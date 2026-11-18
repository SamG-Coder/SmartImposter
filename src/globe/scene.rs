//! The globe as content for the main renderer (renderer.rs): terrain tiles and tree tiles streamed
//! around the camera, each with its own BLAS, turned into TLAS instances in the camera's frame
//! every frame.
//!
//! Frame: the local east / up / south axes at the camera (x east, y up, -z north, the forest
//! scene's convention), origin at the camera. Positions are float64 Earth-centred (ECEF) on the
//! CPU and only the camera-relative, rotated offsets reach the GPU, so precision holds from orbit
//! to the ground.

use super::dem::Terrain;
use super::forest::{self, Roles, Species, TreeTile};
use super::tiles::{self, TileId, TileMesh, R_EARTH, VSTRIDE};
use crate::accel;
use crate::gpu::{Buffer, Gpu, Image};
use anyhow::Result;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{DMat3, DVec3};
use gpu_allocator::MemoryLocation;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{mpsc, Arc, Condvar, Mutex};

/// globe.rchit's uniform block (binding 30). Must match globe.glsl GlobeGlobals.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct GlobeGlobals {
    pub cam_right: [f32; 4],
    pub cam_up: [f32; 4],
    pub cam_fwd: [f32; 4],
    pub sun: [f32; 4],
    /// xyz planet centre relative to the camera (camera frame), w = radius
    pub planet: [f32; 4],
    /// x exposure, y frame index, z debug view, w ground scans present (bit mask)
    pub params: [f32; 4],
    /// x pixel angle (rad); trees: y every tree within, z SPLIT x finest tile edge, w range
    pub misc: [f32; 4],
    /// xy shared index buffer address (lo, hi)
    pub addr: [u32; 4],
}

const CACHE_TILES: usize = 14_000;
/// Terrain tiles not drawn for this many frames are evicted even below CACHE_TILES.
const STALE_FRAMES: u64 = 240;
const UPLOADS_PER_FRAME: usize = 64;
const TREE_UPLOADS_PER_FRAME: usize = 96;
/// Trees uploaded per frame at most (BLAS builds on the GPU, recorded in the frame).
const TREES_PER_FRAME: usize = 80_000;
/// Per-frame staging (host -> device copies) and BLAS build scratch.
const STAGING: u64 = 48 << 20;
/// Bytes per plant record (forest::REC_WORDS words).
const PLANT_REC_BYTES: u64 = forest::REC_WORDS as u64 * 4;
/// Prefetch along the motion: look this many frames ahead (at most MAX_LEAD m; below MIN_LEAD m
/// the camera counts as still).
const LEAD_FRAMES: f64 = 30.0;
const MAX_LEAD: f64 = 15_000.0;
const MIN_LEAD: f64 = 60.0;
const SCRATCH: u64 = 96 << 20;
/// Terrain tiles in the tile table (globe.rchit indexes it by instance custom index).
pub const MAX_TERRAIN_TILES: usize = 16_384;

/// Hit records in the main pipeline (renderer.rs create_rt_pipeline).
pub const SBT_IMPOSTOR: u32 = 1;
pub const SBT_GLOBE_TERRAIN: u32 = 3;

struct TileGpu {
    /// slot in the vertex pool
    slot: u32,
    /// BLAS object, its address and its place in the AS heap (built, then compacted)
    blas: vk::AccelerationStructureKHR,
    blas_addr: u64,
    heap: HeapSlot,
    vtx_addr: u64,
    center: DVec3,
    last_used: u64,
}

struct TreeTileGpu {
    /// first plant record (u32::MAX: no trees)
    base: u32,
    count: u32,
    /// BLAS object, its address and its place in the AS heap
    blas: Option<(vk::AccelerationStructureKHR, u64, HeapSlot)>,
    center: DVec3,
    /// finest tiles: the plant records on the CPU too (mesh LOD near the camera)
    recs: Vec<[u32; forest::REC_WORDS]>,
}

/// Terrain tile vertices: fixed-size slots in device-local chunks of POOL_CHUNK tiles, added as
/// tiles become resident (up to `cap`), freed slots reused lowest first, so the tiles pack into the
/// first chunks and a chunk left empty (the camera moved far away) is given back (trim). (Their
/// BLASes live in the AS heap.)
struct TilePool {
    vtx: Vec<Option<Buffer>>,
    vslot: u64,
    free: std::collections::BTreeSet<u32>,
    cap: u32,
}

const POOL_CHUNK: u32 = 512;

impl TilePool {
    fn new(vtx_bytes: u64, cap: u32) -> Self {
        TilePool { vtx: Vec::new(), vslot: vtx_bytes.div_ceil(256) * 256, free: Default::default(), cap }
    }
    fn chunks(&self) -> usize {
        self.vtx.iter().filter(|c| c.is_some()).count()
    }
    /// The lowest free slot (adding a chunk when all are taken), or None at capacity.
    fn alloc(&mut self, gpu: &Gpu) -> Result<Option<u32>> {
        if self.free.is_empty() && (self.chunks() as u32) * POOL_CHUNK < self.cap {
            let vu = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR | vk::BufferUsageFlags::TRANSFER_DST;
            let buf = gpu.create_buffer(self.vslot * POOL_CHUNK as u64, vu, MemoryLocation::GpuOnly, "terrain vertex pool")?;
            let c = match self.vtx.iter().position(|c| c.is_none()) {
                Some(c) => {
                    self.vtx[c] = Some(buf);
                    c
                }
                None => {
                    self.vtx.push(Some(buf));
                    self.vtx.len() - 1
                }
            } as u32;
            self.free.extend(c * POOL_CHUNK..(c + 1) * POOL_CHUNK);
        }
        Ok(self.free.pop_first())
    }
    fn release(&mut self, slot: u32) {
        self.free.insert(slot);
    }
    fn vtx(&self, i: u32) -> (&Buffer, u64) {
        (self.vtx[(i / POOL_CHUNK) as usize].as_ref().expect("live pool chunk"), (i % POOL_CHUNK) as u64 * self.vslot)
    }
    /// Give back empty chunks beyond `spare` (after the fence: no frame in flight uses them).
    fn trim(&mut self, gpu: &Gpu, spare: usize) {
        let mut empty: Vec<usize> = (0..self.vtx.len())
            .filter(|&c| self.vtx[c].is_some() && self.free.range(c as u32 * POOL_CHUNK..(c as u32 + 1) * POOL_CHUNK).count() == POOL_CHUNK as usize)
            .collect();
        // keep the lowest empty chunks (the next tiles go there), give back the highest
        empty.reverse();
        for &c in empty.iter().take(empty.len().saturating_sub(spare)) {
            if let Some(mut b) = self.vtx[c].take() {
                gpu.destroy_buffer(&mut b);
            }
            for s in c as u32 * POOL_CHUNK..(c as u32 + 1) * POOL_CHUNK {
                self.free.remove(&s);
            }
        }
    }
}

/// Where a tree BLAS lives in the AS heap: size class, chunk, offset.
#[derive(Clone, Copy)]
struct HeapSlot {
    class: usize,
    chunk: usize,
    offset: u64,
}

/// Acceleration structure storage for the streamed BLASes (terrain and trees): size classes
/// 1/8 octave apart (~9%: compacted BLASes vary in size) carved out of 64 MB device chunks,
/// freed slots reused per class, lowest chunk first (no allocator calls per tile). A chunk whose
/// slots are all free again is given back (trim), so the heap shrinks after the camera moves on.
struct AsHeap {
    chunks: Vec<Option<Buffer>>,
    /// live slots per chunk
    live: Vec<u32>,
    /// the chunk new slots are cut from, and how far
    cur: usize,
    bump: u64,
    free: Vec<std::collections::BTreeSet<(usize, u64)>>,
}

const AS_CHUNK: u64 = 64 << 20;
const AS_MIN_CLASS: u64 = 4 << 10;
const AS_CLASSES: usize = 96; // 4 KB .. 16 MB

impl AsHeap {
    fn new() -> Self {
        AsHeap { chunks: Vec::new(), live: Vec::new(), cur: usize::MAX, bump: AS_CHUNK, free: vec![Default::default(); AS_CLASSES] }
    }
    /// The class of a size, and that class's size (a multiple of 256: AS offsets must be).
    fn class_of(size: u64) -> (usize, u64) {
        let c = ((size.max(AS_MIN_CLASS) as f64 / AS_MIN_CLASS as f64).log2() * 8.0).ceil() as usize;
        let c = c.min(AS_CLASSES - 1);
        let bytes = (AS_MIN_CLASS as f64 * 2f64.powf(c as f64 / 8.0)).ceil() as u64;
        (c, bytes.div_ceil(256) * 256)
    }
    fn alloc(&mut self, gpu: &Gpu, size: u64) -> Result<HeapSlot> {
        let (class, csize) = Self::class_of(size);
        if let Some((chunk, offset)) = self.free[class].pop_first() {
            self.live[chunk] += 1;
            return Ok(HeapSlot { class, chunk, offset });
        }
        if self.bump + csize > AS_CHUNK {
            let buf = gpu.create_buffer(AS_CHUNK, vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR, MemoryLocation::GpuOnly, "blas heap")?;
            self.cur = match self.chunks.iter().position(|c| c.is_none()) {
                Some(c) => {
                    self.chunks[c] = Some(buf);
                    self.live[c] = 0;
                    c
                }
                None => {
                    self.chunks.push(Some(buf));
                    self.live.push(0);
                    self.chunks.len() - 1
                }
            };
            self.bump = 0;
        }
        let s = HeapSlot { class, chunk: self.cur, offset: self.bump };
        self.bump += csize;
        self.live[self.cur] += 1;
        Ok(s)
    }
    fn release(&mut self, s: HeapSlot) {
        self.live[s.chunk] -= 1;
        self.free[s.class].insert((s.chunk, s.offset));
    }
    fn raw(&self, chunk: usize) -> vk::Buffer {
        self.chunks[chunk].as_ref().expect("live AS heap chunk").raw
    }
    fn chunk_count(&self) -> usize {
        self.chunks.iter().filter(|c| c.is_some()).count()
    }
    /// Give back chunks with no live slot, beyond `spare` of them (after the fence, once the
    /// BLASes in them are destroyed).
    fn trim(&mut self, gpu: &Gpu, spare: usize) {
        let empty: Vec<usize> = (0..self.chunks.len()).filter(|&c| c != self.cur && self.chunks[c].is_some() && self.live[c] == 0).collect();
        if empty.len() <= spare {
            return;
        }
        let gone: std::collections::HashSet<usize> = empty.iter().rev().take(empty.len() - spare).copied().collect();
        for &c in &gone {
            if let Some(mut b) = self.chunks[c].take() {
                gpu.destroy_buffer(&mut b);
            }
        }
        for f in &mut self.free {
            f.retain(|(c, _)| !gone.contains(c));
        }
    }
}

/// GPU work recorded into the frame's command buffer (renderer.rs, before the TLAS build).
enum Op {
    Copy { src: u64, dst: vk::Buffer, dst_off: u64, size: u64 },
    /// the ocean wave textures (all mips of one image) from staging
    Image { src: u64, img: vk::Image },
    Terrain { vtx: u64, dst: vk::AccelerationStructureKHR, scratch: u64 },
    Trees { aabbs: u64, count: u32, dst: vk::AccelerationStructureKHR, scratch: u64 },
    /// copy a BLAS built in an earlier frame to its compacted size
    Compact { src: vk::AccelerationStructureKHR, dst: vk::AccelerationStructureKHR },
}

/// The ocean wave textures (ocean.rs: A and B per cascade), RGBA16F with mips, ready to sample.
fn ocean_images(gpu: &Gpu) -> Result<Vec<Image>> {
    use super::ocean::{CASCADES, MIPS, N};
    let mut v = Vec::new();
    for k in 0..CASCADES * 2 {
        v.push(gpu.create_image(N as u32, N as u32, MIPS as u32, vk::Format::R16G16B16A16_SFLOAT, vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST, &format!("ocean {k}"))?);
    }
    let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: MIPS as u32, base_array_layer: 0, layer_count: 1 };
    gpu.one_time(|cb| {
        let b: Vec<_> = v
            .iter()
            .map(|i: &Image| {
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(i.raw)
                    .subresource_range(range)
            })
            .collect();
        unsafe { gpu.device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(&b)) };
        for i in &v {
            unsafe {
                gpu.device.cmd_clear_color_image(cb, i.raw, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &vk::ClearColorValue { float32: [0.0; 4] }, &[range]);
            }
        }
        let b: Vec<_> = v
            .iter()
            .map(|i: &Image| {
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                    .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(i.raw)
                    .subresource_range(range)
            })
            .collect();
        unsafe { gpu.device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(&b)) };
    })?;
    Ok(v)
}

/// The sea state under the camera (ocean.rs): the climatological wind there, in the patch
/// coordinates (the cube face's u, v: the texture coordinates the water shading uses), a swell
/// from a little off the wind, an open-sea fetch.
fn sea_state(cam: DVec3) -> super::ocean::SeaState {
    let d = cam.normalize();
    let (lat, lon) = tiles::lat_lon(d);
    let (wind, en) = super::ocean::wind_at(lat, lon);
    let (face, u, v) = forest::dir_to_face_uv(d);
    let e = 1e-6;
    let tu = (tiles::face_dir(face, u + e, v) - tiles::face_dir(face, u - e, v)).normalize();
    let tv = (tiles::face_dir(face, u, v + e) - tiles::face_dir(face, u, v - e)).normalize();
    let east = DVec3::Z.cross(d).normalize_or(DVec3::X);
    let north = d.cross(east);
    let to_patch = |a: [f32; 2]| {
        let w = east * a[0] as f64 + north * a[1] as f64;
        let p = glam::DVec2::new(w.dot(tu), w.dot(tv)).normalize_or(glam::DVec2::X);
        [p.x as f32, p.y as f32]
    };
    let ang = (en[1] as f64).atan2(en[0] as f64) + 0.5;
    super::ocean::SeaState {
        wind,
        dir: to_patch(en),
        fetch: 300_000.0,
        swell_hs: 0.5 + 0.09 * wind,
        swell_t: 9.0 + 0.25 * wind,
        swell_dir: to_patch([ang.cos() as f32, ang.sin() as f32]),
    }
}

/// Most BLAS builds per frame whose compacted size is queried.
const QUERIES: u32 = 1024;

/// Work queues (0 terrain tiles, 1 tree tiles), served by one pool of workers.
type Queue = Arc<(Mutex<[VecDeque<TileId>; 2]>, Condvar)>;

/// Tree workers split: [trees, understory] x (ns, tiles).
static UNDER: [[std::sync::atomic::AtomicU64; 2]; 2] =
    [[std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)], [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)]];

/// Worker time (ns) and tiles made: terrain, trees.
static WORK: [[std::sync::atomic::AtomicU64; 2]; 2] =
    [[std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)], [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)]];

/// `n` workers: half take terrain tiles first, half tree tiles first, each falling back to the
/// other queue, so the whole pool goes where the work is and neither queue starves when the
/// camera moves fast.
/// What each worker is doing (kind, tile, since): diagnostics for work that never finishes.
static DROPPED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static RECEIVED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static ALIVE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static ACTIVE: std::sync::Mutex<Vec<Option<(usize, TileId, std::time::Instant)>>> = std::sync::Mutex::new(Vec::new());

fn spawn_workers(
    n: usize,
    queue: &Queue,
    tx: (mpsc::Sender<TileMesh>, mpsc::Sender<TreeTile>),
    terrain: impl Fn(TileId) -> TileMesh + Send + Sync + 'static,
    trees: impl Fn(TileId) -> TreeTile + Send + Sync + 'static,
) {
    let work = Arc::new((terrain, trees));
    ACTIVE.lock().unwrap().resize(n, None);
    for w in 0..n {
        let (q, tx, work) = (queue.clone(), tx.clone(), work.clone());
        let first = w % 2;
        ALIVE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::thread::spawn(move || loop {
            struct Bye;
            impl Drop for Bye {
                fn drop(&mut self) {
                    if std::thread::panicking() {
                        ALIVE.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                        eprintln!("globe worker died (panic)");
                    }
                }
            }
            let _bye = Bye;
            let (kind, id) = {
                let (lock, cv) = &*q;
                let mut g = lock.lock().unwrap();
                loop {
                    if let Some(id) = g[first].pop_front() {
                        break (first, id);
                    }
                    if let Some(id) = g[1 - first].pop_front() {
                        break (1 - first, id);
                    }
                    g = cv.wait(g).unwrap();
                }
            };
            let t = std::time::Instant::now();
            ACTIVE.lock().unwrap()[w] = Some((kind, id, t));
            let sent = if kind == 0 { tx.0.send((work.0)(id)).is_ok() } else { tx.1.send((work.1)(id)).is_ok() };
            WORK[kind][0].fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            WORK[kind][1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            ACTIVE.lock().unwrap()[w] = None;
            if !sent {
                return;
            }
        });
    }
}

/// Replace a work queue's not-yet-started entries with `want` (nearest first).
fn requeue(queue: &Queue, kind: usize, pending: &mut HashSet<TileId>, want: impl Iterator<Item = TileId>, have: impl Fn(&TileId) -> bool, limit: usize) {
    let (lock, cv) = &**queue;
    let mut g = lock.lock().unwrap();
    let q = &mut g[kind];
    for id in q.drain(..) {
        pending.remove(&id);
    }
    for id in want.take(limit) {
        if !pending.contains(&id) && !have(&id) {
            pending.insert(id);
            q.push_back(id);
        }
    }
    cv.notify_all();
}

/// First-fit allocator of record ranges in the plant buffer (free ranges sorted, merged on free).
struct RangeAlloc {
    free: Vec<(u32, u32)>,
}

impl RangeAlloc {
    fn new(n: u32) -> Self {
        RangeAlloc { free: vec![(0, n)] }
    }
    fn alloc(&mut self, n: u32) -> Option<u32> {
        let i = self.free.iter().position(|r| r.1 >= n)?;
        let start = self.free[i].0;
        self.free[i].0 += n;
        self.free[i].1 -= n;
        if self.free[i].1 == 0 {
            self.free.remove(i);
        }
        Some(start)
    }
    fn release(&mut self, start: u32, n: u32) {
        let i = self.free.partition_point(|r| r.0 < start);
        self.free.insert(i, (start, n));
        if i + 1 < self.free.len() && self.free[i].0 + self.free[i].1 == self.free[i + 1].0 {
            self.free[i].1 += self.free[i + 1].1;
            self.free.remove(i + 1);
        }
        if i > 0 && self.free[i - 1].0 + self.free[i - 1].1 == self.free[i].0 {
            self.free[i - 1].1 += self.free[i].1;
            self.free.remove(i);
        }
    }
}

/// Rotation from ECEF into the camera frame at `pos`: rows east, up, south.
pub fn frame_at(pos: DVec3) -> DMat3 {
    let up = pos.normalize();
    let mut east = DVec3::Z.cross(up);
    if east.length_squared() < 1e-12 {
        east = DVec3::X;
    }
    let east = east.normalize();
    let north = up.cross(east);
    DMat3::from_cols(east, up, -north).transpose()
}

/// Carry a camera frame (rows east, up, south as frame_at) from wherever it was to `to` (ECEF)
/// without turning it: the horizontal axes are projected onto the new tangent plane (parallel
/// transport). Unlike frame_at, which is tied to north, this keeps a straight flight straight
/// over the poles and never spins near them.
pub fn transport(frame: DMat3, to: DVec3) -> DMat3 {
    let up = to.normalize();
    let east0 = frame.row(0);
    let mut east = east0 - up * east0.dot(up);
    if east.length_squared() < 1e-12 {
        // the old east points straight up (a 90 degree jump): fall back to the old south
        let s0 = frame.row(2);
        east = up.cross(s0 - up * s0.dot(up));
    }
    let east = east.normalize();
    let south = east.cross(up);
    DMat3::from_cols(east, up, south).transpose()
}

/// Heading of a camera-frame direction in degrees from north (clockwise), as `--at` takes it.
pub fn heading(frame: DMat3, dir: DVec3) -> f64 {
    let d = frame.transpose() * dir; // Earth axes
    let n = frame_at(frame.row(1)); // north-tied frame at the same place
    let l = n * d;
    l.x.atan2(-l.z).to_degrees().rem_euclid(360.0)
}

pub struct GlobeScene {
    pub terrain: Arc<Terrain>,
    /// animation time (s) for the ocean: main.rs State.clock (a fixed step per frame in a tour)
    pub clock: f64,
    tiles: HashMap<TileId, TileGpu>,
    pending: HashSet<TileId>,
    queue: Queue,
    done_rx: mpsc::Receiver<TileMesh>,
    trees: HashMap<TileId, TreeTileGpu>,
    records: RangeAlloc,
    tree_pending: HashSet<TileId>,
    tree_rx: mpsc::Receiver<TreeTile>,
    idx: Buffer,
    tri_count: u32,
    /// device-local pools: terrain tile vertices and BLASes (fixed slots), tree BLAS heap,
    /// tree AABBs (indexed like the plant records)
    pool: TilePool,
    /// compaction: the compacted-size queries of this frame's builds (terrain?, tile, BLAS), read
    /// next frame; and BLASes replaced by their compacted copies, freed after the next fence
    query_pool: vk::QueryPool,
    queried: Vec<(bool, TileId, vk::AccelerationStructureKHR)>,
    built: Vec<(bool, TileId, vk::AccelerationStructureKHR)>,
    retire: Vec<(vk::AccelerationStructureKHR, HeapSlot)>,
    blas_size: u64,
    terrain_scratch: u64,
    as_heap: AsHeap,
    /// per-frame staging ring and build scratch, and the work recorded for this frame
    staging: Buffer,
    staging_used: u64,
    scratch: Buffer,
    scratch_used: u64,
    scratch_align: u64,
    ops: Vec<Op>,
    /// finished tiles waiting for room in a frame
    backlog: VecDeque<TileMesh>,
    tree_backlog: VecDeque<TreeTile>,
    /// The whole ETOPO grid (R16_SNORM, mips) for per-pixel coastlines / relief far away.
    pub elev_tex: Image,
    /// ocean waves (ocean.rs): the simulation thread, its textures (A, B per cascade), and the
    /// statistics of the frame they hold (Globals.ocean*)
    ocean: super::ocean::Ocean,
    pub ocean_tex: Vec<Image>,
    ocean_stats: [[f32; 4]; 3],
    /// globe.rchit globals (binding 30) and terrain tile table (binding 31)
    pub ubo: Buffer,
    pub table: Buffer,
    /// plant records and LOD state of the globe's trees (bindings 23 / 24 in globe mode)
    pub plants: Buffer,
    pub plant_state: Buffer,
    frame: u64,
    /// last frame: terrain tiles drawn, tiles still wanted, trees in the TLAS, tree range (m)
    pub drawn: usize,
    pub waiting: usize,
    pub trees_drawn: usize,
    pub range: f64,
    /// this frame's tree tiles (prepare) and terrain tile table (written after the fence)
    keep: HashSet<TileId>,
    table_cpu: Vec<u64>,
    held: usize,
    /// running mean plant records per tree tile and per understory tile (the request budget)
    recs_mean: [f64; 2],
    /// camera of the previous frame (motion for prefetching)
    prev_cam: Option<DVec3>,
}

impl GlobeScene {
    pub fn new(gpu: &Gpu, terrain: Arc<Terrain>, species: Species) -> Result<Self> {
        let indices = tiles::index_buffer();
        let idx = gpu.upload_buffer(
            bytemuck::cast_slice(&indices),
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            "tile indices",
        )?;
        let t = std::time::Instant::now();
        let elev_tex = crate::textures::upload_with_mips(
            gpu,
            terrain.etopo.w as u32,
            terrain.etopo.h as u32,
            bytemuck::cast_slice(terrain.etopo.raw()),
            vk::Format::R16_SNORM,
            "etopo",
        )?;
        println!("Elevation grid on the GPU ({:.1}s)", t.elapsed().as_secs_f32());
        let ubo = gpu.create_buffer(256, vk::BufferUsageFlags::UNIFORM_BUFFER, MemoryLocation::CpuToGpu, "globe globals")?;
        let table = gpu.create_buffer(MAX_TERRAIN_TILES as u64 * 8, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "tile table")?;
        let recs = forest::CAPACITY as u64;
        let plants = gpu.create_buffer(recs * PLANT_REC_BYTES, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, MemoryLocation::GpuOnly, "globe plants")?;
        let mut plant_state = gpu.create_buffer(recs * 4, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "globe plant state")?;
        plant_state.mapped().fill(0);
        // pools: every terrain tile has the same vertex count and topology, so the same BLAS size
        let tri_count = (indices.len() / 3) as u32;
        let nverts = (tiles::GRID * tiles::GRID + 4 * tiles::GRID) as u32;
        let sample = terrain_geometry(0, idx.addr, nverts);
        let (blas_size, terrain_scratch) = accel::blas_sizes(gpu, &sample, tri_count);
        // a slot: the tile's vertices, then its shoreline field (tiles.rs TILE_FLOATS)
        let mut pool = TilePool::new((tiles::TILE_FLOATS * 4) as u64, (CACHE_TILES + 512) as u32);
        let query_pool = unsafe {
            gpu.device.create_query_pool(
                &vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR).query_count(QUERIES),
                None,
            )?
        };
        // a typical view keeps 5-9k tiles: allocate the first chunks now, not mid-flight
        let mut first = Vec::new();
        for _ in 0..12 * POOL_CHUNK {
            first.extend(pool.alloc(gpu)?);
        }
        pool.free.extend(first);
        let staging = gpu.create_buffer(
            STAGING,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            MemoryLocation::CpuToGpu,
            "globe staging",
        )?;
        let scratch = gpu.create_buffer(SCRATCH + 4096, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, "globe build scratch")?;
        println!(
            "Globe pools: terrain tiles {:.0} KB vertices + {:.0} KB BLAS each before compaction (chunks of {}), {:.0} MB plant records",
            pool.vslot as f64 / 1e3,
            blas_size as f64 / 1e3,
            POOL_CHUNK,
            (recs * PLANT_REC_BYTES) as f64 / 1e6
        );

        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let queue: Queue = Arc::new((Mutex::new([VecDeque::new(), VecDeque::new()]), Condvar::new()));
        let (done_tx, done_rx) = mpsc::channel();
        let (tree_tx, tree_rx) = mpsc::channel();
        let (et, et2, roles, sp) = (terrain.clone(), terrain.clone(), Arc::new(Roles::new(&species)), Arc::new(species));
        // all cores but the render thread, the driver and a little slack
        spawn_workers(
            cores.saturating_sub(3).max(2),
            &queue,
            (done_tx, tree_tx),
            move |id| tiles::build_tile(id, &et),
            move |id| {
                let t = std::time::Instant::now();
                let under = id.level == forest::UNDER_LEVEL;
                let out = if under { forest::grow_understory(id, &et2, &sp) } else { forest::grow_tile(id, &et2, &sp, &roles) };
                UNDER[under as usize][0].fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                UNDER[under as usize][1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                out
            },
        );

        let mut s = GlobeScene {
            terrain: terrain.clone(),
            tiles: HashMap::new(),
            pending: HashSet::new(),
            queue,
            done_rx,
            trees: HashMap::new(),
            records: RangeAlloc::new(forest::CAPACITY),
            tree_pending: HashSet::new(),
            tree_rx,
            idx,
            tri_count,
            pool,
            query_pool,
            queried: Vec::new(),
            built: Vec::new(),
            retire: Vec::new(),
            blas_size,
            terrain_scratch,
            as_heap: AsHeap::new(),
            staging,
            staging_used: 0,
            scratch,
            scratch_used: 0,
            scratch_align: gpu.rt.scratch_alignment.max(256),
            ops: Vec::new(),
            backlog: VecDeque::new(),
            tree_backlog: VecDeque::new(),
            elev_tex,
            ocean: super::ocean::Ocean::spawn(),
            clock: 0.0,
            ocean_tex: ocean_images(gpu)?,
            ocean_stats: [[0.0; 4]; 3],
            ubo,
            table,
            plants,
            plant_state,
            frame: 0,
            drawn: 0,
            waiting: 0,
            trees_drawn: 0,
            range: 0.0,
            keep: HashSet::new(),
            table_cpu: Vec::new(),
            held: 0,
            recs_mean: [1500.0, 1500.0],
            prev_cam: None,
        };
        // the coarse levels everywhere, built up front so the globe is never missing pieces
        let mut ids = Vec::new();
        let mut level = (0..6).map(TileId::root).collect::<Vec<_>>();
        for _ in 0..=2 {
            ids.extend_from_slice(&level);
            level = level.iter().flat_map(|t| t.children()).collect();
        }
        let meshes: Vec<TileMesh> = std::thread::scope(|sc| {
            let hs: Vec<_> = ids.chunks(16).map(|c| sc.spawn(|| c.iter().map(|&id| tiles::build_tile(id, &terrain)).collect::<Vec<_>>())).collect();
            hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        // through the same path as streaming, in batches that fit a frame's staging / scratch
        s.backlog.extend(meshes);
        while !s.backlog.is_empty() {
            s.begin_frame();
            s.compact(gpu)?;
            s.upload(gpu)?;
            gpu.one_time(|cb| s.record(gpu, cb))?;
        }
        Ok(s)
    }

    fn begin_frame(&mut self) {
        self.staging_used = 0;
        self.scratch_used = 0;
        self.ops.clear();
    }

    /// Room in this frame's staging ring (offset), or None.
    fn stage(&mut self, bytes: &[u8]) -> Option<u64> {
        let off = self.staging_used.div_ceil(256) * 256;
        if off + bytes.len() as u64 > STAGING {
            return None;
        }
        self.staging.mapped()[off as usize..off as usize + bytes.len()].copy_from_slice(bytes);
        self.staging_used = off + bytes.len() as u64;
        Some(off)
    }

    /// Scratch for one BLAS build this frame (device address), or None.
    fn scratch_for(&mut self, size: u64) -> Option<u64> {
        let off = self.scratch_used.div_ceil(self.scratch_align) * self.scratch_align;
        if off + size > SCRATCH {
            return None;
        }
        self.scratch_used = off + size;
        Some(crate::gpu::align_up(self.scratch.addr, self.scratch_align) + off)
    }

    /// Record this frame's uploads and BLAS builds (copies, then builds, with the barriers the TLAS
    /// build and the shaders need). Called by the renderer before the TLAS build.
    /// The ocean statistics for Globals.ocean0..2 (main.rgen).
    pub fn ocean_globals(&self) -> [[f32; 4]; 3] {
        self.ocean_stats
    }

    pub fn record(&mut self, gpu: &Gpu, cb: vk::CommandBuffer) {
        if self.ops.is_empty() {
            return;
        }
        let d = &gpu.device;
        let staging = self.staging.raw;
        // ocean textures: shader-read -> transfer, every mip from staging, back to shader-read
        let images: Vec<(u64, vk::Image)> = self.ops.iter().filter_map(|o| if let Op::Image { src, img } = o { Some((*src, *img)) } else { None }).collect();
        if !images.is_empty() {
            let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: super::ocean::MIPS as u32, base_array_layer: 0, layer_count: 1 };
            let barrier = |img: vk::Image, old, new, sa, da, ss, ds| {
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(ss)
                    .src_access_mask(sa)
                    .dst_stage_mask(ds)
                    .dst_access_mask(da)
                    .old_layout(old)
                    .new_layout(new)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(img)
                    .subresource_range(range)
            };
            let to_dst: Vec<_> = images
                .iter()
                .map(|(_, img)| {
                    barrier(
                        *img,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::AccessFlags2::SHADER_READ,
                        vk::AccessFlags2::TRANSFER_WRITE,
                        vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                        vk::PipelineStageFlags2::TRANSFER,
                    )
                })
                .collect();
            unsafe { d.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(&to_dst)) };
            for (src, img) in &images {
                let mut off = *src;
                let mut regions = Vec::new();
                for m in 0..super::ocean::MIPS as u32 {
                    let n = (super::ocean::N as u32 >> m).max(1);
                    regions.push(vk::BufferImageCopy {
                        buffer_offset: off,
                        image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: m, base_array_layer: 0, layer_count: 1 },
                        image_extent: vk::Extent3D { width: n, height: n, depth: 1 },
                        ..Default::default()
                    });
                    off += (n * n * 8) as u64;
                }
                unsafe { d.cmd_copy_buffer_to_image(cb, staging, *img, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &regions) };
            }
            let to_read: Vec<_> = images
                .iter()
                .map(|(_, img)| {
                    barrier(
                        *img,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::AccessFlags2::TRANSFER_WRITE,
                        vk::AccessFlags2::SHADER_READ,
                        vk::PipelineStageFlags2::TRANSFER,
                        vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    )
                })
                .collect();
            unsafe { d.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(&to_read)) };
        }
        for op in &self.ops {
            if let Op::Copy { src, dst, dst_off, size } = op {
                unsafe { d.cmd_copy_buffer(cb, staging, *dst, &[vk::BufferCopy { src_offset: *src, dst_offset: *dst_off, size: *size }]) };
            }
        }
        gpu.memory_barrier(
            cb,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR | vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags2::SHADER_READ,
        );
        for op in &self.ops {
            match op {
                Op::Terrain { vtx, dst, scratch } => {
                    let nverts = (tiles::GRID * tiles::GRID + 4 * tiles::GRID) as u32;
                    accel::cmd_build_blas(gpu, cb, &terrain_geometry(*vtx, self.idx.addr, nverts), self.tri_count, *dst, *scratch);
                }
                Op::Trees { aabbs, count, dst, scratch } => accel::cmd_build_blas(gpu, cb, &accel::aabb_geometry(*aabbs), *count, *dst, *scratch),
                Op::Copy { .. } | Op::Compact { .. } | Op::Image { .. } => {}
            }
        }
        gpu.memory_barrier(
            cb,
            vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
            vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR | vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
        );
        // earlier frames' builds into their compacted size
        for op in &self.ops {
            if let Op::Compact { src, dst } = op {
                let info = vk::CopyAccelerationStructureInfoKHR::default().src(*src).dst(*dst).mode(vk::CopyAccelerationStructureModeKHR::COMPACT);
                unsafe { gpu.accel_fn.cmd_copy_acceleration_structure(cb, &info) };
            }
        }
        // this frame's builds: their compacted sizes, read next frame (update)
        self.queried = std::mem::take(&mut self.built);
        self.queried.truncate(QUERIES as usize);
        if !self.queried.is_empty() {
            let raws: Vec<vk::AccelerationStructureKHR> = self.queried.iter().map(|q| q.2).collect();
            unsafe {
                d.cmd_reset_query_pool(cb, self.query_pool, 0, raws.len() as u32);
                gpu.accel_fn.cmd_write_acceleration_structures_properties(cb, &raws, vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, self.query_pool, 0);
            }
        }
        gpu.memory_barrier(
            cb,
            vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR | vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_COPY_KHR,
            vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
            vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR | vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
        );
    }

    /// After the previous frame's fence: free the BLASes its compaction copies replaced, then
    /// compact the BLASes it built (their sizes are known now): copies recorded for this frame,
    /// the tiles use the copies from the next frame's instances on.
    fn compact(&mut self, gpu: &Gpu) -> Result<()> {
        for (raw, heap) in self.retire.drain(..) {
            unsafe { gpu.accel_fn.destroy_acceleration_structure(raw, None) };
            self.as_heap.release(heap);
        }
        let queried = std::mem::take(&mut self.queried);
        if queried.is_empty() {
            return Ok(());
        }
        let mut sizes = vec![0u64; queried.len()];
        unsafe { gpu.device.get_query_pool_results(self.query_pool, 0, &mut sizes, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)? };
        for ((terrain, id, raw), size) in queried.into_iter().zip(sizes) {
            if size == 0 {
                continue;
            }
            // still resident with that BLAS (not evicted or replaced since)
            let current = if terrain {
                self.tiles.get(&id).filter(|t| t.blas == raw).map(|t| t.heap)
            } else {
                self.trees.get(&id).and_then(|t| t.blas).filter(|b| b.0 == raw).map(|b| b.2)
            };
            let Some(old_heap) = current else { continue };
            let heap = self.as_heap.alloc(gpu, size)?;
            let (dst, addr) = accel::create_accel_in(gpu, self.as_heap.raw(heap.chunk), heap.offset, size, vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)?;
            self.ops.push(Op::Compact { src: raw, dst });
            self.retire.push((raw, old_heap));
            if terrain {
                let t = self.tiles.get_mut(&id).unwrap();
                (t.blas, t.blas_addr, t.heap) = (dst, addr, heap);
            } else {
                self.trees.get_mut(&id).unwrap().blas = Some((dst, addr, heap));
            }
        }
        Ok(())
    }

    pub fn index_addr(&self) -> u64 {
        self.idx.addr
    }

    /// Terrain and tree tiles still being built.
    /// What keeps the scene busy (screenshot diagnostics).
    pub fn busy_detail(&self) -> String {
        let slow: Vec<String> = ACTIVE
            .lock()
            .unwrap()
            .iter()
            .flatten()
            .filter(|a| a.2.elapsed().as_secs_f64() > 5.0)
            .map(|(k, id, t)| format!("{} {:?} for {:.0} s", if *k == 0 { "terrain" } else { "trees" }, id, t.elapsed().as_secs_f64()))
            .collect();
        let ql = {
            let g = self.queue.0.lock().unwrap();
            (g[0].len(), g[1].len())
        };
        format!(
            "received {} dropped {} resident trees {}; workers alive {} queues {:?}; waiting {} tree_pending {} backlog {} tree_backlog {}; stuck: {:?}",
            RECEIVED.load(std::sync::atomic::Ordering::Relaxed),
            DROPPED.load(std::sync::atomic::Ordering::Relaxed),
            self.trees.values().map(|t| t.count as usize).sum::<usize>(),
            ALIVE.load(std::sync::atomic::Ordering::Relaxed),
            ql,
            self.waiting,
            self.tree_pending.len(),
            self.backlog.len(),
            self.tree_backlog.len(),
            slow
        )
    }

    pub fn busy(&self) -> bool {
        // nothing requested before the first frames
        self.frame < 2 || self.waiting > 0 || !self.tree_pending.is_empty() || !self.backlog.is_empty() || !self.tree_backlog.is_empty()
    }

    /// Terrain tiles from the backlog into pool slots, as far as this frame's staging and scratch
    /// allow: vertex copy + BLAS build recorded for the frame (no waiting).
    fn upload(&mut self, gpu: &Gpu) -> Result<()> {
        while let Some(m) = self.backlog.pop_front() {
            if self.tiles.contains_key(&m.id) {
                continue;
            }
            let bytes: &[u8] = bytemuck::cast_slice(&m.verts);
            if self.staging_used + bytes.len() as u64 + 256 > STAGING || self.scratch_used + self.terrain_scratch + self.scratch_align > SCRATCH {
                self.backlog.push_front(m);
                break;
            }
            let Some(slot) = self.pool.alloc(gpu)? else {
                self.backlog.push_front(m);
                break;
            };
            let scratch = self.scratch_for(self.terrain_scratch).unwrap();
            let src = self.stage(bytes).unwrap();
            let heap = self.as_heap.alloc(gpu, self.blas_size)?;
            let (blas, blas_addr) =
                accel::create_accel_in(gpu, self.as_heap.raw(heap.chunk), heap.offset, self.blas_size, vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)?;
            let (vbuf, off) = self.pool.vtx(slot);
            let vtx_addr = vbuf.addr + off;
            self.ops.push(Op::Copy { src, dst: vbuf.raw, dst_off: off, size: bytes.len() as u64 });
            self.ops.push(Op::Terrain { vtx: vtx_addr, dst: blas, scratch });
            self.built.push((true, m.id, blas));
            self.tiles.insert(m.id, TileGpu { slot, blas, blas_addr, heap, vtx_addr, center: m.center, last_used: self.frame });
        }
        Ok(())
    }

    /// Tree tiles from the backlog: plant records and AABBs (copies into their range of the record
    /// buffers), a BLAS from the AS heap, build recorded for the frame. Empty tiles take nothing.
    fn upload_trees(&mut self, gpu: &Gpu) -> Result<()> {
        while let Some(t) = self.tree_backlog.pop_front() {
            if self.trees.contains_key(&t.id) {
                continue;
            }
            let n = t.recs.len();
            if n == 0 {
                self.trees.insert(t.id, TreeTileGpu { base: u32::MAX, count: 0, blas: None, center: t.center, recs: Vec::new() });
                continue;
            }
            let g = accel::aabb_geometry(0);
            let (as_size, sc_size) = accel::blas_sizes(gpu, &g, n as u32);
            let need = (n * PLANT_REC_BYTES as usize + n * 24 + 512) as u64;
            if self.staging_used + need > STAGING {
                self.tree_backlog.push_front(t);
                break;
            }
            let Some(scratch) = self.scratch_for(sc_size) else {
                self.tree_backlog.push_front(t);
                break;
            };
            let Some(base) = self.records.alloc(n as u32) else {
                if std::env::var("SI_TREE_STATS").is_ok() {
                    println!("  plant records full ({} trees resident)", self.trees.values().map(|t| t.count as usize).sum::<usize>());
                }
                self.tree_backlog.push_front(t);
                break;
            };
            let src_r = self.stage(bytemuck::cast_slice(&t.recs)).unwrap();
            let src_a = self.stage(bytemuck::cast_slice(&t.aabbs)).unwrap();
            self.ops.push(Op::Copy { src: src_r, dst: self.plants.raw, dst_off: base as u64 * PLANT_REC_BYTES, size: n as u64 * PLANT_REC_BYTES });
            let hs = self.as_heap.alloc(gpu, as_size)?;
            let (blas, addr) = accel::create_accel_in(gpu, self.as_heap.raw(hs.chunk), hs.offset, as_size, vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)?;
            // the build reads the AABBs straight from staging (only BLAS build input: no copy kept)
            self.ops.push(Op::Trees { aabbs: self.staging.addr + src_a, count: n as u32, dst: blas, scratch });
            self.built.push((false, t.id, blas));
            let keep_recs = if t.id.level >= forest::TREE_LEVEL { t.recs } else { Vec::new() };
            self.trees.insert(t.id, TreeTileGpu { base, count: n as u32, blas: Some((blas, addr, hs)), center: t.center, recs: keep_recs });
        }
        Ok(())
    }

    fn evict(&mut self, gpu: &Gpu) {
        // tiles nothing has drawn for STALE_FRAMES (the camera moved on: a jump, or a long
        // flight) go whatever the cache holds, so their memory is given back
        if self.frame > STALE_FRAMES && self.frame % 30 == 0 {
            let stale: Vec<TileId> = self.tiles.iter().filter(|(id, t)| id.level > 2 && t.last_used + STALE_FRAMES < self.frame).map(|(id, _)| *id).collect();
            for id in stale {
                if let Some(t) = self.tiles.remove(&id) {
                    unsafe { gpu.accel_fn.destroy_acceleration_structure(t.blas, None) };
                    self.as_heap.release(t.heap);
                    self.pool.release(t.slot);
                }
            }
        }
        if self.tiles.len() <= CACHE_TILES {
            return;
        }
        // least recently used first, and among those the finest (coarse tiles are cheap fallbacks)
        let mut old: Vec<(u64, std::cmp::Reverse<u8>, TileId)> = self
            .tiles
            .iter()
            .filter(|(id, t)| id.level > 2 && t.last_used + 2 < self.frame)
            .map(|(id, t)| (t.last_used, std::cmp::Reverse(id.level), *id))
            .collect();
        old.sort();
        let excess = self.tiles.len() - CACHE_TILES * 9 / 10;
        for (_, _, id) in old.into_iter().take(excess) {
            if let Some(t) = self.tiles.remove(&id) {
                unsafe { gpu.accel_fn.destroy_acceleration_structure(t.blas, None) };
                self.as_heap.release(t.heap);
                self.pool.release(t.slot);
            }
        }
    }

    fn evict_trees(&mut self, gpu: &Gpu, keep: &HashSet<TileId>) {
        let gone: Vec<TileId> = self.trees.keys().filter(|id| !keep.contains(id)).copied().collect();
        for id in gone {
            if let Some(mut t) = self.trees.remove(&id) {
                if let Some((b, _, hs)) = t.blas.take() {
                    unsafe { gpu.accel_fn.destroy_acceleration_structure(b, None) };
                    self.as_heap.release(hs);
                }
                if t.base != u32::MAX {
                    self.records.release(t.base, t.count);
                }
            }
        }
    }

    /// CPU half of a frame, run while the previous frame is still on the GPU (touches no memory
    /// the GPU reads): which tiles the camera at `cam` (ECEF) needs, worker requests, and the TLAS
    /// instances in the camera frame (`rot` from frame_at). The instances use only tiles built in
    /// earlier frames that this frame keeps. Returns CPU ms: tree tile list, select + instances.
    pub fn prepare(&mut self, cam: DVec3, rot: DMat3, out: &mut Vec<vk::AccelerationStructureInstanceKHR>) -> [f64; 2] {
        self.frame += 1;
        let t = std::time::Instant::now();
        // Moving fast, request tiles by their distance from where the camera will be in about
        // LEAD_FRAMES frames: the workers fill the ground ahead instead of chasing the tiles under
        // the camera, which are behind it by the time they are done. Still or slow: nearest first.
        let step = self.prev_cam.map(|p| cam - p).unwrap_or(DVec3::ZERO);
        self.prev_cam = Some(cam);
        let lead = step * LEAD_FRAMES;
        let lead = if lead.length() > MAX_LEAD { lead * (MAX_LEAD / lead.length()) } else { lead };
        let ahead = (lead.length() > MIN_LEAD).then(|| cam + lead);
        let eye = cam.length() - tiles::ground_radius(&self.terrain, cam);
        self.ocean.post(sea_state(cam), self.clock as f32);
        self.range = forest::tree_range(eye);
        let mut near = forest::under_tiles(cam);
        near.extend(forest::tiles_near(cam, self.range));
        // Only what fits the plant record pool, nearest first: in dense forest seen from high up
        // the tiles out to the horizon hold more trees than the pool; the farthest are left to the
        // terrain's canopy colour instead of staying requested (and missing) for ever.
        // (resident tiles stay: only new requests must fit what is left, so arriving tiles never
        // move the cut and evict others that would be requested again)
        {
            near.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let budget = forest::CAPACITY as f64 * 0.85;
            let mut sum: f64 = near.iter().filter_map(|x| self.trees.get(&x.1)).map(|t| t.count as f64).sum();
            let mut full = false;
            near.retain(|(_, id)| {
                if self.trees.contains_key(id) {
                    return true;
                }
                if full {
                    return false;
                }
                let est = self.recs_mean[(id.level == forest::UNDER_LEVEL) as usize] * 1.2;
                if sum + est > budget {
                    full = true;
                    return false;
                }
                sum += est;
                true
            });
        }
        self.keep = near.iter().map(|x| x.1).collect();
        // Hand-over: a resident tree tile that left the list stays (drawn) while a tile replacing
        // it over the same ground, finer or coarser, is still growing; no bare patches meanwhile.
        self.held = 0;
        let missing: HashSet<TileId> =
            near.iter().map(|x| x.1).filter(|id| id.level <= forest::TREE_LEVEL && !self.trees.contains_key(id)).collect();
        if !missing.is_empty() {
            let mut above = HashSet::new(); // ancestors of missing tiles
            for id in &missing {
                let mut p = id.parent();
                while let Some(q) = p {
                    if !above.insert(q) {
                        break;
                    }
                    p = q.parent();
                }
            }
            let held: Vec<TileId> = self
                .trees
                .keys()
                .filter(|id| id.level <= forest::TREE_LEVEL && !self.keep.contains(*id))
                .filter(|id| {
                    above.contains(*id) || {
                        let mut p = id.parent();
                        loop {
                            match p {
                                Some(q) if missing.contains(&q) => break true,
                                Some(q) => p = q.parent(),
                                None => break false,
                            }
                        }
                    }
                })
                .copied()
                .collect();
            self.held = held.len();
            self.keep.extend(held);
        }
        {
            // resident or finished and waiting for a frame with room: not requested again
            let trees = &self.trees;
            let queued: HashSet<TileId> = self.tree_backlog.iter().map(|t| t.id).collect();
            let mut order: Vec<(f64, TileId)> = match ahead {
                Some(a) => near.iter().map(|x| (tiles::tile_distance(x.1, a), x.1)).collect(),
                None => Vec::new(),
            };
            order.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let order = if ahead.is_some() { &order } else { &near };
            requeue(&self.queue, 1, &mut self.tree_pending, order.iter().map(|x| x.1), |id| trees.contains_key(id) || queued.contains(id), 8192);
        }
        let ms_list = t.elapsed().as_secs_f64() * 1e3;

        let t = std::time::Instant::now();
        let (draw, mut want) = {
            let tiles = &self.tiles;
            tiles::select(cam, &|id| tiles.contains_key(&id), &self.terrain)
        };
        if let Some(a) = ahead {
            for w in want.iter_mut() {
                w.0 = tiles::tile_distance(w.1, a);
            }
            want.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        }
        {
            let tiles = &self.tiles;
            let queued: HashSet<TileId> = self.backlog.iter().map(|m| m.id).collect();
            requeue(&self.queue, 0, &mut self.pending, want.iter().map(|x| x.1), |id| tiles.contains_key(id) || queued.contains(id), 1024);
        }
        self.waiting = want.len();
        let r = glam::Mat3::from_cols(rot.x_axis.as_vec3(), rot.y_axis.as_vec3(), rot.z_axis.as_vec3());
        let place = |c: DVec3| glam::Affine3A::from_mat3_translation(r, (rot * (c - cam)).as_vec3());
        let n = draw.len().min(MAX_TERRAIN_TILES);
        self.table_cpu.clear();
        for (k, id) in draw.iter().take(n).enumerate() {
            let t = self.tiles.get_mut(id).unwrap();
            t.last_used = self.frame; // not evicted this frame
            out.push(accel::make_instance(
                &place(t.center),
                k as u32,
                1,
                SBT_GLOBE_TERRAIN,
                vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE | vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE,
                t.blas_addr,
            ));
            self.table_cpu.push(t.vtx_addr);
        }
        self.drawn = n;
        // The ancestors of drawn tiles are the fallback where the ground ahead is not in yet, and
        // select() only descends into a tile while all four children are resident (also the ones
        // over the horizon, never drawn): keep both, or a whole region drops to a coarse tile
        // and refines back down one round trip per level.
        for id in draw.iter().take(n) {
            let mut p = id.parent();
            while let Some(q) = p {
                match self.tiles.get_mut(&q) {
                    Some(t) if t.last_used != self.frame => t.last_used = self.frame,
                    _ => break,
                }
                for k in q.children() {
                    if let Some(t) = self.tiles.get_mut(&k) {
                        t.last_used = self.frame;
                    }
                }
                p = q.parent();
            }
        }
        let mut trees_drawn = 0usize;
        for (id, t) in &self.trees {
            // tiles outside `keep` are evicted after the fence: not drawn
            if let (Some((_, addr, _)), true) = (&t.blas, self.keep.contains(id)) {
                out.push(accel::make_instance(&place(t.center), t.base, 2, SBT_IMPOSTOR, vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE, *addr));
                trees_drawn += t.count as usize;
            }
        }
        self.trees_drawn = trees_drawn;
        if std::env::var("SI_STREAM_LOG").is_ok() {
            let (lat, lon) = tiles::lat_lon(cam.normalize());
            println!(
                "stream {} {:.4},{:.4} eye {:.0} | terrain drawn {} want {} resident {} pending {} backlog {} | trees drawn {} tiles {} keep {} held {} pending {} backlog {}",
                self.frame, lat, lon, eye, self.drawn, self.waiting, self.tiles.len(), self.pending.len(), self.backlog.len(),
                self.trees_drawn, self.trees.len(), self.keep.len(), self.held, self.tree_pending.len(), self.tree_backlog.len()
            );
        }
        [ms_list, t.elapsed().as_secs_f64() * 1e3]
    }

    /// GPU half, after the previous frame's fence: this frame's terrain tile table, evictions,
    /// and finished tiles staged with their BLAS builds recorded for this frame (record()). Tiles
    /// uploaded here are drawn from the next frame on. Returns CPU ms: terrain, trees.
    pub fn update(&mut self, gpu: &Gpu) -> Result<[f64; 2]> {
        self.begin_frame();
        // the newest ocean frame: its textures into the frame's staging, copied in record()
        if let Some(f) = self.ocean.take() {
            for (k, data) in f.tex.iter().enumerate() {
                if let Some(src) = self.stage(bytemuck::cast_slice(data)) {
                    self.ops.push(Op::Image { src, img: self.ocean_tex[k].raw });
                }
            }
            let s = f.state;
            self.ocean_stats = [
                [s.wind, f.residual, f.amplitude, super::ocean::CHOP],
                [s.dir[0], s.dir[1], f.coverage, 1.0],
                f.slope_var,
            ];
        }
        self.table.mapped()[..self.table_cpu.len() * 8].copy_from_slice(bytemuck::cast_slice(&self.table_cpu));
        while self.backlog.len() < UPLOADS_PER_FRAME {
            match self.done_rx.try_recv() {
                Ok(m) => {
                    self.pending.remove(&m.id);
                    if !self.tiles.contains_key(&m.id) {
                        self.backlog.push_back(m);
                    }
                }
                Err(_) => break,
            }
        }
        let tu = std::time::Instant::now();
        self.evict(gpu);
        let ms_terrain = tu.elapsed().as_secs_f64() * 1e3;

        // Finished tree tiles join the backlog (up to a budget of trees per frame).
        let tt = std::time::Instant::now();
        let keep = std::mem::take(&mut self.keep);
        let mut budget: usize = self.tree_backlog.iter().map(|t| t.recs.len()).sum();
        while self.tree_backlog.len() < TREE_UPLOADS_PER_FRAME && budget < TREES_PER_FRAME {
            match self.tree_rx.try_recv() {
                Ok(t) => {
                    self.tree_pending.remove(&t.id);
                    let k = (t.id.level == forest::UNDER_LEVEL) as usize;
                    self.recs_mean[k] += (t.recs.len() as f64 - self.recs_mean[k]) * 0.02;
                    RECEIVED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if !keep.contains(&t.id) {
                        DROPPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    if keep.contains(&t.id) {
                        budget += t.recs.len();
                        self.tree_backlog.push_back(t);
                    }
                }
                Err(_) => break,
            }
        }
        self.tree_backlog.retain(|t| keep.contains(&t.id));
        self.evict_trees(gpu, &keep);
        // (evicted tiles are gone before their compaction would be recorded)
        self.compact(gpu)?;
        if self.frame % 30 == 0 {
            self.as_heap.trim(gpu, 2);
            self.pool.trim(gpu, 1);
        }
        self.upload(gpu)?;
        self.upload_trees(gpu)?;
        self.keep = keep;
        Ok([ms_terrain, tt.elapsed().as_secs_f64() * 1e3])
    }

    /// Trees of the finest tiles within `radius` of the camera (ECEF): (plant record index, tile
    /// centre, record: object -> tile rows, variant | seed << 8).
    pub fn near_trees(&self, cam: DVec3, radius: f64, mut f: impl FnMut(u32, DVec3, &[u32; forest::REC_WORDS])) {
        let edge = 0.5 * std::f64::consts::PI * R_EARTH / (1u64 << forest::TREE_LEVEL) as f64;
        let ground = cam.normalize() * R_EARTH; // tile centres are at sea level: compare along the ground
        for (id, t) in &self.trees {
            if t.recs.is_empty() || !self.keep.contains(id) || (t.center - ground).length() > radius + edge {
                continue;
            }
            for (i, r) in t.recs.iter().enumerate() {
                f(t.base + i as u32, t.center, r);
            }
        }
    }

    /// Distance (m) from the camera (ECEF) to the nearest plant of each variant within `radius`
    /// (finest tree and understory tiles; `out` keeps its value where a variant is not near).
    pub fn variants_near(&self, cam: DVec3, radius: f64, out: &mut [f32]) {
        let edge = 0.5 * std::f64::consts::PI * R_EARTH / (1u64 << forest::TREE_LEVEL) as f64;
        let ground = cam.normalize() * R_EARTH;
        for (id, t) in &self.trees {
            if t.recs.is_empty() || !self.keep.contains(id) || (t.center - ground).length() > radius + edge {
                continue;
            }
            for r in &t.recs {
                let v = (r[12] & 0xFF) as usize;
                let p = t.center + DVec3::new(f32::from_bits(r[3]) as f64, f32::from_bits(r[7]) as f64, f32::from_bits(r[11]) as f64);
                let d = (p - cam).length() as f32;
                if v < out.len() && d < out[v] {
                    out[v] = d;
                }
            }
        }
    }

    pub fn write_globals(&mut self, g: &GlobeGlobals) {
        self.ubo.mapped()[..std::mem::size_of::<GlobeGlobals>()].copy_from_slice(bytemuck::bytes_of(g));
    }

    /// Residency summary for the benchmark output.
    pub fn stats_line(&self) -> String {
        format!(
            "globe: {} terrain tiles resident ({} vertex pool chunks, {:.0} MB), {} drawn; {} tree tiles, {} trees drawn; BLAS heap (terrain + trees, compacted) {} MB",
            self.tiles.len(),
            self.pool.chunks(),
            (self.pool.chunks() as u64 * POOL_CHUNK as u64 * self.pool.vslot) as f64 / 1e6,
            self.drawn,
            self.trees.len(),
            self.trees_drawn,
            self.as_heap.chunk_count() as u64 * (AS_CHUNK >> 20)
        ) + &{
            let w = |k: usize, i: usize| WORK[k][i].load(std::sync::atomic::Ordering::Relaxed) as f64;
            format!(
                "
globe workers: {} terrain tiles at {:.2} ms ({:.2} ms shoreline field), {} tree tiles at {:.2} ms (thread time each)",
                w(0, 1),
                w(0, 0) / w(0, 1).max(1.0) / 1e6,
                tiles::FIELD_NS.load(std::sync::atomic::Ordering::Relaxed) as f64 / w(0, 1).max(1.0) / 1e6,
                w(1, 1),
                w(1, 0) / w(1, 1).max(1.0) / 1e6
            ) + &{
                let u = |k: usize, i: usize| UNDER[k][i].load(std::sync::atomic::Ordering::Relaxed) as f64;
                format!(
                    " [{} tree tiles at {:.2} ms, {} understory tiles at {:.2} ms]",
                    u(0, 1),
                    u(0, 0) / u(0, 1).max(1.0) / 1e6,
                    u(1, 1),
                    u(1, 0) / u(1, 1).max(1.0) / 1e6
                )
            }
        }
    }
}

/// Earth radius for the shaders.
pub const RADIUS: f32 = R_EARTH as f32;

/// Terrain tile geometry (vertex buffer address, shared index buffer).
fn terrain_geometry(vtx: u64, idx: u64, nverts: u32) -> vk::AccelerationStructureGeometryKHR<'static> {
    vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .flags(vk::GeometryFlagsKHR::OPAQUE)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                .vertex_format(vk::Format::R32G32B32_SFLOAT)
                .vertex_data(vk::DeviceOrHostAddressConstKHR { device_address: vtx })
                .vertex_stride((VSTRIDE * 4) as u64)
                .max_vertex(nverts - 1)
                .index_type(vk::IndexType::UINT32)
                .index_data(vk::DeviceOrHostAddressConstKHR { device_address: idx }),
        })
}
