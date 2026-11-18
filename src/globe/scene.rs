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
const UPLOADS_PER_FRAME: usize = 160;
const TREE_UPLOADS_PER_FRAME: usize = 192;
/// Trees uploaded per frame at most (BLAS builds on the GPU, recorded in the frame).
const TREES_PER_FRAME: usize = 200_000;
/// Per-frame staging (host -> device copies) and BLAS build scratch.
const STAGING: u64 = 48 << 20;
const SCRATCH: u64 = 96 << 20;
/// Terrain tiles in the tile table (globe.rchit indexes it by instance custom index).
pub const MAX_TERRAIN_TILES: usize = 16_384;

/// Hit records in the main pipeline (renderer.rs create_rt_pipeline).
pub const SBT_IMPOSTOR: u32 = 1;
pub const SBT_GLOBE_TERRAIN: u32 = 3;

struct TileGpu {
    /// slot in the vertex and BLAS pools
    slot: u32,
    blas: vk::AccelerationStructureKHR,
    blas_addr: u64,
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
    recs: Vec<[u32; 16]>,
}

/// Terrain tile storage: fixed-size slots for vertices and BLASes in device-local chunks of
/// POOL_CHUNK tiles, added as tiles become resident (up to `cap`), freed slots reused.
struct TilePool {
    vtx: Vec<Buffer>,
    blas: Vec<Buffer>,
    vslot: u64,
    bslot: u64,
    free: Vec<u32>,
    cap: u32,
}

const POOL_CHUNK: u32 = 512;

impl TilePool {
    fn new(vtx_bytes: u64, blas_bytes: u64, cap: u32) -> Self {
        TilePool { vtx: Vec::new(), blas: Vec::new(), vslot: vtx_bytes.div_ceil(256) * 256, bslot: blas_bytes.div_ceil(256) * 256, free: Vec::new(), cap }
    }
    /// A free slot (adding a chunk when all are taken), or None at capacity.
    fn alloc(&mut self, gpu: &Gpu) -> Result<Option<u32>> {
        let have = self.vtx.len() as u32 * POOL_CHUNK;
        if self.free.is_empty() && have < self.cap {
            let vu = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR | vk::BufferUsageFlags::TRANSFER_DST;
            self.vtx.push(gpu.create_buffer(self.vslot * POOL_CHUNK as u64, vu, MemoryLocation::GpuOnly, "terrain vertex pool")?);
            let bu = vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR;
            self.blas.push(gpu.create_buffer(self.bslot * POOL_CHUNK as u64, bu, MemoryLocation::GpuOnly, "terrain blas pool")?);
            self.free.extend((have..have + POOL_CHUNK).rev());
        }
        Ok(self.free.pop())
    }
    fn vtx(&self, i: u32) -> (&Buffer, u64) {
        (&self.vtx[(i / POOL_CHUNK) as usize], (i % POOL_CHUNK) as u64 * self.vslot)
    }
    fn blas(&self, i: u32) -> (&Buffer, u64) {
        (&self.blas[(i / POOL_CHUNK) as usize], (i % POOL_CHUNK) as u64 * self.bslot)
    }
}

/// Where a tree BLAS lives in the AS heap: size class, chunk, offset.
#[derive(Clone, Copy)]
struct HeapSlot {
    class: usize,
    chunk: usize,
    offset: u64,
}

/// Acceleration structure storage for variable-size BLASes: power-of-two size classes carved
/// out of 64 MB device chunks, freed slots reused per class (no allocator calls per tile).
struct AsHeap {
    chunks: Vec<Buffer>,
    bump: u64,
    free: Vec<Vec<(usize, u64)>>,
}

const AS_CHUNK: u64 = 64 << 20;
const AS_MIN_CLASS: u64 = 4 << 10;

impl AsHeap {
    fn new() -> Self {
        AsHeap { chunks: Vec::new(), bump: AS_CHUNK, free: vec![Vec::new(); 16] }
    }
    fn alloc(&mut self, gpu: &Gpu, size: u64) -> Result<HeapSlot> {
        let csize = size.max(AS_MIN_CLASS).next_power_of_two();
        let class = (csize / AS_MIN_CLASS).trailing_zeros() as usize;
        if let Some((chunk, offset)) = self.free[class].pop() {
            return Ok(HeapSlot { class, chunk, offset });
        }
        if self.bump + csize > AS_CHUNK {
            self.chunks.push(gpu.create_buffer(AS_CHUNK, vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR, MemoryLocation::GpuOnly, "tree blas heap")?);
            self.bump = 0;
        }
        let s = HeapSlot { class, chunk: self.chunks.len() - 1, offset: self.bump };
        self.bump += csize;
        Ok(s)
    }
    fn release(&mut self, s: HeapSlot) {
        self.free[s.class].push((s.chunk, s.offset));
    }
}

/// GPU work recorded into the frame's command buffer (renderer.rs, before the TLAS build).
enum Op {
    Copy { src: u64, dst: vk::Buffer, dst_off: u64, size: u64 },
    Terrain { vtx: u64, dst: vk::AccelerationStructureKHR, scratch: u64 },
    Trees { aabbs: u64, count: u32, dst: vk::AccelerationStructureKHR, scratch: u64 },
}

/// Work queues (0 terrain tiles, 1 tree tiles), served by one pool of workers.
type Queue = Arc<(Mutex<[VecDeque<TileId>; 2]>, Condvar)>;

/// Worker time (ns) and tiles made: terrain, trees.
static WORK: [[std::sync::atomic::AtomicU64; 2]; 2] =
    [[std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)], [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)]];

/// `n` workers taking terrain tiles first (ground holes are worse than late trees), tree tiles
/// otherwise, so the whole pool goes where the work is.
fn spawn_workers(
    n: usize,
    queue: &Queue,
    tx: (mpsc::Sender<TileMesh>, mpsc::Sender<TreeTile>),
    terrain: impl Fn(TileId) -> TileMesh + Send + Sync + 'static,
    trees: impl Fn(TileId) -> TreeTile + Send + Sync + 'static,
) {
    let work = Arc::new((terrain, trees));
    for _ in 0..n {
        let (q, tx, work) = (queue.clone(), tx.clone(), work.clone());
        std::thread::spawn(move || loop {
            let (kind, id) = {
                let (lock, cv) = &*q;
                let mut g = lock.lock().unwrap();
                loop {
                    if let Some(id) = g[0].pop_front() {
                        break (0, id);
                    }
                    if let Some(id) = g[1].pop_front() {
                        break (1, id);
                    }
                    g = cv.wait(g).unwrap();
                }
            };
            let t = std::time::Instant::now();
            let sent = if kind == 0 { tx.0.send((work.0)(id)).is_ok() } else { tx.1.send((work.1)(id)).is_ok() };
            WORK[kind][0].fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            WORK[kind][1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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

pub struct GlobeScene {
    pub terrain: Arc<Terrain>,
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
    blas_size: u64,
    terrain_scratch: u64,
    as_heap: AsHeap,
    aabbs: Buffer,
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
        let plants = gpu.create_buffer(recs * 64, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, MemoryLocation::GpuOnly, "globe plants")?;
        let mut plant_state = gpu.create_buffer(recs * 4, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "globe plant state")?;
        plant_state.mapped().fill(0);
        // pools: every terrain tile has the same vertex count and topology, so the same BLAS size
        let tri_count = (indices.len() / 3) as u32;
        let nverts = (tiles::GRID * tiles::GRID + 4 * tiles::GRID) as u32;
        let sample = terrain_geometry(0, idx.addr, nverts);
        let (blas_size, terrain_scratch) = accel::blas_sizes(gpu, &sample, tri_count);
        let mut pool = TilePool::new((nverts as usize * VSTRIDE * 4) as u64, blas_size, (CACHE_TILES + 512) as u32);
        // a typical view keeps 5-9k tiles: allocate the first chunks now, not mid-flight
        let mut first = Vec::new();
        for _ in 0..8 * POOL_CHUNK {
            first.extend(pool.alloc(gpu)?);
        }
        pool.free.extend(first.into_iter().rev());
        let aabbs = gpu.create_buffer(
            recs * 24,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            "tree aabbs",
        )?;
        let staging = gpu.create_buffer(STAGING, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu, "globe staging")?;
        let scratch = gpu.create_buffer(SCRATCH + 4096, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, "globe build scratch")?;
        println!(
            "Globe pools: terrain tiles {:.0} KB vertices + {:.0} KB BLAS each (chunks of {}), {:.0} MB plant records + AABBs",
            pool.vslot as f64 / 1e3,
            pool.bslot as f64 / 1e3,
            POOL_CHUNK,
            (recs * 88) as f64 / 1e6
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
                if id.level == forest::UNDER_LEVEL {
                    forest::grow_understory(id, &et2, &sp)
                } else {
                    forest::grow_tile(id, &et2, &sp, &roles)
                }
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
            blas_size,
            terrain_scratch,
            as_heap: AsHeap::new(),
            aabbs,
            staging,
            staging_used: 0,
            scratch,
            scratch_used: 0,
            scratch_align: gpu.rt.scratch_alignment.max(256),
            ops: Vec::new(),
            backlog: VecDeque::new(),
            tree_backlog: VecDeque::new(),
            elev_tex,
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
    pub fn record(&mut self, gpu: &Gpu, cb: vk::CommandBuffer) {
        if self.ops.is_empty() {
            return;
        }
        let d = &gpu.device;
        let staging = self.staging.raw;
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
                Op::Copy { .. } => {}
            }
        }
        gpu.memory_barrier(
            cb,
            vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
            vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR | vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
        );
    }

    pub fn index_addr(&self) -> u64 {
        self.idx.addr
    }

    /// Terrain and tree tiles still being built.
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
            let (bbuf, boff) = self.pool.blas(slot);
            let (blas, blas_addr) = accel::create_accel_in(gpu, bbuf.raw, boff, self.blas_size, vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)?;
            let (vbuf, off) = self.pool.vtx(slot);
            let vtx_addr = vbuf.addr + off;
            self.ops.push(Op::Copy { src, dst: vbuf.raw, dst_off: off, size: bytes.len() as u64 });
            self.ops.push(Op::Terrain { vtx: vtx_addr, dst: blas, scratch });
            self.tiles.insert(m.id, TileGpu { slot, blas, blas_addr, vtx_addr, center: m.center, last_used: self.frame });
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
            let need = (n * 64 + n * 24 + 512) as u64;
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
            self.ops.push(Op::Copy { src: src_r, dst: self.plants.raw, dst_off: base as u64 * 64, size: (n * 64) as u64 });
            self.ops.push(Op::Copy { src: src_a, dst: self.aabbs.raw, dst_off: base as u64 * 24, size: (n * 24) as u64 });
            let hs = self.as_heap.alloc(gpu, as_size)?;
            let (blas, addr) = accel::create_accel_in(gpu, self.as_heap.chunks[hs.chunk].raw, hs.offset, as_size, vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)?;
            self.ops.push(Op::Trees { aabbs: self.aabbs.addr + base as u64 * 24, count: n as u32, dst: blas, scratch });
            let keep_recs = if t.id.level >= forest::TREE_LEVEL { t.recs } else { Vec::new() };
            self.trees.insert(t.id, TreeTileGpu { base, count: n as u32, blas: Some((blas, addr, hs)), center: t.center, recs: keep_recs });
        }
        Ok(())
    }

    fn evict(&mut self, gpu: &Gpu) {
        if self.tiles.len() <= CACHE_TILES {
            return;
        }
        let mut old: Vec<(u64, TileId)> =
            self.tiles.iter().filter(|(id, t)| id.level > 2 && t.last_used + 2 < self.frame).map(|(id, t)| (t.last_used, *id)).collect();
        old.sort();
        let excess = self.tiles.len() - CACHE_TILES * 9 / 10;
        for (_, id) in old.into_iter().take(excess) {
            if let Some(t) = self.tiles.remove(&id) {
                unsafe { gpu.accel_fn.destroy_acceleration_structure(t.blas, None) };
                self.pool.free.push(t.slot);
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
        let eye = cam.length() - tiles::ground_radius(&self.terrain, cam);
        self.range = forest::tree_range(eye);
        let mut near = forest::under_tiles(cam);
        near.extend(forest::tiles_near(cam, self.range));
        self.keep = near.iter().map(|x| x.1).collect();
        // Hand-over: a resident tree tile that left the list stays (drawn) while a tile replacing
        // it over the same ground, finer or coarser, is still growing; no bare patches meanwhile.
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
            self.keep.extend(held);
        }
        {
            // resident or finished and waiting for a frame with room: not requested again
            let trees = &self.trees;
            let queued: HashSet<TileId> = self.tree_backlog.iter().map(|t| t.id).collect();
            requeue(&self.queue, 1, &mut self.tree_pending, near.iter().map(|x| x.1), |id| trees.contains_key(id) || queued.contains(id), 8192);
        }
        let ms_list = t.elapsed().as_secs_f64() * 1e3;

        let t = std::time::Instant::now();
        let (draw, want) = {
            let tiles = &self.tiles;
            tiles::select(cam, &|id| tiles.contains_key(&id), &self.terrain)
        };
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
        let mut trees_drawn = 0usize;
        for (id, t) in &self.trees {
            // tiles outside `keep` are evicted after the fence: not drawn
            if let (Some((_, addr, _)), true) = (&t.blas, self.keep.contains(id)) {
                out.push(accel::make_instance(&place(t.center), t.base, 2, SBT_IMPOSTOR, vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE, *addr));
                trees_drawn += t.count as usize;
            }
        }
        self.trees_drawn = trees_drawn;
        [ms_list, t.elapsed().as_secs_f64() * 1e3]
    }

    /// GPU half, after the previous frame's fence: this frame's terrain tile table, evictions,
    /// and finished tiles staged with their BLAS builds recorded for this frame (record()). Tiles
    /// uploaded here are drawn from the next frame on. Returns CPU ms: terrain, trees.
    pub fn update(&mut self, gpu: &Gpu) -> Result<[f64; 2]> {
        self.begin_frame();
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
        self.upload(gpu)?;
        let ms_terrain = tu.elapsed().as_secs_f64() * 1e3;

        // Finished tree tiles join the backlog (up to a budget of trees per frame).
        let tt = std::time::Instant::now();
        let keep = std::mem::take(&mut self.keep);
        let mut budget: usize = self.tree_backlog.iter().map(|t| t.recs.len()).sum();
        while self.tree_backlog.len() < TREE_UPLOADS_PER_FRAME && budget < TREES_PER_FRAME {
            match self.tree_rx.try_recv() {
                Ok(t) => {
                    self.tree_pending.remove(&t.id);
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
        self.upload_trees(gpu)?;
        self.keep = keep;
        Ok([ms_terrain, tt.elapsed().as_secs_f64() * 1e3])
    }

    /// Trees of the finest tiles within `radius` of the camera (ECEF): (plant record index, tile
    /// centre, record: object -> tile rows, variant | seed << 8).
    pub fn near_trees(&self, cam: DVec3, radius: f64, mut f: impl FnMut(u32, DVec3, &[u32; 16])) {
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

    pub fn write_globals(&mut self, g: &GlobeGlobals) {
        self.ubo.mapped()[..std::mem::size_of::<GlobeGlobals>()].copy_from_slice(bytemuck::bytes_of(g));
    }

    /// Residency summary for the benchmark output.
    pub fn stats_line(&self) -> String {
        format!(
            "globe: {} terrain tiles resident ({} pool chunks, {:.0} MB), {} drawn; {} tree tiles, {} trees drawn; tree BLAS heap {} MB",
            self.tiles.len(),
            self.pool.vtx.len(),
            (self.pool.vtx.len() as u64 * POOL_CHUNK as u64 * (self.pool.vslot + self.pool.bslot)) as f64 / 1e6,
            self.drawn,
            self.trees.len(),
            self.trees_drawn,
            self.as_heap.chunks.len() as u64 * (AS_CHUNK >> 20)
        ) + &{
            let w = |k: usize, i: usize| WORK[k][i].load(std::sync::atomic::Ordering::Relaxed) as f64;
            format!(
                "
globe workers: {} terrain tiles at {:.2} ms, {} tree tiles at {:.2} ms (thread time each)",
                w(0, 1),
                w(0, 0) / w(0, 1).max(1.0) / 1e6,
                w(1, 1),
                w(1, 0) / w(1, 1).max(1.0) / 1e6
            )
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
