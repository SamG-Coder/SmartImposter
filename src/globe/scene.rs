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
use crate::accel::{self, Accel};
use crate::gpu::{Buffer, Gpu, Image};
use anyhow::{Context, Result};
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
const TREE_UPLOADS_PER_FRAME: usize = 128;
/// Trees uploaded per frame at most (~3 ms of BLAS builds).
const TREES_PER_FRAME: usize = 120_000;
/// Terrain tiles in the tile table (globe.rchit indexes it by instance custom index).
pub const MAX_TERRAIN_TILES: usize = 16_384;

/// Hit records in the main pipeline (renderer.rs create_rt_pipeline).
pub const SBT_IMPOSTOR: u32 = 1;
pub const SBT_GLOBE_TERRAIN: u32 = 3;

struct TileGpu {
    buf: Buffer,
    blas: Accel,
    center: DVec3,
    last_used: u64,
}

struct TreeTileGpu {
    /// first plant record (u32::MAX: no trees)
    base: u32,
    count: u32,
    blas: Option<Accel>,
    aabbs: Option<Buffer>,
    center: DVec3,
    /// finest tiles: the plant records on the CPU too (mesh LOD near the camera)
    recs: Vec<[u32; 16]>,
}

type Queue = Arc<(Mutex<VecDeque<TileId>>, Condvar)>;

fn spawn_workers<T: Send + 'static>(n: usize, queue: &Queue, tx: mpsc::Sender<T>, work: impl Fn(TileId) -> T + Send + Sync + 'static) {
    let work = Arc::new(work);
    for _ in 0..n {
        let (q, tx, work) = (queue.clone(), tx.clone(), work.clone());
        std::thread::spawn(move || loop {
            let id = {
                let (lock, cv) = &*q;
                let mut g = lock.lock().unwrap();
                loop {
                    if let Some(id) = g.pop_front() {
                        break id;
                    }
                    g = cv.wait(g).unwrap();
                }
            };
            if tx.send(work(id)).is_err() {
                return;
            }
        });
    }
}

/// Replace a work queue's not-yet-started entries with `want` (nearest first).
fn requeue(queue: &Queue, pending: &mut HashSet<TileId>, want: impl Iterator<Item = TileId>, have: impl Fn(&TileId) -> bool, limit: usize) {
    let (lock, cv) = &**queue;
    let mut q = lock.lock().unwrap();
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
    tree_queue: Queue,
    tree_rx: mpsc::Receiver<TreeTile>,
    idx: Buffer,
    tri_count: u32,
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

        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let queue: Queue = Arc::new((Mutex::new(VecDeque::new()), Condvar::new()));
        let (done_tx, done_rx) = mpsc::channel();
        let et = terrain.clone();
        spawn_workers(cores.saturating_sub(5).max(2), &queue, done_tx, move |id| tiles::build_tile(id, &et));
        let tree_queue: Queue = Arc::new((Mutex::new(VecDeque::new()), Condvar::new()));
        let (tree_tx, tree_rx) = mpsc::channel();
        let (et, roles, sp) = (terrain.clone(), Arc::new(Roles::new(&species)), Arc::new(species));
        spawn_workers(4, &tree_queue, tree_tx, move |id| {
            if id.level == forest::UNDER_LEVEL {
                forest::grow_understory(id, &et, &sp)
            } else {
                forest::grow_tile(id, &et, &sp, &roles)
            }
        });

        let mut s = GlobeScene {
            terrain: terrain.clone(),
            tiles: HashMap::new(),
            pending: HashSet::new(),
            queue,
            done_rx,
            trees: HashMap::new(),
            records: RangeAlloc::new(forest::CAPACITY),
            tree_pending: HashSet::new(),
            tree_queue,
            tree_rx,
            idx,
            tri_count: (indices.len() / 3) as u32,
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
        s.upload(gpu, meshes)?;
        Ok(s)
    }

    pub fn index_addr(&self) -> u64 {
        self.idx.addr
    }

    /// Terrain and tree tiles still being built.
    pub fn busy(&self) -> bool {
        self.waiting > 0 || !self.tree_pending.is_empty()
    }

    fn upload(&mut self, gpu: &Gpu, meshes: Vec<TileMesh>) -> Result<()> {
        if meshes.is_empty() {
            return Ok(());
        }
        let mut bufs = Vec::with_capacity(meshes.len());
        let mut geoms = Vec::with_capacity(meshes.len());
        for m in &meshes {
            let bytes: &[u8] = bytemuck::cast_slice(&m.verts);
            let mut buf = gpu.create_buffer(
                bytes.len() as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                MemoryLocation::CpuToGpu,
                "tile vertices",
            )?;
            buf.mapped()[..bytes.len()].copy_from_slice(bytes);
            let nv = (m.verts.len() / VSTRIDE) as u32;
            let g = vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .flags(vk::GeometryFlagsKHR::OPAQUE)
                .geometry(vk::AccelerationStructureGeometryDataKHR {
                    triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                        .vertex_format(vk::Format::R32G32B32_SFLOAT)
                        .vertex_data(vk::DeviceOrHostAddressConstKHR { device_address: buf.addr })
                        .vertex_stride((VSTRIDE * 4) as u64)
                        .max_vertex(nv - 1)
                        .index_type(vk::IndexType::UINT32)
                        .index_data(vk::DeviceOrHostAddressConstKHR { device_address: self.idx.addr }),
                });
            geoms.push((g, self.tri_count));
            bufs.push(buf);
        }
        let blases = accel::build_blases(gpu, &geoms).context("building tile BLASes")?;
        for ((m, buf), blas) in meshes.into_iter().zip(bufs).zip(blases) {
            self.tiles.insert(m.id, TileGpu { buf, blas, center: m.center, last_used: self.frame });
        }
        Ok(())
    }

    fn upload_trees(&mut self, gpu: &Gpu, grown: Vec<TreeTile>) -> Result<()> {
        let grown: Vec<TreeTile> = grown.into_iter().filter(|t| !self.trees.contains_key(&t.id)).collect();
        if grown.is_empty() {
            return Ok(());
        }
        let total: usize = grown.iter().map(|t| t.recs.len()).sum();
        let mut staging = gpu.create_buffer((total.max(1) * 64) as u64, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu, "plant staging")?;
        let mut copies = Vec::new();
        let mut geoms = Vec::new();
        let mut entries = Vec::new();
        let mut off = 0usize;
        for t in &grown {
            let n = t.recs.len();
            // empty tiles (no forest, or no land cover data there) take no records
            let base = if n > 0 {
                match self.records.alloc(n as u32) {
                    Some(b) => b,
                    None => {
                        if std::env::var("SI_TREE_STATS").is_ok() {
                            println!("  plant records full ({} trees resident)", self.trees.values().map(|t| t.count as usize).sum::<usize>());
                        }
                        break;
                    }
                }
            } else {
                u32::MAX
            };
            let mut aabbs = None;
            if n > 0 {
                staging.mapped()[off * 64..(off + n) * 64].copy_from_slice(bytemuck::cast_slice(&t.recs));
                copies.push(vk::BufferCopy { src_offset: (off * 64) as u64, dst_offset: base as u64 * 64, size: (n * 64) as u64 });
                off += n;
                let bytes: &[u8] = bytemuck::cast_slice(&t.aabbs);
                let mut ab = gpu.create_buffer(bytes.len() as u64, vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR, MemoryLocation::CpuToGpu, "tree aabbs")?;
                ab.mapped()[..bytes.len()].copy_from_slice(bytes);
                geoms.push((accel::aabb_geometry(ab.addr), n as u32));
                aabbs = Some(ab);
            }
            let keep_recs = if t.id.level >= forest::TREE_LEVEL { t.recs.clone() } else { Vec::new() };
            entries.push((t.id, base, n as u32, aabbs, t.center, keep_recs));
        }
        if !copies.is_empty() {
            let (src, dst) = (staging.raw, self.plants.raw);
            gpu.one_time(|cb| unsafe { gpu.device.cmd_copy_buffer(cb, src, dst, &copies) })?;
        }
        gpu.destroy_buffer(&mut staging);
        let mut blases = accel::build_blases(gpu, &geoms)?.into_iter();
        for (id, base, count, aabbs, center, recs) in entries {
            let blas = if count > 0 { blases.next() } else { None };
            self.trees.insert(id, TreeTileGpu { base, count, blas, aabbs, center, recs });
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
            if let Some(mut t) = self.tiles.remove(&id) {
                accel::destroy_accel(gpu, &mut t.blas);
                gpu.destroy_buffer(&mut t.buf);
            }
        }
    }

    fn evict_trees(&mut self, gpu: &Gpu, keep: &HashSet<TileId>) {
        let gone: Vec<TileId> = self.trees.keys().filter(|id| !keep.contains(id)).copied().collect();
        for id in gone {
            if let Some(mut t) = self.trees.remove(&id) {
                if let Some(mut b) = t.blas.take() {
                    accel::destroy_accel(gpu, &mut b);
                }
                if let Some(mut a) = t.aabbs.take() {
                    gpu.destroy_buffer(&mut a);
                }
                if t.base != u32::MAX {
                    self.records.release(t.base, t.count);
                }
            }
        }
    }

    /// Stream: upload finished tiles, evict far ones, request the ones the camera at `cam` (ECEF)
    /// needs next. Call after the previous frame's fence (GPU buffers are freed here).
    pub fn update(&mut self, gpu: &Gpu, cam: DVec3) -> Result<()> {
        self.frame += 1;
        let mut got = Vec::new();
        while got.len() < UPLOADS_PER_FRAME {
            match self.done_rx.try_recv() {
                Ok(m) => {
                    self.pending.remove(&m.id);
                    if !self.tiles.contains_key(&m.id) {
                        got.push(m);
                    }
                }
                Err(_) => break,
            }
        }
        let n_got = got.len();
        let tu = std::time::Instant::now();
        self.upload(gpu, got)?;
        self.evict(gpu);
        if std::env::var("SI_TREE_STATS").is_ok() && tu.elapsed().as_secs_f64() > 0.004 {
            println!("  terrain upload {} tiles {:.1} ms (cached {})", n_got, tu.elapsed().as_secs_f64() * 1e3, self.tiles.len());
        }

        let t = std::time::Instant::now();
        let eye = cam.length() - tiles::ground_radius(&self.terrain, cam);
        self.range = forest::tree_range(eye);
        let mut near = forest::under_tiles(cam);
        near.extend(forest::tiles_near(cam, self.range));
        if std::env::var("SI_TREE_STATS").is_ok() && self.frame % 60 == 0 {
            println!("  tiles_near: {} tiles in {:.2} ms", near.len(), t.elapsed().as_secs_f64() * 1e3);
        }
        let keep: HashSet<TileId> = near.iter().map(|x| x.1).collect();
        // A frame takes finished tiles up to a budget of trees (BLAS builds are synchronous).
        let mut grown = Vec::new();
        let mut budget = 0usize;
        while grown.len() < TREE_UPLOADS_PER_FRAME && budget < TREES_PER_FRAME {
            match self.tree_rx.try_recv() {
                Ok(t) => {
                    self.tree_pending.remove(&t.id);
                    if keep.contains(&t.id) {
                        budget += t.recs.len();
                        grown.push(t);
                    }
                }
                Err(_) => break,
            }
        }
        let n_grown = grown.len();
        let tt = std::time::Instant::now();
        self.evict_trees(gpu, &keep);
        self.upload_trees(gpu, grown)?;
        if std::env::var("SI_TREE_STATS").is_ok() && tt.elapsed().as_secs_f64() > 0.004 {
            println!("  tree upload {} tiles {:.1} ms", n_grown, tt.elapsed().as_secs_f64() * 1e3);
        }
        {
            let trees = &self.trees;
            requeue(&self.tree_queue, &mut self.tree_pending, near.iter().map(|x| x.1), |id| trees.contains_key(id), 8192);
        }
        Ok(())
    }

    /// TLAS instances of the terrain and tree tiles in the camera frame (camera at `cam`, ECEF;
    /// `rot` from frame_at). Also fills the terrain tile table.
    pub fn instances(&mut self, cam: DVec3, rot: DMat3, out: &mut Vec<vk::AccelerationStructureInstanceKHR>) {
        let (draw, want) = {
            let tiles = &self.tiles;
            tiles::select(cam, &|id| tiles.contains_key(&id), &self.terrain)
        };
        {
            let tiles = &self.tiles;
            requeue(&self.queue, &mut self.pending, want.iter().map(|x| x.1), |id| tiles.contains_key(id), 1024);
        }
        self.waiting = want.len();
        let r = glam::Mat3::from_cols(rot.x_axis.as_vec3(), rot.y_axis.as_vec3(), rot.z_axis.as_vec3());
        let place = |c: DVec3| glam::Affine3A::from_mat3_translation(r, (rot * (c - cam)).as_vec3());
        let n = draw.len().min(MAX_TERRAIN_TILES);
        let mut table: Vec<u64> = Vec::with_capacity(n);
        for (k, id) in draw.iter().take(n).enumerate() {
            let t = self.tiles.get_mut(id).unwrap();
            t.last_used = self.frame;
            out.push(accel::make_instance(
                &place(t.center),
                k as u32,
                1,
                SBT_GLOBE_TERRAIN,
                vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE | vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE,
                t.blas.addr,
            ));
            table.push(t.buf.addr);
        }
        self.drawn = n;
        self.table.mapped()[..table.len() * 8].copy_from_slice(bytemuck::cast_slice(&table));
        let mut trees_drawn = 0usize;
        for t in self.trees.values() {
            if let Some(b) = &t.blas {
                out.push(accel::make_instance(&place(t.center), t.base, 2, SBT_IMPOSTOR, vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE, b.addr));
                trees_drawn += t.count as usize;
            }
        }
        self.trees_drawn = trees_drawn;
    }

    /// Trees of the finest tiles within `radius` of the camera (ECEF): (plant record index, tile
    /// centre, record: object -> tile rows, variant | seed << 8).
    pub fn near_trees(&self, cam: DVec3, radius: f64, mut f: impl FnMut(u32, DVec3, &[u32; 16])) {
        let edge = 0.5 * std::f64::consts::PI * R_EARTH / (1u64 << forest::TREE_LEVEL) as f64;
        let ground = cam.normalize() * R_EARTH; // tile centres are at sea level: compare along the ground
        for t in self.trees.values() {
            if t.recs.is_empty() || (t.center - ground).length() > radius + edge {
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

    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }
}

/// Earth radius for the shaders.
pub const RADIUS: f32 = R_EARTH as f32;
