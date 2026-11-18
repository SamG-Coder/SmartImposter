//! Globe renderer: one BLAS per terrain tile, a TLAS rebuilt every frame with the tiles placed
//! relative to the camera (float64 on the CPU, float32 on the GPU), tiles built on worker threads
//! and uploaded as they arrive, trees near the camera as impostors (per-tile cluster BLAS of tree
//! AABBs, the forest scene's impostor shaders and baked atlases), one ray tracing pipeline.

use super::dem::Terrain;
use super::forest::{self, Impostors, Roles, TreeTile};
use super::tiles::{self, TileId, TileMesh, VSTRIDE};
use crate::accel::{self, Accel, Tlas};
use crate::gpu::{align_up, Buffer, Gpu, Image, Swapchain};
use crate::renderer::Globals;
use anyhow::{Context, Result};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::DVec3;
use gpu_allocator::MemoryLocation;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{mpsc, Arc, Condvar, Mutex};

macro_rules! spv {
    ($n:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $n, ".spv"))
    };
}

/// Must match globe.glsl GlobeGlobals.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct GlobeGlobals {
    pub cam_right: [f32; 4],
    pub cam_up: [f32; 4],
    /// w = tan(fov_y / 2)
    pub cam_fwd: [f32; 4],
    /// xyz direction to the sun, w = sun illuminance scale
    pub sun: [f32; 4],
    /// xyz planet centre relative to the camera, w = radius
    pub planet: [f32; 4],
    /// x exposure, y frame index, z debug view, w ground scans present (bit mask)
    pub params: [f32; 4],
    /// x pixel angle (rad)
    pub misc: [f32; 4],
    /// xy shared index buffer address (lo, hi)
    pub addr: [u32; 4],
}

const MAX_INSTANCES: u32 = 20_000;
const CACHE_TILES: usize = 14_000;
const UPLOADS_PER_FRAME: usize = 160;
const TREE_UPLOADS_PER_FRAME: usize = 128;
/// Ground scans (catalog.rs GROUND_LAYERS) x albedo / normal / ORM. Must match globe.rchit.
pub const GROUND_TEX: u32 = 36;
/// Impostor atlas slots (shared.glsl MAX_VARIANTS).
const MAX_VARIANTS: u32 = crate::renderer::MAX_VARIANTS as u32;

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
}

struct Pipe {
    pipe: vk::Pipeline,
    rgen: vk::StridedDeviceAddressRegionKHR,
    miss: vk::StridedDeviceAddressRegionKHR,
    hit: vk::StridedDeviceAddressRegionKHR,
    _sbt: Buffer,
}

type Queue = Arc<(Mutex<VecDeque<TileId>>, Condvar)>;

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
        // merge with the neighbours
        if i + 1 < self.free.len() && self.free[i].0 + self.free[i].1 == self.free[i + 1].0 {
            self.free[i].1 += self.free[i + 1].1;
            self.free.remove(i + 1);
        }
        if i > 0 && self.free[i - 1].0 + self.free[i - 1].1 == self.free[i].0 {
            self.free[i - 1].1 += self.free[i].1;
            self.free.remove(i);
        }
    }
    fn free_records(&self) -> u32 {
        self.free.iter().map(|r| r.1).sum()
    }
}

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

pub struct GlobeRenderer {
    pub gpu: Gpu,
    pub swap: Swapchain,
    out: Image,
    readback: Buffer,
    set_layout: vk::DescriptorSetLayout,
    pipe_layout: vk::PipelineLayout,
    pipe: Pipe,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    ubo: Buffer,
    fubo: Buffer,
    table: Buffer,
    tlas: Tlas,
    idx: Buffer,
    tri_count: u32,
    elev_tex: Image,
    sampler: vk::Sampler,
    ground: Vec<Image>,
    ground_sampler: vk::Sampler,
    /// bit per ground scan present
    pub ground_mask: u32,
    imps: Impostors,
    imp_info: Buffer,
    atlas_sampler: vk::Sampler,
    plants: Buffer,
    plant_state: Buffer,
    cb: vk::CommandBuffer,
    fence: vk::Fence,
    acquire_sem: vk::Semaphore,
    tiles: HashMap<TileId, TileGpu>,
    terrain: Arc<Terrain>,
    pending: HashSet<TileId>,
    queue: Queue,
    done_rx: mpsc::Receiver<TileMesh>,
    trees: HashMap<TileId, TreeTileGpu>,
    records: RangeAlloc,
    tree_pending: HashSet<TileId>,
    tree_queue: Queue,
    tree_rx: mpsc::Receiver<TreeTile>,
    frame: u64,
    vsync: bool,
    pub need_resize: bool,
    /// Tiles drawn / waiting for the last frame.
    pub drawn: usize,
    pub waiting: usize,
    /// Trees in the TLAS for the last frame.
    pub trees_drawn: usize,
}

impl GlobeRenderer {
    pub fn new(display: RawDisplayHandle, window: RawWindowHandle, w: u32, h: u32, terrain: Arc<Terrain>, vsync: bool) -> Result<Self> {
        let gpu = Gpu::new(display, window, false)?;
        println!("GPU: {}", gpu.device_name);
        let swap = gpu.create_swapchain(w, h, vk::SwapchainKHR::null(), vsync)?;
        let (out, readback) = Self::create_out(&gpu, swap.extent)?;
        let d = gpu.device.clone();

        // descriptors: the impostor shaders' bindings (rt.glsl / impostor.rint: 3 globals, 5 impostor
        // infos, 6 / 7 / 11 / 14 atlases, 23 plant records, 24 plant LOD state) plus the globe's own
        let b = |i: u32, ty: vk::DescriptorType, n: u32| {
            let st = vk::ShaderStageFlags::RAYGEN_KHR
                | vk::ShaderStageFlags::CLOSEST_HIT_KHR
                | vk::ShaderStageFlags::MISS_KHR
                | vk::ShaderStageFlags::INTERSECTION_KHR;
            vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(ty).descriptor_count(n).stage_flags(st)
        };
        use vk::DescriptorType as DT;
        let bindings = [
            b(0, DT::ACCELERATION_STRUCTURE_KHR, 1),
            b(1, DT::STORAGE_IMAGE, 1),
            b(3, DT::UNIFORM_BUFFER, 1),
            b(5, DT::STORAGE_BUFFER, 1),
            b(6, DT::COMBINED_IMAGE_SAMPLER, MAX_VARIANTS),
            b(7, DT::COMBINED_IMAGE_SAMPLER, MAX_VARIANTS),
            b(11, DT::COMBINED_IMAGE_SAMPLER, MAX_VARIANTS),
            b(14, DT::COMBINED_IMAGE_SAMPLER, MAX_VARIANTS),
            b(23, DT::STORAGE_BUFFER, 1),
            b(24, DT::STORAGE_BUFFER, 1),
            b(30, DT::UNIFORM_BUFFER, 1),
            b(31, DT::STORAGE_BUFFER, 1),
            b(32, DT::COMBINED_IMAGE_SAMPLER, 1),
            b(33, DT::COMBINED_IMAGE_SAMPLER, GROUND_TEX),
        ];
        let set_layout = unsafe { d.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)? };
        let pipe_layout = unsafe { d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]), None)? };
        let sizes = [
            vk::DescriptorPoolSize { ty: DT::ACCELERATION_STRUCTURE_KHR, descriptor_count: 1 },
            vk::DescriptorPoolSize { ty: DT::STORAGE_IMAGE, descriptor_count: 1 },
            vk::DescriptorPoolSize { ty: DT::UNIFORM_BUFFER, descriptor_count: 2 },
            vk::DescriptorPoolSize { ty: DT::STORAGE_BUFFER, descriptor_count: 4 },
            vk::DescriptorPoolSize { ty: DT::COMBINED_IMAGE_SAMPLER, descriptor_count: 4 * MAX_VARIANTS + 1 + GROUND_TEX },
        ];
        let pool = unsafe { d.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None)? };
        let set = unsafe { d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[set_layout]))?[0] };
        let pipe = Self::create_pipeline(&gpu, pipe_layout)?;

        let ubo = gpu.create_buffer(256, vk::BufferUsageFlags::UNIFORM_BUFFER, MemoryLocation::CpuToGpu, "globe globals")?;
        let fubo = gpu.create_buffer(std::mem::size_of::<Globals>() as u64, vk::BufferUsageFlags::UNIFORM_BUFFER, MemoryLocation::CpuToGpu, "impostor globals")?;
        let table = gpu.create_buffer(MAX_INSTANCES as u64 * 8, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "tile table")?;
        let tlas = Tlas::new(&gpu, MAX_INSTANCES)?;
        let indices = tiles::index_buffer();
        let idx = gpu.upload_buffer(
            bytemuck::cast_slice(&indices),
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            "tile indices",
        )?;
        let tri_count = (indices.len() / 3) as u32;
        // The whole elevation grid on the GPU (int16 metres as R16_SNORM, with mips): per-pixel
        // coastlines and relief shading wherever the tile mesh is coarser than the data.
        let t = std::time::Instant::now();
        let elev_tex = crate::textures::upload_with_mips(
            &gpu,
            terrain.etopo.w as u32,
            terrain.etopo.h as u32,
            bytemuck::cast_slice(terrain.etopo.raw()),
            vk::Format::R16_SNORM,
            "etopo",
        )?;
        let linear = |u: vk::SamplerAddressMode, v: vk::SamplerAddressMode| -> Result<vk::Sampler> {
            Ok(unsafe {
                d.create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::LINEAR)
                        .min_filter(vk::Filter::LINEAR)
                        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                        .address_mode_u(u)
                        .address_mode_v(v)
                        .max_lod(vk::LOD_CLAMP_NONE),
                    None,
                )?
            })
        };
        let sampler = linear(vk::SamplerAddressMode::REPEAT, vk::SamplerAddressMode::CLAMP_TO_EDGE)?;
        println!("Elevation grid on the GPU ({:.1}s)", t.elapsed().as_secs_f32());
        // Quixel ground scans (the forest scene's layers): missing ones are 1x1 placeholders
        let found = crate::catalog::find_ground();
        let ground_mask = found.iter().enumerate().map(|(i, g)| (g.is_some() as u32) << i).sum::<u32>();
        let mut ground = crate::textures::load_ground(&gpu, &found)?;
        while ground.len() < GROUND_TEX as usize {
            ground.push(crate::textures::upload_with_mips(&gpu, 1, 1, &[128, 128, 128, 255], vk::Format::R8G8B8A8_UNORM, "ground placeholder")?);
        }
        ground.truncate(GROUND_TEX as usize);
        let ground_sampler = linear(vk::SamplerAddressMode::REPEAT, vk::SamplerAddressMode::REPEAT)?;
        let atlas_sampler = linear(vk::SamplerAddressMode::CLAMP_TO_EDGE, vk::SamplerAddressMode::CLAMP_TO_EDGE)?;

        // trees: the forest scene's baked impostors, plant record slots
        let imps = forest::load_impostors(&gpu)?;
        let imp_info = gpu.upload_buffer(bytemuck::cast_slice(&imps.infos), vk::BufferUsageFlags::STORAGE_BUFFER, "globe impostor info")?;
        let recs = forest::CAPACITY as u64;
        let plants = gpu.create_buffer(recs * 64, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, MemoryLocation::GpuOnly, "globe plants")?;
        let plant_state = gpu.upload_buffer(&vec![0u8; recs as usize * 4], vk::BufferUsageFlags::STORAGE_BUFFER, "globe plant state")?;

        let cb = unsafe {
            d.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default().command_pool(gpu.cmd_pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1),
            )?[0]
        };
        let fence = unsafe { d.create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None)? };
        let acquire_sem = unsafe { d.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)? };

        // terrain tile builders and tree tile growers
        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let queue: Queue = Arc::new((Mutex::new(VecDeque::new()), Condvar::new()));
        let (done_tx, done_rx) = mpsc::channel();
        let et = terrain.clone();
        spawn_workers(cores.saturating_sub(4).max(2), &queue, done_tx, move |id| tiles::build_tile(id, &et));
        let tree_queue: Queue = Arc::new((Mutex::new(VecDeque::new()), Condvar::new()));
        let (tree_tx, tree_rx) = mpsc::channel();
        let (et, roles) = (terrain.clone(), Arc::new(Roles::new(&imps)));
        let info: Arc<forest::Impostors> = Arc::new(Impostors {
            albedo: vec![],
            normal: vec![],
            depth: vec![],
            vis: vec![],
            infos: imps.infos.clone(),
            species: imps.species.clone(),
        });
        spawn_workers(4, &tree_queue, tree_tx, move |id| forest::grow_tile(id, &et, &info, &roles));

        let mut r = GlobeRenderer {
            gpu,
            swap,
            out,
            readback,
            set_layout,
            pipe_layout,
            pipe,
            pool,
            set,
            ubo,
            fubo,
            table,
            tlas,
            idx,
            tri_count,
            elev_tex,
            sampler,
            ground,
            ground_sampler,
            ground_mask,
            imps,
            imp_info,
            atlas_sampler,
            plants,
            plant_state,
            cb,
            fence,
            acquire_sem,
            tiles: HashMap::new(),
            terrain: terrain.clone(),
            pending: HashSet::new(),
            queue,
            done_rx,
            trees: HashMap::new(),
            records: RangeAlloc::new(forest::CAPACITY),
            tree_pending: HashSet::new(),
            tree_queue,
            tree_rx,
            frame: 0,
            vsync,
            need_resize: false,
            drawn: 0,
            waiting: 0,
            trees_drawn: 0,
        };
        // the coarse levels everywhere, built up front so the globe is never missing pieces
        let mut ids = Vec::new();
        let mut level = (0..6).map(TileId::root).collect::<Vec<_>>();
        for _ in 0..=2 {
            ids.extend_from_slice(&level);
            level = level.iter().flat_map(|t| t.children()).collect();
        }
        let meshes: Vec<TileMesh> = std::thread::scope(|s| {
            let hs: Vec<_> = ids.chunks(16).map(|c| s.spawn(|| c.iter().map(|&id| tiles::build_tile(id, &terrain)).collect::<Vec<_>>())).collect();
            hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        r.upload(meshes)?;
        r.write_set();
        Ok(r)
    }

    fn create_out(gpu: &Gpu, e: vk::Extent2D) -> Result<(Image, Buffer)> {
        let out = gpu.create_image(
            e.width,
            e.height,
            1,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            "globe out",
        )?;
        gpu.one_time(|cb| {
            gpu.image_barrier(
                cb,
                out.raw,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::SHADER_WRITE,
            )
        })?;
        let readback = gpu.create_buffer(e.width as u64 * e.height as u64 * 4, vk::BufferUsageFlags::TRANSFER_DST, MemoryLocation::GpuToCpu, "readback")?;
        Ok((out, readback))
    }

    fn create_pipeline(gpu: &Gpu, layout: vk::PipelineLayout) -> Result<Pipe> {
        let modules = [
            gpu.shader_module(spv!("globe.rgen"))?,
            gpu.shader_module(spv!("globe.rmiss"))?,
            gpu.shader_module(spv!("shadow.rmiss"))?,
            gpu.shader_module(spv!("globe.rchit"))?,
            gpu.shader_module(spv!("impostor.rint"))?,
            gpu.shader_module(spv!("impostor.rchit"))?,
        ];
        let kinds = [
            vk::ShaderStageFlags::RAYGEN_KHR,
            vk::ShaderStageFlags::MISS_KHR,
            vk::ShaderStageFlags::MISS_KHR,
            vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            vk::ShaderStageFlags::INTERSECTION_KHR,
            vk::ShaderStageFlags::CLOSEST_HIT_KHR,
        ];
        let stages: Vec<_> =
            modules.iter().zip(kinds).map(|(m, k)| vk::PipelineShaderStageCreateInfo::default().stage(k).module(*m).name(c"main")).collect();
        let u = vk::SHADER_UNUSED_KHR;
        let group = |ty, general, closest, intersection| {
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(ty)
                .general_shader(general)
                .closest_hit_shader(closest)
                .any_hit_shader(u)
                .intersection_shader(intersection)
        };
        use vk::RayTracingShaderGroupTypeKHR as G;
        // hit records: 0 terrain triangles, 1 impostor trees
        let groups = [
            group(G::GENERAL, 0, u, u),
            group(G::GENERAL, 1, u, u),
            group(G::GENERAL, 2, u, u),
            group(G::TRIANGLES_HIT_GROUP, u, 3, u),
            group(G::PROCEDURAL_HIT_GROUP, u, 5, 4),
        ];
        let info = vk::RayTracingPipelineCreateInfoKHR::default().stages(&stages).groups(&groups).max_pipeline_ray_recursion_depth(1).layout(layout);
        let pipe = unsafe {
            gpu.rt_fn.create_ray_tracing_pipelines(vk::DeferredOperationKHR::null(), vk::PipelineCache::null(), &[info], None).map_err(|(_, e)| e)?[0]
        };
        for m in modules {
            unsafe { gpu.device.destroy_shader_module(m, None) };
        }
        let hs = gpu.rt.handle_size as u64;
        let stride = align_up(hs, gpu.rt.handle_alignment as u64);
        let ba = gpu.rt.base_alignment as u64;
        let off_miss = align_up(stride, ba);
        let off_hit = align_up(off_miss + 2 * stride, ba);
        let total = off_hit + 2 * stride;
        let handles = unsafe { gpu.rt_fn.get_ray_tracing_shader_group_handles(pipe, 0, 5, 5 * hs as usize)? };
        let mut sbt = gpu.create_buffer(total + ba, vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR, MemoryLocation::CpuToGpu, "globe sbt")?;
        let base = align_up(sbt.addr, ba);
        let shift = (base - sbt.addr) as usize;
        {
            let m = sbt.mapped();
            for (g, off) in [(0usize, 0u64), (1, off_miss), (2, off_miss + stride), (3, off_hit), (4, off_hit + stride)] {
                let dst = shift + off as usize;
                m[dst..dst + hs as usize].copy_from_slice(&handles[g * hs as usize..(g + 1) * hs as usize]);
            }
        }
        let region = |off: u64, size: u64| vk::StridedDeviceAddressRegionKHR { device_address: base + off, stride, size };
        Ok(Pipe { pipe, rgen: region(0, stride), miss: region(off_miss, 2 * stride), hit: region(off_hit, 2 * stride), _sbt: sbt })
    }

    fn write_set(&self) {
        let accels = [self.tlas.accel.raw];
        let mut as_info = vk::WriteDescriptorSetAccelerationStructureKHR::default().acceleration_structures(&accels);
        let img = [vk::DescriptorImageInfo::default().image_view(self.out.view).image_layout(vk::ImageLayout::GENERAL)];
        let buf = |b: &Buffer| [vk::DescriptorBufferInfo::default().buffer(b.raw).range(vk::WHOLE_SIZE)];
        let (ubo, fubo, table, infos, plants, state) =
            (buf(&self.ubo), buf(&self.fubo), buf(&self.table), buf(&self.imp_info), buf(&self.plants), buf(&self.plant_state));
        let tex = |s: vk::Sampler, v: vk::ImageView| vk::DescriptorImageInfo::default().sampler(s).image_view(v).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let elev = [tex(self.sampler, self.elev_tex.view)];
        let ground: Vec<_> = self.ground.iter().map(|g| tex(self.ground_sampler, g.view)).collect();
        // atlas arrays: unused slots repeat the first variant (every descriptor must be valid)
        let atlas = |list: &Vec<Image>| -> Vec<vk::DescriptorImageInfo> {
            (0..MAX_VARIANTS as usize).map(|i| tex(self.atlas_sampler, list[i.min(list.len() - 1)].view)).collect()
        };
        let (albedo, depth, vis, normal) = (atlas(&self.imps.albedo), atlas(&self.imps.depth), atlas(&self.imps.vis), atlas(&self.imps.normal));
        let mut tlas_write =
            vk::WriteDescriptorSet::default().dst_set(self.set).dst_binding(0).descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR).push_next(&mut as_info);
        tlas_write.descriptor_count = 1;
        let w = |binding: u32, ty: vk::DescriptorType| vk::WriteDescriptorSet::default().dst_set(self.set).dst_binding(binding).descriptor_type(ty);
        use vk::DescriptorType as DT;
        let writes = [
            tlas_write,
            w(1, DT::STORAGE_IMAGE).image_info(&img),
            w(3, DT::UNIFORM_BUFFER).buffer_info(&fubo),
            w(5, DT::STORAGE_BUFFER).buffer_info(&infos),
            w(6, DT::COMBINED_IMAGE_SAMPLER).image_info(&albedo),
            w(7, DT::COMBINED_IMAGE_SAMPLER).image_info(&depth),
            w(11, DT::COMBINED_IMAGE_SAMPLER).image_info(&vis),
            w(14, DT::COMBINED_IMAGE_SAMPLER).image_info(&normal),
            w(23, DT::STORAGE_BUFFER).buffer_info(&plants),
            w(24, DT::STORAGE_BUFFER).buffer_info(&state),
            w(30, DT::UNIFORM_BUFFER).buffer_info(&ubo),
            w(31, DT::STORAGE_BUFFER).buffer_info(&table),
            w(32, DT::COMBINED_IMAGE_SAMPLER).image_info(&elev),
            w(33, DT::COMBINED_IMAGE_SAMPLER).image_info(&ground),
        ];
        unsafe { self.gpu.device.update_descriptor_sets(&writes, &[]) };
    }

    /// Vertex buffers (host visible) and one BLAS per tile, all BLAS builds in one submission.
    fn upload(&mut self, meshes: Vec<TileMesh>) -> Result<()> {
        if meshes.is_empty() {
            return Ok(());
        }
        let mut bufs = Vec::with_capacity(meshes.len());
        let mut geoms = Vec::with_capacity(meshes.len());
        for m in &meshes {
            let bytes: &[u8] = bytemuck::cast_slice(&m.verts);
            let mut buf = self.gpu.create_buffer(
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
        let blases = accel::build_blases(&self.gpu, &geoms).context("building tile BLASes")?;
        for ((m, buf), blas) in meshes.into_iter().zip(bufs).zip(blases) {
            self.tiles.insert(m.id, TileGpu { buf, blas, center: m.center, last_used: self.frame });
        }
        Ok(())
    }

    /// Tree tiles: plant records into a free slot of the plant buffer, AABBs, one cluster BLAS each.
    fn upload_trees(&mut self, grown: Vec<TreeTile>) -> Result<()> {
        let grown: Vec<TreeTile> = grown.into_iter().filter(|t| !self.trees.contains_key(&t.id)).collect();
        if grown.is_empty() {
            return Ok(());
        }
        let total: usize = grown.iter().map(|t| t.recs.len()).sum();
        let mut staging = self.gpu.create_buffer((total.max(1) * 64) as u64, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu, "plant staging")?;
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
                    None => break,
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
                let mut ab = self.gpu.create_buffer(bytes.len() as u64, vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR, MemoryLocation::CpuToGpu, "tree aabbs")?;
                ab.mapped()[..bytes.len()].copy_from_slice(bytes);
                geoms.push((accel::aabb_geometry(ab.addr), n as u32));
                aabbs = Some(ab);
            }
            entries.push((t.id, base, n as u32, aabbs, t.center));
        }
        if !copies.is_empty() {
            let (src, dst) = (staging.raw, self.plants.raw);
            self.gpu.one_time(|cb| unsafe { self.gpu.device.cmd_copy_buffer(cb, src, dst, &copies) })?;
        }
        self.gpu.destroy_buffer(&mut staging);
        let mut blases = accel::build_blases(&self.gpu, &geoms)?.into_iter();
        for (id, base, count, aabbs, center) in entries {
            let blas = if count > 0 { blases.next() } else { None };
            self.trees.insert(id, TreeTileGpu { base, count, blas, aabbs, center });
        }
        Ok(())
    }

    fn evict(&mut self) {
        if self.tiles.len() <= CACHE_TILES {
            return;
        }
        let mut old: Vec<(u64, TileId)> =
            self.tiles.iter().filter(|(id, t)| id.level > 2 && t.last_used + 2 < self.frame).map(|(id, t)| (t.last_used, *id)).collect();
        old.sort();
        let excess = self.tiles.len() - CACHE_TILES * 9 / 10;
        for (_, id) in old.into_iter().take(excess) {
            if let Some(mut t) = self.tiles.remove(&id) {
                accel::destroy_accel(&self.gpu, &mut t.blas);
                self.gpu.destroy_buffer(&mut t.buf);
            }
        }
    }

    fn evict_trees(&mut self, keep: &HashSet<TileId>) {
        let gone: Vec<TileId> = self.trees.keys().filter(|id| !keep.contains(id)).copied().collect();
        for id in gone {
            if let Some(mut t) = self.trees.remove(&id) {
                if let Some(mut b) = t.blas.take() {
                    accel::destroy_accel(&self.gpu, &mut b);
                }
                if let Some(mut a) = t.aabbs.take() {
                    self.gpu.destroy_buffer(&mut a);
                }
                if t.base != u32::MAX {
                    self.records.release(t.base, t.count);
                }
            }
        }
    }

    pub fn resize(&mut self, w: u32, h: u32) -> Result<()> {
        unsafe { self.gpu.device.device_wait_idle()? };
        let old = self.swap.raw;
        let new = self.gpu.create_swapchain(w, h, old, self.vsync)?;
        let mut prev = std::mem::replace(&mut self.swap, new);
        self.gpu.destroy_swapchain(&mut prev);
        self.gpu.destroy_image(&mut self.out);
        self.gpu.destroy_buffer(&mut self.readback);
        let (out, readback) = Self::create_out(&self.gpu, self.swap.extent)?;
        self.out = out;
        self.readback = readback;
        self.write_set();
        self.need_resize = false;
        Ok(())
    }

    pub fn extent(&self) -> (u32, u32) {
        (self.swap.extent.width, self.swap.extent.height)
    }

    /// Render one frame with the camera at `cam` (ECEF metres). Returns the RGBA8 image if `capture`.
    pub fn render(&mut self, mut g: GlobeGlobals, cam: DVec3, capture: bool) -> Result<Option<Vec<u8>>> {
        let d = self.gpu.device.clone();
        unsafe { d.wait_for_fences(&[self.fence], true, u64::MAX)? };
        self.frame += 1;

        // finished terrain tiles -> GPU
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
        self.upload(got)?;
        self.evict();

        // trees near the camera
        let eye = cam.length() - tiles::ground_radius(&self.terrain, cam);
        let range = forest::tree_range(eye);
        let near = forest::tiles_near(cam, range);
        if self.frame % 120 == 0 && std::env::var("SI_TREE_STATS").is_ok() {
            println!("trees: eye {eye:.0} m, range {:.0} km, {} tiles wanted, {} resident, {} pending, {:.1}M records free", range / 1000.0, near.len(), self.trees.len(), self.tree_pending.len(), self.records.free_records() as f64 / 1e6);
        }
        let keep: HashSet<TileId> = near.iter().map(|x| x.1).collect();
        let mut grown = Vec::new();
        while grown.len() < TREE_UPLOADS_PER_FRAME {
            match self.tree_rx.try_recv() {
                Ok(t) => {
                    self.tree_pending.remove(&t.id);
                    if keep.contains(&t.id) {
                        grown.push(t);
                    }
                }
                Err(_) => break,
            }
        }
        self.evict_trees(&keep);
        self.upload_trees(grown)?;
        {
            let trees = &self.trees;
            requeue(&self.tree_queue, &mut self.tree_pending, near.iter().map(|x| x.1), |id| trees.contains_key(id), 8192);
        }

        // which terrain tiles to draw, which to build next
        let (draw, want) = {
            let tiles = &self.tiles;
            tiles::select(cam, &|id| tiles.contains_key(&id), &self.terrain)
        };
        {
            let tiles = &self.tiles;
            requeue(&self.queue, &mut self.pending, want.iter().map(|x| x.1), |id| tiles.contains_key(id), 1024);
        }
        self.waiting = want.len();

        let max_tiles = (MAX_INSTANCES as usize).saturating_sub(self.trees.len());
        let n = draw.len().min(max_tiles);
        let mut insts = Vec::with_capacity(n + self.trees.len());
        let mut table: Vec<u64> = Vec::with_capacity(n);
        let place = |c: DVec3| {
            let off = c - cam;
            glam::Affine3A::from_translation(glam::Vec3::new(off.x as f32, off.y as f32, off.z as f32))
        };
        for (k, id) in draw.iter().take(n).enumerate() {
            let t = self.tiles.get_mut(id).unwrap();
            t.last_used = self.frame;
            insts.push(accel::make_instance(
                &place(t.center),
                k as u32,
                0xFF,
                0,
                vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE | vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE,
                t.blas.addr,
            ));
            table.push(t.buf.addr);
        }
        let mut trees_drawn = 0usize;
        for t in self.trees.values() {
            if let Some(b) = &t.blas {
                insts.push(accel::make_instance(&place(t.center), t.base, 0xFF, 1, vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE, b.addr));
                trees_drawn += t.count as usize;
            }
        }
        self.trees_drawn = trees_drawn;
        self.drawn = n;
        self.tlas.write(&insts);
        self.table.mapped()[..table.len() * 8].copy_from_slice(bytemuck::cast_slice(&table));
        g.addr = [self.idx.addr as u32, (self.idx.addr >> 32) as u32, 0, 0];
        g.params[1] = self.frame as f32;
        g.params[3] = self.ground_mask as f32;
        let (w, h) = self.extent();
        g.misc[0] = 2.0 * g.cam_fwd[3] / h as f32;
        g.misc[1] = forest::NEAR as f32;
        g.misc[2] = (forest::SPLIT * 0.5 * std::f64::consts::PI * tiles::R_EARTH / (1u64 << forest::TREE_LEVEL) as f64) as f32;
        g.misc[3] = range as f32;
        self.ubo.mapped()[..std::mem::size_of::<GlobeGlobals>()].copy_from_slice(bytemuck::bytes_of(&g));
        // the impostor shaders' globals: camera at the origin (camera-relative world), pixel
        // spread, frame counter; no mesh band (impostors only)
        let mut fg = Globals::default();
        fg.cam_fwd[3] = g.misc[0];
        fg.iparams[2] = self.frame as u32;
        self.fubo.mapped()[..std::mem::size_of::<Globals>()].copy_from_slice(bytemuck::bytes_of(&fg));

        let idx = match unsafe { self.gpu.swapchain_fn.acquire_next_image(self.swap.raw, u64::MAX, self.acquire_sem, vk::Fence::null()) } {
            Ok((i, _)) => i,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.need_resize = true;
                return Ok(None);
            }
            Err(e) => return Err(e.into()),
        };
        unsafe { d.reset_fences(&[self.fence])? };
        let cb = self.cb;
        let swap_img = self.swap.images[idx as usize];
        unsafe {
            d.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;
            d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            self.tlas.cmd_build(&self.gpu, cb, insts.len() as u32);
            self.gpu.memory_barrier(
                cb,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
                vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                vk::AccessFlags2::SHADER_WRITE,
            );
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::RAY_TRACING_KHR, self.pipe.pipe);
            d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::RAY_TRACING_KHR, self.pipe_layout, 0, &[self.set], &[]);
            self.gpu.rt_fn.cmd_trace_rays(cb, &self.pipe.rgen, &self.pipe.miss, &self.pipe.hit, &vk::StridedDeviceAddressRegionKHR::default(), w, h, 1);
            self.gpu.memory_barrier(
                cb,
                vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                vk::AccessFlags2::SHADER_WRITE,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
            );
            self.gpu.image_barrier(
                cb,
                swap_img,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
            );
            let layers = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 };
            let corner = vk::Offset3D { x: w as i32, y: h as i32, z: 1 };
            d.cmd_blit_image(
                cb,
                self.out.raw,
                vk::ImageLayout::GENERAL,
                swap_img,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::ImageBlit { src_subresource: layers, src_offsets: [vk::Offset3D::default(), corner], dst_subresource: layers, dst_offsets: [vk::Offset3D::default(), corner] }],
                vk::Filter::NEAREST,
            );
            if capture {
                d.cmd_copy_image_to_buffer(
                    cb,
                    self.out.raw,
                    vk::ImageLayout::GENERAL,
                    self.readback.raw,
                    &[vk::BufferImageCopy {
                        buffer_offset: 0,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: layers,
                        image_offset: vk::Offset3D::default(),
                        image_extent: vk::Extent3D { width: w, height: h, depth: 1 },
                    }],
                );
            }
            self.gpu.image_barrier(
                cb,
                swap_img,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::NONE,
                vk::AccessFlags2::NONE,
            );
            d.end_command_buffer(cb)?;
            let waits = [self.acquire_sem];
            let stages = [vk::PipelineStageFlags::TRANSFER];
            let cbs = [cb];
            let signals = [self.swap.done[idx as usize]];
            d.queue_submit(
                self.gpu.queue,
                &[vk::SubmitInfo::default().wait_semaphores(&waits).wait_dst_stage_mask(&stages).command_buffers(&cbs).signal_semaphores(&signals)],
                self.fence,
            )?;
            let swaps = [self.swap.raw];
            let idxs = [idx];
            match self.gpu.swapchain_fn.queue_present(self.gpu.queue, &vk::PresentInfoKHR::default().wait_semaphores(&signals).swapchains(&swaps).image_indices(&idxs)) {
                Ok(_) => {}
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.need_resize = true,
                Err(e) => return Err(e.into()),
            }
        }
        if capture {
            unsafe { d.wait_for_fences(&[self.fence], true, u64::MAX)? };
            let n = (w * h * 4) as usize;
            return Ok(Some(self.readback.mapped()[..n].to_vec()));
        }
        Ok(None)
    }

    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }

    /// Tree tiles still being grown.
    pub fn trees_waiting(&self) -> usize {
        self.tree_pending.len()
    }

    #[allow(dead_code)]
    pub fn destroy(&mut self) {
        unsafe {
            let d = &self.gpu.device;
            let _ = d.device_wait_idle();
            d.destroy_pipeline(self.pipe.pipe, None);
            d.destroy_pipeline_layout(self.pipe_layout, None);
            d.destroy_descriptor_pool(self.pool, None);
            d.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}
