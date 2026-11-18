//! Globe renderer: one BLAS per terrain tile, a TLAS rebuilt every frame with the tiles placed
//! relative to the camera (float64 on the CPU, float32 on the GPU), tiles built on worker threads
//! and uploaded as they arrive, and one ray tracing pipeline (globe.rgen / globe.rchit).

use super::dem::Terrain;
use super::tiles::{self, TileId, TileMesh, VSTRIDE};
use crate::accel::{self, Accel, Tlas};
use crate::gpu::{align_up, Buffer, Gpu, Image, Swapchain};
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
    /// x exposure, y frame index, z debug view, w unused
    pub params: [f32; 4],
    /// xy shared index buffer address (lo, hi)
    pub addr: [u32; 4],
}

const MAX_TILES_DRAWN: u32 = 16_384;
/// Ground scans (catalog.rs GROUND_LAYERS) x albedo / normal / ORM. Must match globe.rgen.
pub const GROUND_TEX: u32 = 36;
const CACHE_TILES: usize = 14_000;
const UPLOADS_PER_FRAME: usize = 160;

struct TileGpu {
    buf: Buffer,
    blas: Accel,
    center: DVec3,
    last_used: u64,
}

struct Pipe {
    pipe: vk::Pipeline,
    rgen: vk::StridedDeviceAddressRegionKHR,
    miss: vk::StridedDeviceAddressRegionKHR,
    hit: vk::StridedDeviceAddressRegionKHR,
    _sbt: Buffer,
}

type Queue = Arc<(Mutex<VecDeque<TileId>>, Condvar)>;

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
    cb: vk::CommandBuffer,
    fence: vk::Fence,
    acquire_sem: vk::Semaphore,
    tiles: HashMap<TileId, TileGpu>,
    terrain: Arc<Terrain>,
    pending: HashSet<TileId>,
    queue: Queue,
    done_rx: mpsc::Receiver<TileMesh>,
    frame: u64,
    vsync: bool,
    pub need_resize: bool,
    /// Tiles drawn / waiting for the last frame.
    pub drawn: usize,
    pub waiting: usize,
}

impl GlobeRenderer {
    pub fn new(display: RawDisplayHandle, window: RawWindowHandle, w: u32, h: u32, terrain: Arc<Terrain>, vsync: bool) -> Result<Self> {
        let gpu = Gpu::new(display, window, false)?;
        println!("GPU: {}", gpu.device_name);
        let swap = gpu.create_swapchain(w, h, vk::SwapchainKHR::null(), vsync)?;
        let (out, readback) = Self::create_out(&gpu, swap.extent)?;
        let d = gpu.device.clone();

        // descriptors: TLAS, output image, globals, tile table
        let b = |i: u32, ty: vk::DescriptorType, st: vk::ShaderStageFlags| {
            vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(ty).descriptor_count(1).stage_flags(st)
        };
        let rt = vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR | vk::ShaderStageFlags::MISS_KHR;
        let bindings = [
            b(0, vk::DescriptorType::ACCELERATION_STRUCTURE_KHR, rt),
            b(1, vk::DescriptorType::STORAGE_IMAGE, rt),
            b(2, vk::DescriptorType::UNIFORM_BUFFER, rt),
            b(3, vk::DescriptorType::STORAGE_BUFFER, rt),
            b(4, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, rt),
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(GROUND_TEX)
                .stage_flags(rt),
        ];
        let set_layout = unsafe { d.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)? };
        let pipe_layout = unsafe { d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]), None)? };
        let sizes = [
            vk::DescriptorPoolSize { ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR, descriptor_count: 1 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: 1 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: 1 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 1 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: 1 + GROUND_TEX },
        ];
        let pool = unsafe { d.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None)? };
        let set = unsafe { d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[set_layout]))?[0] };
        let pipe = Self::create_pipeline(&gpu, pipe_layout)?;

        let ubo = gpu.create_buffer(256, vk::BufferUsageFlags::UNIFORM_BUFFER, MemoryLocation::CpuToGpu, "globe globals")?;
        let table = gpu.create_buffer(MAX_TILES_DRAWN as u64 * 8, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "tile table")?;
        let tlas = Tlas::new(&gpu, MAX_TILES_DRAWN)?;
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
        let sampler = unsafe {
            d.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::REPEAT)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .max_lod(vk::LOD_CLAMP_NONE),
                None,
            )?
        };
        println!("Elevation grid on the GPU ({:.1}s)", t.elapsed().as_secs_f32());
        // Quixel ground scans (the forest scene's layers): missing ones are 1x1 placeholders
        let found = crate::catalog::find_ground();
        let ground_mask = found.iter().enumerate().map(|(i, g)| (g.is_some() as u32) << i).sum::<u32>();
        let mut ground = crate::textures::load_ground(&gpu, &found)?;
        while ground.len() < GROUND_TEX as usize {
            ground.push(crate::textures::upload_with_mips(&gpu, 1, 1, &[128, 128, 128, 255], vk::Format::R8G8B8A8_UNORM, "ground placeholder")?);
        }
        ground.truncate(GROUND_TEX as usize);
        let ground_sampler = unsafe {
            d.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::REPEAT)
                    .address_mode_v(vk::SamplerAddressMode::REPEAT)
                    .max_lod(vk::LOD_CLAMP_NONE),
                None,
            )?
        };
        let cb = unsafe {
            d.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default().command_pool(gpu.cmd_pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1),
            )?[0]
        };
        let fence = unsafe { d.create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None)? };
        let acquire_sem = unsafe { d.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)? };

        // tile builders
        let queue: Queue = Arc::new((Mutex::new(VecDeque::new()), Condvar::new()));
        let (done_tx, done_rx) = mpsc::channel();
        let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).saturating_sub(2).max(2);
        for _ in 0..workers {
            let (q, tx, et) = (queue.clone(), done_tx.clone(), terrain.clone());
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
                if tx.send(tiles::build_tile(id, &et)).is_err() {
                    return;
                }
            });
        }

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
            table,
            tlas,
            idx,
            tri_count,
            elev_tex,
            sampler,
            ground,
            ground_sampler,
            ground_mask,
            cb,
            fence,
            acquire_sem,
            tiles: HashMap::new(),
            terrain: terrain.clone(),
            pending: HashSet::new(),
            queue,
            done_rx,
            frame: 0,
            vsync,
            need_resize: false,
            drawn: 0,
            waiting: 0,
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
        ];
        let kinds = [
            vk::ShaderStageFlags::RAYGEN_KHR,
            vk::ShaderStageFlags::MISS_KHR,
            vk::ShaderStageFlags::MISS_KHR,
            vk::ShaderStageFlags::CLOSEST_HIT_KHR,
        ];
        let stages: Vec<_> =
            modules.iter().zip(kinds).map(|(m, k)| vk::PipelineShaderStageCreateInfo::default().stage(k).module(*m).name(c"main")).collect();
        let u = vk::SHADER_UNUSED_KHR;
        let general = |s| {
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(s)
                .closest_hit_shader(u)
                .any_hit_shader(u)
                .intersection_shader(u)
        };
        let groups = [
            general(0),
            general(1),
            general(2),
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                .general_shader(u)
                .closest_hit_shader(3)
                .any_hit_shader(u)
                .intersection_shader(u),
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
        let total = off_hit + stride;
        let handles = unsafe { gpu.rt_fn.get_ray_tracing_shader_group_handles(pipe, 0, 4, 4 * hs as usize)? };
        let mut sbt = gpu.create_buffer(total + ba, vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR, MemoryLocation::CpuToGpu, "globe sbt")?;
        let base = align_up(sbt.addr, ba);
        let shift = (base - sbt.addr) as usize;
        {
            let m = sbt.mapped();
            for (g, off) in [(0usize, 0u64), (1, off_miss), (2, off_miss + stride), (3, off_hit)] {
                let dst = shift + off as usize;
                m[dst..dst + hs as usize].copy_from_slice(&handles[g * hs as usize..(g + 1) * hs as usize]);
            }
        }
        let region = |off: u64, size: u64| vk::StridedDeviceAddressRegionKHR { device_address: base + off, stride, size };
        Ok(Pipe { pipe, rgen: region(0, stride), miss: region(off_miss, 2 * stride), hit: region(off_hit, stride), _sbt: sbt })
    }

    fn write_set(&self) {
        let accels = [self.tlas.accel.raw];
        let mut as_info = vk::WriteDescriptorSetAccelerationStructureKHR::default().acceleration_structures(&accels);
        let img = [vk::DescriptorImageInfo::default().image_view(self.out.view).image_layout(vk::ImageLayout::GENERAL)];
        let ubo = [vk::DescriptorBufferInfo::default().buffer(self.ubo.raw).range(vk::WHOLE_SIZE)];
        let table = [vk::DescriptorBufferInfo::default().buffer(self.table.raw).range(vk::WHOLE_SIZE)];
        let elev = [vk::DescriptorImageInfo::default().sampler(self.sampler).image_view(self.elev_tex.view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let ground: Vec<_> = self
            .ground
            .iter()
            .map(|g| vk::DescriptorImageInfo::default().sampler(self.ground_sampler).image_view(g.view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL))
            .collect();
        let mut tlas_write =
            vk::WriteDescriptorSet::default().dst_set(self.set).dst_binding(0).descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR).push_next(&mut as_info);
        tlas_write.descriptor_count = 1;
        let writes = [
            tlas_write,
            vk::WriteDescriptorSet::default().dst_set(self.set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&img),
            vk::WriteDescriptorSet::default().dst_set(self.set).dst_binding(2).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&ubo),
            vk::WriteDescriptorSet::default().dst_set(self.set).dst_binding(3).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&table),
            vk::WriteDescriptorSet::default().dst_set(self.set).dst_binding(4).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&elev),
            vk::WriteDescriptorSet::default().dst_set(self.set).dst_binding(5).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&ground),
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

        // finished tiles -> GPU
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

        // which tiles to draw, which to build next
        let (draw, want) = {
            let tiles = &self.tiles;
            tiles::select(cam, &|id| tiles.contains_key(&id), &self.terrain)
        };
        {
            let (lock, cv) = &*self.queue;
            let mut q = lock.lock().unwrap();
            for id in q.drain(..) {
                self.pending.remove(&id); // not started: re-prioritised below
            }
            for (_, id) in want.iter().take(1024) {
                if !self.pending.contains(id) && !self.tiles.contains_key(id) {
                    self.pending.insert(*id);
                    q.push_back(*id);
                }
            }
            cv.notify_all();
        }
        self.waiting = want.len();

        let n = draw.len().min(MAX_TILES_DRAWN as usize);
        let mut insts = Vec::with_capacity(n);
        let mut table: Vec<u64> = Vec::with_capacity(n);
        for (k, id) in draw.iter().take(n).enumerate() {
            let t = self.tiles.get_mut(id).unwrap();
            t.last_used = self.frame;
            let off = t.center - cam;
            let m = glam::Affine3A::from_translation(glam::Vec3::new(off.x as f32, off.y as f32, off.z as f32));
            insts.push(accel::make_instance(
                &m,
                k as u32,
                0xFF,
                0,
                vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE | vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE,
                t.blas.addr,
            ));
            table.push(t.buf.addr);
        }
        self.drawn = n;
        self.tlas.write(&insts);
        self.table.mapped()[..table.len() * 8].copy_from_slice(bytemuck::cast_slice(&table));
        g.addr = [self.idx.addr as u32, (self.idx.addr >> 32) as u32, 0, 0];
        g.params[1] = self.frame as f32;
        g.params[3] = self.ground_mask as f32;
        self.ubo.mapped()[..std::mem::size_of::<GlobeGlobals>()].copy_from_slice(bytemuck::bytes_of(&g));

        let idx = match unsafe { self.gpu.swapchain_fn.acquire_next_image(self.swap.raw, u64::MAX, self.acquire_sem, vk::Fence::null()) } {
            Ok((i, _)) => i,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.need_resize = true;
                return Ok(None);
            }
            Err(e) => return Err(e.into()),
        };
        unsafe { d.reset_fences(&[self.fence])? };
        let (w, h) = self.extent();
        let cb = self.cb;
        let swap_img = self.swap.images[idx as usize];
        unsafe {
            d.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;
            d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            self.tlas.cmd_build(&self.gpu, cb, n as u32);
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
