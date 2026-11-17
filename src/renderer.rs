//! Ray tracing renderer: pipelines, impostor baking, per-frame LOD selection and tracing.

use crate::accel::{self, Accel, Tlas};
use crate::asset::{MeshData, TreePack};
use crate::gpu::{align_up, Buffer, Gpu, Image, Swapchain};
use crate::world::World;
use anyhow::{Context, Result};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use gpu_allocator::MemoryLocation;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::time::Instant;

macro_rules! spv {
    ($n:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $n, ".spv"))
    };
}

/// Atlas slots (tree variants x impostor tiers) bound as sampler arrays.
pub const MAX_VARIANTS: usize = 16;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct Globals {
    pub cam_pos: [f32; 4],
    pub cam_right: [f32; 4],
    pub cam_up: [f32; 4],
    pub cam_fwd: [f32; 4],
    pub sun_dir: [f32; 4],
    pub params: [f32; 4],
    pub iparams: [u32; 4],
    /// x mesh<->impostor band width, y hi->lo impostor distance, z its band width, w variant count (set by the renderer)
    pub lod_params: [f32; 4],
    /// Previous frame camera (same layout as cam_pos..cam_fwd) for TAA reprojection.
    pub prev_pos: [f32; 4],
    pub prev_right: [f32; 4],
    pub prev_up: [f32; 4],
    pub prev_fwd: [f32; 4],
}

/// Render-resolution targets: this frame's sample + hit distance, and the TAA history ping-pong.
struct Targets {
    cur: Image,
    depth: Image,
    hist: [Image; 2],
}

/// Per TLAS instance data for the LOD cross-fade shaders (indexed by gl_InstanceID).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InstInfo {
    center: [f32; 3],
    data: u32, // tree index | flags << 24 (bit 0: tree also has mesh instances)
}

const INST_HAS_MESH: u32 = 1 << 24;

/// Per-tree LOD state.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Lod {
    Impostor,
    Both, // inside the mesh <-> impostor band: both copies in the TLAS, shaders dissolve
    Mesh,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ImpostorInfo {
    center_radius: [f32; 4],
    bmin: [f32; 4],
    bmax: [f32; 4],
    grid: u32,
    res: u32,
    mips: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BakePush {
    center_radius: [f32; 4],
    grid: u32,
    res: u32,
    frame_offset: u32,
    ss: u32,
    ao_rays: u32,
    ao_range: f32,
    vis_rays: u32,
    pad: u32,
}

#[derive(Clone, Copy)]
pub struct ImpostorSettings {
    pub grid: u32,
    pub res: u32,
    pub ss: u32,
    pub ao_rays: u32,
    pub ao_range: f32,
    pub vis_rays: u32,
}

/// Per tree variant: albedo+AO, normal+depth+coverage, directional self-visibility.
type Atlas = (Image, Image, Image);

impl ImpostorSettings {
    pub fn mips(&self) -> u32 {
        let (mut m, mut r) = (1, self.res);
        while r % 2 == 0 && r / 2 >= 6 {
            r /= 2;
            m += 1;
        }
        m
    }
}

struct MeshGpu {
    pos: Buffer,
    nrm: Buffer,
    idx: Buffer,
    mat: Buffer,
    uv: Buffer,
}

struct RtPipeline {
    pipe: vk::Pipeline,
    _sbt: Buffer,
    rgen: vk::StridedDeviceAddressRegionKHR,
    miss: vk::StridedDeviceAddressRegionKHR,
    hit: vk::StridedDeviceAddressRegionKHR,
}

#[derive(Default, Clone, Copy)]
pub struct FrameStats {
    pub near_trees: usize,
    pub instances: usize,
    pub rebuilt: bool,
}

pub struct Renderer {
    pub gpu: Gpu,
    swap: Swapchain,
    vsync: bool,
    out_img: Image,
    tg: Targets,
    readback: Buffer,
    taa_layout: vk::PipelineLayout,
    taa_pipe: vk::Pipeline,
    taa_sets: [vk::DescriptorSet; 2],
    resolve_sets: [vk::DescriptorSet; 2],
    parity: usize,
    taa_reset: bool,
    pub taa_enabled: bool,
    resize_count: u32,
    mat_tex: Vec<Image>,
    mat_sampler: vk::Sampler,
    has_tex: bool,
    sharp_img: Image,
    sharpen_layout: vk::PipelineLayout,
    sharpen_pipe: vk::Pipeline,
    sharpen_set: vk::DescriptorSet,
    /// Contrast adaptive sharpening strength (0 = off .. 1).
    pub sharpness: f32,
    set_layout: vk::DescriptorSetLayout,
    pipe_layout: vk::PipelineLayout,
    main_set: vk::DescriptorSet,
    bake_set: vk::DescriptorSet,
    main_pipe: RtPipeline,
    bake_pipe: RtPipeline,
    mip_layout: vk::PipelineLayout,
    mip_pipe: vk::Pipeline,
    mip_sets: Vec<Vec<vk::DescriptorSet>>, // [slot][level-1]
    _meshes: Vec<MeshGpu>,
    mesh_info: Buffer,
    blas: Vec<Accel>,
    _aabbs: Vec<Buffer>,
    imp_blas: Vec<Accel>,
    imp_info: Buffer,
    atlases: Vec<Atlas>,
    sampler: vk::Sampler,
    tlas: Tlas,
    ubo: Buffer,
    cb: vk::CommandBuffer,
    fence: vk::Fence,
    acquire_sem: vk::Semaphore,
    pub world: World,
    pub pack: TreePack,
    /// Impostor tiers: [standard (far), high detail (mid range)]. Atlas slot = tier * variants + variant.
    pub tiers: Vec<ImpostorSettings>,
    lod_state: Vec<Lod>,
    insts: Vec<vk::AccelerationStructureInstanceKHR>,
    inst_data: Vec<InstInfo>,
    inst_info: Buffer,
    tlas_dirty: bool,
    first_update: bool,
    near_count: usize,
    pub need_resize: bool,
    render_scale: f32,
    render_extent: vk::Extent2D,
    resolve_layout: vk::PipelineLayout,
    resolve_pipe: vk::Pipeline,
}

fn scaled_extent(e: vk::Extent2D, scale: f32) -> vk::Extent2D {
    vk::Extent2D {
        width: ((e.width as f32 * scale).round() as u32).max(1),
        height: ((e.height as f32 * scale).round() as u32).max(1),
    }
}

fn upload_mesh(gpu: &Gpu, m: &MeshData) -> Result<MeshGpu> {
    let geo = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;
    Ok(MeshGpu {
        pos: gpu.upload_buffer(bytemuck::cast_slice(&m.positions), geo, &m.name)?,
        nrm: gpu.upload_buffer(bytemuck::cast_slice(&m.normals), vk::BufferUsageFlags::STORAGE_BUFFER, &m.name)?,
        idx: gpu.upload_buffer(bytemuck::cast_slice(&m.indices), geo, &m.name)?,
        mat: gpu.upload_buffer(&m.materials, vk::BufferUsageFlags::STORAGE_BUFFER, &m.name)?,
        uv: gpu.upload_buffer(bytemuck::cast_slice(&m.uvs), vk::BufferUsageFlags::STORAGE_BUFFER, &m.name)?,
    })
}

fn create_rt_pipeline(gpu: &Gpu, layout: vk::PipelineLayout, rgen: &[u8]) -> Result<RtPipeline> {
    let modules = [
        gpu.shader_module(rgen)?,
        gpu.shader_module(spv!("miss.rmiss"))?,
        gpu.shader_module(spv!("shadow.rmiss"))?,
        gpu.shader_module(spv!("tri.rchit"))?,
        gpu.shader_module(spv!("impostor.rint"))?,
        gpu.shader_module(spv!("impostor.rchit"))?,
        gpu.shader_module(spv!("tri.rahit"))?,
    ];
    let kinds = [
        vk::ShaderStageFlags::RAYGEN_KHR,
        vk::ShaderStageFlags::MISS_KHR,
        vk::ShaderStageFlags::MISS_KHR,
        vk::ShaderStageFlags::CLOSEST_HIT_KHR,
        vk::ShaderStageFlags::INTERSECTION_KHR,
        vk::ShaderStageFlags::CLOSEST_HIT_KHR,
        vk::ShaderStageFlags::ANY_HIT_KHR,
    ];
    let stages: Vec<_> = modules
        .iter()
        .zip(kinds)
        .map(|(m, k)| vk::PipelineShaderStageCreateInfo::default().stage(k).module(*m).name(c"main"))
        .collect();
    let group = |ty, general, closest, any, intersection| {
        vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(ty)
            .general_shader(general)
            .closest_hit_shader(closest)
            .any_hit_shader(any)
            .intersection_shader(intersection)
    };
    let u = vk::SHADER_UNUSED_KHR;
    // Hit records (instance SBT offset): 0 opaque triangles, 1 impostor AABBs,
    // 2 triangles of trees inside the mesh <-> impostor band (any-hit dissolve).
    let groups = [
        group(vk::RayTracingShaderGroupTypeKHR::GENERAL, 0, u, u, u),
        group(vk::RayTracingShaderGroupTypeKHR::GENERAL, 1, u, u, u),
        group(vk::RayTracingShaderGroupTypeKHR::GENERAL, 2, u, u, u),
        group(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP, u, 3, u, u),
        group(vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP, u, 5, u, 4),
        group(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP, u, 3, 6, u),
    ];
    let info = vk::RayTracingPipelineCreateInfoKHR::default()
        .stages(&stages)
        .groups(&groups)
        .max_pipeline_ray_recursion_depth(1)
        .layout(layout);
    let pipe = unsafe {
        gpu.rt_fn
            .create_ray_tracing_pipelines(vk::DeferredOperationKHR::null(), vk::PipelineCache::null(), &[info], None)
            .map_err(|(_, e)| e)?[0]
    };
    for m in modules {
        unsafe { gpu.device.destroy_shader_module(m, None) };
    }

    // Shader binding table: raygen | miss, shadow miss | triangle hit, impostor hit, dissolving triangle hit.
    let hs = gpu.rt.handle_size as u64;
    let stride = align_up(hs, gpu.rt.handle_alignment as u64);
    let ba = gpu.rt.base_alignment as u64;
    let off_miss = align_up(stride, ba);
    let off_hit = align_up(off_miss + 2 * stride, ba);
    let total = off_hit + 3 * stride;
    let handles = unsafe { gpu.rt_fn.get_ray_tracing_shader_group_handles(pipe, 0, 6, 6 * hs as usize)? };
    let mut sbt = gpu.create_buffer(total + ba, vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR, MemoryLocation::CpuToGpu, "sbt")?;
    let base = align_up(sbt.addr, ba);
    let shift = (base - sbt.addr) as usize;
    {
        let m = sbt.mapped();
        for (group, off) in
            [(0usize, 0u64), (1, off_miss), (2, off_miss + stride), (3, off_hit), (4, off_hit + stride), (5, off_hit + 2 * stride)]
        {
            let dst = shift + off as usize;
            m[dst..dst + hs as usize].copy_from_slice(&handles[group * hs as usize..(group + 1) * hs as usize]);
        }
    }
    let region = |off: u64, size: u64| vk::StridedDeviceAddressRegionKHR { device_address: base + off, stride, size };
    Ok(RtPipeline {
        pipe,
        rgen: region(0, stride),
        miss: region(off_miss, 2 * stride),
        hit: region(off_hit, 3 * stride),
        _sbt: sbt,
    })
}

impl Renderer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        display: RawDisplayHandle,
        window: RawWindowHandle,
        width: u32,
        height: u32,
        pack: TreePack,
        world: World,
        tiers: Vec<ImpostorSettings>,
        vsync: bool,
        validation: bool,
        render_scale: f32,
        tex_dir: &str,
    ) -> Result<Self> {
        let t0 = Instant::now();
        let gpu = Gpu::new(display, window, validation)?;
        println!("GPU: {}", gpu.device_name);
        let (mat_tex, has_tex) = crate::textures::load(&gpu, tex_dir)?;
        let swap = gpu.create_swapchain(width, height, vk::SwapchainKHR::null(), vsync)?;
        let render_extent = scaled_extent(swap.extent, render_scale);
        let (out_img, tg, readback) = Self::create_targets(&gpu, swap.extent, render_extent)?;
        let sharp_img = Self::create_sharp(&gpu, swap.extent)?;

        // ---- geometry: tree meshes followed by terrain
        let mut meshes = Vec::new();
        for m in pack.meshes.iter().chain(std::iter::once(&world.terrain)) {
            meshes.push(upload_mesh(&gpu, m)?);
        }
        let infos: Vec<[u64; 5]> = meshes.iter().map(|m| [m.pos.addr, m.nrm.addr, m.idx.addr, m.mat.addr, m.uv.addr]).collect();
        let mesh_info = gpu.upload_buffer(bytemuck::cast_slice(&infos), vk::BufferUsageFlags::STORAGE_BUFFER, "mesh info")?;
        let all_meshes: Vec<&MeshData> = pack.meshes.iter().chain(std::iter::once(&world.terrain)).collect();
        let geoms: Vec<_> = meshes
            .iter()
            .zip(&all_meshes)
            .map(|(g, m)| (accel::triangles_geometry(g.pos.addr, m.vertex_count(), g.idx.addr), m.tri_count()))
            .collect();
        let blas = accel::build_blases(&gpu, &geoms)?;
        println!("BLAS: {} meshes built in {:.2}s", blas.len(), t0.elapsed().as_secs_f32());

        // ---- impostor proxies: one AABB per tree variant (shared by every tier)
        let nv = pack.trees.len();
        assert!(nv * tiers.len() <= MAX_VARIANTS, "too many tree variants x impostor tiers");
        let mut aabbs = Vec::new();
        let mut imp_infos = Vec::new();
        for tier in &tiers {
            for t in &pack.trees {
                let pad = Vec3::splat(0.05);
                let (lo, hi) = (t.bmin - pad, t.bmax + pad);
                let c = (t.bmin + t.bmax) * 0.5;
                let r = (t.bmax - t.bmin).length() * 0.5;
                imp_infos.push(ImpostorInfo {
                    center_radius: [c.x, c.y, c.z, r],
                    bmin: [lo.x, lo.y, lo.z, 0.0],
                    bmax: [hi.x, hi.y, hi.z, 0.0],
                    grid: tier.grid,
                    res: tier.res,
                    mips: tier.mips(),
                    pad: 0,
                });
            }
        }
        for t in &pack.trees {
            let pad = Vec3::splat(0.05);
            let (lo, hi) = (t.bmin - pad, t.bmax + pad);
            let data = [lo.x, lo.y, lo.z, hi.x, hi.y, hi.z];
            aabbs.push(gpu.upload_buffer(
                bytemuck::cast_slice(&data),
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                "aabb",
            )?);
        }
        let imp_geoms: Vec<_> = aabbs.iter().map(|b| (accel::aabb_geometry(b.addr), 1)).collect();
        let imp_blas = accel::build_blases(&gpu, &imp_geoms)?;
        let imp_info = gpu.upload_buffer(bytemuck::cast_slice(&imp_infos), vk::BufferUsageFlags::STORAGE_BUFFER, "impostor info")?;

        // ---- impostor atlases, one set per (tier, variant) slot
        let usage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC;
        let mut atlases = Vec::new();
        let mut atlas_bytes = 0u64;
        let mut slot_mips = Vec::new();
        for (ti, tier) in tiers.iter().enumerate() {
            let size = tier.grid * tier.res;
            let mips = tier.mips();
            for t in &pack.trees {
                let name = format!("{} tier{}", t.name, ti);
                let a = gpu.create_image(size, size, mips, vk::Format::R8G8B8A8_UNORM, usage, &format!("{name} albedo"))?;
                let b = gpu.create_image(size, size, mips, vk::Format::R16G16B16A16_SFLOAT, usage, &format!("{name} normal-depth"))?;
                let c = gpu.create_image(size, size, mips, vk::Format::R8G8B8A8_UNORM, usage, &format!("{name} visibility"))?;
                atlases.push((a, b, c));
                atlas_bytes += (size as u64 * size as u64 * 16) * 4 / 3;
                slot_mips.push(mips);
            }
        }
        println!("Impostor atlases: {} slots, {:.2} GB", atlases.len(), atlas_bytes as f64 / 1e9);
        let mips = *slot_mips.iter().max().unwrap();
        gpu.one_time(|cb| {
            for (a, b, c) in &atlases {
                for img in [a.raw, b.raw, c.raw] {
                    gpu.image_barrier(
                        cb,
                        img,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::GENERAL,
                        vk::PipelineStageFlags2::NONE,
                        vk::AccessFlags2::NONE,
                        vk::PipelineStageFlags2::ALL_COMMANDS,
                        vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ,
                    );
                }
            }
        })?;
        let sampler = unsafe {
            gpu.device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .max_lod(mips as f32),
                None,
            )?
        };

        let ubo = gpu.create_buffer(256, vk::BufferUsageFlags::UNIFORM_BUFFER, MemoryLocation::CpuToGpu, "globals")?;
        let mat_sampler = unsafe {
            gpu.device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::REPEAT)
                    .address_mode_v(vk::SamplerAddressMode::REPEAT)
                    .max_lod(16.0),
                None,
            )?
        };

        // ---- descriptors
        let rt_all = vk::ShaderStageFlags::RAYGEN_KHR
            | vk::ShaderStageFlags::CLOSEST_HIT_KHR
            | vk::ShaderStageFlags::ANY_HIT_KHR
            | vk::ShaderStageFlags::INTERSECTION_KHR
            | vk::ShaderStageFlags::MISS_KHR;
        let b = |binding: u32, ty: vk::DescriptorType, count: u32, stages: vk::ShaderStageFlags| {
            vk::DescriptorSetLayoutBinding::default().binding(binding).descriptor_type(ty).descriptor_count(count).stage_flags(stages)
        };
        let bindings = [
            b(0, vk::DescriptorType::ACCELERATION_STRUCTURE_KHR, 1, rt_all),
            b(1, vk::DescriptorType::STORAGE_IMAGE, 1, rt_all),
            b(2, vk::DescriptorType::STORAGE_IMAGE, 1, rt_all),
            b(3, vk::DescriptorType::UNIFORM_BUFFER, 1, rt_all),
            b(4, vk::DescriptorType::STORAGE_BUFFER, 1, rt_all),
            b(5, vk::DescriptorType::STORAGE_BUFFER, 1, rt_all),
            b(6, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, MAX_VARIANTS as u32, rt_all),
            b(7, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, MAX_VARIANTS as u32, rt_all),
            b(8, vk::DescriptorType::STORAGE_IMAGE, 1, rt_all),
            b(9, vk::DescriptorType::STORAGE_IMAGE, 1, rt_all),
            b(10, vk::DescriptorType::STORAGE_IMAGE, 1, rt_all),
            b(11, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, MAX_VARIANTS as u32, rt_all),
            b(12, vk::DescriptorType::STORAGE_BUFFER, 1, rt_all),
            b(13, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 4, rt_all),
        ];
        let set_layout = unsafe {
            gpu.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)?
        };
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
            .offset(0)
            .size(std::mem::size_of::<BakePush>() as u32)];
        let layouts = [set_layout];
        let pipe_layout = unsafe {
            gpu.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&push),
                None,
            )?
        };
        let mip_bindings: Vec<_> = (0..6).map(|i| b(i, vk::DescriptorType::STORAGE_IMAGE, 1, vk::ShaderStageFlags::COMPUTE)).collect();
        let mip_set_layout = unsafe {
            gpu.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&mip_bindings), None)?
        };
        let mip_layouts = [mip_set_layout];
        let mip_layout = unsafe {
            gpu.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&mip_layouts), None)?
        };
        let mip_module = gpu.shader_module(spv!("mip.comp"))?;
        let mip_pipe = unsafe {
            gpu.device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::COMPUTE)
                                .module(mip_module)
                                .name(c"main"),
                        )
                        .layout(mip_layout)],
                    None,
                )
                .map_err(|(_, e)| e)?[0]
        };
        unsafe { gpu.device.destroy_shader_module(mip_module, None) };

        let n_mip_sets: u32 = slot_mips.iter().sum();
        let pool_sizes = [
            vk::DescriptorPoolSize { ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR, descriptor_count: 2 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: 26 + 6 * n_mip_sets },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: 4 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 6 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: 6 * MAX_VARIANTS as u32 + 8 },
        ];
        let desc_pool = unsafe {
            gpu.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().pool_sizes(&pool_sizes).max_sets(7 + n_mip_sets),
                None,
            )?
        };
        let sets = unsafe {
            gpu.device
                .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(desc_pool).set_layouts(&[set_layout, set_layout]))?
        };
        let mut mip_sets = Vec::new();
        for ((a, b, c), &slot_mip_count) in atlases.iter().zip(&slot_mips) {
            let mut per = Vec::new();
            for level in 1..slot_mip_count {
                let set = unsafe {
                    gpu.device.allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default().descriptor_pool(desc_pool).set_layouts(&mip_layouts),
                    )?[0]
                };
                let (src, dst) = (level as usize - 1, level as usize);
                let views = [a.mip_views[src], b.mip_views[src], a.mip_views[dst], b.mip_views[dst], c.mip_views[src], c.mip_views[dst]];
                let infos: Vec<[vk::DescriptorImageInfo; 1]> = views
                    .iter()
                    .map(|v| [vk::DescriptorImageInfo::default().image_view(*v).image_layout(vk::ImageLayout::GENERAL)])
                    .collect();
                let writes: Vec<_> = infos
                    .iter()
                    .enumerate()
                    .map(|(i, info)| {
                        vk::WriteDescriptorSet::default()
                            .dst_set(set)
                            .dst_binding(i as u32)
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .image_info(info)
                    })
                    .collect();
                unsafe { gpu.device.update_descriptor_sets(&writes, &[]) };
                per.push(set);
            }
            mip_sets.push(per);
        }

        let main_pipe = create_rt_pipeline(&gpu, pipe_layout, spv!("main.rgen"))?;
        let bake_pipe = create_rt_pipeline(&gpu, pipe_layout, spv!("bake.rgen"))?;

        // ---- supersampling resolve (render resolution HDR -> display resolution)
        let resolve_bindings: Vec<_> = (0..2).map(|i| b(i, vk::DescriptorType::STORAGE_IMAGE, 1, vk::ShaderStageFlags::COMPUTE)).collect();
        let resolve_set_layout = unsafe {
            gpu.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&resolve_bindings), None)?
        };
        let resolve_layouts = [resolve_set_layout];
        let resolve_push = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(16)];
        let resolve_layout = unsafe {
            gpu.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&resolve_layouts).push_constant_ranges(&resolve_push),
                None,
            )?
        };
        let resolve_module = gpu.shader_module(spv!("resolve.comp"))?;
        let resolve_pipe = unsafe {
            gpu.device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::COMPUTE)
                                .module(resolve_module)
                                .name(c"main"),
                        )
                        .layout(resolve_layout)],
                    None,
                )
                .map_err(|(_, e)| e)?[0]
        };
        unsafe { gpu.device.destroy_shader_module(resolve_module, None) };
        // ---- CAS sharpening (display resolution)
        let sharpen_bindings: Vec<_> = (0..2).map(|i| b(i, vk::DescriptorType::STORAGE_IMAGE, 1, vk::ShaderStageFlags::COMPUTE)).collect();
        let sharpen_set_layout = unsafe {
            gpu.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&sharpen_bindings), None)?
        };
        let sharpen_layouts = [sharpen_set_layout];
        let sharpen_push = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(4)];
        let sharpen_layout = unsafe {
            gpu.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&sharpen_layouts).push_constant_ranges(&sharpen_push),
                None,
            )?
        };
        let sharpen_module = gpu.shader_module(spv!("sharpen.comp"))?;
        let sharpen_pipe = unsafe {
            gpu.device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(sharpen_module).name(c"main"))
                        .layout(sharpen_layout)],
                    None,
                )
                .map_err(|(_, e)| e)?[0]
        };
        unsafe { gpu.device.destroy_shader_module(sharpen_module, None) };
        let sharpen_set = unsafe {
            gpu.device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(desc_pool).set_layouts(&sharpen_layouts))?[0]
        };
        let resolve_sets: [vk::DescriptorSet; 2] = unsafe {
            gpu.device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default().descriptor_pool(desc_pool).set_layouts(&[resolve_set_layout, resolve_set_layout]),
                )?
                .try_into()
                .unwrap()
        };

        // ---- TAA (render resolution): current sample + depth + history ping-pong + globals
        let taa_bindings = [
            b(0, vk::DescriptorType::STORAGE_IMAGE, 1, vk::ShaderStageFlags::COMPUTE),
            b(1, vk::DescriptorType::STORAGE_IMAGE, 1, vk::ShaderStageFlags::COMPUTE),
            b(2, vk::DescriptorType::STORAGE_IMAGE, 1, vk::ShaderStageFlags::COMPUTE),
            b(3, vk::DescriptorType::STORAGE_IMAGE, 1, vk::ShaderStageFlags::COMPUTE),
            b(4, vk::DescriptorType::UNIFORM_BUFFER, 1, vk::ShaderStageFlags::COMPUTE),
        ];
        let taa_set_layout = unsafe {
            gpu.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&taa_bindings), None)?
        };
        let taa_layouts = [taa_set_layout];
        let taa_push = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(4)];
        let taa_layout = unsafe {
            gpu.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&taa_layouts).push_constant_ranges(&taa_push),
                None,
            )?
        };
        let taa_module = gpu.shader_module(spv!("taa.comp"))?;
        let taa_pipe = unsafe {
            gpu.device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(taa_module).name(c"main"))
                        .layout(taa_layout)],
                    None,
                )
                .map_err(|(_, e)| e)?[0]
        };
        unsafe { gpu.device.destroy_shader_module(taa_module, None) };
        let taa_sets: [vk::DescriptorSet; 2] = unsafe {
            gpu.device
                .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(desc_pool).set_layouts(&[taa_set_layout, taa_set_layout]))?
                .try_into()
                .unwrap()
        };

        let est = world.trees.len() as u32 + 400_000;
        let tlas = Tlas::new(&gpu, est)?;
        let inst_info = Self::create_inst_info(&gpu, tlas.capacity)?;

        let cb = unsafe {
            gpu.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(gpu.cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0]
        };
        let fence = unsafe { gpu.device.create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None)? };
        let acquire_sem = unsafe { gpu.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)? };
        let lod_state = vec![Lod::Both; world.trees.len()];

        let mut r = Renderer {
            gpu,
            swap,
            vsync,
            out_img,
            tg,
            readback,
            taa_layout,
            taa_pipe,
            taa_sets,
            resolve_sets,
            parity: 0,
            taa_reset: true,
            taa_enabled: true,
            resize_count: 0,
            mat_tex,
            mat_sampler,
            has_tex,
            sharp_img,
            sharpen_layout,
            sharpen_pipe,
            sharpen_set,
            sharpness: 0.5,
            set_layout,
            pipe_layout,
            main_set: sets[0],
            bake_set: sets[1],
            main_pipe,
            bake_pipe,
            mip_layout,
            mip_pipe,
            mip_sets,
            _meshes: meshes,
            mesh_info,
            blas,
            _aabbs: aabbs,
            imp_blas,
            imp_info,
            atlases,
            sampler,
            tlas,
            ubo,
            cb,
            fence,
            acquire_sem,
            world,
            pack,
            tiers,
            lod_state,
            insts: Vec::new(),
            inst_data: Vec::new(),
            inst_info,
            tlas_dirty: true,
            first_update: true,
            near_count: 0,
            need_resize: false,
            render_scale,
            render_extent,
            resolve_layout,
            resolve_pipe,
        };
        let _ = r.set_layout;
        r.write_set(r.main_set, r.tlas.accel.raw, 0);
        r.write_set(r.bake_set, r.tlas.accel.raw, 0);
        r.write_resolve_set();
        r.bake_impostors()?;
        println!("Renderer ready in {:.2}s", t0.elapsed().as_secs_f32());
        Ok(r)
    }

    fn create_inst_info(gpu: &Gpu, capacity: u32) -> Result<Buffer> {
        gpu.create_buffer(
            capacity as u64 * std::mem::size_of::<InstInfo>() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::CpuToGpu,
            "instance info",
        )
    }

    /// Output image at display resolution; frame sample, depth and TAA history at render (supersampled) resolution.
    fn create_targets(gpu: &Gpu, extent: vk::Extent2D, render: vk::Extent2D) -> Result<(Image, Targets, Buffer)> {
        let out = gpu.create_image(
            extent.width,
            extent.height,
            1,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            "output",
        )?;
        let (w, h) = (render.width, render.height);
        let st = vk::ImageUsageFlags::STORAGE;
        let tg = Targets {
            cur: gpu.create_image(w, h, 1, vk::Format::R16G16B16A16_SFLOAT, st, "frame sample")?,
            depth: gpu.create_image(w, h, 1, vk::Format::R32_SFLOAT, st, "hit distance")?,
            hist: [
                gpu.create_image(w, h, 1, vk::Format::R16G16B16A16_SFLOAT, st, "taa history 0")?,
                gpu.create_image(w, h, 1, vk::Format::R16G16B16A16_SFLOAT, st, "taa history 1")?,
            ],
        };
        gpu.one_time(|cb| {
            for img in [out.raw, tg.cur.raw, tg.depth.raw, tg.hist[0].raw, tg.hist[1].raw] {
                gpu.image_barrier(
                    cb,
                    img,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::GENERAL,
                    vk::PipelineStageFlags2::NONE,
                    vk::AccessFlags2::NONE,
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                    vk::AccessFlags2::SHADER_WRITE,
                );
            }
        })?;
        let readback = gpu.create_buffer(
            extent.width as u64 * extent.height as u64 * 4,
            vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuToCpu,
            "readback",
        )?;
        Ok((out, tg, readback))
    }

    /// Write every binding of a ray tracing descriptor set. `bake_variant` selects the atlas
    /// bound as the bake target (bindings 8/9).
    fn write_set(&self, set: vk::DescriptorSet, tlas: vk::AccelerationStructureKHR, bake_variant: usize) {
        let handles = [tlas];
        let mut as_info = vk::WriteDescriptorSetAccelerationStructureKHR::default().acceleration_structures(&handles);
        let img = |v: vk::ImageView| [vk::DescriptorImageInfo::default().image_view(v).image_layout(vk::ImageLayout::GENERAL)];
        let out = img(self.tg.depth.mip_views[0]); // binding 1: hit distance
        let accum = img(self.tg.cur.mip_views[0]); // binding 2: this frame's sample
        let bake_a = img(self.atlases[bake_variant].0.mip_views[0]);
        let bake_b = img(self.atlases[bake_variant].1.mip_views[0]);
        let bake_c = img(self.atlases[bake_variant].2.mip_views[0]);
        let ubo = [vk::DescriptorBufferInfo::default().buffer(self.ubo.raw).range(vk::WHOLE_SIZE)];
        let meshes = [vk::DescriptorBufferInfo::default().buffer(self.mesh_info.raw).range(vk::WHOLE_SIZE)];
        let imps = [vk::DescriptorBufferInfo::default().buffer(self.imp_info.raw).range(vk::WHOLE_SIZE)];
        let inst_info = [vk::DescriptorBufferInfo::default().buffer(self.inst_info.raw).range(vk::WHOLE_SIZE)];
        let mat_tex: Vec<vk::DescriptorImageInfo> = self
            .mat_tex
            .iter()
            .map(|t| vk::DescriptorImageInfo::default().sampler(self.mat_sampler).image_view(t.view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL))
            .collect();
        let sampled = |pick: fn(&Atlas) -> vk::ImageView| -> Vec<vk::DescriptorImageInfo> {
            (0..MAX_VARIANTS)
                .map(|i| {
                    let a = &self.atlases[i.min(self.atlases.len() - 1)];
                    vk::DescriptorImageInfo::default()
                        .sampler(self.sampler)
                        .image_view(pick(a))
                        .image_layout(vk::ImageLayout::GENERAL)
                })
                .collect()
        };
        let albedo = sampled(|a| a.0.view);
        let nd = sampled(|a| a.1.view);
        let visib = sampled(|a| a.2.view);
        let w = |binding: u32, ty: vk::DescriptorType| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(binding).descriptor_type(ty);
        let mut as_write = w(0, vk::DescriptorType::ACCELERATION_STRUCTURE_KHR).push_next(&mut as_info);
        as_write.descriptor_count = 1;
        let writes = [
            as_write,
            w(1, vk::DescriptorType::STORAGE_IMAGE).image_info(&out),
            w(2, vk::DescriptorType::STORAGE_IMAGE).image_info(&accum),
            w(3, vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&ubo),
            w(4, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&meshes),
            w(5, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&imps),
            w(6, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&albedo),
            w(7, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&nd),
            w(8, vk::DescriptorType::STORAGE_IMAGE).image_info(&bake_a),
            w(9, vk::DescriptorType::STORAGE_IMAGE).image_info(&bake_b),
            w(10, vk::DescriptorType::STORAGE_IMAGE).image_info(&bake_c),
            w(11, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&visib),
            w(12, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&inst_info),
            w(13, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&mat_tex),
        ];
        unsafe { self.gpu.device.update_descriptor_sets(&writes, &[]) };
    }

    /// Bake every (tier, tree variant) slot into its hemi-octahedral atlas using hardware ray tracing.
    fn bake_impostors(&mut self) -> Result<()> {
        let nv = self.pack.trees.len();
        for slot in 0..self.atlases.len() {
            let (tier, v) = (slot / nv, slot % nv);
            let imp = self.tiers[tier];
            let mips = imp.mips();
            let t0 = Instant::now();
            let tree = &self.pack.trees[v];
            let flags = vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE | vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE;
            let insts: Vec<_> = tree
                .parts
                .iter()
                .map(|p| accel::make_instance(&p.xform, p.mesh, 0xFF, 0, flags, self.blas[p.mesh as usize].addr))
                .collect();
            let mut tlas = Tlas::new(&self.gpu, insts.len() as u32)?;
            tlas.write(&insts);
            self.gpu.one_time(|cb| tlas.cmd_build(&self.gpu, cb, insts.len() as u32))?;
            self.write_set(self.bake_set, tlas.accel.raw, slot);

            let c = (tree.bmin + tree.bmax) * 0.5;
            let r = (tree.bmax - tree.bmin).length() * 0.5;
            // Bake rays are orthographic: fixed texture footprint = one bake sub-sample (negative
            // spread in tri.rchit), and the material textures switched on.
            let mut bg = Globals::default();
            bg.cam_fwd[3] = -(2.0 * r / imp.res as f32 / imp.ss as f32);
            bg.params[1] = self.has_tex as u32 as f32;
            self.ubo.mapped()[..std::mem::size_of::<Globals>()].copy_from_slice(bytemuck::bytes_of(&bg));
            let mut push = BakePush {
                center_radius: [c.x, c.y, c.z, r],
                grid: imp.grid,
                res: imp.res,
                frame_offset: 0,
                ss: imp.ss,
                ao_rays: imp.ao_rays,
                ao_range: imp.ao_range,
                vis_rays: imp.vis_rays,
                pad: 0,
            };
            for row in 0..imp.grid {
                push.frame_offset = row * imp.grid;
                let d = &self.gpu.device;
                self.gpu.one_time(|cb| unsafe {
                    d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::RAY_TRACING_KHR, self.bake_pipe.pipe);
                    d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::RAY_TRACING_KHR, self.pipe_layout, 0, &[self.bake_set], &[]);
                    d.cmd_push_constants(cb, self.pipe_layout, vk::ShaderStageFlags::RAYGEN_KHR, 0, bytemuck::bytes_of(&push));
                    self.gpu.rt_fn.cmd_trace_rays(
                        cb,
                        &self.bake_pipe.rgen,
                        &self.bake_pipe.miss,
                        &self.bake_pipe.hit,
                        &vk::StridedDeviceAddressRegionKHR::default(),
                        imp.res,
                        imp.res,
                        imp.grid,
                    );
                })?;
            }

            let d = &self.gpu.device;
            let size = imp.grid * imp.res;
            self.gpu.one_time(|cb| unsafe {
                self.gpu.memory_barrier(
                    cb,
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::SHADER_WRITE,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_READ,
                );
                d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.mip_pipe);
                for level in 1..mips {
                    let s = (size >> level).max(1);
                    d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, self.mip_layout, 0, &[self.mip_sets[slot][level as usize - 1]], &[]);
                    d.cmd_dispatch(cb, s.div_ceil(8), s.div_ceil(8), 1);
                    self.gpu.memory_barrier(
                        cb,
                        vk::PipelineStageFlags2::COMPUTE_SHADER,
                        vk::AccessFlags2::SHADER_WRITE,
                        vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                        vk::AccessFlags2::SHADER_READ,
                    );
                }
            })?;
            tlas.destroy(&self.gpu);
            println!(
                "Baked {} impostor {} ({} twig/trunk instances, {:.0}M tris) -> {}x{} atlas, {} mips in {:.2}s",
                if tier == 0 { "standard" } else { "high-detail" },
                tree.name,
                tree.parts.len(),
                tree.expanded_tris(&self.pack.meshes) as f64 / 1e6,
                size,
                size,
                mips,
                t0.elapsed().as_secs_f32()
            );
        }
        self.write_set(self.bake_set, self.tlas.accel.raw, 0);
        Ok(())
    }

    /// Per tree: mesh only, impostor only, or both while inside the mesh <-> impostor band (the
    /// shaders then dissolve between the two copies). Rebuilds the TLAS only when a tree changes state.
    fn update_instances(&mut self, cam: Vec3, lod: f32, band: f32) -> Result<FrameStats> {
        let mut changed = self.first_update;
        self.first_update = false;
        // The shader fade is exactly 1 below a and exactly 0 above b, so "mesh only" / "impostor
        // only" are safe there; the hysteresis just keeps "both" a little longer.
        let (a, b) = if lod <= 0.0 { (-1.0, -1.0) } else { (lod - band * 0.5, lod + band * 0.5) };
        let mut near = 0;
        for (i, t) in self.world.trees.iter().enumerate() {
            let d = t.center.distance(cam);
            let cur = self.lod_state[i];
            let next = match cur {
                Lod::Mesh if d > a - 0.5 => Lod::Both,
                Lod::Impostor if d < b + 0.5 => Lod::Both,
                Lod::Both if d < a - 2.0 => Lod::Mesh,
                Lod::Both if d > b + 2.0 => Lod::Impostor,
                s => s,
            };
            if next != cur {
                self.lod_state[i] = next;
                changed = true;
            }
            near += (next != Lod::Impostor) as usize;
        }
        self.near_count = near;
        if changed {
            let mesh_flags = vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE | vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE;
            // Band trees: not opaque, so the dissolve any-hit runs (hit record 2).
            let band_flags = vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE | vk::GeometryInstanceFlagsKHR::FORCE_NO_OPAQUE;
            let imp_flags = vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE;
            let terrain_id = self.pack.meshes.len();
            self.insts.clear();
            self.inst_data.clear();
            self.inst_data.push(InstInfo { center: [0.0, -1e9, 0.0], data: 0 });
            // Mask bit 0: meshes and terrain. Bits 1..7: impostors by lattice colour, so shadow
            // rays leaving an impostor can skip that impostor (its self-shadowing is baked).
            self.insts.push(accel::make_instance(
                &glam::Affine3A::IDENTITY,
                terrain_id as u32,
                1,
                0,
                mesh_flags,
                self.blas[terrain_id].addr,
            ));
            // Instance budget: beyond it, further trees stay impostors (protects VRAM when the
            // LOD distance is pushed very far in a dense forest).
            const MAX_INSTANCES: usize = 12_000_000;
            for (i, t) in self.world.trees.iter().enumerate() {
                let seed = ((t.seed as u32) & 0x1FFF) << 11;
                let parts = self.pack.trees[t.variant as usize].parts.len();
                if self.lod_state[i] == Lod::Mesh && self.insts.len() + parts > MAX_INSTANCES {
                    self.lod_state[i] = Lod::Impostor;
                }
                let state = self.lod_state[i];
                let center = t.center.to_array();
                let tree = i as u32 & 0xFF_FFFF;
                if state != Lod::Impostor {
                    let (flags, sbt) = if state == Lod::Mesh { (mesh_flags, 0) } else { (band_flags, 2) };
                    for p in &self.pack.trees[t.variant as usize].parts {
                        let m = t.xform * p.xform;
                        self.insts.push(accel::make_instance(&m, p.mesh | seed, 1, sbt, flags, self.blas[p.mesh as usize].addr));
                        self.inst_data.push(InstInfo { center, data: tree });
                    }
                }
                if state != Lod::Mesh {
                    self.insts.push(accel::make_instance(
                        &t.xform,
                        t.variant as u32 | ((t.color as u32) << 8) | seed,
                        2u8 << t.color,
                        1,
                        imp_flags,
                        self.imp_blas[t.variant as usize].addr,
                    ));
                    let has_mesh = if state == Lod::Both { INST_HAS_MESH } else { 0 };
                    self.inst_data.push(InstInfo { center, data: tree | has_mesh });
                }
            }
            if self.insts.len() > self.tlas.capacity as usize {
                let cap = (self.insts.len() as f32 * 1.5) as u32;
                println!("Growing TLAS capacity to {cap} instances");
                unsafe { self.gpu.device.device_wait_idle()? };
                let mut old = std::mem::replace(&mut self.tlas, Tlas::new(&self.gpu, cap)?);
                old.destroy(&self.gpu);
                let mut old_info = std::mem::replace(&mut self.inst_info, Self::create_inst_info(&self.gpu, cap)?);
                self.gpu.destroy_buffer(&mut old_info);
                self.write_set(self.main_set, self.tlas.accel.raw, 0);
            }
            self.tlas.write(&self.insts);
            let bytes: &[u8] = bytemuck::cast_slice(&self.inst_data);
            self.inst_info.mapped()[..bytes.len()].copy_from_slice(bytes);
            self.tlas_dirty = true;
        }
        Ok(FrameStats { near_trees: self.near_count, instances: self.insts.len(), rebuilt: changed })
    }

    pub fn extent(&self) -> (u32, u32) {
        (self.swap.extent.width, self.swap.extent.height)
    }

    /// Supersampled tracing resolution.
    pub fn render_extent(&self) -> (u32, u32) {
        (self.render_extent.width, self.render_extent.height)
    }

    pub fn render_scale(&self) -> f32 {
        self.render_scale
    }

    /// TAA set p reads history p and writes history 1-p; resolve set p reads history 1-p.
    fn write_resolve_set(&self) {
        let info = |img: &Image| [vk::DescriptorImageInfo::default().image_view(img.mip_views[0]).image_layout(vk::ImageLayout::GENERAL)];
        let ubo = [vk::DescriptorBufferInfo::default().buffer(self.ubo.raw).range(vk::WHOLE_SIZE)];
        for p in 0..2 {
            let (cur, depth, prev, next, out) =
                (info(&self.tg.cur), info(&self.tg.depth), info(&self.tg.hist[p]), info(&self.tg.hist[1 - p]), info(&self.out_img));
            fn st<'a>(set: vk::DescriptorSet, binding: u32, i: &'a [vk::DescriptorImageInfo; 1]) -> vk::WriteDescriptorSet<'a> {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(i)
            }
            let t = self.taa_sets[p];
            let r = self.resolve_sets[p];
            let writes = [
                st(t, 0, &cur),
                st(t, 1, &depth),
                st(t, 2, &prev),
                st(t, 3, &next),
                vk::WriteDescriptorSet::default().dst_set(t).dst_binding(4).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&ubo),
                st(r, 0, &next),
                st(r, 1, &out),
            ];
            unsafe { self.gpu.device.update_descriptor_sets(&writes, &[]) };
        }
        let (src, dst) = (info(&self.out_img), info(&self.sharp_img));
        let writes = [0u32, 1].map(|bnd| {
            vk::WriteDescriptorSet::default()
                .dst_set(self.sharpen_set)
                .dst_binding(bnd)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(if bnd == 0 { &src } else { &dst })
        });
        unsafe { self.gpu.device.update_descriptor_sets(&writes, &[]) };
    }

    fn create_sharp(gpu: &Gpu, extent: vk::Extent2D) -> Result<Image> {
        let img = gpu.create_image(
            extent.width,
            extent.height,
            1,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            "sharpened output",
        )?;
        gpu.one_time(|cb| {
            gpu.image_barrier(
                cb,
                img.raw,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags2::NONE,
                vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::SHADER_WRITE,
            )
        })?;
        Ok(img)
    }

    fn recreate_targets(&mut self) -> Result<()> {
        unsafe { self.gpu.device.device_wait_idle()? };
        self.gpu.destroy_image(&mut self.out_img);
        self.gpu.destroy_image(&mut self.sharp_img);
        self.sharp_img = Self::create_sharp(&self.gpu, self.swap.extent)?;
        for img in [&mut self.tg.cur, &mut self.tg.depth] {
            self.gpu.destroy_image(img);
        }
        for img in self.tg.hist.iter_mut() {
            self.gpu.destroy_image(img);
        }
        self.gpu.destroy_buffer(&mut self.readback);
        self.render_extent = scaled_extent(self.swap.extent, self.render_scale);
        let (o, tg, r) = Self::create_targets(&self.gpu, self.swap.extent, self.render_extent)?;
        self.out_img = o;
        self.tg = tg;
        self.readback = r;
        self.write_set(self.main_set, self.tlas.accel.raw, 0);
        self.write_set(self.bake_set, self.tlas.accel.raw, 0);
        self.write_resolve_set();
        self.taa_reset = true;
        Ok(())
    }

    /// Change the supersampling factor (1 = native, 2 = 2x2 rays per pixel).
    pub fn set_render_scale(&mut self, scale: f32) -> Result<()> {
        self.render_scale = scale.clamp(0.5, 3.0);
        self.recreate_targets()
    }

    pub fn resize(&mut self, w: u32, h: u32) -> Result<()> {
        if w == 0 || h == 0 {
            return Ok(());
        }
        unsafe { self.gpu.device.device_wait_idle()? };
        self.resize_count += 1;
        if self.resize_count <= 3 || self.resize_count % 100 == 0 {
            println!("swapchain resize #{} to {}x{}", self.resize_count, w, h);
        }
        let new = self.gpu.create_swapchain(w, h, self.swap.raw, self.vsync)?;
        let same_size = new.extent == self.swap.extent;
        let mut old = std::mem::replace(&mut self.swap, new);
        self.gpu.destroy_swapchain(&mut old);
        // A same-size rebuild (e.g. "suboptimal" after a move between monitors) keeps the render
        // targets and the TAA history.
        if !same_size {
            self.recreate_targets()?;
        }
        self.need_resize = false;
        Ok(())
    }

    /// Render one frame. With `capture`, the tonemapped output is returned as RGBA8.
    pub fn render(&mut self, globals: &Globals, cam: Vec3, lod: f32, capture: bool) -> Result<(FrameStats, Option<Vec<u8>>)> {
        let d = self.gpu.device.clone();
        unsafe { d.wait_for_fences(&[self.fence], true, u64::MAX).context("waiting for previous frame")? };
        let idx = match unsafe { self.gpu.swapchain_fn.acquire_next_image(self.swap.raw, u64::MAX, self.acquire_sem, vk::Fence::null()) } {
            Ok((i, _)) => i,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.need_resize = true;
                return Ok((FrameStats::default(), None));
            }
            Err(e) => return Err(e.into()),
        };
        unsafe { d.reset_fences(&[self.fence])? };

        let mut g = *globals;
        if lod <= 0.0 {
            g.lod_params[0] = 0.0; // all impostors: no dissolve band
        }
        if self.tiers.len() < 2 {
            g.lod_params[1] = 0.0; // no high-detail tier
        }
        g.lod_params[3] = self.pack.trees.len() as f32;
        g.params[1] = self.has_tex as u32 as f32; // material textures available
        let globals = &g;
        let stats = self.update_instances(cam, lod, g.lod_params[0])?;
        self.ubo.mapped()[..std::mem::size_of::<Globals>()].copy_from_slice(bytemuck::bytes_of(globals));

        let (w, h) = self.extent();
        let (rw, rh) = self.render_extent();
        let cb = self.cb;
        let swap_img = self.swap.images[idx as usize];
        unsafe {
            d.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;
            d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            if self.tlas_dirty {
                self.tlas.cmd_build(&self.gpu, cb, self.insts.len() as u32);
                self.tlas_dirty = false;
            }
            self.gpu.memory_barrier(
                cb,
                vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR | vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_READ,
                vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
            );
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::RAY_TRACING_KHR, self.main_pipe.pipe);
            d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::RAY_TRACING_KHR, self.pipe_layout, 0, &[self.main_set], &[]);
            self.gpu.rt_fn.cmd_trace_rays(
                cb,
                &self.main_pipe.rgen,
                &self.main_pipe.miss,
                &self.main_pipe.hit,
                &vk::StridedDeviceAddressRegionKHR::default(),
                rw,
                rh,
                1,
            );
            // TAA: blend this frame into the reprojected history (render resolution).
            self.gpu.memory_barrier(
                cb,
                vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR | vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::SHADER_READ,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
            );
            let p = self.parity;
            let reset = [(self.taa_reset || !self.taa_enabled) as u32];
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.taa_pipe);
            d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, self.taa_layout, 0, &[self.taa_sets[p]], &[]);
            d.cmd_push_constants(cb, self.taa_layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::cast_slice(&reset));
            d.cmd_dispatch(cb, rw.div_ceil(8), rh.div_ceil(8), 1);
            self.gpu.memory_barrier(
                cb,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_READ,
            );
            self.taa_reset = false;
            self.parity = 1 - p;

            // Supersampling resolve: tonemap + area-weighted downsample to display resolution.
            let push: [u32; 4] = [
                (rw as f32 / w as f32).to_bits(),
                1.0f32.to_bits(), // history is already an average
                globals.params[2].to_bits(),
                (globals.iparams[1] == 0) as u32,
            ];
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.resolve_pipe);
            d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, self.resolve_layout, 0, &[self.resolve_sets[p]], &[]);
            d.cmd_push_constants(cb, self.resolve_layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::cast_slice(&push));
            d.cmd_dispatch(cb, w.div_ceil(8), h.div_ceil(8), 1);
            // CAS sharpening on the display image.
            self.gpu.memory_barrier(
                cb,
                vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_READ,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
            );
            let sharp = [if globals.iparams[1] == 0 { self.sharpness } else { 0.0 }];
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.sharpen_pipe);
            d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, self.sharpen_layout, 0, &[self.sharpen_set], &[]);
            d.cmd_push_constants(cb, self.sharpen_layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::cast_slice(&sharp));
            d.cmd_dispatch(cb, w.div_ceil(8), h.div_ceil(8), 1);
            self.gpu.memory_barrier(
                cb,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
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
                self.sharp_img.raw,
                vk::ImageLayout::GENERAL,
                swap_img,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::ImageBlit {
                    src_subresource: layers,
                    src_offsets: [vk::Offset3D::default(), corner],
                    dst_subresource: layers,
                    dst_offsets: [vk::Offset3D::default(), corner],
                }],
                vk::Filter::NEAREST,
            );
            if capture {
                d.cmd_copy_image_to_buffer(
                    cb,
                    self.sharp_img.raw,
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
                &[vk::SubmitInfo::default()
                    .wait_semaphores(&waits)
                    .wait_dst_stage_mask(&stages)
                    .command_buffers(&cbs)
                    .signal_semaphores(&signals)],
                self.fence,
            )?;
            let swaps = [self.swap.raw];
            let idxs = [idx];
            match self.gpu.swapchain_fn.queue_present(
                self.gpu.queue,
                &vk::PresentInfoKHR::default().wait_semaphores(&signals).swapchains(&swaps).image_indices(&idxs),
            ) {
                Ok(false) => {}
                // "Suboptimal" is only worth a rebuild if the surface size actually changed;
                // otherwise it would rebuild (and reset TAA) every single frame.
                Ok(true) => {
                    let caps = self.gpu.surface_fn.get_physical_device_surface_capabilities(self.gpu.pdev, self.gpu.surface)?;
                    if caps.current_extent != self.swap.extent {
                        self.need_resize = true;
                    }
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.need_resize = true,
                Err(e) => return Err(e.into()),
            }
        }

        let image = if capture {
            unsafe { d.wait_for_fences(&[self.fence], true, u64::MAX)? };
            let n = (w * h * 4) as usize;
            Some(self.readback.mapped()[..n].to_vec())
        } else {
            None
        };
        Ok((stats, image))
    }

    /// Read back mip 0 of a variant's atlases as display-friendly RGBA8 images
    /// (albedo, normals, depth/coverage).
    pub fn dump_atlas(&mut self, variant: usize) -> Result<(u32, Vec<u8>, Vec<u8>, Vec<u8>)> {
        let size = self.tiers[0].grid * self.tiers[0].res;
        let n = size as u64 * size as u64;
        let mut ba = self.gpu.create_buffer(n * 4, vk::BufferUsageFlags::TRANSFER_DST, MemoryLocation::GpuToCpu, "dump a")?;
        let mut bb = self.gpu.create_buffer(n * 8, vk::BufferUsageFlags::TRANSFER_DST, MemoryLocation::GpuToCpu, "dump b")?;
        let (a, b, _) = &self.atlases[variant];
        let layers = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 };
        let region = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: layers,
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D { width: size, height: size, depth: 1 },
        };
        let d = &self.gpu.device;
        self.gpu.one_time(|cb| unsafe {
            d.cmd_copy_image_to_buffer(cb, a.raw, vk::ImageLayout::GENERAL, ba.raw, &[region]);
            d.cmd_copy_image_to_buffer(cb, b.raw, vk::ImageLayout::GENERAL, bb.raw, &[region]);
        })?;
        let albedo = ba.mapped()[..(n * 4) as usize].to_vec();
        let halves: Vec<u16> = bytemuck::pod_collect_to_vec(&bb.mapped()[..(n * 8) as usize]);
        let mut normals = vec![0u8; (n * 4) as usize];
        let mut depth = vec![0u8; (n * 4) as usize];
        for i in 0..n as usize {
            let px = [half_to_f32(halves[i * 4]), half_to_f32(halves[i * 4 + 1]), half_to_f32(halves[i * 4 + 2]), half_to_f32(halves[i * 4 + 3])];
            let cov = px[3].clamp(0.0, 1.0);
            normals[i * 4] = ((px[0] * 0.5 + 0.5) * 255.0 * cov) as u8;
            normals[i * 4 + 1] = ((px[1] * 0.5 + 0.5) * 255.0 * cov) as u8;
            normals[i * 4 + 2] = (cov * 255.0) as u8;
            normals[i * 4 + 3] = 255;
            let dd = ((px[2] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0 * cov) as u8;
            depth[i * 4] = dd;
            depth[i * 4 + 1] = dd;
            depth[i * 4 + 2] = dd;
            depth[i * 4 + 3] = 255;
        }
        let mut albedo_vis = albedo;
        for px in albedo_vis.chunks_mut(4) {
            for c in px.iter_mut().take(3) {
                *c = ((*c as f32) * 2.2).min(255.0) as u8;
            }
            px[3] = 255;
        }
        self.gpu.destroy_buffer(&mut ba);
        self.gpu.destroy_buffer(&mut bb);
        Ok((size, albedo_vis, normals, depth))
    }
}

fn half_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let man = (h & 0x3ff) as f32;
    match exp {
        0 => sign * man * 2f32.powi(-24),
        31 => sign * f32::INFINITY,
        _ => sign * (1.0 + man / 1024.0) * 2f32.powi(exp - 15),
    }
}
