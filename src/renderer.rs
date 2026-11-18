//! Ray tracing renderer: pipelines, impostor baking, per-frame LOD selection and tracing.

use crate::accel::{self, Accel, Tlas};
use crate::asset::{MeshData, TreePack};
use crate::catalog::Catalog;
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

/// Atlas slots (one per plant variant) bound as sampler arrays. Must match shared.glsl.
pub const MAX_VARIANTS: usize = 128;
/// Texture sets (4 textures each) bound as a sampler array. Must match shared.glsl.
pub const MAX_SPECIES: usize = 16;
/// Leaf transmission relative to albedo (calibrated so Black Alder keeps its earlier back-lit look).
const TRANS_GAIN: f32 = 8.0;
/// Ground material textures (binding 16). Must match shared.glsl.
pub const MAX_GROUND_TEX: usize = 40;

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
    /// x aerial perspective strength (1 = clear day, 0 = off)
    pub extra: [f32; 4],
    /// Globe mode: xyz planet centre relative to the camera (camera frame), w = 1. 0 otherwise.
    pub planet: [f32; 4],
}

/// Render-resolution targets: this frame's sample + hit distance, and the TAA history ping-pong.
struct Targets {
    cur: Image,
    depth: Image,
    hist: [Image; 2],
    // DLSS Ray Reconstruction inputs (render resolution), written by main.rgen when DLSS is on
    viewz: Image,  // r32f linear view depth
    motion: Image, // rg16f pixel motion, previous - current
    nrm: Image,    // rgba16f world normal, roughness
    diff: Image,   // rgba8 diffuse albedo
    spec: Image,   // rgba8 specular albedo
    dlss_out: Image, // rgba16f denoised + upscaled HDR (display resolution)
}

/// Per TLAS instance data for the LOD cross-fade shaders (indexed by gl_InstanceID).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InstInfo {
    center: [f32; 3],
    data: u32, // tree index | flags << 24 (bit 0: tree also has mesh instances)
}

const INST_HAS_MESH: u32 = 1 << 24;
const INST_VARIANT_SHIFT: u32 = 25;

/// Per-tree LOD state.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Lod {
    Impostor,
    Both, // inside the mesh <-> impostor band: both copies in the TLAS, shaders dissolve
    Mesh,
}

/// Per plant variant (= atlas slot); see rt.glsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ImpostorInfo {
    pub center_radius: [f32; 4],
    pub bmin: [f32; 4],
    pub bmax: [f32; 4],
    pub trans: [f32; 4],
    pub grid: u32,
    pub res: u32,
    pub mips: u32,
    pub lod_scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MeshInfoGpu {
    addr: [u64; 5], // pos, nrm, idx, mat, uv
    tex_set: u32,
    flags: u32, // MESH_ALPHA_TESTED
}

/// Mesh flag: the leaf texture's alpha is an opacity mask (alpha-tested in tri.rahit).
const MESH_ALPHA_TESTED: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SpeciesGpu {
    leaf_tint: [f32; 4],
    bark_tint: [f32; 4],
    trans_tint: [f32; 4],
}

/// One BC-compressed impostor atlas per plant variant.
struct BcAtlas {
    albedo: Image, // BC3: sqrt(albedo).rgb, AO
    normal: Image, // BC5: octahedral normal * 0.5 + 0.5
    depth: Image,  // BC5: depth * 0.5 + 0.5, coverage
    vis: Image,    // BC3, half resolution: directional visibility a, b/3 + 0.5
    grid: u32,
    res: u32,
    mips: u32,
}

impl BcAtlas {
    fn bytes(&self) -> u64 {
        let s = (self.grid * self.res) as u64;
        // BC3/BC5: 1 byte per texel; vis at quarter area; mips +1/3
        (s * s * 3 + s * s / 4) * 4 / 3
    }
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

/// Impostor atlas settings. Each variant gets its own resolution: texels of `texel` metres
/// (scaled by the variant's LOD scale, since small plants become impostors closer to the camera),
/// rounded to a multiple of 32 and clamped to [32, max_res]. Mip-mapping serves the far field.
#[derive(Clone, Copy)]
pub struct ImpostorSettings {
    pub grid: u32,
    pub texel: f32,
    pub max_res: u32,
    pub ss: u32,
    pub ao_rays: u32,
    pub ao_range: f32,
    pub vis_rays: u32,
}

impl ImpostorSettings {
    pub fn res_for(&self, radius: f32, lod_scale: f32) -> u32 {
        // Small plants switch to impostors closer, but only down to 0.35x the tree texel density.
        let r = (2.0 * radius / (self.texel * lod_scale.max(0.35))).round() as u32;
        (r.div_ceil(32) * 32).clamp(32, self.max_res.max(32))
    }
}

/// Mip count for a frame resolution: every encoded level must stay a multiple of 4 (BC blocks)
/// and frames keep at least 8 texels.
fn atlas_mips(res: u32) -> u32 {
    let mut m = 1;
    while (res >> m) >= 8 && (res >> m) % 4 == 0 {
        m += 1;
    }
    m
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

/// Per-pass GPU times and CPU times (ms), summed over the frames since the last reset.
#[derive(Default, Clone, Copy)]
pub struct Profile {
    /// TLAS build, ray tracing (main.rgen + hit shaders), denoise/TAA + resolve, sharpen + blit
    pub gpu: [f64; 4],
    /// update_instances (LOD + instance list), command recording + submit, waiting for the GPU
    pub cpu_lod: f64,
    /// parts of cpu_lod: LOD state loop, instance list build, upload to the mapped buffers
    pub cpu_lod_parts: [f64; 3],
    pub rebuilds: u32,
    pub cpu_record: f64,
    pub cpu_wait: f64,
    pub frames: u32,
}

impl Profile {
    pub fn line(&self) -> String {
        let n = self.frames.max(1) as f64;
        format!(
            "gpu: tlas {:.2} | trace {:.2} | denoise+resolve {:.2} | sharpen+blit {:.2} = {:.2} ms; cpu: lod {:.2} (states {:.2}, list {:.2}, upload {:.2}, {} rebuilds) | record {:.2} | wait {:.2} ms ({} frames)",
            self.gpu[0] / n,
            self.gpu[1] / n,
            self.gpu[2] / n,
            self.gpu[3] / n,
            self.gpu.iter().sum::<f64>() / n,
            self.cpu_lod / n,
            self.cpu_lod_parts[0] / n,
            self.cpu_lod_parts[1] / n,
            self.cpu_lod_parts[2] / n,
            self.rebuilds,
            self.cpu_record / n,
            self.cpu_wait / n,
            self.frames
        )
    }
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
    dlss_resolve_set: vk::DescriptorSet,
    /// NGX initialised and Ray Reconstruction available.
    dlss_ok: bool,
    /// Active DLSS mode (index into dlss::MODES), None = TAA + supersampling.
    dlss_mode: Option<usize>,
    last_frame: Instant,
    /// Jitter sign convention passed to NGX (SI_DLSS_JITTER="+x,+y" overrides for testing).
    dlss_jitter_sign: (f32, f32),
    parity: usize,
    taa_reset: bool,
    pub taa_enabled: bool,
    resize_count: u32,
    mat_tex: Vec<Image>,
    ground_tex: Vec<Image>,
    ground_layers: u32,
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
    mip_set_layout: vk::DescriptorSetLayout,
    bcenc_layout: vk::PipelineLayout,
    bcenc_pipe: vk::Pipeline,
    bcenc_set_layout: vk::DescriptorSetLayout,
    _meshes: Vec<MeshGpu>,
    mesh_info: Buffer,
    species_buf: Buffer,
    blas: Vec<Accel>,
    _aabbs: Vec<Buffer>,
    imp_blas: Vec<Accel>,
    imp_info: Buffer,
    imp_infos: Vec<ImpostorInfo>,
    atlases: Vec<BcAtlas>,
    sampler: vk::Sampler,
    pub catalog: Catalog,
    tlas: Tlas,
    ubo: Buffer,
    cb: vk::CommandBuffer,
    fence: vk::Fence,
    acquire_sem: vk::Semaphore,
    query_pool: vk::QueryPool,
    queries_pending: bool,
    pub profile: Profile,
    pub world: World,
    pub pack: TreePack,
    /// Impostor atlas settings (one compressed atlas per variant; slot = variant index).
    pub imp: ImpostorSettings,
    /// Write the first baked atlas as PNGs with this prefix (debugging).
    pub dump_prefix: Option<String>,
    lod_state: Vec<Lod>,
    /// Plants (world.trees indices) currently in Mesh / Both state.
    active: Vec<u32>,
    /// Per plant: last LOD frame it was examined (dedupe of the candidate list).
    visited: Vec<u32>,
    lod_frame: u32,
    /// Spatial grid of the plants for the near-camera LOD query.
    grid: PlantGrid,
    max_lod_scale: f32,
    /// Plant record index (cluster order) of every world.trees entry.
    rec_of: Vec<u32>,
    /// Per plant record: object->world and variant/seed (binding 23); LOD state (binding 24).
    plant_rec: Buffer,
    plant_state: Buffer,
    state_writes: Vec<(u32, u32)>,
    /// Cluster BLASes (one AABB per plant) and their AABB buffer.
    _cluster_blas: Vec<Accel>,
    _cluster_aabbs: Buffer,
    /// TLAS instances that never change: terrain and the plant clusters.
    static_insts: Vec<vk::AccelerationStructureInstanceKHR>,
    static_inst_data: Vec<InstInfo>,
    insts: Vec<vk::AccelerationStructureInstanceKHR>,
    inst_data: Vec<InstInfo>,
    inst_info: Buffer,
    tlas_dirty: bool,
    /// insts / inst_data changed and must be copied to the GPU buffers (upload_instances).
    upload_pending: bool,
    first_update: bool,
    near_count: usize,
    pub need_resize: bool,
    render_scale: f32,
    render_extent: vk::Extent2D,
    resolve_layout: vk::PipelineLayout,
    resolve_pipe: vk::Pipeline,
    /// Globe mode: the streamed terrain and trees.
    pub globe: Option<crate::globe::scene::GlobeScene>,
    /// Placeholders for the globe bindings when not in globe mode.
    globe_dummy: Option<(Buffer, Buffer, Image)>,
    /// Globe mode: this frame's camera (ECEF), its frame rotation and globe.rchit's globals.
    globe_view: Option<(glam::DVec3, glam::DMat3, crate::globe::scene::GlobeGlobals)>,
    /// Globe mode: plant records currently drawn as meshes (1 both / 2 mesh only).
    globe_states: std::collections::HashMap<u32, u32>,
}

/// World -> view for NGX: row-major, row vectors (v_view = v_world * M), x right, y up, z forward.
fn world_to_view(g: &Globals) -> [f32; 16] {
    let p = Vec3::from_slice(&g.cam_pos[..3]);
    let r = Vec3::from_slice(&g.cam_right[..3]);
    let u = Vec3::from_slice(&g.cam_up[..3]);
    let f = Vec3::from_slice(&g.cam_fwd[..3]);
    [
        r.x, u.x, f.x, 0.0, //
        r.y, u.y, f.y, 0.0, //
        r.z, u.z, f.z, 0.0, //
        -r.dot(p), -u.dot(p), -f.dot(p), 1.0,
    ]
}

/// View -> clip (DirectX-style perspective, depth 0..1), matching the camera's tan(fov/2) values.
fn view_to_clip(g: &Globals) -> [f32; 16] {
    let (sx, sy) = (1.0 / g.cam_right[3], 1.0 / g.cam_up[3]);
    let (n, f) = (0.02f32, 1.0e5f32);
    let a = f / (f - n);
    [
        sx, 0.0, 0.0, 0.0, //
        0.0, sy, 0.0, 0.0, //
        0.0, 0.0, a, 1.0, //
        0.0, 0.0, -n * a, 0.0,
    ]
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
        gpu.shader_module(spv!("globe.rchit"))?,
    ];
    let kinds = [
        vk::ShaderStageFlags::RAYGEN_KHR,
        vk::ShaderStageFlags::MISS_KHR,
        vk::ShaderStageFlags::MISS_KHR,
        vk::ShaderStageFlags::CLOSEST_HIT_KHR,
        vk::ShaderStageFlags::INTERSECTION_KHR,
        vk::ShaderStageFlags::CLOSEST_HIT_KHR,
        vk::ShaderStageFlags::ANY_HIT_KHR,
        vk::ShaderStageFlags::CLOSEST_HIT_KHR,
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
    // 2 triangles of trees inside the mesh <-> impostor band (any-hit dissolve), 3 globe terrain.
    let groups = [
        group(vk::RayTracingShaderGroupTypeKHR::GENERAL, 0, u, u, u),
        group(vk::RayTracingShaderGroupTypeKHR::GENERAL, 1, u, u, u),
        group(vk::RayTracingShaderGroupTypeKHR::GENERAL, 2, u, u, u),
        group(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP, u, 3, u, u),
        group(vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP, u, 5, u, 4),
        group(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP, u, 3, 6, u),
        group(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP, u, 7, u, u),
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

    // Shader binding table: raygen | miss, shadow miss | triangle hit, impostor hit, dissolving triangle hit, globe terrain hit.
    let hs = gpu.rt.handle_size as u64;
    let stride = align_up(hs, gpu.rt.handle_alignment as u64);
    let ba = gpu.rt.base_alignment as u64;
    let off_miss = align_up(stride, ba);
    let off_hit = align_up(off_miss + 2 * stride, ba);
    let total = off_hit + 4 * stride;
    let handles = unsafe { gpu.rt_fn.get_ray_tracing_shader_group_handles(pipe, 0, 7, 7 * hs as usize)? };
    let mut sbt = gpu.create_buffer(total + ba, vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR, MemoryLocation::CpuToGpu, "sbt")?;
    let base = align_up(sbt.addr, ba);
    let shift = (base - sbt.addr) as usize;
    {
        let m = sbt.mapped();
        for (group, off) in
            [(0usize, 0u64), (1, off_miss), (2, off_miss + stride), (3, off_hit), (4, off_hit + stride), (5, off_hit + 2 * stride), (6, off_hit + 3 * stride)]
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
        hit: region(off_hit, 4 * stride),
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
        catalog: Catalog,
        world: World,
        imp: ImpostorSettings,
        vsync: bool,
        validation: bool,
        render_scale: f32,
        dlss_mode: Option<usize>,
        globe: Option<std::sync::Arc<crate::globe::dem::Terrain>>,
    ) -> Result<Self> {
        let t0 = Instant::now();
        let gpu = Gpu::new(display, window, validation)?;
        println!("GPU: {}", gpu.device_name);
        let dlss_ok = match crate::dlss::init(&gpu.entry, &gpu.instance, gpu.pdev, &gpu.device) {
            Ok(()) => {
                println!("DLSS Ray Reconstruction available");
                true
            }
            Err(e) => {
                println!("DLSS Ray Reconstruction off: {e}");
                false
            }
        };
        let dlss_mode = if dlss_ok { dlss_mode } else { None };
        let (mat_tex, leaf_scale, has_tex) = crate::textures::load_species(&gpu, &catalog)?;
        let mut ground_tex = crate::textures::load_ground(&gpu, &catalog.ground)?;
        // bit per scanned ground layer present (tri.rchit uses procedural stand-ins for the others)
        let ground_layers = catalog.ground.iter().enumerate().map(|(i, g)| (g.is_some() as u32) << i).sum::<u32>();
        {
            // biome mask in the last ground slot (see World::ground_mask)
            let t = Instant::now();
            const MASK_RES: usize = crate::world::COVER_RES;
            let mask = world.ground_mask(&pack.trees, &catalog);
            while ground_tex.len() < MAX_GROUND_TEX - 2 {
                ground_tex.push(crate::textures::upload_with_mips(&gpu, 1, 1, &[128, 128, 128, 255], vk::Format::R8G8B8A8_UNORM, "ground placeholder")?);
            }
            ground_tex.truncate(MAX_GROUND_TEX - 2);
            const FAR_RES: u32 = 2048;
            let far = world.far_mask(FAR_RES as usize, &pack.trees, &catalog);
            ground_tex.push(crate::textures::upload_with_mips(&gpu, FAR_RES, FAR_RES, &far, vk::Format::R8G8B8A8_UNORM, "far ground mask")?);
            let m = MASK_RES as u32;
            ground_tex.push(crate::textures::upload_with_mips(&gpu, m, m, &mask, vk::Format::R8G8B8A8_UNORM, "ground mask")?);
            println!("Ground mask {MASK_RES}^2 in {:.2}s", t.elapsed().as_secs_f32());
        }
        let swap = gpu.create_swapchain(width, height, vk::SwapchainKHR::null(), vsync)?;
        let render_extent = match dlss_mode {
            Some(m) => scaled_extent(swap.extent, crate::dlss::MODES[m].1),
            None => scaled_extent(swap.extent, render_scale),
        };
        let (out_img, tg, readback) = Self::create_targets(&gpu, swap.extent, render_extent)?;
        let sharp_img = Self::create_sharp(&gpu, swap.extent)?;

        // ---- geometry: plant meshes followed by terrain
        assert!(pack.meshes.len() < 2048, "mesh ids are 11 bits in the instance custom index");
        let mut meshes = Vec::new();
        for m in pack.meshes.iter().chain(std::iter::once(&world.terrain)) {
            meshes.push(upload_mesh(&gpu, m)?);
        }
        let infos: Vec<MeshInfoGpu> = meshes
            .iter()
            .enumerate()
            .map(|(i, m)| MeshInfoGpu {
                addr: [m.pos.addr, m.nrm.addr, m.idx.addr, m.mat.addr, m.uv.addr],
                tex_set: catalog.mesh_species.get(i).copied().unwrap_or(0),
                flags: if catalog.mesh_alpha.get(i).copied().unwrap_or(false) { MESH_ALPHA_TESTED } else { 0 },
            })
            .collect();
        let mesh_info = gpu.upload_buffer(bytemuck::cast_slice(&infos), vk::BufferUsageFlags::STORAGE_BUFFER, "mesh info")?;
        let mut species: Vec<SpeciesGpu> = catalog
            .species
            .iter()
            .enumerate()
            .map(|(i, s)| SpeciesGpu {
                // w: foliage brightness normalisation (textures::FOLIAGE_LUMINANCE)
                leaf_tint: [s.leaf_tint[0], s.leaf_tint[1], s.leaf_tint[2], leaf_scale.get(i).map(|x| x.0).unwrap_or(1.0)],
                bark_tint: [s.bark_tint[0], s.bark_tint[1], s.bark_tint[2], 0.0],
                // Transmitted light = leaf albedo x Unreal's translucency tint (0.5 = neutral) x
                // the translucency mask normalised to a mean of 0.33 (w) x TRANS_GAIN.
                trans_tint: [
                    2.0 * TRANS_GAIN * s.trans_tint[0],
                    2.0 * TRANS_GAIN * s.trans_tint[1],
                    2.0 * TRANS_GAIN * s.trans_tint[2],
                    0.33 / leaf_scale.get(i).map(|x| x.1).unwrap_or(0.33),
                ],
            })
            .collect();
        if species.is_empty() {
            species.push(SpeciesGpu { leaf_tint: [0.5, 0.5, 0.5, 1.0], bark_tint: [0.5; 4], trans_tint: [TRANS_GAIN, TRANS_GAIN, TRANS_GAIN, 1.0] });
        }
        let species_buf = gpu.upload_buffer(bytemuck::cast_slice(&species), vk::BufferUsageFlags::STORAGE_BUFFER, "species")?;
        let all_meshes: Vec<&MeshData> = pack.meshes.iter().chain(std::iter::once(&world.terrain)).collect();
        let geoms: Vec<_> = meshes
            .iter()
            .zip(&all_meshes)
            .map(|(g, m)| (accel::triangles_geometry(g.pos.addr, m.vertex_count(), g.idx.addr), m.tri_count()))
            .collect();
        let blas = accel::build_blases(&gpu, &geoms)?;
        println!("BLAS: {} meshes built in {:.2}s", blas.len(), t0.elapsed().as_secs_f32());

        // ---- impostor proxies and atlases: one per plant variant, resolution from its size
        let nv = pack.trees.len();
        assert!(nv <= MAX_VARIANTS, "too many plant variants ({nv} > {MAX_VARIANTS})");
        let mut aabbs = Vec::new();
        let mut imp_infos = Vec::new();
        let mut atlases = Vec::new();
        let mut atlas_bytes = 0u64;
        let mut raw_bytes = 0u64;
        let usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST;
        for (v, t) in pack.trees.iter().enumerate() {
            let vi = &catalog.variants[v];
            let sp = &species[vi.species.min(species.len() - 1)];
            let pad = Vec3::splat(0.05);
            let (lo, hi) = (t.bmin - pad, t.bmax + pad);
            let c = (t.bmin + t.bmax) * 0.5;
            let r = (t.bmax - t.bmin).length() * 0.5;
            // Forest tiles are seen from > 1 km only: fixed 256 px frames (~0.6 m texels).
            let res = imp.res_for(r / vi.texel_scale, vi.lod_scale);
            let mips = atlas_mips(res);
            imp_infos.push(ImpostorInfo {
                center_radius: [c.x, c.y, c.z, r],
                // w: 1 = foliage only (grass, ferns: no bark, so never classify texels as bark by colour)
                bmin: [lo.x, lo.y, lo.z, matches!(vi.kind, crate::catalog::Kind::Grass | crate::catalog::Kind::Plant) as u32 as f32],
                bmax: [hi.x, hi.y, hi.z, 0.0],
                // mean transmission (x albedo in impostor.rchit); w = draw distance
                trans: [0.33 * sp.trans_tint[0], 0.33 * sp.trans_tint[1], 0.33 * sp.trans_tint[2], vi.cull],
                grid: imp.grid,
                res,
                mips,
                lod_scale: vi.lod_scale,
            });
            let data = [lo.x, lo.y, lo.z, hi.x, hi.y, hi.z];
            aabbs.push(gpu.upload_buffer(
                bytemuck::cast_slice(&data),
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                "aabb",
            )?);
            let size = imp.grid * res;
            let a = BcAtlas {
                albedo: gpu.create_image(size, size, mips, vk::Format::BC3_UNORM_BLOCK, usage, &format!("{} albedo", t.name))?,
                normal: gpu.create_image(size, size, mips, vk::Format::BC5_UNORM_BLOCK, usage, &format!("{} normal", t.name))?,
                depth: gpu.create_image(size, size, mips, vk::Format::BC5_UNORM_BLOCK, usage, &format!("{} depth", t.name))?,
                vis: gpu.create_image(size / 2, size / 2, mips - 1, vk::Format::BC3_UNORM_BLOCK, usage, &format!("{} visibility", t.name))?,
                grid: imp.grid,
                res,
                mips,
            };
            atlas_bytes += a.bytes();
            raw_bytes += (size as u64 * size as u64 * 16) * 4 / 3; // the old rgba8 + rgba16f + rgba8 layout
            atlases.push(a);
        }
        println!(
            "Impostor atlases: {} variants, {:.2} GB BC-compressed (uncompressed would be {:.2} GB)",
            atlases.len(),
            atlas_bytes as f64 / 1e9,
            raw_bytes as f64 / 1e9
        );
        let imp_geoms: Vec<_> = aabbs.iter().map(|b| (accel::aabb_geometry(b.addr), 1)).collect();
        let imp_blas = accel::build_blases(&gpu, &imp_geoms)?;
        let imp_info = gpu.upload_buffer(bytemuck::cast_slice(&imp_infos), vk::BufferUsageFlags::STORAGE_BUFFER, "impostor info")?;
        gpu.one_time(|cb| {
            for a in &atlases {
                for img in [a.albedo.raw, a.normal.raw, a.depth.raw, a.vis.raw] {
                    gpu.image_barrier(
                        cb,
                        img,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::PipelineStageFlags2::NONE,
                        vk::AccessFlags2::NONE,
                        vk::PipelineStageFlags2::ALL_COMMANDS,
                        vk::AccessFlags2::SHADER_READ,
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
                    .max_lod(16.0),
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
                    .anisotropy_enable(false)
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
        let nvar = MAX_VARIANTS as u32;
        let bindings = [
            b(0, vk::DescriptorType::ACCELERATION_STRUCTURE_KHR, 1, rt_all),
            b(1, vk::DescriptorType::STORAGE_IMAGE, 1, rt_all),
            b(2, vk::DescriptorType::STORAGE_IMAGE, 1, rt_all),
            b(3, vk::DescriptorType::UNIFORM_BUFFER, 1, rt_all),
            b(4, vk::DescriptorType::STORAGE_BUFFER, 1, rt_all),
            b(5, vk::DescriptorType::STORAGE_BUFFER, 1, rt_all),
            b(6, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, nvar, rt_all),
            b(7, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, nvar, rt_all),
            b(8, vk::DescriptorType::STORAGE_IMAGE, 1, rt_all),
            b(9, vk::DescriptorType::STORAGE_IMAGE, 1, rt_all),
            b(10, vk::DescriptorType::STORAGE_IMAGE, 1, rt_all),
            b(11, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, nvar, rt_all),
            b(12, vk::DescriptorType::STORAGE_BUFFER, 1, rt_all),
            b(13, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 4 * MAX_SPECIES as u32, rt_all),
            b(14, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, nvar, rt_all),
            b(15, vk::DescriptorType::STORAGE_BUFFER, 1, rt_all),
            b(16, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, MAX_GROUND_TEX as u32, rt_all),
            // DLSS Ray Reconstruction G-buffer (main.rgen): view depth, motion, normal+roughness,
            // diffuse albedo, specular albedo
            b(17, vk::DescriptorType::STORAGE_IMAGE, 1, rt_all),
            b(18, vk::DescriptorType::STORAGE_IMAGE, 1, rt_all),
            b(19, vk::DescriptorType::STORAGE_IMAGE, 1, rt_all),
            b(20, vk::DescriptorType::STORAGE_IMAGE, 1, rt_all),
            b(21, vk::DescriptorType::STORAGE_IMAGE, 1, rt_all),
            // plant records and per-plant LOD state (impostor clusters)
            b(23, vk::DescriptorType::STORAGE_BUFFER, 1, rt_all),
            b(24, vk::DescriptorType::STORAGE_BUFFER, 1, rt_all),
            // globe mode (globe.rchit): its globals, terrain tile table, elevation grid
            b(30, vk::DescriptorType::UNIFORM_BUFFER, 1, rt_all),
            b(31, vk::DescriptorType::STORAGE_BUFFER, 1, rt_all),
            b(32, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 1, rt_all),
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
        let compute_pipe = |module: &[u8], layout: vk::PipelineLayout| -> Result<vk::Pipeline> {
            let m = gpu.shader_module(module)?;
            let p = unsafe {
                gpu.device
                    .create_compute_pipelines(
                        vk::PipelineCache::null(),
                        &[vk::ComputePipelineCreateInfo::default()
                            .stage(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(m).name(c"main"))
                            .layout(layout)],
                        None,
                    )
                    .map_err(|(_, e)| e)?[0]
            };
            unsafe { gpu.device.destroy_shader_module(m, None) };
            Ok(p)
        };
        let mip_bindings: Vec<_> = (0..6).map(|i| b(i, vk::DescriptorType::STORAGE_IMAGE, 1, vk::ShaderStageFlags::COMPUTE)).collect();
        let mip_set_layout = unsafe {
            gpu.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&mip_bindings), None)?
        };
        let mip_layout = unsafe {
            gpu.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[mip_set_layout]), None)?
        };
        let mip_pipe = compute_pipe(spv!("mip.comp"), mip_layout)?;
        // BC encoder: 3 scratch atlas levels in, block buffer out.
        let bcenc_bindings = [
            b(0, vk::DescriptorType::STORAGE_IMAGE, 1, vk::ShaderStageFlags::COMPUTE),
            b(1, vk::DescriptorType::STORAGE_IMAGE, 1, vk::ShaderStageFlags::COMPUTE),
            b(2, vk::DescriptorType::STORAGE_IMAGE, 1, vk::ShaderStageFlags::COMPUTE),
            b(3, vk::DescriptorType::STORAGE_BUFFER, 1, vk::ShaderStageFlags::COMPUTE),
        ];
        let bcenc_set_layout = unsafe {
            gpu.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bcenc_bindings), None)?
        };
        let bcenc_push = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(16)];
        let bcenc_layout = unsafe {
            gpu.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&[bcenc_set_layout]).push_constant_ranges(&bcenc_push),
                None,
            )?
        };
        let bcenc_pipe = compute_pipe(spv!("bcenc.comp"), bcenc_layout)?;

        let pool_sizes = [
            vk::DescriptorPoolSize { ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR, descriptor_count: 2 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: 64 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: 8 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 24 },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 2 * (4 * nvar + 4 * MAX_SPECIES as u32 + MAX_GROUND_TEX as u32 + 1),
            },
        ];
        let desc_pool = unsafe {
            gpu.device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().pool_sizes(&pool_sizes).max_sets(9), None)?
        };
        let sets = unsafe {
            gpu.device
                .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(desc_pool).set_layouts(&[set_layout, set_layout]))?
        };

        let main_pipe = if gpu.ser {
            println!("Shader Execution Reordering on");
            create_rt_pipeline(&gpu, pipe_layout, spv!("main_ser.rgen"))?
        } else {
            create_rt_pipeline(&gpu, pipe_layout, spv!("main.rgen"))?
        };
        let bake_pipe = create_rt_pipeline(&gpu, pipe_layout, spv!("bake.rgen"))?;

        // ---- supersampling resolve (render resolution HDR -> display resolution)
        let resolve_bindings: Vec<_> = (0..2).map(|i| b(i, vk::DescriptorType::STORAGE_IMAGE, 1, vk::ShaderStageFlags::COMPUTE)).collect();
        let resolve_set_layout = unsafe {
            gpu.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&resolve_bindings), None)?
        };
        let resolve_push = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(16)];
        let resolve_layout = unsafe {
            gpu.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&[resolve_set_layout]).push_constant_ranges(&resolve_push),
                None,
            )?
        };
        let resolve_pipe = compute_pipe(spv!("resolve.comp"), resolve_layout)?;
        // ---- CAS sharpening (display resolution)
        let sharpen_bindings: Vec<_> = (0..2).map(|i| b(i, vk::DescriptorType::STORAGE_IMAGE, 1, vk::ShaderStageFlags::COMPUTE)).collect();
        let sharpen_set_layout = unsafe {
            gpu.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&sharpen_bindings), None)?
        };
        let sharpen_push = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(4)];
        let sharpen_layout = unsafe {
            gpu.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&[sharpen_set_layout]).push_constant_ranges(&sharpen_push),
                None,
            )?
        };
        let sharpen_pipe = compute_pipe(spv!("sharpen.comp"), sharpen_layout)?;
        let sharpen_set = unsafe {
            gpu.device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(desc_pool).set_layouts(&[sharpen_set_layout]))?[0]
        };
        let [rs0, rs1, rs2]: [vk::DescriptorSet; 3] = unsafe {
            gpu.device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(desc_pool)
                        .set_layouts(&[resolve_set_layout, resolve_set_layout, resolve_set_layout]),
                )?
                .try_into()
                .unwrap()
        };
        let resolve_sets = [rs0, rs1];
        let dlss_resolve_set = rs2;

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
        let taa_push = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(4)];
        let taa_layout = unsafe {
            gpu.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&[taa_set_layout]).push_constant_ranges(&taa_push),
                None,
            )?
        };
        let taa_pipe = compute_pipe(spv!("taa.comp"), taa_layout)?;
        let taa_sets: [vk::DescriptorSet; 2] = unsafe {
            gpu.device
                .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(desc_pool).set_layouts(&[taa_set_layout, taa_set_layout]))?
                .try_into()
                .unwrap()
        };

        let est = 600_000u32;
        let world_trees_len = world.trees.len();
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
        let query_pool = unsafe {
            gpu.device.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(5), None)?
        };
        let lod_state = vec![Lod::Impostor; world.trees.len()];
        let (clusters, max_lod_scale) = Self::build_clusters(&gpu, &world, &pack, &catalog, blas[pack.meshes.len()].addr)?;
        let grid = PlantGrid::new(&world);

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
            dlss_resolve_set,
            dlss_ok,
            dlss_mode,
            last_frame: Instant::now(),
            dlss_jitter_sign: {
                let v = std::env::var("SI_DLSS_JITTER").unwrap_or_default();
                // NGX wants the projection offset, the opposite of where the samples sit (measured:
                // sharpest with both flipped). "+x"/"+y" restore the raw sample offset for testing.
                (if v.contains("+x") { 1.0 } else { -1.0 }, if v.contains("+y") { 1.0 } else { -1.0 })
            },
            parity: 0,
            taa_reset: true,
            taa_enabled: true,
            resize_count: 0,
            mat_tex,
            ground_tex,
            ground_layers,
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
            mip_set_layout,
            bcenc_layout,
            bcenc_pipe,
            bcenc_set_layout,
            _meshes: meshes,
            mesh_info,
            species_buf,
            blas,
            _aabbs: aabbs,
            imp_blas,
            imp_info,
            imp_infos,
            atlases,
            sampler,
            catalog,
            tlas,
            ubo,
            cb,
            fence,
            acquire_sem,
            query_pool,
            queries_pending: false,
            profile: Profile::default(),
            world,
            pack,
            imp,
            dump_prefix: std::env::var("SI_DUMP_ATLAS").ok(),
            lod_state,
            active: Vec::new(),
            visited: vec![0; world_trees_len],
            lod_frame: 0,
            grid,
            max_lod_scale,
            rec_of: clusters.rec_of,
            plant_rec: clusters.rec,
            plant_state: clusters.state,
            state_writes: Vec::new(),
            _cluster_blas: clusters.blas,
            _cluster_aabbs: clusters.aabbs,
            static_insts: clusters.insts,
            static_inst_data: clusters.inst_data,
            insts: Vec::new(),
            inst_data: Vec::new(),
            inst_info,
            tlas_dirty: true,
            upload_pending: false,
            first_update: true,
            near_count: 0,
            need_resize: false,
            render_scale,
            render_extent,
            resolve_layout,
            resolve_pipe,
            globe: None,
            globe_dummy: None,
            globe_view: None,
            globe_states: std::collections::HashMap::new(),
        };
        let _ = r.set_layout;
        // Globe mode: terrain and trees stream in around the camera (globe/scene.rs); the static
        // world is empty. Otherwise the globe bindings get 1-element placeholders.
        match globe {
            Some(terrain) => {
                let mut species: Vec<(String, Vec<usize>)> = Vec::new();
                for sp in &r.catalog.species {
                    if sp.kind == crate::catalog::Kind::Tree {
                        let mut v: Vec<usize> = sp.variants.clone().collect();
                        v.sort_by(|a, b| r.catalog.variants[*b].height.partial_cmp(&r.catalog.variants[*a].height).unwrap());
                        species.push((sp.name.clone(), v));
                    }
                }
                // understory variants by kind / name (catalog.rs Kind)
                let mut sp = crate::globe::forest::Species { infos: r.imp_infos.clone(), species, ..Default::default() };
                for s in &r.catalog.species {
                    let v: Vec<usize> = s.variants.clone().collect();
                    let n = s.name.to_ascii_lowercase();
                    match s.kind {
                        crate::catalog::Kind::Grass if n.contains("dry") => sp.dry_grass.extend(v),
                        crate::catalog::Kind::Grass if n.contains("cotton") => sp.wet_grass.extend(v),
                        crate::catalog::Kind::Grass => sp.grass.extend(v),
                        crate::catalog::Kind::Plant => sp.ferns.extend(v),
                        crate::catalog::Kind::Shrub if n.contains("heather") => sp.heath.extend(v),
                        crate::catalog::Kind::Shrub => sp.shrubs.extend(v),
                        _ => {}
                    }
                }
                if sp.dry_grass.is_empty() {
                    sp.dry_grass = sp.grass.clone();
                }
                if sp.wet_grass.is_empty() {
                    sp.wet_grass = sp.grass.clone();
                }
                if sp.heath.is_empty() {
                    sp.heath = sp.shrubs.clone();
                }
                r.globe = Some(crate::globe::scene::GlobeScene::new(&r.gpu, terrain, sp)?);
            }
            None => {
                let ubo = r.gpu.create_buffer(256, vk::BufferUsageFlags::UNIFORM_BUFFER, MemoryLocation::CpuToGpu, "globe placeholder")?;
                let buf = r.gpu.create_buffer(16, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "globe placeholder")?;
                let img = crate::textures::upload_with_mips(&r.gpu, 1, 1, &[0, 0, 0, 0], vk::Format::R8G8B8A8_UNORM, "globe placeholder")?;
                r.globe_dummy = Some((ubo, buf, img));
            }
        }
        r.write_set(r.main_set, r.tlas.accel.raw, None);
        r.write_resolve_set();
        r.create_dlss_feature()?;
        r.bake_impostors()?;
        println!("Renderer ready in {:.2}s", t0.elapsed().as_secs_f32());
        Ok(r)
    }

    /// Group the plants into CLUSTER x CLUSTER m cells, one AABB BLAS per cell with one box per
    /// plant (its world bounds), and write the plant records in cluster order. The TLAS gets one
    /// instance per cluster (custom index = first record of the cluster) plus the terrain.
    fn build_clusters(gpu: &Gpu, world: &World, pack: &TreePack, catalog: &Catalog, terrain_blas: u64) -> Result<(Clusters, f32)> {
        const CLUSTER: f32 = 192.0;
        let t0 = Instant::now();
        let n = world.trees.len();
        let key = |p: Vec3| (((p.z / CLUSTER).floor() as i32) as i64) << 32 | (((p.x / CLUSTER).floor() as i32) as u32 as i64);
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_by_key(|&i| key(world.trees[i as usize].center));
        let mut rec_of = vec![0u32; n];
        let mut recs: Vec<PlantRecGpu> = Vec::with_capacity(n);
        let mut aabbs: Vec<[f32; 6]> = Vec::with_capacity(n);
        let mut ranges: Vec<(u32, u32)> = Vec::new(); // (first record, count)
        let mut last_key = None;
        for (k, &i) in order.iter().enumerate() {
            let t = &world.trees[i as usize];
            rec_of[i as usize] = k as u32;
            let kk = key(t.center);
            if last_key != Some(kk) {
                ranges.push((k as u32, 0));
                last_key = Some(kk);
            }
            ranges.last_mut().unwrap().1 += 1;
            let c = t.xform.matrix3;
            let o = t.xform.translation;
            recs.push(PlantRecGpu {
                rows: [
                    c.x_axis.x, c.y_axis.x, c.z_axis.x, o.x, //
                    c.x_axis.y, c.y_axis.y, c.z_axis.y, o.y, //
                    c.x_axis.z, c.y_axis.z, c.z_axis.z, o.z,
                ],
                data: t.variant as u32 | ((t.seed as u32 & 0x1FFF) << 8),
                pad: [0; 3],
            });
            let asset = &pack.trees[t.variant as usize];
            let (lo, hi) = (asset.bmin - Vec3::splat(0.05), asset.bmax + Vec3::splat(0.05));
            let (mut mn, mut mx) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            for corner in 0..8 {
                let v = Vec3::new(
                    if corner & 1 == 0 { lo.x } else { hi.x },
                    if corner & 2 == 0 { lo.y } else { hi.y },
                    if corner & 4 == 0 { lo.z } else { hi.z },
                );
                let w = t.xform.transform_point3(v);
                mn = mn.min(w);
                mx = mx.max(w);
            }
            aabbs.push([mn.x, mn.y, mn.z, mx.x, mx.y, mx.z]);
        }
        let aabb_buf = gpu.upload_buffer(
            bytemuck::cast_slice(&aabbs),
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            "plant aabbs",
        )?;
        let geoms: Vec<_> = ranges.iter().map(|&(first, count)| (accel::aabb_geometry(aabb_buf.addr + first as u64 * 24), count)).collect();
        let blas = accel::build_blases(gpu, &geoms)?;
        let rec = gpu.upload_buffer(bytemuck::cast_slice(&recs), vk::BufferUsageFlags::STORAGE_BUFFER, "plant records")?;
        let mut state = gpu.create_buffer((n.max(1) * 4) as u64, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "plant state")?;
        state.mapped().fill(0);
        let terrain_id = pack.meshes.len();
        let mesh_flags = vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE | vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE;
        let mut insts = vec![accel::make_instance(&glam::Affine3A::IDENTITY, terrain_id as u32, 1, 0, mesh_flags, terrain_blas)];
        let mut inst_data = vec![InstInfo { center: [0.0, -1e9, 0.0], data: 0 }];
        for (ci, &(first, _)) in ranges.iter().enumerate() {
            // mask bit 1: impostors; hit record 1: impostor intersection + closest hit
            insts.push(accel::make_instance(
                &glam::Affine3A::IDENTITY,
                first,
                2,
                1,
                vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE,
                blas[ci].addr,
            ));
            inst_data.push(InstInfo { center: [0.0, -1e9, 0.0], data: 0 });
        }
        let max_ls = catalog.variants.iter().map(|v| v.lod_scale).fold(0.0f32, f32::max);
        println!("Impostor clusters: {} plants in {} cluster BLASes ({:.2}s)", n, ranges.len(), t0.elapsed().as_secs_f32());
        Ok((Clusters { rec_of, rec, state, blas, aabbs: aabb_buf, insts, inst_data }, max_ls))
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
        let ss = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED; // also read by NGX
        let tg = Targets {
            cur: gpu.create_image(w, h, 1, vk::Format::R16G16B16A16_SFLOAT, ss, "frame sample")?,
            depth: gpu.create_image(w, h, 1, vk::Format::R32_SFLOAT, st, "hit distance")?,
            hist: [
                gpu.create_image(w, h, 1, vk::Format::R16G16B16A16_SFLOAT, st, "taa history 0")?,
                gpu.create_image(w, h, 1, vk::Format::R16G16B16A16_SFLOAT, st, "taa history 1")?,
            ],
            viewz: gpu.create_image(w, h, 1, vk::Format::R32_SFLOAT, ss, "view depth")?,
            motion: gpu.create_image(w, h, 1, vk::Format::R16G16_SFLOAT, ss, "motion")?,
            nrm: gpu.create_image(w, h, 1, vk::Format::R16G16B16A16_SFLOAT, ss, "normal roughness")?,
            diff: gpu.create_image(w, h, 1, vk::Format::R8G8B8A8_UNORM, ss, "diffuse albedo")?,
            spec: gpu.create_image(w, h, 1, vk::Format::R8G8B8A8_UNORM, ss, "specular albedo")?,
            dlss_out: gpu.create_image(extent.width, extent.height, 1, vk::Format::R16G16B16A16_SFLOAT, ss, "dlss output")?,
        };
        gpu.one_time(|cb| {
            for img in [
                out.raw,
                tg.cur.raw,
                tg.depth.raw,
                tg.hist[0].raw,
                tg.hist[1].raw,
                tg.viewz.raw,
                tg.motion.raw,
                tg.nrm.raw,
                tg.diff.raw,
                tg.spec.raw,
                tg.dlss_out.raw,
            ] {
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

    /// Write every binding of a ray tracing descriptor set. `bake` = scratch atlas views bound as
    /// the bake targets (bindings 8/9/10); only the bake pipeline uses them.
    fn write_set(&self, set: vk::DescriptorSet, tlas: vk::AccelerationStructureKHR, bake: Option<[vk::ImageView; 3]>) {
        let handles = [tlas];
        let mut as_info = vk::WriteDescriptorSetAccelerationStructureKHR::default().acceleration_structures(&handles);
        let img = |v: vk::ImageView| [vk::DescriptorImageInfo::default().image_view(v).image_layout(vk::ImageLayout::GENERAL)];
        let out = img(self.tg.depth.mip_views[0]); // binding 1: hit distance
        let accum = img(self.tg.cur.mip_views[0]); // binding 2: this frame's sample
        let bake_imgs = bake.map(|v| [img(v[0]), img(v[1]), img(v[2])]);
        let buf = |b: &Buffer| [vk::DescriptorBufferInfo::default().buffer(b.raw).range(vk::WHOLE_SIZE)];
        let ubo = buf(&self.ubo);
        let meshes = buf(&self.mesh_info);
        let imps = buf(&self.imp_info);
        let inst_info = buf(&self.inst_info);
        let species = buf(&self.species_buf);
        let (plant_rec, plant_state) = match &self.globe {
            Some(g) => (buf(&g.plants), buf(&g.plant_state)),
            None => (buf(&self.plant_rec), buf(&self.plant_state)),
        };
        let (globe_ubo, globe_table, globe_elev) = match (&self.globe, &self.globe_dummy) {
            (Some(g), _) => (buf(&g.ubo), buf(&g.table), g.elev_tex.view),
            (None, Some(d)) => (buf(&d.0), buf(&d.1), d.2.view),
            _ => unreachable!("globe bindings without resources"),
        };
        let globe_elev = [vk::DescriptorImageInfo::default().sampler(self.mat_sampler).image_view(globe_elev).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let tex = |list: &[Image], n: usize| -> Vec<vk::DescriptorImageInfo> {
            (0..n)
                .map(|i| {
                    vk::DescriptorImageInfo::default()
                        .sampler(self.mat_sampler)
                        .image_view(list[i.min(list.len() - 1)].view)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                })
                .collect()
        };
        let mat_tex = tex(&self.mat_tex, 4 * MAX_SPECIES);
        let ground_tex = tex(&self.ground_tex, MAX_GROUND_TEX);
        let sampled = |pick: fn(&BcAtlas) -> vk::ImageView| -> Vec<vk::DescriptorImageInfo> {
            (0..MAX_VARIANTS)
                .map(|i| {
                    let a = &self.atlases[i.min(self.atlases.len() - 1)];
                    vk::DescriptorImageInfo::default()
                        .sampler(self.sampler)
                        .image_view(pick(a))
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                })
                .collect()
        };
        let albedo = sampled(|a| a.albedo.view);
        let depth = sampled(|a| a.depth.view);
        let visib = sampled(|a| a.vis.view);
        let normal = sampled(|a| a.normal.view);
        let w = |binding: u32, ty: vk::DescriptorType| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(binding).descriptor_type(ty);
        let mut as_write = w(0, vk::DescriptorType::ACCELERATION_STRUCTURE_KHR).push_next(&mut as_info);
        as_write.descriptor_count = 1;
        let mut writes = vec![
            as_write,
            w(1, vk::DescriptorType::STORAGE_IMAGE).image_info(&out),
            w(2, vk::DescriptorType::STORAGE_IMAGE).image_info(&accum),
            w(3, vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&ubo),
            w(4, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&meshes),
            w(5, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&imps),
            w(6, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&albedo),
            w(7, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&depth),
            w(11, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&visib),
            w(12, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&inst_info),
            w(13, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&mat_tex),
            w(14, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&normal),
            w(15, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&species),
            w(16, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&ground_tex),
            w(23, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&plant_rec),
            w(24, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&plant_state),
            w(30, vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&globe_ubo),
            w(31, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&globe_table),
            w(32, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&globe_elev),
        ];
        let gbuf = [img(self.tg.viewz.view), img(self.tg.motion.view), img(self.tg.nrm.view), img(self.tg.diff.view), img(self.tg.spec.view)];
        for (k, info) in gbuf.iter().enumerate() {
            writes.push(w(17 + k as u32, vk::DescriptorType::STORAGE_IMAGE).image_info(info));
        }
        if let Some(bi) = &bake_imgs {
            for (k, info) in bi.iter().enumerate() {
                writes.push(w(8 + k as u32, vk::DescriptorType::STORAGE_IMAGE).image_info(info));
            }
        }
        unsafe { self.gpu.device.update_descriptor_sets(&writes, &[]) };
    }

    /// Bake every plant variant into its hemi-octahedral atlas with hardware ray tracing, build
    /// coverage-weighted mips, then BC-compress every level on the GPU (bcenc.comp) into the
    /// sampled atlases: BC3 albedo+AO, BC5 normal, BC5 depth+coverage, BC3 visibility at half
    /// resolution. The uncompressed scratch atlas is shared by all variants and freed afterwards.
    fn bake_impostors(&mut self) -> Result<()> {
        let t_all = Instant::now();
        let imp = self.imp;
        let max_size = self.atlases.iter().map(|a| a.grid * a.res).max().unwrap_or(32);
        let smips = self.atlases.iter().map(|a| a.mips).max().unwrap_or(1);
        let st = vk::ImageUsageFlags::STORAGE;
        let mut scratch = [
            self.gpu.create_image(max_size, max_size, smips, vk::Format::R8G8B8A8_UNORM, st, "bake albedo")?,
            self.gpu.create_image(max_size, max_size, smips, vk::Format::R16G16B16A16_SFLOAT, st, "bake normal-depth")?,
            self.gpu.create_image(max_size, max_size, smips, vk::Format::R8G8B8A8_UNORM, st, "bake visibility")?,
        ];
        self.gpu.one_time(|cb| {
            for s in &scratch {
                self.gpu.image_barrier(
                    cb,
                    s.raw,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::GENERAL,
                    vk::PipelineStageFlags2::NONE,
                    vk::AccessFlags2::NONE,
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                    vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ,
                );
            }
        })?;
        // Encoded blocks of one variant (all levels of all four atlases): < 4.5 bytes per level-0 texel.
        let enc_size = max_size as u64 * max_size as u64 * 9 / 2;
        let mut enc = self.gpu.create_buffer(
            enc_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::GpuOnly,
            "bc blocks",
        )?;

        // Transient descriptor sets: mip level L-1 -> L, and encoder inputs at level L.
        let d = self.gpu.device.clone();
        let pool_sizes = [
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: 9 * smips },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: smips },
        ];
        let pool = unsafe { d.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().pool_sizes(&pool_sizes).max_sets(2 * smips), None)? };
        let alloc = |layout: vk::DescriptorSetLayout| -> Result<vk::DescriptorSet> {
            Ok(unsafe { d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[layout]))?[0] })
        };
        let storage = |set: vk::DescriptorSet, views: &[vk::ImageView]| {
            let infos: Vec<[vk::DescriptorImageInfo; 1]> =
                views.iter().map(|v| [vk::DescriptorImageInfo::default().image_view(*v).image_layout(vk::ImageLayout::GENERAL)]).collect();
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
            unsafe { d.update_descriptor_sets(&writes, &[]) };
        };
        let [sa, sb, sc] = &scratch;
        let mut mip_sets = vec![vk::DescriptorSet::null()];
        for l in 1..smips as usize {
            let set = alloc(self.mip_set_layout)?;
            storage(set, &[sa.mip_views[l - 1], sb.mip_views[l - 1], sa.mip_views[l], sb.mip_views[l], sc.mip_views[l - 1], sc.mip_views[l]]);
            mip_sets.push(set);
        }
        let mut enc_sets = Vec::new();
        for l in 0..smips as usize {
            let set = alloc(self.bcenc_set_layout)?;
            storage(set, &[sa.mip_views[l], sb.mip_views[l], sc.mip_views[l]]);
            let bi = [vk::DescriptorBufferInfo::default().buffer(enc.raw).range(vk::WHOLE_SIZE)];
            let wr = [vk::WriteDescriptorSet::default().dst_set(set).dst_binding(3).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&bi)];
            unsafe { d.update_descriptor_sets(&wr, &[]) };
            enc_sets.push(set);
        }
        let bake_views = [sa.mip_views[0], sb.mip_views[0], sc.mip_views[0]];

        let mut total_tris = 0u64;
        for v in 0..self.atlases.len() {
            let t0 = Instant::now();
            let (res, grid, mips) = (self.atlases[v].res, self.atlases[v].grid, self.atlases[v].mips);
            let tree = &self.pack.trees[v];
            let flags = vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE | vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE;
            let alpha_flags = vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE | vk::GeometryInstanceFlagsKHR::FORCE_NO_OPAQUE;
            let insts: Vec<_> = tree
                .parts
                .iter()
                .map(|p| {
                    let alpha = self.catalog.mesh_alpha.get(p.mesh as usize).copied().unwrap_or(false);
                    let (f, sbt) = if alpha { (alpha_flags, 2) } else { (flags, 0) };
                    accel::make_instance(&p.xform, p.mesh, 0xFF, sbt, f, self.blas[p.mesh as usize].addr)
                })
                .collect();
            let mut tlas = Tlas::new(&self.gpu, insts.len() as u32)?;
            tlas.write(&insts);
            self.gpu.one_time(|cb| tlas.cmd_build(&self.gpu, cb, insts.len() as u32))?;
            self.write_set(self.bake_set, tlas.accel.raw, Some(bake_views));

            let c = (tree.bmin + tree.bmax) * 0.5;
            let r = (tree.bmax - tree.bmin).length() * 0.5;
            // Bake rays are orthographic: fixed texture footprint = one bake sub-sample (negative
            // spread in tri.rchit), and the material textures switched on.
            let mut bg = Globals::default();
            bg.cam_fwd[3] = -(2.0 * r / res as f32 / imp.ss as f32);
            bg.params[1] = self.has_tex as u32 as f32;
            bg.lod_params[2] = -1.0; // bake: tri.rahit only alpha-tests, no LOD dissolve
            self.ubo.mapped()[..std::mem::size_of::<Globals>()].copy_from_slice(bytemuck::bytes_of(&bg));
            let mut push = BakePush {
                center_radius: [c.x, c.y, c.z, r],
                grid,
                res,
                frame_offset: 0,
                ss: imp.ss,
                ao_rays: imp.ao_rays,
                // AO range is in metres for full-size trees; small plants get a proportionally shorter one.
                ao_range: imp.ao_range * (r / 8.0).clamp(0.2, 1.0),
                vis_rays: imp.vis_rays,
                pad: 0,
            };
            for row in 0..grid {
                push.frame_offset = row * grid;
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
                        res,
                        res,
                        grid,
                    );
                })?;
            }

            let size = grid * res;
            let a = &self.atlases[v];
            self.gpu.one_time(|cb| unsafe {
                let rw = vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE;
                let cs = vk::PipelineStageFlags2::COMPUTE_SHADER;
                self.gpu.memory_barrier(cb, vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR, vk::AccessFlags2::SHADER_WRITE, cs, rw);
                // coverage-weighted mips of the uncompressed scratch atlas
                d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.mip_pipe);
                for level in 1..mips {
                    let s = size >> level;
                    d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, self.mip_layout, 0, &[mip_sets[level as usize]], &[]);
                    d.cmd_dispatch(cb, s.div_ceil(8), s.div_ceil(8), 1);
                    self.gpu.memory_barrier(cb, cs, vk::AccessFlags2::SHADER_WRITE, cs, rw);
                }
                // BC encode every level: (mode, scratch level, target image, target level)
                let mut jobs = Vec::new();
                for level in 0..mips {
                    for (mode, img) in [(0u32, &a.albedo), (1, &a.normal), (2, &a.depth)] {
                        jobs.push((mode, level, img, level));
                    }
                }
                for level in 1..mips {
                    jobs.push((3u32, level, &a.vis, level - 1)); // half resolution: from scratch level+1
                }
                d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.bcenc_pipe);
                let mut offset = 0u32; // in blocks
                let mut copies = Vec::new();
                for &(mode, level, img, dst_level) in &jobs {
                    let s = size >> level;
                    let bx = s / 4;
                    d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, self.bcenc_layout, 0, &[enc_sets[level as usize]], &[]);
                    let pc = [mode, bx, bx, offset];
                    d.cmd_push_constants(cb, self.bcenc_layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::cast_slice(&pc));
                    d.cmd_dispatch(cb, bx.div_ceil(8), bx.div_ceil(8), 1);
                    copies.push((
                        img.raw,
                        vk::BufferImageCopy {
                            buffer_offset: offset as u64 * 16,
                            buffer_row_length: 0,
                            buffer_image_height: 0,
                            image_subresource: vk::ImageSubresourceLayers {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                mip_level: dst_level,
                                base_array_layer: 0,
                                layer_count: 1,
                            },
                            image_offset: vk::Offset3D::default(),
                            image_extent: vk::Extent3D { width: s, height: s, depth: 1 },
                        },
                    ));
                    offset += bx * bx;
                }
                assert!(offset as u64 * 16 <= enc_size, "bc block buffer too small");
                self.gpu.memory_barrier(cb, cs, vk::AccessFlags2::SHADER_WRITE, vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_READ);
                let targets = [a.albedo.raw, a.normal.raw, a.depth.raw, a.vis.raw];
                for img in targets {
                    self.gpu.image_barrier(
                        cb,
                        img,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::PipelineStageFlags2::ALL_COMMANDS,
                        vk::AccessFlags2::NONE,
                        vk::PipelineStageFlags2::TRANSFER,
                        vk::AccessFlags2::TRANSFER_WRITE,
                    );
                }
                for (img, region) in &copies {
                    d.cmd_copy_buffer_to_image(cb, enc.raw, *img, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[*region]);
                }

                for img in targets {
                    self.gpu.image_barrier(
                        cb,
                        img,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::PipelineStageFlags2::TRANSFER,
                        vk::AccessFlags2::TRANSFER_WRITE,
                        vk::PipelineStageFlags2::ALL_COMMANDS,
                        vk::AccessFlags2::SHADER_READ,
                    );
                }
            })?;
            if v == 0 {
                if let Some(prefix) = self.dump_prefix.clone() {
                    self.dump_scratch(&scratch[0], size, &prefix)?;
                }
            }
            tlas.destroy(&self.gpu);
            let tris = tree.expanded_tris(&self.pack.meshes);
            total_tris += tris;
            println!(
                "Baked {:<28} {:>4} parts {:>6.1}M tris -> {:>4}px frames, {:>5}^2 atlas, {:>5.1} MB, {:.2}s",
                tree.name,
                tree.parts.len(),
                tris as f64 / 1e6,
                res,
                size,
                a.bytes() as f64 / 1e6,
                t0.elapsed().as_secs_f32()
            );
        }
        unsafe {
            d.device_wait_idle()?;
            d.destroy_descriptor_pool(pool, None);
        }
        for s in scratch.iter_mut() {
            self.gpu.destroy_image(s);
        }
        self.gpu.destroy_buffer(&mut enc);
        println!(
            "Baked {} impostors ({:.0}M source triangles) in {:.1}s",
            self.atlases.len(),
            total_tris as f64 / 1e6,
            t_all.elapsed().as_secs_f32()
        );
        Ok(())
    }

    /// Debug: write the albedo of the first baked variant (uncompressed scratch) as a PNG.
    fn dump_scratch(&self, img: &Image, size: u32, prefix: &str) -> Result<()> {
        let n = size as u64 * size as u64;
        let mut buf = self.gpu.create_buffer(n * 4, vk::BufferUsageFlags::TRANSFER_DST, MemoryLocation::GpuToCpu, "dump")?;
        let layers = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 };
        let region = vk::BufferImageCopy {
            image_subresource: layers,
            image_extent: vk::Extent3D { width: size, height: size, depth: 1 },
            ..Default::default()
        };
        self.gpu.one_time(|cb| unsafe {
            self.gpu.memory_barrier(cb, vk::PipelineStageFlags2::ALL_COMMANDS, vk::AccessFlags2::SHADER_WRITE, vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_READ);
            self.gpu.device.cmd_copy_image_to_buffer(cb, img.raw, vk::ImageLayout::GENERAL, buf.raw, &[region]);
        })?;
        let mut px = buf.mapped()[..(n * 4) as usize].to_vec();
        for p in px.chunks_mut(4) {
            for c in p.iter_mut().take(3) {
                *c = ((*c as f32) * 2.2).min(255.0) as u8;
            }
            p[3] = 255;
        }
        self.gpu.destroy_buffer(&mut buf);
        let path = format!("{prefix}_albedo.png");
        let file = std::io::BufWriter::new(std::fs::File::create(&path)?);
        let mut enc = png::Encoder::new(file, size, size);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header()?.write_image_data(&px)?;
        println!("wrote {path}");
        Ok(())
    }

    /// Every plant is always an impostor primitive of its cluster BLAS (static). Plants near the
    /// camera additionally get mesh instances: mesh only, or both while inside the mesh <-> impostor
    /// band (the shaders then dissolve between the two copies). Only plants in grid cells within
    /// reach of the largest mesh distance are examined; the TLAS (clusters + terrain + near meshes)
    /// is rebuilt only when a plant changes state.
    fn update_instances(&mut self, cam: Vec3, lod: f32, band: f32) -> Result<FrameStats> {
        let t_begin = Instant::now();
        let mut changed = self.first_update;
        self.first_update = false;
        self.lod_frame = self.lod_frame.wrapping_add(1);
        let stamp = self.lod_frame;
        // Candidates: currently active plants plus every plant in reach.
        let mut cand: Vec<u32> = std::mem::take(&mut self.active);
        for &i in &cand {
            self.visited[i as usize] = stamp;
        }
        if lod > 0.0 {
            let reach = (lod + band) * self.max_lod_scale + 4.0;
            let g = &self.grid;
            let to = |v: f32| (((v + g.half) / g.cell).floor() as isize).clamp(0, g.n as isize - 1) as usize;
            for j in to(cam.z - reach)..=to(cam.z + reach) {
                for i in to(cam.x - reach)..=to(cam.x + reach) {
                    for &p in &g.cells[j * g.n + i] {
                        if self.visited[p as usize] != stamp {
                            self.visited[p as usize] = stamp;
                            cand.push(p);
                        }
                    }
                }
            }
        }
        let mut active = Vec::new();
        for &pi in &cand {
            let i = pi as usize;
            let t = &self.world.trees[i];
            let ls = self.catalog.variants[t.variant as usize].lod_scale;
            // The shader fade is exactly 1 below a and exactly 0 above b, so "mesh only" /
            // "impostor only" are safe there; the hysteresis just keeps "both" a little longer.
            let (a, b) = if lod <= 0.0 { (-1.0, -1.0) } else { ((lod - band * 0.5) * ls, (lod + band * 0.5) * ls) };
            let (h1, h2) = (0.5 * ls, 2.0 * ls);
            let d = t.center.distance(cam);
            let cur = self.lod_state[i];
            let next = match cur {
                Lod::Mesh if d > a - h1 => Lod::Both,
                Lod::Impostor if d < b + h1 => Lod::Both,
                Lod::Both if d < a - h2 => Lod::Mesh,
                Lod::Both if d > b + h2 => Lod::Impostor,
                s => s,
            };
            if next != cur {
                self.lod_state[i] = next;
                changed = true;
                let flag = match next {
                    Lod::Impostor => 0u32,
                    Lod::Both => 1,
                    Lod::Mesh => 2,
                };
                self.state_writes.push((self.rec_of[i], flag));
            }
            if next != Lod::Impostor {
                active.push(pi);
            }
        }
        active.sort_unstable();
        self.near_count = active.len();
        self.active = active;
        let t_states = Instant::now();
        self.profile.cpu_lod_parts[0] += (t_states - t_begin).as_secs_f64() * 1000.0;
        if changed {
            let mesh_flags = vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE | vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE;
            // Band plants and alpha-tested cards: not opaque, so tri.rahit runs (hit record 2).
            let band_flags = vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE | vk::GeometryInstanceFlagsKHR::FORCE_NO_OPAQUE;
            self.insts.clear();
            self.inst_data.clear();
            self.insts.extend_from_slice(&self.static_insts);
            self.inst_data.extend_from_slice(&self.static_inst_data);
            const MAX_INSTANCES: usize = 8_000_000;
            for &pi in &self.active {
                let i = pi as usize;
                let t = &self.world.trees[i];
                let state = self.lod_state[i];
                let variant = &self.pack.trees[t.variant as usize];
                if self.insts.len() + variant.parts.len() > MAX_INSTANCES {
                    break;
                }
                let seed = ((t.seed as u32) & 0x1FFF) << 11;
                let center = t.center.to_array();
                let rec = self.rec_of[i];
                let tree = (rec & 0xFF_FFFF) | ((t.variant as u32) << INST_VARIANT_SHIFT);
                let band = if state == Lod::Both { INST_HAS_MESH } else { 0 }; // tri.rahit: dissolve
                for p in &variant.parts {
                    let alpha = self.catalog.mesh_alpha[p.mesh as usize];
                    let (flags, sbt) = if state == Lod::Mesh && !alpha { (mesh_flags, 0) } else { (band_flags, 2) };
                    let m = t.xform * p.xform;
                    self.insts.push(accel::make_instance(&m, p.mesh | seed, 1, sbt, flags, self.blas[p.mesh as usize].addr));
                    self.inst_data.push(InstInfo { center, data: tree | band });
                }
            }
            if self.insts.len() > self.tlas.capacity as usize {
                let cap = ((self.insts.len() as f32 * 1.5) as usize).min(MAX_INSTANCES + 500_000) as u32;
                println!("Growing TLAS capacity to {cap} instances");
                unsafe { self.gpu.device.device_wait_idle()? };
                let mut old = std::mem::replace(&mut self.tlas, Tlas::new(&self.gpu, cap)?);
                old.destroy(&self.gpu);
                let mut old_info = std::mem::replace(&mut self.inst_info, Self::create_inst_info(&self.gpu, cap)?);
                self.gpu.destroy_buffer(&mut old_info);
                self.write_set(self.main_set, self.tlas.accel.raw, None);
            }
            self.profile.cpu_lod_parts[1] += t_states.elapsed().as_secs_f64() * 1000.0;
            self.profile.rebuilds += 1;
            // The instance buffers are still read by the frame in flight: upload after its fence.
            self.upload_pending = true;
        }
        Ok(FrameStats { near_trees: self.near_count, instances: self.insts.len(), rebuilt: changed })
    }

    /// Copy the instance list built by update_instances into the TLAS instance buffer and the
    /// per-instance info buffer (call after the previous frame's fence).
    fn upload_instances(&mut self) {
        if !self.upload_pending {
            return;
        }
        let t = Instant::now();
        self.tlas.write(&self.insts);
        {
            let st: &mut [u32] = bytemuck::cast_slice_mut(self.plant_state.mapped());
            for (k, v) in self.state_writes.drain(..) {
                st[k as usize] = v;
            }
        }
        let bytes: &[u8] = bytemuck::cast_slice(&self.inst_data);
        self.inst_info.mapped()[..bytes.len()].copy_from_slice(bytes);
        self.tlas_dirty = true;
        self.upload_pending = false;
        self.profile.cpu_lod_parts[2] += t.elapsed().as_secs_f64() * 1000.0;
    }

    /// Globe mode: the camera for the next frame (ECEF position, rotation into the camera frame
    /// from globe::scene::frame_at, globe.rchit's globals: planet centre, sun, ...).
    pub fn set_globe_view(&mut self, ecef: glam::DVec3, rot: glam::DMat3, gg: crate::globe::scene::GlobeGlobals) {
        self.globe_view = Some((ecef, rot, gg));
    }

    pub fn extent(&self) -> (u32, u32) {
        (self.swap.extent.width, self.swap.extent.height)
    }

    /// Supersampled tracing resolution.
    pub fn render_extent(&self) -> (u32, u32) {
        (self.render_extent.width, self.render_extent.height)
    }

    /// Drop the TAA history (camera cut).
    pub fn reset_taa(&mut self) {
        self.taa_reset = true;
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
        // DLSS path: tonemap the denoised, upscaled output (already display resolution).
        let (dsrc, ddst) = (info(&self.tg.dlss_out), info(&self.out_img));
        let writes = [0u32, 1].map(|bnd| {
            vk::WriteDescriptorSet::default()
                .dst_set(self.dlss_resolve_set)
                .dst_binding(bnd)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(if bnd == 0 { &dsrc } else { &ddst })
        });
        unsafe { self.gpu.device.update_descriptor_sets(&writes, &[]) };
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
        for img in [
            &mut self.tg.cur,
            &mut self.tg.depth,
            &mut self.tg.viewz,
            &mut self.tg.motion,
            &mut self.tg.nrm,
            &mut self.tg.diff,
            &mut self.tg.spec,
            &mut self.tg.dlss_out,
        ] {
            self.gpu.destroy_image(img);
        }
        for img in self.tg.hist.iter_mut() {
            self.gpu.destroy_image(img);
        }
        self.gpu.destroy_buffer(&mut self.readback);
        self.render_extent = self.wanted_render_extent();
        let (o, tg, r) = Self::create_targets(&self.gpu, self.swap.extent, self.render_extent)?;
        self.out_img = o;
        self.tg = tg;
        self.readback = r;
        self.write_set(self.main_set, self.tlas.accel.raw, None);
        self.write_resolve_set();
        self.taa_reset = true;
        self.create_dlss_feature()?;
        Ok(())
    }

    /// Render resolution: the DLSS mode's input resolution when Ray Reconstruction is on,
    /// otherwise the supersampling scale.
    fn wanted_render_extent(&self) -> vk::Extent2D {
        match self.dlss_mode {
            Some(m) => scaled_extent(self.swap.extent, crate::dlss::MODES[m].1),
            None => scaled_extent(self.swap.extent, self.render_scale),
        }
    }

    /// (Re)create the NGX Ray Reconstruction feature for the current resolutions.
    fn create_dlss_feature(&mut self) -> Result<()> {
        let Some(m) = self.dlss_mode else { return Ok(()) };
        let (r, o) = (self.render_extent, self.swap.extent);
        let res = self.gpu.one_time(|cb| crate::dlss::create(cb, (r.width, r.height), (o.width, o.height), m))?;
        match res {
            Ok(()) => println!(
                "DLSS Ray Reconstruction: {} ({}x{} -> {}x{})",
                crate::dlss::MODES[m].0,
                r.width,
                r.height,
                o.width,
                o.height
            ),
            Err(e) => {
                println!("{e}; falling back to TAA");
                self.dlss_mode = None;
                self.recreate_targets()?;
            }
        }
        Ok(())
    }

    /// Turn DLSS Ray Reconstruction on (with a quality mode index into dlss::MODES) or off.
    pub fn set_dlss(&mut self, mode: Option<usize>) -> Result<()> {
        let mode = if self.dlss_ok { mode } else { None };
        if mode == self.dlss_mode {
            return Ok(());
        }
        self.dlss_mode = mode;
        self.recreate_targets()
    }

    pub fn dlss_mode(&self) -> Option<usize> {
        self.dlss_mode
    }

    pub fn dlss_available(&self) -> bool {
        self.dlss_ok
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
        // CPU work that does not touch GPU-read memory runs while the previous frame is still on
        // the GPU (LOD states, instance list); the uploads wait for its fence.
        let mut g = *globals;
        if lod <= 0.0 {
            g.lod_params[0] = 0.0; // all impostors: no dissolve band
        }
        g.lod_params[1] = 0.0; // single impostor tier (mips serve the far field)
        g.lod_params[3] = self.ground_layers as f32; // bit mask of the scanned ground layers loaded
        g.params[1] = self.has_tex as u32 as f32; // material textures available
        // Profiling switch (main.rgen): SI_PROF_SKIP=1 no sun shadow rays, 2 no AO rays, 3 neither.
        g.params[0] = std::env::var("SI_PROF_SKIP").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(0) as f32;
        let dlss = self.dlss_mode.is_some();
        if dlss {
            g.iparams[0] = 0; // no still-camera accumulation: DLSS keeps its own history (and needs the Halton jitter)
            g.lod_params[2] = 1.0; // main.rgen writes the G-buffer
            // Texture mip bias for upscaling: footprint of an output pixel, not a render pixel.
            g.cam_fwd[3] *= self.render_extent.height as f32 / self.swap.extent.height as f32;
        } else {
            g.lod_params[2] = 0.0;
        }
        let globals = &g;
        let t_lod = Instant::now();
        // Globe mode: the instances come from the streamed scene after the fence (below).
        let stats = if self.globe.is_some() { FrameStats::default() } else { self.update_instances(cam, lod, g.lod_params[0])? };
        self.profile.cpu_lod += t_lod.elapsed().as_secs_f64() * 1000.0;
        let d = self.gpu.device.clone();
        let t_wait = Instant::now();
        unsafe { d.wait_for_fences(&[self.fence], true, u64::MAX).context("waiting for previous frame")? };
        self.profile.cpu_wait += t_wait.elapsed().as_secs_f64() * 1000.0;
        if self.queries_pending {
            // Previous frame's GPU timestamps (its fence has signalled).
            let mut ts = [0u64; 5];
            if unsafe { d.get_query_pool_results(self.query_pool, 0, &mut ts, vk::QueryResultFlags::TYPE_64) }.is_ok() {
                let ms = self.gpu.timestamp_period as f64 * 1e-6;
                for k in 0..4 {
                    self.profile.gpu[k] += ts[k + 1].saturating_sub(ts[k]) as f64 * ms;
                }
                self.profile.frames += 1;
            }
            self.queries_pending = false;
        }
        let idx = match unsafe { self.gpu.swapchain_fn.acquire_next_image(self.swap.raw, u64::MAX, self.acquire_sem, vk::Fence::null()) } {
            Ok((i, _)) => i,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.need_resize = true;
                return Ok((FrameStats::default(), None));
            }
            Err(e) => return Err(e.into()),
        };
        unsafe { d.reset_fences(&[self.fence])? };

        if let (Some(scene), Some((ecef, rot, mut gg))) = (self.globe.as_mut(), self.globe_view) {
            // Terrain and tree tiles around the camera, placed in the camera frame every frame.
            let t0 = Instant::now();
            scene.update(&self.gpu, ecef)?;
            let t1 = Instant::now();
            self.insts.clear();
            scene.instances(ecef, rot, &mut self.insts);
            if std::env::var("SI_TREE_STATS").is_ok() && (self.lod_frame % 60 == 0 || (t1 - t0).as_secs_f64() > 0.005) {
                println!("globe cpu: update {:.2} ms, instances {:.2} ms", (t1 - t0).as_secs_f64() * 1e3, t1.elapsed().as_secs_f64() * 1e3);
            }
            self.lod_frame = self.lod_frame.wrapping_add(1);
            self.inst_data.clear();
            self.inst_data.resize(self.insts.len(), InstInfo { center: [0.0, -1e9, 0.0], data: 0 });
            // Mesh LOD as in the forest scene: trees near the camera get their mesh parts as
            // instances (dissolving against the impostor inside the band).
            let mut states = std::collections::HashMap::new();
            if lod > 0.0 {
                let band = g.lod_params[0];
                let reach = ((lod + band) * self.max_lod_scale + 4.0) as f64;
                let rotf = glam::Mat3::from_cols(rot.x_axis.as_vec3(), rot.y_axis.as_vec3(), rot.z_axis.as_vec3());
                let mesh_flags = vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE | vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE;
                let band_flags = vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE | vk::GeometryInstanceFlagsKHR::FORCE_NO_OPAQUE;
                let (insts, inst_data) = (&mut self.insts, &mut self.inst_data);
                let (catalog, pack, blas, infos) = (&self.catalog, &self.pack, &self.blas, &self.imp_infos);
                scene.near_trees(ecef, reach, |rec, tile_c, r| {
                    let f = |k: usize| f32::from_bits(r[k]);
                    let v = (r[12] & 0xFF) as usize;
                    let ls = catalog.variants[v].lod_scale;
                    let m = glam::Mat3::from_cols(glam::Vec3::new(f(0), f(4), f(8)), glam::Vec3::new(f(1), f(5), f(9)), glam::Vec3::new(f(2), f(6), f(10)));
                    // plant -> camera frame
                    let tile = glam::Affine3A::from_mat3_translation(rotf, (rot * (tile_c - ecef)).as_vec3());
                    let plant = tile * glam::Affine3A::from_mat3_translation(m, glam::Vec3::new(f(3), f(7), f(11)));
                    let c = infos[v].center_radius;
                    let center = plant.transform_point3(glam::Vec3::new(c[0], c[1], c[2]));
                    let d = center.length();
                    let (a, b) = ((lod - band * 0.5) * ls, (lod + band * 0.5) * ls);
                    let state = if d < a { 2u32 } else if d < b { 1 } else { return };
                    states.insert(rec, state);
                    let seed = ((r[12] >> 8) & 0x1FFF) << 11;
                    let data = (rec & 0xFF_FFFF) | ((v as u32) << INST_VARIANT_SHIFT) | if state == 1 { INST_HAS_MESH } else { 0 };
                    for p in &pack.trees[v].parts {
                        let alpha = catalog.mesh_alpha[p.mesh as usize];
                        let (flags, sbt) = if state == 2 && !alpha { (mesh_flags, 0) } else { (band_flags, 2) };
                        insts.push(accel::make_instance(&(plant * p.xform), p.mesh | seed, 1, sbt, flags, blas[p.mesh as usize].addr));
                        inst_data.push(InstInfo { center: center.to_array(), data });
                    }
                });
            }
            {
                // per-plant LOD state for impostor.rint (dissolve / skip where the mesh draws)
                let st: &mut [u32] = bytemuck::cast_slice_mut(scene.plant_state.mapped());
                for (k, _) in self.globe_states.iter() {
                    if !states.contains_key(k) {
                        st[*k as usize] = 0;
                    }
                }
                for (k, v) in &states {
                    st[*k as usize] = *v;
                }
            }
            self.globe_states = states;
            self.near_count = self.globe_states.len();
            let a = scene.index_addr();
            gg.addr = [a as u32, (a >> 32) as u32, 0, 0];
            gg.params[1] = self.lod_frame as f32;
            gg.params[2] = g.iparams[1] as f32;
            gg.params[3] = self.ground_layers as f32;
            gg.misc = [
                g.cam_fwd[3],
                crate::globe::forest::NEAR as f32,
                (crate::globe::forest::SPLIT * 0.5 * std::f64::consts::PI * crate::globe::tiles::R_EARTH
                    / (1u64 << crate::globe::forest::TREE_LEVEL) as f64) as f32,
                scene.range as f32,
            ];
            scene.write_globals(&gg);
            self.upload_pending = true;
        }
        self.upload_instances();
        let t_rec = Instant::now();
        self.ubo.mapped()[..std::mem::size_of::<Globals>()].copy_from_slice(bytemuck::bytes_of(globals));

        let (w, h) = self.extent();
        let (rw, rh) = self.render_extent();
        let cb = self.cb;
        let swap_img = self.swap.images[idx as usize];
        unsafe {
            d.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;
            d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            let qp = self.query_pool;
            d.cmd_reset_query_pool(cb, qp, 0, 5);
            d.cmd_write_timestamp2(cb, vk::PipelineStageFlags2::TOP_OF_PIPE, qp, 0);
            if self.tlas_dirty {
                self.tlas.cmd_build(&self.gpu, cb, self.insts.len() as u32);
                self.tlas_dirty = false;
            }
            d.cmd_write_timestamp2(cb, vk::PipelineStageFlags2::ALL_COMMANDS, qp, 1);
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
            d.cmd_write_timestamp2(cb, vk::PipelineStageFlags2::ALL_COMMANDS, qp, 2);
            if dlss {
                // DLSS Ray Reconstruction: denoise + upscale the noisy frame to display resolution.
                self.gpu.memory_barrier(
                    cb,
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR | vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::TRANSFER_READ,
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                    vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                );
                let si = |img: &Image| crate::dlss::SiImage { image: img.raw, view: img.view, format: img.format, width: img.width, height: img.height };
                let (jx, jy) = crate::dlss::jitter(globals.iparams[2]);
                let now = Instant::now();
                let e = crate::dlss::SiDlssEval {
                    color: si(&self.tg.cur),
                    output: si(&self.tg.dlss_out),
                    depth: si(&self.tg.viewz),
                    motion: si(&self.tg.motion),
                    normal_rough: si(&self.tg.nrm),
                    diff_albedo: si(&self.tg.diff),
                    spec_albedo: si(&self.tg.spec),
                    // offset of this frame's samples from the pixel centres, render pixels
                    jitter_x: (jx - 0.5) * self.dlss_jitter_sign.0,
                    jitter_y: (jy - 0.5) * self.dlss_jitter_sign.1,
                    render_w: rw,
                    render_h: rh,
                    reset: self.taa_reset as i32,
                    world_to_view: world_to_view(globals),
                    view_to_clip: view_to_clip(globals),
                    frame_time_ms: (now - self.last_frame).as_secs_f32() * 1000.0,
                };
                self.last_frame = now;
                self.taa_reset = false;
                if let Err(err) = crate::dlss::evaluate(cb, &e) {
                    eprintln!("{err}");
                }
                self.gpu.memory_barrier(
                    cb,
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                    vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::MEMORY_WRITE,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                );
                let push: [u32; 4] = [1.0f32.to_bits(), 1.0f32.to_bits(), globals.params[2].to_bits(), (globals.iparams[1] == 0) as u32];
                d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.resolve_pipe);
                d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, self.resolve_layout, 0, &[self.dlss_resolve_set], &[]);
                d.cmd_push_constants(cb, self.resolve_layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::cast_slice(&push));
                d.cmd_dispatch(cb, w.div_ceil(8), h.div_ceil(8), 1);
            } else {
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
            }
            d.cmd_write_timestamp2(cb, vk::PipelineStageFlags2::ALL_COMMANDS, qp, 3);
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
            d.cmd_write_timestamp2(cb, vk::PipelineStageFlags2::ALL_COMMANDS, qp, 4);
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
            self.queries_pending = true;
            self.profile.cpu_record += t_rec.elapsed().as_secs_f64() * 1000.0;
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
}

/// Per plant record (rt.glsl PlantRec): object->world 3x4 rows, variant | seed << 8.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PlantRecGpu {
    rows: [f32; 12],
    data: u32,
    pad: [u32; 3],
}

struct Clusters {
    rec_of: Vec<u32>,
    rec: Buffer,
    state: Buffer,
    blas: Vec<Accel>,
    aabbs: Buffer,
    insts: Vec<vk::AccelerationStructureInstanceKHR>,
    inst_data: Vec<InstInfo>,
}

/// Uniform grid of plant indices for the near-camera LOD query.
struct PlantGrid {
    half: f32,
    cell: f32,
    n: usize,
    cells: Vec<Vec<u32>>,
}

impl PlantGrid {
    fn new(world: &World) -> Self {
        let cell = 32.0;
        let half = world.trees.iter().map(|t| t.center.x.abs().max(t.center.z.abs())).fold(64.0f32, f32::max) + cell;
        let n = ((2.0 * half / cell).ceil() as usize).max(1);
        let mut cells = vec![Vec::new(); n * n];
        for (i, t) in world.trees.iter().enumerate() {
            let ix = (((t.center.x + half) / cell) as usize).min(n - 1);
            let iz = (((t.center.z + half) / cell) as usize).min(n - 1);
            cells[iz * n + ix].push(i as u32);
        }
        PlantGrid { half, cell, n, cells }
    }
}
