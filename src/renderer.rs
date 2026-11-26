//! Ray tracing renderer: pipelines, impostor baking, per-frame LOD selection and tracing.

use crate::accel::{self, Accel, Tlas};
use crate::asset::{MeshData, TreePack};
use crate::atlas::Res;
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
pub const MAX_VARIANTS: usize = 512; // PlantRec keeps the variant in 9 bits (rt.glsl plantVariant, forest.rs rec_data)
/// Texture sets (4 textures each) bound as a sampler array. Must match shared.glsl.
pub const MAX_SPECIES: usize = 64;
/// Ocean wave textures (globe/ocean.rs: A and B per cascade), binding 33.
const OCEAN_TEX: usize = crate::globe::ocean::CASCADES * 2;
/// Leaf transmission relative to albedo (calibrated so Black Alder keeps its earlier back-lit look).
const TRANS_GAIN: f32 = 8.0;
/// Ground material textures (binding 16). Must match shared.glsl.
pub const MAX_GROUND_TEX: usize = 81;

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
    /// x mesh<->impostor band width, y hi->lo impostor distance, z its band width, w the scanned
    /// ground layers loaded (a bit mask, set by the renderer)
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
    /// Globe mode: direction to the moon (camera frame), w = its angular radius (rad).
    pub moon: [f32; 4],
    /// Globe mode: direction to the sun (camera frame); w = pre-exposure (eye adaptation: 1 by
    /// day, up to ~1e5 by moonlight), applied before the 16-bit targets. 1 outside globe mode.
    pub sun_true: [f32; 4],
    /// Globe mode: rows of the camera frame -> Earth axes (ECEF) rotation (clouds.glsl).
    pub earth: [[f32; 4]; 3],
    /// Globe mode: x cloud cover (0..1), y clouds on (1) / off (0), z smooth tree thinning (1) / stepped (0).
    pub cloud: [f32; 4],
    /// Globe mode, the sea (globe/ocean.rs): x wind m/s, y residual slope variance, z largest wave
    /// height m, w chop; then wind direction (patch coordinates), whitecap coverage, ocean on;
    /// then the slope variance of each cascade.
    pub ocean: [[f32; 4]; 3],
    /// Forest scene: the biome's ground map (biome.rs Preset::ground_packed), one scan index per
    /// ground slot, 4 per word.
    pub biome: [u32; 4],
    /// Forest scene, the lawn (lawn.rs; set by the renderer): blade height (0 = no lawn), tile
    /// edge, views (azimuths, slopes); texels per view, gutter texels, atlas levels, atlas slot;
    /// slope of the steepest and the flattest views.
    pub lawn: [[f32; 4]; 4],
    /// `--test-flight` (the aircraft; main.rgen, aircraft.glsl, clouds.glsl): the renderer's frame
    /// -> the aircraft's model frame (rows, w the translation)
    pub ac_inv: [[f32; 4]; 3],
    /// a point on the aircraft this frame -> where it was last frame (rows): motion vectors
    pub ac_prev: [[f32; 4]; 3],
    /// x 1 = an aircraft is flying, y afterburner 0..1, z Mach number, w vapour cone 0..1
    pub ac_state: [f32; 4],
    /// x vapour over the wings 0..1, y points in the wake, z air density / sea level's, w seconds flown
    pub ac_state2: [f32; 4],
    /// xyz which way the air goes past the aircraft (model frame, unit), w engine power 0..1
    pub ac_vel: [f32; 4],
    /// where the aircraft has been, newest first (the renderer's frame); w age in seconds
    pub wake: [[f32; 4]; WAKE_POINTS],
    /// x condensation trail left there 0..1, y how much the air was stirred there 0..1
    pub wake_b: [[f32; 4]; WAKE_POINTS],
    /// Globe, near a city (globe/city.rs): x 1 = one traced bounce of light on the buildings and
    /// the ground (main.rgen), y extra coastal haze (atmosphere.glsl atmoHaze - 1), z 1 = the
    /// softer tone curve (resolve.comp), w unused
    pub city: [f32; 4],
}

/// Points of the aircraft's wake kept for the shaders (globals.glsl).
pub const WAKE_POINTS: usize = 32;

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
    data: u32, // tree index (24 bits) | variant << 24 (8 bits)
    flags: u32, // bit 0: the tree also has mesh instances (tri.rahit dissolves them); bits 1..4 dryness; from bit 5 the part
}

const INST_HAS_MESH: u32 = 1;
const INST_VARIANT_SHIFT: u32 = 24;
/// InstInfo.flags: the variant's ninth bit (rt.glsl instVariant).
const INST_VARIANT_HI: u32 = 31;
/// InstInfo.flags: which part of its plant the instance is (from this bit up; tri.rchit's per-leaf
/// colour variation).
const INST_PART_SHIFT: u32 = 5;

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
    /// texels per frame: across | up << 16
    pub res: u32,
    pub mips: u32,
    pub lod_scale: f32,
    /// xy: half the width and height of a frame (m)
    pub frame: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MeshInfoGpu {
    addr: [u64; 5], // pos, nrm, idx, mat, uv
    tex_set: u32,
    flags: u32, // MESH_ALPHA_TESTED; >> 8: a grid mesh's vertices per side
}

/// Mesh flag: the leaf texture's alpha is an opacity mask (alpha-tested in tri.rahit).
const MESH_ALPHA_TESTED: u32 = 1;
/// Mesh flag: the lawn tile's blades (tri.rchit colours them itself).
const MESH_LAWN_BLADE: u32 = 2;
/// Mesh flag: a procedural building (city.rs): its material bytes are facade materials.
const MESH_BUILDING: u32 = 4;
/// Hit records of the buildings' meshes (city.rs): opaque, and dissolving in the LOD band.
const SBT_BUILDING: u32 = 5;
const SBT_BUILDING_BAND: u32 = 6;

/// Building impostors may use frames up to this many texels: a 250 m tower at the plants' 448
/// would have 0.6 m texels.
const BUILDING_MAX_RES: u32 = 768;

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
    res: Res,
    mips: u32,
}

/// GPU bytes of an atlas of `grid` x `grid` frames of `res` texels with all its levels.
fn atlas_bytes(grid: u32, res: Res) -> u64 {
    let s = (grid * res[0]) as u64 * (grid * res[1]) as u64;
    // BC3/BC5: 1 byte per texel; vis at quarter area; mips +1/3
    (s * 3 + s / 4) * 4 / 3
}

impl BcAtlas {
    fn bytes(&self) -> u64 {
        atlas_bytes(self.grid, self.res)
    }
    fn destroy(&mut self, gpu: &Gpu) {
        for img in [&mut self.albedo, &mut self.normal, &mut self.depth, &mut self.vis] {
            gpu.destroy_image(img);
        }
    }
}

fn create_bc_atlas(gpu: &Gpu, name: &str, grid: u32, res: Res, mips: u32) -> Result<BcAtlas> {
    let usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST;
    let (w, h) = (grid * res[0], grid * res[1]);
    Ok(BcAtlas {
        albedo: gpu.create_image(w, h, mips, vk::Format::BC3_UNORM_BLOCK, usage, &format!("{name} albedo"))?,
        normal: gpu.create_image(w, h, mips, vk::Format::BC5_UNORM_BLOCK, usage, &format!("{name} normal"))?,
        depth: gpu.create_image(w, h, mips, vk::Format::BC5_UNORM_BLOCK, usage, &format!("{name} depth"))?,
        vis: gpu.create_image(w / 2, h / 2, mips - 1, vk::Format::BC3_UNORM_BLOCK, usage, &format!("{name} visibility"))?,
        grid,
        res,
        mips,
    })
}

/// A plant variant's impostor atlases (atlas.rs): the base one always on the GPU, the full one
/// while the variant grows near the camera; the bake itself lives in the cache file.
struct VariantAtlas {
    name: String,
    grid: u32,
    /// the full atlas's frame resolution and levels
    res: Res,
    mips: u32,
    /// the square frame of the bounding sphere the frames are cropped from: its texels per side
    /// and the first texel of the crop (bake.rgen)
    square: u32,
    crop: [u32; 2],
    /// levels the base atlas drops
    skip: u32,
    key: u64,
    path: std::path::PathBuf,
    /// (the globe; the forest scene keeps the full atlases of its own variants and nothing else)
    base: Option<BcAtlas>,
    full: Option<BcAtlas>,
    loading: bool,
    /// the upload buffer range its full atlas is being read into
    upload: Option<u64>,
    /// update_atlases check that last wanted the full atlas
    wanted: u64,
}

impl VariantAtlas {
    fn shown(&self) -> Option<&BcAtlas> {
        self.full.as_ref().or(self.base.as_ref())
    }
    fn full_bytes(&self) -> u64 {
        atlas_bytes(self.grid, self.res)
    }
    /// The texel size of the full atlas (m), from the bounding sphere's radius.
    fn texel(&self, radius: f32) -> f32 {
        2.0 * radius / self.square as f32
    }
}

/// Variants growing within this distance (m) of the camera are candidates for their full atlas.
const ATLAS_RADIUS: f64 = 900.0;
/// Upload buffer for full atlases in flight (the largest stream is ~125 MB).
const ATLAS_STAGING: u64 = 256 << 20;

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
    size: [u32; 2],
    crop: [u32; 2],
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
    /// short grass as a parallax layer over the grass ground (lawn.rs; forest scene)
    pub lawn: bool,
}

/// The lawn on the GPU: its atlas is one more slot after the plant variants', its tile mesh one
/// more mesh after the terrain (only the bake traces it).
struct LawnGpu {
    slot: usize,
    mesh: u32,
    tris: u32,
}

impl ImpostorSettings {
    pub fn res_for(&self, radius: f32, lod_scale: f32) -> u32 {
        // Small plants switch to impostors closer, but only down to 0.35x the tree texel density.
        let r = (2.0 * radius / (self.texel * lod_scale.max(0.35))).round() as u32;
        (r.div_ceil(32) * 32).clamp(32, self.max_res.max(32))
    }
}

/// The frame of a plant in texels (across, up) and where it starts in the square frame of
/// `square` texels that shows the plant's whole bounding sphere: the square cropped, about its
/// centre, to what the plant can cover in any view. `across`, `up`: how far the plant reaches
/// from the sphere's centre sideways and at all (asset.rs TreeAsset::reach), in texels. A
/// margin of one texel of the coarsest level stays empty around the plant (what a filtered
/// lookup at the plant's edge reads). The frame is a whole number of BC blocks of the coarsest
/// level and at least 8 of its texels, as the square is, and the crop starts on a whole block of
/// every level: the blocks that remain, at every level, are the square bake's.
fn frame_res(square: u32, mips: u32, across: f32, up: f32) -> (Res, [u32; 2]) {
    let step = 4 << (mips - 1);
    let margin = (1 << (mips - 1)) as f32 + 1.0;
    let side = |reach: f32| {
        let want = (2.0 * (reach + margin)).ceil() as u32;
        let mut s = (want.div_ceil(step).max(2) * step).min(square);
        // (the crop, half the difference, a whole step too)
        if !(square - s).is_multiple_of(2 * step) {
            s += step;
        }
        s
    };
    let res = [side(across), side(up)];
    (res, [(square - res[0]) / 2, (square - res[1]) / 2])
}

/// Mip count for a frame resolution: every encoded level must stay a multiple of 4 (BC blocks)
/// and frames keep at least 8 texels.
fn atlas_mips(res: u32) -> u32 {
    let mut m = 1;
    while (res >> m) >= 8 && (res >> m).is_multiple_of(4) {
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

/// What the scene can show. On the globe any species may come into view; the forest scene (a
/// biome, a real site, the gallery) only ever shows the variants that grow in it, so only their
/// meshes, textures, BLASes and atlases go to the GPU.
struct Used {
    variant: Vec<bool>,
    mesh: Vec<bool>,
    species: Vec<bool>,
    /// scanned ground surfaces (catalog.rs GROUND_LAYERS)
    ground: Vec<bool>,
}

impl Used {
    fn new(pack: &TreePack, catalog: &Catalog, world: &World, globe: bool) -> Self {
        let mut u = Used {
            variant: vec![globe; pack.trees.len()],
            mesh: vec![globe; pack.meshes.len()],
            species: vec![globe; catalog.species.len()],
            ground: vec![globe; catalog.ground.len()],
        };
        if globe {
            return u;
        }
        for t in &world.trees {
            u.variant[t.variant as usize] = true;
        }
        for (v, t) in pack.trees.iter().enumerate().filter(|(v, _)| u.variant[*v]) {
            if let Some(s) = catalog.variants.get(v).and_then(|vi| u.species.get_mut(vi.species)) {
                *s = true;
            }
            for p in &t.parts {
                u.mesh[p.mesh as usize] = true;
                if let Some(s) = catalog.mesh_species.get(p.mesh as usize).and_then(|&s| u.species.get_mut(s as usize)) {
                    *s = true;
                }
            }
        }
        // tri.rchit groundMaterial: a slot shows the biome's scan, its own one where that scan is
        // not downloaded, or (river slots) scans 0 / 1; scan 0 also sets the texel size
        let has = |i: usize| catalog.ground.get(i).is_some_and(|g| g.is_some());
        for (slot, &scan) in crate::biome::current().ground.iter().enumerate() {
            let src = if has(scan as usize) { scan as usize } else { slot };
            for i in [src, 0, 1] {
                if let Some(g) = u.ground.get_mut(i) {
                    *g = true;
                }
            }
        }
        u
    }
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
    /// globe mode, CPU ms: terrain uploads, tree uploads, tree tile list, terrain select +
    /// instances, mesh LOD + plant state (sums and worst single frame)
    pub globe: [f64; 5],
    pub globe_max: [f64; 5],
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
        ) + &if self.globe_max.iter().any(|&m| m > 0.0) {
            let n = self.frames.max(1) as f64;
            let names = ["terrain upload", "tree upload", "tree tiles", "select+instances", "mesh LOD"];
            let parts: Vec<String> = (0..5).map(|i| format!("{} {:.2} (max {:.1})", names[i], self.globe[i] / n, self.globe_max[i])).collect();
            format!("
globe cpu ms: {}", parts.join(" | "))
        } else {
            String::new()
        }
    }
}

#[derive(Default, Clone, Copy)]
pub struct FrameStats {
    pub near_trees: usize,
    pub instances: usize,
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
    /// volumetric cloud noise (cloudnoise.rs): 3D shape and detail textures, bindings 34 / 35
    cloud_noise: [Image; 2],
    ground_layers: u32,
    mat_sampler: vk::Sampler,
    has_tex: bool,
    lawn: Option<LawnGpu>,
    /// the lawn is drawn (key toggle; nothing happens without one)
    pub lawn_on: bool,
    sharp_img: Image,
    sharpen_layout: vk::PipelineLayout,
    sharpen_pipe: vk::Pipeline,
    sharpen_set: vk::DescriptorSet,
    /// Contrast adaptive sharpening strength (0 = off .. 1).
    pub sharpness: f32,
    /// output brightness for fades (0 black .. 1), applied in the resolve after DLSS / TAA
    pub fade: f32,
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
    /// per mesh (a null one for meshes the scene never shows)
    blas: Vec<Accel>,
    imp_info: Buffer,
    imp_infos: Vec<ImpostorInfo>,
    /// per variant (atlas.rs): on the globe the base atlas always and the full one near the
    /// camera; in the forest scene the full atlas of every variant that grows in it
    atlases: Vec<VariantAtlas>,
    /// bound in the slots of variants without an atlas (never sampled)
    atlas_blank: BcAtlas,
    /// the variants the scene can show (Used)
    used_variant: Vec<bool>,
    atlas_loader: crate::atlas::Loader,
    /// GPU memory the full atlases may take (SI_IMPOSTOR_MB)
    atlas_budget: u64,
    atlas_frame: u64,
    /// persistently mapped upload buffer the loader reads full atlases into, its free ranges,
    /// the uploads to record into this frame (variant, offset) and the ranges to free once the
    /// frame that copied them is done (offset, length)
    atlas_staging: Option<Buffer>,
    atlas_ranges: crate::atlas::Ranges,
    atlas_uploads: Vec<(usize, u64)>,
    atlas_retire: Vec<(u64, u64)>,
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
    /// TLAS instances that never change: terrain and the plant clusters.
    static_insts: Vec<vk::AccelerationStructureInstanceKHR>,
    /// cells with draw-distance plants, their complete BLASes to build in this frame, and the
    /// ones to free once the frame in flight is done
    near_cells: Vec<NearCell>,
    near_builds: Vec<usize>,
    near_retire: Vec<NearBlas>,
    near_scratch: Buffer,
    tree_of: Vec<u32>,
    insts: Vec<vk::AccelerationStructureInstanceKHR>,
    inst_data: Vec<InstInfo>,
    inst_info: Buffer,
    tlas_dirty: bool,
    /// insts / inst_data changed and must be copied to the GPU buffers (upload_instances).
    upload_pending: bool,
    first_update: bool,
    /// The lawn's real blades around the camera (lawn.rs near_radius): the tile's instances, and
    /// where the camera was when they were placed (they are placed again once it has moved).
    grass_insts: Vec<vk::AccelerationStructureInstanceKHR>,
    grass_at: Vec3,
    grass_radius: f32,
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
    /// `--test-flight`: the aircraft's skin maps and materials (placeholders without one), its
    /// parts (meshes ac_first..), and where it is this frame
    ac_tex: Vec<Image>,
    ac_mats: Buffer,
    ac_parts: Vec<crate::aircraft::Part>,
    ac_first: usize,
    pub ac_pose: Option<crate::aircraft::Pose>,
    /// Globe mode: plant records currently drawn as meshes (1 both / 2 mesh only).
    globe_states: std::collections::HashMap<u32, u32>,
    /// Globe mode: trees in the TLAS last frame (window title).
    pub globe_trees: usize,
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

impl MeshGpu {
    /// A mesh the scene never shows: nothing on the GPU.
    fn none() -> Self {
        MeshGpu { pos: Buffer::none(), nrm: Buffer::none(), idx: Buffer::none(), mat: Buffer::none(), uv: Buffer::none() }
    }
}

fn upload_mesh(gpu: &Gpu, m: &MeshData) -> Result<MeshGpu> {
    let geo = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;
    Ok(MeshGpu {
        pos: gpu.upload_buffer(bytemuck::cast_slice(&m.positions), geo, &m.name)?,
        nrm: gpu.upload_buffer(bytemuck::cast_slice(&m.normals), vk::BufferUsageFlags::STORAGE_BUFFER, &m.name)?,
        idx: gpu.upload_buffer(bytemuck::cast_slice(&m.indices), geo, &m.name)?,
        // (a grid mesh's triangles are all one material: rt.glsl meshTri, tri.rchit)
        mat: if m.grid == 0 { gpu.upload_buffer(&m.materials, vk::BufferUsageFlags::STORAGE_BUFFER, &m.name)? } else { Buffer::none() },
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
        gpu.shader_module(spv!("aircraft.rchit"))?,
        gpu.shader_module(spv!("aircraft.rahit"))?,
        gpu.shader_module(spv!("building.rchit"))?,
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
    // 2 triangles of trees inside the mesh <-> impostor band (any-hit dissolve), 3 globe terrain,
    // 4 the aircraft (its glass is seen through by the any-hit), 5 buildings (building.rchit),
    // 6 buildings inside the mesh <-> impostor band (any-hit dissolve).
    let groups = [
        group(vk::RayTracingShaderGroupTypeKHR::GENERAL, 0, u, u, u),
        group(vk::RayTracingShaderGroupTypeKHR::GENERAL, 1, u, u, u),
        group(vk::RayTracingShaderGroupTypeKHR::GENERAL, 2, u, u, u),
        group(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP, u, 3, u, u),
        group(vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP, u, 5, u, 4),
        group(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP, u, 3, 6, u),
        group(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP, u, 7, u, u),
        group(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP, u, 8, 9, u),
        group(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP, u, 10, u, u),
        group(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP, u, 10, 6, u),
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
    let total = off_hit + 7 * stride;
    let handles = unsafe { gpu.rt_fn.get_ray_tracing_shader_group_handles(pipe, 0, 10, 10 * hs as usize)? };
    let mut sbt = gpu.create_buffer(total + ba, vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR, MemoryLocation::CpuToGpu, "sbt")?;
    let base = align_up(sbt.addr, ba);
    let shift = (base - sbt.addr) as usize;
    {
        let m = sbt.mapped();
        for (group, off) in
            [(0usize, 0u64), (1, off_miss), (2, off_miss + stride), (3, off_hit), (4, off_hit + stride), (5, off_hit + 2 * stride), (6, off_hit + 3 * stride), (7, off_hit + 4 * stride), (8, off_hit + 5 * stride), (9, off_hit + 6 * stride)]
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
        hit: region(off_hit, 7 * stride),
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
        mut world: World,
        imp: ImpostorSettings,
        vsync: bool,
        validation: bool,
        render_scale: f32,
        dlss_mode: Option<usize>,
        globe: Option<std::sync::Arc<crate::globe::dem::Terrain>>,
        aircraft: Option<crate::aircraft::Model>,
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
        let on_globe = globe.is_some();
        let mut used = Used::new(&pack, &catalog, &world, on_globe);
        // the lawn's square of blades (forest scene): a mesh after the terrain
        let lawn_tile = (imp.lawn && !on_globe).then(crate::lawn::tile);
        if !on_globe {
            let n = |v: &[bool]| v.iter().filter(|&&b| b).count();
            println!(
                "Scene: {} of {} variants, {} of {} meshes, {} of {} species, {} of {} ground scans",
                n(&used.variant),
                used.variant.len(),
                n(&used.mesh),
                used.mesh.len(),
                n(&used.species),
                used.species.len(),
                n(&used.ground),
                used.ground.len()
            );
        }
        let (mat_tex, leaf_scale, has_tex) = crate::textures::load_species(&gpu, &catalog, &used.species)?;
        let cloud_noise = {
            let (shape, detail) = crate::cloudnoise::load_or_build();
            [
                gpu.upload_volume(crate::cloudnoise::SHAPE as u32, &shape, "cloud shape noise")?,
                gpu.upload_volume(crate::cloudnoise::DETAIL as u32, &detail, "cloud detail noise")?,
            ]
        };
        let mut ground_tex = crate::textures::load_ground(&gpu, &catalog.ground, &used.ground)?;
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
            // the slot before the masks: the grass map of `--lawn auto` (World::build_grass_map)
            if imp.lawn && !on_globe && crate::lawn::auto_short() > 0.0 {
                world.build_grass_map(&pack.trees, &catalog);
                let rgba: Vec<u8> = world.grass.iter().flat_map(|&v| [v, v, v, 255]).collect();
                ground_tex[MAX_GROUND_TEX - 3] = crate::textures::upload_with_mips(&gpu, MASK_RES as u32, MASK_RES as u32, &rgba, vk::Format::R8G8B8A8_UNORM, "grass map")?;
            }
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
        // (the terrain, after the plant meshes, is always shown; so is the lawn tile after it)
        let shown = |i: usize| used.mesh.get(i).copied().unwrap_or(true);
        // (and the aircraft's parts last: --test-flight)
        let no_meshes: Vec<MeshData> = Vec::new();
        let ac_meshes = aircraft.as_ref().map(|a| &a.meshes).unwrap_or(&no_meshes);
        let ac_first = pack.meshes.len() + 1 + lawn_tile.iter().count();
        let all_meshes: Vec<&MeshData> = pack.meshes.iter().chain(std::iter::once(&world.terrain)).chain(lawn_tile.iter()).chain(ac_meshes.iter()).collect();
        assert!(all_meshes.len() < 2048, "mesh ids are 11 bits in the instance custom index");
        let lawn_mesh = lawn_tile.as_ref().map(|_| pack.meshes.len() + 1);
        for (i, m) in all_meshes.iter().enumerate() {
            meshes.push(if shown(i) { upload_mesh(&gpu, m)? } else { MeshGpu::none() });
        }
        let infos: Vec<MeshInfoGpu> = meshes
            .iter()
            .enumerate()
            .map(|(i, m)| {
                // a grid mesh (the terrain): its index buffer is only BLAS build input
                let grid = all_meshes[i].grid;
                let alpha = if catalog.mesh_alpha.get(i).copied().unwrap_or(false) { MESH_ALPHA_TESTED } else { 0 };
                let building = catalog.mesh_species.get(i).and_then(|&s| catalog.species.get(s as usize)).is_some_and(|s| s.kind == crate::catalog::Kind::Building);
                let alpha = alpha | if building && i < pack.meshes.len() { MESH_BUILDING } else { 0 };
                MeshInfoGpu {
                    addr: [m.pos.addr, m.nrm.addr, if grid == 0 { m.idx.addr } else { 0 }, m.mat.addr, m.uv.addr],
                    tex_set: catalog.mesh_species.get(i).copied().unwrap_or(0),
                    flags: alpha | if Some(i) == lawn_mesh { MESH_LAWN_BLADE } else { 0 } | grid << 8,
                }
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
        let built: Vec<usize> = (0..meshes.len()).filter(|&i| shown(i)).collect();
        let geoms: Vec<_> = built
            .iter()
            .map(|&i| (accel::triangles_geometry(meshes[i].pos.addr, all_meshes[i].vertex_count(), meshes[i].idx.addr), all_meshes[i].tri_count()))
            .collect();
        let mut blas: Vec<Accel> = (0..meshes.len()).map(|_| Accel::none()).collect();
        for (&i, a) in built.iter().zip(accel::build_blases(&gpu, &geoms)?) {
            blas[i] = a;
            if all_meshes[i].grid != 0 {
                gpu.destroy_buffer(&mut meshes[i].idx); // the shaders work a grid's triangles out
                meshes[i].idx = Buffer::none();
            }
        }
        println!("BLAS: {} meshes built in {:.2}s", built.len(), t0.elapsed().as_secs_f32());

        // ---- impostor proxies and atlases: one per plant variant, resolution from its size
        let nv = pack.trees.len();
        assert!(nv <= MAX_VARIANTS, "too many plant variants ({nv} > {MAX_VARIANTS})");
        let mut imp_infos = Vec::new();
        let mut atlases = Vec::new();
        let mut full_bytes = 0u64;
        let mut raw_bytes = 0u64;
        // GPU memory for the full atlases of the variants near the camera (SI_IMPOSTOR_MB, default
        // 25% of the largest device-local heap); every variant keeps its base atlas besides
        let atlas_budget = {
            let heap = unsafe { gpu.instance.get_physical_device_memory_properties(gpu.pdev) };
            let vram = heap.memory_heaps[..heap.memory_heap_count as usize]
                .iter()
                .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
                .map(|h| h.size)
                .max()
                .unwrap_or(8 << 30);
            std::env::var("SI_IMPOSTOR_MB").ok().and_then(|v| v.parse::<u64>().ok()).map(|mb| mb << 20).unwrap_or(vram / 4)
        };
        // (full atlases stream in on the globe only)
        let atlas_staging = match on_globe {
            true => Some(gpu.create_buffer(ATLAS_STAGING, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu, "atlas uploads")?),
            false => None,
        };
        // everything that changes a bake (atlas.rs cache key), shared by all variants
        let bake_key = crate::atlas::Key::new()
            .bytes(spv!("bake.rgen"))
            .bytes(spv!("tri.rchit"))
            .bytes(spv!("tri.rahit"))
            .bytes(spv!("mip.comp"))
            .bytes(spv!("bcenc.comp"))
            .u64(imp.grid as u64)
            .u64(imp.ss as u64)
            .u64(imp.ao_rays as u64)
            .u64(imp.ao_range.to_bits() as u64)
            .u64(imp.vis_rays as u64)
            .u64(has_tex as u64)
            .get();
        // How far every plant reaches from the centre of its bounding box (its frame: frame_res),
        // on every core: the outer parts' vertices are walked. Every variant, also those this
        // scene does not show: the frame is part of the cache key.
        let reach: Vec<(f32, f32)> = {
            let bounds: Vec<(Vec3, Vec3)> = pack.meshes.iter().map(|m| m.bounds()).collect();
            let next = std::sync::atomic::AtomicUsize::new(0);
            let out: Vec<std::sync::Mutex<(f32, f32)>> = pack.trees.iter().map(|_| std::sync::Mutex::new((0.0, 0.0))).collect();
            std::thread::scope(|s| {
                for _ in 0..std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4) {
                    s.spawn(|| loop {
                        let v = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(t) = pack.trees.get(v) else { break };
                        *out[v].lock().unwrap() = t.reach(&pack.meshes, &bounds);
                    });
                }
            });
            out.into_iter().map(|m| m.into_inner().unwrap()).collect()
        };
        let mut square_bytes = 0u64;
        for (v, t) in pack.trees.iter().enumerate() {
            let vi = &catalog.variants[v];
            let sp = &species[vi.species.min(species.len() - 1)];
            let pad = Vec3::splat(0.05);
            let (lo, hi) = (t.bmin - pad, t.bmax + pad);
            let c = (t.bmin + t.bmax) * 0.5;
            let r = (t.bmax - t.bmin).length() * 0.5;
            // The square frame of the bounding sphere sets the texel size; the frame that is baked
            // and kept is that square cropped to the plant.
            let building = vi.kind == crate::catalog::Kind::Building;
            let square = if building { ImpostorSettings { max_res: BUILDING_MAX_RES.max(imp.max_res), ..imp }.res_for(r, vi.lod_scale) } else { imp.res_for(r / vi.texel_scale, vi.lod_scale) };
            let mips = atlas_mips(square);
            let texels = square as f32 / (2.0 * r);
            let (res, crop) = frame_res(square, mips, reach[v].0 * texels, reach[v].1 * texels);
            square_bytes += atlas_bytes(imp.grid, [square, square]);
            imp_infos.push(ImpostorInfo {
                center_radius: [c.x, c.y, c.z, r],
                // w: 1 = foliage only (grass, ferns: no bark, so never classify texels as bark by
                // colour); 2 = solid (rocks, logs, cacti: never foliage, however green)
                bmin: [
                    lo.x,
                    lo.y,
                    lo.z,
                    match vi.kind {
                        crate::catalog::Kind::Grass | crate::catalog::Kind::Plant => 1.0,
                        crate::catalog::Kind::Rock => 2.0,
                        // 3 = a building (impostor.rchit: facade, glass; impostor.rint: no thinning)
                        crate::catalog::Kind::Building => 3.0,
                        _ => 0.0,
                    },
                ],
                // w: 1 = a car (impostor.rchit: its paint by the record's seed)
                bmax: [hi.x, hi.y, hi.z, if t.name.starts_with("Building_Car_") { 1.0 } else { 0.0 }],
                // mean transmission (x albedo in impostor.rchit); w = draw distance
                trans: [0.33 * sp.trans_tint[0], 0.33 * sp.trans_tint[1], 0.33 * sp.trans_tint[2], vi.cull],
                grid: imp.grid,
                res: 0,
                mips,
                lod_scale: vi.lod_scale,
                frame: [r * res[0] as f32 / square as f32, r * res[1] as f32 / square as f32, 0.0, 0.0],
            });
            let skip = crate::atlas::skip_for(mips);
            let base_res = [res[0] >> skip, res[1] >> skip];
            let base = match on_globe {
                true => Some(create_bc_atlas(&gpu, &t.name, imp.grid, base_res, mips - skip)?),
                false => None,
            };
            // the shaders start on the base atlas (update_atlases swaps the full one in)
            let info = imp_infos.last_mut().unwrap();
            (info.res, info.mips) = (base_res[0] | base_res[1] << 16, mips - skip);
            let mesh_key = t.parts.iter().fold(crate::atlas::Key::new().u64(bake_key).bytes(t.name.as_bytes()).u64(t.parts.len() as u64), |k, p| {
                let m = &pack.meshes[p.mesh as usize];
                k.u64(p.mesh as u64).u64(m.vertex_count() as u64).u64(m.tri_count() as u64).bytes(bytemuck::cast_slice(&p.xform.to_cols_array()))
            });
            // (building variants: their hit shader too; the plants' keys stay as they were)
            let mesh_key = if building { mesh_key.bytes(spv!("building.rchit")) } else { mesh_key };
            let key = mesh_key.u64(square as u64).u64(res[0] as u64).u64(res[1] as u64).u64(mips as u64).u64(vi.texel_scale.to_bits() as u64).get();
            let path = crate::atlas::path_for(&t.name, key);
            let full = VariantAtlas { name: t.name.clone(), grid: imp.grid, res, mips, square, crop, skip, key, path, base, full: None, loading: false, upload: None, wanted: 0 };
            full_bytes += full.full_bytes();
            raw_bytes += (imp.grid * res[0]) as u64 * (imp.grid * res[1]) as u64 * 16 * 4 / 3; // the old rgba8 + rgba16f + rgba8 layout
            atlases.push(full);
        }
        // The lawn: one more atlas, baked from its tile (bake_impostors) and cached like the plants'.
        let lawn = match (&lawn_tile, lawn_mesh) {
            (Some(tile), Some(mesh)) => {
                use crate::lawn::{AZIMUTHS, GUTTER, RES, SLOPES, SLOPE_MAX, SLOPE_MIN};
                #[allow(non_snake_case)]
                let TILE = crate::lawn::tile_size();
                let mips = atlas_mips(RES);
                let key = crate::atlas::Key::new()
                    .u64(bake_key)
                    .bytes(b"lawn 4")
                    .bytes(bytemuck::cast_slice(&[AZIMUTHS, SLOPES, RES, GUTTER, crate::lawn::height().to_bits(), TILE.to_bits(), SLOPE_MIN.to_bits(), SLOPE_MAX.to_bits()]))
                    .bytes(bytemuck::cast_slice(&tile.positions))
                    .get();
                let path = crate::atlas::path_for("Lawn", key);
                // (to the atlas code: a 4 x 4 grid of "frames", each a block of views)
                let res = [AZIMUTHS / 4 * RES, SLOPES / 4 * RES];
                let full = VariantAtlas { name: "Lawn".into(), grid: 4, res, mips, square: RES, crop: [0, 0], skip: 0, key, path, base: None, full: None, loading: false, upload: None, wanted: 0 };
                // (a place in the impostor table, so every atlas has one; the shaders read the
                // lawn's numbers from the globals)
                imp_infos.push(ImpostorInfo { center_radius: [0.0; 4], bmin: [0.0; 4], bmax: [0.0; 4], trans: [0.0; 4], grid: 4, res: res[0] | res[1] << 16, mips, lod_scale: 1.0, frame: [0.0; 4] });
                full_bytes += full.full_bytes();
                atlases.push(full);
                used.variant.push(true);
                Some(LawnGpu { slot: atlases.len() - 1, mesh: mesh as u32, tris: tile.tri_count() })
            }
            _ => None,
        };
        assert!(atlases.len() <= MAX_VARIANTS, "no atlas slot left for the lawn");
        println!(
            "Impostor atlases: {} variants, {:.2} GB BC-compressed at full resolution (frames cropped to the plants; {:.2} GB as squares, {:.2} GB uncompressed){}",
            atlases.len(),
            full_bytes as f64 / 1e9,
            square_bytes as f64 / 1e9,
            raw_bytes as f64 / 1e9,
            if on_globe { format!("; full atlases near the camera within {:.2} GB", atlas_budget as f64 / 1e9) } else { String::new() }
        );
        let atlas_blank = create_bc_atlas(&gpu, "blank", 1, [8, 8], 2)?;
        // host-visible: update_atlases rewrites a variant's resolution when its atlas changes
        let info_bytes: &[u8] = bytemuck::cast_slice(&imp_infos);
        let mut imp_info = gpu.create_buffer(info_bytes.len() as u64, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "impostor info")?;
        imp_info.mapped()[..info_bytes.len()].copy_from_slice(info_bytes);
        gpu.one_time(|cb| {
            for a in atlases.iter().filter_map(|a| a.base.as_ref()).chain(std::iter::once(&atlas_blank)) {
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

        // the aircraft's skin and materials (bindings 36, 37), or placeholders
        let (ac_tex, ac_mats) = match &aircraft {
            Some(a) => {
                let t = Instant::now();
                let tex = crate::aircraft::load_textures(&gpu, &a.dir)?;
                println!("Aircraft: {} parts, {} triangles, skin in {:.2}s", a.parts.len(), a.meshes.iter().map(|m| m.tri_count()).sum::<u32>(), t.elapsed().as_secs_f32());
                (tex, gpu.upload_buffer(bytemuck::cast_slice(&a.materials), vk::BufferUsageFlags::STORAGE_BUFFER, "aircraft materials")?)
            }
            None => (
                vec![crate::textures::upload_with_mips(&gpu, 1, 1, &[128, 128, 128, 255], vk::Format::R8G8B8A8_UNORM, "aircraft placeholder")?],
                gpu.upload_buffer(&[0u8; 48], vk::BufferUsageFlags::STORAGE_BUFFER, "aircraft materials")?,
            ),
        };
        let ac_parts: Vec<crate::aircraft::Part> = aircraft.map(|a| a.parts).unwrap_or_default();
        const _: () = assert!(std::mem::size_of::<Globals>() <= 2048);
        let ubo = gpu.create_buffer(2048, vk::BufferUsageFlags::UNIFORM_BUFFER, MemoryLocation::CpuToGpu, "globals")?;
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
            b(13, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, (crate::textures::SPECIES_SLOTS * MAX_SPECIES) as u32, rt_all),
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
            b(33, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, OCEAN_TEX as u32, rt_all),
            b(34, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 1, rt_all),
            b(35, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 1, rt_all),
            // the aircraft (aircraft.rchit): its skin's three maps, its materials
            b(36, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 3, rt_all),
            b(37, vk::DescriptorType::STORAGE_BUFFER, 1, rt_all),
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
                descriptor_count: 2 * (4 * nvar + (crate::textures::SPECIES_SLOTS * MAX_SPECIES) as u32 + MAX_GROUND_TEX as u32 + 1 + OCEAN_TEX as u32 + 2 + 3),
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

        // (the temperate clearing needs ~270k instances; a denser spot grows it: update_instances)
        let est = 400_000u32;
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
            cloud_noise,
            ground_layers,
            mat_sampler,
            has_tex,
            lawn,
            lawn_on: true,
            sharp_img,
            sharpen_layout,
            sharpen_pipe,
            sharpen_set,
            sharpness: 0.5,
            fade: 1.0,
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
            imp_info,
            imp_infos,
            atlases,
            atlas_blank,
            used_variant: used.variant,
            atlas_loader: crate::atlas::Loader::spawn(),
            atlas_budget,
            atlas_frame: 0,
            atlas_staging,
            atlas_ranges: crate::atlas::Ranges::new(ATLAS_STAGING),
            atlas_uploads: Vec::new(),
            atlas_retire: Vec::new(),
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
            static_insts: clusters.insts,
            near_cells: clusters.near,
            near_builds: Vec::new(),
            near_retire: Vec::new(),
            near_scratch: clusters.near_scratch,
            tree_of: clusters.tree_of,
            insts: Vec::new(),
            inst_data: Vec::new(),
            inst_info,
            tlas_dirty: true,
            upload_pending: false,
            first_update: true,
            grass_insts: Vec::new(),
            grass_at: Vec3::splat(f32::MAX),
            grass_radius: 0.0,
            near_count: 0,
            need_resize: false,
            render_scale,
            render_extent,
            resolve_layout,
            resolve_pipe,
            globe: None,
            globe_dummy: None,
            globe_view: None,
            ac_tex,
            ac_mats,
            ac_parts,
            ac_first,
            ac_pose: None,
            globe_states: std::collections::HashMap::new(),
            globe_trees: 0,
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
                        crate::catalog::Kind::Grass if n.contains("dry") || n.contains("wheat") || n.contains("marram") => sp.dry_grass.extend(v),
                        crate::catalog::Kind::Grass if n.contains("cotton") || n.contains("reed") => sp.wet_grass.extend(v),
                        crate::catalog::Kind::Grass => sp.grass.extend(v),
                        crate::catalog::Kind::Plant => sp.ferns.extend(v),
                        crate::catalog::Kind::Shrub if n.contains("heather") => sp.heath.extend(v),
                        // dry land: desert shrubs, yuccas, dead shrubs (bare ground too)
                        crate::catalog::Kind::Shrub if n.contains("yucca") || n.contains("dead_shrub") => sp.dry.extend(v),
                        crate::catalog::Kind::Shrub if n.contains("greasewood") => {
                            sp.dry.extend(v.iter().copied());
                            sp.shrubs.extend(v)
                        }
                        crate::catalog::Kind::Shrub => sp.shrubs.extend(v),
                        crate::catalog::Kind::Rock if n.contains("cact") => sp.dry.extend(v),
                        crate::catalog::Kind::Rock if n.contains("log") || n.contains("snag") => sp.deadwood.extend(v),
                        crate::catalog::Kind::Rock => sp.rocks.extend(v),
                        _ => {}
                    }
                }
                // the building archetypes (city.rs) for the city plan (globe/city.rs)
                for s in r.catalog.species.iter().filter(|s| s.kind == crate::catalog::Kind::Building) {
                    for v in s.variants.clone() {
                        let t = &r.pack.trees[v];
                        sp.buildings.push(crate::globe::city::ArchRef {
                            variant: v,
                            class: crate::city::class_of(&t.name),
                            size: [t.bmax.x - t.bmin.x, t.bmax.z - t.bmin.z],
                            height: t.bmax.y,
                        });
                    }
                }
                // the city's trees: the tall palms (catalog.rs add_city_palms: street rows, most
                // trees in a city) and broad,
                // spreading crowns standing in for live oaks and figs (no conifers, poplars or
                // bamboo in a Florida city)
                for s in &r.catalog.species {
                    let n = s.name.to_ascii_lowercase();
                    let tall = |v: &usize| r.catalog.variants[*v].height > 3.5;
                    if n == "city_palm" {
                        sp.palms.extend(s.variants.clone().filter(tall));
                    }
                    if n.contains("english_oak") || n.contains("norway_maple") || n.contains("european_beech") {
                        // (mature crowns only: the young ones read as shrubs from the air)
                        sp.city_trees.extend(s.variants.clone().filter(|v| r.catalog.variants[*v].height > 12.0));
                    }
                }
                println!("Cities: {} building archetypes, {} palm and {} broadleaf variants", sp.buildings.len(), sp.palms.len(), sp.city_trees.len());
                if sp.dry_grass.is_empty() {
                    sp.dry_grass = sp.grass.clone();
                }
                if sp.wet_grass.is_empty() {
                    sp.wet_grass = sp.grass.clone();
                }
                if sp.heath.is_empty() {
                    sp.heath = sp.shrubs.clone();
                }
                sp.biomes = vec![u16::MAX; r.imp_infos.len()];
                for s in &r.catalog.species {
                    let m = crate::globe::forest::understory_biomes(&s.name);
                    for v in s.variants.clone() {
                        if let Some(b) = sp.biomes.get_mut(v) {
                            *b = m;
                        }
                    }
                }
                r.globe = Some(crate::globe::scene::GlobeScene::new(&r.gpu, terrain, sp)?);
            }
            None => {
                let ubo = r.gpu.create_buffer(512, vk::BufferUsageFlags::UNIFORM_BUFFER, MemoryLocation::CpuToGpu, "globe placeholder")?;
                let buf = r.gpu.create_buffer(16, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "globe placeholder")?;
                let img = crate::textures::upload_with_mips(&r.gpu, 1, 1, &[0, 0, 0, 0], vk::Format::R8G8B8A8_UNORM, "globe placeholder")?;
                r.globe_dummy = Some((ubo, buf, img));
            }
        }
        r.write_set(r.main_set, r.tlas.accel.raw, None);
        r.write_resolve_set();
        r.create_dlss_feature()?;
        r.bake_impostors()?;
        // the atlases the bake installed (full ones outside the globe)
        r.write_set(r.main_set, r.tlas.accel.raw, None);
        println!("Renderer ready in {:.2}s", t0.elapsed().as_secs_f32());
        Ok(r)
    }

    /// Group the plants into CLUSTER x CLUSTER m cells and write the plant records in cell order,
    /// the plants without a draw distance first. Every cell gets an AABB BLAS with one box per
    /// such plant (its world bounds): always in the TLAS, one instance per cell (custom index =
    /// the cell's first record), next to the terrain. The plants with a draw distance
    /// (impostor.rint drops them beyond it) are only in the BLAS a cell gets while the camera is
    /// within that distance (build_near): rays in the far field do not stop at boxes that can
    /// never be hit, and those boxes take no memory.
    fn build_clusters(gpu: &Gpu, world: &World, pack: &TreePack, catalog: &Catalog, terrain_blas: u64) -> Result<(Clusters, f32)> {
        const CLUSTER: f32 = 192.0;
        let t0 = Instant::now();
        let n = world.trees.len();
        let culled = |t: &crate::world::TreeInstance| catalog.variants[t.variant as usize].cull > 0.0;
        let key = |t: &crate::world::TreeInstance| {
            let p = t.center;
            ((((p.z / CLUSTER).floor() as i32) as i64) << 32 | (((p.x / CLUSTER).floor() as i32) as u32 as i64), culled(t))
        };
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_by_key(|&i| key(&world.trees[i as usize]));
        let mut rec_of = vec![0u32; n];
        let mut recs: Vec<PlantRecGpu> = Vec::with_capacity(n);
        let mut aabbs: Vec<[f32; 6]> = Vec::with_capacity(n);
        // per cell: first record, plants without a draw distance, plants; the box of the culled
        // plants' centres and their largest draw distance
        struct Cell {
            first: u32,
            keep: u32,
            count: u32,
            lo: Vec3,
            hi: Vec3,
            cull: f32,
        }
        let mut cells: Vec<Cell> = Vec::new();
        let mut last_cell = None;
        for (k, &i) in order.iter().enumerate() {
            let t = &world.trees[i as usize];
            rec_of[i as usize] = k as u32;
            let (cell, far) = key(t);
            if last_cell != Some(cell) {
                cells.push(Cell { first: k as u32, keep: 0, count: 0, lo: Vec3::splat(f32::MAX), hi: Vec3::splat(f32::MIN), cull: 0.0 });
                last_cell = Some(cell);
            }
            let c = cells.last_mut().unwrap();
            c.count += 1;
            if far {
                (c.lo, c.hi, c.cull) = (c.lo.min(t.center), c.hi.max(t.center), c.cull.max(catalog.variants[t.variant as usize].cull));
            } else {
                c.keep += 1;
            }
            let data = crate::globe::forest::rec_data(t.variant as usize, ((t.seed as u32 & 0x1FFF) << 8) | ((t.dry as u32 & 15) << 21));
            recs.push(PlantRecGpu::new(t.xform.matrix3.into(), t.xform.translation.into(), data));
            aabbs.push(plant_box(t, &pack.trees[t.variant as usize]));
        }
        let mut aabb_buf = gpu.upload_buffer(
            bytemuck::cast_slice(&aabbs),
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            "plant aabbs",
        )?;
        let kept: Vec<usize> = (0..cells.len()).filter(|&c| cells[c].keep > 0).collect();
        let geoms: Vec<_> = kept.iter().map(|&c| (accel::aabb_geometry(aabb_buf.addr + cells[c].first as u64 * 24), cells[c].keep)).collect();
        let blas = accel::build_blases(gpu, &geoms)?;
        gpu.destroy_buffer(&mut aabb_buf); // build input only
        let rec = gpu.upload_buffer(bytemuck::cast_slice(&recs), vk::BufferUsageFlags::STORAGE_BUFFER, "plant records")?;
        let mut state = gpu.create_buffer((n.max(1) * 4) as u64, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "plant state")?;
        state.mapped().fill(0);
        let terrain_id = pack.meshes.len();
        let mesh_flags = vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE | vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE;
        let mut insts = vec![accel::make_instance(&glam::Affine3A::IDENTITY, terrain_id as u32, 1, 0, mesh_flags, terrain_blas)];
        let mut far_inst: Vec<Option<vk::AccelerationStructureInstanceKHR>> = vec![None; cells.len()];
        for (&c, b) in kept.iter().zip(&blas) {
            far_inst[c] = Some(cluster_instance(cells[c].first, b.addr));
        }
        let mut near = Vec::new();
        let (mut near_plants, mut scratch) = (0usize, 0u64);
        for (ci, c) in cells.iter().enumerate() {
            if c.count > c.keep {
                near.push(NearCell { first: c.first, count: c.count, far: far_inst[ci], lo: c.lo, hi: c.hi, cull: c.cull, on: false, full: None });
                near_plants += (c.count - c.keep) as usize;
                scratch = scratch.max(accel::blas_sizes(gpu, &accel::aabb_geometry(0), c.count).1);
            } else if let Some(i) = far_inst[ci] {
                insts.push(i);
            }
        }
        let align = gpu.rt.scratch_alignment.max(256);
        let near_scratch = gpu.create_buffer(scratch + align, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, "near blas scratch")?;
        let max_ls = catalog.variants.iter().map(|v| v.lod_scale).fold(0.0f32, f32::max);
        println!(
            "Impostor clusters: {} plants in {} cells; {} plants with a draw distance join their cell near the camera ({:.2}s)",
            n,
            cells.len(),
            near_plants,
            t0.elapsed().as_secs_f32()
        );
        Ok((Clusters { rec_of, tree_of: order, rec, state, blas, insts, near, near_scratch }, max_ls))
    }

    /// A cell with draw-distance plants has come within their reach: its complete BLAS (every
    /// plant of the cell), created now and built in this frame (record_near_builds).
    fn build_near(&mut self, ci: usize) -> Result<()> {
        let c = &self.near_cells[ci];
        let boxes: Vec<[f32; 6]> = (c.first..c.first + c.count)
            .map(|r| {
                let t = &self.world.trees[self.tree_of[r as usize] as usize];
                plant_box(t, &self.pack.trees[t.variant as usize])
            })
            .collect();
        let bytes: &[u8] = bytemuck::cast_slice(&boxes);
        let mut input = self.gpu.create_buffer(
            bytes.len() as u64,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            MemoryLocation::CpuToGpu,
            "near plant aabbs",
        )?;
        input.mapped()[..bytes.len()].copy_from_slice(bytes);
        let (size, _) = accel::blas_sizes(&self.gpu, &accel::aabb_geometry(input.addr), c.count);
        let accel = accel::create_accel(&self.gpu, vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL, size)?;
        self.near_cells[ci].full = Some(NearBlas { accel, input });
        self.near_builds.push(ci);
        Ok(())
    }

    /// The BLAS builds of the cells that came into reach, before this frame's TLAS build.
    fn record_near_builds(&mut self, cb: vk::CommandBuffer) {
        let align = self.gpu.rt.scratch_alignment.max(256);
        let scratch = align_up(self.near_scratch.addr, align);
        for ci in std::mem::take(&mut self.near_builds) {
            let c = &self.near_cells[ci];
            let Some(f) = &c.full else { continue }; // left again before it was built
            accel::cmd_build_blas(&self.gpu, cb, &accel::aabb_geometry(f.input.addr), c.count, f.accel.raw, scratch);
            // the scratch buffer is shared: one build after the other
            self.gpu.memory_barrier(
                cb,
                vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
                vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
            );
        }
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
        let cloud_info = |v: vk::ImageView| [vk::DescriptorImageInfo::default().sampler(self.mat_sampler).image_view(v).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let (cloud_shape, cloud_detail) = (cloud_info(self.cloud_noise[0].view), cloud_info(self.cloud_noise[1].view));
        let ocean_tex: Vec<vk::DescriptorImageInfo> = (0..OCEAN_TEX)
            .map(|k| {
                let view = match (&self.globe, &self.globe_dummy) {
                    (Some(g), _) => g.ocean_tex[k].view,
                    (None, Some(d)) => d.2.view,
                    _ => unreachable!(),
                };
                vk::DescriptorImageInfo::default().sampler(self.mat_sampler).image_view(view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            })
            .collect();
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
        let mat_tex = tex(&self.mat_tex, crate::textures::SPECIES_SLOTS * MAX_SPECIES);
        let ground_tex = tex(&self.ground_tex, MAX_GROUND_TEX);
        let ac_tex = tex(&self.ac_tex, 3);
        let ac_mats = buf(&self.ac_mats);
        let sampled = |pick: fn(&BcAtlas) -> vk::ImageView| -> Vec<vk::DescriptorImageInfo> {
            (0..MAX_VARIANTS)
                .map(|i| {
                    let a = self.atlases.get(i).and_then(|a| a.shown()).unwrap_or(&self.atlas_blank);
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
            w(33, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&ocean_tex),
            w(34, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&cloud_shape),
            w(35, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&cloud_detail),
            w(36, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&ac_tex),
            w(37, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&ac_mats),
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
        // only what the cache (atlas.rs) does not hold yet, or everything with SI_REBAKE
        let rebake = std::env::var("SI_REBAKE").is_ok();
        let missing: Vec<usize> = (0..self.atlases.len())
            .filter(|&v| {
                let a = &self.atlases[v];
                self.used_variant[v] && (rebake || !crate::atlas::valid(&a.path, a.key, a.grid, a.res, a.mips))
            })
            .collect();
        if missing.is_empty() {
            println!("Impostor atlases: all {} in the cache ({})", self.used_variant.iter().filter(|&&u| u).count(), crate::atlas::DIR);
            return self.load_base_atlases(t_all);
        }
        let max_w = missing.iter().map(|&v| self.atlases[v].grid * self.atlases[v].res[0]).max().unwrap_or(32);
        let max_h = missing.iter().map(|&v| self.atlases[v].grid * self.atlases[v].res[1]).max().unwrap_or(32);
        let smips = missing.iter().map(|&v| self.atlases[v].mips).max().unwrap_or(1);
        let st = vk::ImageUsageFlags::STORAGE;
        let mut scratch = [
            self.gpu.create_image(max_w, max_h, smips, vk::Format::R8G8B8A8_UNORM, st, "bake albedo")?,
            self.gpu.create_image(max_w, max_h, smips, vk::Format::R16G16B16A16_SFLOAT, st, "bake normal-depth")?,
            self.gpu.create_image(max_w, max_h, smips, vk::Format::R8G8B8A8_UNORM, st, "bake visibility")?,
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
        let enc_size = max_w as u64 * max_h as u64 * 9 / 2;
        let mut enc = self.gpu.create_buffer(
            enc_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::GpuOnly,
            "bc blocks",
        )?;
        let mut enc_host = self.gpu.create_buffer(enc_size, vk::BufferUsageFlags::TRANSFER_DST, MemoryLocation::GpuToCpu, "bc blocks readback")?;

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
        for &v in &missing {
            let t0 = Instant::now();
            let (res, grid, mips) = (self.atlases[v].res, self.atlases[v].grid, self.atlases[v].mips);
            let (square, crop) = (self.atlases[v].square, self.atlases[v].crop);
            let flags = vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE | vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE;
            let alpha_flags = vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE | vk::GeometryInstanceFlagsKHR::FORCE_NO_OPAQUE;
            // the lawn: its tile between copies of itself, so the views wrap around
            let lawn = self.lawn.as_ref().filter(|l| l.slot == v);
            let tree = self.pack.trees.get(v).filter(|_| lawn.is_none());
            let insts: Vec<_> = match (lawn, tree) {
                // (the flattest view's rays run SLOPE_MAX blade heights sideways: lawn::copies)
                (Some(l), _) => {
                    let c = crate::lawn::copies();
                    (0..(2 * c + 1) * (2 * c + 1))
                        .map(|k| {
                            let at = Vec3::new((k % (2 * c + 1) - c) as f32, 0.0, (k / (2 * c + 1) - c) as f32) * crate::lawn::tile_size();
                            accel::make_instance(&glam::Affine3A::from_translation(at), l.mesh, 0xFF, 0, flags, self.blas[l.mesh as usize].addr)
                        })
                        .collect()
                }
                (None, Some(tree)) => tree
                    .parts
                    .iter()
                    .map(|p| {
                        let alpha = self.catalog.mesh_alpha.get(p.mesh as usize).copied().unwrap_or(false);
                        let (f, sbt) = if alpha { (alpha_flags, 2) } else if self.catalog.is_building_mesh(p.mesh as usize) { (flags, SBT_BUILDING) } else { (flags, 0) };
                        accel::make_instance(&p.xform, p.mesh, 0xFF, sbt, f, self.blas[p.mesh as usize].addr)
                    })
                    .collect(),
                (None, None) => unreachable!("an atlas is a plant variant's or the lawn's"),
            };
            let (name, parts, tris) = match (lawn, tree) {
                (Some(l), _) => ("Lawn tile".to_string(), insts.len(), insts.len() as u64 * l.tris as u64),
                (_, Some(tree)) => (tree.name.clone(), tree.parts.len(), tree.expanded_tris(&self.pack.meshes)),
                _ => unreachable!(),
            };
            let mut tlas = Tlas::new(&self.gpu, insts.len() as u32)?;
            tlas.write(&insts);
            self.gpu.one_time(|cb| tlas.cmd_build(&self.gpu, cb, insts.len() as u32))?;
            self.write_set(self.bake_set, tlas.accel.raw, Some(bake_views));

            // what bake.rgen gets (centre and radius of the plant; the lawn: tile edge, blade
            // height, slope of the steepest and flattest views), the size of a bake sub-sample,
            // the range of the occlusion rays, its mode (1 = lawn, with the gutter texels above
            // it) and the frame's crop (the lawn: its views)
            let (centre, sample, ao_range, mode, crop) = match tree {
                Some(tree) => {
                    let c = (tree.bmin + tree.bmax) * 0.5;
                    let r = (tree.bmax - tree.bmin).length() * 0.5;
                    // (occlusion range in metres for full-size trees; small plants a proportionally shorter one)
                    ([c.x, c.y, c.z, r], 2.0 * r / square as f32, imp.ao_range * (r / 8.0).clamp(0.2, 1.0), 0, crop)
                }
                None => {
                    use crate::lawn::{AZIMUTHS, GUTTER, RES, SLOPES, SLOPE_MAX, SLOPE_MIN};
                #[allow(non_snake_case)]
                let TILE = crate::lawn::tile_size();
                    let height = crate::lawn::height();
                    ([TILE, height, SLOPE_MIN, SLOPE_MAX], TILE / (RES - 2 * GUTTER) as f32, 2.5 * height, 1 | GUTTER << 8, [AZIMUTHS, SLOPES])
                }
            };
            // Bake rays are orthographic: fixed texture footprint = one bake sub-sample (negative
            // spread in tri.rchit), and the material textures switched on.
            let mut bg = Globals::default();
            bg.cam_fwd[3] = -(sample / imp.ss as f32);
            bg.params[1] = self.has_tex as u32 as f32;
            bg.lod_params[2] = -1.0; // bake: tri.rahit only alpha-tests, no LOD dissolve
            self.ubo.mapped()[..std::mem::size_of::<Globals>()].copy_from_slice(bytemuck::bytes_of(&bg));
            let mut push = BakePush { center_radius: centre, grid, res: square, frame_offset: 0, ss: imp.ss, ao_rays: imp.ao_rays, ao_range, vis_rays: imp.vis_rays, pad: mode, size: res, crop };
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
                        res[0],
                        res[1],
                        grid,
                    );
                })?;
            }

            let size = [grid * res[0], grid * res[1]];
            let regions = crate::atlas::layout(grid, res, mips);
            let stream = crate::atlas::stream_bytes(grid, res, mips);
            assert!(stream <= enc_size, "bc block buffer too small");
            self.gpu.one_time(|cb| unsafe {
                let rw = vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE;
                let cs = vk::PipelineStageFlags2::COMPUTE_SHADER;
                self.gpu.memory_barrier(cb, vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR, vk::AccessFlags2::SHADER_WRITE, cs, rw);
                // coverage-weighted mips of the uncompressed scratch atlas
                d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.mip_pipe);
                for level in 1..mips {
                    d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, self.mip_layout, 0, &[mip_sets[level as usize]], &[]);
                    d.cmd_dispatch(cb, (size[0] >> level).div_ceil(8), (size[1] >> level).div_ceil(8), 1);
                    self.gpu.memory_barrier(cb, cs, vk::AccessFlags2::SHADER_WRITE, cs, rw);
                }
                // BC encode every region of the stream (atlas.rs layout): mode = image; the
                // visibility levels read the scratch level above their own (half resolution)
                d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.bcenc_pipe);
                for r in &regions {
                    let scratch_level = if r.img == 3 { r.level + 1 } else { r.level };
                    let (bx, by) = (r.size[0] / 4, r.size[1] / 4);
                    d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, self.bcenc_layout, 0, &[enc_sets[scratch_level as usize]], &[]);
                    let pc = [r.img as u32, bx, by, (r.offset / 16) as u32];
                    d.cmd_push_constants(cb, self.bcenc_layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::cast_slice(&pc));
                    d.cmd_dispatch(cb, bx.div_ceil(8), by.div_ceil(8), 1);
                }
                self.gpu.memory_barrier(cb, cs, vk::AccessFlags2::SHADER_WRITE, vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_READ);
                d.cmd_copy_buffer(cb, enc.raw, enc_host.raw, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: stream }]);
            })?;
            let a = &self.atlases[v];
            crate::atlas::write(&a.path, a.key, grid, res, mips, &enc_host.mapped()[..stream as usize])?;
            if v == 0 {
                if let Some(prefix) = self.dump_prefix.clone() {
                    self.dump_scratch(&scratch[0], size, &prefix)?;
                }
            }
            tlas.destroy(&self.gpu);
            total_tris += tris;
            println!(
                "Baked {:<28} {:>4} parts {:>6.1}M tris -> {:>3}x{:<3} px frames (of {:>3}), {:>4}x{:<4} atlas, {:>5.1} MB, {:.2}s",
                name,
                parts,
                tris as f64 / 1e6,
                res[0],
                res[1],
                square,
                size[0],
                size[1],
                stream as f64 / 1e6,
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
        self.gpu.destroy_buffer(&mut enc_host);
        println!(
            "Baked {} impostors ({:.0}M source triangles) into {} in {:.1}s",
            missing.len(),
            total_tris as f64 / 1e6,
            crate::atlas::DIR,
            t_all.elapsed().as_secs_f32()
        );
        self.load_base_atlases(t_all)
    }

    /// The atlases from the cache. The globe: every variant's base atlas (the full ones stream in
    /// near the camera). The forest scene: the full atlas of every variant that grows in it (one
    /// forest, every variant in view), and nothing for the others.
    fn load_base_atlases(&mut self, t_all: Instant) -> Result<()> {
        // (every variant's file stays, also those this scene does not show: the next biome's)
        crate::atlas::prune(&self.atlases.iter().map(|a| a.path.clone()).collect::<Vec<_>>());
        if self.globe.is_some() {
            let mut bytes = 0u64;
            for v in 0..self.atlases.len() {
                let a = &self.atlases[v];
                let Some(base) = &a.base else { continue };
                let regions: Vec<_> = crate::atlas::layout(a.grid, a.res, a.mips).into_iter().filter(|r| r.level >= a.skip).collect();
                let data = crate::atlas::read_regions(&a.path, &regions)?;
                self.upload_atlas(base, &regions, &data, a.skip)?;
                bytes += base.bytes();
            }
            println!(
                "Impostor base atlases: {:.2} GB on the GPU for {} variants (full atlases stream in near the camera)",
                bytes as f64 / 1e9,
                self.atlases.len()
            );
        } else {
            let mut bytes = 0u64;
            let scene: Vec<usize> = (0..self.atlases.len()).filter(|&v| self.used_variant[v]).collect();
            for v in scene {
                let a = &self.atlases[v];
                let stream = crate::atlas::read_regions(&a.path, &crate::atlas::layout(a.grid, a.res, a.mips))?;
                self.install_full(v, &stream)?;
                bytes += self.atlases[v].full_bytes();
            }
            println!("Impostor atlases: {:.2} GB on the GPU for the scene's {} variants", bytes as f64 / 1e9, self.used_variant.iter().filter(|&&u| u).count());
        }
        println!("Impostors ready in {:.1}s", t_all.elapsed().as_secs_f32());
        Ok(())
    }

    /// BC blocks (concatenated `regions` of an atlas.rs stream) into an atlas whose level 0 is
    /// the stream's level `skip`, now (startup: a one-time submit and wait).
    fn upload_atlas(&self, a: &BcAtlas, regions: &[crate::atlas::Region], data: &[u8], skip: u32) -> Result<()> {
        let mut staging = self.gpu.create_buffer(data.len() as u64, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu, "atlas staging")?;
        staging.mapped()[..data.len()].copy_from_slice(data);
        self.gpu.one_time(|cb| self.cmd_upload_atlas(cb, a, regions, staging.raw, 0, skip))?;
        self.gpu.destroy_buffer(&mut staging);
        Ok(())
    }

    /// Record the copies of an atlas.rs stream at `offset` in `src` into `a` (level 0 = the
    /// stream's level `skip`), with the layout transitions; the atlas is readable by the ray
    /// tracing shaders recorded after it.
    fn cmd_upload_atlas(&self, cb: vk::CommandBuffer, a: &BcAtlas, regions: &[crate::atlas::Region], src: vk::Buffer, offset: u64, skip: u32) {
        let imgs = [a.albedo.raw, a.normal.raw, a.depth.raw, a.vis.raw];
        unsafe {
            for img in imgs {
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
            let mut at = offset;
            for r in regions {
                let region = vk::BufferImageCopy {
                    buffer_offset: at,
                    image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: r.level - skip, base_array_layer: 0, layer_count: 1 },
                    image_extent: vk::Extent3D { width: r.size[0], height: r.size[1], depth: 1 },
                    ..Default::default()
                };
                self.gpu.device.cmd_copy_buffer_to_image(cb, src, imgs[r.img], vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region]);
                at += r.bytes;
            }
            for img in imgs {
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
        }
    }

    /// This frame's full-atlas uploads, recorded before the ray tracing; their upload ranges are
    /// freed after the frame's fence.
    fn record_atlas_uploads(&mut self, cb: vk::CommandBuffer) {
        for (v, offset) in std::mem::take(&mut self.atlas_uploads) {
            let (grid, res, mips) = (self.atlases[v].grid, self.atlases[v].res, self.atlases[v].mips);
            if let Some(full) = &self.atlases[v].full {
                let src = self.atlas_staging.as_ref().expect("uploads only on the globe").raw;
                self.cmd_upload_atlas(cb, full, &crate::atlas::layout(grid, res, mips), src, offset, 0);
            }
            self.atlas_retire.push((offset, crate::atlas::stream_bytes(grid, res, mips)));
        }
    }

    /// A variant's full atlas from its whole cache stream, and the info the shaders read.
    fn install_full(&mut self, v: usize, stream: &[u8]) -> Result<()> {
        let (grid, res, mips) = (self.atlases[v].grid, self.atlases[v].res, self.atlases[v].mips);
        let full = create_bc_atlas(&self.gpu, &format!("{} full", self.atlases[v].name), grid, res, mips)?;
        self.upload_atlas(&full, &crate::atlas::layout(grid, res, mips), stream, 0)?;
        if let Some(mut old) = self.atlases[v].full.replace(full) {
            old.destroy(&self.gpu);
        }
        self.set_atlas_info(v);
        Ok(())
    }

    /// The resolution and levels of the atlas a variant shows now, into the impostor info.
    fn set_atlas_info(&mut self, v: usize) {
        let Some(a) = self.atlases[v].shown() else { return };
        let (res, mips) = (a.res, a.mips);
        self.imp_infos[v].res = res[0] | res[1] << 16;
        self.imp_infos[v].mips = mips;
        let n = std::mem::size_of::<ImpostorInfo>();
        self.imp_info.mapped()[v * n..(v + 1) * n].copy_from_slice(bytemuck::bytes_of(&self.imp_infos[v]));
    }

    /// After the fence (no frame in flight): full atlases for the variants growing near the
    /// camera, base atlases back for the ones left behind, within the full-atlas budget. `pixel`:
    /// the angle of one display pixel (radians).
    fn update_atlases(&mut self, pixel: f32) -> Result<()> {
        self.atlas_frame += 1;
        // the previous frame has copied its uploads out
        for (offset, len) in std::mem::take(&mut self.atlas_retire) {
            self.atlas_ranges.release(offset, len);
        }
        let mut dirty = false;
        while let Ok((v, r)) = self.atlas_loader.rx.try_recv() {
            let a = &mut self.atlases[v];
            a.loading = false;
            let offset = a.upload.take().expect("an upload range for every request");
            match r {
                Ok(()) => {
                    // the images now, filled by copies recorded into this frame (record_atlas_uploads)
                    let t0 = std::time::Instant::now();
                    let full = create_bc_atlas(&self.gpu, &format!("{} full", a.name), a.grid, a.res, a.mips)?;
                    if std::env::var("SI_ATLAS_LOG").is_ok() {
                        println!("atlas {} full: {:.1} MB, images created in {:.2} ms", self.atlases[v].name, full.bytes() as f64 / 1e6, t0.elapsed().as_secs_f64() * 1e3);
                    }
                    if let Some(mut old) = self.atlases[v].full.replace(full) {
                        old.destroy(&self.gpu);
                    }
                    self.set_atlas_info(v);
                    self.atlas_uploads.push((v, offset));
                    dirty = true;
                }
                Err(e) => {
                    eprintln!("impostor atlas {}: {e:#}", a.name);
                    let len = crate::atlas::stream_bytes(a.grid, a.res, a.mips);
                    self.atlas_ranges.release(offset, len);
                }
            }
        }
        if self.atlas_frame.is_multiple_of(10) {
            if let (Some(scene), Some((ecef, _, _))) = (&self.globe, self.globe_view) {
                let n = self.atlases.len();
                let mut dist = vec![f32::INFINITY; n];
                scene.variants_near(ecef, ATLAS_RADIUS, &mut dist);
                // a variant needs its full atlas while the base atlas's level 0 would be sampled:
                // its texels larger than a pixel there
                let need: Vec<f32> = (0..n)
                    .map(|v| {
                        let a = &self.atlases[v];
                        let texel = a.texel(self.imp_infos[v].center_radius[3]);
                        (texel * (1u32 << a.skip) as f32 / pixel.max(1e-5)).clamp(30.0, ATLAS_RADIUS as f32)
                    })
                    .collect();
                // nearest (relative to need) first, within the budget; a full atlas stays while
                // within 1.5x its distance (no flicker at the edge)
                let mut order: Vec<usize> = (0..n).filter(|&v| dist[v] < need[v] * if self.atlases[v].full.is_some() { 1.5 } else { 1.0 }).collect();
                order.sort_by(|&a, &b| (dist[a] / need[a]).partial_cmp(&(dist[b] / need[b])).unwrap());
                let mut used = 0u64;
                for v in order {
                    let bytes = self.atlases[v].full_bytes();
                    if used + bytes > self.atlas_budget {
                        continue;
                    }
                    used += bytes;
                    let a = &mut self.atlases[v];
                    a.wanted = self.atlas_frame;
                    if a.full.is_none() && !a.loading {
                        // read straight into the upload buffer (later, when it is full)
                        let len = crate::atlas::stream_bytes(a.grid, a.res, a.mips);
                        if let (Some(staging), Some(offset)) = (self.atlas_staging.as_mut(), self.atlas_ranges.alloc(len)) {
                            let addr = staging.mapped().as_mut_ptr() as usize + offset as usize;
                            a.loading = true;
                            a.upload = Some(offset);
                            self.atlas_loader.request(v, a.path.clone(), crate::atlas::Dest { addr, len: len as usize });
                        }
                    }
                }
                for v in 0..n {
                    // not wanted for a while (10 frames per check): back to the base atlas
                    if self.atlases[v].full.is_some() && self.atlases[v].wanted + 30 < self.atlas_frame {
                        if let Some(mut f) = self.atlases[v].full.take() {
                            f.destroy(&self.gpu);
                        }
                        self.set_atlas_info(v);
                        dirty = true;
                    }
                }
            }
        }
        if dirty {
            let t0 = std::time::Instant::now();
            self.write_set(self.main_set, self.tlas.accel.raw, None);
            if std::env::var("SI_ATLAS_LOG").is_ok() {
                println!("atlas descriptors rewritten in {:.2} ms", t0.elapsed().as_secs_f64() * 1e3);
            }
        }
        Ok(())
    }

    /// Full atlases on their way from the cache (the tour waits for them before a shot).
    pub fn atlases_pending(&self) -> bool {
        self.atlases.iter().any(|a| a.loading)
    }

    /// GPU memory of the impostor atlases: (base, full) bytes and how many variants are full.
    pub fn atlas_memory(&self) -> (u64, u64, usize) {
        let base = self.atlases.iter().filter_map(|a| a.base.as_ref()).map(|a| a.bytes()).sum();
        let full: Vec<_> = self.atlases.iter().filter_map(|a| a.full.as_ref()).collect();
        (base, full.iter().map(|a| a.bytes()).sum(), full.len())
    }

    /// Debug: write the albedo of the first baked variant (uncompressed scratch) as a PNG.
    fn dump_scratch(&self, img: &Image, size: [u32; 2], prefix: &str) -> Result<()> {
        let n = size[0] as u64 * size[1] as u64;
        let mut buf = self.gpu.create_buffer(n * 4, vk::BufferUsageFlags::TRANSFER_DST, MemoryLocation::GpuToCpu, "dump")?;
        let layers = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 };
        let region = vk::BufferImageCopy {
            image_subresource: layers,
            image_extent: vk::Extent3D { width: size[0], height: size[1], depth: 1 },
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
        let mut enc = png::Encoder::new(file, size[0], size[1]);
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
        // Cells with draw-distance plants: complete while the camera is within that distance
        // (impostor.rint measures to the plant's centre; a few metres early, and back later, so a
        // camera at the edge does not rebuild the TLAS every frame). A complete BLAS is kept a
        // while longer for a camera that turns back.
        let mut build = Vec::new();
        for (ci, c) in self.near_cells.iter_mut().enumerate() {
            let d = (cam.clamp(c.lo, c.hi) - cam).length();
            let on = d < c.cull + if c.on { 24.0 } else { 8.0 };
            changed |= on != c.on;
            c.on = on;
            if on && c.full.is_none() {
                build.push(ci);
            } else if !on && d > c.cull + 150.0 {
                self.near_retire.extend(c.full.take());
            }
        }
        for ci in build {
            self.build_near(ci)?;
        }
        changed |= self.place_grass(cam);
        let t_states = Instant::now();
        self.profile.cpu_lod_parts[0] += (t_states - t_begin).as_secs_f64() * 1000.0;
        if changed {
            let mesh_flags = vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE | vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE;
            // Band plants and alpha-tested cards: not opaque, so tri.rahit runs (hit record 2).
            let band_flags = vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE | vk::GeometryInstanceFlagsKHR::FORCE_NO_OPAQUE;
            self.insts.clear();
            self.inst_data.clear();
            self.insts.extend_from_slice(&self.static_insts);
            self.insts.extend(self.near_cells.iter().filter_map(|c| match (&c.full, c.on) {
                (Some(f), true) => Some(cluster_instance(c.first, f.accel.addr)),
                _ => c.far,
            }));
            self.insts.extend_from_slice(&self.grass_insts);
            // (the terrain, the clusters and the grass have no per-instance info: tri.rahit never
            // runs on them)
            self.inst_data.resize(self.insts.len(), InstInfo { center: [0.0, -1e9, 0.0], data: 0, flags: 0 });
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
                // tri.rahit: dissolve; bits 1..4 the biome's dryness tint (tri.rchit)
                let band = if state == Lod::Both { INST_HAS_MESH } else { 0 } | ((t.dry as u32 & 15) << 1);
                for (pi, p) in variant.parts.iter().enumerate() {
                    let alpha = self.catalog.mesh_alpha[p.mesh as usize];
                    let (flags, sbt) = if state == Lod::Mesh && !alpha { (mesh_flags, 0) } else { (band_flags, 2) };
                    let m = t.xform * p.xform;
                    self.insts.push(accel::make_instance(&m, p.mesh | seed, 1, sbt, flags, self.blas[p.mesh as usize].addr));
                    self.inst_data.push(InstInfo { center, data: tree, flags: band | (pi as u32) << INST_PART_SHIFT });
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
        Ok(FrameStats { near_trees: self.near_count, instances: self.insts.len() })
    }

    /// The lawn's real blades around the camera (lawn.rs): one instance of the tile's mesh per
    /// square of the lawn's grid within reach, standing on the terrain (sheared to its slope, so
    /// the blades stay upright), turned one of four ways (blades are whole, so any placing of a
    /// tile joins its neighbours), as tall as the layer is there, and thinning out blade by blade
    /// towards the edge of the circle, where the terrain shader's layer takes over (tri.rahit,
    /// tri.rchit, lawn3.w). Placed again when the
    /// camera has moved a quarter of a metre. Returns whether the instances changed.
    fn place_grass(&mut self, cam: Vec3) -> bool {
        let radius = crate::lawn::near_radius();
        let Some(mesh) = self.lawn.as_ref().filter(|_| self.lawn_on && radius > 0.0).map(|l| l.mesh) else {
            let had = !self.grass_insts.is_empty();
            self.grass_insts.clear();
            self.grass_at = Vec3::splat(f32::MAX);
            return had;
        };
        // (or the radius has changed: the grass quality, key B)
        if (cam - self.grass_at).length() < 0.25 && radius == self.grass_radius {
            return false;
        }
        self.grass_at = cam;
        self.grass_radius = radius;
        let before = self.grass_insts.len();
        self.grass_insts.clear();
        // (well above the ground there are no blades to see: the layer alone)
        if cam.y - self.world.height_at(cam.x, cam.z) > radius {
            return before > 0;
        }
        let tile = crate::lawn::tile_size();
        let flags = vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE | vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE;
        let thin = vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE | vk::GeometryInstanceFlagsKHR::FORCE_NO_OPAQUE;
        let addr = self.blas[mesh as usize].addr;
        let cell = |v: f32| (v / tile).floor() as i32;
        for j in cell(cam.z - radius)..=cell(cam.z + radius) {
            for i in cell(cam.x - radius)..=cell(cam.x + radius) {
                let (x, z) = ((i as f32 + 0.5) * tile, (j as f32 + 0.5) * tile);
                let y = self.world.height_at(x, z);
                let d = (Vec3::new(x, y, z) - cam).length();
                if d - 0.75 * tile > radius || !self.world.grows_lawn(x, z) {
                    continue;
                }
                // Towards the edge of the circle the blades go one by one (tri.rahit, which runs
                // on the tiles that reach into that band), not shorter.
                let band = d + 0.75 * tile > 0.5 * radius;
                // Its height: what the layer stands at (the patches, the auto mode's grass map),
                // which changes across a tile. The tile is as tall as its tallest place and
                // tri.rahit trims every blade to its own: one field, no step between tiles. (A
                // tile on even ground needs no trimming and stays opaque.)
                let short = crate::lawn::auto_short();
                let stands = |x: f32, z: f32| crate::lawn::patch_height(x, z) * (short + (1.0 - short) * self.world.grass_at(x, z));
                let e = 0.5 * tile;
                let at = [stands(x, z), stands(x - e, z - e), stands(x + e, z - e), stands(x - e, z + e), stands(x + e, z + e), stands(x - e, z), stands(x + e, z), stands(x, z - e), stands(x, z + e)];
                let (lo, hi) = at.iter().fold((f32::MAX, 0.0f32), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
                let grown = (hi * 1.04 + 0.01).min(1.0);
                let band = band || grown - lo > 0.03;
                let (gx, gz) = ((self.world.height_at(x + e, z) - self.world.height_at(x - e, z)) / tile, (self.world.height_at(x, z + e) - self.world.height_at(x, z - e)) / tile);
                let h = (i as u32).wrapping_mul(0x9E37_79B9) ^ (j as u32).wrapping_mul(0x85EB_CA6B);
                let turn = glam::Affine3A::from_rotation_y(((h >> 13) & 3) as f32 * std::f32::consts::FRAC_PI_2);
                let shear = glam::Affine3A::from_mat3(glam::Mat3::from_cols(Vec3::new(1.0, gx, 0.0), Vec3::new(0.0, grown, 0.0), Vec3::new(0.0, gz, 1.0)));
                let m = glam::Affine3A::from_translation(Vec3::new(x, y - 0.004, z)) * shear * turn * glam::Affine3A::from_translation(Vec3::new(-e, 0.0, -e));
                let (f, sbt) = if band { (thin, 2) } else { (flags, 0) };
                self.grass_insts.push(accel::make_instance(&m, mesh, 1, sbt, f, addr));
            }
        }
        before > 0 || !self.grass_insts.is_empty()
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
        if self.resize_count <= 3 || self.resize_count.is_multiple_of(100) {
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
        g.lod_params[3] = f32::from_bits(self.ground_layers); // bit mask of the scanned ground layers loaded (bits, not a value: 26 layers)
        g.params[1] = self.has_tex as u32 as f32; // material textures available
        g.lawn = match &self.lawn {
            Some(l) if self.lawn_on => {
                use crate::lawn::{AZIMUTHS, GUTTER, RES, SLOPES, SLOPE_MAX, SLOPE_MIN};
                #[allow(non_snake_case)]
                let TILE = crate::lawn::tile_size();
                [
                    [crate::lawn::height(), TILE, AZIMUTHS as f32, SLOPES as f32],
                    [RES as f32, GUTTER as f32, self.atlases[l.slot].mips as f32, l.slot as f32],
                    [SLOPE_MIN, SLOPE_MAX, crate::world::park() as u32 as f32, crate::lawn::near_radius()],
                    [crate::lawn::segments() as f32, crate::lawn::auto_short(), crate::lawn::patch_low(), crate::lawn::fade_distance()],
                ]
            }
            _ => [[0.0; 4]; 4],
        };
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
        // the sea state of the last ocean frame uploaded (globe/ocean.rs, main.rgen)
        if let Some(scene) = self.globe.as_ref() {
            g.ocean = scene.ocean_globals();
        }
        let globals = &g;
        let t_lod = Instant::now();
        // Globe mode: the instances come from the streamed scene (below).
        let mut stats = if self.globe.is_some() { FrameStats::default() } else { self.update_instances(cam, lod, g.lod_params[0])? };
        self.profile.cpu_lod += t_lod.elapsed().as_secs_f64() * 1000.0;
        // Globe mode: the CPU half of streaming (tile choice, worker requests, TLAS instances, mesh
        // LOD) overlaps the previous frame's GPU work; uploads follow the fence.
        let mut globe_states = std::collections::HashMap::new();
        let mut globe_cpu = [0.0f64; 3];
        if let (Some(scene), Some((ecef, rot, _))) = (self.globe.as_mut(), self.globe_view) {
            self.insts.clear();
            let p = scene.prepare(ecef, rot, &mut self.insts);
            self.lod_frame = self.lod_frame.wrapping_add(1);
            self.inst_data.clear();
            self.inst_data.resize(self.insts.len(), InstInfo { center: [0.0, -1e9, 0.0], data: 0, flags: 0 });
            // Mesh LOD as in the forest scene: trees near the camera get their mesh parts as
            // instances (dissolving against the impostor inside the band).
            let mut states = std::collections::HashMap::new();
            let t2 = Instant::now();
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
                    let v = crate::globe::forest::rec_variant(r[12]);
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
                    let data = (rec & 0xFF_FFFF) | ((v as u32) << INST_VARIANT_SHIFT);
                    let iflags = if state == 1 { INST_HAS_MESH } else { 0 } | ((r[12] >> 21) & 15) << 1 | ((v as u32 >> 8) & 1) << INST_VARIANT_HI;
                    for (pi, p) in pack.trees[v].parts.iter().enumerate() {
                        let alpha = catalog.mesh_alpha[p.mesh as usize];
                        let (flags, sbt) = if state == 2 && !alpha { (mesh_flags, 0) } else { (band_flags, 2) };
                        let sbt = if catalog.is_building_mesh(p.mesh as usize) { if sbt == 0 { SBT_BUILDING } else { SBT_BUILDING_BAND } } else { sbt };
                        insts.push(accel::make_instance(&(plant * p.xform), p.mesh | seed, 1, sbt, flags, blas[p.mesh as usize].addr));
                        inst_data.push(InstInfo { center: center.to_array(), data, flags: iflags | (pi as u32) << INST_PART_SHIFT });
                    }
                });
            }
            // the aircraft: one instance per part, the moving surfaces turned about their hinges
            if let Some(pose) = &self.ac_pose {
                let opaque = vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE | vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE;
                let seen_through = vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE | vk::GeometryInstanceFlagsKHR::FORCE_NO_OPAQUE;
                for (k, part) in self.ac_parts.iter().enumerate() {
                    let mesh = self.ac_first + k;
                    let m = pose.xform * pose.part(part);
                    // (mask 4, its own bit: the rays its shock waves bend look past it, main.rgen)
                    self.insts.push(accel::make_instance(&m, mesh as u32, 4, 4, if part.glass { seen_through } else { opaque }, self.blas[mesh].addr));
                    self.inst_data.push(InstInfo { center: [0.0, -1e9, 0.0], data: 0, flags: 0 });
                }
            }
            globe_cpu = [p[0], p[1], t2.elapsed().as_secs_f64() * 1e3];
            globe_states = states;
        }
        let d = self.gpu.device.clone();
        let t_wait = Instant::now();
        unsafe { d.wait_for_fences(&[self.fence], true, u64::MAX).context("waiting for previous frame")? };
        self.profile.cpu_wait += t_wait.elapsed().as_secs_f64() * 1000.0;
        // (the frame that may still have traced them is done; this frame's TLAS is without them)
        for mut f in std::mem::take(&mut self.near_retire) {
            accel::destroy_accel(&self.gpu, &mut f.accel);
            self.gpu.destroy_buffer(&mut f.input);
        }
        if self.globe.is_some() && self.insts.len() > self.tlas.capacity as usize {
            // dense understory, rocks and near tree meshes can outgrow the first estimate
            let cap = (self.insts.len() as f32 * 1.5) as u32;
            println!("Growing TLAS capacity to {cap} instances");
            unsafe { self.gpu.device.device_wait_idle()? };
            let mut old = std::mem::replace(&mut self.tlas, Tlas::new(&self.gpu, cap)?);
            old.destroy(&self.gpu);
            let mut old_info = std::mem::replace(&mut self.inst_info, Self::create_inst_info(&self.gpu, cap)?);
            self.gpu.destroy_buffer(&mut old_info);
            self.write_set(self.main_set, self.tlas.accel.raw, None);
        }
        if self.globe.is_some() {
            let (_, h) = self.extent();
            self.update_atlases(2.0 * globals.cam_up[3] / h as f32)?;
        }
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

        if let (Some(scene), Some((_, _, mut gg))) = (self.globe.as_mut(), self.globe_view) {
            // uploads and BLAS builds of finished tiles (recorded for this frame), tile table
            let up = scene.update(&self.gpu)?;
            let states = std::mem::take(&mut globe_states);
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
            for (i, v) in [up[0], up[1], globe_cpu[0], globe_cpu[1], globe_cpu[2]].into_iter().enumerate() {
                self.profile.globe[i] += v;
                self.profile.globe_max[i] = self.profile.globe_max[i].max(v);
            }
            let a = scene.index_addr();
            // z: smooth tree thinning on (Globals.cloud.z; globe.rchit's canopy colour follows it)
            gg.addr = [a as u32, (a >> 32) as u32, (g.cloud[2] > 0.5) as u32, 0];
            gg.params[1] = self.lod_frame as f32;
            gg.params[2] = g.iparams[1] as f32;
            gg.params[3] = f32::from_bits(self.ground_layers);
            gg.misc = [
                g.cam_fwd[3],
                crate::globe::forest::NEAR as f32,
                (crate::globe::forest::SPLIT * 0.5 * std::f64::consts::PI * crate::globe::tiles::R_EARTH
                    / (1u64 << crate::globe::forest::TREE_LEVEL) as f64) as f32,
                scene.range as f32,
            ];
            if let Some((ecef, rot, _)) = self.globe_view {
                if let Some(c) = crate::globe::city::shader_city(ecef) {
                    let v = |x: glam::DVec3| {
                        let r = (rot * x).as_vec3();
                        [r.x, r.y, r.z, 0.0]
                    };
                    (gg.city_o, gg.city_e, gg.city_n, gg.city_p, gg.city_cores) = (v(c.centre - ecef), v(c.east), v(c.north), c.params, c.cores);
                }
            }
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
            if let Some(scene) = self.globe.as_mut() {
                // this frame's streamed tiles: copies and BLAS builds before the TLAS build
                scene.record(&self.gpu, cb);
            }
            self.record_atlas_uploads(cb);
            self.record_near_builds(cb);
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
                let push: [u32; 4] = [1.0f32.to_bits(), 1.0f32.to_bits(), (globals.params[2] * self.fade).to_bits(), if globals.iparams[1] != 0 { 0 } else if globals.city[2] > 0.5 { 2 } else { 1 }];
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
                (globals.params[2] * self.fade).to_bits(),
                if globals.iparams[1] != 0 { 0 } else if globals.city[2] > 0.5 { 2 } else { 1 },
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
        if let Some(scene) = &self.globe {
            // globe: trees drawn as meshes, all trees in the TLAS (window title)
            stats.near_trees = self.near_count;
            stats.instances = self.insts.len();
            self.globe_trees = scene.trees_drawn;
        }
        Ok((stats, image))
    }
}

/// Per plant record (rt.glsl PlantRec): object->world 3x4 rows, variant | seed << 8 | ...
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PlantRecGpu {
    rows: [f32; 12],
    data: u32,
}

impl PlantRecGpu {
    /// From the plant's object -> world transform.
    pub fn new(m: glam::Mat3, pos: Vec3, data: u32) -> Self {
        let rows = [
            m.x_axis.x, m.y_axis.x, m.z_axis.x, pos.x, //
            m.x_axis.y, m.y_axis.y, m.z_axis.y, pos.y, //
            m.x_axis.z, m.y_axis.z, m.z_axis.z, pos.z,
        ];
        PlantRecGpu { rows, data }
    }
}

struct Clusters {
    /// plant -> record, record -> plant
    rec_of: Vec<u32>,
    tree_of: Vec<u32>,
    rec: Buffer,
    state: Buffer,
    blas: Vec<Accel>,
    /// always in the TLAS: the terrain and the cells without draw-distance plants
    insts: Vec<vk::AccelerationStructureInstanceKHR>,
    near: Vec<NearCell>,
    near_scratch: Buffer,
}

/// A cell that has plants with a draw distance. While the camera is within `cull` of the box of
/// their centres the TLAS holds `full`, a BLAS of all the cell's plants (records first..first +
/// count); otherwise `far`, the permanent BLAS of its plants without a draw distance (if any).
struct NearCell {
    first: u32,
    count: u32,
    far: Option<vk::AccelerationStructureInstanceKHR>,
    lo: Vec3,
    hi: Vec3,
    cull: f32,
    on: bool,
    full: Option<NearBlas>,
}

struct NearBlas {
    accel: Accel,
    /// its boxes (build input)
    input: Buffer,
}

/// A cell's TLAS instance. Mask bit 1: impostors; hit record 1: impostor intersection + closest
/// hit; custom index: the cell's first plant record.
fn cluster_instance(first: u32, blas: u64) -> vk::AccelerationStructureInstanceKHR {
    accel::make_instance(&glam::Affine3A::IDENTITY, first, 2, 1, vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE, blas)
}

/// The world box of a plant's impostor: the box of its turned bounding box.
fn plant_box(t: &crate::world::TreeInstance, asset: &crate::asset::TreeAsset) -> [f32; 6] {
    let (lo, hi) = (asset.bmin - Vec3::splat(0.05), asset.bmax + Vec3::splat(0.05));
    let (mut mn, mut mx) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for corner in 0..8 {
        let v = Vec3::new(if corner & 1 == 0 { lo.x } else { hi.x }, if corner & 2 == 0 { lo.y } else { hi.y }, if corner & 4 == 0 { lo.z } else { hi.z });
        let w = t.xform.transform_point3(v);
        mn = mn.min(w);
        mx = mx.max(w);
    }
    [mn.x, mn.y, mn.z, mx.x, mx.y, mx.z]
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
