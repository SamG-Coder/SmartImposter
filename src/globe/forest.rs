//! Trees on the globe: the forest scene's baked impostors (impcache.rs) placed per 300 m terrain
//! tile wherever ESA WorldCover has tree cover, standing on the Copernicus / ETOPO terrain.
//! Every tree is one AABB of a per-tile cluster BLAS, drawn by impostor.rint / impostor.rchit.

use super::dem::Terrain;
use super::tiles::{face_dir, lat_lon, TileId, R_EARTH};
use crate::gpu::{Gpu, Image};
use crate::impcache;
use crate::renderer::ImpostorInfo;
use anyhow::Result;
use ash::vk;
use glam::DVec3;
use gpu_allocator::MemoryLocation;

/// Finest tree tiles: this quadtree level (~305 m on a side), every tree.
pub const TREE_LEVEL: u8 = 15;
/// Coarsest tree tiles (~39 km).
pub const TREE_LEVEL_MIN: u8 = 8;
/// Every tree out to this distance (m): beyond it a crown is ~1 pixel.
pub const NEAR: f64 = 6_000.0;
/// Further out a tree tile is used while it is more than SPLIT edges away. A tile `k` levels coarser
/// than TREE_LEVEL keeps 1 / 4^k of the trees (same candidates per tile): the forest thins with
/// distance at a constant density on screen, the trees keep their true size, and the terrain shader
/// adds the canopy of the dropped ones as the pixel average (globe.rchit).
pub const SPLIT: f64 = 8.0;
/// Most trees in one tile (~3000 candidates), and plant records on the GPU in all (the instance
/// custom index addressing them has 24 bits).
pub const MAX_PER_TILE: usize = 4_096;
pub const CAPACITY: u32 = 16_000_000;
/// Candidate spacing (m) on the finest tiles: a closed stand reaches ~300 trees / ha.
const SPACING: f64 = 5.5;

/// How far the trees go: the horizon for the eye height (sqrt(2 R h)) plus the distance at which
/// a 1000 m high ridge still shows above it.
pub fn tree_range(eye_height: f64) -> f64 {
    let h = eye_height.max(1.0);
    ((2.0 * R_EARTH * h + h * h).sqrt() + (2.0 * R_EARTH * 1000.0).sqrt()).min(400_000.0)
}

pub struct Impostors {
    pub albedo: Vec<Image>,
    pub normal: Vec<Image>,
    pub depth: Vec<Image>,
    pub vis: Vec<Image>,
    pub infos: Vec<ImpostorInfo>,
    /// tree species -> variant indices, tallest first
    pub species: Vec<(String, Vec<usize>)>,
}

/// Load the baked atlases written by the forest scene and upload them (BC3 / BC5 blocks per mip).
pub fn load_impostors(gpu: &Gpu) -> Result<Impostors> {
    let t = std::time::Instant::now();
    let vars = impcache::read(std::path::Path::new(impcache::CACHE_PATH))?;
    let usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST;
    let mut im = Impostors { albedo: vec![], normal: vec![], depth: vec![], vis: vec![], infos: vec![], species: vec![] };
    let d = &gpu.device;
    for (vi, v) in vars.iter().enumerate() {
        let size = v.info.grid * v.info.res;
        let mips = v.info.mips;
        let imgs = [
            gpu.create_image(size, size, mips, vk::Format::BC3_UNORM_BLOCK, usage, "globe albedo atlas")?,
            gpu.create_image(size, size, mips, vk::Format::BC5_UNORM_BLOCK, usage, "globe normal atlas")?,
            gpu.create_image(size, size, mips, vk::Format::BC5_UNORM_BLOCK, usage, "globe depth atlas")?,
            gpu.create_image(size / 2, size / 2, (mips - 1).max(1), vk::Format::BC3_UNORM_BLOCK, usage, "globe vis atlas")?,
        ];
        let mut staging = gpu.create_buffer(v.blocks.len() as u64, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu, "atlas staging")?;
        staging.mapped()[..v.blocks.len()].copy_from_slice(&v.blocks);
        gpu.one_time(|cb| unsafe {
            for img in &imgs {
                gpu.image_barrier(
                    cb,
                    img.raw,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::PipelineStageFlags2::NONE,
                    vk::AccessFlags2::NONE,
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_WRITE,
                );
            }
            for j in &v.jobs {
                let region = vk::BufferImageCopy {
                    buffer_offset: j[3] as u64,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: j[1], base_array_layer: 0, layer_count: 1 },
                    image_offset: vk::Offset3D::default(),
                    image_extent: vk::Extent3D { width: j[2], height: j[2], depth: 1 },
                };
                d.cmd_copy_buffer_to_image(cb, staging.raw, imgs[j[0] as usize].raw, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region]);
            }
            for img in &imgs {
                gpu.image_barrier(
                    cb,
                    img.raw,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                    vk::AccessFlags2::SHADER_READ,
                );
            }
        })?;
        gpu.destroy_buffer(&mut staging);
        let [a, n, dp, vs] = imgs;
        im.albedo.push(a);
        im.normal.push(n);
        im.depth.push(dp);
        im.vis.push(vs);
        im.infos.push(v.info);
        if v.kind == 0 {
            match im.species.iter_mut().find(|s| s.0 == v.species) {
                Some(s) => s.1.push(vi),
                None => im.species.push((v.species.clone(), vec![vi])),
            }
        }
    }
    for s in &mut im.species {
        s.1.sort_by(|a, b| vars[*b].height.partial_cmp(&vars[*a].height).unwrap());
    }
    println!(
        "Globe trees: {} impostor variants, {} tree species ({}) ({:.1}s)",
        vars.len(),
        im.species.len(),
        im.species.iter().map(|s| s.0.trim_start_matches("Tree_").to_string()).collect::<Vec<_>>().join(", "),
        t.elapsed().as_secs_f32()
    );
    Ok(im)
}

pub struct TreeTile {
    pub id: TileId,
    pub center: DVec3,
    /// rt.glsl PlantRec: object -> tile 3x4 rows, then variant | seed << 8 and padding (16 x 4 bytes)
    pub recs: Vec<[u32; 16]>,
    /// tile-local AABB per tree
    pub aabbs: Vec<[f32; 6]>,
}

fn hash(a: u64) -> u64 {
    let mut x = a.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 31;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^ (x >> 29)
}

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f64 {
        self.0 = hash(self.0.wrapping_add(0x632B_E59B_D9B4_E019));
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Which species groups the catalog has: indices into `Impostors::species` by role.
pub struct Roles {
    beech: Option<usize>,
    alder: Option<usize>,
    poplar: Option<usize>,
    aspen: Option<usize>,
    hazel: Option<usize>,
    any: Vec<usize>,
}

impl Roles {
    pub fn new(im: &Impostors) -> Self {
        let find = |k: &str| im.species.iter().position(|s| s.0.contains(k));
        Roles {
            beech: find("Beech"),
            alder: find("Alder"),
            poplar: find("Poplar"),
            aspen: find("Aspen"),
            hazel: find("Hazel"),
            any: (0..im.species.len()).collect(),
        }
    }
}

/// Grow the trees of one tree tile: candidates every SPACING m (jittered), kept with the
/// probability of the WorldCover tree fraction around them, species from the site (water nearby,
/// altitude), size class weighted towards mature trees.
pub fn grow_tile(id: TileId, terrain: &Terrain, im: &Impostors, roles: &Roles) -> TreeTile {
    let center = id.center_dir() * R_EARTH;
    let mut tile = TreeTile { id, center, recs: Vec::new(), aabbs: Vec::new() };
    if roles.any.is_empty() || terrain.cover.is_empty() {
        return tile;
    }
    let (u0, v0, s) = id.uv();
    // same number of candidates on every level: coarser tiles keep 1 / 4^k of the trees
    let spacing = SPACING * (1u64 << (TREE_LEVEL - id.level)) as f64;
    let n = (id.edge() / spacing).ceil() as usize;
    let mut rng = Rng(hash(((id.face as u64) << 56) ^ ((id.level as u64) << 48) ^ ((id.x as u64) << 24) ^ id.y as u64));
    for j in 0..n {
        for i in 0..n {
            let (fu, fv) = ((i as f64 + rng.f()) / n as f64, (j as f64 + rng.f()) / n as f64);
            let r_keep = rng.f();
            let d = face_dir(id.face, u0 + fu * s, v0 + fv * s);
            let (lat, lon) = lat_lon(d);
            let Some(f) = terrain.cover.fractions(lat, lon, 30.0) else { continue };
            let tree = f[super::landcover::TREE];
            if tree <= 0.0 || r_keep > tree as f64 * 0.95 {
                continue;
            }
            let h = terrain.height(lat, lon).max(0.0) as f64;
            // site: wet ground near water, altitude
            let wet = terrain.cover.fractions(lat, lon, 60.0).map(|g| g[super::landcover::WATER] + g[8]).unwrap_or(0.0);
            let alt = h as f32;
            let r = rng.f() as f32;
            let lowland = 1.0 - ((alt - 900.0) / 600.0).clamp(0.0, 1.0);
            let mut w: Vec<(Option<usize>, f32)> = vec![
                (roles.beech, 0.55 * (0.3 + 0.7 * lowland) * (1.0 - wet)),
                (roles.aspen, 0.22 + 0.35 * (1.0 - lowland)),
                (roles.alder, 0.05 + 2.5 * wet),
                (roles.poplar, (0.06 + 0.8 * wet) * lowland),
                (roles.hazel, 0.12 * lowland),
            ];
            w.retain(|x| x.0.is_some() && x.1 > 0.0);
            let sp = if w.is_empty() {
                roles.any[(r * roles.any.len() as f32) as usize % roles.any.len()]
            } else {
                let total: f32 = w.iter().map(|x| x.1).sum();
                let mut acc = 0.0;
                let mut pick = w[0].0.unwrap();
                for (sp, wt) in &w {
                    acc += wt / total;
                    if r <= acc {
                        pick = sp.unwrap();
                        break;
                    }
                }
                pick
            };
            let vars = &im.species[sp].1;
            let k = ((rng.f().powf(1.7) * vars.len() as f64) as usize).min(vars.len() - 1); // mostly mature
            let vid = vars[k];
            let info = &im.infos[vid];
            // frame: up = local vertical, random yaw
            let up = d;
            let east = DVec3::Z.cross(up).normalize();
            let north = up.cross(east);
            let a = rng.f() * std::f64::consts::TAU;
            let x = east * a.cos() + north * a.sin();
            let z = x.cross(up);
            let sc = 0.85 + 0.3 * rng.f();
            let (x, y, z) = (x * sc, up * sc, z * sc);
            let t = d * (R_EARTH + h - 0.2) - center;
            let rows = [[x.x, y.x, z.x, t.x], [x.y, y.y, z.y, t.y], [x.z, y.z, z.z, t.z]];
            let seed = 1 + (rng.f() * 8190.0) as u32;
            let mut rec = [0u32; 16];
            for (ri, row) in rows.iter().enumerate() {
                for (ci, v) in row.iter().enumerate() {
                    rec[ri * 4 + ci] = (*v as f32).to_bits();
                }
            }
            rec[12] = vid as u32 | seed << 8;
            // tile-local bounds of the transformed box
            let (bmin, bmax) = (info.bmin, info.bmax);
            let mut lo = [f32::MAX; 3];
            let mut hi = [f32::MIN; 3];
            for c in 0..8 {
                let p = [
                    if c & 1 == 0 { bmin[0] } else { bmax[0] } as f64,
                    if c & 2 == 0 { bmin[1] } else { bmax[1] } as f64,
                    if c & 4 == 0 { bmin[2] } else { bmax[2] } as f64,
                ];
                for (k, row) in rows.iter().enumerate() {
                    let v = (row[0] * p[0] + row[1] * p[1] + row[2] * p[2] + row[3]) as f32;
                    lo[k] = lo[k].min(v);
                    hi[k] = hi[k].max(v);
                }
            }
            if tile.recs.len() >= MAX_PER_TILE {
                break;
            }
            tile.recs.push(rec);
            tile.aabbs.push([lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]]);
        }
    }
    tile
}

/// Tree tiles from the camera to `range` (the horizon), nearest first: TREE_LEVEL within NEAR,
/// then coarser tiles further out (quadtree descent on all six faces, over-the-horizon tiles
/// skipped).
pub fn tiles_near(cam: DVec3, range: f64) -> Vec<(f64, TileId)> {
    fn visit(id: TileId, cam: DVec3, range: f64, out: &mut Vec<(f64, TileId)>) {
        let d = super::tiles::tile_distance(id, cam);
        if d > range || (id.level > 2 && super::tiles::over_horizon(id, cam)) {
            return;
        }
        let split = id.level < TREE_LEVEL && (id.level < TREE_LEVEL_MIN || d < NEAR || d < SPLIT * id.edge());
        if split {
            for k in id.children() {
                visit(k, cam, range, out);
            }
        } else {
            out.push((d, id));
        }
    }
    let mut out = Vec::new();
    for f in 0..6 {
        visit(TileId::root(f), cam, range, &mut out);
    }
    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    out
}

/// Inverse of tiles::face_dir.
pub fn dir_to_face_uv(d: DVec3) -> (u8, f64, f64) {
    let (ax, ay, az) = (d.x.abs(), d.y.abs(), d.z.abs());
    let (face, a, b) = if ax >= ay && ax >= az {
        if d.x > 0.0 {
            (0, d.y / d.x, d.z / d.x)
        } else {
            (1, d.y / d.x, -d.z / d.x)
        }
    } else if ay >= az {
        if d.y > 0.0 {
            (2, -d.x / d.y, d.z / d.y)
        } else {
            (3, -d.x / d.y, -d.z / d.y)
        }
    } else if d.z > 0.0 {
        (4, d.y / d.z, -d.x / d.z)
    } else {
        (5, -d.y / d.z, -d.x / d.z)
    };
    let k = 4.0 / std::f64::consts::PI;
    (face, a.atan() * k, b.atan() * k)
}
