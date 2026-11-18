//! Trees on the globe: the renderer's plant variants placed per terrain tile wherever ESA
//! WorldCover has tree cover, standing on the Copernicus / ETOPO terrain. Every tree is one AABB
//! of a per-tile cluster BLAS, drawn by impostor.rint / impostor.rchit (scene.rs).

use super::biomes as b;
use super::dem::Terrain;
use super::tiles::{face_dir, lat_lon, TileId, R_EARTH};
use crate::renderer::ImpostorInfo;
use glam::DVec3;

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

/// What the tree growth needs from the renderer's catalog: every variant's impostor bounds, and
/// the tree species with their variants (tallest first).
#[derive(Clone, Default)]
pub struct Species {
    pub infos: Vec<ImpostorInfo>,
    pub species: Vec<(String, Vec<usize>)>,
    /// understory variants: grass patches (meadow, dry, wet), ferns, shrubs (heath, other)
    pub grass: Vec<usize>,
    pub dry_grass: Vec<usize>,
    pub wet_grass: Vec<usize>,
    pub ferns: Vec<usize>,
    pub heath: Vec<usize>,
    pub shrubs: Vec<usize>,
    /// dry land (desert shrubs, yuccas, cacti: on bare ground too), rocks, deadwood (logs, snags)
    pub dry: Vec<usize>,
    pub rocks: Vec<usize>,
    pub deadwood: Vec<usize>,
    /// per variant: the biomes it grows in (bit = biomes.rs number; bit 0 = no biome data)
    pub biomes: Vec<u16>,
}

/// Biomes an understory species belongs to (bit per biomes.rs number), by catalog name: desert
/// shrubs stay out of the Alps, bilberry and anemone in temperate and boreal forests. Unknown
/// names grow everywhere.
pub fn understory_biomes(name: &str) -> u16 {
    let bits = |b: &[u8]| b.iter().fold(1u16, |m, x| m | 1 << x);
    let n = name.to_ascii_lowercase();
    if n.contains("cact") {
        bits(&[b::DESERT])
    } else if n.contains("yucca") {
        bits(&[b::TEMPERATE_GRASSLAND, b::MEDITERRANEAN, b::DESERT])
    } else if n.contains("wheat") {
        bits(&[b::TEMPERATE_BROADLEAF, b::TEMPERATE_CONIFER, b::SAVANNA, b::TEMPERATE_GRASSLAND, b::MEDITERRANEAN])
    } else if n.contains("desert_rock") {
        bits(&[b::SAVANNA, b::MEDITERRANEAN, b::DESERT])
    } else if n.contains("dead_shrub") {
        bits(&[b::SAVANNA, b::TEMPERATE_GRASSLAND, b::MEDITERRANEAN, b::DESERT])
    } else if n.contains("palm") || n.contains("xanadu") {
        bits(&[b::TROPICAL_MOIST, b::TROPICAL_DRY, b::TROPICAL_CONIFER, b::SAVANNA, b::MANGROVES])
    } else if n.contains("leather_fern") {
        bits(&[b::TROPICAL_MOIST, b::TROPICAL_DRY, b::TROPICAL_CONIFER, b::TEMPERATE_BROADLEAF, b::MANGROVES])
    } else if n.contains("tundra_grass") {
        bits(&[b::BOREAL, b::MONTANE, b::TUNDRA])
    } else if n.contains("ribbon") || n.contains("poppy") || n.contains("dandelion") {
        bits(&[b::TEMPERATE_BROADLEAF, b::TEMPERATE_CONIFER, b::TEMPERATE_GRASSLAND, b::MEDITERRANEAN])
    } else if n.contains("marram") {
        bits(&[b::MEDITERRANEAN])
    } else if n.contains("reed") {
        bits(&[b::TROPICAL_MOIST, b::TROPICAL_DRY, b::TROPICAL_CONIFER, b::TEMPERATE_BROADLEAF, b::SAVANNA, b::TEMPERATE_GRASSLAND, b::FLOODED, b::MEDITERRANEAN, b::MANGROVES])
    } else if n.contains("boulder") {
        bits(&[b::TEMPERATE_BROADLEAF, b::TEMPERATE_CONIFER, b::BOREAL, b::MONTANE, b::TUNDRA])
    } else if n.contains("logs") {
        bits(&[b::TEMPERATE_BROADLEAF, b::TEMPERATE_CONIFER, b::BOREAL, b::MONTANE])
    } else if n.contains("snag") {
        bits(&[b::TEMPERATE_CONIFER, b::BOREAL, b::MONTANE, b::TUNDRA])
    } else if n.contains("greasewood") {
        bits(&[b::SAVANNA, b::TEMPERATE_GRASSLAND, b::MEDITERRANEAN, b::DESERT])
    } else if n.contains("bilberry") || n.contains("lingonberry") {
        bits(&[b::TEMPERATE_BROADLEAF, b::TEMPERATE_CONIFER, b::BOREAL, b::MONTANE, b::TUNDRA])
    } else if n.contains("anemone") || n.contains("spindle") {
        bits(&[b::TEMPERATE_BROADLEAF, b::TEMPERATE_CONIFER])
    } else if n.contains("heather") {
        bits(&[b::TEMPERATE_BROADLEAF, b::TEMPERATE_CONIFER, b::BOREAL, b::TEMPERATE_GRASSLAND, b::MONTANE, b::TUNDRA])
    } else if n.contains("huckleberry") {
        bits(&[b::TEMPERATE_BROADLEAF, b::TEMPERATE_CONIFER, b::MEDITERRANEAN])
    } else if n.contains("bracken") || n.contains("fern") {
        bits(&[b::TROPICAL_MOIST, b::TROPICAL_DRY, b::TROPICAL_CONIFER, b::TEMPERATE_BROADLEAF, b::TEMPERATE_CONIFER, b::BOREAL])
    } else if n.contains("cotton") {
        bits(&[b::TEMPERATE_CONIFER, b::BOREAL, b::FLOODED, b::MONTANE, b::TUNDRA])
    } else if n.contains("dry_grass") || n.contains("dry grass") {
        bits(&[b::SAVANNA, b::TEMPERATE_GRASSLAND, b::MONTANE, b::MEDITERRANEAN, b::DESERT])
    } else {
        u16::MAX
    }
}

/// How dry a biome's plants look (0..15; rt.glsl plantDry): desert scrub is dusty grey-olive,
/// Mediterranean, savanna and steppe plants in between, moist biomes keep their colour.
pub fn dryness(biome: u8) -> u32 {
    match biome {
        b::DESERT => 15,
        b::SAVANNA | b::MEDITERRANEAN => 10,
        b::TEMPERATE_GRASSLAND => 7,
        b::TROPICAL_DRY => 5,
        _ => 0,
    }
}

/// Understory tiles: this quadtree level (~76 m), within UNDER_RADIUS of the camera.
pub const UNDER_LEVEL: u8 = 17;
pub const UNDER_RADIUS: f64 = 700.0;

/// Understory tiles around the camera (same quadtree, its own level), nearest first.
pub fn under_tiles(cam: DVec3) -> Vec<(f64, TileId)> {
    let d = cam.normalize();
    let ground = d * R_EARTH;
    let (face, u, v) = dir_to_face_uv(d);
    let cells = 1u32 << UNDER_LEVEL;
    let to_cell = |c: f64| (((c + 1.0) * 0.5 * cells as f64).floor() as i64).clamp(0, cells as i64 - 1);
    let (cx, cy) = (to_cell(u), to_cell(v));
    let edge = 0.5 * std::f64::consts::PI * R_EARTH / cells as f64;
    let k = (UNDER_RADIUS / edge).ceil() as i64 + 1;
    let mut out = Vec::new();
    if std::env::var("SI_NO_UNDER").is_ok() {
        return out; // profiling: no understory
    }
    for y in cy - k..=cy + k {
        for x in cx - k..=cx + k {
            if x < 0 || y < 0 || x >= cells as i64 || y >= cells as i64 {
                continue;
            }
            let id = TileId { face, level: UNDER_LEVEL, x: x as u32, y: y as u32 };
            let dist = (id.center_dir() * R_EARTH - ground).length();
            if dist < UNDER_RADIUS {
                out.push((dist, id));
            }
        }
    }
    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    out
}

/// Grass patches, ferns and shrubs of one understory tile from the 20 m land cover, as in the
/// forest scene: meadows and fields get grass (dry on cropland), forest floor gets ferns and a
/// few shrubs, shrubland heather and shrubs, wetland wet grass. Grass patches lie on the slope.
pub fn grow_understory(id: TileId, terrain: &Terrain, sp: &Species) -> TreeTile {
    let center = id.center_dir() * R_EARTH;
    let mut tile = TreeTile { id, center, recs: Vec::new(), aabbs: Vec::new() };
    if terrain.cover.is_empty() {
        return tile;
    }
    let (u0, v0, s) = id.uv();
    let edge = id.edge();
    let mut rng = Rng(hash(0xA5A5 ^ ((id.face as u64) << 56) ^ ((id.x as u64) << 24) ^ id.y as u64));
    // (spacing m, layer): 0 shrubs, 1 ferns, 2 grass, 3 deadwood, 4 rocks
    for (spacing, layer) in [(3.4f64, 0u32), (2.1, 1), (2.3, 2), (6.0, 3), (5.0, 4)] {
        let n = (edge / spacing).ceil() as usize;
        for j in 0..n {
            for i in 0..n {
                let (fu, fv) = ((i as f64 + rng.f()) / n as f64, (j as f64 + rng.f()) / n as f64);
                let (r_keep, r_pick, r_size, r_yaw) = (rng.f(), rng.f(), rng.f(), rng.f());
                let d = face_dir(id.face, u0 + fu * s, v0 + fv * s);
                let (lat, lon) = lat_lon(d);
                let Some(g) = terrain.cover.class_group(lat, lon) else { continue };
                use super::landcover as lc;
                let biome = terrain.biome(lat, lon);
                // deserts and dry scrub: dry grass only, and far fewer plants (mostly bare ground)
                let arid = biome == b::DESERT;
                let dryish = matches!(biome, b::SAVANNA | b::MEDITERRANEAN);
                let (p, list): (f64, &Vec<usize>) = match (layer, g) {
                    (2, lc::GRASS) | (2, lc::SHRUB) | (2, lc::BARE) if arid => (0.12, &sp.dry_grass),
                    (2, lc::GRASS) | (2, lc::SHRUB) if dryish => (if g == lc::GRASS { 0.8 } else { 0.4 }, &sp.dry_grass),
                    (0, lc::SHRUB) if arid => (0.22, if r_pick < 0.5 && !sp.dry.is_empty() { &sp.dry } else { &sp.shrubs }),
                    (0, lc::BARE) if arid => (0.06, &sp.dry),
                    // deadwood on the forest floor (rocks: below, where the ground is rocky)
                    (3, lc::TREE) => (0.09, &sp.deadwood),
                    (3, lc::SHRUB) => (0.03, &sp.deadwood),
                    (4, lc::BARE) | (4, lc::MOSS) => (0.9, &sp.rocks),
                    (4, lc::SHRUB) | (4, lc::GRASS) | (4, lc::TREE) => (0.5, &sp.rocks),
                    // dry land plants: cacti, yuccas, dead shrubs (the biome masks keep them in the
                    // deserts and dry scrub)
                    (0, lc::BARE) => (0.12, &sp.dry),
                    (0, lc::SHRUB) if r_pick < 0.35 && !sp.dry.is_empty() => (0.55, &sp.dry),
                    (2, lc::GRASS) => (0.9, &sp.grass),
                    (2, lc::CROP) => (0.8, &sp.dry_grass),
                    (2, lc::WETLAND) => (0.85, &sp.wet_grass),
                    (2, lc::SHRUB) => (0.5, &sp.grass),
                    (2, lc::BARE) | (2, lc::MOSS) => (0.15, &sp.dry_grass),
                    (2, lc::TREE) => (0.06, &sp.grass),
                    (1, lc::TREE) => (0.45, &sp.ferns),
                    (1, lc::WETLAND) => (0.2, &sp.ferns),
                    (0, lc::SHRUB) => (0.55, if r_pick < 0.6 { &sp.heath } else { &sp.shrubs }),
                    (0, lc::TREE) => (0.12, &sp.shrubs),
                    (0, lc::GRASS) | (0, lc::MOSS) => (0.05, &sp.heath),
                    _ => continue,
                };
                // Rocks are not strewn evenly: they lie in fields (scree, moraine, rocky knolls) and
                // under steep ground, where the soil is thin
                if layer == 4 {
                    let m = 4.0 / 111_320.0;
                    let c = lat.to_radians().cos().max(0.05);
                    let sx = (terrain.height(lat, lon + m / c) - terrain.height(lat, lon - m / c)) as f64 / 8.0;
                    let sz = (terrain.height(lat + m, lon) - terrain.height(lat - m, lon)) as f64 / 8.0;
                    let steep = smooth(0.25, 0.6, sx.hypot(sz));
                    let p = d * R_EARTH;
                    let field = smooth(0.62, 0.8, vnoise3(p / 220.0) * 0.65 + vnoise3(p / 45.0 + DVec3::splat(7.0)) * 0.35);
                    if rng.f() > (field + steep * 0.7).min(1.0) {
                        continue;
                    }
                }
                let list = if list.is_empty() && layer < 3 && !(layer == 0 && g == lc::BARE) { if layer == 2 { &sp.grass } else { &sp.shrubs } } else { list };
                if list.is_empty() || r_keep > p {
                    continue;
                }
                // only what belongs in this biome
                let bit = 1u16 << biome.min(15);
                let ok: Vec<usize> = list.iter().copied().filter(|v| sp.biomes.get(*v).is_none_or(|m| m & bit != 0)).collect();
                if ok.is_empty() {
                    continue;
                }
                let vid = ok[((r_pick * 7.31).fract() * ok.len() as f64) as usize % ok.len()];
                let info = &sp.infos[vid];
                // rocks and logs: big ones are rarer (one 20 m ledge per many boulders), and the
                // big ledges are outcrops: only where the ground slopes
                if layer >= 3 {
                    let foot = ((info.bmax[0] - info.bmin[0]) * (info.bmax[2] - info.bmin[2])).max(0.01) as f64;
                    if rng.f() > (6.0 / foot).min(1.0) {
                        continue;
                    }
                    if foot > 20.0 {
                        let m = 4.0 / 111_320.0;
                        let c = lat.to_radians().cos().max(0.05);
                        let sx = (terrain.height(lat, lon + m / c) - terrain.height(lat, lon - m / c)) as f64 / 8.0;
                        let sz = (terrain.height(lat + m, lon) - terrain.height(lat - m, lon)) as f64 / 8.0;
                        if sx.hypot(sz) < 0.2 {
                            continue;
                        }
                    }
                }
                let h = terrain.height(lat, lon).max(0.0) as f64;
                let up = d;
                let east = DVec3::Z.cross(up).normalize();
                let north = up.cross(east);
                // grass patches, rocks and logs lie on the slope
                let mut y = up;
                if layer >= 2 {
                    let m = 1.0 / 111_320.0;
                    let dx = (terrain.height(lat, lon + 1.5 * m / lat.to_radians().cos()) - terrain.height(lat, lon - 1.5 * m / lat.to_radians().cos())) as f64 / 3.0;
                    let dz = (terrain.height(lat + 1.5 * m, lon) - terrain.height(lat - 1.5 * m, lon)) as f64 / 3.0;
                    y = (up - east * dx - north * dz).normalize();
                }
                let a = r_yaw * std::f64::consts::TAU;
                let x0 = east * a.cos() + north * a.sin();
                let x = (x0 - y * x0.dot(y)).normalize();
                let z = x.cross(y);
                let sc = 0.7 + 0.6 * r_size;
                let (x, y, z) = (x * sc, y * sc, z * sc);
                // rocks sink into the ground a little (their lowest point is the base)
                let sink = if layer == 4 { 0.08 + 0.2 * (info.bmax[1] - info.bmin[1]) as f64 * sc } else if layer == 3 { 0.03 + 0.1 * (info.bmax[1] - info.bmin[1]) as f64 * sc } else { 0.03 };
                let t = d * (R_EARTH + h - sink) - center;
                let rows = [[x.x, y.x, z.x, t.x], [x.y, y.y, z.y, t.y], [x.z, y.z, z.z, t.z]];
                let seed = 1 + (rng.f() * 8190.0) as u32;
                // rocks keep their colour
                let dry = if layer >= 3 { 0 } else { dryness(biome) };
                let mut rec = [0u32; REC_WORDS];
                for (ri, row) in rows.iter().enumerate() {
                    for (ci, v) in row.iter().enumerate() {
                        rec[ri * 4 + ci] = (*v as f32).to_bits();
                    }
                }
                rec[12] = vid as u32 | seed << 8 | dry << 21;
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
                    return tile;
                }
                tile.recs.push(rec);
                tile.aabbs.push([lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]]);
            }
        }
    }
    tile
}

/// Words per plant record (rt.glsl PlantRec: 3x4 rows, variant | seed << 8; 52 bytes).
pub const REC_WORDS: usize = 13;

pub struct TreeTile {
    pub id: TileId,
    pub center: DVec3,
    /// rt.glsl PlantRec: object -> tile 3x4 rows, then variant | seed << 8 and padding (16 x 4 bytes)
    pub recs: Vec<[u32; REC_WORDS]>,
    /// tile-local AABB per tree
    pub aabbs: Vec<[f32; 6]>,
}

/// How strongly the species fields concentrate a species (e^(+-CONTRAST/2) around its mean share).
const CONTRAST: f64 = 4.0;

/// Smooth value noise in 3D (Earth-fixed metres / scale), in [0, 1].
fn smooth(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn vnoise3(p: DVec3) -> f64 {
    let c = p.floor();
    let f = p - c;
    let u = f * f * (DVec3::splat(3.0) - 2.0 * f);
    let h = |x: f64, y: f64, z: f64| {
        let k = hash(((c.x + x) as i64 as u64).wrapping_mul(0x9E37) ^ ((c.y + y) as i64 as u64).wrapping_mul(0x85EB_CA6B) ^ ((c.z + z) as i64 as u64).wrapping_mul(0xC2B2_AE35));
        (k >> 11) as f64 / (1u64 << 53) as f64
    };
    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    let x00 = lerp(h(0.0, 0.0, 0.0), h(1.0, 0.0, 0.0), u.x);
    let x10 = lerp(h(0.0, 1.0, 0.0), h(1.0, 1.0, 0.0), u.x);
    let x01 = lerp(h(0.0, 0.0, 1.0), h(1.0, 0.0, 1.0), u.x);
    let x11 = lerp(h(0.0, 1.0, 1.0), h(1.0, 1.0, 1.0), u.x);
    lerp(lerp(x00, x10, u.y), lerp(x01, x11, u.y), u.z)
}

/// A smooth field (0..1) for `seed` at a point (ECEF, m) with local vertical `up` and altitude
/// `alt`: ~260 m across horizontally but ~70 m vertically, so its highs and lows run as bands
/// along the contours of a slope, not as blobs.
fn contour_field(p: DVec3, up: DVec3, alt: f64, seed: f64) -> f64 {
    let q = p / 260.0 + up * (alt * (1.0 / 70.0 - 1.0 / 260.0)) + DVec3::new(seed * 31.7, seed * 17.3, seed * 7.9);
    0.65 * vnoise3(q) + 0.35 * vnoise3(q * 2.3 + DVec3::splat(5.1))
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

/// Trees of each biome (biomes.rs numbers; index 0: no biome data), as catalog species names
/// (substrings) with weights. Species not downloaded are skipped, so a new Quixel species grows
/// where it belongs as soon as it is in the catalog; where none of a biome's are present, the
/// temperate list stands in. Stand-ins for species Quixel has no scan of follow docs/EARTH_PLAN.md
/// section 5 (e.g. birch for eucalyptus, oak for acacia and teak, poplar crowns for rainforest).
const BIOME_TREES: [&[(&str, f32)]; 15] = [
    /* 0 none     */ &[("Silver_Birch", 0.5), ("Norway_Spruce", 0.5)],
    // (broad, rounded crowns stand in for the rainforest canopy; no columnar poplars)
    /* 1 tropical moist */ &[("English_Oak", 0.3), ("European_Beech", 0.25), ("Norway_Maple", 0.15), ("Black_Alder", 0.12), ("Giant_Bamboo", 0.1), ("Common_Hazel", 0.08)],
    /* 2 tropical dry   */ &[("English_Oak", 0.45), ("Norway_Maple", 0.2), ("Giant_Bamboo", 0.15), ("Common_Hazel", 0.1), ("Baltic_Pine", 0.1)],
    /* 3 tropical conifer */ &[("Baltic_Pine", 0.6), ("English_Oak", 0.4)],
    /* 4 temperate broadleaf */ &[("European_Beech", 0.3), ("English_Oak", 0.22), ("Norway_Maple", 0.1), ("European_Hornbeam", 0.06), ("Silver_Birch", 0.08), ("European_Aspen", 0.07), ("Common_Hazel", 0.07), ("Rowan", 0.04), ("Norway_Spruce", 0.04), ("Elder", 0.02)],
    /* 5 temperate conifer */ &[("Norway_Spruce", 0.4), ("Baltic_Pine", 0.2), ("European_Beech", 0.16), ("Silver_Birch", 0.06), ("European_Aspen", 0.06), ("Rowan", 0.05), ("Norway_Maple", 0.04), ("Dead_Pine", 0.03)],
    /* 6 boreal   */ &[("Norway_Spruce", 0.38), ("Baltic_Pine", 0.3), ("Silver_Birch", 0.2), ("European_Aspen", 0.08), ("Rowan", 0.02), ("Dead_Pine", 0.02)],
    /* 7 savanna  */ &[("English_Oak", 0.45), ("Silver_Birch", 0.2), ("Aleppo_Pine", 0.15), ("Tropical_Palm", 0.1), ("Giant_Bamboo", 0.1)],
    /* 8 temperate grassland */ &[("English_Oak", 0.4), ("Black_Poplar", 0.25), ("Goat_Willow", 0.2), ("Silver_Birch", 0.15)],
    /* 9 flooded  */ &[("Goat_Willow", 0.45), ("Black_Alder", 0.3), ("Black_Poplar", 0.25)],
    /* 10 montane */ &[("Baltic_Pine", 0.45), ("Norway_Spruce", 0.3), ("Silver_Birch", 0.15), ("Rowan", 0.1)],
    /* 11 tundra  */ &[("Silver_Birch", 0.5), ("Goat_Willow", 0.3), ("Baltic_Pine", 0.2)],
    /* 12 Mediterranean */ &[("Aleppo_Pine", 0.5), ("English_Oak", 0.35), ("Baltic_Pine", 0.15)],
    /* 13 desert  */ &[("Tropical_Palm", 0.5), ("Aleppo_Pine", 0.3), ("Goat_Willow", 0.2)],
    /* 14 mangroves */ &[("Goat_Willow", 0.5), ("Black_Alder", 0.5)],
];
/// Wet ground (water nearby) favours these, in any biome where they are not out of place.
const WET_TREES: &[(&str, f32)] = &[("Black_Alder", 2.5), ("Goat_Willow", 1.5), ("Black_Poplar", 0.8)];
/// Conifers: they take over with altitude in the temperate and montane biomes (4, 5, 10).
const CONIFERS: &[&str] = &["Spruce", "Pine"];

/// The catalog species of each biome (indices into `Species::species`, weights, conifer?).
pub struct Roles {
    biome: Vec<Vec<(usize, f32, bool)>>,
    wet: Vec<(usize, f32)>,
}

impl Roles {
    pub fn new(im: &Species) -> Self {
        let find = |k: &str| im.species.iter().position(|s| s.0.contains(k) && !s.1.is_empty());
        let resolve = |list: &[(&str, f32)]| -> Vec<(usize, f32, bool)> {
            list.iter().filter_map(|(n, w)| find(n).map(|i| (i, *w, CONIFERS.iter().any(|c| n.contains(c))))).collect()
        };
        let mut biome: Vec<Vec<(usize, f32, bool)>> = BIOME_TREES.iter().map(|l| resolve(l)).collect();
        let temperate = biome[4].clone();
        let any: Vec<(usize, f32, bool)> = (0..im.species.len()).filter(|i| !im.species[*i].1.is_empty()).map(|i| (i, 1.0, false)).collect();
        for b in biome.iter_mut() {
            if b.is_empty() {
                *b = if temperate.is_empty() { any.clone() } else { temperate.clone() };
            }
        }
        let wet = WET_TREES.iter().filter_map(|(n, w)| find(n).map(|i| (i, *w))).collect();
        Roles { biome, wet }
    }

    fn is_empty(&self) -> bool {
        self.biome.iter().all(|b| b.is_empty())
    }

    /// A species for a tree at this site: the biome's mix, conifers higher up in the temperate
    /// and montane biomes, alders / willows / poplars on wet ground. `r` in [0, 1).
    fn pick(&self, biome: u8, alt: f32, wet: f32, cool: f32, field: &dyn Fn(usize) -> f32, r: f32) -> Option<usize> {
        let list = self.biome.get(biome as usize).unwrap_or(&self.biome[0]);
        let lowland = 1.0 - ((alt - 800.0) / 700.0).clamp(0.0, 1.0);
        let mountains = matches!(biome, b::TEMPERATE_BROADLEAF | b::TEMPERATE_CONIFER | b::MONTANE);
        let mut w: Vec<(usize, f32)> = list
            .iter()
            .map(|&(i, wt, conifer)| {
                let alt_k = if !mountains { 1.0 } else if conifer { 0.35 + 0.65 * (1.0 - lowland) } else { 0.3 + 0.7 * lowland };
                let aspect = if conifer { 1.0 + 0.7 * cool } else { 1.0 - 0.4 * cool };
                (i, wt * alt_k * aspect.max(0.1) * (1.0 - 0.6 * wet) * field(i))
            })
            .collect();
        if wet > 0.0 && biome != b::DESERT {
            for &(i, wt) in &self.wet {
                w.push((i, wt * wet * if biome == b::BOREAL || biome == b::TUNDRA { 0.5 } else { 1.0 } * field(i)));
            }
        }
        let total: f32 = w.iter().map(|x| x.1).sum();
        if total <= 0.0 {
            return list.first().map(|x| x.0);
        }
        let mut acc = 0.0;
        for (i, wt) in &w {
            acc += wt / total;
            if r <= acc {
                return Some(*i);
            }
        }
        w.last().map(|x| x.0)
    }
}

/// Grow the trees of one tree tile: candidates every SPACING m (jittered), kept with the
/// probability of the WorldCover tree fraction around them, species from the site (water nearby,
/// altitude), size class weighted towards mature trees.
pub fn grow_tile(id: TileId, terrain: &Terrain, im: &Species, roles: &Roles) -> TreeTile {
    let center = id.center_dir() * R_EARTH;
    let mut tile = TreeTile { id, center, recs: Vec::new(), aabbs: Vec::new() };
    if roles.is_empty() || terrain.cover.is_empty() {
        return tile;
    }
    let (u0, v0, s) = id.uv();
    // same number of candidates on every level: coarser tiles keep 1 / 4^k of the trees
    let spacing = SPACING * (1u64 << (TREE_LEVEL - id.level)) as f64;
    let n = (id.edge() / spacing).ceil() as usize;
    let mut rng = Rng(hash(((id.face as u64) << 56) ^ ((id.level as u64) << 48) ^ ((id.x as u64) << 24) ^ id.y as u64));
    // The slope's facing (cool slopes favour conifers) from a coarse grid of heights over the
    // tile: 17 x 17 samples instead of two extra 30 m lookups per tree.
    const G: usize = 17;
    let fine = id.level >= TREE_LEVEL - 1;
    let gdir: Vec<DVec3> = (0..G * G).map(|k| face_dir(id.face, u0 + s * (k % G) as f64 / (G - 1) as f64, v0 + s * (k / G) as f64 / (G - 1) as f64)).collect();
    let grid: Vec<f32> = gdir
        .iter()
        .map(|d| {
            let (la, lo) = lat_lon(*d);
            terrain.height_at(la, lo, spacing)
        })
        .collect();
    let gstep = id.edge() / (G - 1) as f64;
    // The species and age fields vary over 70-260 m: evaluated on the grid (per species, when a
    // tree first asks for it) and interpolated per tree, not ~20 noise lookups for every tree.
    let fields: std::cell::RefCell<Vec<Option<Vec<f32>>>> = std::cell::RefCell::new(vec![None; im.species.len() + 1]);
    let field_at = |slot: usize, seed: f64, fu: f64, fv: f64| -> f32 {
        let mut f = fields.borrow_mut();
        let g = f[slot].get_or_insert_with(|| {
            gdir.iter().zip(&grid).map(|(d, h)| contour_field(*d * (R_EARTH + *h as f64), *d, *h as f64, seed) as f32).collect()
        });
        let (x, y) = (fu * (G - 1) as f64, fv * (G - 1) as f64);
        let (x0, y0) = ((x as usize).min(G - 2), (y as usize).min(G - 2));
        let (tx, ty) = ((x - x0 as f64) as f32, (y - y0 as f64) as f32);
        let at = |i: usize, j: usize| g[j * G + i];
        let a = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * tx;
        let b = at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * tx;
        a + (b - a) * ty
    };
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
            let h = terrain.height_at(lat, lon, spacing).max(0.0) as f64;
            // site: wet ground near water, altitude
            let wet = terrain.cover.fractions(lat, lon, 60.0).map(|g| g[super::landcover::WATER] + g[8]).unwrap_or(0.0);
            let alt = h as f32;
            // Where species grow: each has a smooth field that runs along the contours, so a
            // species dominates bands of the slope and gives way gradually to its neighbours
            // (stands merge at their edges); cooler slopes (facing the pole) and steep ones favour
            // conifers. Each tree then draws from those shares.

            let cool = if fine {
                // the grid cell's gradient, in tile axes, turned to north: d(height) / d(north) < 0
                // means the slope faces the pole (in the north; the sign flips in the south)
                let (gx, gy) = (((fu * (G - 1) as f64) as usize).min(G - 2), ((fv * (G - 1) as f64) as usize).min(G - 2));
                let at = |x: usize, y: usize| grid[y * G + x] as f64;
                let du = (at(gx + 1, gy) + at(gx + 1, gy + 1) - at(gx, gy) - at(gx, gy + 1)) / (2.0 * gstep);
                let dv = (at(gx, gy + 1) + at(gx + 1, gy + 1) - at(gx, gy) - at(gx + 1, gy)) / (2.0 * gstep);
                let up = d;
                let east = DVec3::Z.cross(up).normalize();
                let north = up.cross(east);
                let eu = (face_dir(id.face, u0 + (fu + 0.01) * s, v0 + fv * s) - d).normalize();
                let ev = (face_dir(id.face, u0 + fu * s, v0 + (fv + 0.01) * s) - d).normalize();
                // gradient vector on the ground from its components along the tile axes
                let g = eu * du + ev * dv;
                let dn = g.dot(north);
                (-dn * 3.0 * lat.signum()).clamp(-1.0, 1.0) as f32
            } else {
                0.0
            };
            let field = |sp: usize| (CONTRAST as f32 * (field_at(sp, sp as f64, fu, fv) - 0.5)).exp();
            let biome = terrain.biome(lat, lon);
            let Some(sp) = roles.pick(biome, alt, wet, cool, &field, rng.f() as f32) else { continue };
            let vars = &im.species[sp].1;
            // variants run from old (large) to young: stands of an age, also as smooth bands
            let age = (field_at(im.species.len(), 101.0, fu, fv) as f64 * 1.3 - 0.15 + (rng.f() - 0.5) * 0.35).clamp(0.0, 0.999);
            let k = ((age.powf(1.7) * vars.len() as f64) as usize).min(vars.len() - 1); // mostly mature
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
            let mut rec = [0u32; REC_WORDS];
            for (ri, row) in rows.iter().enumerate() {
                for (ci, v) in row.iter().enumerate() {
                    rec[ri * 4 + ci] = (*v as f32).to_bits();
                }
            }
            // levels coarser than the finest (impostor.rint thins the trees on with distance)
            let k = (TREE_LEVEL - id.level).min(7) as u32;
            rec[12] = vid as u32 | seed << 8 | dryness(biome) << 21 | k << 25;
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
