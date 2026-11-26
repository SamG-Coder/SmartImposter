//! Cities on the globe: a street grid laid over a real place, its blocks cut into lots and every
//! lot given one of the procedural building archetypes (city.rs) by how central it is: towers
//! in the downtown cores, office and apartment blocks around them, shops with their parking,
//! then houses in rows facing their streets. Buildings go wherever the 10 m land cover says the
//! ground is built up and dry, so the real coast, bay and islands shape the city. Each building
//! is a plant record of the tree tiles (forest.rs): an impostor far away, its mesh near, like a
//! tree. Palms line the sidewalks the same way.
//!
//! The plan is a pure function of the place, made of integer hashes of the block and lot
//! numbers: any tile can work out its own buildings, the trees can ask whether a spot is street
//! or building, and the terrain shader (globe.rchit cityGround) draws the same streets, zones and
//! lots on the ground. `PLAN` below and that shader must stay in step.

use super::dem::Terrain;
use super::forest::{Species, TreeTile, MAX_PER_TILE, REC_WORDS, TREE_LEVEL};
use super::landcover as lc;
use super::tiles::{face_dir, lat_lon, TileId, R_EARTH};
use crate::city::Class;
use crate::renderer::ImpostorInfo;
use glam::DVec3;

/// A building archetype as the plan needs it (renderer.rs fills these from the catalog).
#[derive(Clone, Copy, Debug)]
pub struct ArchRef {
    pub variant: usize,
    pub class: Class,
    /// footprint (x, z) and height, m
    pub size: [f32; 2],
    pub height: f32,
}

struct Core {
    lat: f64,
    lon: f64,
    /// how tall it builds (0..1) and how far that reaches (m)
    strength: f64,
    radius: f64,
}

struct City {
    lat: f64,
    lon: f64,
    /// the plan reaches this far from the centre (m)
    radius: f64,
    cores: &'static [Core],
}

/// Most cores a city has (the shader's globals hold this many).
pub const MAX_CORES: usize = 6;

/// Miami: downtown and Brickell on the west shore of Biscayne Bay, Edgewater north of them, the
/// high-rises along Miami Beach across the bay.
const CITIES: &[City] = &[City {
    lat: 25.7800,
    lon: -80.1700,
    radius: 14_000.0,
    cores: &[
        Core { lat: 25.7745, lon: -80.1905, strength: 1.0, radius: 900.0 },   // downtown
        Core { lat: 25.7640, lon: -80.1925, strength: 1.0, radius: 800.0 },   // Brickell
        Core { lat: 25.7930, lon: -80.1875, strength: 0.75, radius: 900.0 },  // Edgewater
        Core { lat: 25.7790, lon: -80.1310, strength: 0.6, radius: 700.0 },   // South Beach
        Core { lat: 25.8100, lon: -80.1240, strength: 0.65, radius: 1200.0 }, // Mid-Beach
        Core { lat: 25.7520, lon: -80.2050, strength: 0.4, radius: 1200.0 },  // Coconut Grove side
    ],
}];

// ---- the plan (globe.rchit cityGround has the same numbers and rules)

/// Street lines every BX m east-west (the north-south streets, x = i BX) and BY m north-south.
pub const BX: f64 = 128.0;
pub const BY: f64 = 104.0;
/// Arterials: every 6th north-south street, every 5th east-west one.
const ART_X: i64 = 6;
const ART_Y: i64 = 5;

/// Half the width of a street (kerb to kerb plus sidewalks), and its sidewalk.
fn half_width(arterial: bool) -> f64 {
    if arterial {
        11.0
    } else {
        7.0
    }
}
fn sidewalk(arterial: bool) -> f64 {
    if arterial {
        4.0
    } else {
        3.0
    }
}
fn art_x(i: i64) -> bool {
    i.rem_euclid(ART_X) == 0
}
fn art_y(j: i64) -> bool {
    j.rem_euclid(ART_Y) == 0
}

/// The plan's integer hash (globe.rchit cHash).
fn ihash(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}
/// Hash of block (i, j) and a salt.
fn bh(i: i64, j: i64, salt: u32) -> u32 {
    ihash((i as i32 as u32).wrapping_mul(0x9E37_79B1) ^ ihash((j as i32 as u32).wrapping_add(salt.wrapping_mul(0x85EB_CA6B))))
}
/// ... as a number in [0, 1).
fn bu(i: i64, j: i64, salt: u32) -> f64 {
    (bh(i, j, salt) >> 8) as f64 / 16_777_216.0
}

/// Block zones: 0 houses, 1 shops and walk-ups, 2 office and apartment blocks, 3 towers.
fn zone(dens: f64) -> u32 {
    if dens > 0.5 {
        3
    } else if dens > 0.28 {
        2
    } else if dens > 0.1 {
        1
    } else {
        0
    }
}

/// The inside of block (i, j) (between the sidewalks): x0, y0, width, height.
fn interior(i: i64, j: i64) -> (f64, f64, f64, f64) {
    let x0 = i as f64 * BX + half_width(art_x(i));
    let x1 = (i + 1) as f64 * BX - half_width(art_x(i + 1));
    let y0 = j as f64 * BY + half_width(art_y(j));
    let y1 = (j + 1) as f64 * BY - half_width(art_y(j + 1));
    (x0, y0, x1 - x0, y1 - y0)
}

/// House lots: two rows back to back, each lot facing its street (south row: the street at y0,
/// north row: the street at y1), about 21 m wide. The house stands 6 m back from the sidewalk,
/// the driveway along one side.
const HOUSE_LOT: f64 = 21.0;
const FRONT_YARD: f64 = 6.0;

// ----

/// A city's plane: the tangent plane at its centre (sea level), x east, y north. The terrain
/// shader works out the same coordinates (globe.rchit cityGround), so streets and buildings agree.
struct Frame {
    centre: DVec3,
    east: DVec3,
    north: DVec3,
    /// the cores in the plane: x, y
    cores: Vec<(f64, f64)>,
}

fn frames() -> &'static [Frame] {
    static F: std::sync::OnceLock<Vec<Frame>> = std::sync::OnceLock::new();
    F.get_or_init(|| {
        CITIES
            .iter()
            .map(|c| {
                let up = super::tiles::dir_from_lat_lon(c.lat, c.lon);
                let east = DVec3::Z.cross(up).normalize();
                let north = up.cross(east);
                let centre = up * R_EARTH;
                let cores = c
                    .cores
                    .iter()
                    .map(|k| {
                        let v = super::tiles::dir_from_lat_lon(k.lat, k.lon) * R_EARTH - centre;
                        (v.dot(east), v.dot(north))
                    })
                    .collect();
                Frame { centre, east, north, cores }
            })
            .collect()
    })
}

fn plane(k: usize, d: DVec3) -> (f64, f64) {
    let f = &frames()[k];
    let v = d * R_EARTH - f.centre;
    (v.dot(f.east), v.dot(f.north))
}

/// The city whose plan covers lat / lon, and the point in its plane (m east, m north).
fn city_at(lat: f64, lon: f64) -> Option<(usize, f64, f64)> {
    CITIES.iter().enumerate().find_map(|(k, c)| {
        // (a quick test first)
        if (lat - c.lat).abs() > c.radius / 100_000.0 || (lon - c.lon).abs() * c.lat.to_radians().cos() > c.radius / 100_000.0 {
            return None;
        }
        let (x, y) = plane(k, super::tiles::dir_from_lat_lon(lat, lon));
        (x * x + y * y < c.radius * c.radius).then_some((k, x, y))
    })
}

fn to_lat_lon(k: usize, x: f64, y: f64) -> (f64, f64) {
    let f = &frames()[k];
    lat_lon(f.centre + f.east * x + f.north * y)
}

/// How central a place is: 1 in a downtown core, falling off with distance.
fn density(k: usize, x: f64, y: f64) -> f64 {
    CITIES[k]
        .cores
        .iter()
        .zip(&frames()[k].cores)
        .map(|(c, &(cx, cy))| {
            let d2 = ((x - cx).powi(2) + (y - cy).powi(2)) / (c.radius * c.radius);
            c.strength * (-d2).exp()
        })
        .fold(0.0, f64::max)
}

fn block_zone(k: usize, i: i64, j: i64) -> u32 {
    let (x0, y0, w, h) = interior(i, j);
    zone(density(k, x0 + w * 0.5, y0 + h * 0.5) + (bu(i, j, 1) - 0.5) * 0.12)
}

/// What the terrain shader needs to draw the plan (globe.rchit cityGround).
pub struct ShaderCity {
    pub centre: DVec3,
    pub east: DVec3,
    pub north: DVec3,
    /// block x, block y, radius, number of cores
    pub params: [f32; 4],
    /// cores: x, y in the plane, strength, radius
    pub cores: [[f32; 4]; MAX_CORES],
}

/// The city nearest the camera (ECEF), if it is near enough to matter.
pub fn shader_city(cam: DVec3) -> Option<ShaderCity> {
    let d = cam.normalize();
    let eye = (cam.length() - R_EARTH).max(0.0);
    let k = (0..CITIES.len())
        .filter(|&k| (d * R_EARTH - frames()[k].centre).length() < CITIES[k].radius * 3.0 + eye * 4.0)
        .min_by(|&a, &b| (d - frames()[a].centre.normalize()).length().total_cmp(&(d - frames()[b].centre.normalize()).length()))?;
    let (c, f) = (&CITIES[k], &frames()[k]);
    let mut cores = [[0.0f32; 4]; MAX_CORES];
    for (n, (core, &(x, y))) in c.cores.iter().zip(&f.cores).take(MAX_CORES).enumerate() {
        cores[n] = [x as f32, y as f32, core.strength as f32, core.radius as f32];
    }
    Some(ShaderCity { centre: f.centre, east: f.east, north: f.north, params: [BX as f32, BY as f32, c.radius as f32, c.cores.len().min(MAX_CORES) as f32], cores })
}

/// Terrain heights in a city: the 30 m elevation is a surface model (it has the real city's
/// buildings and trees in it, blurred); there the ground is the plain the city stands on, a few
/// metres above the sea. Water and the sea floor stay as they are.
pub fn ground_cap(lat: f64, lon: f64) -> Option<f32> {
    let (k, x, y) = city_at(lat, lon)?;
    let r = (x * x + y * y).sqrt() / CITIES[k].radius;
    // fade back to the data over the last fifth of the radius
    let t = ((r - 0.8) / 0.2).clamp(0.0, 1.0);
    Some(2.5 + 400.0 * (t * t) as f32)
}

/// One building of the plan: archetype, centre (plane m), yaw in quarter turns, scale.
struct Placed {
    arch: usize,
    x: f64,
    y: f64,
    quarter: u32,
    scale: f64,
    /// footprint half extents in the plane after the turn
    hx: f64,
    hy: f64,
}

/// Is the plane point on built-up, dry ground? `strict`: also needs enough built-up land.
fn buildable(k: usize, terrain: &Terrain, x: f64, y: f64) -> bool {
    let (la, lo) = to_lat_lon(k, x, y);
    !matches!(terrain.cover.class_group(la, lo), Some(lc::WATER) | Some(lc::WETLAND) | None)
}

/// The buildings of block (i, j) (land permitting). Only buildings at least `need` m tall are
/// worked out; every choice comes from hashes of the block and lot, so a lot has the same
/// building whoever asks.
fn block(k: usize, archs: &[ArchRef], terrain: &Terrain, i: i64, j: i64, need: f32) -> Vec<Placed> {
    let mut out = Vec::new();
    let (x0, y0, w, h) = interior(i, j);
    let z = block_zone(k, i, j);
    // (the zones below the cores hold nothing that tall)
    let tallest = |cls: &[Class]| archs.iter().filter(|a| cls.contains(&a.class)).map(|a| a.height).fold(0.0f32, f32::max);
    let classes: &[Class] = match z {
        3 => &[Class::Tower, Class::Tower, Class::Tower, Class::MidRise],
        2 => &[Class::MidRise, Class::MidRise, Class::Tower, Class::LowRise],
        1 => &[Class::LowRise, Class::LowRise, Class::MidRise],
        _ => &[Class::House],
    };
    if need > 0.0 && tallest(classes) < need {
        return out;
    }
    let dens = density(k, x0 + w * 0.5, y0 + h * 0.5);
    let built_at = |x: f64, y: f64| {
        let (la, lo) = to_lat_lon(k, x, y);
        terrain.cover.fractions(la, lo, 40.0).map(|f| f[4]).unwrap_or(0.0)
    };
    if z == 0 {
        // houses: two rows facing their streets
        let n = (w / HOUSE_LOT).floor().max(1.0) as usize;
        let lw = w / n as f64;
        let depth = h * 0.5;
        for row in 0..2 {
            for l in 0..n {
                let lot = (row * n + l) as u32;
                let drive_right = bu(i, j, 30 + lot) < 0.5;
                // the archetypes that fit: along the street at most the lot less the driveway,
                // at most 16 m deep
                let mut fits: Vec<(usize, u32)> = Vec::new();
                for (a, r) in archs.iter().enumerate() {
                    if r.class != Class::House {
                        continue;
                    }
                    let (ax, az) = (r.size[0] as f64, r.size[1] as f64);
                    if ax <= lw - 6.0 && az <= 16.0 {
                        fits.push((a, 0));
                    }
                    if az <= lw - 6.0 && ax <= 16.0 {
                        fits.push((a, 1));
                    }
                }
                if fits.is_empty() {
                    continue;
                }
                let (a, q) = fits[(bh(i, j, 10 + lot) as usize) % fits.len()];
                let r = &archs[a];
                if r.height < need {
                    continue;
                }
                let quarter = q + if bu(i, j, 20 + lot) < 0.5 { 2 } else { 0 };
                let scale = 0.94 + 0.08 * bu(i, j, 40 + lot);
                let (ax, az) = if quarter % 2 == 0 { (r.size[0] as f64, r.size[1] as f64) } else { (r.size[1] as f64, r.size[0] as f64) };
                let (hx, hy) = (ax * 0.5 * scale, az * 0.5 * scale);
                // along the street: centred in what the driveway leaves; back from the street
                let u = if drive_right { 1.5 + (lw - 6.0) * 0.5 } else { 4.5 + (lw - 6.0) * 0.5 };
                let v = FRONT_YARD + hy + 0.5;
                let px = x0 + l as f64 * lw + u;
                let py = if row == 0 { y0 + v } else { y0 + h - v };
                if built_at(px, py) < 0.25 {
                    continue;
                }
                let (ex, ey) = (hx + 1.0, hy + 1.0);
                if !(buildable(k, terrain, px, py) && buildable(k, terrain, px - ex, py - ey) && buildable(k, terrain, px + ex, py + ey) && buildable(k, terrain, px + ex, py - ey) && buildable(k, terrain, px - ex, py + ey)) {
                    continue;
                }
                let _ = depth;
                out.push(Placed { arch: a, x: px, y: py, quarter, scale, hx, hy });
            }
        }
        return out;
    }
    let (nx, ny) = match z {
        3 => {
            if bu(i, j, 2) < 0.35 {
                (1, 1)
            } else {
                (2, 1)
            }
        }
        2 => (2, 2),
        _ => (3, 2),
    };
    let (lw, lh) = (w / nx as f64, h / ny as f64);
    for ly in 0..ny {
        for lx in 0..nx {
            let lot = (ly * nx + lx) as u32;
            let class = classes[(bh(i, j, 10 + lot) as usize) % classes.len()];
            let (cx, cy) = (x0 + (lx as f64 + 0.5) * lw, y0 + (ly as f64 + 0.5) * lh);
            let mut fits: Vec<(usize, u32)> = Vec::new();
            for (a, r) in archs.iter().enumerate() {
                if r.class != class {
                    continue;
                }
                let (ax, az) = (r.size[0] as f64, r.size[1] as f64);
                if ax <= lw - 4.0 && az <= lh - 4.0 {
                    fits.push((a, 0));
                }
                if az <= lw - 4.0 && ax <= lh - 4.0 {
                    fits.push((a, 1));
                }
            }
            if fits.is_empty() {
                continue;
            }
            let pick = if class == Class::Tower {
                // taller towers nearer the core centres, and now and then a much taller one
                let mut want = 70.0 + 170.0 * dens.powf(1.3) * (0.55 + 0.6 * bu(i, j, 90 + lot));
                if bu(i, j, 95 + lot) < 0.15 {
                    want *= 1.6;
                }
                let mut best = fits[0];
                let mut bd = f64::MAX;
                for (n, &(a, q)) in fits.iter().enumerate() {
                    let d = (archs[a].height as f64 - want).abs() + bu(i, j, 200 + lot * 97 + n as u32) * 35.0;
                    if d < bd {
                        bd = d;
                        best = (a, q);
                    }
                }
                best
            } else {
                fits[(bh(i, j, 12 + lot) as usize) % fits.len()]
            };
            let r = &archs[pick.0];
            if r.height < need {
                continue;
            }
            let quarter = pick.1 + if bu(i, j, 20 + lot) < 0.5 { 2 } else { 0 };
            let (hx, hy) = if quarter % 2 == 0 { (r.size[0] as f64 * 0.5, r.size[1] as f64 * 0.5) } else { (r.size[1] as f64 * 0.5, r.size[0] as f64 * 0.5) };
            let scale = 0.93 + 0.1 * bu(i, j, 40 + lot);
            // slack in the lot: set back from the street by a random amount
            let sx = (lw * 0.5 - 2.0 - hx * scale).max(0.0);
            let sy = (lh * 0.5 - 2.0 - hy * scale).max(0.0);
            let (px, py) = (cx + (bu(i, j, 60 + lot) * 2.0 - 1.0) * sx * 0.7, cy + (bu(i, j, 80 + lot) * 2.0 - 1.0) * sy * 0.7);
            if built_at(px, py) < if class == Class::Tower { 0.15 } else { 0.3 } {
                continue;
            }
            let (ex, ey) = (hx * scale + 1.0, hy * scale + 1.0);
            if !(buildable(k, terrain, px, py) && buildable(k, terrain, px - ex, py - ey) && buildable(k, terrain, px + ex, py - ey) && buildable(k, terrain, px + ex, py + ey) && buildable(k, terrain, px - ex, py + ey)) {
                continue;
            }
            out.push(Placed { arch: pick.0, x: px, y: py, quarter, scale, hx: hx * scale, hy: hy * scale });
        }
    }
    out
}

/// Is lat / lon in a city? (A cheap test before `Blocks::blocked`.)
pub fn in_city(lat: f64, lon: f64) -> bool {
    city_at(lat, lon).is_some()
}

/// The buildings' footprints of the blocks asked about so far (one tile's trees ask about the
/// same few blocks thousands of times).
#[derive(Default)]
pub struct Blocks {
    cache: std::collections::HashMap<(usize, i64, i64), Vec<[f64; 4]>>,
}

impl Blocks {
    /// Is lat / lon in a city's roadway, or under a building? (Trees keep out of both.)
    pub fn blocked(&mut self, lat: f64, lon: f64, archs: &[ArchRef], terrain: &Terrain) -> bool {
        let Some((k, x, y)) = city_at(lat, lon) else { return false };
        if archs.is_empty() {
            return false;
        }
        let (i, j) = ((x / BX).floor() as i64, (y / BY).floor() as i64);
        let (fx, fy) = (x - i as f64 * BX, y - j as f64 * BY);
        // the roadways (the sidewalks may have trees)
        let road = |a: bool| half_width(a) - sidewalk(a);
        if fx < road(art_x(i)) || fx > BX - road(art_x(i + 1)) || fy < road(art_y(j)) || fy > BY - road(art_y(j + 1)) {
            return true;
        }
        let b = self.cache.entry((k, i, j)).or_insert_with(|| block(k, archs, terrain, i, j, 0.0).iter().map(|p| [p.x, p.y, p.hx, p.hy]).collect());
        b.iter().any(|p| (x - p[0]).abs() < p[2] + 1.5 && (y - p[1]).abs() < p[3] + 1.5)
    }
}

/// Height (m) a building must reach to be kept in a tree tile `k` levels coarser than the finest
/// (the far tiles: small buildings are below a pixel there).
fn min_height(k: u8) -> f32 {
    match k {
        0 => 0.0,
        1 => 14.0,
        2 => 30.0,
        3 => 60.0,
        _ => 110.0,
    }
}

/// Is the plane point's foot in tree tile `id`?
fn in_tile(k: usize, x: f64, y: f64, id: TileId) -> Option<DVec3> {
    let (lat, lon) = to_lat_lon(k, x, y);
    if city_at(lat, lon).map(|v| v.0) != Some(k) {
        return None;
    }
    let d = super::tiles::dir_from_lat_lon(lat, lon);
    let (face, u, v) = super::forest::dir_to_face_uv(d);
    let cells = 1u64 << id.level;
    let to_cell = |c: f64| (((c + 1.0) * 0.5 * cells as f64).floor() as i64).clamp(0, cells as i64 - 1) as u32;
    (face == id.face && to_cell(u) == id.x && to_cell(v) == id.y).then_some(d)
}

/// The buildings whose centres lie in tree tile `id`, as plant records into `tile`; on the
/// finest tiles also the palms along the sidewalks.
pub fn grow(id: TileId, terrain: &Terrain, sp: &Species, tile: &mut TreeTile) {
    let (archs, infos) = (&sp.buildings[..], &sp.infos[..]);
    if archs.is_empty() {
        return;
    }
    let (u0, v0, s) = id.uv();
    let corners: Vec<DVec3> = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)].iter().map(|(a, b)| face_dir(id.face, u0 + a * s, v0 + b * s)).collect();
    for (k, c) in CITIES.iter().enumerate() {
        // the tile's box in the city plane
        let pts: Vec<(f64, f64)> = corners.iter().map(|d| plane(k, *d)).collect();
        let (xl, xh) = pts.iter().fold((f64::MAX, f64::MIN), |a, p| (a.0.min(p.0), a.1.max(p.0)));
        let (yl, yh) = pts.iter().fold((f64::MAX, f64::MIN), |a, p| (a.0.min(p.1), a.1.max(p.1)));
        if xh < -c.radius || xl > c.radius || yh < -c.radius || yl > c.radius {
            continue;
        }
        let (xl, xh, yl, yh) = (xl.max(-c.radius), xh.min(c.radius), yl.max(-c.radius), yh.min(c.radius));
        let lvl = TREE_LEVEL.saturating_sub(id.level);
        let need = min_height(lvl);
        for j in (yl / BY).floor() as i64 - 1..=(yh / BY).floor() as i64 {
            for i in (xl / BX).floor() as i64 - 1..=(xh / BX).floor() as i64 {
                let placed = block(k, archs, terrain, i, j, need);
                if lvl == 0 && !sp.palms.is_empty() {
                    street_palms(k, terrain, sp, i, j, id, tile);
                    lot_trees(k, terrain, sp, i, j, id, tile, &placed);
                }
                for p in placed {
                    let Some(d) = in_tile(k, p.x, p.y, id) else { continue };
                    if tile.recs.len() >= MAX_PER_TILE {
                        return;
                    }
                    let (lat, lon) = lat_lon(d);
                    let r = &archs[p.arch];
                    let seed = 1 + (bh((p.x * 4.0) as i64, (p.y * 4.0) as i64, 7) % 8190);
                    // (sunk a little: the ground under a footprint is never quite flat; no
                    // thinning level: impostor.rint never thins buildings)
                    let rows = frame_rows(d, tile.center, terrain.height(lat, lon).max(0.0) as f64 - 0.4, p.quarter as f64 * std::f64::consts::FRAC_PI_2, p.scale);
                    push_rec(tile, rows, r.variant as u32 | seed << 8, &infos[r.variant]);
                }
            }
        }
    }
}

/// Palms along the sidewalks on the west and south sides of block (i, j)'s street lines (each
/// street line belongs to one block column or row: the street at x = i BX and the one at
/// y = j BY), both sides of the street, one every ~12 m near the kerb, on built-up dry land;
/// whole streets have palms or none. Those whose foot lies in tile `id`.
fn street_palms(k: usize, terrain: &Terrain, sp: &Species, i: i64, j: i64, id: TileId, tile: &mut TreeTile) {
    let mut spots = Vec::new();
    // the north-south street at x = i BX, along this block's y span
    if bh(i, 0, 50) % 3 != 0 {
        let a = art_x(i);
        let off = half_width(a) - sidewalk(a) + 1.1;
        let (ys, ye) = (j as f64 * BY + half_width(art_y(j)) + 4.0, (j + 1) as f64 * BY - half_width(art_y(j + 1)) - 4.0);
        let n = ((ye - ys) / 12.0).floor().max(1.0) as usize;
        for s in 0..=n {
            let y = ys + (ye - ys) * s as f64 / n as f64;
            spots.push((i as f64 * BX - off, y, s as u32));
            spots.push((i as f64 * BX + off, y, 100 + s as u32));
        }
    }
    // the east-west street at y = j BY, along this block's x span
    if bh(0, j, 51) % 3 != 0 {
        let a = art_y(j);
        let off = half_width(a) - sidewalk(a) + 1.1;
        let (xs, xe) = (i as f64 * BX + half_width(art_x(i)) + 4.0, (i + 1) as f64 * BX - half_width(art_x(i + 1)) - 4.0);
        let n = ((xe - xs) / 12.0).floor().max(1.0) as usize;
        for s in 0..=n {
            let x = xs + (xe - xs) * s as f64 / n as f64;
            spots.push((x, j as f64 * BY - off, 200 + s as u32));
            spots.push((x, j as f64 * BY + off, 300 + s as u32));
        }
    }
    for (x, y, n) in spots {
        if bu(i, j, 1000 + n) > 0.85 {
            continue;
        }
        let Some(d) = in_tile(k, x, y, id) else { continue };
        let (lat, lon) = lat_lon(d);
        let built = terrain.cover.fractions(lat, lon, 30.0).map(|f| f[4]).unwrap_or(0.0);
        if built < 0.25 || !buildable(k, terrain, x, y) {
            continue;
        }
        if tile.recs.len() >= MAX_PER_TILE {
            return;
        }
        let vid = sp.palms[(bh(i, j, 2000 + n) as usize) % sp.palms.len()];
        let rows = frame_rows(d, tile.center, terrain.height(lat, lon).max(0.0) as f64 - 0.1, bu(i, j, 3000 + n) * std::f64::consts::TAU, 0.85 + 0.35 * bu(i, j, 4000 + n));
        push_rec(tile, rows, vid as u32 | (1 + bh(i, j, 5000 + n) % 8190) << 8, &sp.infos[vid]);
    }
}

/// Trees on the lots of block (i, j), where globe.rchit cityGround leaves room for them: in the
/// houses' front yards (on the side away from the driveway) and back yards (beyond the pools),
/// palms along the planted edges of the parking lots, palms in the plazas' planters. Never under
/// a building; those whose foot lies in tile `id`.
#[allow(clippy::too_many_arguments)]
fn lot_trees(k: usize, terrain: &Terrain, sp: &Species, i: i64, j: i64, id: TileId, tile: &mut TreeTile, placed: &[Placed]) {
    let (x0, y0, w, h) = interior(i, j);
    let z = block_zone(k, i, j);
    // (x, y, broad crown?, salt)
    let mut spots: Vec<(f64, f64, bool, u32)> = Vec::new();
    if z == 0 {
        let n = (w / HOUSE_LOT).floor().max(1.0) as usize;
        let lw = w / n as f64;
        for row in 0..2 {
            for l in 0..n {
                let lot = (row * n + l) as u32;
                let right = bu(i, j, 30 + lot) < 0.5;
                let at = |u: f64, v: f64| (x0 + l as f64 * lw + u, if row == 0 { y0 + v } else { y0 + h - v });
                if bu(i, j, 3100 + lot) < 0.6 {
                    let (x, y) = at(if right { 2.6 } else { lw - 2.6 }, 2.8);
                    spots.push((x, y, bu(i, j, 3200 + lot) < 0.4, 3300 + lot));
                }
                let backs = 1 + (bu(i, j, 3400 + lot) * 2.2) as u32;
                for t in 0..backs {
                    let u = 3.0 + (lw - 6.0) * bu(i, j, 3500 + lot * 4 + t);
                    let (x, y) = at(u, h * 0.5 - 3.5 - 2.0 * bu(i, j, 3600 + lot * 4 + t));
                    spots.push((x, y, bu(i, j, 3700 + lot * 4 + t) < 0.7, 3800 + lot * 4 + t));
                }
            }
        }
    } else if z == 1 || (z == 2 && bu(i, j, 3) < 0.5) {
        // the planted strip round a parking lot (2.2 m wide)
        let per = 2.0 * (w + h);
        let n = (per / 15.0).floor() as u32;
        for t in 0..n {
            if bu(i, j, 4100 + t) > 0.5 {
                continue;
            }
            let s = (t as f64 + 0.5) / n as f64 * per;
            let (u, v) = if s < w {
                (s, 1.1)
            } else if s < w + h {
                (w - 1.1, s - w)
            } else if s < 2.0 * w + h {
                (2.0 * w + h - s, h - 1.1)
            } else {
                (1.1, per - s)
            };
            spots.push((x0 + u, y0 + v, false, 4200 + t));
        }
    } else if bu(i, j, 5) >= 0.4 {
        // plaza planters: one every 14 x 11 m, a palm in each
        let (nx, ny) = ((w / 14.0).floor() as i64, (h / 11.0).floor() as i64);
        for a in 0..nx {
            for b in 0..ny {
                spots.push((x0 + a as f64 * 14.0 + 3.0, y0 + b as f64 * 11.0 + 1.8, false, 4400 + (a * 31 + b) as u32));
            }
        }
    }
    for (x, y, broad, salt) in spots {
        if placed.iter().any(|p| (x - p.x).abs() < p.hx + 1.5 && (y - p.y).abs() < p.hy + 1.5) {
            continue;
        }
        let list = if broad && !sp.city_trees.is_empty() { &sp.city_trees } else { &sp.palms };
        let Some(d) = in_tile(k, x, y, id) else { continue };
        let (lat, lon) = lat_lon(d);
        if !buildable(k, terrain, x, y) || terrain.cover.fractions(lat, lon, 30.0).map(|f| f[4]).unwrap_or(0.0) < 0.2 {
            continue;
        }
        if tile.recs.len() >= MAX_PER_TILE {
            return;
        }
        let vid = list[(bh(i, j, salt) as usize) % list.len()];
        let sc = if broad { 0.6 + 0.35 * bu(i, j, salt + 50_000) } else { 0.85 + 0.35 * bu(i, j, salt + 50_000) };
        let rows = frame_rows(d, tile.center, terrain.height(lat, lon).max(0.0) as f64 - 0.1, bu(i, j, salt + 60_000) * std::f64::consts::TAU, sc);
        push_rec(tile, rows, vid as u32 | (1 + bh(i, j, salt + 70_000) % 8190) << 8, &sp.infos[vid]);
    }
}

/// Object -> tile rows for something standing at direction `d` at height `h`, turned `yaw`
/// from east towards north, scaled.
fn frame_rows(d: DVec3, center: DVec3, h: f64, yaw: f64, sc: f64) -> [[f64; 4]; 3] {
    let up = d;
    let east = DVec3::Z.cross(up).normalize();
    let north = up.cross(east);
    let x = east * yaw.cos() + north * yaw.sin();
    let z = x.cross(up);
    let (x, y, z) = (x * sc, up * sc, z * sc);
    let t = d * (R_EARTH + h) - center;
    [[x.x, y.x, z.x, t.x], [x.y, y.y, z.y, t.y], [x.z, y.z, z.z, t.z]]
}

fn push_rec(tile: &mut TreeTile, rows: [[f64; 4]; 3], data: u32, info: &ImpostorInfo) {
    let mut rec = [0u32; REC_WORDS];
    for (ri, row) in rows.iter().enumerate() {
        for (ci, v) in row.iter().enumerate() {
            rec[ri * 4 + ci] = (*v as f32).to_bits();
        }
    }
    rec[12] = data;
    let (bmin, bmax) = (info.bmin, info.bmax);
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    for c in 0..8 {
        let q = [
            if c & 1 == 0 { bmin[0] } else { bmax[0] } as f64,
            if c & 2 == 0 { bmin[1] } else { bmax[1] } as f64,
            if c & 4 == 0 { bmin[2] } else { bmax[2] } as f64,
        ];
        for (k, row) in rows.iter().enumerate() {
            let v = (row[0] * q[0] + row[1] * q[1] + row[2] * q[2] + row[3]) as f32;
            lo[k] = lo[k].min(v);
            hi[k] = hi[k].max(v);
        }
    }
    tile.recs.push(rec);
    tile.aabbs.push([lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]]);
}
