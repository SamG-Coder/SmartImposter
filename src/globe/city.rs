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
pub const MAX_CORES: usize = 8;

/// Miami: downtown and Brickell on the west shore of Biscayne Bay, Edgewater north of them, the
/// high-rises along Miami Beach across the bay.
const CITIES: &[City] = &[City {
    lat: 25.7800,
    lon: -80.1700,
    radius: 14_000.0,
    cores: &[
        Core { lat: 25.7745, lon: -80.1905, strength: 1.0, radius: 750.0 },   // downtown
        Core { lat: 25.7640, lon: -80.1925, strength: 1.0, radius: 650.0 },   // Brickell
        Core { lat: 25.7930, lon: -80.1875, strength: 0.75, radius: 700.0 },  // Edgewater
        Core { lat: 25.7790, lon: -80.1310, strength: 0.6, radius: 600.0 },   // South Beach
        Core { lat: 25.8100, lon: -80.1240, strength: 0.65, radius: 900.0 },  // Mid-Beach
        Core { lat: 25.7520, lon: -80.2050, strength: 0.4, radius: 1200.0 },  // Coconut Grove side
        // the broad urban fields round them: Miami Beach mid-rise all along, the city round
        // downtown (walk-ups and apartment blocks rather than houses)
        Core { lat: 25.8000, lon: -80.1320, strength: 0.4, radius: 2600.0 },  // Miami Beach
        Core { lat: 25.7760, lon: -80.2000, strength: 0.32, radius: 3200.0 }, // downtown's fringe
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
/// The street lines: every BX / BY m, each moved by up to +-12 / +-10 m (blocks of
/// different sizes, not one even grid; globe.rchit cLineX / cLineY).
pub fn line_x(i: i64) -> f64 {
    i as f64 * BX + (bu(i, 0, 40) - 0.5) * 24.0
}
pub fn line_y(j: i64) -> f64 {
    j as f64 * BY + (bu(0, j, 41) - 0.5) * 20.0
}
/// The block (i, j) a plane point is in.
fn cell(x: f64, y: f64) -> (i64, i64) {
    let mut i = (x / BX).floor() as i64;
    if x < line_x(i) {
        i -= 1;
    } else if x >= line_x(i + 1) {
        i += 1;
    }
    let mut j = (y / BY).floor() as i64;
    if y < line_y(j) {
        j -= 1;
    } else if y >= line_y(j + 1) {
        j += 1;
    }
    (i, j)
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
    if dens > 0.55 {
        3
    } else if dens > 0.2 {
        2
    } else if dens > 0.05 {
        1
    } else {
        0
    }
}

/// The inside of block (i, j) (between the sidewalks): x0, y0, width, height.
fn interior(i: i64, j: i64) -> (f64, f64, f64, f64) {
    let x0 = line_x(i) + half_width(art_x(i));
    let x1 = line_x(i + 1) - half_width(art_x(i + 1));
    let y0 = line_y(j) + half_width(art_y(j));
    let y1 = line_y(j + 1) - half_width(art_y(j + 1));
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

fn density_at_block(k: usize, i: i64, j: i64) -> f64 {
    let (x0, y0, w, h) = interior(i, j);
    density(k, x0 + w * 0.5, y0 + h * 0.5)
}

fn block_zone(k: usize, i: i64, j: i64) -> u32 {
    let (x0, y0, w, h) = interior(i, j);
    let z = zone(density(k, x0 + w * 0.5, y0 + h * 0.5) + (bu(i, j, 1) - 0.5) * 0.12);
    // shops and walk-ups along the arterials, between the house blocks
    if z == 0 && (art_x(i) || art_x(i + 1) || art_y(j) || art_y(j + 1)) && bu(i, j, 6) < 0.6 {
        1
    } else {
        z
    }
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
        // (a tower in the mid-rise zone only next to a core)
        2 if density_at_block(k, i, j) > 0.38 => &[Class::MidRise, Class::MidRise, Class::Tower, Class::LowRise],
        2 => &[Class::MidRise, Class::MidRise, Class::MidRise, Class::LowRise],
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
        2 => {
            if bu(i, j, 7) < 0.5 {
                (3, 2)
            } else {
                (2, 2)
            }
        }
        _ => {
            if bu(i, j, 4) < 0.4 {
                (2, 2)
            } else {
                (3, 2)
            }
        }
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
                // (outside the cores only the shortest towers: the mid-rise city round them is
                // 10-25 storeys)
                let mut want = if z == 3 { 70.0 + 170.0 * dens.powf(1.3) * (0.55 + 0.6 * bu(i, j, 90 + lot)) } else { 55.0 + 35.0 * bu(i, j, 90 + lot) };
                if z == 3 && bu(i, j, 95 + lot) < 0.15 {
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
            let (mut px, mut py) = (cx + (bu(i, j, 60 + lot) * 2.0 - 1.0) * sx * 0.7, cy + (bu(i, j, 80 + lot) * 2.0 - 1.0) * sy * 0.7);
            // downtown a tower stands at one end of its lot, a podium (garage or shops) fills
            // the rest, as the parking podiums of real towers do
            let podium = class == Class::Tower && z == 3 && sy > 14.0;
            let side = if bu(i, j, 85 + lot) < 0.5 { 1.0 } else { -1.0 };
            if podium {
                py = cy + side * sy;
                px = cx + (bu(i, j, 60 + lot) * 2.0 - 1.0) * sx * 0.3;
            }
            if built_at(px, py) < if class == Class::Tower { 0.15 } else { 0.3 } {
                continue;
            }
            let (ex, ey) = (hx * scale + 1.0, hy * scale + 1.0);
            if !(buildable(k, terrain, px, py) && buildable(k, terrain, px - ex, py - ey) && buildable(k, terrain, px + ex, py - ey) && buildable(k, terrain, px + ex, py + ey) && buildable(k, terrain, px - ex, py + ey)) {
                continue;
            }
            if podium {
                // the free span: from the lot's far edge to the tower, less a gap
                let free = 2.0 * sy - 1.5;
                let gy = py - side * (hy * scale + 1.5 + free * 0.5);
                let mut best: Option<(usize, u32, f64)> = None;
                for (a, r) in archs.iter().enumerate() {
                    if r.class != Class::LowRise {
                        continue;
                    }
                    for q in 0..2u32 {
                        let (ax, ay) = if q == 0 { (r.size[0] as f64, r.size[1] as f64) } else { (r.size[1] as f64, r.size[0] as f64) };
                        if ax <= lw - 4.0 && ay <= free && best.is_none_or(|b| ax * ay > b.2) {
                            best = Some((a, q, ax * ay));
                        }
                    }
                }
                if let Some((a, q, _)) = best {
                    let r = &archs[a];
                    let (ax, ay) = if q == 0 { (r.size[0] as f64, r.size[1] as f64) } else { (r.size[1] as f64, r.size[0] as f64) };
                    if r.height >= need && buildable(k, terrain, px, gy) && buildable(k, terrain, px, gy - side * ay * 0.5) {
                        out.push(Placed { arch: a, x: px, y: gy, quarter: q, scale: 1.0, hx: ax * 0.5, hy: ay * 0.5 });
                    }
                }
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
        let (i, j) = cell(x, y);
        let (fx, fy) = (x - line_x(i), y - line_y(j));
        let (bxw, byw) = (line_x(i + 1) - line_x(i), line_y(j + 1) - line_y(j));
        // the roadways (the sidewalks may have trees)
        let road = |a: bool| half_width(a) - sidewalk(a);
        if fx < road(art_x(i)) || fx > bxw - road(art_x(i + 1)) || fy < road(art_y(j)) || fy > byw - road(art_y(j + 1)) {
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
        for j in (yl / BY).floor() as i64 - 1..=(yh / BY).floor() as i64 + 1 {
            for i in (xl / BX).floor() as i64 - 1..=(xh / BX).floor() as i64 + 1 {
                let placed = block(k, archs, terrain, i, j, need);
                for p in &placed {
                    let Some(d) = in_tile(k, p.x, p.y, id) else { continue };
                    if tile.recs.len() >= MAX_PER_TILE {
                        break;
                    }
                    let (lat, lon) = lat_lon(d);
                    let r = &archs[p.arch];
                    let seed = 1 + (bh((p.x * 4.0) as i64, (p.y * 4.0) as i64, 7) % 8190);
                    // (sunk a little: the ground under a footprint is never quite flat; no
                    // thinning level: impostor.rint never thins buildings)
                    let rows = frame_rows(d, tile.center, terrain.height(lat, lon).max(0.0) as f64 - 0.4, p.quarter as f64 * std::f64::consts::FRAC_PI_2, p.scale);
                    push_rec(tile, rows, super::forest::rec_data(r.variant, seed << 8), &infos[r.variant]);
                }
                // (the buildings first: a tile full of cars must not lose a tower)
                if lvl == 0 && !sp.palms.is_empty() {
                    street_palms(k, terrain, sp, i, j, id, tile);
                    lot_trees(k, terrain, sp, i, j, id, tile, &placed);
                }
                if lvl == 0 {
                    docks(k, terrain, sp, i, j, id, tile, &placed);
                    parked(k, terrain, sp, i, j, id, tile, &placed);
                }
                if lvl <= 2 {
                    streets(k, terrain, sp, i, j, id, tile, lvl == 0);
                }
            }
        }
        // boats out on the water: one cell of BOAT_CELL m at a time, a boat in some
        if lvl <= 1 {
            for b in (yl / BOAT_CELL).floor() as i64..=(yh / BOAT_CELL).floor() as i64 {
                for a in (xl / BOAT_CELL).floor() as i64..=(xh / BOAT_CELL).floor() as i64 {
                    open_water_boat(k, terrain, sp, a, b, lvl, id, tile);
                }
            }
        }
    }
}


/// The car archetypes (compact, sedan, SUV, pickup, van, bus) and the bridge decks (arterial,
/// side street).
fn car_kinds(archs: &[ArchRef]) -> (Vec<&ArchRef>, Vec<&ArchRef>) {
    (archs.iter().filter(|a| a.class == Class::Car).collect(), archs.iter().filter(|a| a.class == Class::Bridge).collect())
}

/// Does the street through (x, y) cross water here on a bridge? Short crossings (canals, the
/// river: land within 160 m both ways along the street) always; long ones (across the bay, to
/// 1.6 km) only on every other arterial, as causeways.
fn bridged(k: usize, terrain: &Terrain, x: f64, y: f64, along_x: bool, arterial: bool, line: i64) -> bool {
    if !water_at(k, terrain, x, y) {
        return false;
    }
    // a crossing has water on both sides of the street (a street running along a shore just off
    // it does not)
    let (px, py) = if along_x { (0.0, 40.0) } else { (40.0, 0.0) };
    if !water_at(k, terrain, x + px, y + py) || !water_at(k, terrain, x - px, y - py) {
        return false;
    }
    // (canals for the side streets, the river and the like for the arterials; the bay only on
    // the causeways)
    let reach = if arterial && bh(line, along_x as i64, 77) % 2 == 0 { 1600.0 } else if arterial { 250.0 } else { 70.0 };
    let land = |dir: f64| {
        let mut s = 10.0;
        while s <= reach {
            let (px, py) = if along_x { (x + dir * s, y) } else { (x, y + dir * s) };
            if !water_at(k, terrain, px, py) {
                return true;
            }
            s += 10.0;
        }
        false
    };
    // (both banks within reach: the whole crossing is shorter than about twice that)
    land(1.0) && land(-1.0)
}

/// Bridge decks and traffic on the street lines of block (i, j) (as street_palms: the
/// north-south street at x = i BX along this block's y span, the east-west one at y = j BY
/// along its x span, crossings included). `cars`: also the traffic (finest tiles only).
#[allow(clippy::too_many_arguments)]
fn streets(k: usize, terrain: &Terrain, sp: &Species, i: i64, j: i64, id: TileId, tile: &mut TreeTile, cars: bool) {
    let (kinds, decks) = car_kinds(&sp.buildings);
    if decks.len() < 2 {
        return;
    }
    for along_x in [false, true] {
        let (line, a) = if along_x { (j, art_y(j)) } else { (i, art_x(i)) };
        let (start, len) = if along_x { (line_x(i), line_x(i + 1) - line_x(i)) } else { (line_y(j), line_y(j + 1) - line_y(j)) };
        let fixed = if along_x { line_y(j) } else { line_x(i) };
        let at = |s: f64, off: f64| if along_x { (start + s, fixed + off) } else { (fixed + off, start + s) };
        let yaw0 = if along_x { 0.0 } else { std::f64::consts::FRAC_PI_2 };
        // decks every 20 m where the street is over water
        let mut s = 10.0;
        let mut n = 0u32;
        while s < len {
            let (x, y) = at(s, 0.0);
            if bridged(k, terrain, x, y, along_x, a, line) {
                put_at(k, sp, decks[if a { 0 } else { 1 }], x, y, 0.0, yaw0, bh(i, j, 6000 + n + along_x as u32 * 50), id, tile);
            }
            s += 20.0;
            n += 1;
        }
        if !cars || kinds.len() < 6 {
            continue;
        }
        // traffic, driving on the right: lanes at +-1.75 and +-5.25 m on an arterial, +-2 m on
        // a side street; none in the crossings
        let lanes: &[f64] = if a { &[-5.25, -1.75, 1.75, 5.25] } else { &[-2.0, 2.0] };
        let road_x = |ii: i64| half_width(art_x(ii)) + 2.0;
        let road_y = |jj: i64| half_width(art_y(jj)) + 2.0;
        let (c0, c1) = if along_x { (road_x(i), len - road_x(i + 1)) } else { (road_y(j), len - road_y(j + 1)) };
        for (li, &off) in lanes.iter().enumerate() {
            let mut s = c0 + bu(i, j, 6500 + li as u32 + along_x as u32 * 10) * 20.0;
            let mut n = 0u32;
            while s < c1 {
                let salt = 7000 + (li as u32) * 100 + n + along_x as u32 * 1000;
                let busy = if a { 0.75 } else { 0.3 };
                if bu(i, j, salt) < busy {
                    let (x, y) = at(s, off);
                    // northbound / eastbound on the right of the centre line
                    let forward = if along_x { off < 0.0 } else { off > 0.0 };
                    let yaw = yaw0 + if forward { 0.0 } else { std::f64::consts::PI };
                    let kind = if a && bu(i, j, salt + 500) < 0.04 { 5 } else { (bh(i, j, salt + 600) % 5) as usize };
                    let over = water_at(k, terrain, x, y);
                    if !over || bridged(k, terrain, x, y, along_x, a, line) {
                        let h = if over { 2.92 } else { ground(k, terrain, x, y) };
                        put_at(k, sp, kinds[kind], x, y, h, yaw, bh(i, j, salt + 700), id, tile);
                    }
                }
                s += if a { 16.0 + 30.0 * bu(i, j, salt + 800) } else { 25.0 + 50.0 * bu(i, j, salt + 800) };
                n += 1;
            }
        }
    }
}

fn ground(k: usize, terrain: &Terrain, x: f64, y: f64) -> f64 {
    let (la, lo) = to_lat_lon(k, x, y);
    terrain.height(la, lo).max(0.0) as f64
}

/// Parked cars: in the stalls of block (i, j)'s parking lots (the bays globe.rchit cityGround
/// draws) and in the houses' driveways. Never under a building.
#[allow(clippy::too_many_arguments)]
fn parked(k: usize, terrain: &Terrain, sp: &Species, i: i64, j: i64, id: TileId, tile: &mut TreeTile, placed: &[Placed]) {
    let (kinds, _) = car_kinds(&sp.buildings);
    if kinds.len() < 6 {
        return;
    }
    let (x0, y0, w, h) = interior(i, j);
    let z = block_zone(k, i, j);
    let free = |x: f64, y: f64| !placed.iter().any(|p| (x - p.x).abs() < p.hx + 1.2 && (y - p.y).abs() < p.hy + 1.2);
    let mut n = 0u32;
    if z == 1 || (z == 2 && bu(i, j, 3) < 0.3) {
        // stalls of 2.6 m either side of 7 m aisles, rows every 18 m (from 2 m in)
        let mut row = 0;
        loop {
            let base = 2.0 + 18.0 * row as f64;
            if base + 15.25 + 2.5 > h - 2.2 && base + 2.75 + 2.5 > h - 2.2 {
                break;
            }
            for (dy, flip) in [(2.75, 0.0), (15.25, std::f64::consts::PI)] {
                let by = base + dy;
                if by < 4.6 || by > h - 4.6 {
                    continue;
                }
                let stalls = ((w - 4.4) / 2.6).floor() as i64;
                for st in 0..stalls {
                    n += 1;
                    if bu(i, j, 20_000 + n) > 0.62 {
                        continue;
                    }
                    let bx = 2.2 + (st as f64 + 0.5) * 2.6 + (w - 4.4 - stalls as f64 * 2.6) * 0.5;
                    let (x, y) = (x0 + bx, y0 + by);
                    if !free(x, y) || !buildable(k, terrain, x, y) {
                        continue;
                    }
                    let kind = (bh(i, j, 21_000 + n) % 5) as usize;
                    let yaw = std::f64::consts::FRAC_PI_2 + flip + (bu(i, j, 22_000 + n) - 0.5) * 0.08;
                    put_at(k, sp, kinds[kind], x, y, ground(k, terrain, x, y), yaw, bh(i, j, 23_000 + n), id, tile);
                }
            }
            row += 1;
        }
    } else if z == 0 {
        let nl = (w / HOUSE_LOT).floor().max(1.0) as usize;
        let lw = w / nl as f64;
        for r in 0..2 {
            for l in 0..nl {
                let lot = (r * nl + l) as u32;
                let right = bu(i, j, 30 + lot) < 0.5;
                let u = if right { lw - 3.0 } else { 3.0 };
                for (c, v) in [(0u32, 3.4), (1, 9.0)] {
                    if bu(i, j, 24_000 + lot * 2 + c) > if c == 0 { 0.65 } else { 0.3 } {
                        continue;
                    }
                    let (x, y) = (x0 + l as f64 * lw + u, if r == 0 { y0 + v } else { y0 + h - v });
                    if !free(x, y) || !buildable(k, terrain, x, y) {
                        continue;
                    }
                    // nose in towards the house
                    let yaw = if r == 0 { std::f64::consts::FRAC_PI_2 } else { -std::f64::consts::FRAC_PI_2 };
                    let kind = (bh(i, j, 25_000 + lot * 2 + c) % 5) as usize;
                    put_at(k, sp, kinds[kind], x, y, ground(k, terrain, x, y), yaw, bh(i, j, 26_000 + lot * 2 + c), id, tile);
                }
            }
        }
    }
}

/// Place archetype `r` at plane point (x, y), height h, facing `yaw`, if its foot is in tile
/// `id`.
#[allow(clippy::too_many_arguments)]
fn put_at(k: usize, sp: &Species, r: &ArchRef, x: f64, y: f64, h: f64, yaw: f64, seed: u32, id: TileId, tile: &mut TreeTile) {
    let Some(d) = in_tile(k, x, y, id) else { return };
    if tile.recs.len() >= MAX_PER_TILE {
        return;
    }
    let rows = frame_rows(d, tile.center, h, yaw, 1.0);
    push_rec(tile, rows, super::forest::rec_data(r.variant, (1 + seed % 8190) << 8), &sp.infos[r.variant]);
}

const BOAT_CELL: f64 = 130.0;

/// The boat archetypes (runabout, cruiser, yacht) and the dock, in that order.
fn boat_kinds(archs: &[ArchRef]) -> (Vec<&ArchRef>, Option<&ArchRef>) {
    (archs.iter().filter(|a| a.class == Class::Boat).collect(), archs.iter().find(|a| a.class == Class::Dock))
}

fn water_at(k: usize, terrain: &Terrain, x: f64, y: f64) -> bool {
    let (la, lo) = to_lat_lon(k, x, y);
    terrain.cover.class_group(la, lo) == Some(lc::WATER)
}

/// Place a boat (variant of `r`) at plane point (x, y), bow towards `yaw` (from east towards
/// north), floating at sea level, if its foot is in tile `id`.
#[allow(clippy::too_many_arguments)]
fn put_boat(k: usize, sp: &Species, r: &ArchRef, x: f64, y: f64, yaw: f64, seed: u32, id: TileId, tile: &mut TreeTile) {
    let Some(d) = in_tile(k, x, y, id) else { return };
    if tile.recs.len() >= MAX_PER_TILE {
        return;
    }
    let rows = frame_rows(d, tile.center, 0.0, yaw, 1.0);
    push_rec(tile, rows, super::forest::rec_data(r.variant, (1 + seed % 8190) << 8), &sp.infos[r.variant]);
}

/// A boat on open water in cell (a, b), clear of the shore: runabouts and cruisers near
/// (finest tiles), the yachts also a level out.
#[allow(clippy::too_many_arguments)]
fn open_water_boat(k: usize, terrain: &Terrain, sp: &Species, a: i64, b: i64, lvl: u8, id: TileId, tile: &mut TreeTile) {
    let (boats, _) = boat_kinds(&sp.buildings);
    let under: Vec<&ArchRef> = sp.buildings.iter().filter(|a| a.class == Class::Underway).collect();
    if boats.len() < 3 || bu(a, b, 9000) > 0.5 {
        return;
    }
    let kind = if bu(a, b, 9001) < 0.5 { 0 } else if bu(a, b, 9001) < 0.85 { 1 } else { 2 };
    // (each tile level covers its own band of distance: the yachts are in the next level's tiles
    // too, so they are still seen beyond the finest tiles' 6 km)
    if lvl == 1 && kind != 2 {
        return;
    }
    let (x, y) = ((a as f64 + 0.15 + 0.7 * bu(a, b, 9002)) * BOAT_CELL, (b as f64 + 0.15 + 0.7 * bu(a, b, 9003)) * BOAT_CELL);
    let yaw = bu(a, b, 9004) * std::f64::consts::TAU;
    // most are under way, with their wake
    let r = if under.len() == 3 && bu(a, b, 9006) < 0.75 { under[kind] } else { boats[kind] };
    let half = r.size[0] as f64 * 0.5 + 25.0;
    let (c, s) = (yaw.cos(), yaw.sin());
    let clear = [(0.0, 0.0), (half, 0.0), (-half, 0.0), (0.0, 20.0), (0.0, -20.0)]
        .iter()
        .all(|&(u, v)| water_at(k, terrain, x + u * c - v * s, y + u * s + v * c));
    if clear {
        put_boat(k, sp, r, x, y, yaw, bh(a, b, 9005), id, tile);
    }
}

/// Docks behind the waterfront houses of block (i, j): where the land beyond a house lot's back
/// line is water, a timber dock runs out from the lot, a boat moored alongside most of them.
#[allow(clippy::too_many_arguments)]
fn docks(k: usize, terrain: &Terrain, sp: &Species, i: i64, j: i64, id: TileId, tile: &mut TreeTile, placed: &[Placed]) {
    let (boats, dock) = boat_kinds(&sp.buildings);
    let Some(dock) = dock else { return };
    if block_zone(k, i, j) != 0 || placed.is_empty() {
        return;
    }
    let (x0, y0, w, h) = interior(i, j);
    let n = (w / HOUSE_LOT).floor().max(1.0) as usize;
    let lw = w / n as f64;
    for row in 0..2 {
        for l in 0..n {
            let lot = (row * n + l) as u32;
            let u = l as f64 * lw + lw * (0.3 + 0.4 * bu(i, j, 9100 + lot));
            // the lot's back line, and outwards (away from its street)
            let (yb, out) = if row == 0 { (y0 + h * 0.5, 1.0) } else { (y0 + h * 0.5, -1.0) };
            let x = x0 + u;
            // a house on this lot, land at the lot, water beyond the back line
            let has_house = placed.iter().any(|p| (p.x - (x0 + l as f64 * lw + lw * 0.5)).abs() < lw * 0.5 && (p.y - yb).abs() < h * 0.5 && (p.y - yb) * out < 0.0);
            if !has_house || !water_at(k, terrain, x, yb + out * 6.0) || !water_at(k, terrain, x, yb + out * 14.0) {
                continue;
            }
            // the dock starts at the shore: walk back from the water to the first land
            let mut s = 6.0;
            while s > -10.0 && water_at(k, terrain, x, yb + out * s) {
                s -= 2.0;
            }
            let ys = yb + out * (s + 1.0);
            let yaw = if out > 0.0 { std::f64::consts::FRAC_PI_2 } else { -std::f64::consts::FRAC_PI_2 };
            put_boat(k, sp, dock, x, ys, yaw, bh(i, j, 9200 + lot), id, tile);
            if boats.len() >= 2 && bu(i, j, 9300 + lot) < 0.75 {
                let kind = if bu(i, j, 9400 + lot) < 0.55 { 0 } else { 1 };
                let r = boats[kind];
                let side = if bu(i, j, 9500 + lot) < 0.5 { 1.0 } else { -1.0 };
                let bx = x + side * (1.4 + r.size[1] as f64 * 0.5 + 0.4);
                let by = ys + out * (7.0 + r.size[0] as f64 * 0.1);
                if water_at(k, terrain, bx, by + out * r.size[0] as f64 * 0.5) {
                    // bow out or in
                    let flip = if bu(i, j, 9600 + lot) < 0.5 { 0.0 } else { std::f64::consts::PI };
                    put_boat(k, sp, r, bx, by, yaw + flip, bh(i, j, 9700 + lot), id, tile);
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
    // (x, y, salt, palm?) : two streets in three are palm-lined, the rest have shade trees
    let mut spots = Vec::new();
    // the north-south street at x = i BX, along this block's y span
    {
        let palm = bh(i, 0, 50) % 3 != 0;
        let a = art_x(i);
        let off = half_width(a) - sidewalk(a) + 1.1;
        let (ys, ye) = (line_y(j) + half_width(art_y(j)) + 4.0, line_y(j + 1) - half_width(art_y(j + 1)) - 4.0);
        let n = ((ye - ys) / if palm { 12.0 } else { 15.0 }).floor().max(1.0) as usize;
        for s in 0..=n {
            let y = ys + (ye - ys) * s as f64 / n as f64;
            spots.push((line_x(i) - off, y, s as u32, palm));
            spots.push((line_x(i) + off, y, 100 + s as u32, palm));
        }
    }
    // the east-west street at y = j BY, along this block's x span
    {
        let palm = bh(0, j, 51) % 3 != 0;
        let a = art_y(j);
        let off = half_width(a) - sidewalk(a) + 1.1;
        let (xs, xe) = (line_x(i) + half_width(art_x(i)) + 4.0, line_x(i + 1) - half_width(art_x(i + 1)) - 4.0);
        let n = ((xe - xs) / if palm { 12.0 } else { 15.0 }).floor().max(1.0) as usize;
        for s in 0..=n {
            let x = xs + (xe - xs) * s as f64 / n as f64;
            spots.push((x, line_y(j) - off, 200 + s as u32, palm));
            spots.push((x, line_y(j) + off, 300 + s as u32, palm));
        }
    }
    for (x, y, n, palm) in spots {
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
        let list = if palm || sp.city_trees.is_empty() { &sp.palms } else { &sp.city_trees };
        let vid = list[(bh(i, j, 2000 + n) as usize) % list.len()];
        let sc = if palm { 0.85 + 0.35 * bu(i, j, 4000 + n) } else { 0.55 + 0.3 * bu(i, j, 4000 + n) };
        let rows = frame_rows(d, tile.center, terrain.height(lat, lon).max(0.0) as f64 - 0.1, bu(i, j, 3000 + n) * std::f64::consts::TAU, sc);
        push_rec(tile, rows, super::forest::rec_data(vid, (1 + bh(i, j, 5000 + n) % 8190) << 8), &sp.infos[vid]);
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
                if bu(i, j, 3100 + lot) < 0.85 {
                    let (x, y) = at(if right { 2.6 } else { lw - 2.6 }, 2.8);
                    spots.push((x, y, bu(i, j, 3200 + lot) < 0.4, 3300 + lot));
                }
                let backs = 2 + (bu(i, j, 3400 + lot) * 2.5) as u32;
                for t in 0..backs {
                    let u = 3.0 + (lw - 6.0) * bu(i, j, 3500 + lot * 4 + t);
                    let (x, y) = at(u, h * 0.5 - 3.5 - 2.0 * bu(i, j, 3600 + lot * 4 + t));
                    spots.push((x, y, bu(i, j, 3700 + lot * 4 + t) < 0.7, 3800 + lot * 4 + t));
                }
            }
        }
    } else if z == 1 || (z == 2 && bu(i, j, 3) < 0.3) {
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
    // the waterfront: palms and broad crowns along every edge of the block that the water
    // comes within 12 m of (the canal banks, the bay shore), every ~8 m
    {
        let per = 2.0 * (w + h);
        let n = (per / 8.0).floor() as u32;
        for t in 0..n {
            let s = (t as f64 + 0.5) / n as f64 * per;
            let (u, v, nx, ny) = if s < w {
                (s, 1.5, 0.0, -1.0)
            } else if s < w + h {
                (w - 1.5, s - w, 1.0, 0.0)
            } else if s < 2.0 * w + h {
                (2.0 * w + h - s, h - 1.5, 0.0, 1.0)
            } else {
                (1.5, per - s, -1.0, 0.0)
            };
            let (x, y) = (x0 + u, y0 + v);
            // water just beyond this edge (past the street) or right here at a lot's back
            let near = water_at(k, terrain, x + nx * 12.0, y + ny * 12.0) || water_at(k, terrain, x - nx * 6.0 + ny * 6.0, y - ny * 6.0 + nx * 6.0);
            if near && bu(i, j, 8000 + t) < 0.8 {
                spots.push((x, y, bu(i, j, 8100 + t) < 0.35, 8200 + t));
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
        let sc = if broad { 0.7 + 0.4 * bu(i, j, salt + 50_000) } else { 0.85 + 0.35 * bu(i, j, salt + 50_000) };
        let rows = frame_rows(d, tile.center, terrain.height(lat, lon).max(0.0) as f64 - 0.1, bu(i, j, salt + 60_000) * std::f64::consts::TAU, sc);
        push_rec(tile, rows, super::forest::rec_data(vid, (1 + bh(i, j, salt + 70_000) % 8190) << 8), &sp.infos[vid]);
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
