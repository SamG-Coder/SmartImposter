//! Procedural terrain, forest growth and layout.
//!
//! * Terrain: rolling hills on a graded grid (3 m cells in the middle, up to 300 m at the edge)
//!   reaching TERRAIN_HALF in every direction, with the Earth's curvature (d^2 / 2R).
//! * Canopy: a seed-dispersal simulation. Founder trees start in pairs; every generation each tree
//!   drops seeds with its species' dispersal kernel, and a seedling survives depending on the site
//!   (biome), crowding and shade tolerance. Stands, groves and ragged edges emerge on their own.
//! * Understory and grass: jittered grids driven by the simulated canopy cover.
//! * Every plant is an individual instance with its own transform; the renderer groups them into
//!   cluster acceleration structures (one impostor box per plant), so the forest can reach far out.

use crate::asset::{MeshData, TreeAsset};
use crate::catalog::{Catalog, Kind};
use glam::{Affine3A, Quat, Vec3};
use crate::erosion::Erosion;
use std::sync::OnceLock;

pub const MAT_TERRAIN: u8 = 2;
/// Mean Earth radius: heights include the drop d^2 / 2R from the world origin.
pub const EARTH_RADIUS: f32 = 6_371_000.0;
/// Half extent of the terrain mesh (m). From 50 m up the horizon is ~25 km away.
pub const TERRAIN_HALF: f32 = 60_000.0;
/// Half extent of the fine terrain grid, the canopy-cover grid and the near ground mask.
pub const NEAR_HALF: f32 = 2048.0;
/// Fine terrain cell size (m).
const NEAR_STEP: f32 = 3.0;
/// Canopy cover grid resolution over +-NEAR_HALF (2 m cells).
pub const COVER_RES: usize = 2048;

pub struct TreeInstance {
    pub xform: Affine3A,
    pub center: Vec3, // world-space bounding sphere centre, used for LOD distance
    pub variant: u8,
    pub seed: u16, // 13 bits, never 0 (0 = untinted, used by the baker)
    pub color: u8, // impostor instance-mask colour
}

pub struct World {
    pub terrain: MeshData,
    pub trees: Vec<TreeInstance>,
    /// Graded grid coordinates (same along x and z) and heights (row-major, z rows).
    xs: Vec<f32>,
    heights: Vec<f32>,
    /// Per vertex: in the river channel (0..1) and proximity to the water (0..1).
    water: Vec<f32>,
    /// Per vertex: water depth over the bed (m).
    depth: Vec<f32>,
    bank: Vec<f32>,
    /// Canopy cover of the simulated trees, COVER_RES^2 over +-NEAR_HALF (0 for the gallery).
    cover: Vec<f32>,
    /// Radius of the grown forest (beyond it the ground shader paints distant forest).
    pub forest_radius: f32,
}

// ------------------------------------------------------------------------------------ noise

fn hash(ix: i32, iz: i32, seed: u32) -> f32 {
    let mut h = (ix as u32).wrapping_mul(0x8da6_b343) ^ (iz as u32).wrapping_mul(0xd816_3841) ^ seed.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^= h >> 15;
    h as f32 / u32::MAX as f32
}

fn value_noise(x: f32, z: f32, seed: u32) -> f32 {
    let (ix, iz) = (x.floor(), z.floor());
    let (fx, fz) = (x - ix, z - iz);
    let (ux, uz) = (fx * fx * (3.0 - 2.0 * fx), fz * fz * (3.0 - 2.0 * fz));
    let (ix, iz) = (ix as i32, iz as i32);
    let a = hash(ix, iz, seed);
    let b = hash(ix + 1, iz, seed);
    let c = hash(ix, iz + 1, seed);
    let d = hash(ix + 1, iz + 1, seed);
    let ab = a + (b - a) * ux;
    let cd = c + (d - c) * ux;
    ab + (cd - ab) * uz
}

/// Fractal noise in roughly [-1, 1].
fn fbm(x: f32, z: f32, octaves: u32, seed: u32) -> f32 {
    let (mut sum, mut amp, mut freq, mut norm) = (0.0, 1.0, 1.0, 0.0);
    for o in 0..octaves {
        sum += (value_noise(x * freq, z * freq, seed + o * 101) * 2.0 - 1.0) * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.03;
    }
    sum / norm
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Small deterministic PRNG (PCG-style) for the simulation.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let x = ((self.0 >> 18) ^ self.0) >> 27;
        (x as u32).rotate_right((self.0 >> 59) as u32)
    }
    fn f(&mut self) -> f32 {
        self.next() as f32 / 4294967296.0
    }
}

// ---------------------------------------------------------------------------------- terrain

/// Ridged multifractal in [0, 1]: sharp crests and gullies (detail on the eroded mountains).
fn ridged(x: f32, z: f32, octaves: u32, seed: u32) -> f32 {
    let (mut sum, mut amp, mut freq, mut norm, mut weight) = (0.0, 1.0, 1.0, 0.0, 1.0f32);
    for o in 0..octaves {
        let n = 1.0 - (value_noise(x * freq, z * freq, seed + o * 131) * 2.0 - 1.0).abs();
        let v = n * n * weight;
        weight = (v * 1.6).clamp(0.0, 1.0);
        sum += v * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.1;
    }
    sum / norm
}

/// Half size of the erosion simulation: the visible terrain (+-TERRAIN_HALF) plus room to move the
/// start point onto a river bank.
const SIM_HALF: f32 = 72_000.0;

/// The eroded landscape (erosion.rs). World (0, 0) sits at `start` in simulation coordinates.
struct Land {
    e: Erosion,
    sx: f32,
    sz: f32,
    h0: f32,
}

fn land() -> &'static Land {
    static L: OnceLock<Land> = OnceLock::new();
    L.get_or_init(|| {
        let e = Erosion::generate(SIM_HALF);
        let (sx, sz) = e.start_point();
        let mut l = Land { e, sx, sz, h0: 0.0 };
        l.h0 = raw_sample(&l, 0.0, 0.0).0;
        println!("Start: simulation ({sx:.0}, {sz:.0}), height {:.0} m", l.h0);
        l
    })
}

// --------------------------------------------------------------------------- real places

/// A real place on Earth (`--site lat,lon`): the world is the local tangent plane there (x east,
/// -z north, y up, with the Earth's curvature), heights from Copernicus GLO-30 / ETOPO and lakes,
/// rivers and forest from ESA WorldCover (globe/dem.rs, globe/landcover.rs) instead of the
/// erosion simulation.
pub struct Site {
    terrain: std::sync::Arc<crate::globe::dem::Terrain>,
    pub lat: f64,
    pub lon: f64,
    up: glam::DVec3,
    east: glam::DVec3,
    north: glam::DVec3,
    /// streams and rivers from the real heights (30 m grid over +-15 km)
    drainage: Option<Erosion>,
}

static SITE: OnceLock<Site> = OnceLock::new();

pub fn set_site(terrain: std::sync::Arc<crate::globe::dem::Terrain>, lat: f64, lon: f64) {
    let up = crate::globe::tiles::dir_from_lat_lon(lat, lon);
    let east = glam::DVec3::Z.cross(up).normalize();
    let north = up.cross(east);
    let mut s = Site { terrain, lat, lon, up, east, north, drainage: None };
    // drainage of the real terrain around the site: the small streams the land cover map misses
    let half = 15_360.0;
    let n = Erosion::GRID;
    let cell = 2.0 * half / (n - 1) as f32;
    let mut h = vec![0f32; n * n];
    par_rows(n, n, &mut h, |j, row| {
        for (i, v) in row.iter_mut().enumerate() {
            let (_, la, lo) = s.geo(i as f32 * cell - half, j as f32 * cell - half);
            *v = s.terrain.height(la, lo).max(0.0);
        }
    });
    s.drainage = Some(Erosion::from_heights(half, h, 3.0e6));
    let _ = SITE.set(s);
}

pub fn site() -> Option<&'static Site> {
    SITE.get()
}

impl Site {
    /// Direction from the Earth's centre under local (x, z), and its latitude / longitude.
    fn geo(&self, x: f32, z: f32) -> (glam::DVec3, f64, f64) {
        let d = (self.up * EARTH_RADIUS as f64 + self.east * x as f64 - self.north * z as f64).normalize();
        let (lat, lon) = crate::globe::tiles::lat_lon(d);
        (d, lat, lon)
    }

    /// Latitude / longitude under local (x, z).
    pub fn lat_lon(&self, x: f32, z: f32) -> (f64, f64) {
        let (_, lat, lon) = self.geo(x, z);
        (lat, lon)
    }

    /// Water fraction (0..1) from the 20 m land cover, bilinear so shores are smooth.
    fn water(&self, lat: f64, lon: f64) -> f32 {
        let c = &self.terrain.cover;
        let step = 1.0 / 6000.0; // 20 m cells (landcover.rs)
        let (fy, fx) = (lat / step - 0.5, lon / step - 0.5);
        let (y0, x0) = (fy.floor(), fx.floor());
        let (ty, tx) = ((fy - y0) as f32, (fx - x0) as f32);
        let at = |dy: f64, dx: f64| (c.class((y0 + dy + 0.5) * step, (x0 + dx + 0.5) * step) == 80) as u32 as f32;
        let a = at(0.0, 0.0) + (at(0.0, 1.0) - at(0.0, 0.0)) * tx;
        let b = at(1.0, 0.0) + (at(1.0, 1.0) - at(1.0, 0.0)) * tx;
        a + (b - a) * ty
    }

    /// Bed height (local y, with the curvature), water fraction, bank proximity and water depth.
    fn sample(&self, x: f32, z: f32) -> (f32, f32, f32, f32) {
        let (d, lat, lon) = self.geo(x, z);
        let h = self.terrain.height(lat, lon).max(0.0) as f64;
        let y = (EARTH_RADIUS as f64 + h) * d.dot(self.up) - EARTH_RADIUS as f64;
        let water = self.water(lat, lon);
        // bank: water within ~50 m
        let mut bank = water;
        let m = 1.0 / 111_320.0;
        for (k, r) in [(0.6f32, 25.0), (0.3, 55.0)] {
            for (a, b) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
                bank = bank.max(k * self.water(lat + a * r * m, lon + b * r * m / lat.to_radians().cos()));
            }
        }
        // lakes (the DEM holds their flat surface): a bed shelving down from the shore
        let y = y as f32;
        let lake_depth = (0.25 + 5.0 * smoothstep(0.5, 1.0, water)) * smoothstep(0.02, 0.5, water);
        let (mut y, mut water, mut bank, mut depth) = (y - lake_depth, water, bank, lake_depth);
        if let Some((dist, w, level, _)) = self.drainage.as_ref().and_then(|e| e.river(x, z)) {
            // a channel cut into the real valley floor along the drainage line: banks down to the
            // water surface, the bed below it
            let curv = (x * x + z * z) / (2.0 * EARTH_RADIUS);
            let surf = (level - curv).min(y + depth) - 0.35;
            let top = surf + (y - surf) * smoothstep(w - 1.0, w + 4.0, dist);
            let bed = top - (0.35 + 0.05 * w).min(2.5) * (1.0 - (dist / w).powi(2)).max(0.0);
            if bed < y {
                depth = depth.max(surf - bed);
                y = bed;
            } else {
                y = top.min(y);
            }
            water = water.max(1.0 - smoothstep(w - 1.5, w + 0.5, dist));
            bank = bank.max((1.0 - smoothstep(w, w + 12.0 + 1.5 * w, dist)) * 0.9);
        }
        (y, water, bank, depth.max(0.0))
    }

    /// Tree cover fraction (0..1) over ~60 m around the point.
    fn tree_cover(&self, x: f32, z: f32) -> f32 {
        let (_, lat, lon) = self.geo(x, z);
        self.terrain.cover.fractions(lat, lon, 60.0).map(|f| f[crate::globe::landcover::TREE]).unwrap_or(0.0)
    }

    /// Mountain zone shift (m): the forest scene's altitude zones (treeline ~1350 m, snow ~1700 m)
    /// are set for the generated landscape; real ranges at lower latitudes sit higher.
    pub fn alt_shift(&self) -> f32 {
        (900.0 - 60.0 * (self.lat.abs() as f32 - 46.0)).clamp(0.0, 1800.0)
    }
}

/// Mountain-ness (0..1) from the tectonic uplift field.
fn mountain_n(x: f32, z: f32) -> f32 {
    if let Some(s) = site() {
        let (_, lat, lon) = s.geo(x, z);
        return smoothstep(1800.0, 2800.0, s.terrain.height(lat, lon));
    }
    let l = land();
    smoothstep(0.35, 1.1, l.e.uplift(x + l.sx, z + l.sz))
}

/// Topographic wetness (0..1) from the simulated drainage: valley bottoms wet, ridges dry.
fn twi(x: f32, z: f32) -> f32 {
    if site().is_some() {
        return 0.3;
    }
    let l = land();
    l.e.wetness(x + l.sx, z + l.sz)
}

/// Raw terrain height, water (0..1) and bank proximity (0..1). The eroded heightfield (bicubic),
/// small-scale relief the 141 m simulation grid cannot hold, and the river channels carved along
/// the drainage network: flat floodplains, banks, and a water surface that only falls downstream.
fn raw_sample(l: &Land, x: f32, z: f32) -> (f32, f32, f32, f32) {
    let (sx, sz) = (x + l.sx, z + l.sz);
    let e = &l.e;
    let mtn = smoothstep(0.35, 1.1, e.uplift(sx, sz));
    let detail = 12.0 * fbm(sx / 900.0, sz / 900.0, 4, 19)
        + 4.0 * fbm(sx / 260.0, sz / 260.0, 3, 23)
        + 1.2 * fbm(sx / 70.0, sz / 70.0, 3, 29)
        + (0.8 + 1.6 * mtn) * fbm(sx / 28.0, sz / 28.0, 3, 31)
        + (0.35 + 0.5 * mtn) * fbm(sx / 11.0, sz / 11.0, 2, 33)
        + mtn * 80.0 * (ridged(sx / 1500.0, sz / 1500.0, 5, 67) - 0.45);
    let mut h = e.height(sx, sz) + detail;
    let (mut water, mut bank, mut depth) = (0.0, 0.0, 0.0f32);
    if let Some((d, w, level, area)) = e.river(sx, sz) {
        // floodplain: wider for larger rivers, narrow in the mountains (gorges)
        let fw = (w * 4.0 + 25.0 + 4.0 * (area / 1.0e6).sqrt()).min(450.0) * (1.0 - 0.6 * mtn);
        let floor_h = level + 0.9 + 0.6 * fbm(sx / 70.0, sz / 70.0, 2, 37);
        h = floor_h + (h - floor_h) * smoothstep(w, w + fw, d);
        let surf = level - 0.3;
        h = surf + (h - surf) * smoothstep(w - 1.0, w + 5.0, d);
        // the riverbed below the surface: deeper in the middle, deeper for bigger rivers
        let bed = h - (0.35 + 0.05 * w).min(2.5) * (1.0 - (d / w).powi(2)).max(0.0);
        depth = (surf - bed).max(0.0) * (d < w + 1.0) as u32 as f32;
        h = bed;
        water = 1.0 - smoothstep(w - 1.5, w + 0.5, d);
        bank = (1.0 - smoothstep(w, w + 20.0 + 1.5 * w, d)) * (0.85 + 0.15 * smoothstep(4.0, 12.0, w));
    }
    (h, water, bank, depth)
}

/// Terrain (bed) height (with the start clearing's reduced relief and the Earth's curvature),
/// water fraction, bank proximity and water depth over the bed.
pub fn terrain_sample(x: f32, z: f32) -> (f32, f32, f32, f32) {
    if let Some(s) = site() {
        return s.sample(x, z);
    }
    let l = land();
    let r2 = x * x + z * z;
    // The start clearing keeps a third of the local relief (around the height at the origin).
    let f = 0.35 + 0.65 * smoothstep(0.0, 250.0, r2.sqrt());
    let (h, water, bank, depth) = raw_sample(l, x, z);
    (l.h0 + (h - l.h0) * f - r2 / (2.0 * EARTH_RADIUS), water, bank, depth)
}


/// Biome fields at a point, all in [0, 1].
#[derive(Clone, Copy, Default)]
struct Biome {
    dens: f32,
    wet: f32,
    edge: f32,
    beech: f32,
    aspen: f32,
}

impl World {
    /// Dry ground (0..1) from the landform: convex knolls and ridges shed water and dry out,
    /// hollows collect it; plus the soil moisture. `e`: curvature scale (m). Sun-facing slopes
    /// are added in the shader (it knows the sun).
    pub fn dryness(&self, x: f32, z: f32, wet: f32, e: f32) -> f32 {
        let h = self.height_at(x, z);
        let ring = 0.25 * (self.height_at(x + e, z) + self.height_at(x - e, z) + self.height_at(x, z + e) + self.height_at(x, z - e));
        let convex = ((h - ring) / (0.06 * e)).clamp(-1.0, 1.0);
        smoothstep(-0.25, 0.55, 0.6 * (0.5 - wet) + 0.4 * convex + 0.12 * fbm(x / 140.0, z / 140.0, 3, 83))
    }

    /// Soil moisture in [0, 1]: valleys and wet patches, river banks; uplands are drier.
    pub fn wetness(&self, x: f32, z: f32) -> f32 {
        let (_, bank) = self.river_at(x, z);
        (0.15 + 0.6 * twi(x, z) + 0.4 * fbm(x / 380.0, z / 380.0, 4, 31) - 0.25 * mountain_n(x, z) + 0.8 * bank).clamp(0.0, 1.0)
    }

    /// Where forest can establish (0 = open country, 1 = forest site). Follows the land like a real
    /// cultural landscape: large forests and open country (~4.5 km), woods on steep slopes and hill
    /// tops, meadows on valley floors, ragged and wide edges (the seed spread fills them in), no
    /// trees in the water or on gravel bars, none above the treeline, and a clearing at the origin.
    pub fn forest_density(&self, x: f32, z: f32) -> f32 {
        let r2 = x * x + z * z;
        let clearing = ((r2.sqrt() - 24.0) / 10.0).clamp(0.0, 1.0);
        if let Some(s) = site() {
            // the forest stands where the satellite map says it does
            let (water, _) = self.river_at(x, z);
            return clearing * (1.0 - water) * s.tree_cover(x, z);
        }
        let e = 20.0;
        let h = self.height_at(x, z);
        let gx = (self.height_at(x + e, z) - h) / e;
        let gz = (self.height_at(x, z + e) - h) / e;
        let slope = (gx * gx + gz * gz).sqrt();
        let alt = h + r2 / (2.0 * EARTH_RADIUS);
        let (water, bank) = self.river_at(x, z);
        let large = fbm(x / 4500.0, z / 4500.0, 4, 13);
        let ragged = fbm(x / 380.0, z / 380.0, 3, 7);
        // meadows on flat, wet valley floors (farmed / grazed), woods on the slopes
        let floodplain = twi(x, z) * (1.0 - smoothstep(0.03, 0.15, slope));
        // a strip of alder / poplar lines the rivers (not on the gravel bars)
        let riparian = smoothstep(0.25, 0.6, bank);
        let v = 0.85 * large - 0.45 * floodplain + 1.6 * slope.min(0.35) + 0.22 * ragged + 0.2 + 0.7 * riparian;
        let treeline = 1.0 - smoothstep(1250.0, 1450.0, alt);
        let river = (1.0 - water) * (1.0 - smoothstep(0.8, 0.95, bank));
        clearing * treeline * river * smoothstep(-0.12, 0.28, v)
    }

    /// Water (0..1) and bank proximity (0..1) of the triangulated terrain at (x, z).
    pub fn river_at(&self, x: f32, z: f32) -> (f32, f32) {
        if self.water.is_empty() {
            return (0.0, 0.0);
        }
        (self.interp(&self.water, x, z), self.interp(&self.bank, x, z))
    }
}

fn biome(world: &World, x: f32, z: f32) -> Biome {
    let dens = world.forest_density(x, z);
    let wet = world.wetness(x, z);
    let edge = smoothstep(0.05, 0.45, dens) * (1.0 - smoothstep(0.55, 0.95, dens));
    let beech = smoothstep(-0.15, 0.3, fbm(x / 700.0, z / 700.0, 3, 41));
    let aspen = smoothstep(0.12, 0.42, fbm(x / 170.0, z / 170.0, 3, 53));
    Biome { dens, wet, edge, beech, aspen }
}

/// Weighted choice among (weight, variant list); variants of a species are equally likely.
fn pick(groups: &[(f32, &[usize])], r1: f32, r2: f32) -> Option<usize> {
    let live = || groups.iter().filter(|g| !g.1.is_empty() && g.0 > 0.0);
    let total: f32 = live().map(|g| g.0).sum();
    if total <= 0.0 {
        return None;
    }
    let mut acc = 0.0;
    for (w, vs) in live() {
        acc += w / total;
        if r1 <= acc {
            return Some(vs[((r2 * vs.len() as f32) as usize).min(vs.len() - 1)]);
        }
    }
    live().last().map(|g| g.1[0])
}

/// Run `f(row)` for every row on all cores; rows are written into `out` (row length `w`).
fn par_rows<T: Send + Copy + Default>(n_rows: usize, w: usize, out: &mut [T], f: impl Fn(usize, &mut [T]) + Sync) {
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let per = n_rows.div_ceil(threads);
    std::thread::scope(|s| {
        for (c, chunk) in out.chunks_mut(per * w).enumerate() {
            let f = &f;
            s.spawn(move || {
                for (k, row) in chunk.chunks_mut(w).enumerate() {
                    f(c * per + k, row);
                }
            });
        }
    });
}

// ---------------------------------------------------------------------- forest simulation

/// Species behaviour: mean seed dispersal distance (m), seeds per tree per generation, shade
/// tolerance (survival under an older crown), minimum spacing to a neighbour (m), lifespan and
/// age of first seeding (generations of ~15 years).
#[derive(Clone, Copy)]
struct Habit {
    disperse: f32,
    seeds: u32,
    shade: f32,
    space: f32,
    life: u8,
    mature: u8,
}

// beech: heavy nuts (gravity, jays) and very shade tolerant; alder: water / wind; poplar: cotton
// seeds on the wind, pioneer, intolerant; aspen: clonal root suckers form groves; hazel: nuts
// cached by rodents, understory.
const HABITS: [Habit; 6] = [
    Habit { disperse: 14.0, seeds: 3, shade: 0.75, space: 5.0, life: 20, mature: 3 }, // beech
    Habit { disperse: 22.0, seeds: 3, shade: 0.30, space: 4.5, life: 7, mature: 1 },  // alder
    Habit { disperse: 45.0, seeds: 3, shade: 0.10, space: 6.0, life: 6, mature: 1 },  // poplar
    Habit { disperse: 9.0, seeds: 3, shade: 0.12, space: 3.5, life: 5, mature: 1 },   // aspen
    Habit { disperse: 7.0, seeds: 3, shade: 0.85, space: 3.0, life: 6, mature: 1 },   // hazel
    Habit { disperse: 15.0, seeds: 3, shade: 0.50, space: 5.0, life: 12, mature: 2 }, // other
];
/// ~150 years of succession.
const GENERATIONS: u8 = 10;

fn suitability(b: &Biome) -> [f32; 6] {
    [
        0.35 + 2.6 * b.beech * (1.0 - b.wet),
        3.2 * b.wet * b.wet,
        0.08 + 1.4 * b.edge * (0.4 + b.wet),
        0.08 + 2.4 * b.aspen,
        0.25 + 1.1 * b.edge,
        0.5,
    ]
}

#[derive(Clone, Copy)]
struct Seedling {
    x: f32,
    z: f32,
    sp: u8,
    gen: u8,
    alive: bool,
}

/// Uniform grid of biome samples over the simulated disc (8 m).
struct BiomeGrid {
    half: f32,
    cell: f32,
    n: usize,
    b: Vec<Biome>,
}

impl BiomeGrid {
    fn new(world: &World, half: f32, cell: f32) -> Self {
        let n = (2.0 * half / cell).ceil() as usize + 1;
        let mut b = vec![Biome::default(); n * n];
        par_rows(n, n, &mut b, |j, row| {
            for (i, v) in row.iter_mut().enumerate() {
                *v = biome(world, i as f32 * cell - half, j as f32 * cell - half);
            }
        });
        BiomeGrid { half, cell, n, b }
    }
    fn at(&self, x: f32, z: f32) -> &Biome {
        let i = (((x + self.half) / self.cell).round() as isize).clamp(0, self.n as isize - 1) as usize;
        let j = (((z + self.half) / self.cell).round() as isize).clamp(0, self.n as isize - 1) as usize;
        &self.b[j * self.n + i]
    }
}

/// Grow the canopy inside a disc of `radius`: founder pairs, then GENERATIONS of seed dispersal
/// with site, crowding and shade filters. `avail[k]`: species k has variants.
fn grow_forest(world: &World, radius: f32, avail: [bool; 6]) -> Vec<Seedling> {
    let bg = BiomeGrid::new(world, radius, if radius > 3000.0 { 16.0 } else { 8.0 });
    let cell = 10.0f32;
    let n = (2.0 * radius / cell).ceil() as usize + 1;
    let mut grid: Vec<Vec<u32>> = vec![Vec::new(); n * n];
    let gi = |x: f32, z: f32| -> Option<usize> {
        let i = ((x + radius) / cell) as isize;
        let j = ((z + radius) / cell) as isize;
        (i >= 0 && j >= 0 && (i as usize) < n && (j as usize) < n).then(|| j as usize * n + i as usize)
    };
    let mut trees: Vec<Seedling> = Vec::new();
    let mut rng = Rng(0x5eed_f0e5_7123_4567);

    let suit_of = |b: &Biome| {
        let mut s = suitability(b);
        for k in 0..6 {
            if !avail[k] {
                s[k] = 0.0;
            }
        }
        s
    };
    // Try to establish a tree; returns false when crowded / shaded out.
    let mut try_add = |trees: &mut Vec<Seedling>, grid: &mut Vec<Vec<u32>>, rng: &mut Rng, t: Seedling| -> bool {
        if t.x * t.x + t.z * t.z > radius * radius {
            return false;
        }
        let Some(c) = gi(t.x, t.z) else { return false };
        let h = HABITS[t.sp as usize];
        let (ci, cj) = ((c % n) as isize, (c / n) as isize);
        let mut shaded = 0;
        for dj in -1..=1 {
            for di in -1..=1 {
                let (ii, jj) = (ci + di, cj + dj);
                if ii < 0 || jj < 0 || ii >= n as isize || jj >= n as isize {
                    continue;
                }
                for &k in &grid[jj as usize * n + ii as usize] {
                    let o = trees[k as usize];
                    if !o.alive {
                        continue;
                    }
                    let d2 = (o.x - t.x).powi(2) + (o.z - t.z).powi(2);
                    let sp = h.space.max(HABITS[o.sp as usize].space) * 0.8;
                    if d2 < sp * sp {
                        return false;
                    }
                    // shade comes from a clearly taller (two generations older) crown overhead
                    if d2 < 5.5 * 5.5 && o.gen + 2 <= t.gen {
                        shaded += 1;
                    }
                }
            }
        }
        if shaded > 0 && rng.f() > h.shade.powi(shaded) {
            return false;
        }
        grid[c].push(trees.len() as u32);
        trees.push(t);
        true
    };

    // Founders: in pairs, on a jittered 80 m lattice, where the site allows forest.
    let lattice = 60.0;
    let m = (radius / lattice).ceil() as i32;
    for gz in -m..m {
        for gx in -m..m {
            let x = (gx as f32 + rng.f()) * lattice;
            let z = (gz as f32 + rng.f()) * lattice;
            let b = *bg.at(x, z);
            if rng.f() > 0.85 * smoothstep(0.3, 0.6, b.dens) {
                continue;
            }
            let s = suit_of(&b);
            let total: f32 = s.iter().sum();
            if total <= 0.0 {
                continue;
            }
            let mut r = rng.f() * total;
            let mut sp = 0;
            for (k, w) in s.iter().enumerate() {
                if r < *w {
                    sp = k;
                    break;
                }
                r -= w;
            }
            if try_add(&mut trees, &mut grid, &mut rng, Seedling { x, z, sp: sp as u8, gen: 0, alive: true }) {
                let a = rng.f() * std::f32::consts::TAU;
                let d = 4.0 + 3.0 * rng.f();
                try_add(&mut trees, &mut grid, &mut rng, Seedling { x: x + d * a.cos(), z: z + d * a.sin(), sp: sp as u8, gen: 0, alive: true });
            }
        }
    }
    let founders = trees.len();

    // Generations of succession: disturbance, mortality, then seed dispersal and establishment.
    // Pioneers (aspen, poplar, alder) colonise open ground and gaps quickly but need light and
    // live short; shade-tolerant beech and hazel establish under them and take over. Windthrow
    // gaps of all sizes reset patches, so the forest ends up a mosaic of stand ages and species.
    // Canopy trees that grow up through the pioneers and overtop them (hazel stays an understory
    // shrub and never does).
    let tolerant = |sp: u8| sp == 0 || sp == 5;
    let mut died = 0usize;
    for gen in 1..GENERATIONS {
        // -- disturbance: small gaps (one to a few crowns) and rare large blowdowns
        let disc = std::f32::consts::PI * radius * radius;
        let small = (disc * 0.004 / 900.0) as usize;
        // large blowdowns only early on: a recent one would still be bare, a few saplings in rows
        let large = if gen + 3 < GENERATIONS { (disc * 0.002 / 25_000.0) as usize } else { 0 };
        for g in 0..small + large {
            let r = if g < small { 8.0 + 22.0 * rng.f() } else { 60.0 + 120.0 * rng.f() };
            let a = rng.f() * std::f32::consts::TAU;
            let d = radius * rng.f().sqrt();
            let (cx, cz) = (d * a.cos(), d * a.sin());
            let cells = (r * 1.25 / cell).ceil() as isize + 1;
            let Some(c) = gi(cx, cz) else { continue };
            let (ci, cj) = ((c % n) as isize, (c / n) as isize);
            for dj in -cells..=cells {
                for di in -cells..=cells {
                    let (ii, jj) = (ci + di, cj + dj);
                    if ii < 0 || jj < 0 || ii >= n as isize || jj >= n as isize {
                        continue;
                    }
                    for &k in &grid[jj as usize * n + ii as usize] {
                        let t = &mut trees[k as usize];
                        // ragged gap edges; a few survivors (seed trees) stay standing in the gap
                        let rr = r * (0.75 + 0.5 * hash(k as i32, gen as i32, 17));
                        if t.alive && (t.x - cx).powi(2) + (t.z - cz).powi(2) < rr * rr && hash(k as i32, gen as i32, 19) > 0.06 {
                            t.alive = false;
                            died += 1;
                        }
                    }
                }
            }
        }
        // -- old age, and pioneers overtopped by the shade-tolerant trees growing up beneath them
        for k in 0..trees.len() {
            let t = trees[k];
            if !t.alive {
                continue;
            }
            let h = HABITS[t.sp as usize];
            let age = gen - t.gen;
            let mut p_die: f32 = if age > h.life { 0.5 } else { 0.0 };
            if matches!(t.sp, 1..=3) && age >= 2 {
                if let Some(c) = gi(t.x, t.z) {
                    let (ci, cj) = ((c % n) as isize, (c / n) as isize);
                    let mut under = 0;
                    for dj in -1..=1 {
                        for di in -1..=1 {
                            let (ii, jj) = (ci + di, cj + dj);
                            if ii < 0 || jj < 0 || ii >= n as isize || jj >= n as isize {
                                continue;
                            }
                            for &q in &grid[jj as usize * n + ii as usize] {
                                let o = trees[q as usize];
                                if o.alive && tolerant(o.sp) && o.gen + 2 <= gen && (o.x - t.x).powi(2) + (o.z - t.z).powi(2) < 49.0 {
                                    under += 1;
                                }
                            }
                        }
                    }
                    if under >= 2 {
                        p_die = p_die.max(0.3);
                    }
                }
            }
            if p_die > 0.0 && rng.f() < p_die {
                trees[k].alive = false;
                died += 1;
            }
        }
        // -- seed dispersal and establishment (only trees old enough to bear seed)
        let parents = trees.len();
        for p in 0..parents {
            let par = trees[p];
            let h = HABITS[par.sp as usize];
            if !par.alive || gen - par.gen < h.mature {
                continue;
            }
            for _ in 0..h.seeds + 1 {
                let d = (-h.disperse * (1.0 - rng.f()).max(1e-6).ln()).clamp(1.5, h.disperse * 5.0);
                let a = rng.f() * std::f32::consts::TAU;
                let (x, z) = (par.x + d * a.cos(), par.z + d * a.sin());
                let b = *bg.at(x, z);
                let s = suit_of(&b);
                let smax = s.iter().cloned().fold(1e-6, f32::max);
                // Forest or open land, not a thin scatter: establishment switches on over a narrow
                // band of site quality, with the threshold wandering (~90 m) so stand edges are ragged
                // and groups and clearings form at the margins.
                let site = b.dens + 0.22 * fbm(x / 90.0, z / 90.0, 3, 113);
                let p_est = smoothstep(0.32, 0.44, site) * (0.1 + 0.9 * s[par.sp as usize] / smax);
                if rng.f() > p_est {
                    continue;
                }
                if try_add(&mut trees, &mut grid, &mut rng, Seedling { x, z, sp: par.sp, gen, alive: true }) {
                    // Pioneers come up in groups, not as lone saplings: aspen suckers from its roots
                    // into groves, poplar and alder seed into thickets around the first one.
                    let clump = match par.sp {
                        3 => 2,
                        1 | 2 => 1,
                        _ => 0,
                    };
                    for _ in 0..clump {
                        let a = rng.f() * std::f32::consts::TAU;
                        let d = 3.0 + 4.0 * rng.f();
                        try_add(&mut trees, &mut grid, &mut rng, Seedling { x: x + d * a.cos(), z: z + d * a.sin(), sp: par.sp, gen, alive: true });
                    }
                }
            }
        }
        // compact the spatial grid now and then (dead trees no longer block anything)
        if gen % 3 == 0 {
            for cv in grid.iter_mut() {
                cv.retain(|&k| trees[k as usize].alive);
            }
        }
    }
    let total = trees.len();
    trees.retain(|t| t.alive);
    let mut per = [0usize; 6];
    for t in &trees {
        per[t.sp as usize] += 1;
    }
    println!(
        "Forest succession: {} established, {} died (age, overtopped, windthrow); living beech/alder/poplar/aspen/hazel/other {:?}",
        total, died, per
    );
    println!("Forest growth: {} founders -> {} trees after {} generations", founders, trees.len(), GENERATIONS);
    trees
}

// ------------------------------------------------------------------------------------ world

impl World {
    /// Flat test scene: every variant on a flat plane, one row per species.
    pub fn gallery(trees: &[TreeAsset], cat: &Catalog) -> Self {
        let mut w = Self::build_flat();
        let mut z = 0.0f32;
        for sp in &cat.species {
            let vs: Vec<usize> = sp.variants.clone().collect();
            let width: f32 = vs.iter().map(|&v| trees[v].bmax.x - trees[v].bmin.x + 1.5).sum();
            let depth = vs.iter().map(|&v| trees[v].bmax.z - trees[v].bmin.z).fold(0.0f32, f32::max);
            let mut x = -width * 0.5;
            println!("gallery row {:<24} z {:7.1} .. {:7.1}, x {:6.1} .. {:6.1}", sp.name, z, z - depth, -width * 0.5, width * 0.5);
            for &v in &vs {
                let t = &trees[v];
                let tw = t.bmax.x - t.bmin.x + 1.5;
                let pos = Vec3::new(x + tw * 0.5 - (t.bmin.x + t.bmax.x) * 0.5, 0.0, z - depth * 0.5 - (t.bmin.z + t.bmax.z) * 0.5);
                let xform = Affine3A::from_translation(pos);
                let center = xform.transform_point3((t.bmin + t.bmax) * 0.5);
                w.trees.push(TreeInstance { xform, center, variant: v as u8, seed: 0, color: (v % 7) as u8 });
                x += tw;
            }
            z -= depth + 2.0;
        }
        w
    }

    /// Terrain, the grown canopy (within `radius` of the origin), understory and grass layers.
    pub fn generate(trees: &[TreeAsset], cat: &Catalog, radius: f32) -> Self {
        let t0 = std::time::Instant::now();
        let mut world = Self::build_terrain();
        println!("Terrain: {}x{} graded grid to +-{:.0} km, {:.1}M triangles ({:.2}s)", world.xs.len(), world.xs.len(), TERRAIN_HALF / 1000.0, world.terrain.tri_count() as f64 / 1e6, t0.elapsed().as_secs_f32());
        world.forest_radius = radius;

        let beech = cat.variants_of("Beech");
        let alder = cat.variants_of("Alder");
        let poplar = cat.variants_of("Poplar");
        let aspen = cat.variants_of("Aspen");
        let hazel = cat.variants_of("Hazel");
        let heather = cat.variants_of("Heather");
        let huck = cat.variants_of("Huckleberry");
        let fern = cat.variants_of("Fern");
        let g_clumps = cat.variants_of("Grass_Clumps");
        let g_wild = cat.variants_of("Wild_Grass");
        let g_tall = cat.variants_of("Tall_Grass");
        let g_tufted = cat.variants_of("Tufted_Grass");
        let g_dry = cat.variants_of("Dry_Grass");
        let g_cotton = cat.variants_of("Cotton_Grass");
        let known: Vec<usize> =
            [&beech, &alder, &poplar, &aspen, &hazel, &heather, &huck, &fern, &g_clumps, &g_wild, &g_tall, &g_tufted, &g_dry, &g_cotton]
                .iter()
                .flat_map(|v| v.iter().copied())
                .collect();
        let other_trees: Vec<usize> = (0..trees.len()).filter(|v| !known.contains(v) && cat.variants[*v].kind == Kind::Tree).collect();
        let other_small: Vec<usize> =
            (0..trees.len()).filter(|v| !known.contains(v) && matches!(cat.variants[*v].kind, Kind::Shrub | Kind::Plant)).collect();
        let other_grass: Vec<usize> = (0..trees.len()).filter(|v| !known.contains(v) && cat.variants[*v].kind == Kind::Grass).collect();

        // ---- canopy: seed-dispersal growth
        let t1 = std::time::Instant::now();
        let mut by_species: Vec<Vec<usize>> = vec![beech.clone(), alder.clone(), poplar.clone(), aspen.clone(), hazel.clone(), other_trees.clone()];
        for vs in &mut by_species {
            vs.sort_by(|a, b| cat.variants[*b].height.partial_cmp(&cat.variants[*a].height).unwrap()); // tallest first
        }
        let avail = [0, 1, 2, 3, 4, 5].map(|k| !by_species[k].is_empty());
        let grown = grow_forest(&world, radius, avail);
        let mut rng = Rng(0x1234_5678_9abc_def0);
        let mut counts = [0usize; 5];
        let mut canopy: Vec<usize> = Vec::new(); // indices into world.trees of the simulated trees
        for s in &grown {
            let vs = &by_species[s.sp as usize];
            // Old trees (early generations) are the large variants, young ones the small.
            let age = s.gen as f32 / (GENERATIONS - 1).max(1) as f32;
            let k = (((age * 0.8 + rng.f() * 0.35) * vs.len() as f32) as usize).min(vs.len() - 1);
            let variant = vs[k];
            let y = world.height_at(s.x, s.z) - 0.15;
            let xform = Affine3A::from_scale_rotation_translation(
                Vec3::splat(0.85 + 0.3 * rng.f()),
                Quat::from_rotation_y(rng.f() * std::f32::consts::TAU),
                Vec3::new(s.x, y, s.z),
            );
            let t = &trees[variant];
            let center = xform.transform_point3((t.bmin + t.bmax) * 0.5);
            let seed = 1 + (rng.f() * 8190.0) as u16;
            canopy.push(world.trees.len());
            world.trees.push(TreeInstance { xform, center, variant: variant as u8, seed, color: (rng.next() % 7) as u8 });
            counts[0] += 1;
        }
        world.cover = world.splat_cover(trees, cat);
        println!("Canopy simulated and placed in {:.2}s", t1.elapsed().as_secs_f32());

        // ---- understory and grass, driven by the simulated canopy cover
        let under_radius = radius.min(900.0);
        let grass_radius = radius.min(600.0);
        let layers = [(1u32, 3.4, under_radius), (2, 2.1, under_radius), (3, 2.3, grass_radius)];
        for (layer, step, rad) in layers {
            let count = (rad * 2.0 / step) as i32;
            let hs = |gx: i32, gz: i32, k: u32| hash(gx, gz, k + layer * 100);
            for gz in -count / 2..count / 2 {
                for gx in -count / 2..count / 2 {
                    let x = (gx as f32 + 0.5 + (hs(gx, gz, 1) - 0.5) * 0.9) * step;
                    let z = (gz as f32 + 0.5 + (hs(gx, gz, 2) - 0.5) * 0.9) * step;
                    if x * x + z * z > rad * rad {
                        continue;
                    }
                    let (water, bank) = world.river_at(x, z);
                    if water > 0.3 || (layer != 2 && bank > 0.9) {
                        continue; // no plants in the water, gravel bars only get ferns
                    }
                    let mut b = biome(&world, x, z);
                    // canopy from the simulated trees (clearing at the origin kept)
                    b.dens = world.cover_at(x, z).min(world.forest_density(x, z).max(0.05) * 1.5).clamp(0.0, 1.0);
                    b.edge = smoothstep(0.05, 0.45, b.dens) * (1.0 - smoothstep(0.55, 0.95, b.dens));
                    let (r_place, r1, r2) = (hs(gx, gz, 3), hs(gx, gz, 4), hs(gx, gz, 5));
                    let choice = match layer {
                        1 => {
                            let open = 1.0 - b.dens;
                            let clump = smoothstep(-0.15, 0.35, fbm(x / 28.0, z / 28.0, 3, 61));
                            let w_heather = 1.6 * open * open * (1.0 - b.wet) * clump;
                            let w_huck = 0.9 * b.edge + 0.25 * b.dens * clump;
                            let w_other = if other_small.is_empty() { 0.0 } else { 0.2 };
                            if r_place > (w_heather + w_huck + w_other).min(0.85) {
                                continue;
                            }
                            pick(&[(w_heather, &heather), (w_huck, &huck), (w_other, &other_small)], r1, r2)
                        }
                        3 => {
                            let open = 1.0 - b.dens;
                            let dry = world.dryness(x, z, b.wet, 12.0);
                            let patchy = smoothstep(-0.3, 0.25, fbm(x / 18.0, z / 18.0, 3, 97));
                            let w = (open * 1.15 + 0.35 * b.edge) * (0.85 + 0.15 * patchy) * smoothstep(0.0, 0.25, open + 0.5 * b.edge);
                            if r_place > w.min(0.97) {
                                continue;
                            }
                            let tallz = smoothstep(0.1, 0.5, fbm(x / 45.0, z / 45.0, 3, 101));
                            pick(
                                &[
                                    (1.6 * (1.0 - dry) * (1.0 - 0.6 * b.wet), &g_wild),
                                    (1.6 * (1.0 - dry), &g_clumps),
                                    (1.4 * dry * (1.0 - b.wet), &g_dry),
                                    (2.0 * b.wet * b.wet, &g_cotton),
                                    (0.9 * tallz + 0.8 * b.edge, &g_tall),
                                    (0.6 * tallz * (1.0 - dry) + 0.3 * b.edge, &g_tufted),
                                    (0.5, &other_grass),
                                ],
                                r1,
                                r2,
                            )
                        }
                        _ => {
                            let clump = smoothstep(-0.25, 0.3, fbm(x / 22.0, z / 22.0, 3, 71));
                            let w = b.dens.powf(1.3) * (0.35 + 0.65 * b.wet) * clump + 0.06 * b.edge;
                            if r_place > (1.4 * w).min(0.85) {
                                continue;
                            }
                            pick(&[(1.0, &fern)], r1, r2)
                        }
                    };
                    let Some(variant) = choice else { continue };
                    let yaw = hs(gx, gz, 6) * std::f32::consts::TAU;
                    let scale = 0.7 + 0.6 * hs(gx, gz, 7);
                    let y = world.height_at(x, z) - 0.03;
                    let mut rot = Quat::from_rotation_y(yaw);
                    if layer == 3 {
                        // Grass patches are flat discs a few metres across: follow the slope.
                        let e = 1.0;
                        let nrm = Vec3::new(
                            world.height_at(x - e, z) - world.height_at(x + e, z),
                            2.0 * e,
                            world.height_at(x, z - e) - world.height_at(x, z + e),
                        )
                        .normalize();
                        rot = Quat::from_rotation_arc(Vec3::Y, nrm) * rot;
                    }
                    let xform = Affine3A::from_scale_rotation_translation(Vec3::splat(scale), rot, Vec3::new(x, y, z));
                    let t = &trees[variant];
                    let center = xform.transform_point3((t.bmin + t.bmax) * 0.5);
                    let seed = 1 + (hs(gx, gz, 8) * 8190.0) as u16;
                    let color = (gx + 3 * gz + 2 * layer as i32).rem_euclid(7) as u8;
                    world.trees.push(TreeInstance { xform, center, variant: variant as u8, seed, color });
                    counts[layer as usize] += 1;
                }
            }
        }

        let _ = &canopy;
        println!(
            "Placed {} trees, {} shrubs, {} ferns/plants, {} grass patches ({:.2}s)",
            counts[0],
            counts[1],
            counts[2],
            counts[3],
            t0.elapsed().as_secs_f32()
        );
        let mut per = vec![0usize; trees.len()];
        for t in &world.trees {
            if (t.variant as usize) < trees.len() {
                per[t.variant as usize] += 1;
            }
        }
        for sp in &cat.species {
            let n: usize = sp.variants.clone().map(|v| per[v]).sum();
            println!("  {:<24} {:>7}", sp.name, n);
        }
        world
    }

    fn build_flat() -> Self {
        let xs: Vec<f32> = (0..=256).map(|i| i as f32 * 12.0 - 1536.0).collect();
        let heights = vec![0.0; xs.len() * xs.len()];
        let terrain = Self::mesh(&xs, &heights, &[], &[]);
        World { terrain, trees: Vec::new(), xs, heights, water: Vec::new(), bank: Vec::new(), depth: Vec::new(), cover: Vec::new(), forest_radius: 0.0 }
    }

    fn build_terrain() -> Self {
        // Graded coordinates: 3 m inside +-NEAR_HALF, then growing 3.5% per cell up to 300 m.
        let mut half = vec![0.0f32];
        let mut x = 0.0;
        while x < NEAR_HALF {
            x += NEAR_STEP;
            half.push(x);
        }
        let mut step = NEAR_STEP;
        while x < TERRAIN_HALF {
            step = (step * 1.02).min(200.0);
            x += step;
            half.push(x.min(TERRAIN_HALF));
        }
        let mut xs: Vec<f32> = half.iter().skip(1).rev().map(|v| -v).collect();
        xs.extend_from_slice(&half);
        let n = xs.len();
        let mut samples = vec![(0.0f32, 0.0f32, 0.0f32, 0.0f32); n * n];
        par_rows(n, n, &mut samples, |j, row| {
            for (i, s) in row.iter_mut().enumerate() {
                *s = terrain_sample(xs[i], xs[j]);
            }
        });
        let heights: Vec<f32> = samples.iter().map(|s| s.0).collect();
        let water: Vec<f32> = samples.iter().map(|s| s.1).collect();
        let bank: Vec<f32> = samples.iter().map(|s| s.2).collect();
        let depth: Vec<f32> = samples.iter().map(|s| s.3).collect();
        let terrain = Self::mesh(&xs, &heights, &depth, &bank);
        World { terrain, trees: Vec::new(), xs, heights, water, bank, depth, cover: Vec::new(), forest_radius: 0.0 }
    }

    fn mesh(xs: &[f32], heights: &[f32], water: &[f32], bank: &[f32]) -> MeshData {
        let n = xs.len();
        let mut positions = vec![0.0f32; n * n * 3];
        let mut normals = vec![0.0f32; n * n * 3];
        par_rows(n, n * 3, &mut positions, |j, row| {
            for i in 0..n {
                row[i * 3..i * 3 + 3].copy_from_slice(&[xs[i], heights[j * n + i], xs[j]]);
            }
        });
        par_rows(n, n * 3, &mut normals, |j, row| {
            for i in 0..n {
                let (il, ir) = (i.saturating_sub(1), (i + 1).min(n - 1));
                let (jd, ju) = (j.saturating_sub(1), (j + 1).min(n - 1));
                let dx = (heights[j * n + ir] - heights[j * n + il]) / (xs[ir] - xs[il]).max(1e-3);
                let dz = (heights[ju * n + i] - heights[jd * n + i]) / (xs[ju] - xs[jd]).max(1e-3);
                let nrm = Vec3::new(-dx, 1.0, -dz).normalize();
                row[i * 3..i * 3 + 3].copy_from_slice(&[nrm.x, nrm.y, nrm.z]);
            }
        });
        let cells = n - 1;
        let mut indices = Vec::with_capacity(cells * cells * 6);
        for j in 0..cells {
            for i in 0..cells {
                let a = (j * n + i) as u32;
                let b = a + 1;
                let c = a + n as u32;
                let d = c + 1;
                indices.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }
        let tri_count = indices.len() / 3;
        let materials = vec![MAT_TERRAIN; (tri_count + 3) / 4 * 4];
        // UV channel carries the water data to the terrain shader: u = water depth over the bed
        // (m), v = bank proximity.
        let mut uvs = vec![0.0; n * n * 2];
        if !water.is_empty() {
            for k in 0..n * n {
                uvs[k * 2] = water[k];
                uvs[k * 2 + 1] = bank[k];
            }
        }
        MeshData { name: "terrain".into(), positions, normals, uvs, indices, materials }
    }

    /// Canopy cover of the trees (Kind::Tree, >= 4 m) on the COVER_RES grid, softened.
    fn splat_cover(&self, trees: &[TreeAsset], cat: &Catalog) -> Vec<f32> {
        let res = COVER_RES;
        let cell = 2.0 * NEAR_HALF / res as f32;
        let mut cover = vec![0.0f32; res * res];
        for t in &self.trees {
            let v = t.variant as usize;
            if v >= cat.variants.len() || cat.variants[v].kind != Kind::Tree || cat.variants[v].height < 4.0 {
                continue;
            }
            let a = &trees[v];
            let scale = t.xform.matrix3.x_axis.length();
            let rc = 0.5 * (a.bmax.x - a.bmin.x).max(a.bmax.z - a.bmin.z) * scale;
            splat(&mut cover, res, cell, t.center.x, t.center.z, rc, 0.9);
        }
        blur(&blur(&blur(&cover, res), res), res)
    }

    fn cover_at(&self, x: f32, z: f32) -> f32 {
        if self.cover.is_empty() {
            return 0.0;
        }
        let res = COVER_RES;
        let cell = 2.0 * NEAR_HALF / res as f32;
        let i = (((x + NEAR_HALF) / cell) as isize).clamp(0, res as isize - 1) as usize;
        let j = (((z + NEAR_HALF) / cell) as isize).clamp(0, res as isize - 1) as usize;
        self.cover[j * res + i]
    }

    /// Near ground mask (COVER_RES^2 RGBA8 over +-NEAR_HALF): R canopy cover, G moisture,
    /// B dryness, A understory density. The terrain shader blends the scanned ground with it.
    pub fn ground_mask(&self, trees: &[TreeAsset], cat: &Catalog) -> Vec<u8> {
        let res = COVER_RES;
        let cell = 2.0 * NEAR_HALF / res as f32;
        let mut under = vec![0.0f32; res * res];
        for t in &self.trees {
            let v = t.variant as usize;
            if v >= cat.variants.len() || !matches!(cat.variants[v].kind, Kind::Shrub | Kind::Plant) && !(cat.variants[v].kind == Kind::Tree && cat.variants[v].height < 4.0) {
                continue;
            }
            let a = &trees[v];
            let rc = 0.8 * (a.bmax.x - a.bmin.x).max(a.bmax.z - a.bmin.z) * t.xform.matrix3.x_axis.length();
            splat(&mut under, res, cell, t.center.x, t.center.z, rc, 0.5);
        }
        let under = blur(&under, res);
        let mut px = vec![0u8; res * res * 4];
        par_rows(res, res * 4, &mut px, |j, row| {
            for i in 0..res {
                let x = (i as f32 + 0.5) * cell - NEAR_HALF;
                let z = (j as f32 + 0.5) * cell - NEAR_HALF;
                let wet = self.wetness(x, z);
                let dry = self.dryness(x, z, wet, 12.0);
                let k = j * res + i;
                let c = if self.cover.is_empty() { 0.0 } else { self.cover[k] };
                let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                row[i * 4..i * 4 + 4].copy_from_slice(&[q(c), q(wet), q(dry), q(under[k])]);
            }
        });
        px
    }

    /// Far ground mask (res^2 RGBA8 over +-TERRAIN_HALF): R canopy cover of the simulated trees
    /// (crown area per texel) inside the grown forest, G moisture, B dryness, A forest site
    /// density beyond the grown forest (painted canopy there).
    pub fn far_mask(&self, res: usize, trees: &[TreeAsset], cat: &Catalog) -> Vec<u8> {
        let cell = 2.0 * TERRAIN_HALF / res as f32;
        let mut crown = vec![0.0f32; res * res];
        for t in &self.trees {
            let v = t.variant as usize;
            if v >= cat.variants.len() || cat.variants[v].kind != Kind::Tree || cat.variants[v].height < 4.0 {
                continue;
            }
            let a = &trees[v];
            let rc = 0.5 * (a.bmax.x - a.bmin.x).max(a.bmax.z - a.bmin.z) * t.xform.matrix3.x_axis.length();
            let i = (((t.center.x + TERRAIN_HALF) / cell) as isize).clamp(0, res as isize - 1) as usize;
            let j = (((t.center.z + TERRAIN_HALF) / cell) as isize).clamp(0, res as isize - 1) as usize;
            crown[j * res + i] += std::f32::consts::PI * rc * rc * 0.8 / (cell * cell);
        }
        let crown = blur(&crown, res);
        let fr = self.forest_radius;
        let mut px = vec![0u8; res * res * 4];
        par_rows(res, res * 4, &mut px, |j, row| {
            for i in 0..res {
                let x = (i as f32 + 0.5) * cell - TERRAIN_HALF;
                let z = (j as f32 + 0.5) * cell - TERRAIN_HALF;
                let wet = self.wetness(x, z);
                let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                let inside = 1.0 - smoothstep(fr * 0.97, fr, (x * x + z * z).sqrt());
                let site = if inside < 1.0 { self.forest_density(x, z) * (1.0 - inside) } else { 0.0 };
                row[i * 4..i * 4 + 4].copy_from_slice(&[q(crown[j * res + i].min(1.0) * inside), q(wet), q(self.dryness(x, z, wet, cell)), q(site)]);
            }
        });
        px
    }

    /// Height of the ground or, over water, of the water surface (for the camera).
    pub fn surface_at(&self, x: f32, z: f32) -> f32 {
        let h = self.height_at(x, z);
        if self.depth.is_empty() {
            h
        } else {
            h + self.interp(&self.depth, x, z)
        }
    }

    /// Height of the triangulated terrain surface (matches the mesh exactly).
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        self.interp(&self.heights, x, z)
    }

    /// Per-vertex value interpolated over the terrain triangles at (x, z).
    fn interp(&self, values: &[f32], x: f32, z: f32) -> f32 {
        let n = self.xs.len();
        let find = |v: f32| -> (usize, f32) {
            let k = self.xs.partition_point(|&c| c <= v).clamp(1, n - 1) - 1;
            let t = ((v - self.xs[k]) / (self.xs[k + 1] - self.xs[k])).clamp(0.0, 1.0);
            (k, t)
        };
        let ((i, u), (j, v)) = (find(x), find(z));
        let h = |a: usize, b: usize| values[b * n + a];
        // Triangles per cell: (a, c, b) and (b, c, d) split along the b-c diagonal (u + v = 1).
        if u + v <= 1.0 {
            h(i, j) + (h(i + 1, j) - h(i, j)) * u + (h(i, j + 1) - h(i, j)) * v
        } else {
            h(i + 1, j + 1) + (h(i, j + 1) - h(i + 1, j + 1)) * (1.0 - u) + (h(i + 1, j) - h(i + 1, j + 1)) * (1.0 - v)
        }
    }
}

fn splat(grid: &mut [f32], res: usize, cell: f32, cx: f32, cz: f32, rc: f32, strength: f32) {
    let r = rc * 1.2 + cell;
    let to = |v: f32| (v + NEAR_HALF) / cell;
    let (i0, i1) = ((to(cx - r).floor().max(0.0)) as usize, (to(cx + r).ceil() as usize).min(res - 1));
    let (j0, j1) = ((to(cz - r).floor().max(0.0)) as usize, (to(cz + r).ceil() as usize).min(res - 1));
    if to(cx + r) < 0.0 || to(cz + r) < 0.0 {
        return;
    }
    for j in j0..=j1 {
        for i in i0..=i1 {
            let x = (i as f32 + 0.5) * cell - NEAR_HALF;
            let z = (j as f32 + 0.5) * cell - NEAR_HALF;
            let d = ((x - cx).powi(2) + (z - cz).powi(2)).sqrt();
            let f = 1.0 - smoothstep(0.5 * rc, 1.2 * rc, d);
            let k = j * res + i;
            grid[k] = 1.0 - (1.0 - grid[k]) * (1.0 - f * strength);
        }
    }
}

fn blur(src: &[f32], res: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; res * res];
    par_rows(res, res, &mut out, |j, row| {
        for (i, o) in row.iter_mut().enumerate() {
            let mut s = 0.0;
            for dj in -1i32..=1 {
                for di in -1i32..=1 {
                    let ii = (i as i32 + di).clamp(0, res as i32 - 1) as usize;
                    let jj = (j as i32 + dj).clamp(0, res as i32 - 1) as usize;
                    s += src[jj * res + ii];
                }
            }
            *o = s / 9.0;
        }
    });
    out
}
