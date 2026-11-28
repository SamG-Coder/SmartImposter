//! The globe's weather: one equirectangular map of the whole Earth (RGBA8: low cloud cover, low
//! cloud type, mid cover, high cover), built once on the CPU and read by clouds.glsl with a single
//! texture lookup (weatherAt). Built here, not per ray in the shader, it can afford the structure
//! real weather has, the shapes one sees from orbit:
//!
//! - the climate of cloud cover (ISCCP / MODIS): the equatorial belt of convection (ITCZ), clear
//!   subtropical highs and deserts, marine stratocumulus decks off the subtropical west coasts,
//!   trade-wind cumulus over the subtropical oceans, the storm tracks, polar stratus;
//! - extratropical cyclones along the storm tracks (Norwegian / Shapiro-Keyser models): a comma of
//!   cloud round each low, with the cold front as a long narrowing band that trails equatorward
//!   and wraps into the centre, the dry slot behind it, the cloud head on the poleward side, fields
//!   of cumulus in the cold air behind and a shield of cirrus ahead; anticlockwise in the north,
//!   clockwise in the south;
//! - the background cloud wound round those lows (the noise is sampled at points rotated about
//!   each low, more near its centre): streaks and swirls instead of blotches;
//! - mesoscale convective clusters along the ITCZ with their cirrus anvils, and a few tropical
//!   cyclones (central dense overcast, spiral rain bands, an eye).
//!
//! The shader adds what is smaller than a texel (~10 km): cells of ~12 km, the clouds themselves.

use super::etopo::Etopo;
use glam::DVec3;
use std::f64::consts::{PI, TAU};

pub const W: usize = 4096;
pub const H: usize = 2048;
const CACHE: &str = "assets/cache/weather_map.bin";
const MAGIC: &[u8; 8] = b"SIWX\0\0\0\x08";
const R_KM: f64 = 6371.0;

fn hash(x: i32, y: i32, z: i32, seed: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x8da6b343) ^ (y as u32).wrapping_mul(0xd8163841) ^ (z as u32).wrapping_mul(0xcb1ab31f) ^ seed.wrapping_mul(0x9e3779b9);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846ca68b);
    ((h ^ (h >> 16)) >> 8) as f32 / (1u32 << 24) as f32
}

/// Value noise, 0..1.
fn vnoise(p: DVec3, seed: u32) -> f32 {
    let f = p.floor();
    let (x, y, z) = (f.x as i32, f.y as i32, f.z as i32);
    let t = p - f;
    let u = t * t * (DVec3::splat(3.0) - 2.0 * t);
    let (ux, uy, uz) = (u.x as f32, u.y as f32, u.z as f32);
    let l = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let h = |dx: i32, dy: i32, dz: i32| hash(x + dx, y + dy, z + dz, seed);
    l(l(l(h(0, 0, 0), h(1, 0, 0), ux), l(h(0, 1, 0), h(1, 1, 0), ux), uy), l(l(h(0, 0, 1), h(1, 0, 1), ux), l(h(0, 1, 1), h(1, 1, 1), ux), uy), uz)
}

/// Fractal value noise, 0..1 (mean 0.5).
fn fbm(p: DVec3, octaves: u32, seed: u32) -> f32 {
    let (mut s, mut a, mut n, mut q) = (0.0, 1.0, 0.0, p);
    for o in 0..octaves {
        s += a * vnoise(q, seed + o * 7);
        n += a;
        a *= 0.5;
        q = q * 2.03 + DVec3::splat(11.7);
    }
    s / n
}

fn smooth(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Land fraction on a half-degree grid, blurred.
struct Grid {
    v: Vec<f32>,
}
const GW: usize = 720;
const GH: usize = 360;

impl Grid {
    fn sample(&self, lat: f64, lon: f64) -> f32 {
        let fx = (lon + 180.0).rem_euclid(360.0) * 2.0 - 0.5;
        let fy = ((90.0 - lat) * 2.0 - 0.5).clamp(0.0, GH as f64 - 1.0);
        let (x0, y0) = (fx.floor(), fy.floor());
        let (tx, ty) = ((fx - x0) as f32, (fy - y0) as f32);
        let at = |x: i64, y: i64| self.v[(y.clamp(0, GH as i64 - 1) as usize) * GW + x.rem_euclid(GW as i64) as usize];
        let (x0, y0) = (x0 as i64, y0 as i64);
        mix(mix(at(x0, y0), at(x0 + 1, y0), tx), mix(at(x0, y0 + 1), at(x0 + 1, y0 + 1), tx), ty)
    }

    /// Box blur of `r` cells (in latitude; widened in longitude towards the poles), twice.
    fn blurred(&self, r: usize) -> Grid {
        let mut v = self.v.clone();
        for _ in 0..2 {
            let mut t = vec![0f32; GW * GH];
            for y in 0..GH {
                let lat = 90.0 - (y as f64 + 0.5) * 0.5;
                let rx = ((r as f64 / lat.to_radians().cos().max(0.1)) as usize).min(GW / 4);
                let row = &v[y * GW..(y + 1) * GW];
                let mut sum: f32 = (0..=2 * rx).map(|k| row[(k + GW - rx) % GW]).sum();
                for x in 0..GW {
                    t[y * GW + x] = sum / (2 * rx + 1) as f32;
                    sum += row[(x + rx + 1) % GW] - row[(x + GW - rx) % GW];
                }
            }
            for x in 0..GW {
                for y in 0..GH {
                    let (y0, y1) = (y.saturating_sub(r), (y + r).min(GH - 1));
                    v[y * GW + x] = (y0..=y1).map(|k| t[k * GW + x]).sum::<f32>() / (y1 - y0 + 1) as f32;
                }
            }
        }
        Grid { v }
    }
}

/// An extratropical cyclone: its centre, the local east and north there, its radius (km), its
/// strength (0..1) and its hemisphere.
struct Low {
    c: DVec3,
    east: DVec3,
    north: DVec3,
    r: f64,
    s: f32,
    south: bool,
    seed: u32,
    /// how tightly the frontal band winds (a young wave ~1.4: an open arc; an occluded low ~2.5:
    /// wrapped round its centre), and where the band starts (radians from east)
    wind: f64,
    start: f64,
    /// the size of the cloud head (1: that of a mature low)
    head: f32,
}

fn at(lat: f64, lon: f64) -> DVec3 {
    let (la, lo) = (lat.to_radians(), lon.to_radians());
    DVec3::new(la.cos() * lo.cos(), la.cos() * lo.sin(), la.sin())
}

fn frame(c: DVec3) -> (DVec3, DVec3) {
    let east = DVec3::Z.cross(c).normalize_or(DVec3::X);
    (east, c.cross(east))
}

/// The lows of the two storm tracks: one every ~40 degrees of longitude, each at its own stage.
fn lows() -> Vec<Low> {
    let mut v = Vec::new();
    for (south, n, lat0) in [(false, 9, 50.0), (true, 10, -53.0)] {
        for k in 0..n {
            let h = |i: i32| hash(k, i, south as i32, 77) as f64;
            let lon = -180.0 + 360.0 * (k as f64 + 0.5 + 0.6 * (h(0) - 0.5)) / n as f64;
            let lat: f64 = lat0 + 16.0 * (h(1) - 0.5);
            let c = at(lat, lon);
            let (east, north) = frame(c);
            // (each at its own stage of life: a young frontal wave, a mature comma, an old occlusion)
            let age = h(4);
            v.push(Low {
                c,
                east,
                north,
                r: 800.0 + 800.0 * h(2),
                s: 0.55 + 0.45 * h(3) as f32,
                south,
                seed: 100 + k as u32 * 3 + south as u32 * 50,
                wind: 1.4 + 1.1 * age,
                start: (35.0 + 60.0 * h(5)).to_radians(),
                head: (0.45 + 1.0 * age) as f32,
            });
        }
    }
    v
}

/// Tropical cyclones: (lat, lon, radius km), over the warm oceans, far from each other.
const CYCLONES: [(f64, f64, f64); 4] = [(17.5, 136.0, 380.0), (15.0, -118.0, 300.0), (24.0, -49.0, 340.0), (-16.5, 78.0, 330.0)];

fn rotate(v: DVec3, axis: DVec3, ang: f64) -> DVec3 {
    let (s, c) = ang.sin_cos();
    v * c + axis.cross(v) * s + axis * axis.dot(v) * (1.0 - c)
}

struct World {
    near: Grid,
    far: Grid,
    lows: Vec<Low>,
}

/// The weather at the unit direction `u` (Earth axes): low cover, low type (0 stratus, 0.5
/// cumulus, 1 cumulonimbus), mid cover, high cover.
fn weather(u: DVec3, w: &World) -> [f32; 4] {
    let lat = u.z.clamp(-1.0, 1.0).asin().to_degrees();
    let lon = u.y.atan2(u.x).to_degrees();
    let a = lat.abs() as f32;
    let p = u * R_KM;
    let land = w.near.sample(lat, lon);
    let inland = smooth(0.5, 0.9, w.far.sample(lat, lon));
    // climate bands
    let itcz = (-((lat as f32 - 6.0) / 7.0).powi(2)).exp();
    let subtrop = smooth(10.0, 18.0, a) * (1.0 - smooth(30.0, 40.0, a));
    let storm = smooth(35.0, 45.0, a) * (1.0 - smooth(65.0, 75.0, a));
    let polar = smooth(62.0, 72.0, a);
    // marine stratocumulus: subtropical ocean with land 2-12 degrees to the east (west coasts)
    let mut deck = 0.0;
    if land < 0.9 && a > 8.0 && a < 38.0 {
        let step = 2.0 / lat.to_radians().cos().max(0.2);
        let mut l = 0f32;
        for i in 1..=6 {
            l = l.max(w.near.sample(lat, lon + step * i as f64) * (1.12 - 0.15 * i as f32));
        }
        deck = (1.0 - land) * smooth(0.0, 1.0, l) * (0.75 + 0.5 * vnoise(p / 300.0 + DVec3::splat(4.0), 5)) * smooth(8.0, 16.0, a) * (1.0 - smooth(30.0, 38.0, a));
    }
    // the subtropics are clear where the air sinks dry (deserts, the cool eastern oceans); the
    // trade-wind oceans and the humid coasts have fair-weather cumulus most days
    let dry = inland.max(deck).min(1.0);
    let trade = subtrop * (1.0 - dry);

    // extratropical cyclones
    let (mut front, mut head, mut slot, mut post, mut ci) = (0f32, 0f32, 0f32, 0f32, 0f32);
    let mut uw = u; // the background is sampled here: wound round the lows
    for l in &w.lows {
        let dist = u.dot(l.c).clamp(-1.0, 1.0).acos() * R_KM;
        if dist > 2.6 * l.r {
            continue;
        }
        let d = u - l.c;
        let (mut x, mut y) = (d.dot(l.east) * R_KM, d.dot(l.north) * R_KM);
        if l.south {
            y = -y; // (mirrored: the same comma turns clockwise in the south)
        }
        // ragged, not drawn with compasses: bent over ~650 km and frayed over ~180 km
        x += 340.0 * (fbm(p / 650.0 + DVec3::splat(3.0), 3, l.seed) as f64 - 0.5) + 110.0 * (fbm(p / 180.0 + DVec3::splat(5.0), 3, l.seed + 2) as f64 - 0.5);
        y += 340.0 * (fbm(p / 650.0 + DVec3::splat(9.0), 3, l.seed + 1) as f64 - 0.5) + 110.0 * (fbm(p / 180.0 + DVec3::splat(7.0), 3, l.seed + 3) as f64 - 0.5);
        let r = (x * x + y * y).sqrt().max(1.0);
        let rr = (r / l.r) as f32;
        // the frontal band: north-east of the centre at 0.3 R, trailing round the east and south to
        // the south-west at 2 R (a log spiral); s: the angle past the band (+ ahead, - behind)
        let s = y.atan2(x) - (l.start - l.wind * (r / (0.3 * l.r)).ln());
        let s = ((s + PI).rem_euclid(TAU) - PI) as f32;
        // (the band: 400-900 km of cloud near the low, where the warm air climbs over the cold,
        // narrowing to a line of ~250 km at its tail; wider ahead of the front than behind it)
        let width = mix(480.0, 150.0, (rr / 2.0).min(1.0)) / r.max(150.0) as f32 * if s > 0.0 { 1.35 } else { 0.75 }; // (radians)
        let env = smooth(0.1, 0.3, rr) * (1.0 - smooth(1.7, 2.4, rr));
        // its cloud is not even: thick and thin along it, breaking into pieces towards the tail
        let tex = fbm(p / 260.0 + DVec3::splat(13.0), 4, l.seed + 5);
        let broken = mix(1.0, smooth(0.38, 0.6, fbm(p / 420.0 + DVec3::splat(17.0), 3, l.seed + 6)), smooth(0.9, 1.9, rr));
        front = front.max(l.s * (-(s / width).powi(2)).exp() * env * (0.55 + 0.9 * tex) * broken);
        // the cloud head on the poleward, western side: larger the older the low
        let hs = l.head as f64;
        let (hx, hy) = (((x + 0.25 * l.r) / (0.5 * l.r * hs)) as f32, ((y - 0.3 * l.r) / (0.42 * l.r * hs)) as f32);
        head = head.max(l.s * (-(hx * hx + hy * hy)).exp() * (0.6 + 0.8 * tex));
        // the dry slot behind the front, the cumulus fields further back in the cold air, the
        // cirrus shield ahead
        slot = slot.max(l.s * (-((s + 0.9) / 0.45).powi(2)).exp() * smooth(0.2, 0.45, rr) * (1.0 - smooth(1.0, 1.5, rr)));
        post = post.max(l.s * smooth(-2.9, -2.3, s) * (1.0 - smooth(-1.6, -1.2, s)) * smooth(0.5, 0.8, rr) * (1.0 - smooth(1.6, 2.2, rr)));
        ci = ci.max(l.s * (-((s - 0.7) / 0.55).powi(2)).exp() * env);
        // wind the background round the low: more the nearer its centre
        let turn = 1.5 * l.s as f64 * (-(dist / (0.8 * l.r)).powf(1.3)).exp();
        uw = rotate(uw, l.c, if l.south { turn } else { -turn });
    }
    // weather systems between them: stretched along the flow (east-west), wound round the lows
    let q = DVec3::new(uw.x, uw.y, uw.z * 1.7) * (R_KM / 900.0);
    let sys = ((fbm(q, 7, 17) - 0.5) * 2.4 + 0.5).clamp(0.0, 1.0);
    // convective clusters along the ITCZ (100-400 km) and their anvils
    let clus = fbm(p / 420.0 + DVec3::splat(31.0), 4, 23);
    let (cl, anvil) = (smooth(0.52, 0.7, clus), smooth(0.46, 0.68, clus));
    // jet-stream cirrus: long streaks drawn out along the flow of the storm tracks
    let jet = smooth(0.56, 0.72, fbm(DVec3::new(uw.x, uw.y, uw.z * 4.5) * (R_KM / 800.0) + DVec3::splat(41.0), 4, 29));
    // the cold air behind the fronts and the storm tracks at large: fields of broken cloud
    let field = fbm(DVec3::new(uw.x, uw.y, uw.z * 1.4) * (R_KM / 330.0) + DVec3::splat(23.0), 5, 37);
    // and the fine grain of every cloud field: thick and thin over 10-40 km
    let grain = fbm(uw * (R_KM / 45.0) + DVec3::splat(57.0), 3, 59);
    // tropical cyclones
    let (mut tc, mut cdo) = (0f32, 0f32);
    for (la, lo, rad) in CYCLONES {
        let c = at(la, lo);
        let dist = u.dot(c).clamp(-1.0, 1.0).acos() * R_KM;
        if dist > 2.2 * rad {
            continue;
        }
        let (east, north) = frame(c);
        let d = u - c;
        let (x, mut y) = (d.dot(east) * R_KM, d.dot(north) * R_KM);
        if la < 0.0 {
            y = -y;
        }
        let r = (x * x + y * y).sqrt().max(1.0);
        let th = y.atan2(x) + 0.5 * (fbm(p / 160.0, 3, 61) as f64 - 0.5);
        let k = (r / rad) as f32;
        let arms = (0.5 + 0.5 * (2.0 * (th + 2.3 * (r / 40.0).ln())).cos() as f32).powf(1.6) * (-(k / 0.8).powi(2)).exp() * smooth(0.25, 0.45, k);
        let core = (-(k / 0.34).powi(2)).exp();
        let eye = smooth(0.035, 0.08, k);
        tc = tc.max(eye * core.max(arms));
        cdo = cdo.max(eye * core);
    }

    let mut c = 0.14 + 0.2 * polar + 0.5 * deck - 0.4 * subtrop * dry + 0.02 * trade + 0.1 * storm;
    // (the trades: the same scattered cumulus day after day, rarely overcast)
    c += (sys - 0.5) * (0.55 + 0.3 * storm) * (1.0 - 0.9 * trade);
    c += storm * 0.5 * (smooth(0.45, 0.7, field) - 0.35) * (0.5 + sys);
    // (the cold air behind a front: broken fields, patchy over 20-80 km)
    c += 0.85 * front + 0.65 * head + post * (0.1 + 0.45 * smooth(0.4, 0.65, field) + 0.25 * (grain - 0.5)) - 0.6 * slot;
    // (grain: where there is cloud at all, never in the clear or the scattered trades)
    c += (grain - 0.5) * 0.45 * smooth(0.22, 0.45, c);
    c += itcz * (0.08 + 0.55 * cl);
    let c = c.max(tc).clamp(0.0, 1.0);
    let ty = (0.5 + 0.45 * itcz * cl - 0.5 * deck - 0.35 * polar - 0.25 * front - 0.2 * head).max(cdo).clamp(0.0, 1.0);
    let mid = (0.8 * front + 0.7 * head + 0.35 * itcz * cl + 0.8 * tc).clamp(0.0, 1.0);
    let high = ((0.06 + 0.65 * ci + 0.3 * front + 0.5 * itcz * anvil + 0.2 * storm * smooth(0.55, 0.8, sys) + 0.5 * jet * storm.max(0.25 * polar)) * (1.0 - 0.7 * subtrop)).max((tc * 1.3).min(1.0) * smooth(0.0, 0.3, cdo + 0.1)).clamp(0.0, 1.0);
    [c, ty, mid, high]
}

fn world(etopo: &Etopo) -> World {
    let mut v = vec![0f32; GW * GH];
    for y in 0..GH {
        for x in 0..GW {
            let (lat, lon) = (90.0 - (y as f64 + 0.5) * 0.5, -180.0 + (x as f64 + 0.5) * 0.5);
            v[y * GW + x] = if etopo.height(lat, lon) > 0.0 { 1.0 } else { 0.0 };
        }
    }
    let land = Grid { v };
    // (~60 km: coastlines change the weather gradually; ~500 km: continental interiors)
    World { near: land.blurred(1), far: land.blurred(6), lows: lows() }
}

/// The weather map, W x H RGBA8, row 0 at 90 N, column 0 at 180 W (as the elevation grid).
pub fn load_or_build(etopo: &Etopo) -> Vec<u8> {
    let n = W * H * 4;
    if let Ok(b) = std::fs::read(CACHE) {
        if b.len() == 8 + n && &b[..8] == MAGIC {
            return b[8..].to_vec();
        }
    }
    let t = std::time::Instant::now();
    let wd = world(etopo);
    let mut out = vec![0u8; n];
    let threads = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(4);
    let rows = H.div_ceil(threads);
    std::thread::scope(|s| {
        for (k, chunk) in out.chunks_mut(rows * W * 4).enumerate() {
            let wd = &wd;
            s.spawn(move || {
                for (i, px) in chunk.chunks_mut(4).enumerate() {
                    let (x, y) = (i % W, k * rows + i / W);
                    let lat = 90.0 - (y as f64 + 0.5) * 180.0 / H as f64;
                    let lon = -180.0 + (x as f64 + 0.5) * 360.0 / W as f64;
                    let v = weather(at(lat, lon), wd);
                    for c in 0..4 {
                        px[c] = (v[c].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                    }
                }
            });
        }
    });
    if let Some(dir) = std::path::Path::new(CACHE).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let mut file = Vec::with_capacity(8 + n);
    file.extend_from_slice(MAGIC);
    file.extend_from_slice(&out);
    let _ = std::fs::write(CACHE, file);
    println!("Weather map {W}x{H} built in {:.1}s", t.elapsed().as_secs_f32());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fronts_trail_equatorward_in_both_hemispheres() {
        // the band at 1.5 R lies equatorward of the low's centre, in the north and in the south
        // (each low on its own: the neighbours' fronts reach across)
        for one in lows() {
            let w = World { near: Grid { v: vec![0.0; GW * GH] }, far: Grid { v: vec![0.0; GW * GH] }, lows: vec![one] };
            let l = &w.lows[0];
            // (cover-weighted over the ring: the band is ragged and broken, its side is not)
            let mut north = 0.0;
            for k in 0..360 {
                let th = (k as f64).to_radians();
                let u = (l.c + (l.east * th.cos() + l.north * th.sin()) * (1.3 * l.r / R_KM)).normalize();
                north += weather(u, &w)[2] as f64 * th.sin(); // mid cover: the front and the head
            }
            assert!(if l.south { north > 0.1 } else { north < -0.1 }, "south {} north-weighted cover {north:.2}", l.south);
        }
    }

    /// Rebuild the cached map (cargo test --release weather_map_dump -- --ignored).
    #[test]
    #[ignore]
    fn weather_map_dump() {
        let _ = std::fs::remove_file(CACHE);
        let e = Etopo::load(std::path::Path::new("assets/earth/etopo")).unwrap();
        assert_eq!(load_or_build(&e).len(), W * H * 4);
    }

    #[test]
    fn rotation_keeps_the_axis() {
        let a = at(40.0, 10.0);
        assert!((rotate(a, a, 1.0) - a).length() < 1e-12);
        assert!((rotate(at(41.0, 10.0), a, 0.3).dot(a) - at(41.0, 10.0).dot(a)).abs() < 1e-12);
    }
}
