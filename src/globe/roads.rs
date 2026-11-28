//! Roads from OpenStreetMap, live: the lines of the main roads round the camera (motorways down
//! to tertiary roads, no streets, lanes or tracks: the network one sees from the air, a few
//! thousand ways for a city and its surroundings) are fetched from the Overpass API as one flies
//! (a worker thread, one query per zoom-10 map tile, ~40 km), kept on disk (assets/earth/osm), and
//! laid out for the shaders as a grid of line segments round an anchor point (roads.glsl). The
//! ground shader finds the road a pixel lies on from the exact vectors, any distance away, and
//! paints RealisticDrive's road surface there: no mesh, a cell lookup and a few segments a pixel.
//!
//! Two grids over the same lines: 16 m cells within ~2 km of the anchor (at the height of the
//! ground, to the centimetre), 128 m cells out to ~16 km. The anchor moves with the camera: a new
//! layout is built when it has gone ~500 m, or when a tile has arrived.
//!
//! Map data (c) OpenStreetMap contributors, Open Database Licence (openstreetmap.org/copyright).

use super::dem::Terrain;
use super::tiles::R_EARTH;
use glam::{DVec2, DVec3};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Cells per side of each grid, and their sizes (m): the fine grid reaches +-2048 m round the
/// anchor, the coarse one +-16.4 km. (roads.glsl has the same numbers.)
pub const GRID: usize = 256;
pub const FINE: f32 = 16.0;
pub const COARSE: f32 = 128.0;
/// Words of one layout on the GPU: the two grids' cells, the segment indices they point into, the
/// segments (6 words each: a, b in metres from the anchor, the length along the way at a, info).
pub const IDX_CAP: usize = 1_000_000;
pub const SEG_CAP: usize = 200_000;
pub const SEG_WORDS: usize = 6;
pub const IDX_AT: usize = 2 * GRID * GRID;
pub const SEG_AT: usize = IDX_AT + IDX_CAP;
pub const HALF_WORDS: usize = SEG_AT + SEG_CAP * SEG_WORDS;

/// Road classes (the shader's markings and widths go by them).
pub mod class {
    pub const MOTORWAY: u8 = 0;
    pub const TRUNK: u8 = 1;
    pub const PRIMARY: u8 = 2;
    pub const SECONDARY: u8 = 3;
    pub const TERTIARY: u8 = 4;
    pub const STREET: u8 = 5;
    pub const SERVICE: u8 = 6;
    pub const TRACK: u8 = 7;
    pub const LINK: u8 = 8;
}
pub const ONEWAY: u8 = 1;
pub const BRIDGE: u8 = 2;
pub const UNPAVED: u8 = 8;

/// The map tiles asked for: zoom 10, ~40 km across.
const TILE_Z: u32 = 10;
/// Public Overpass servers. The first is the main one: often at capacity for a moment (it says
/// so at once), but it answers. The mirrors are tried now and then (attempts 3 and 6 of every
/// six), and given up on quickly: they are often down altogether.
const API: [&str; 3] = ["https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter", "https://overpass.kumi.systems/api/interpreter"];
const ORDER: [usize; 6] = [0, 0, 1, 0, 0, 2];
const FILTER: &str = "motorway|trunk|primary|secondary|tertiary|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link";

#[derive(Clone)]
struct Way {
    id: u64,
    class: u8,
    lanes: u8,
    flags: u8,
    /// (lat, lon) in degrees
    pts: Vec<[f64; 2]>,
}

/// Half the paved width (m) of a road: as roads.glsl roadHalfWidth.
pub fn half_width(class: u8, lanes: u8, flags: u8) -> f32 {
    let oneway = flags & ONEWAY != 0;
    let n = if lanes > 0 {
        lanes as f32
    } else {
        match class {
            class::MOTORWAY => 3.0,
            class::TRUNK => 2.0,
            class::PRIMARY => if oneway { 2.0 } else { 4.0 },
            class::SECONDARY => if oneway { 2.0 } else { 2.0 },
            class::TERTIARY | class::STREET => if oneway { 1.0 } else { 2.0 },
            _ => 1.0,
        }
    };
    match class {
        class::MOTORWAY | class::TRUNK => n * 1.85 + 1.5,
        class::PRIMARY | class::SECONDARY => n * 1.7 + 0.4,
        class::TERTIARY => n * 1.6 + 0.3,
        class::STREET => (n * 1.5 + 0.4).max(2.8),
        class::SERVICE => 1.9,
        class::TRACK => 1.5,
        _ => n * 1.85 + 0.8,
    }
}

fn tile_of(lat: f64, lon: f64, z: u32) -> (i32, i32) {
    let n = (1u32 << z) as f64;
    let x = ((lon + 180.0) / 360.0 * n).floor();
    let y = ((1.0 - lat.to_radians().tan().asinh() / std::f64::consts::PI) / 2.0 * n).floor();
    (x.clamp(0.0, n - 1.0) as i32, y.clamp(0.0, n - 1.0) as i32)
}

/// (south, west, north, east) of a map tile, degrees.
fn tile_box(x: i32, y: i32, z: u32) -> (f64, f64, f64, f64) {
    let n = (1u32 << z) as f64;
    let lat = |y: f64| (std::f64::consts::PI * (1.0 - 2.0 * y / n)).sinh().atan().to_degrees();
    (lat(y as f64 + 1.0), x as f64 / n * 360.0 - 180.0, lat(y as f64), (x as f64 + 1.0) / n * 360.0 - 180.0)
}

fn cache_path(x: i32, y: i32, z: u32) -> PathBuf {
    PathBuf::from(format!("assets/earth/osm/{z}/{x}_{y}.roads"))
}

const MAGIC: &[u8; 8] = b"SIROAD\0\x02";

fn save(path: &PathBuf, ways: &[Way]) {
    let mut b = Vec::new();
    b.extend_from_slice(MAGIC);
    b.extend_from_slice(&(ways.len() as u32).to_le_bytes());
    for w in ways {
        b.extend_from_slice(&w.id.to_le_bytes());
        b.extend_from_slice(&[w.class, w.lanes, w.flags, 0]);
        b.extend_from_slice(&(w.pts.len() as u32).to_le_bytes());
        for p in &w.pts {
            b.extend_from_slice(&((p[0] * 1e7).round() as i32).to_le_bytes());
            b.extend_from_slice(&((p[1] * 1e7).round() as i32).to_le_bytes());
        }
    }
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(path, b);
}

fn load(path: &PathBuf) -> Option<Vec<Way>> {
    let b = std::fs::read(path).ok()?;
    if b.len() < 12 || &b[..8] != MAGIC {
        return None;
    }
    let u32_at = |i: usize| b.get(i..i + 4).map(|s| u32::from_le_bytes(s.try_into().unwrap()));
    let n = u32_at(8)? as usize;
    let mut i = 12;
    let mut ways = Vec::with_capacity(n);
    for _ in 0..n {
        let id = u64::from_le_bytes(b.get(i..i + 8)?.try_into().ok()?);
        let (class, lanes, flags) = (*b.get(i + 8)?, *b.get(i + 9)?, *b.get(i + 10)?);
        let np = u32_at(i + 12)? as usize;
        i += 16;
        let mut pts = Vec::with_capacity(np);
        for _ in 0..np {
            let la = i32::from_le_bytes(b.get(i..i + 4)?.try_into().ok()?);
            let lo = i32::from_le_bytes(b.get(i + 4..i + 8)?.try_into().ok()?);
            pts.push([la as f64 * 1e-7, lo as f64 * 1e-7]);
            i += 8;
        }
        ways.push(Way { id, class, lanes, flags, pts });
    }
    Some(ways)
}

/// One Overpass query (through curl, which every current Windows, macOS and Linux has: no TLS
/// stack of our own): the ways of the box as JSON.
fn fetch(x: i32, y: i32, z: u32, server: usize) -> Option<Vec<Way>> {
    let (s, w, n, e) = tile_box(x, y, z);
    let query = format!("data=[out:json][timeout:120];way[\"highway\"~\"^({FILTER})$\"]({s:.6},{w:.6},{n:.6},{e:.6});out geom qt;");
    let out = std::process::Command::new("curl")
        .args(["-s", "--connect-timeout", "8", "-m", if ORDER[server % ORDER.len()] == 0 { "120" } else { "40" }, "-A", "SmartImposter-globe/0.1", "--data-urlencode", &query, API[ORDER[server % ORDER.len()]]])
        .stdin(std::process::Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse(&out.stdout)
}

fn parse(json: &[u8]) -> Option<Vec<Way>> {
    let v: serde_json::Value = serde_json::from_slice(json).ok()?;
    let mut ways = Vec::new();
    for e in v.get("elements")?.as_array()? {
        let tags = &e["tags"];
        let tag = |k: &str| tags[k].as_str().unwrap_or("");
        let hw = tag("highway");
        let class = match hw {
            "motorway" => class::MOTORWAY,
            "trunk" => class::TRUNK,
            "primary" => class::PRIMARY,
            "secondary" => class::SECONDARY,
            "tertiary" => class::TERTIARY,
            "unclassified" | "residential" | "living_street" => class::STREET,
            "service" => class::SERVICE,
            "track" => class::TRACK,
            h if h.ends_with("_link") => class::LINK,
            _ => continue,
        };
        // (tunnels are under the ground: not drawn; private driveways neither)
        if !matches!(tag("tunnel"), "" | "no") || tag("service") == "driveway" || tag("area") == "yes" {
            continue;
        }
        let mut flags = 0;
        if matches!(tag("oneway"), "yes" | "1" | "true") || class == class::MOTORWAY || tag("junction") == "roundabout" {
            flags |= ONEWAY;
        }
        if !matches!(tag("bridge"), "" | "no") {
            flags |= BRIDGE;
        }
        if class == class::TRACK || matches!(tag("surface"), "unpaved" | "gravel" | "dirt" | "sand" | "ground" | "earth" | "grass" | "compacted" | "fine_gravel" | "shells") {
            flags |= UNPAVED;
        }
        let lanes = tag("lanes").parse::<u8>().unwrap_or(0).min(12);
        let pts: Vec<[f64; 2]> = e["geometry"].as_array().map(|g| g.iter().filter_map(|p| Some([p["lat"].as_f64()?, p["lon"].as_f64()?])).collect()).unwrap_or_default();
        if pts.len() >= 2 {
            ways.push(Way { id: e["id"].as_u64().unwrap_or(0), class, lanes, flags, pts });
        }
    }
    Some(ways)
}

/// A layout of the roads round an anchor, as the shaders read it.
pub struct Build {
    /// the anchor (ECEF, on the ground) and the unit east and north there: a point P maps to
    /// ((P - anchor) . east, (P - anchor) . north)
    pub anchor: DVec3,
    pub east: DVec3,
    pub north: DVec3,
    /// 2 x GRID x GRID cells (fine, then coarse): offset << 8 | count into `index`
    pub cells: Vec<u32>,
    pub index: Vec<u32>,
    /// SEG_WORDS words per segment
    pub segs: Vec<u32>,
    pub ways: usize,
}

impl Build {
    fn seg(&self, i: u32) -> (DVec2, DVec2, f32) {
        let k = i as usize * SEG_WORDS;
        let f = |j: usize| f32::from_bits(self.segs[k + j]) as f64;
        let info = self.segs[k + 5];
        (DVec2::new(f(0), f(1)), DVec2::new(f(2), f(3)), half_width((info & 15) as u8, ((info >> 4) & 15) as u8, ((info >> 8) & 15) as u8))
    }

    /// Is the point `q` (metres east and north of the anchor) on a road, or within `margin` of
    /// its edge? (The fine grid: within ~2 km of the anchor.)
    pub fn on_road(&self, q: DVec2, margin: f64) -> bool {
        let half = GRID as f64 * 0.5 * FINE as f64;
        if q.x.abs() >= half || q.y.abs() >= half {
            return false;
        }
        let (cx, cy) = (((q.x + half) / FINE as f64) as usize, ((q.y + half) / FINE as f64) as usize);
        let c = self.cells[cy.min(GRID - 1) * GRID + cx.min(GRID - 1)];
        let (off, n) = ((c >> 8) as usize, (c & 255) as usize);
        (0..n).any(|k| {
            let (a, b, hw) = self.seg(self.index[off + k]);
            let ab = b - a;
            let t = ((q - a).dot(ab) / ab.length_squared().max(1e-9)).clamp(0.0, 1.0);
            (q - (a + ab * t)).length() < hw as f64 + margin
        })
    }
}

fn hash16(id: u64) -> u32 {
    let mut h = id.wrapping_mul(0x9e3779b97f4a7c15);
    h ^= h >> 29;
    (h as u32) & 0xFFFF
}

/// Lay the ways out round the ground point under `cam` (ECEF).
fn build(cam: DVec3, terrain: &Terrain, sets: &[Arc<Vec<Way>>]) -> Build {
    let dir = cam.normalize();
    let (lat0, lon0) = super::tiles::lat_lon(dir);
    let anchor = dir * (R_EARTH + terrain.height_at(lat0, lon0, 30.0).max(0.0) as f64);
    let east = DVec3::Z.cross(dir).normalize_or(DVec3::X);
    let north = dir.cross(east);
    let xy = |p: &[f64; 2], fine: bool| {
        let (la, lo) = (p[0].to_radians(), p[1].to_radians());
        let d = DVec3::new(la.cos() * lo.cos(), la.cos() * lo.sin(), la.sin());
        // (at the height of the ground there: the shader maps the point it has hit the same way)
        let h = if fine { terrain.height_at(p[0], p[1], 30.0).max(0.0) as f64 } else { 0.0 };
        let v = d * (R_EARTH + h) - anchor;
        DVec2::new(v.dot(east), v.dot(north))
    };
    let mut segs: Vec<u32> = Vec::new();
    let mut lists: Vec<Vec<u32>> = vec![Vec::new(); 2 * GRID * GRID];
    let mut ways = 0;
    for (level, cell) in [FINE as f64, COARSE as f64].into_iter().enumerate() {
        let half = GRID as f64 * 0.5 * cell;
        let mut seen = HashSet::new();
        for w in sets.iter().flat_map(|s| s.iter()) {
            if !seen.insert(w.id) {
                continue;
            }
            let hw = half_width(w.class, w.lanes, w.flags) as f64;
            let reach = hw + 3.6 + cell * 0.71;
            let info = w.class as u32 | (w.lanes.min(15) as u32) << 4 | (w.flags as u32 & 15) << 8 | hash16(w.id) << 16;
            let mut a = xy(&w.pts[0], level == 0);
            let mut s0 = 0.0f64;
            let mut used = false;
            for p in &w.pts[1..] {
                let b = xy(p, level == 0);
                let len = (b - a).length();
                let (lo, hi) = (a.min(b) - DVec2::splat(reach), a.max(b) + DVec2::splat(reach));
                if len > 0.01 && hi.x > -half && hi.y > -half && lo.x < half && lo.y < half && segs.len() / SEG_WORDS < SEG_CAP {
                    let si = (segs.len() / SEG_WORDS) as u32;
                    segs.extend_from_slice(&[(a.x as f32).to_bits(), (a.y as f32).to_bits(), (b.x as f32).to_bits(), (b.y as f32).to_bits(), (s0 as f32).to_bits(), info]);
                    used = true;
                    let c0 = (((lo.x + half) / cell).floor().max(0.0) as usize, ((lo.y + half) / cell).floor().max(0.0) as usize);
                    let c1 = ((((hi.x + half) / cell).floor() as isize).min(GRID as isize - 1), (((hi.y + half) / cell).floor() as isize).min(GRID as isize - 1));
                    for cy in c0.1 as isize..=c1.1 {
                        for cx in c0.0 as isize..=c1.0 {
                            let c = DVec2::new((cx as f64 + 0.5) * cell - half, (cy as f64 + 0.5) * cell - half);
                            let ab = b - a;
                            let t = ((c - a).dot(ab) / (len * len)).clamp(0.0, 1.0);
                            if (c - (a + ab * t)).length() < reach {
                                lists[level * GRID * GRID + cy as usize * GRID + cx as usize].push(si);
                            }
                        }
                    }
                }
                s0 += len;
                a = b;
            }
            ways += used as usize;
        }
    }
    let mut cells = vec![0u32; 2 * GRID * GRID];
    let mut index = Vec::new();
    for (c, l) in cells.iter_mut().zip(&lists) {
        if index.len() + l.len() > IDX_CAP {
            break;
        }
        *c = (index.len() as u32) << 8 | l.len().min(255) as u32;
        index.extend_from_slice(&l[..l.len().min(255)]);
    }
    Build { anchor, east, north, cells, index, segs, ways }
}

/// The worker: camera positions in, layouts out.
pub struct Roads {
    req: Sender<DVec3>,
    done: Receiver<Arc<Build>>,
    last_req: Option<DVec3>,
    /// the worker has tiles to read or a layout to make for the last position it was given
    working: Arc<std::sync::atomic::AtomicBool>,
}

impl Roads {
    pub fn spawn(terrain: Arc<Terrain>) -> Self {
        let (req, req_rx) = channel::<DVec3>();
        let (done_tx, done) = channel();
        let working = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let busy = working.clone();
        std::thread::Builder::new()
            .name("roads".into())
            .spawn(move || {
                let mut loaded: HashMap<(u32, i32, i32), Arc<Vec<Way>>> = HashMap::new();
                let mut cam: Option<DVec3> = None;
                let mut built_at: Option<DVec3> = None;
                let mut dirty = false;
                // (after a refusal: wait, twice as long each time, and ask the next server)
                let (mut server, mut backoff, mut wait_until) = (0usize, Duration::from_secs(4), Instant::now());
                loop {
                    // the latest position; wait for one when there is nothing to do
                    match if cam.is_none() { req_rx.recv().map_err(|_| RecvTimeoutError::Disconnected) } else { req_rx.recv_timeout(Duration::from_millis(if dirty { 0 } else { 300 })) } {
                        Ok(c) => cam = Some(c),
                        Err(RecvTimeoutError::Disconnected) => return,
                        Err(RecvTimeoutError::Timeout) => {}
                    }
                    while let Ok(c) = req_rx.try_recv() {
                        cam = Some(c);
                    }
                    let c = cam.unwrap();
                    let (lat, lon) = super::tiles::lat_lon(c.normalize());
                    // (from high up no road is wider than a pixel)
                    if c.length() - R_EARTH > 60_000.0 {
                        busy.store(false, std::sync::atomic::Ordering::Relaxed);
                        continue;
                    }
                    // the map tiles the two grids reach into, nearest first
                    let mut want: Vec<(f64, (u32, i32, i32))> = Vec::new();
                    {
                        let (z, reach) = (TILE_Z, GRID as f64 * 0.5 * COARSE as f64);
                        let (dlat, dlon) = (reach / 111_320.0, reach / (111_320.0 * lat.to_radians().cos().max(0.05)));
                        let (x0, y0) = tile_of(lat + dlat, lon - dlon, z);
                        let (x1, y1) = tile_of(lat - dlat, lon + dlon, z);
                        for y in y0..=y1 {
                            for x in x0..=x1 {
                                let (s, w, n, e) = tile_box(x, y, z);
                                let d = ((0.5 * (s + n) - lat).powi(2) + ((0.5 * (w + e) - lon) * lat.to_radians().cos()).powi(2)).sqrt();
                                want.push((d, (z, x, y)));
                            }
                        }
                    }
                    want.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                    // from the disk what is there
                    for (_, key) in &want {
                        if !loaded.contains_key(key) {
                            if let Some(w) = load(&cache_path(key.1, key.2, key.0)) {
                                loaded.insert(*key, Arc::new(w));
                                dirty = true;
                            }
                        }
                    }
                    // a layout of what there is: when a tile has come, or the camera has moved on
                    let moved = built_at.is_none_or(|b| (b - c).length() > 500.0);
                    if (dirty || moved) && want.iter().any(|(_, k)| loaded.contains_key(k)) {
                        let sets: Vec<_> = want.iter().filter_map(|(_, k)| loaded.get(k).cloned()).collect();
                        let b = build(c, &terrain, &sets);
                        built_at = Some(c);
                        dirty = false;
                        if done_tx.send(Arc::new(b)).is_err() {
                            return;
                        }
                    }
                    // then one missing tile from the network, the nearest (this takes its time)
                    let online = std::env::var("SI_ROADS").map(|v| v != "offline").unwrap_or(true);
                    // (the nearest missing tile, again and again: it is the one under the camera)
                    let next = want.iter().map(|(_, k)| *k).find(|k| !loaded.contains_key(k));
                    if let (true, Some(key)) = (online && Instant::now() >= wait_until, next) {
                        let t = Instant::now();
                        match fetch(key.1, key.2, key.0, server) {
                            Some(w) => {
                                println!("Roads: OpenStreetMap tile {}/{}/{}: {} ways ({:.0}s)", key.0, key.1, key.2, w.len(), t.elapsed().as_secs_f32());
                                save(&cache_path(key.1, key.2, key.0), &w);
                                loaded.insert(key, Arc::new(w));
                                dirty = true;
                                backoff = Duration::from_secs(4);
                            }
                            None => {
                                println!("Roads: OpenStreetMap tile {}/{}/{} not received (trying again in {} s)", key.0, key.1, key.2, backoff.as_secs());
                                server += 1;
                                wait_until = Instant::now() + backoff;
                                backoff = (backoff * 2).min(Duration::from_secs(30));
                            }
                        }
                    }
                    // (a --shot waits for the layout of what is on the disk, not for the network)
                    busy.store(dirty, std::sync::atomic::Ordering::Relaxed);
                    // (keep what is near: tiles far behind are dropped)
                    if loaded.len() > 96 {
                        let keep: HashSet<_> = want.iter().map(|(_, k)| *k).collect();
                        loaded.retain(|k, _| keep.contains(k));
                    }
                }
            })
            .expect("roads thread");
        Roads { req, done, last_req: None, working }
    }

    /// Tell the worker where the camera is (ECEF).
    pub fn at(&mut self, cam: DVec3) {
        if self.last_req.is_none_or(|p| (p - cam).length() > 50.0) {
            self.last_req = Some(cam);
            self.working.store(true, std::sync::atomic::Ordering::Relaxed);
            let _ = self.req.send(cam);
        }
    }

    /// Tiles are still being read or fetched for the last position.
    pub fn working(&self) -> bool {
        self.working.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The newest layout, if one has arrived.
    pub fn take(&self) -> Option<Arc<Build>> {
        let mut last = None;
        while let Ok(b) = self.done.try_recv() {
            last = Some(b);
        }
        last
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_tiles_round_trip() {
        // Miami: the tile's box holds the point it was asked for
        let (x, y) = tile_of(25.92, -80.13, TILE_Z);
        let (s, w, n, e) = tile_box(x, y, TILE_Z);
        assert!(s < 25.92 && 25.92 < n && w < -80.13 && -80.13 < e, "{s} {w} {n} {e}");
    }

    /// Fill the cache round a place (cargo test --release roads_prefetch -- --ignored; the place
    /// from SI_ROADS_AT=lat,lon, Miami's Oleta River otherwise).
    #[test]
    #[ignore]
    fn roads_prefetch() {
        let at = std::env::var("SI_ROADS_AT").ok().and_then(|s| {
            let v: Vec<f64> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            (v.len() == 2).then(|| (v[0], v[1]))
        });
        let (lat, lon) = at.unwrap_or((25.92, -80.14));
        let keys = vec![(TILE_Z, tile_of(lat, lon, TILE_Z))];
        for (z, (x, y)) in keys {
            let path = cache_path(x, y, z);
            if load(&path).is_some() {
                continue;
            }
            let mut wait = 4;
            for attempt in 0..8 {
                match fetch(x, y, z, attempt) {
                    Some(w) => {
                        println!("{z}/{x}/{y}: {} ways", w.len());
                        save(&path, &w);
                        break;
                    }
                    None => {
                        println!("{z}/{x}/{y}: not received, again in {wait} s");
                        std::thread::sleep(Duration::from_secs(wait));
                        wait = (wait * 2).min(60);
                    }
                }
            }
        }
    }

    #[test]
    fn overpass_json_and_cache() {
        let json = br#"{"elements":[
            {"type":"way","id":7,"tags":{"highway":"primary","lanes":"4","oneway":"yes"},"geometry":[{"lat":25.9,"lon":-80.1},{"lat":25.901,"lon":-80.1}]},
            {"type":"way","id":8,"tags":{"highway":"footway"},"geometry":[{"lat":25.9,"lon":-80.1},{"lat":25.901,"lon":-80.1}]},
            {"type":"way","id":9,"tags":{"highway":"track"},"geometry":[{"lat":25.9,"lon":-80.1},{"lat":25.9,"lon":-80.099}]}]}"#;
        let ways = parse(json).unwrap();
        assert_eq!(ways.len(), 2);
        assert_eq!((ways[0].class, ways[0].lanes, ways[0].flags), (class::PRIMARY, 4, ONEWAY));
        assert_eq!(ways[1].flags & UNPAVED, UNPAVED);
        let path = std::env::temp_dir().join("si_roads_test.roads");
        save(&path, &ways);
        let back = load(&path).unwrap();
        assert_eq!(back.len(), 2);
        assert!((back[0].pts[1][0] - 25.901).abs() < 1e-6 && back[1].class == class::TRACK);
    }
}
