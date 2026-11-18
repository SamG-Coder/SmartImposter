//! Terrain elevation for the globe: Copernicus DEM GLO-30 (1 arc-second, ~30 m) wherever its tiles
//! are in assets/earth/glo30, NOAA ETOPO 2022 (60") everywhere else (open ocean, where GLO-30
//! publishes no tiles, and the seafloor).
//!
//! GLO-30 tiles are 1 x 1 degree, 3600 rows by 3600 columns below 50 degrees latitude, fewer
//! columns towards the poles (2400 / 1800 / 1200 / 720 / 360), pixel-is-point: pixel (i, j) of
//! tile N46 E008 sits at lat 47 - j / 3600, lon 8 + i / w.
//!
//! The whole planet is ~26,000 tiles, so nothing is loaded up front: the published GeoTIFFs are
//! converted once to a compact `.dem` file (half-metre int16, gradient-predicted residuals,
//! deflate: ~4.5x smaller than the GeoTIFF; `--convert-dem`, tools/fetch_earth.py does it while downloading), and tiles are
//! decoded on first use by the workers and kept in an LRU of RESIDENT tiles.

use super::etopo::{decode_tiff_f32, Etopo};
use super::landcover::LandCover;
use anyhow::{bail, Context, Result};
use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Rows per tile (1 arc-second) at every latitude.
const ROWS: i64 = 3600;
const MAGIC: &[u8; 8] = b"SIDEM2\0\0";
/// Decoded tiles kept in memory (up to 26 MB each).
const RESIDENT: usize = 160;

/// Columns of the GLO-30 tile whose south edge is at `lat`.
pub fn cols_at(lat: i32) -> i64 {
    // the band is set by the tile's latitude nearest the equator
    let a = if lat >= 0 { lat } else { -lat - 1 };
    match a {
        0..=49 => 3600,
        50..=59 => 2400,
        60..=69 => 1800,
        70..=79 => 1200,
        80..=84 => 720,
        _ => 360,
    }
}

fn tile_name_coords(name: &str) -> Option<(i32, i32)> {
    // Copernicus_DSM_COG_10_N46_00_E008_00_DEM.tif
    let p: Vec<&str> = name.split('_').collect();
    let ns = p.iter().position(|s| (s.starts_with('N') || s.starts_with('S')) && s.len() == 3)?;
    let lat: i32 = p[ns][1..].parse().ok()?;
    let lat = if p[ns].starts_with('S') { -lat } else { lat };
    let ew = p.iter().position(|s| (s.starts_with('E') || s.starts_with('W')) && s.len() == 4)?;
    let lon: i32 = p[ew][1..].parse().ok()?;
    let lon = if p[ew].starts_with('W') { -lon } else { lon };
    Some((lat, lon))
}

/// Convert a published GLO-30 GeoTIFF to `.dem` next to it; returns the new path.
pub fn convert_tif(tif: &Path) -> Result<PathBuf> {
    let name = tif.file_name().unwrap_or_default().to_string_lossy().to_string();
    let (lat, lon) = tile_name_coords(&name).with_context(|| format!("not a GLO tile name: {name}"))?;
    let bytes = std::fs::read(tif).with_context(|| format!("reading {}", tif.display()))?;
    let (w, h, v) = decode_tiff_f32(&bytes)?;
    if h as i64 != ROWS || w as i64 != cols_at(lat) {
        bail!("{name}: {w} x {h}, expected {} x {ROWS}", cols_at(lat));
    }
    // half-metre steps, stored as the residual of the gradient predictor left + up - up-left
    // (small numbers deflate well)
    let q: Vec<i16> = v.iter().map(|x| (x * 2.0).round().clamp(-32767.0, 32767.0) as i16).collect();
    let mut deltas = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            deltas.push(q[y * w + x].wrapping_sub(predict(&q, w, x, y)));
        }
    }
    let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(6));
    z.write_all(bytemuck::cast_slice(&deltas))?;
    let body = z.finish()?;
    let mut out = Vec::with_capacity(24 + body.len());
    out.extend_from_slice(MAGIC);
    for v in [w as u32, h as u32, lat as u32, lon as u32] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&body);
    let dem = tif.with_extension("dem");
    let part = tif.with_extension("dem.part");
    std::fs::write(&part, out)?;
    std::fs::rename(&part, &dem)?;
    Ok(dem)
}

/// `--convert-dem a.tif b.tif ...`: convert in parallel and delete each GeoTIFF once its `.dem`
/// is written.
pub fn convert_cli(paths: &[String]) -> Result<()> {
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let next = AtomicU64::new(0);
    let failed = AtomicU64::new(0);
    std::thread::scope(|s| {
        for _ in 0..threads.min(paths.len().max(1)) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed) as usize;
                let Some(p) = paths.get(i) else { break };
                match convert_tif(Path::new(p)) {
                    Ok(_) => {
                        let _ = std::fs::remove_file(p);
                    }
                    Err(e) => {
                        eprintln!("convert {p}: {e:#}");
                        failed.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });
        }
    });
    match failed.into_inner() {
        0 => Ok(()),
        n => bail!("{n} of {} tiles failed", paths.len()),
    }
}

/// Gradient predictor from the already known neighbours (0 outside the tile).
#[inline]
fn predict(q: &[i16], w: usize, x: usize, y: usize) -> i16 {
    let at = |x: usize, y: usize| q[y * w + x] as i32;
    let l = if x > 0 { at(x - 1, y) } else { 0 };
    let u = if y > 0 { at(x, y - 1) } else { 0 };
    let ul = if x > 0 && y > 0 { at(x - 1, y - 1) } else { 0 };
    (l + u - ul) as i16
}

struct Tile {
    w: i64,
    /// half metres, row-major, row 0 at the north edge
    data: Vec<i16>,
}

fn load_dem(path: &Path) -> Result<Tile> {
    let b = std::fs::read(path)?;
    if b.len() < 24 || &b[..8] != MAGIC {
        bail!("not a .dem file");
    }
    let u = |k: usize| u32::from_le_bytes([b[8 + 4 * k], b[9 + 4 * k], b[10 + 4 * k], b[11 + 4 * k]]);
    let (w, h) = (u(0) as usize, u(1) as usize);
    let mut raw = Vec::with_capacity(w * h * 2);
    flate2::read::ZlibDecoder::new(&b[24..]).read_to_end(&mut raw)?;
    if raw.len() != w * h * 2 {
        bail!("truncated");
    }
    let mut data: Vec<i16> = bytemuck::pod_collect_to_vec(&raw);
    for y in 0..h {
        for x in 0..w {
            let p = predict(&data, w, x, y);
            data[y * w + x] = data[y * w + x].wrapping_add(p);
        }
    }
    Ok(Tile { w: w as i64, data })
}

struct Slot {
    tile: OnceLock<Option<Arc<Tile>>>,
    used: AtomicU64,
}

thread_local! {
    /// Each worker's recent tiles, held and read in place: a sample touches no shared counter
    /// (13 workers cloning one tile's Arc per sample made the cache line bounce between cores).
    static LAST: std::cell::RefCell<([((i32, i32), Option<Arc<Tile>>); 4], usize)> =
        std::cell::RefCell::new((std::array::from_fn(|_| ((i32::MIN, 0), None)), 0));
}

pub struct Glo30 {
    dir: PathBuf,
    /// tiles on disk
    index: HashSet<(i32, i32)>,
    slots: Mutex<HashMap<(i32, i32), Arc<Slot>>>,
    clock: AtomicU64,
}

impl Glo30 {
    /// Index the `.dem` tiles (GeoTIFFs without one are converted first, and kept).
    pub fn load(dir: &Path) -> Result<Self> {
        let t = std::time::Instant::now();
        let mut index = HashSet::new();
        let mut tifs = Vec::new();
        if let Ok(rd) = std::fs::read_dir(dir) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                let Some(c) = tile_name_coords(&name) else { continue };
                if name.ends_with("_DEM.dem") {
                    index.insert(c);
                } else if name.ends_with("_DEM.tif") && !e.path().with_extension("dem").exists() {
                    tifs.push((c, e.path()));
                }
            }
        }
        if !tifs.is_empty() {
            println!("GLO-30: converting {} GeoTIFFs to .dem ...", tifs.len());
            let done: Vec<Option<(i32, i32)>> = std::thread::scope(|s| {
                let hs: Vec<_> = tifs
                    .iter()
                    .map(|(c, p)| {
                        s.spawn(move || match convert_tif(p) {
                            Ok(_) => Some(*c),
                            Err(e) => {
                                eprintln!("GLO-30 {}: {e:#}", p.display());
                                None
                            }
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            index.extend(done.into_iter().flatten());
        }
        if !index.is_empty() {
            println!("Copernicus GLO-30: {} tiles indexed ({:.1}s)", index.len(), t.elapsed().as_secs_f32());
        }
        Ok(Glo30 { dir: dir.to_path_buf(), index, slots: Mutex::new(HashMap::new()), clock: AtomicU64::new(0) })
    }

    pub fn has(&self, lat: i32, lon: i32) -> bool {
        self.index.contains(&(lat, lon))
    }

    pub fn tile_count(&self) -> usize {
        self.index.len()
    }

    fn path(&self, (lat, lon): (i32, i32)) -> PathBuf {
        let ns = if lat >= 0 { 'N' } else { 'S' };
        let ew = if lon >= 0 { 'E' } else { 'W' };
        self.dir.join(format!("Copernicus_DSM_COG_10_{ns}{:02}_00_{ew}{:03}_00_DEM.dem", lat.abs(), lon.abs()))
    }

    /// The decoded tile, loading it on first use (concurrent callers wait for the same load).
    /// Read from tile `key` (in this worker's cache, or loaded into it).
    fn with_tile<R>(&self, key: (i32, i32), f: impl FnOnce(&Tile) -> R) -> Option<R> {
        let mut f = Some(f);
        let hit = LAST.with(|l| {
            let l = l.borrow();
            l.0.iter().find(|e| e.0 == key).and_then(|e| e.1.as_ref()).map(|t| (f.take().unwrap())(t))
        });
        if hit.is_some() {
            return hit;
        }
        let t = self.tile(key)?;
        let r = (f.take().unwrap())(&t);
        LAST.with(|l| {
            let mut l = l.borrow_mut();
            let i = l.1;
            l.0[i] = (key, Some(t));
            l.1 = (i + 1) % 4;
        });
        Some(r)
    }

    fn tile(&self, key: (i32, i32)) -> Option<Arc<Tile>> {
        if !self.index.contains(&key) {
            return None;
        }
        let slot = {
            let mut m = self.slots.lock().unwrap();
            if !m.contains_key(&key) && m.len() >= RESIDENT {
                // drop the least recently used decoded tile
                if let Some(old) = m.iter().filter(|(_, s)| s.tile.get().is_some()).min_by_key(|(_, s)| s.used.load(Ordering::Relaxed)).map(|(k, _)| *k) {
                    m.remove(&old);
                }
            }
            m.entry(key).or_insert_with(|| Arc::new(Slot { tile: OnceLock::new(), used: AtomicU64::new(0) })).clone()
        };
        slot.used.store(self.clock.fetch_add(1, Ordering::Relaxed), Ordering::Relaxed);
        let t = slot
            .tile
            .get_or_init(|| match load_dem(&self.path(key)) {
                Ok(t) => Some(Arc::new(t)),
                Err(e) => {
                    eprintln!("GLO-30 {}: {e:#}", self.path(key).display());
                    None
                }
            })
            .clone()?;
        Some(t)
    }

    /// Elevation (m), bicubic (Catmull-Rom); None where a needed tile is missing. Rows are 1"
    /// apart everywhere; each row is interpolated at its own tile's column spacing, so tiles of
    /// different widths (above 50 degrees) join without seams.
    pub fn height(&self, lat: f64, lon: f64) -> Option<f32> {
        let cr = |p0: f32, p1: f32, p2: f32, p3: f32, t: f32| {
            p1 + 0.5 * t * (p2 - p0 + t * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + t * (3.0 * (p1 - p2) + p3 - p0)))
        };
        let fy = (90.0 - lat) * ROWS as f64;
        let y0 = fy.floor();
        let ty = (fy - y0) as f32;
        let y0 = y0 as i64;
        let mut rows = [0f32; 4];
        for (k, dy) in (-1..=2).enumerate() {
            let r = (y0 + dy).clamp(0, 180 * ROWS - 1);
            let tlat = (89 - r.div_euclid(ROWS)) as i32;
            let j = r.rem_euclid(ROWS);
            let w = cols_at(tlat);
            let fx = (lon + 180.0) * w as f64;
            let x0 = fx.floor();
            let tx = (fx - x0) as f32;
            let x0 = x0 as i64;
            let cols = [-1i64, 0, 1, 2].map(|dx| (x0 + dx).rem_euclid(360 * w));
            let key = |c: i64| (tlat, (c.div_euclid(w) - 180) as i32);
            let mut p = [0f32; 4];
            if key(cols[0]) == key(cols[3]) {
                // the usual case: the row's four samples in one tile, one cache lookup
                self.with_tile(key(cols[0]), |t| {
                    for m in 0..4 {
                        p[m] = t.data[(j * t.w + cols[m].rem_euclid(w)) as usize] as f32 * 0.5;
                    }
                })?;
            } else {
                for m in 0..4 {
                    p[m] = self.with_tile(key(cols[m]), |t| t.data[(j * t.w + cols[m].rem_euclid(w)) as usize] as f32 * 0.5)?;
                }
            }
            rows[k] = cr(p[0], p[1], p[2], p[3], tx);
        }
        Some(cr(rows[0], rows[1], rows[2], rows[3], ty))
    }
}

/// All elevation sources.
pub struct Terrain {
    pub etopo: Etopo,
    pub glo: Glo30,
    pub cover: LandCover,
    pub biomes: Option<super::biomes::Biomes>,
}

impl Terrain {
    pub fn load(root: &Path) -> Result<Self> {
        Ok(Terrain {
            etopo: Etopo::load(&root.join("etopo"))?,
            glo: Glo30::load(&root.join("glo30"))?,
            cover: LandCover::load(&root.join("worldcover"))?,
            biomes: super::biomes::Biomes::load(&root.join("ecoregions"))?,
        })
    }

    /// RESOLVE biome number at lat / lon (biomes.rs; 0 = none or no biome data).
    pub fn biome(&self, lat: f64, lon: f64) -> u8 {
        self.biomes.as_ref().map(|b| b.biome(lat, lon)).unwrap_or(0)
    }

    /// Elevation (m): 30 m data where its tiles exist, ETOPO elsewhere. GLO-30 flattens the sea to
    /// exactly 0; there ETOPO's seafloor is used instead.
    pub fn height(&self, lat: f64, lon: f64) -> f32 {
        let coarse = self.etopo.height(lat, lon);
        match self.glo.height(lat, lon) {
            Some(fine) if !(fine == 0.0 && coarse < 0.0) => fine,
            _ => coarse,
        }
    }

    /// Elevation for points `spacing` metres apart: the 30 m data only where the points are close
    /// enough to show it; coarser samples (far terrain, the planet-wide tiles) read ETOPO, without
    /// decoding a 26 MB tile for a sample every few kilometres.
    pub fn height_at(&self, lat: f64, lon: f64, spacing: f64) -> f32 {
        if spacing > 400.0 {
            self.etopo.height(lat, lon)
        } else {
            self.height(lat, lon)
        }
    }

    /// Is any of the 30 m data inside this lat / lon box?
    pub fn fine_in(&self, lat0: f64, lat1: f64, lon0: f64, lon1: f64) -> bool {
        if self.glo.index.is_empty() {
            return false;
        }
        let (a0, a1) = (lat0.floor() as i32, (lat1.ceil() as i32 - 1).max(lat0.floor() as i32));
        let (o0, o1) = (lon0.floor() as i32, (lon1.ceil() as i32 - 1).max(lon0.floor() as i32));
        if (a1 - a0 + 1) * (o1 - o0 + 1) > 4096 || o1 - o0 > 180 {
            return true; // huge box (or across the date line): refine and decide further down
        }
        (a0..=a1).any(|a| (o0..=o1).any(|o| self.glo.has(a, o)))
    }
}
