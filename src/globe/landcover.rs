//! ESA WorldCover 2021 (10 m land cover, 11 classes) for the globe, wherever its 3 x 3 degree
//! tiles are in assets/earth/worldcover (the whole planet is 2651 tiles, 124 GB).
//!
//! Nothing is decoded up front: each GeoTIFF's tile index is read on first use (the first 64 KB
//! of a cloud-optimised GeoTIFF), and its 1024 x 1024 blocks are inflated on demand by the
//! workers and kept in an LRU of RESIDENT blocks (1 MB each). Samples spread over a large area
//! (far terrain) read the file's overviews (20 m ... ~600 m pixels) instead of single 10 m pixels.
//!
//! Class codes: 10 tree cover, 20 shrubland, 30 grassland, 40 cropland, 50 built-up, 60 bare /
//! sparse vegetation, 70 snow and ice, 80 permanent water, 90 herbaceous wetland, 95 mangroves,
//! 100 moss and lichen, 0 no data.

use super::etopo::{read_tile, tiff_layouts, TiffLayout};
use anyhow::{bail, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// 10 m cells per 3 degree tile side.
const N: usize = 36_000;
/// Block side, and the levels every WorldCover file has (10 m ... 36000 >> 6 = 562 pixels).
const BLOCK: usize = 1024;
const LEVELS: usize = 7;
/// Decoded blocks kept in memory (1 MB each).
const RESIDENT: usize = 2048;

/// Class groups stored per terrain vertex (tiles.rs), in this order.
pub const GROUPS: usize = 10;
pub const TREE: usize = 0;
pub const SHRUB: usize = 1;
pub const GRASS: usize = 2;
pub const CROP: usize = 3;
pub const BARE: usize = 5;
pub const WATER: usize = 7;
pub const WETLAND: usize = 8;
pub const MOSS: usize = 9;

pub fn group(class: u8) -> Option<usize> {
    Some(match class {
        10 | 95 => 0, // tree cover (mangroves too)
        20 => 1,      // shrubland
        30 => 2,      // grassland
        40 => 3,      // cropland
        50 => 4,      // built-up
        60 => 5,      // bare / sparse
        70 => 6,      // snow and ice
        80 => 7,      // water
        90 => 8,      // herbaceous wetland
        100 => 9,     // moss and lichen
        _ => return None,
    })
}

/// Index of a 3 degree tile (south-west corner) in the presence grid.
fn cell((lat, lon): (i32, i32)) -> Option<usize> {
    let (y, x) = ((lat + 90) / 3, (lon + 180) / 3);
    (lat >= -90 && lat < 90 && lon >= -180 && lon < 180).then(|| y as usize * 120 + x as usize)
}

fn coords(name: &str) -> Option<(i32, i32)> {
    // ESA_WorldCover_10m_2021_v200_N45E006_Map.tif
    let part = name.split('_').find(|p| p.len() == 7 && (p.starts_with('N') || p.starts_with('S')))?;
    let lat: i32 = part[1..3].parse().ok()?;
    let lat = if part.starts_with('S') { -lat } else { lat };
    let lon: i32 = part[4..7].parse().ok()?;
    let lon = if &part[3..4] == "W" { -lon } else { lon };
    Some((lat, lon))
}

/// Read `len` bytes at `off` (positioned, so threads share the file handle).
fn read_at(f: &std::fs::File, off: u64, len: usize) -> std::io::Result<Vec<u8>> {
    let mut buf = vec![0u8; len];
    let mut done = 0;
    while done < len {
        #[cfg(windows)]
        let n = std::os::windows::fs::FileExt::seek_read(f, &mut buf[done..], off + done as u64)?;
        #[cfg(unix)]
        let n = std::os::unix::fs::FileExt::read_at(f, &mut buf[done..], off + done as u64)?;
        if n == 0 {
            buf.truncate(done);
            break;
        }
        done += n;
    }
    Ok(buf)
}

/// An open WorldCover GeoTIFF: its file and the block index of each level (full resolution,
/// then the overviews).
struct Cog {
    file: std::fs::File,
    levels: Vec<TiffLayout>,
}

fn open_cog(path: &Path) -> Result<Cog> {
    let file = std::fs::File::open(path)?;
    let mut len = 64 << 10;
    loop {
        let head = read_at(&file, 0, len)?;
        match tiff_layouts(&head) {
            Ok(levels) => {
                let l = &levels[0];
                // level i is N >> i square in 1024 x 1024 blocks (class_at relies on it)
                let regular = levels.iter().enumerate().all(|(i, o)| o.w == N >> i && o.h == N >> i && o.tw == BLOCK && o.th == BLOCK);
                if l.bits != 8 || !regular || l.pred != 1 || levels.iter().any(|o| o.bits != 8 || o.pred != 1) {
                    bail!("unexpected WorldCover layout {}x{} {} bits", l.w, l.h, l.bits);
                }
                return Ok(Cog { file, levels });
            }
            Err(_) if head.len() == len && len < 16 << 20 => len *= 4,
            Err(e) => return Err(e),
        }
    }
}

type Lazy<T> = Arc<(OnceLock<Option<Arc<T>>>, AtomicU64)>;

thread_local! {
    /// Each worker's last block (weak: never keeps an evicted block alive).
    /// Each worker's recent blocks (weak: never keeps an evicted block alive). Several: a tile's
    /// vertices read full-resolution blocks and overviews in turn, and samples straddle blocks.
    static LAST: std::cell::RefCell<([((i32, i32, u32), Option<Arc<Vec<u8>>>); 8], usize)> =
        std::cell::RefCell::new((std::array::from_fn(|_| ((i32::MIN, 0, 0), None)), 0));
}

pub struct LandCover {
    /// 3 degree tiles on disk, by south-west corner
    index: HashMap<(i32, i32), PathBuf>,
    /// the same as a 60 x 120 grid of 3 degree cells (the hot path: no hashing)
    present: Vec<bool>,
    cogs: Mutex<HashMap<(i32, i32), Lazy<Cog>>>,
    blocks: Mutex<HashMap<(i32, i32, u32), Lazy<Vec<u8>>>>,
    clock: AtomicU64,
}

impl LandCover {
    pub fn load(dir: &Path) -> Result<Self> {
        let mut index = HashMap::new();
        if let Ok(rd) = std::fs::read_dir(dir) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                if name.ends_with("_Map.tif") {
                    if let Some(c) = coords(&name) {
                        index.insert(c, e.path());
                    }
                }
            }
        }
        if !index.is_empty() {
            println!("ESA WorldCover: {} tiles indexed (10 m, decoded on demand)", index.len());
        }
        let mut present = vec![false; 60 * 120];
        for k in index.keys() {
            if let Some(i) = cell(*k) {
                present[i] = true;
            }
        }
        Ok(LandCover { index, present, cogs: Mutex::new(HashMap::new()), blocks: Mutex::new(HashMap::new()), clock: AtomicU64::new(0) })
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    fn cog(&self, key: (i32, i32)) -> Option<Arc<Cog>> {
        let path = self.index.get(&key)?;
        let slot = self.cogs.lock().unwrap().entry(key).or_insert_with(|| Arc::new((OnceLock::new(), AtomicU64::new(0)))).clone();
        slot.0
            .get_or_init(|| match open_cog(path) {
                Ok(c) => Some(Arc::new(c)),
                Err(e) => {
                    eprintln!("WorldCover {}: {e:#}", path.display());
                    None
                }
            })
            .clone()
    }

    /// Decoded block `b` of level `level` of tile `key`.
    fn block(&self, key: (i32, i32), level: usize, b: u32) -> Option<Arc<Vec<u8>>> {
        let bk = (key.0, key.1, (level as u32) << 24 | b);

        if !self.index.contains_key(&key) {
            return None;
        }
        let slot = {
            let mut m = self.blocks.lock().unwrap();
            if !m.contains_key(&bk) && m.len() >= RESIDENT {
                // drop the least recently used eighth of the decoded blocks
                let mut old: Vec<(u64, (i32, i32, u32))> =
                    m.iter().filter(|(_, s)| s.0.get().is_some()).map(|(k, s)| (s.1.load(Ordering::Relaxed), *k)).collect();
                old.sort_unstable();
                for (_, k) in old.into_iter().take(RESIDENT / 8) {
                    m.remove(&k);
                }
            }
            m.entry(bk).or_insert_with(|| Arc::new((OnceLock::new(), AtomicU64::new(0)))).clone()
        };
        slot.1.store(self.clock.fetch_add(1, Ordering::Relaxed), Ordering::Relaxed);
        let v = slot
            .0
            .get_or_init(|| {
                let cog = self.cog(key)?;
                let l = cog.levels.get(level)?;
                let t = b as usize;
                let src = read_at(&cog.file, l.offs[t], l.lens[t] as usize).ok()?;
                // read_tile indexes the source by the layout's offsets: give it a one-tile layout
                let one = TiffLayout { offs: vec![0], lens: vec![src.len() as u64], ..l.clone() };
                let mut raw = vec![0u8; l.tw * l.th];
                match read_tile(&one, &src, 0, &mut raw) {
                    Ok(()) => Some(Arc::new(raw)),
                    Err(e) => {
                        eprintln!("WorldCover block {b} (level {level}) of {key:?}: {e:#}");
                        None
                    }
                }
            })
            .clone()?;

        Some(v)
    }

    /// Class code at lat / lon (0 = no data or no tile), full resolution.
    pub fn class(&self, lat: f64, lon: f64) -> u8 {
        self.class_at(lat, lon, 0)
    }

    /// Class code from level `level` (0 = 10 m, each further level halves the resolution; past
    /// the coarsest overview the coarsest is used).
    fn class_at(&self, lat: f64, lon: f64, level: usize) -> u8 {
        let key = ((lat / 3.0).floor() as i32 * 3, (lon / 3.0).floor() as i32 * 3);
        if !cell(key).is_some_and(|i| self.present[i]) {
            return 0;
        }
        let level = level.min(LEVELS - 1);
        let w = N >> level;
        let y = (((key.0 as f64 + 3.0 - lat) * w as f64 / 3.0) as usize).min(w - 1);
        let x = (((lon - key.1 as f64) * w as f64 / 3.0) as usize).min(w - 1);
        let b = ((y / BLOCK) * w.div_ceil(BLOCK) + x / BLOCK) as u32;
        let bk = (key.0, key.1, (level as u32) << 24 | b);
        let off = (y % BLOCK) * BLOCK + x % BLOCK;
        // each worker's recent blocks, read in place: no shared counters or locks per sample
        if let Some(v) = LAST.with(|l| l.borrow().0.iter().find(|e| e.0 == bk).and_then(|e| e.1.as_ref().map(|b| b[off]))) {
            return v;
        }
        let Some(blk) = self.block(key, level, b) else { return 0 };
        let v = blk[off];
        LAST.with(|l| {
            let mut l = l.borrow_mut();
            let i = l.1;
            l.0[i] = (bk, Some(blk));
            l.1 = (i + 1) % 8;
        });
        v
    }

    /// Is it water (permanent water bodies) at lat / lon, read at a resolution for samples
    /// `spacing` metres apart? None where there is no land cover data.
    pub fn water_at(&self, lat: f64, lon: f64, spacing: f64) -> Option<bool> {
        let level = ((spacing / (3.0 * 111_320.0 / N as f64)).log2().floor().max(0.0)) as usize;
        match self.class_at(lat, lon, level) {
            0 => None,
            c => Some(c == 80),
        }
    }

    /// Class group at lat / lon (None = no data).
    pub fn class_group(&self, lat: f64, lon: f64) -> Option<usize> {
        group(self.class(lat, lon))
    }

    /// Class fractions over a square of `size` metres around lat / lon (k x k samples).
    pub fn fractions(&self, lat: f64, lon: f64, size: f64) -> Option<[f32; GROUPS]> {
        if self.index.is_empty() {
            return None;
        }
        let k = ((size / 20.0).ceil() as usize).clamp(1, 6);
        // the coarsest level whose pixels are no larger than the sample spacing
        let spacing = size / k as f64;
        let level = ((spacing / (3.0 * 111_320.0 / N as f64)).log2().floor().max(0.0)) as usize;
        let dlat = size / 111_320.0;
        let dlon = dlat / lat.to_radians().cos().max(0.05);
        let mut f = [0f32; GROUPS];
        let mut n = 0.0;
        for j in 0..k {
            for i in 0..k {
                let (a, b) = ((i as f64 + 0.5) / k as f64 - 0.5, (j as f64 + 0.5) / k as f64 - 0.5);
                if let Some(g) = group(self.class_at(lat + b * dlat, lon + a * dlon, level)) {
                    f[g] += 1.0;
                    n += 1.0;
                }
            }
        }
        if n == 0.0 {
            return None;
        }
        f.iter_mut().for_each(|v| *v /= n);
        Some(f)
    }
}
