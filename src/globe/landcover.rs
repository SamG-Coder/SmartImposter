//! ESA WorldCover 2021 (10 m land cover, 11 classes) for the globe, where its 3 x 3 degree tiles
//! are present in assets/earth/worldcover. Decoded once to 20 m (2 x 2 majority) and cached next
//! to the tile (324 MB each).
//!
//! Class codes: 10 tree cover, 20 shrubland, 30 grassland, 40 cropland, 50 built-up, 60 bare /
//! sparse vegetation, 70 snow and ice, 80 permanent water, 90 herbaceous wetland, 95 mangroves,
//! 100 moss and lichen, 0 no data.

use super::etopo::{read_tile, tiff_layout};
use anyhow::{bail, Result};
use std::collections::HashMap;
use std::path::Path;

/// 20 m cells per 3 degree tile side.
const N: usize = 18_000;
const PER_DEG: f64 = N as f64 / 3.0;

/// Class groups stored per terrain vertex (tiles.rs), in this order.
pub const GROUPS: usize = 10;
pub const TREE: usize = 0;
pub const WATER: usize = 7;

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

pub struct LandCover {
    /// key: (lat, lon) of the tile's south-west corner (multiples of 3); rows from the north edge
    tiles: HashMap<(i32, i32), Vec<u8>>,
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

fn decode_half(b: &[u8]) -> Result<Vec<u8>> {
    let l = tiff_layout(b)?;
    if l.bits != 8 || l.w != 2 * N || l.h != 2 * N || l.pred != 1 {
        bail!("unexpected WorldCover layout {}x{} {} bits", l.w, l.h, l.bits);
    }
    let (tw, th) = (l.tw, l.th);
    let tiles_x = l.w.div_ceil(tw);
    let n_tiles = l.offs.len();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let parts: Vec<Vec<(usize, Vec<u8>)>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|k| {
                let l = &l;
                s.spawn(move || -> Result<Vec<(usize, Vec<u8>)>> {
                    let mut raw = vec![0u8; tw * th];
                    let mut out = Vec::new();
                    for t in (k..n_tiles).step_by(threads) {
                        read_tile(l, b, t, &mut raw)?;
                        // 2 x 2 majority (ties: the first)
                        let mut half = vec![0u8; (tw / 2) * (th / 2)];
                        for y in 0..th / 2 {
                            for x in 0..tw / 2 {
                                let c = [raw[2 * y * tw + 2 * x], raw[2 * y * tw + 2 * x + 1], raw[(2 * y + 1) * tw + 2 * x], raw[(2 * y + 1) * tw + 2 * x + 1]];
                                let mut best = (0, c[0]);
                                for &v in &c {
                                    let n = c.iter().filter(|&&u| u == v).count();
                                    if n > best.0 {
                                        best = (n, v);
                                    }
                                }
                                half[y * (tw / 2) + x] = best.1;
                            }
                        }
                        out.push((t, half));
                    }
                    Ok(out)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect::<Result<Vec<_>>>()
    })?;
    let mut img = vec![0u8; N * N];
    let (hw, hh) = (tw / 2, th / 2);
    for (t, half) in parts.into_iter().flatten() {
        let (tx, ty) = ((t % tiles_x) * hw, (t / tiles_x) * hh);
        for y in 0..hh {
            let gy = ty + y;
            if gy >= N {
                break;
            }
            let row_len = hw.min(N.saturating_sub(tx));
            img[gy * N + tx..gy * N + tx + row_len].copy_from_slice(&half[y * hw..y * hw + row_len]);
        }
    }
    Ok(img)
}

impl LandCover {
    pub fn load(dir: &Path) -> Result<Self> {
        let t = std::time::Instant::now();
        let mut tiles = HashMap::new();
        if let Ok(rd) = std::fs::read_dir(dir) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                if !name.ends_with("_Map.tif") {
                    continue;
                }
                let Some(c) = coords(&name) else { continue };
                let cache = e.path().with_extension("20m");
                let data = match std::fs::read(&cache) {
                    Ok(b) if b.len() == N * N => b,
                    _ => {
                        let b = std::fs::read(e.path())?;
                        let img = decode_half(&b)?;
                        let _ = std::fs::write(&cache, &img);
                        img
                    }
                };
                tiles.insert(c, data);
            }
        }
        if !tiles.is_empty() {
            println!("ESA WorldCover: {} tiles at 20 m ({:.1}s)", tiles.len(), t.elapsed().as_secs_f32());
        }
        Ok(LandCover { tiles })
    }

    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }

    /// Class code at lat / lon (0 = no data).
    pub fn class(&self, lat: f64, lon: f64) -> u8 {
        let (tlat, tlon) = ((lat / 3.0).floor() as i32 * 3, (lon / 3.0).floor() as i32 * 3);
        let Some(t) = self.tiles.get(&(tlat, tlon)) else { return 0 };
        let y = ((tlat as f64 + 3.0 - lat) * PER_DEG) as usize;
        let x = ((lon - tlon as f64) * PER_DEG) as usize;
        t[y.min(N - 1) * N + x.min(N - 1)]
    }

    /// Class fractions over a square of `size` metres around lat / lon (k x k samples).
    pub fn fractions(&self, lat: f64, lon: f64, size: f64) -> Option<[f32; GROUPS]> {
        if self.tiles.is_empty() {
            return None;
        }
        let k = ((size / 20.0).ceil() as usize).clamp(1, 6);
        let dlat = size / 111_320.0;
        let dlon = dlat / lat.to_radians().cos().max(0.05);
        let mut f = [0f32; GROUPS];
        let mut n = 0.0;
        for j in 0..k {
            for i in 0..k {
                let (a, b) = ((i as f64 + 0.5) / k as f64 - 0.5, (j as f64 + 0.5) / k as f64 - 0.5);
                if let Some(g) = group(self.class(lat + b * dlat, lon + a * dlon)) {
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
