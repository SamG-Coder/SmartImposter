//! Terrain elevation for the globe: Copernicus DEM GLO-30 (1 arc-second, ~30 m) where its tiles
//! are present in assets/earth/glo30, NOAA ETOPO 2022 (60") everywhere else, blended over ~2 km
//! at the edge of the 30 m region.
//!
//! GLO-30 tiles are 1 x 1 degree, 3600 x 3600 float32 (below 50 degrees latitude), pixel-is-point:
//! pixel (i, j) of tile N46 E008 sits at lat 47 - j / 3600, lon 8 + i / 3600. Each is decoded
//! once and cached next to it as int16 metres.

use super::etopo::{decode_tiff, Etopo};
use super::landcover::LandCover;
use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;

const N: i64 = 3600;

pub struct Glo30 {
    tiles: HashMap<(i32, i32), Vec<i16>>,
    /// bounding box of the loaded tiles (deg): lat_min, lat_max, lon_min, lon_max
    pub bbox: Option<[f64; 4]>,
}

fn tile_name_coords(name: &str) -> Option<(i32, i32)> {
    // Copernicus_DSM_COG_10_N46_00_E008_00_DEM.tif
    let p: Vec<&str> = name.split('_').collect();
    let ns = p.iter().position(|s| s.starts_with('N') || s.starts_with('S'))?;
    let lat: i32 = p[ns][1..].parse().ok()?;
    let lat = if p[ns].starts_with('S') { -lat } else { lat };
    let ew = p.iter().position(|s| (s.starts_with('E') || s.starts_with('W')) && s.len() == 4)?;
    let lon: i32 = p[ew][1..].parse().ok()?;
    let lon = if p[ew].starts_with('W') { -lon } else { lon };
    Some((lat, lon))
}

impl Glo30 {
    pub fn load(dir: &Path) -> Result<Self> {
        let t = std::time::Instant::now();
        let mut files = Vec::new();
        if let Ok(rd) = std::fs::read_dir(dir) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                if name.ends_with("_DEM.tif") {
                    if let Some(c) = tile_name_coords(&name) {
                        files.push((c, e.path()));
                    }
                }
            }
        }
        let loaded: Vec<((i32, i32), Option<Vec<i16>>)> = std::thread::scope(|s| {
            let hs: Vec<_> = files
                .iter()
                .map(|(c, path)| {
                    s.spawn(move || {
                        let cache = path.with_extension("i16");
                        if let Ok(b) = std::fs::read(&cache) {
                            if b.len() == (N * N * 2) as usize {
                                return (*c, Some(bytemuck::pod_collect_to_vec::<u8, i16>(&b)));
                            }
                        }
                        match std::fs::read(path).map_err(anyhow::Error::from).and_then(|b| decode_tiff(&b)) {
                            Ok((w, h, data)) if w as i64 == N && h as i64 == N => {
                                let _ = std::fs::write(&cache, bytemuck::cast_slice::<i16, u8>(&data));
                                (*c, Some(data))
                            }
                            Ok((w, h, _)) => {
                                eprintln!("GLO-30 {}: {w}x{h} tiles (above 50 deg) are not supported yet", path.display());
                                (*c, None)
                            }
                            Err(e) => {
                                eprintln!("GLO-30 {}: {e}", path.display());
                                (*c, None)
                            }
                        }
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut tiles = HashMap::new();
        let mut bbox: Option<[f64; 4]> = None;
        for (c, d) in loaded {
            if let Some(d) = d {
                let (la, lo) = (c.0 as f64, c.1 as f64);
                bbox = Some(match bbox {
                    None => [la, la + 1.0, lo, lo + 1.0],
                    Some(b) => [b[0].min(la), b[1].max(la + 1.0), b[2].min(lo), b[3].max(lo + 1.0)],
                });
                tiles.insert(c, d);
            }
        }
        if !tiles.is_empty() {
            let b = bbox.unwrap();
            println!(
                "Copernicus GLO-30: {} tiles, {:.0}-{:.0} N, {:.0}-{:.0} E ({:.1}s)",
                tiles.len(),
                b[0],
                b[1],
                b[2],
                b[3],
                t.elapsed().as_secs_f32()
            );
        }
        Ok(Glo30 { tiles, bbox })
    }

    /// Global row / column (row 0 at 90 N, column 0 at 180 W, 1" steps).
    fn at(&self, r: i64, c: i64) -> Option<f32> {
        // tile N46 holds the rows with lat in (46, 47]: its row 0 is lat 47 = global row 43 * 3600
        let key = ((89 - r.div_euclid(N)) as i32, (c.div_euclid(N) - 180) as i32);
        let t = self.tiles.get(&key)?;
        Some(t[(r.rem_euclid(N) * N + c.rem_euclid(N)) as usize] as f32)
    }

    pub fn height(&self, lat: f64, lon: f64) -> Option<f32> {
        let fy = (90.0 - lat) * N as f64;
        let fx = (lon + 180.0) * N as f64;
        let (y0, x0) = (fy.floor(), fx.floor());
        let (ty, tx) = ((fy - y0) as f32, (fx - x0) as f32);
        let (y0, x0) = (y0 as i64, x0 as i64);
        let cr = |p0: f32, p1: f32, p2: f32, p3: f32, t: f32| {
            p1 + 0.5 * t * (p2 - p0 + t * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + t * (3.0 * (p1 - p2) + p3 - p0)))
        };
        let mut rows = [0f32; 4];
        for (k, dy) in (-1..=2).enumerate() {
            let r = y0 + dy;
            rows[k] = cr(self.at(r, x0 - 1)?, self.at(r, x0)?, self.at(r, x0 + 1)?, self.at(r, x0 + 2)?, tx);
        }
        Some(cr(rows[0], rows[1], rows[2], rows[3], ty))
    }
}

/// All elevation sources.
pub struct Terrain {
    pub etopo: Etopo,
    pub glo: Glo30,
    pub cover: LandCover,
}

impl Terrain {
    pub fn load(root: &Path) -> Result<Self> {
        Ok(Terrain {
            etopo: Etopo::load(&root.join("etopo"))?,
            glo: Glo30::load(&root.join("glo30"))?,
            cover: LandCover::load(&root.join("worldcover"))?,
        })
    }

    /// Elevation (m): 30 m data inside its region, blended into ETOPO over ~2 km at the edges.
    pub fn height(&self, lat: f64, lon: f64) -> f32 {
        let coarse = self.etopo.height(lat, lon);
        let Some(b) = self.glo.bbox else { return coarse };
        let edge = (lat - b[0]).min(b[1] - lat).min((lon - b[2]) * lat.to_radians().cos()).min((b[3] - lon) * lat.to_radians().cos());
        if edge <= 0.0 {
            return coarse;
        }
        match self.glo.height(lat, lon) {
            Some(fine) => {
                let w = (edge / 0.02).clamp(0.0, 1.0) as f32;
                coarse + (fine - coarse) * w * w * (3.0 - 2.0 * w)
            }
            None => coarse,
        }
    }

    /// Is any of the 30 m data inside this lat / lon box?
    pub fn fine_in(&self, lat0: f64, lat1: f64, lon0: f64, lon1: f64) -> bool {
        match self.glo.bbox {
            Some(b) => lat1 > b[0] && lat0 < b[1] && lon1 > b[2] && lon0 < b[3],
            None => false,
        }
    }
}
