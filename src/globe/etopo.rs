//! NOAA ETOPO 2022 global relief (60 arc-second, ice surface elevation): metres above sea level
//! (negative = sea floor) at any latitude / longitude.
//!
//! The GeoTIFF (21600 x 10800 float32, 256 x 256 tiles, deflate + floating-point predictor) is
//! decoded once and cached next to it as int16 metres (466 MB), which loads in a second.

use anyhow::{bail, Context, Result};
use std::io::Read;
use std::path::Path;

pub struct Etopo {
    pub w: usize,
    pub h: usize,
    /// Row-major, row 0 = north edge; centre of pixel (i, j): lon -180 + (i + 0.5) / 60, lat 90 - (j + 0.5) / 60.
    data: Vec<i16>,
    lon0: f64,
    lat0: f64,
    step: f64,
}

const TIF: &str = "ETOPO_2022_v1_60s_N90W180_surface.tif";
const CACHE: &str = "etopo_60s_i16.bin";

impl Etopo {
    pub fn load(dir: &Path) -> Result<Self> {
        let t = std::time::Instant::now();
        let cache = dir.join(CACHE);
        if let Ok(bytes) = std::fs::read(&cache) {
            if bytes.len() > 32 {
                let hdr: &[u64] = bytemuck::cast_slice(&bytes[..16]);
                let (w, h) = (hdr[0] as usize, hdr[1] as usize);
                if bytes.len() == 16 + w * h * 2 {
                    let data: Vec<i16> = bytemuck::pod_collect_to_vec(&bytes[16..]);
                    println!("ETOPO 2022 60\": {w}x{h} from cache ({:.1}s)", t.elapsed().as_secs_f32());
                    return Ok(Etopo { w, h, data, lon0: -180.0, lat0: 90.0, step: 1.0 / 60.0 });
                }
            }
        }
        let path = dir.join(TIF);
        let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        let (w, h, data) = decode_tiff(&bytes)?;
        let mut out = Vec::with_capacity(16 + data.len() * 2);
        out.extend_from_slice(bytemuck::cast_slice(&[w as u64, h as u64]));
        out.extend_from_slice(bytemuck::cast_slice(&data));
        std::fs::write(&cache, out).context("writing ETOPO cache")?;
        println!("ETOPO 2022 60\": decoded {w}x{h} GeoTIFF ({:.1}s), cached as int16", t.elapsed().as_secs_f32());
        Ok(Etopo { w, h, data, lon0: -180.0, lat0: 90.0, step: 1.0 / 60.0 })
    }

    /// The grid (row 0 = north), int16 metres.
    pub fn raw(&self) -> &[i16] {
        &self.data
    }

    fn at(&self, i: isize, j: isize) -> f32 {
        let i = i.rem_euclid(self.w as isize) as usize;
        let j = j.clamp(0, self.h as isize - 1) as usize;
        self.data[j * self.w + i] as f32
    }

    /// Elevation (m) at latitude / longitude in degrees: bicubic (Catmull-Rom), so the 1.85 km
    /// grid shows no creases when seen from low altitude.
    pub fn height(&self, lat: f64, lon: f64) -> f32 {
        let fx = (lon - self.lon0) / self.step - 0.5;
        let fy = (self.lat0 - lat) / self.step - 0.5;
        let (x0, y0) = (fx.floor(), fy.floor());
        let (tx, ty) = ((fx - x0) as f32, (fy - y0) as f32);
        let (x0, y0) = (x0 as isize, y0 as isize);
        let cr = |p0: f32, p1: f32, p2: f32, p3: f32, t: f32| {
            p1 + 0.5 * t * (p2 - p0 + t * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + t * (3.0 * (p1 - p2) + p3 - p0)))
        };
        let row = |dy: isize| cr(self.at(x0 - 1, y0 + dy), self.at(x0, y0 + dy), self.at(x0 + 1, y0 + dy), self.at(x0 + 2, y0 + dy), tx);
        cr(row(-1), row(0), row(1), row(2), ty)
    }
}

/// Minimal baseline-TIFF reader for this file: little-endian, one sample of float32 per pixel,
/// tiled, deflate (8) or none (1), predictor 1 or 3 (floating point).
fn decode_tiff(b: &[u8]) -> Result<(usize, usize, Vec<i16>)> {
    if &b[..4] != b"II*\0" {
        bail!("not a little-endian classic TIFF");
    }
    let u16_at = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
    let u32_at = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    let ifd = u32_at(4) as usize;
    let n = u16_at(ifd) as usize;
    let mut tags = std::collections::HashMap::new();
    for k in 0..n {
        let e = ifd + 2 + k * 12;
        let (tag, typ, cnt) = (u16_at(e), u16_at(e + 2), u32_at(e + 4) as usize);
        let size = match typ { 3 => 2, 4 => 4, 12 => 8, 1 | 2 | 7 => 1, _ => 4 } * cnt;
        let off = if size <= 4 { e + 8 } else { u32_at(e + 8) as usize };
        let vals: Vec<u64> = (0..cnt)
            .map(|i| match typ {
                3 => u16_at(off + i * 2) as u64,
                4 => u32_at(off + i * 4) as u64,
                _ => 0,
            })
            .collect();
        tags.insert(tag, vals);
    }
    let get = |t: u16| tags.get(&t).and_then(|v| v.first().copied()).unwrap_or(0) as usize;
    let (w, h) = (get(256), get(257));
    let (bits, comp, pred, fmt) = (get(258), get(259), get(317).max(1), get(339));
    let (tw, th) = (get(322), get(323));
    if bits != 32 || fmt != 3 || tw == 0 || th == 0 {
        bail!("unsupported TIFF layout: {bits} bits, format {fmt}, tiles {tw}x{th}");
    }
    let offs = tags.get(&324).context("tile offsets")?.clone();
    let lens = tags.get(&325).context("tile byte counts")?.clone();
    let tiles_x = w.div_ceil(tw);
    let mut out = vec![0i16; w * h];
    let mut raw = vec![0u8; tw * th * 4];
    let mut row = vec![0u8; tw * 4];
    for (t, (&o, &l)) in offs.iter().zip(&lens).enumerate() {
        let src = &b[o as usize..(o + l) as usize];
        match comp {
            1 => {
                let len = raw.len();
                raw.copy_from_slice(&src[..len]);
            }
            8 | 32946 => {
                let mut z = flate2::read::ZlibDecoder::new(src);
                z.read_exact(&mut raw).context("inflating a tile")?;
            }
            c => bail!("unsupported TIFF compression {c}"),
        }
        let (tx, ty) = ((t % tiles_x) * tw, (t / tiles_x) * th);
        for r in 0..th {
            let line = &mut raw[r * tw * 4..(r + 1) * tw * 4];
            let vals: Vec<f32> = if pred == 3 {
                // floating-point predictor: byte differences across the row, bytes split into
                // planes (most significant first)
                for k in 1..line.len() {
                    line[k] = line[k].wrapping_add(line[k - 1]);
                }
                row.copy_from_slice(line);
                (0..tw).map(|i| f32::from_be_bytes([row[i], row[tw + i], row[2 * tw + i], row[3 * tw + i]])).collect()
            } else {
                line.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
            };
            let y = ty + r;
            if y >= h {
                break;
            }
            for (i, v) in vals.iter().enumerate() {
                let x = tx + i;
                if x < w {
                    out[y * w + x] = v.round().clamp(-32768.0, 32767.0) as i16;
                }
            }
        }
    }
    Ok((w, h, out))
}
