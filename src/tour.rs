//! Scripted globe camera tours (`--tour file`) for recording showcase videos in real time (for
//! example with OBS). A tour is a list of shots. Before each shot the screen is black while the
//! camera waits at the shot's first key for its terrain, trees and ocean to stream in, then the
//! shot fades in, flies a smooth spline through its keys and fades out.
//!
//! File format (`#` starts a comment):
//!
//! ```text
//! shot 24 17.5 cover=0.3 date=2026-06-21 | Lake Como | Lombardy, Italy
//! key 46.05,9.25,300,200,-8
//! key 46.00,9.27,250,190,-6
//! ```
//!
//! `shot <seconds> <local solar hour> [cover=<cloud cover>] [date=<UTC date>] | title | caption`,
//! then two or more `key lat,lon,height above ground (m),heading (deg),pitch (deg)` lines (one key
//! holds the camera still). Keys are spread evenly over the shot; the flight eases in and out.

use anyhow::{bail, Context, Result};
use glam::DVec3;

use crate::globe;

#[derive(Clone, Copy, Debug)]
pub struct Key {
    pub lat: f64,
    pub lon: f64,
    pub height: f64,
    pub heading: f64,
    pub pitch: f64,
}

#[derive(Clone, Debug)]
pub struct Shot {
    pub title: String,
    pub caption: String,
    pub secs: f64,
    pub hour: f64,
    pub date: String,
    pub cover: Option<f32>,
    pub keys: Vec<Key>,
}

impl Shot {
    /// The shot's clock: its local solar hour on its date (UTC seconds), at its first key.
    pub fn utc(&self) -> f64 {
        let day = globe::sky::parse_utc(&self.date).unwrap_or(0.0);
        day + (self.hour - self.keys[0].lon / 15.0).rem_euclid(24.0) * 3600.0
    }
}

pub fn load(path: &str) -> Result<Vec<Shot>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading tour {path}"))?;
    let mut shots: Vec<Shot> = Vec::new();
    for (n, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let err = || format!("{path}:{}: {raw}", n + 1);
        if let Some(rest) = line.strip_prefix("shot ") {
            let parts: Vec<&str> = rest.split('|').map(str::trim).collect();
            let words: Vec<&str> = parts[0].split_whitespace().collect();
            if words.len() < 2 {
                bail!("{}: expected `shot <seconds> <hour> | title | caption`", err());
            }
            let mut shot = Shot {
                title: parts.get(1).unwrap_or(&"").to_string(),
                caption: parts.get(2).unwrap_or(&"").to_string(),
                secs: words[0].parse().with_context(err)?,
                hour: words[1].parse().with_context(err)?,
                date: "2026-06-21".into(),
                cover: None,
                keys: Vec::new(),
            };
            for w in &words[2..] {
                match w.split_once('=') {
                    Some(("cover", v)) => shot.cover = Some(v.parse().with_context(err)?),
                    Some(("date", v)) => shot.date = v.to_string(),
                    _ => bail!("{}: unknown option {w}", err()),
                }
            }
            shots.push(shot);
        } else if let Some(rest) = line.strip_prefix("key ") {
            let v: Vec<f64> = rest.split(',').map(|s| s.trim().parse()).collect::<Result<_, _>>().with_context(err)?;
            if v.len() != 5 {
                bail!("{}: a key is lat,lon,height,heading,pitch", err());
            }
            let Some(shot) = shots.last_mut() else { bail!("{}: key before any shot", err()) };
            shot.keys.push(Key { lat: v[0], lon: v[1], height: v[2], heading: v[3], pitch: v[4] });
        } else {
            bail!("{}: expected `shot` or `key`", err());
        }
    }
    if let Some(s) = shots.iter().find(|s| s.keys.is_empty()) {
        bail!("tour shot '{}' has no keys", s.title);
    }
    if shots.is_empty() {
        bail!("tour {path} has no shots");
    }
    Ok(shots)
}

/// A shot's keys resolved against the terrain: absolute altitude (above the sphere) per key,
/// headings unwrapped so the spline turns the short way.
pub struct Path {
    lat: Vec<f64>,
    lon: Vec<f64>,
    /// ln(altitude above the sphere + 200 m): even over a descent from orbit, linear near the ground
    alt: Vec<f64>,
    heading: Vec<f64>,
    pitch: Vec<f64>,
    /// lowest height above the ground the camera may take (the lowest key, at most 30 m)
    clearance: f64,
}

const ALT_BIAS: f64 = 200.0;

impl Path {
    pub fn new(shot: &Shot, terrain: &globe::dem::Terrain) -> Self {
        let mut p = Path { lat: vec![], lon: vec![], alt: vec![], heading: vec![], pitch: vec![], clearance: 30.0 };
        let mut last_h: Option<f64> = None;
        for k in &shot.keys {
            let d = globe::tiles::dir_from_lat_lon(k.lat, k.lon);
            let alt = globe::tiles::ground_radius(terrain, d) - globe::tiles::R_EARTH + k.height.max(1.7);
            p.lat.push(k.lat);
            // longitudes unwrapped too (a shot may cross the antimeridian)
            let lon = match p.lon.last() {
                Some(&l) => l + (k.lon - l + 540.0).rem_euclid(360.0) - 180.0,
                None => k.lon,
            };
            p.lon.push(lon);
            p.alt.push((alt + ALT_BIAS).ln());
            let h = match last_h {
                Some(l) => l + (k.heading - l + 540.0).rem_euclid(360.0) - 180.0,
                None => k.heading,
            };
            last_h = Some(h);
            p.heading.push(h);
            p.pitch.push(k.pitch);
            p.clearance = p.clearance.min(k.height.max(1.7));
        }
        p
    }

    /// Camera at shot fraction `f` (0..1): Earth-centred position, heading and pitch (degrees).
    pub fn at(&self, f: f64, terrain: &globe::dem::Terrain) -> (DVec3, f64, f64) {
        let n = self.lat.len();
        let s = ease(f.clamp(0.0, 1.0)) * (n - 1) as f64;
        let i = (s.floor() as usize).min(n.saturating_sub(2));
        let t = if n > 1 { s - i as f64 } else { 0.0 };
        let cr = |v: &[f64]| {
            if n == 1 {
                return v[0];
            }
            // Catmull-Rom with the ends mirrored (zero curvature at the first and last key)
            let p1 = v[i];
            let p2 = v[i + 1];
            let p0 = if i > 0 { v[i - 1] } else { 2.0 * p1 - p2 };
            let p3 = if i + 2 < n { v[i + 2] } else { 2.0 * p2 - p1 };
            let (t2, t3) = (t * t, t * t * t);
            0.5 * (2.0 * p1 + (p2 - p0) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 + (3.0 * p1 - p0 - 3.0 * p2 + p3) * t3)
        };
        let d = globe::tiles::dir_from_lat_lon(cr(&self.lat), cr(&self.lon));
        let alt = cr(&self.alt).exp() - ALT_BIAS;
        let ground = globe::tiles::ground_radius(terrain, d);
        let r = (globe::tiles::R_EARTH + alt).max(ground + self.clearance);
        (d * r, cr(&self.heading).rem_euclid(360.0), cr(&self.pitch))
    }
}

/// Shot time to path fraction: constant speed through the middle, easing in over the first and
/// out over the last 20% (a trapezoidal speed profile).
fn ease(f: f64) -> f64 {
    const A: f64 = 0.2;
    let vmax = 1.0 / (1.0 - A);
    if f < A {
        0.5 * vmax * f * f / A
    } else if f > 1.0 - A {
        let g = 1.0 - f;
        1.0 - 0.5 * vmax * g * g / A
    } else {
        0.5 * vmax * A + vmax * (f - A)
    }
}

/// Fade over the first and last `edge` seconds of a shot (0 black .. 1).
pub fn fade(t: f64, secs: f64, edge: f64) -> f32 {
    let s = |x: f64| {
        let x = x.clamp(0.0, 1.0);
        x * x * (3.0 - 2.0 * x)
    };
    (s(t / edge) * s((secs - t) / edge)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ease_is_monotonic_and_spans() {
        assert!(ease(0.0).abs() < 1e-12 && (ease(1.0) - 1.0).abs() < 1e-12);
        assert!((ease(0.5) - 0.5).abs() < 1e-9);
        let mut last = -1.0;
        for i in 0..=1000 {
            let v = ease(i as f64 / 1000.0);
            assert!(v >= last);
            last = v;
        }
    }

    #[test]
    fn parses_a_tour() {
        let p = std::env::temp_dir().join("si_tour_test.txt");
        std::fs::write(&p, "# test\nshot 20 17.5 cover=0.3 | Como | Italy\nkey 46,9,300,200,-8\nkey 46.1,9.1,200,190,-6\n").unwrap();
        let t = load(p.to_str().unwrap()).unwrap();
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].keys.len(), 2);
        assert_eq!(t[0].cover, Some(0.3));
        assert_eq!(t[0].title, "Como");
    }
}
