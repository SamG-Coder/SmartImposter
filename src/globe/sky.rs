//! Time, sun and moon for the globe: where the sun and the moon are (Earth-fixed, ECEF axes) at a
//! UTC time, from the low-precision formulas of the Astronomical Almanac (sun ~0.01 degrees) and
//! Meeus ch. 47's main terms (moon ~0.3 degrees, distance ~1%). The Earth turns under them: the
//! time of day and the season come out of the date and time.

use glam::DVec3;

/// Moon radius and mean distance (km): its disc is ~0.26 degrees in radius from the Earth.
pub const MOON_RADIUS_KM: f64 = 1737.4;
/// Bond-ish albedo of the lunar surface (full-moon brightness, ~0.12).
pub const MOON_ALBEDO: f64 = 0.12;

/// Julian date of a UTC time in seconds since 1970-01-01.
pub fn julian(unix: f64) -> f64 {
    unix / 86_400.0 + 2_440_587.5
}

/// Greenwich mean sidereal time (radians) at Julian date `jd`.
fn gmst(jd: f64) -> f64 {
    let n = jd - 2_451_545.0;
    (280.460_618_37 + 360.985_647_366_29 * n).to_radians().rem_euclid(std::f64::consts::TAU)
}

/// Equatorial (right ascension, declination) to an Earth-fixed direction.
fn ecef(ra: f64, dec: f64, jd: f64) -> DVec3 {
    let lon = ra - gmst(jd);
    DVec3::new(dec.cos() * lon.cos(), dec.cos() * lon.sin(), dec.sin())
}

/// Ecliptic longitude / latitude (radians) to right ascension / declination.
fn equatorial(lambda: f64, beta: f64, n: f64) -> (f64, f64) {
    let eps = (23.439 - 0.000_000_4 * n).to_radians();
    let (x, y, z) = (beta.cos() * lambda.cos(), beta.cos() * lambda.sin(), beta.sin());
    let (y2, z2) = (y * eps.cos() - z * eps.sin(), y * eps.sin() + z * eps.cos());
    (y2.atan2(x), z2.clamp(-1.0, 1.0).asin())
}

/// Direction to the sun, Earth-fixed.
pub fn sun(unix: f64) -> DVec3 {
    let jd = julian(unix);
    let n = jd - 2_451_545.0;
    let l = (280.460 + 0.985_647_4 * n).to_radians();
    let g = (357.528 + 0.985_600_3 * n).to_radians();
    let lambda = l + (1.915f64.to_radians()) * g.sin() + (0.020f64.to_radians()) * (2.0 * g).sin();
    let (ra, dec) = equatorial(lambda, 0.0, n);
    ecef(ra, dec, jd)
}

/// Position of the moon (km from the Earth's centre), Earth-fixed.
pub fn moon(unix: f64) -> DVec3 {
    let jd = julian(unix);
    let n = jd - 2_451_545.0;
    let d = |deg: f64| deg.to_radians();
    let lp = d(218.316 + 13.176_396 * n); // mean longitude
    let mp = d(134.963 + 13.064_993 * n); // mean anomaly
    let m = d(357.528 + 0.985_600_3 * n); // the sun's mean anomaly
    let dd = d(297.850 + 12.190_749 * n); // mean elongation
    let f = d(93.272 + 13.229_350 * n); // argument of latitude
    let lambda = lp
        + d(6.289) * mp.sin()
        + d(1.274) * (2.0 * dd - mp).sin()
        + d(0.658) * (2.0 * dd).sin()
        + d(0.214) * (2.0 * mp).sin()
        - d(0.186) * m.sin()
        - d(0.114) * (2.0 * f).sin();
    let beta = d(5.128) * f.sin() + d(0.281) * (mp + f).sin() + d(0.278) * (mp - f).sin() + d(0.173) * (2.0 * dd - f).sin();
    let dist = 385_000.56 - 20_905.36 * mp.cos() - 3_699.11 * (2.0 * dd - mp).cos() - 2_955.97 * (2.0 * dd).cos();
    let (ra, dec) = equatorial(lambda, beta, n);
    ecef(ra, dec, jd) * dist
}

/// Fraction of the sun's light the moon sends to the Earth at phase angle `alpha` (radians,
/// 0 = full): a Lambert sphere of the moon's albedo, ~2.4e-6 at full moon.
pub fn moonlight(alpha: f64, dist_km: f64) -> f64 {
    let phase = ((std::f64::consts::PI - alpha) * alpha.cos() + alpha.sin()) / std::f64::consts::PI;
    let solid = std::f64::consts::PI * (MOON_RADIUS_KM / dist_km).powi(2);
    // a Lambert disc at full phase averages 2/3 of its centre's radiance (albedo / pi)
    MOON_ALBEDO / std::f64::consts::PI * solid * (2.0 / 3.0) * phase
}

/// UTC "YYYY-MM-DDTHH:MM[:SS]" (or a date alone) in seconds since 1970-01-01.
pub fn parse_utc(s: &str) -> Option<f64> {
    let (date, time) = s.split_once(['T', ' ']).unwrap_or((s, "00:00"));
    let d: Vec<i64> = date.split('-').map(|v| v.parse().ok()).collect::<Option<_>>()?;
    let t: Vec<f64> = time.trim_end_matches('Z').split(':').map(|v| v.parse().ok()).collect::<Option<_>>()?;
    let (y, mo, da) = (*d.first()?, *d.get(1)?, *d.get(2)?);
    // days from civil (Howard Hinnant)
    let y2 = if mo <= 2 { y - 1 } else { y };
    let era = y2.div_euclid(400);
    let yoe = y2 - era * 400;
    let doy = (153 * (mo + if mo > 2 { -3 } else { 9 }) + 2) / 5 + da - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some(days as f64 * 86_400.0 + t.first()? * 3600.0 + t.get(1).unwrap_or(&0.0) * 60.0 + t.get(2).unwrap_or(&0.0))
}

/// Seconds since 1970-01-01 as "YYYY-MM-DD HH:MM UTC".
pub fn format_utc(unix: f64) -> String {
    let days = unix.div_euclid(86_400.0) as i64;
    let secs = unix.rem_euclid(86_400.0);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + (m <= 2) as i64;
    format!("{y:04}-{m:02}-{d:02} {:02}:{:02} UTC", (secs / 3600.0) as u32, ((secs % 3600.0) / 60.0) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_round_trip() {
        let t = parse_utc("2026-06-21T10:30").unwrap();
        assert_eq!(format_utc(t), "2026-06-21 10:30 UTC");
        assert_eq!(parse_utc("1970-01-01").unwrap(), 0.0);
    }

    #[test]
    fn solstice_sun() {
        // June solstice noon at Greenwich: declination ~23.44 N, overhead near longitude 0
        let s = sun(parse_utc("2026-06-21T12:00").unwrap());
        let lat = s.z.asin().to_degrees();
        let lon = s.y.atan2(s.x).to_degrees();
        assert!((lat - 23.44).abs() < 0.1, "lat {lat}");
        assert!(lon.abs() < 1.0, "lon {lon}");
    }

    #[test]
    fn moon_distance_and_phase() {
        // full moon 2026-03-03 11:38 UTC (total lunar eclipse): moon opposite the sun
        let t = parse_utc("2026-03-03T11:38").unwrap();
        let (m, s) = (moon(t), sun(t));
        let km = m.length();
        assert!((356_000.0..407_000.0).contains(&km), "distance {km}");
        let elong = m.normalize().dot(s).acos().to_degrees();
        assert!(elong > 175.0, "elongation {elong}");
        let full = moonlight(0.0, 384_400.0);
        assert!((1.5e-6..4e-6).contains(&full), "full moon light {full}");
    }
}
