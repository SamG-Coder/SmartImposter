//! Ocean waves for the globe: a wind-driven sea simulated the way film and game oceans are
//! (Tessendorf, "Simulating Ocean Water"), on a background thread.
//!
//! - Spectrum: JONSWAP wind sea (Hasselmann et al. 1973) for the local wind speed and fetch, plus a
//!   narrower swell from distant storms, spread over direction with a cos^2s(theta / 2) law.
//! - Four cascades (CASCADES patches of N x N waves, SIZES metres) split the wavenumbers between
//!   them; each is an inverse FFT of the spectrum advanced to the current time (deep-water
//!   dispersion w^2 = g k), so every wave moves at its own speed.
//! - Per texel: choppy horizontal displacement (sharp crests, round troughs), height, slopes and
//!   the displacement's Jacobian. Where the Jacobian is lowest the crest is about to fold:
//!   whitecaps start there, sized so their area matches the measured whitecap coverage for the
//!   wind (Monahan & O'Muircheartaigh: 3.84e-6 U^3.41), and fade over a few seconds.
//! - The slope variance the cascades do not hold (waves shorter than the finest texel) is the
//!   Cox-Munk total (0.003 + 0.00512 U) minus what the spectrum resolves: main.rgen uses it as the
//!   surface roughness for the sun glitter and the reflections.
//!
//! Output per cascade: texture A = (Dx, h, Dz, foam), texture B = (dh/dx, dh/dz, dDx/dx, dDz/dz),
//! RGBA16F with full mip chains, in patch coordinates (x, z) along the cube face's (u, v).

use glam::DVec3;
use std::sync::{Arc, Condvar, Mutex};

pub const N: usize = 128;
pub const CASCADES: usize = 4;
/// Patch sizes (m): powers of two, so they tile the texture coordinates (tiles.rs TEX_PERIOD).
pub const SIZES: [f32; CASCADES] = [1024.0, 256.0, 64.0, 16.0];
/// Horizontal displacement scale (Tessendorf's lambda): crest sharpness.
pub const CHOP: f32 = 0.9;
pub const MIPS: usize = 8; // 128 .. 1
const G: f32 = 9.81;

/// Wind and sea state at a place.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SeaState {
    /// wind speed 10 m above the sea (m/s)
    pub wind: f32,
    /// direction the wind blows towards, in patch coordinates (unit)
    pub dir: [f32; 2],
    /// fetch (m): how far the wind has blown over open water
    pub fetch: f32,
    /// significant height (m), period (s) and direction (patch coordinates) of the swell
    pub swell_hs: f32,
    pub swell_t: f32,
    pub swell_dir: [f32; 2],
}

/// One simulated frame: the texel data of 2 x CASCADES textures (mips included, f16 bits) and
/// statistics for the shader.
pub struct Frame {
    /// [cascade][A or B]: f16 RGBA texels, level 0 then the smaller levels
    pub tex: Vec<Vec<u16>>,
    /// per cascade: slope variance (total of both directions) the cascade holds
    pub slope_var: [f32; CASCADES],
    /// slope variance below the finest cascade (Cox-Munk total minus the resolved)
    pub residual: f32,
    /// largest wave height above the mean (m): the slab the shader marches through
    pub amplitude: f32,
    /// whitecap coverage (0..1) for the wind
    pub coverage: f32,
    pub state: SeaState,
}

/// Wind over the sea at a place (climatology): trade winds, doldrums, westerlies, polar
/// easterlies, varied by large weather patterns. Returns (speed m/s, towards-direction as
/// (east, north) unit vector).
pub fn wind_at(lat: f64, lon: f64) -> (f32, [f32; 2]) {
    let a = lat.abs();
    let s = |e0: f64, e1: f64, x: f64| {
        let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    // zonal bands: speed and the wind's bearing (from-direction, degrees clockwise from north)
    let doldrums = 1.0 - s(3.0, 8.0, a);
    let trades = s(3.0, 8.0, a) * (1.0 - s(26.0, 32.0, a));
    let horse = s(26.0, 32.0, a) * (1.0 - s(34.0, 40.0, a));
    let westerlies = s(34.0, 40.0, a) * (1.0 - s(60.0, 66.0, a));
    let polar = s(60.0, 66.0, a);
    let south = lat < 0.0;
    let speed = doldrums * 3.0 + trades * 7.0 + horse * 4.5 + westerlies * if south { 12.5 } else { 9.5 } + polar * 7.0;
    // from-bearings: NE trades (SE in the south), W-SW westerlies, E polar
    let from = doldrums * 90.0 + trades * if south { 120.0 } else { 60.0 } + horse * 270.0 + westerlies * if south { 280.0 } else { 250.0 } + polar * 80.0;
    // weather: highs and lows a few hundred km across vary both (stable in time here)
    let p = DVec3::new(lat.to_radians().cos() * lon.to_radians().cos(), lat.to_radians().cos() * lon.to_radians().sin(), lat.to_radians().sin()) * 6371.0;
    let n1 = noise3(p / 900.0) - 0.5;
    let n2 = noise3(p / 900.0 + DVec3::splat(17.3)) - 0.5;
    let speed = (speed * (1.0 + 0.9 * n1)).clamp(1.0, 20.0);
    let from = (from + 70.0 * n2).to_radians();
    // towards = from + 180 degrees; bearing to (east, north)
    let to = from + std::f64::consts::PI;
    (speed as f32, [to.sin() as f32, to.cos() as f32])
}

fn noise3(p: DVec3) -> f64 {
    let c = p.floor();
    let f = p - c;
    let u = f * f * (DVec3::splat(3.0) - 2.0 * f);
    let h = |x: f64, y: f64, z: f64| {
        let n = ((c.x + x) as i64).wrapping_mul(73856093) ^ ((c.y + y) as i64).wrapping_mul(19349663) ^ ((c.z + z) as i64).wrapping_mul(83492791);
        let mut v = n as u64;
        v ^= v >> 33;
        v = v.wrapping_mul(0xff51afd7ed558ccd);
        v ^= v >> 33;
        (v >> 11) as f64 / (1u64 << 53) as f64
    };
    let l = |a: f64, b: f64, t: f64| a + (b - a) * t;
    l(
        l(l(h(0., 0., 0.), h(1., 0., 0.), u.x), l(h(0., 1., 0.), h(1., 1., 0.), u.x), u.y),
        l(l(h(0., 0., 1.), h(1., 0., 1.), u.x), l(h(0., 1., 1.), h(1., 1., 1.), u.x), u.y),
        u.z,
    )
}

/// JONSWAP frequency spectrum S(w) (m^2 s) for wind speed u (m/s) and fetch (m).
fn jonswap(w: f32, u: f32, fetch: f32) -> f32 {
    // beyond the fetch of a fully developed sea (peak at 0.855 g / U, Pierson-Moskowitz) the
    // waves stop growing
    let fetch = fetch.min(G * G / (u * (0.855 * G / u / 22.0).powi(3)));
    let alpha = 0.076 * (u * u / (fetch * G)).powf(0.22);
    let wp = 22.0 * (G * G / (u * fetch)).powf(1.0 / 3.0);
    jonswap_shape(w, alpha, wp)
}

fn jonswap_shape(w: f32, alpha: f32, wp: f32) -> f32 {
    if w <= 1e-4 {
        return 0.0;
    }
    let sigma = if w <= wp { 0.07 } else { 0.09 };
    let r = (-((w - wp) * (w - wp)) / (2.0 * sigma * sigma * wp * wp)).exp();
    alpha * G * G / w.powi(5) * (-1.25 * (wp / w).powi(4)).exp() * 3.3f32.powf(r)
}

/// Normalised cos^2s(theta / 2) spreading.
fn spread(theta: f32, s: f32) -> f32 {
    // normalisation Gamma(s+1) / (2 sqrt(pi) Gamma(s+1/2)), from ln-gamma
    let norm = (lgamma(s + 1.0) - lgamma(s + 0.5)).exp() / (2.0 * std::f32::consts::PI.sqrt());
    norm * (theta * 0.5).cos().abs().powf(2.0 * s)
}

fn lgamma(x: f32) -> f32 {
    // Stirling series (x >= 0.5 here, accurate to ~1e-4)
    let x = x as f64;
    let (x, shift) = if x < 7.0 { (x + 7.0, (0..7).map(|k| (x + k as f64).ln()).sum::<f64>()) } else { (x, 0.0) };
    ((x - 0.5) * x.ln() - x + 0.5 * (2.0 * std::f64::consts::PI).ln() + 1.0 / (12.0 * x) - 1.0 / (360.0 * x * x * x) - shift) as f32
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }
    /// standard normal pair (Box-Muller)
    fn gauss(&mut self) -> (f32, f32) {
        let (a, b) = (self.next(), self.next());
        let r = (-2.0 * a.ln()).sqrt();
        let a = std::f32::consts::TAU * b;
        (r * a.cos(), r * a.sin())
    }
}

#[derive(Clone, Copy, Default)]
struct C(f32, f32);
impl C {
    fn mul(self, o: C) -> C {
        C(self.0 * o.0 - self.1 * o.1, self.0 * o.1 + self.1 * o.0)
    }
    fn add(self, o: C) -> C {
        C(self.0 + o.0, self.1 + o.1)
    }
    fn scale(self, s: f32) -> C {
        C(self.0 * s, self.1 * s)
    }
    fn conj(self) -> C {
        C(self.0, -self.1)
    }
    /// i * self
    fn i(self) -> C {
        C(-self.1, self.0)
    }
}

/// In-place inverse FFT of length N (power of two): x[n] = sum_k X[k] e^{+2 pi i k n / N}.
fn ifft(a: &mut [C], tw: &[C]) {
    let n = a.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            a.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let step = n / len;
        for s in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let w = tw[k * step];
                let u = a[s + k];
                let v = a[s + k + len / 2].mul(w);
                a[s + k] = u.add(v);
                a[s + k + len / 2] = C(u.0 - v.0, u.1 - v.1);
            }
        }
        len <<= 1;
    }
}

fn ifft2(a: &mut [C], tw: &[C], col: &mut [C]) {
    for r in 0..N {
        ifft(&mut a[r * N..(r + 1) * N], tw);
    }
    for c in 0..N {
        for r in 0..N {
            col[r] = a[r * N + c];
        }
        ifft(col, tw);
        for r in 0..N {
            a[r * N + c] = col[r];
        }
    }
}

fn f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let m = b & 0x7f_ffff;
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = (m | 0x80_0000) >> (1 - e);
        return sign | ((m + 0x1000) >> 13) as u16;
    }
    if e >= 31 {
        return sign | 0x7bff; // clamp to the largest finite
    }
    let v = ((e as u32) << 10) | ((m + 0x1000) >> 13);
    sign | v.min(0x7bff) as u16
}

/// The wave spectrum of one cascade at t = 0 (h0) and the dispersion (w) per wavevector.
struct Cascade {
    h0: Vec<C>,
    w: Vec<f32>,
    k: Vec<[f32; 2]>,
    foam: Vec<f32>,
    /// share of the texels that start breaking each frame, steered so the foam on the water
    /// (breaking plus the fading trails) covers this cascade's half of the whitecap coverage
    qfrac: f32,
}

fn kvec(i: usize, size: f32) -> f32 {
    let n = if i < N / 2 { i as i32 } else { i as i32 - N as i32 };
    std::f32::consts::TAU / size * n as f32
}

fn build_cascade(c: usize, st: &SeaState) -> Cascade {
    let size = SIZES[c];
    let dk = std::f32::consts::TAU / size;
    // wavenumbers this cascade holds: [lo, hi) (the next cascade starts where this one's are
    // within 6 texels of a wave)
    let lo = if c == 0 { 0.0 } else { std::f32::consts::TAU / SIZES[c - 1] * (N as f32 / 6.0) * 0.5 };
    let hi = if c + 1 == CASCADES { f32::MAX } else { std::f32::consts::TAU / SIZES[c] * (N as f32 / 6.0) * 0.5 };
    let mut rng = Rng(0x9E3779B97F4A7C15 ^ (c as u64 + 1).wrapping_mul(0xD1B54A32D192ED03));
    let wdir = st.dir[1].atan2(st.dir[0]);
    let sdir = st.swell_dir[1].atan2(st.swell_dir[0]);
    // swell: JONSWAP shape at its peak, scaled to the significant height Hs = 4 sqrt(m0)
    let swp = std::f32::consts::TAU / st.swell_t;
    let swell_alpha = {
        let mut m0 = 0.0;
        let mut w = 0.02;
        while w < 6.0 {
            m0 += jonswap_shape(w, 1.0, swp) * 0.01;
            w += 0.01;
        }
        (st.swell_hs / 4.0).powi(2) / m0.max(1e-9)
    };
    let mut h0 = vec![C::default(); N * N];
    let mut w = vec![0.0; N * N];
    let mut k = vec![[0.0; 2]; N * N];
    for j in 0..N {
        for i in 0..N {
            let (kx, kz) = (kvec(i, size), kvec(j, size));
            let kl = (kx * kx + kz * kz).sqrt();
            let id = j * N + i;
            k[id] = [kx, kz];
            let (g1, g2) = rng.gauss();
            if kl < 1e-6 || kl < lo || kl >= hi {
                continue;
            }
            let om = (G * kl).sqrt();
            w[id] = om;
            let th = kz.atan2(kx);
            // wind sea + swell, S(w) D(theta) -> S(k) d2k: dw/dk = g / (2 w), 1 / k for polar
            let s_w = jonswap(om, st.wind.max(0.5), st.fetch) * spread(th - wdir, 8.0) + jonswap_shape(om, swell_alpha, swp) * spread(th - sdir, 40.0);
            // short waves under 3 cm are capillary: left to the roughness
            let damp = (-(kl * 0.03).powi(2)).exp();
            let s_k = s_w * (G / (2.0 * om)) / kl * damp;
            // E|h0|^2 = S dk^2 / 2: with its mirrored term each mode carries S dk^2 of variance
            let amp = 0.5 * (s_k * dk * dk).sqrt();
            h0[id] = C(g1 * amp, g2 * amp);
        }
    }
    Cascade { h0, w, k, foam: vec![0.0; N * N], qfrac: 0.002 }
}

/// The simulation: owns the spectra, steps them to a time and packs the textures.
pub struct Sim {
    state: Option<SeaState>,
    cascades: Vec<Cascade>,
    tw: Vec<C>,
    last_t: f32,
}

impl Sim {
    pub fn new() -> Self {
        let tw = (0..N / 2).map(|k| {
            let a = std::f32::consts::TAU * k as f32 / N as f32;
            C(a.cos(), a.sin())
        });
        Sim { state: None, cascades: Vec::new(), tw: tw.collect(), last_t: 0.0 }
    }

    pub fn step(&mut self, st: SeaState, t: f32) -> Frame {
        // a new sea state: rebuild the spectra (the random phases stay: the sea changes, not jumps)
        let changed = match self.state {
            None => true,
            Some(s) => (s.wind - st.wind).abs() > 0.3 || (s.dir[0] * st.dir[0] + s.dir[1] * st.dir[1]) < 0.995 || (s.swell_hs - st.swell_hs).abs() > 0.1,
        };
        if changed {
            // (the foam on the water and the breaking rate carry over)
            let foam: Vec<Vec<f32>> = self.cascades.iter().map(|c| c.foam.clone()).collect();
            let qf: Vec<f32> = self.cascades.iter().map(|c| c.qfrac).collect();
            self.cascades = (0..CASCADES).map(|c| build_cascade(c, &st)).collect();
            for (c, f) in self.cascades.iter_mut().zip(foam) {
                c.foam = f;
            }
            for (c, q) in self.cascades.iter_mut().zip(qf) {
                c.qfrac = q;
            }
            self.state = Some(st);
        }
        let dt = (t - self.last_t).clamp(0.0, 0.25);
        self.last_t = t;
        let coverage = (3.84e-6 * st.wind.powf(3.41)).min(0.2);
        let cox_munk = 0.003 + 0.00512 * st.wind;
        let mut tex = Vec::with_capacity(CASCADES * 2);
        let mut slope_var = [0.0; CASCADES];
        let mut amplitude = 0.0f32;
        let mut col = vec![C::default(); N];
        for (ci, cas) in self.cascades.iter_mut().enumerate() {
            // fields: (Dx + i h), (Dz + i sx), (sz + i Dxx), (Dzz + i Dxz)
            let mut f = [vec![C::default(); N * N], vec![C::default(); N * N], vec![C::default(); N * N], vec![C::default(); N * N]];
            for j in 0..N {
                for i in 0..N {
                    let id = j * N + i;
                    let om = cas.w[id];
                    if om == 0.0 {
                        continue;
                    }
                    // h(k, t) = h0(k) e^{iwt} + conj(h0(-k)) e^{-iwt}
                    let mi = (N - i) % N;
                    let mj = (N - j) % N;
                    let (c, s) = ((om * t).cos(), (om * t).sin());
                    let h = cas.h0[id].mul(C(c, s)).add(cas.h0[mj * N + mi].conj().mul(C(c, -s)));
                    let [kx, kz] = cas.k[id];
                    let kl = (kx * kx + kz * kz).sqrt();
                    let (ux, uz) = (kx / kl, kz / kl);
                    // Dx = -i kx/k h CHOP (choppy), slopes i k h, Jacobian terms kx kx / k h CHOP
                    let dx = h.i().scale(-ux * CHOP);
                    let dz = h.i().scale(-uz * CHOP);
                    let sx = h.i().scale(kx);
                    let sz = h.i().scale(kz);
                    let dxx = h.scale(kx * ux * CHOP);
                    let dzz = h.scale(kz * uz * CHOP);
                    let dxz = h.scale(kx * uz * CHOP);
                    f[0][id] = dx.add(h.i());
                    f[1][id] = dz.add(sx.i());
                    f[2][id] = sz.add(dxx.i());
                    f[3][id] = dzz.add(dxz.i());
                }
            }
            for a in f.iter_mut() {
                ifft2(a, &self.tw, &mut col);
            }
            // level 0 texels, statistics, foam
            let mut a = vec![[0.0f32; 4]; N * N];
            let mut b = vec![[0.0f32; 4]; N * N];
            let mut jac = vec![0.0f32; N * N];
            let mut var = 0.0f64;
            for id in 0..N * N {
                let (dx, h) = (f[0][id].0, f[0][id].1);
                let (dz, sx) = (f[1][id].0, f[1][id].1);
                let (sz, dxx) = (f[2][id].0, f[2][id].1);
                let (dzz, dxz) = (f[3][id].0, f[3][id].1);
                a[id] = [dx, h, dz, 0.0];
                b[id] = [sx, sz, dxx, dzz];
                jac[id] = (1.0 + dxx) * (1.0 + dzz) - dxz * dxz;
                var += (sx * sx + sz * sz) as f64;
                amplitude = amplitude.max(h.abs());
            }
            slope_var[ci] = (var / (N * N) as f64) as f32;
            // whitecaps: the cascades with breaking waves (wind sea), the lowest Jacobians up to
            // the coverage, held and fading over ~4 s
            // (the wind sea's cascades: the 1 km swell does not break, and its 8 m texels would
            // draw foam as blobs)
            if ci == 1 || ci == 2 {
                // texels below the coverage quantile are foam, fading out by twice the coverage
                let mut sorted = jac.clone();
                let q = ((cas.qfrac * (N * N) as f32) as usize).min(N * N - 1);
                let q2 = (2 * q + 1).min(N * N - 1);
                sorted.select_nth_unstable_by(q2, |x, y| x.partial_cmp(y).unwrap());
                let thr2 = sorted[q2];
                let thr = *sorted[..=q2].select_nth_unstable_by(q, |x, y| x.partial_cmp(y).unwrap()).1;
                let decay = (-dt / 2.5).exp();
                for id in 0..N * N {
                    let new = if coverage > 0.0 { ((thr2 - jac[id]) / (thr2 - thr).max(1e-5)).clamp(0.0, 1.0) } else { 0.0 };
                    cas.foam[id] = (cas.foam[id] * decay).max(new);
                    a[id][3] = cas.foam[id];
                }
                let mean = cas.foam.iter().sum::<f32>() / (N * N) as f32;
                let target = 0.5 * coverage;
                cas.qfrac = (cas.qfrac * (target / mean.max(1e-6)).sqrt().clamp(0.8, 1.25)).clamp(1e-5, 0.2);
            }
            tex.push(pack_mips(&a));
            tex.push(pack_mips(&b));
        }
        let residual = (cox_munk - slope_var.iter().sum::<f32>()).max(0.002);
        Frame { tex, slope_var, residual, amplitude: amplitude * 1.15 + 0.05, coverage, state: st }
    }
}

/// f16 RGBA texels of a level-0 image and its box-filtered mips, levels concatenated.
fn pack_mips(l0: &[[f32; 4]]) -> Vec<u16> {
    let mut out = Vec::with_capacity(N * N * 4 * 4 / 3 + 16);
    let mut cur = l0.to_vec();
    let mut n = N;
    loop {
        for t in &cur {
            out.extend(t.iter().map(|v| f16(*v)));
        }
        if n == 1 {
            break;
        }
        let m = n / 2;
        let mut next = vec![[0.0f32; 4]; m * m];
        for y in 0..m {
            for x in 0..m {
                for c in 0..4 {
                    next[y * m + x][c] = 0.25 * (cur[2 * y * n + 2 * x][c] + cur[2 * y * n + 2 * x + 1][c] + cur[(2 * y + 1) * n + 2 * x][c] + cur[(2 * y + 1) * n + 2 * x + 1][c]);
                }
            }
        }
        cur = next;
        n = m;
    }
    out
}

/// The next request (sea state, time) and the newest finished frame, with a signal for the thread.
type Shared = Arc<(Mutex<(Option<(SeaState, f32)>, Option<Frame>)>, Condvar)>;

/// The simulation on its own thread: the scene posts the sea state each frame and takes the
/// newest finished frame when there is one (never waits).
pub struct Ocean {
    shared: Shared,
}

impl Ocean {
    pub fn spawn() -> Self {
        let shared: Shared = Arc::new((Mutex::new((None, None)), Condvar::new()));
        let s = shared.clone();
        std::thread::Builder::new()
            .name("ocean".into())
            .spawn(move || {
                let mut sim = Sim::new();
                loop {
                    let (st, t) = {
                        let (lock, cv) = &*s;
                        let mut g = lock.lock().unwrap();
                        // wait for a sea state and for the last frame to be taken
                        while g.0.is_none() || g.1.is_some() {
                            g = cv.wait(g).unwrap();
                        }
                        g.0.unwrap()
                    };
                    let frame = sim.step(st, t);
                    let (lock, _) = &*s;
                    lock.lock().unwrap().1 = Some(frame);
                }
            })
            .expect("ocean thread");
        Ocean { shared }
    }

    /// The sea state and the animation time (s) for the next simulation step.
    pub fn post(&self, st: SeaState, t: f32) {
        let (lock, cv) = &*self.shared;
        lock.lock().unwrap().0 = Some((st, t));
        cv.notify_one();
    }

    pub fn take(&self) -> Option<Frame> {
        let (lock, cv) = &*self.shared;
        let f = lock.lock().unwrap().1.take();
        if f.is_some() {
            cv.notify_one();
        }
        f
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ifft_of_single_mode_is_a_wave() {
        let tw: Vec<C> = (0..N / 2).map(|k| C((std::f32::consts::TAU * k as f32 / N as f32).cos(), (std::f32::consts::TAU * k as f32 / N as f32).sin())).collect();
        let mut a = vec![C::default(); N];
        a[1] = C(1.0, 0.0);
        ifft(&mut a, &tw);
        for (n, v) in a.iter().enumerate() {
            let ang = std::f32::consts::TAU * n as f32 / N as f32;
            assert!((v.0 - ang.cos()).abs() < 1e-4 && (v.1 - ang.sin()).abs() < 1e-4);
        }
    }

    #[test]
    fn sea_matches_measured_statistics() {
        let st = SeaState { wind: 10.0, dir: [1.0, 0.0], fetch: 400_000.0, swell_hs: 0.0, swell_t: 10.0, swell_dir: [1.0, 0.0] };
        let f = Sim::new().step(st, 3.0);
        // JONSWAP Hs for 10 m/s over 400 km: ~2-3 m; the largest crest above the mean ~1-3 m
        assert!(f.amplitude > 0.8 && f.amplitude < 5.0, "amplitude {}", f.amplitude);
        // the resolved slopes stay within the Cox-Munk total (0.054 at 10 m/s)
        let resolved: f32 = f.slope_var.iter().sum();
        assert!(resolved > 0.005 && resolved < 0.08, "resolved slope variance {resolved}");
        assert!((f.coverage - 0.0098).abs() < 0.003, "coverage {}", f.coverage);
        assert_eq!(f.tex.len(), CASCADES * 2);
        // foam over about the whitecap coverage (1.5x: the fading edge), not the whole sea
        let mut sim = Sim::new();
        let mut f = sim.step(st, 1.0);
        for k in 0..60 {
            f = sim.step(st, 1.0 + k as f32 * 0.05);
        }
        let n0 = N * N * 4;
        let foam: f32 = (f.tex[2][..n0].chunks(4).chain(f.tex[4][..n0].chunks(4)).map(|t| f16_to_f32(t[3])).sum::<f32>()) / (N * N) as f32;
        // both breaking cascades together: about the coverage (~1% at 10 m/s)
        assert!(foam > 0.004 && foam < 0.02, "foam area {foam}");
        assert_eq!(f.tex[0].len(), (0..MIPS).map(|m| (N >> m) * (N >> m) * 4).sum::<usize>());
    }

    fn f16_to_f32(h: u16) -> f32 {
        let e = ((h >> 10) & 0x1f) as i32;
        let m = (h & 0x3ff) as f32;
        let v = if e == 0 { m / 1024.0 * 2f32.powi(-14) } else { (1.0 + m / 1024.0) * 2f32.powi(e - 15) };
        if h & 0x8000 != 0 { -v } else { v }
    }

    #[test]
    fn f16_round_trip() {
        assert_eq!(f16(1.0), 0x3c00);
        assert_eq!(f16(-2.0), 0xc000);
        assert_eq!(f16(0.0), 0);
    }
}

#[cfg(test)]
mod timing {
    #[test]
    fn report() {
        let st = super::SeaState { wind: 10.0, dir: [1.0, 0.0], fetch: 400_000.0, swell_hs: 1.2, swell_t: 11.0, swell_dir: [0.8, 0.6] };
        let mut sim = super::Sim::new();
        let f = sim.step(st, 1.0);
        let t = std::time::Instant::now();
        for i in 0..20 {
            sim.step(st, 1.0 + i as f32 * 0.016);
        }
        println!("ocean step {:.2} ms; amplitude {:.2} m, slope var {:?}, residual {:.4}, coverage {:.4}", t.elapsed().as_secs_f64() * 50.0, f.amplitude, f.slope_var, f.residual, f.coverage);
    }
}
