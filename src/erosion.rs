//! Large-scale terrain from tectonic uplift and fluvial erosion (stream power law), after
//! Braun & Willett 2013 / Cordonnier et al. 2016 ("Large Scale Terrain Generation from Tectonic
//! Uplift and Fluvial Erosion").
//!
//! On a coarse grid the land is lifted by an uplift field (high where mountain ranges should form,
//! low under rolling hills) and cut down by rivers: dh/dt = U - K A^m S, solved implicitly in
//! downstream order. Every iteration a priority-flood pass from the map edge (sea level) gives each
//! cell a receiver (no closed depressions: all water reaches the sea) and a topological order;
//! drainage area A is accumulated upstream. A little hillslope diffusion rounds the ridges.
//! Branching valleys, ridges and drainage basins emerge from the physics instead of from noise.
//!
//! Rivers are the cells whose drainage area exceeds a threshold. Their receiver chains are traced
//! into polylines, smoothed (Chaikin), given widths from the catchment area (w ~ A^0.45) and water
//! levels that never rise downstream. The fine terrain samples the eroded heightfield bicubically,
//! adds small-scale detail and carves channels and floodplains along the polylines.

use std::collections::BinaryHeap;
use std::cmp::Reverse;

/// Simple value noise / fbm (same family as world.rs, own seeds).
fn hash(ix: i32, iz: i32, seed: u32) -> f32 {
    let mut h = (ix as u32).wrapping_mul(0x8da6_b343) ^ (iz as u32).wrapping_mul(0xd816_3841) ^ seed.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^= h >> 15;
    h as f32 / u32::MAX as f32
}
fn vnoise(x: f32, z: f32, seed: u32) -> f32 {
    let (ix, iz) = (x.floor(), z.floor());
    let (fx, fz) = (x - ix, z - iz);
    let (ux, uz) = (fx * fx * (3.0 - 2.0 * fx), fz * fz * (3.0 - 2.0 * fz));
    let (ix, iz) = (ix as i32, iz as i32);
    let a = hash(ix, iz, seed);
    let b = hash(ix + 1, iz, seed);
    let c = hash(ix, iz + 1, seed);
    let d = hash(ix + 1, iz + 1, seed);
    let ab = a + (b - a) * ux;
    let cd = c + (d - c) * ux;
    ab + (cd - ab) * uz
}
fn fbm(x: f32, z: f32, octaves: u32, seed: u32) -> f32 {
    let (mut sum, mut amp, mut freq, mut norm) = (0.0, 1.0, 1.0, 0.0);
    for o in 0..octaves {
        sum += (vnoise(x * freq, z * freq, seed + o * 101) * 2.0 - 1.0) * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.03;
    }
    sum / norm
}
fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Grid resolution, iterations and the highest summit after rescaling.
const N: usize = 1024;
const ITERS: usize = 160;
const MAX_HEIGHT: f32 = 2300.0;
/// Rivers start at this drainage area (m^2).
const RIVER_AREA: f32 = 12.0e6;
const CACHE_VERSION: u32 = 2;

#[derive(Clone, Copy)]
pub struct RiverPoint {
    pub x: f32,
    pub z: f32,
    /// channel half width (m)
    pub w: f32,
    /// water surface height (m, before the world's curvature / start-area adjustments)
    pub level: f32,
    /// drainage area (m^2)
    pub area: f32,
}

pub struct Erosion {
    pub half: f32,
    cell: f32,
    h: Vec<f32>,
    area: Vec<f32>,
    /// uplift field (0..~1.5): where the mountain ranges are
    uplift: Vec<f32>,
    pub rivers: Vec<Vec<RiverPoint>>,
    seg_grid: Vec<Vec<(u32, u32)>>, // (polyline, segment start index)
    seg_cell: f32,
    seg_n: usize,
    /// rivers start at this drainage area (m^2)
    river_area: f32,
    /// meander the river lines (generated terrain); off for real elevation data
    meander: bool,
}

fn uplift_at(x: f32, z: f32) -> f32 {
    // mountain ranges where a very large-scale field says so; rolling uplands elsewhere
    let ranges = smoothstep(0.0, 0.35, fbm(x / 30_000.0 + 3.1, z / 30_000.0 - 1.7, 3, 61));
    let ridge_dir = 1.0 - (fbm(x / 11_000.0, z / 11_000.0, 3, 67) * 2.0).abs().min(1.0); // elongated ranges
    let hills = 0.5 + 0.5 * fbm(x / 9_000.0, z / 9_000.0, 4, 13);
    0.06 + 0.22 * hills + 1.25 * ranges * (0.45 + 0.55 * ridge_dir)
}

const NB: [(isize, isize); 8] = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)];

/// Priority-flood (Barnes et al. 2014) from the map edge (sea level 0): fills closed depressions
/// (lakes become gently sloping outlets), gives every cell a receiver on its way to the sea and
/// returns the cells in downstream-first order.
fn flood(h: &mut [f32], parent: &mut [u32], order: &mut Vec<u32>) {
    let n = N;
    let mut visited = vec![false; n * n];
    let mut heap: BinaryHeap<Reverse<u64>> = BinaryHeap::with_capacity(n * 8);
    order.clear();
    for c in 0..n * n {
        let (i, j) = (c % n, c / n);
        if i == 0 || j == 0 || i == n - 1 || j == n - 1 {
            h[c] = 0.0;
            visited[c] = true;
            parent[c] = u32::MAX;
            heap.push(Reverse(c as u64));
        }
    }
    while let Some(Reverse(key)) = heap.pop() {
        let c = (key & 0xFFFF_FFFF) as usize;
        let lvl = h[c];
        order.push(c as u32);
        let (ci, cj) = ((c % n) as isize, (c / n) as isize);
        for &(di, dj) in &NB {
            let (ii, jj) = (ci + di, cj + dj);
            if ii < 0 || jj < 0 || ii >= n as isize || jj >= n as isize {
                continue;
            }
            let q = jj as usize * n + ii as usize;
            if visited[q] {
                continue;
            }
            visited[q] = true;
            parent[q] = c as u32;
            let lv = h[q].max(lvl + 1e-3);
            h[q] = lv;
            heap.push(Reverse(((lv.to_bits() as u64) << 32) | q as u64));
        }
    }
}

/// Drainage area: every cell's own area plus everything upstream of it.
fn accumulate(parent: &[u32], order: &[u32], cell_area: f32) -> Vec<f32> {
    let mut area = vec![cell_area; parent.len()];
    for &c in order.iter().rev() {
        let p = parent[c as usize];
        if p != u32::MAX {
            area[p as usize] += area[c as usize];
        }
    }
    area
}

impl Erosion {
    /// Run (or load from assets/cache) the erosion simulation over +-half metres.
    pub fn generate(half: f32) -> Self {
        let t0 = std::time::Instant::now();
        let cell = 2.0 * half / (N - 1) as f32;
        let cache = format!("assets/cache/erosion_v{}_{}_{}.bin", CACHE_VERSION, N, half as u32);
        let h = match std::fs::read(&cache) {
            Ok(bytes) if bytes.len() == N * N * 4 => bytemuck::pod_collect_to_vec(&bytes),
            _ => {
                println!("Erosion: simulating uplift + stream power ({ITERS} steps, cached afterwards)...");
                let h = Self::simulate(half, cell);
                let _ = std::fs::create_dir_all("assets/cache");
                let _ = std::fs::write(&cache, bytemuck::cast_slice(&h));
                h
            }
        };
        // receivers (every cell drains to the map edge) and drainage area of the final landscape
        let mut filled = h.clone();
        let mut parent = vec![u32::MAX; N * N];
        let mut order = Vec::with_capacity(N * N);
        flood(&mut filled, &mut parent, &mut order);
        let area = accumulate(&parent, &order, cell * cell);
        let mut uplift = vec![0.0; N * N];
        for j in 0..N {
            for i in 0..N {
                uplift[j * N + i] = uplift_at(i as f32 * cell - half, j as f32 * cell - half);
            }
        }
        let mut e = Erosion {
            half,
            cell,
            h,
            area,
            uplift,
            rivers: Vec::new(),
            seg_grid: Vec::new(),
            seg_cell: 400.0,
            seg_n: 0,
            river_area: RIVER_AREA,
            meander: true,
        };
        e.extract_rivers(&parent);
        println!(
            "Erosion terrain: {}x{} cells of {:.0} m, {} river polylines ({:.2}s)",
            N,
            N,
            cell,
            e.rivers.len(),
            t0.elapsed().as_secs_f32()
        );
        e
    }

    /// Drainage of a given heightfield (N x N samples over +-half, row-major, z rows): every cell
    /// drains to the map edge (priority flood), flow accumulation, and the river network of the
    /// cells draining more than `river_area` m^2. For real elevation data: the streams follow the
    /// actual valley bottoms.
    pub fn from_heights(half: f32, h: Vec<f32>, river_area: f32) -> Self {
        let t0 = std::time::Instant::now();
        let cell = 2.0 * half / (N - 1) as f32;
        let mut filled = h.clone();
        let mut parent = vec![u32::MAX; N * N];
        let mut order = Vec::with_capacity(N * N);
        flood(&mut filled, &mut parent, &mut order);
        let area = accumulate(&parent, &order, cell * cell);
        let mut e = Erosion {
            half,
            cell,
            h,
            area,
            uplift: vec![0.0; N * N],
            rivers: Vec::new(),
            seg_grid: Vec::new(),
            seg_cell: 200.0,
            seg_n: 0,
            river_area,
            meander: false,
        };
        e.extract_rivers(&parent);
        println!("Drainage: {}x{} cells of {:.0} m, {} streams and rivers ({:.2}s)", N, N, cell, e.rivers.len(), t0.elapsed().as_secs_f32());
        e
    }

    pub const GRID: usize = N;

    fn simulate(half: f32, cell: f32) -> Vec<f32> {
        let n = N;
        let pos = |i: usize| i as f32 * cell - half;
        let mut u = vec![0.0f32; n * n];
        let mut k = vec![0.0f32; n * n];
        let mut h = vec![0.0f32; n * n];
        for j in 0..n {
            for i in 0..n {
                let (x, z) = (pos(i), pos(j));
                let c = j * n + i;
                u[c] = uplift_at(x, z);
                // erodibility: harder and softer rock bands
                k[c] = 0.6 + 0.8 * (0.5 + 0.5 * fbm(x / 6_000.0, z / 6_000.0, 3, 71));
                h[c] = u[c] * 60.0 + 3.0 * hash(i as i32, j as i32, 5);
            }
        }
        let cell_area = cell * cell;
        let (dt, kk, m, kd) = (12.0f32, 0.035f32, 0.45f32, 0.02f32);
        let mut parent = vec![u32::MAX; n * n];
        let mut order: Vec<u32> = Vec::with_capacity(n * n);
        let mut tmp = vec![0.0f32; n * n];
        for it in 0..ITERS {
            // priority flood from the sea (map edge): receivers + downstream-first order
            flood(&mut h, &mut parent, &mut order);
            let area = accumulate(&parent, &order, cell_area);
            // ---- uplift + implicit stream power erosion, downstream first
            for &c in &order {
                let c = c as usize;
                let p = parent[c];
                if p == u32::MAX {
                    continue;
                }
                let p = p as usize;
                let (ci, cj, pi, pj) = (c % n, c / n, p % n, p / n);
                let dist = cell * if ci != pi && cj != pj { std::f32::consts::SQRT_2 } else { 1.0 };
                let f = dt * kk * k[c] * area[c].powf(m) / dist;
                h[c] = ((h[c] + dt * u[c]) + f * h[p]) / (1.0 + f);
            }
            // ---- hillslope diffusion (rounds the ridges)
            tmp.copy_from_slice(&h);
            for j in 1..n - 1 {
                for i in 1..n - 1 {
                    let c = j * n + i;
                    let lap = tmp[c - 1] + tmp[c + 1] + tmp[c - n] + tmp[c + n] - 4.0 * tmp[c];
                    h[c] = tmp[c] + kd * lap;
                }
            }
            if it % 40 == 0 {
                let mx = h.iter().cloned().fold(0.0f32, f32::max);
                println!("  erosion iteration {it}: max height {mx:.0}");
            }
        }
        // rescale to metres
        let mx = h.iter().cloned().fold(1e-6f32, f32::max);
        let s = MAX_HEIGHT / mx;
        h.iter_mut().for_each(|v| *v *= s);
        h
    }

    // cells are indices into several grids at once (area, height, parent, done): index loops
    #[allow(clippy::needless_range_loop)]
    fn extract_rivers(&mut self, parent: &[u32]) {
        let n = N;
        let pos = |i: usize| i as f32 * self.cell - self.half;
        let river_area = self.river_area;
        let is_river = |c: usize| self.area[c] >= river_area;
        // sources: river cells without a river upstream neighbour draining into them
        let mut has_up = vec![false; n * n];
        for c in 0..n * n {
            if is_river(c) && parent[c] != u32::MAX {
                has_up[parent[c] as usize] = true;
            }
        }
        let mut done = vec![false; n * n];
        let mut lines: Vec<Vec<RiverPoint>> = Vec::new();
        // longest rivers first: start from sources sorted by the area they reach is complex;
        // tracing from every source and stopping at visited cells is enough
        for start in 0..n * n {
            if !is_river(start) || has_up[start] {
                continue;
            }
            let mut pts = Vec::new();
            let mut c = start;
            loop {
                let (i, j) = (c % n, c / n);
                let a = self.area[c];
                pts.push(RiverPoint { x: pos(i), z: pos(j), w: (1.2 * (a / 1.0e6).powf(0.45)).clamp(3.0, 45.0), level: self.h[c], area: a });
                if done[c] {
                    break; // joined an existing river (keep the junction point)
                }
                done[c] = true;
                let p = parent[c];
                if p == u32::MAX {
                    break;
                }
                c = p as usize;
            }
            if pts.len() >= 3 {
                lines.push(pts);
            }
        }
        // smooth (Chaikin x3), meander a little, monotone water level downstream
        let meander = self.meander;
        for pts in &mut lines {
            for _ in 0..3 {
                let mut out = vec![pts[0]];
                for k in 0..pts.len() - 1 {
                    let (a, b) = (pts[k], pts[k + 1]);
                    let lerp = |t: f32| RiverPoint {
                        x: a.x + (b.x - a.x) * t,
                        z: a.z + (b.z - a.z) * t,
                        w: a.w + (b.w - a.w) * t,
                        level: a.level + (b.level - a.level) * t,
                        area: a.area + (b.area - a.area) * t,
                    };
                    out.push(lerp(0.25));
                    out.push(lerp(0.75));
                }
                out.push(*pts.last().unwrap());
                *pts = out;
            }
            for k in 1..pts.len() - 1 {
                if !meander {
                    break;
                }
                let (a, b) = (pts[k - 1], pts[k + 1]);
                let (dx, dz) = (b.x - a.x, b.z - a.z);
                let l = (dx * dx + dz * dz).sqrt().max(1e-3);
                // valley-scale wander plus meanders (wavelength ~10-14 channel widths)
                let w = pts[k].w;
                let m = 0.35 * self.cell * fbm(pts[k].x / 900.0, pts[k].z / 900.0, 2, 91) + 2.4 * w * fbm(pts[k].x / (9.0 * w.max(8.0)), pts[k].z / (9.0 * w.max(8.0)), 2, 93);
                pts[k].x += -dz / l * m;
                pts[k].z += dx / l * m;
            }
            for k in 1..pts.len() {
                pts[k].level = pts[k].level.min(pts[k - 1].level);
            }
        }
        // segment grid
        self.seg_n = (2.0 * self.half / self.seg_cell).ceil() as usize + 1;
        self.seg_grid = vec![Vec::new(); self.seg_n * self.seg_n];
        for (li, pts) in lines.iter().enumerate() {
            for k in 0..pts.len() - 1 {
                let (a, b) = (pts[k], pts[k + 1]);
                let reach = a.w.max(b.w) * 5.0 + 120.0;
                let (x0, x1) = (a.x.min(b.x) - reach, a.x.max(b.x) + reach);
                let (z0, z1) = (a.z.min(b.z) - reach, a.z.max(b.z) + reach);
                let to = |v: f32| (((v + self.half) / self.seg_cell).floor().max(0.0) as usize).min(self.seg_n - 1);
                for j in to(z0)..=to(z1) {
                    for i in to(x0)..=to(x1) {
                        self.seg_grid[j * self.seg_n + i].push((li as u32, k as u32));
                    }
                }
            }
        }
        self.rivers = lines;
    }

    fn idx(&self, x: f32, z: f32) -> (usize, usize, f32, f32) {
        let fx = ((x + self.half) / self.cell).clamp(0.0, (N - 1) as f32 - 1e-3);
        let fz = ((z + self.half) / self.cell).clamp(0.0, (N - 1) as f32 - 1e-3);
        let (i, j) = (fx.floor() as usize, fz.floor() as usize);
        (i, j, fx - i as f32, fz - j as f32)
    }

    /// Bicubic (Catmull-Rom) height of the eroded terrain.
    pub fn height(&self, x: f32, z: f32) -> f32 {
        let (i, j, u, v) = self.idx(x, z);
        let at = |di: isize, dj: isize| {
            let ii = (i as isize + di).clamp(0, N as isize - 1) as usize;
            let jj = (j as isize + dj).clamp(0, N as isize - 1) as usize;
            self.h[jj * N + ii]
        };
        let cr = |p0: f32, p1: f32, p2: f32, p3: f32, t: f32| {
            p1 + 0.5 * t * (p2 - p0 + t * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + t * (3.0 * (p1 - p2) + p3 - p0)))
        };
        let row = |dj: isize| cr(at(-1, dj), at(0, dj), at(1, dj), at(2, dj), u);
        cr(row(-1), row(0), row(1), row(2), v)
    }

    fn bilinear(&self, f: &[f32], x: f32, z: f32) -> f32 {
        let (i, j, u, v) = self.idx(x, z);
        let a = f[j * N + i];
        let b = f[j * N + i + 1];
        let c = f[(j + 1) * N + i];
        let d = f[(j + 1) * N + i + 1];
        (a + (b - a) * u) + ((c + (d - c) * u) - (a + (b - a) * u)) * v
    }

    /// Topographic wetness index, mapped to 0..1 (wet valley bottoms ~1, ridges ~0).
    pub fn wetness(&self, x: f32, z: f32) -> f32 {
        let a = self.bilinear(&self.area, x, z).max(1.0);
        let e = self.cell;
        let gx = (self.height(x + e, z) - self.height(x - e, z)) / (2.0 * e);
        let gz = (self.height(x, z + e) - self.height(x, z - e)) / (2.0 * e);
        let tan_b = (gx * gx + gz * gz).sqrt().max(0.005);
        let twi = (a / self.cell / tan_b).ln();
        smoothstep(7.0, 15.0, twi)
    }

    /// Uplift (mountain-ness) 0..~1.5.
    pub fn uplift(&self, x: f32, z: f32) -> f32 {
        self.bilinear(&self.uplift, x, z)
    }

    /// Nearest river channel: (distance to the centre line, half width, water level, drainage area).
    pub fn river(&self, x: f32, z: f32) -> Option<(f32, f32, f32, f32)> {
        let i = (((x + self.half) / self.seg_cell) as isize).clamp(0, self.seg_n as isize - 1) as usize;
        let j = (((z + self.half) / self.seg_cell) as isize).clamp(0, self.seg_n as isize - 1) as usize;
        let mut best: Option<(f32, f32, f32, f32, f32)> = None; // (d - w, d, w, level, area)
        for &(li, k) in &self.seg_grid[j * self.seg_n + i] {
            let pts = &self.rivers[li as usize];
            let (a, b) = (pts[k as usize], pts[k as usize + 1]);
            let (dx, dz) = (b.x - a.x, b.z - a.z);
            let l2 = (dx * dx + dz * dz).max(1e-6);
            let t = (((x - a.x) * dx + (z - a.z) * dz) / l2).clamp(0.0, 1.0);
            let (px, pz) = (a.x + dx * t, a.z + dz * t);
            let d = ((x - px).powi(2) + (z - pz).powi(2)).sqrt();
            let w = a.w + (b.w - a.w) * t;
            let lv = a.level + (b.level - a.level) * t;
            let ar = a.area + (b.area - a.area) * t;
            if best.is_none_or(|bb| d - w < bb.0) {
                best = Some((d - w, d, w, lv, ar));
            }
        }
        best.map(|b| (b.1, b.2, b.3, b.4))
    }

    /// A start point: on the bank of a medium river in low (hilly, not mountainous) country near
    /// the middle of the map. Returns (x, z) in simulation coordinates.
    pub fn start_point(&self) -> (f32, f32) {
        let mut best = (f32::MAX, (0.0, 0.0));
        for pts in &self.rivers {
            for k in 1..pts.len() - 1 {
                let p = pts[k];
                let r = (p.x * p.x + p.z * p.z).sqrt();
                if r > 25_000.0 || p.area < 60.0e6 || p.area > 900.0e6 {
                    continue;
                }
                let up = self.uplift(p.x, p.z);
                let score = up * 4.0 + r / 25_000.0 + (p.level / 400.0);
                if score < best.0 {
                    let (a, b) = (pts[k - 1], pts[k + 1]);
                    let (dx, dz) = (b.x - a.x, b.z - a.z);
                    let l = (dx * dx + dz * dz).sqrt().max(1e-3);
                    // 220 m from the river centre, on its left bank
                    let off = p.w + 200.0;
                    best = (score, (p.x - dz / l * off, p.z + dx / l * off));
                }
            }
        }
        best.1
    }
}
