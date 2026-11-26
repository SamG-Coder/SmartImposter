//! Procedural buildings: a set of building archetypes (glass office towers, residential towers
//! with balconies, office blocks, stucco apartment blocks, low-rise shops, houses, parking
//! garages) built as real geometry: floor slabs, mullion fins, recessed windows, balconies with
//! glass rails, parapets, rooftop plant. Each archetype is one variant of the plant system, so it
//! is baked into an impostor atlas like a tree and drawn as an impostor far away, as its mesh
//! near the camera (renderer.rs, globe/city.rs places them).
//!
//! Triangle materials (tri.rchit, meshes flagged MESH_BUILDING): low 4 bits the material, high 4
//! bits an entry of its palette. Texture coordinates are facade cells: x the window column, y the
//! storey, so the shader can tell the windows apart (blinds, lights, tint) in the bake and at
//! runtime alike.

use crate::asset::{MeshData, TreeAsset, TreePack, TreePart};
use glam::{Affine3A, Vec2, Vec3};

pub const WALL: u8 = 0;
pub const GLASS: u8 = 1;
pub const FRAME: u8 = 2;
pub const SLAB: u8 = 3;
pub const ROOF: u8 = 4;
pub const TILE: u8 = 5;
pub const EQUIP: u8 = 6;
pub const RAIL: u8 = 7;
pub const GARAGE: u8 = 8;
pub const DARK: u8 = 9;
pub const SPANDREL: u8 = 10;
pub const WOOD: u8 = 11;
pub const PAINT: u8 = 12;
pub const POOL: u8 = 13;

/// What an archetype is: the city plan (globe/city.rs) picks by it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Class {
    Tower,
    MidRise,
    LowRise,
    House,
    /// boats (moored at docks and out on the water) and docks: not for lots
    Boat,
    Dock,
}

pub struct Arch {
    /// height (m), with the crown
    pub height: f32,
}

struct B {
    pos: Vec<f32>,
    nrm: Vec<f32>,
    uv: Vec<f32>,
    idx: Vec<u32>,
    mat: Vec<u8>,
}

fn m(mat: u8, pal: u8) -> u8 {
    mat | pal << 4
}

impl B {
    fn new() -> Self {
        B { pos: Vec::new(), nrm: Vec::new(), uv: Vec::new(), idx: Vec::new(), mat: Vec::new() }
    }
    fn vert(&mut self, p: Vec3, n: Vec3, uv: Vec2) -> u32 {
        let i = (self.pos.len() / 3) as u32;
        self.pos.extend_from_slice(&p.to_array());
        self.nrm.extend_from_slice(&n.to_array());
        self.uv.extend_from_slice(&uv.to_array());
        i
    }
    /// A quad a b c d (in order around it), facing `n`, with cell coordinates per corner.
    fn quad_uv(&mut self, q: [Vec3; 4], uv: [Vec2; 4], mat: u8) {
        let n = (q[1] - q[0]).cross(q[3] - q[0]).normalize_or_zero();
        let i: Vec<u32> = (0..4).map(|k| self.vert(q[k], n, uv[k])).collect();
        self.idx.extend_from_slice(&[i[0], i[1], i[2], i[0], i[2], i[3]]);
        self.mat.extend_from_slice(&[mat, mat]);
    }
    fn quad(&mut self, q: [Vec3; 4], mat: u8) {
        self.quad_uv(q, [Vec2::ZERO; 4], mat);
    }
    /// A vertical quad on the line a -> b (xz), from y0 to y1, `out` metres along the outward
    /// normal; cells: columns of `col` m from `u0`, storeys of `fh` m from `yb`.
    fn wall(&mut self, a: Vec2, b: Vec2, y0: f32, y1: f32, out: f32, mat: u8, cell: Option<(f32, f32, f32, f32)>) {
        let o = outward(a, b) * out;
        let (a, b) = (a + o, b + o);
        let p = [Vec3::new(a.x, y0, a.y), Vec3::new(b.x, y0, b.y), Vec3::new(b.x, y1, b.y), Vec3::new(a.x, y1, a.y)];
        match cell {
            Some((u0, col, yb, fh)) => {
                let len = (b - a).length();
                let (ua, ub) = (u0 / col, (u0 + len) / col);
                let (va, vb) = ((y0 - yb) / fh, (y1 - yb) / fh);
                self.quad_uv(p, [Vec2::new(ua, va), Vec2::new(ub, va), Vec2::new(ub, vb), Vec2::new(ua, vb)], mat);
            }
            None => self.quad(p, mat),
        }
    }
    /// A horizontal convex polygon at height y (fan), facing up or down.
    fn cap(&mut self, poly: &[Vec2], y: f32, up: bool, mat: u8) {
        let n = if up { Vec3::Y } else { -Vec3::Y };
        let base = (self.pos.len() / 3) as u32;
        for p in poly {
            self.vert(Vec3::new(p.x, y, p.y), n, *p * 0.1);
        }
        for k in 1..poly.len() as u32 - 1 {
            if up {
                self.idx.extend_from_slice(&[base, base + k + 1, base + k]);
            } else {
                self.idx.extend_from_slice(&[base, base + k, base + k + 1]);
            }
            self.mat.push(mat);
        }
    }
    /// An axis-aligned box (all six sides; the bottom only if `bottom`).
    fn bx(&mut self, lo: Vec3, hi: Vec3, mat: u8, bottom: bool) {
        let poly = [Vec2::new(lo.x, lo.z), Vec2::new(hi.x, lo.z), Vec2::new(hi.x, hi.z), Vec2::new(lo.x, hi.z)];
        self.prism(&poly, lo.y, hi.y, mat, mat, bottom);
    }
    /// A vertical prism over a convex counter-clockwise polygon: sides, top, optional bottom.
    fn prism(&mut self, poly: &[Vec2], y0: f32, y1: f32, side: u8, top: u8, bottom: bool) {
        for k in 0..poly.len() {
            self.wall(poly[k], poly[(k + 1) % poly.len()], y0, y1, 0.0, side, None);
        }
        self.cap(poly, y1, true, top);
        if bottom {
            self.cap(poly, y0, false, top);
        }
    }
    /// A box spanning the segment a -> b on the outside of a wall: `depth` out, `w` wide
    /// (centred on the segment), y0..y1. Used for fins and bands.
    fn strip(&mut self, a: Vec2, b: Vec2, y0: f32, y1: f32, out0: f32, out1: f32, mat: u8) {
        let o = outward(a, b);
        let (a0, b0, a1, b1) = (a + o * out0, b + o * out0, a + o * out1, b + o * out1);
        let v = |p: Vec2, y: f32| Vec3::new(p.x, y, p.y);
        // front, top, bottom, two ends
        self.quad([v(a1, y0), v(b1, y0), v(b1, y1), v(a1, y1)], mat);
        self.quad([v(a0, y1), v(a1, y1), v(b1, y1), v(b0, y1)], mat);
        self.quad([v(a0, y0), v(b0, y0), v(b1, y0), v(a1, y0)], mat);
        self.quad([v(a0, y0), v(a1, y0), v(a1, y1), v(a0, y1)], mat);
        self.quad([v(b1, y0), v(b0, y0), v(b0, y1), v(b1, y1)], mat);
    }
    fn mesh(self, name: &str) -> MeshData {
        let mut materials = self.mat;
        materials.resize(materials.len().div_ceil(4) * 4, 0);
        MeshData { name: name.into(), positions: self.pos, normals: self.nrm, uvs: self.uv, indices: self.idx, materials, grid: 0 }
    }
}

/// Outward normal (xz) of a counter-clockwise polygon's edge a -> b (y up: x east, z south).
fn outward(a: Vec2, b: Vec2) -> Vec2 {
    let d = (b - a).normalize_or_zero();
    Vec2::new(d.y, -d.x)
}

/// A rectangle w x d (x, z) around the origin with its corners cut by `chamfer`, or rounded with
/// `round` segments per corner; counter-clockwise seen from above (y up).
fn footprint(w: f32, d: f32, chamfer: f32, round: usize) -> Vec<Vec2> {
    let (hx, hz) = (w * 0.5, d * 0.5);
    if round > 0 {
        let r = chamfer.min(hx).min(hz);
        let mut out = Vec::new();
        for (cx, cz, a0) in [(hx - r, hz - r, 0.0f32), (hx - r, -hz + r, -90.0), (-hx + r, -hz + r, -180.0), (-hx + r, hz - r, -270.0)] {
            for k in 0..=round {
                let a = (a0 - 90.0 * k as f32 / round as f32).to_radians();
                out.push(Vec2::new(cx + r * a.cos(), cz + r * a.sin()));
            }
        }
        return ccw(out);
    }
    let c = chamfer.min(hx * 0.5).min(hz * 0.5);
    let pts = if c > 0.0 {
        vec![
            Vec2::new(-hx + c, -hz),
            Vec2::new(hx - c, -hz),
            Vec2::new(hx, -hz + c),
            Vec2::new(hx, hz - c),
            Vec2::new(hx - c, hz),
            Vec2::new(-hx + c, hz),
            Vec2::new(-hx, hz - c),
            Vec2::new(-hx, -hz + c),
        ]
    } else {
        vec![Vec2::new(-hx, -hz), Vec2::new(hx, -hz), Vec2::new(hx, hz), Vec2::new(-hx, hz)]
    };
    ccw(pts)
}

/// Put a polygon in the order `outward` expects (edges with the inside on their left seen from
/// above, y up: positive area by the shoelace formula in x, z).
fn ccw(mut p: Vec<Vec2>) -> Vec<Vec2> {
    let a: f32 = (0..p.len()).map(|k| p[k].x * p[(k + 1) % p.len()].y - p[(k + 1) % p.len()].x * p[k].y).sum();
    if a < 0.0 {
        p.reverse();
    }
    p
}

/// The polygon moved inwards by `d` (convex: every edge moved along its normal, corners kept on
/// the bisectors).
fn inset(p: &[Vec2], d: f32) -> Vec<Vec2> {
    let n = p.len();
    (0..n)
        .map(|k| {
            let (a, b, c) = (p[(k + n - 1) % n], p[k], p[(k + 1) % n]);
            let (n0, n1) = (outward(a, b), outward(b, c));
            let bis = (n0 + n1).normalize_or_zero();
            let cos = bis.dot(n0).max(0.3);
            b - bis * (d / cos)
        })
        .collect()
}

fn perimeter_u(p: &[Vec2], k: usize) -> f32 {
    (0..k).map(|i| (p[(i + 1) % p.len()] - p[i]).length()).sum()
}

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }
    fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.f()
    }
    fn pick(&mut self, n: usize) -> usize {
        ((self.f() * n as f32) as usize).min(n - 1)
    }
}

/// How a volume's walls are dressed.
#[derive(Clone, Copy)]
enum Fac {
    /// floor-to-ceiling glass, mullion fins every `col` m standing `fin` m out, a band at every
    /// floor
    Curtain { col: f32, fin: f32 },
    /// glass walls set back `depth` behind slab edges, glass rails at every floor
    Balcony { depth: f32 },
    /// horizontal bands: glass ribbon `win` of the storey, recessed, spandrel between
    Ribbon { win: f32, col: f32 },
    /// a solid wall with recessed windows of w x h every `col` m
    Punched { col: f32, w: f32, h: f32 },
    /// open parking decks
    Garage,
}

struct Pal {
    wall: u8,
    glass: u8,
    frame: u8,
}

/// Dress one prism volume: `poly` from y0 to y1 in storeys of `fh` (a taller first storey of
/// `fh0` when y0 is 0: a glazed lobby or shop fronts), its roof at y1 with a parapet (`roof`).
fn volume(b: &mut B, poly: &[Vec2], y0: f32, y1: f32, fh0: f32, fh: f32, fac: Fac, pal: &Pal, roof: bool, rng: &mut Rng) {
    let n = poly.len();
    let lobby = y0 == 0.0 && fh0 > 0.0;
    let ys = if lobby { y0 + fh0 } else { y0 };
    if lobby {
        // shop fronts: glass set back under a canopy band
        for k in 0..n {
            let (a, c) = (poly[k], poly[(k + 1) % n]);
            let u0 = perimeter_u(poly, k);
            b.wall(a, c, 0.0, fh0 - 0.6, -0.4, m(GLASS, pal.glass), Some((u0, 3.0, 0.0, fh0)));
            b.wall(a, c, fh0 - 0.6, fh0, 0.0, m(WALL, pal.wall), None);
            b.strip(a, c, fh0 - 0.35, fh0 - 0.15, 0.0, 1.6, m(FRAME, pal.frame)); // canopy
            // the corners of the recess
            b.strip(a, a + (c - a).normalize_or_zero() * 0.5, 0.0, fh0 - 0.6, -0.4, 0.0, m(WALL, pal.wall));
        }
    }
    let floors = ((y1 - ys) / fh).round().max(1.0) as usize;
    let fh = (y1 - ys) / floors as f32;
    match fac {
        Fac::Curtain { col, fin } => {
            for k in 0..n {
                let (a, c) = (poly[k], poly[(k + 1) % n]);
                let u0 = perimeter_u(poly, k);
                b.wall(a, c, ys, y1, 0.0, m(GLASS, pal.glass), Some((u0, col, ys, fh)));
                let len = (c - a).length();
                let cols = (len / col).round().max(1.0) as usize;
                let dir = (c - a) / len;
                if fin > 0.0 {
                    for j in 1..cols {
                        let p = a + dir * (len * j as f32 / cols as f32);
                        b.strip(p - dir * 0.06, p + dir * 0.06, ys, y1, 0.0, fin, m(FRAME, pal.frame));
                    }
                }
                for f in 0..=floors {
                    let y = ys + f as f32 * fh;
                    b.strip(a, c, y - 0.12, y + 0.12, 0.0, 0.08 + fin * 0.3, m(FRAME, pal.frame));
                }
            }
        }
        Fac::Balcony { depth } => {
            let core = inset(poly, depth);
            for k in 0..n {
                let (a, c) = (core[k], core[(k + 1) % n]);
                let u0 = perimeter_u(&core, k);
                b.wall(a, c, ys, y1, 0.0, m(GLASS, pal.glass), Some((u0, 2.4, ys, fh)));
            }
            for f in 0..floors {
                let y = ys + f as f32 * fh;
                // the slab (a plate over the whole footprint: its edge and soffit show)
                for k in 0..n {
                    b.wall(poly[k], poly[(k + 1) % n], y - 0.3, y, 0.0, m(SLAB, 0), None);
                    // glass rail on the slab edge, with a top rail
                    let u0 = perimeter_u(poly, k);
                    b.wall(poly[k], poly[(k + 1) % n], y, y + 1.05, -0.05, m(RAIL, pal.glass), Some((u0, 1.5, y, 1.05)));
                    b.strip(poly[k], poly[(k + 1) % n], y + 1.02, y + 1.1, -0.1, 0.0, m(FRAME, pal.frame));
                }
                b.cap(poly, y - 0.3, false, m(SLAB, 0));
                let ring: Vec<Vec2> = poly.to_vec();
                // plate top between the rail and the glass: an annulus as quads
                for k in 0..n {
                    let (a0, a1, c0, c1) = (ring[k], ring[(k + 1) % n], core[k], core[(k + 1) % n]);
                    b.quad([Vec3::new(a0.x, y, a0.y), Vec3::new(c0.x, y, c0.y), Vec3::new(c1.x, y, c1.y), Vec3::new(a1.x, y, a1.y)], m(SLAB, 0));
                }
                // balcony dividers every ~9 m
                for k in 0..n {
                    let (a, c) = (poly[k], poly[(k + 1) % n]);
                    let len = (c - a).length();
                    let segs = (len / 9.0).round().max(1.0) as usize;
                    for j in 1..segs {
                        let t = j as f32 / segs as f32;
                        let p = a + (c - a) * t;
                        let q = core[k] + (core[(k + 1) % n] - core[k]) * t;
                        let w = (c - a).normalize_or_zero() * 0.12;
                        let v = |p: Vec2, y: f32| Vec3::new(p.x, y, p.y);
                        b.quad([v(q - w, y), v(p - w, y), v(p - w, y + fh - 0.3), v(q - w, y + fh - 0.3)], m(SLAB, 0));
                        b.quad([v(p + w, y), v(q + w, y), v(q + w, y + fh - 0.3), v(p + w, y + fh - 0.3)], m(SLAB, 0));
                    }
                }
            }
        }
        Fac::Ribbon { win, col } => {
            for k in 0..n {
                let (a, c) = (poly[k], poly[(k + 1) % n]);
                let u0 = perimeter_u(poly, k);
                for f in 0..floors {
                    let y = ys + f as f32 * fh;
                    let (w0, w1) = (y + fh * (1.0 - win) * 0.55, y + fh - fh * (1.0 - win) * 0.45);
                    b.wall(a, c, y, w0, 0.0, m(SPANDREL, pal.wall), Some((u0, col, y, fh)));
                    b.wall(a, c, w1, y + fh, 0.0, m(SPANDREL, pal.wall), Some((u0, col, y, fh)));
                    b.wall(a, c, w0, w1, -0.25, m(GLASS, pal.glass), Some((u0, col, ys, fh)));
                    // reveals: sill and head of the ribbon
                    let o = outward(a, c);
                    let v = |p: Vec2, y: f32| Vec3::new(p.x, y, p.y);
                    b.quad([v(a - o * 0.25, w0), v(c - o * 0.25, w0), v(c, w0), v(a, w0)], m(SPANDREL, pal.wall));
                    b.quad([v(a, w1), v(c, w1), v(c - o * 0.25, w1), v(a - o * 0.25, w1)], m(SPANDREL, pal.wall));
                }
            }
        }
        Fac::Punched { col, w, h } => {
            for k in 0..n {
                let (a, c) = (poly[k], poly[(k + 1) % n]);
                let u0 = perimeter_u(poly, k);
                let len = (c - a).length();
                let cols = (len / col).floor().max(1.0) as usize;
                let dir = (c - a) / len;
                let o = outward(a, c);
                let step = len / cols as f32;
                let ww = w.min(step * 0.7);
                for f in 0..floors {
                    let y = ys + f as f32 * fh;
                    let (s0, s1) = (y + (fh - h) * 0.55, y + (fh - h) * 0.55 + h);
                    // below the sill and above the head: whole strips
                    b.wall(a, c, y, s0, 0.0, m(WALL, pal.wall), None);
                    b.wall(a, c, s1, y + fh, 0.0, m(WALL, pal.wall), None);
                    for j in 0..cols {
                        let mid = (j as f32 + 0.5) * step;
                        let (p0, p1) = (a + dir * (mid - ww * 0.5), a + dir * (mid + ww * 0.5));
                        let (e0, e1) = (a + dir * (j as f32 * step), a + dir * ((j + 1) as f32 * step));
                        // piers either side
                        b.wall(e0, p0, s0, s1, 0.0, m(WALL, pal.wall), None);
                        b.wall(p1, e1, s0, s1, 0.0, m(WALL, pal.wall), None);
                        // the window, recessed, and its reveals
                        let r = 0.2;
                        // (cells: this window is column (u0 / step) + j, storey f)
                        let cu = (u0 / step).floor() + j as f32;
                        b.wall(p0, p1, s0, s1, -r, m(GLASS, pal.glass), Some((cu * ww, ww, s0 - f as f32 * h, h)));
                        let v = |p: Vec2, y: f32| Vec3::new(p.x, y, p.y);
                        let (q0, q1) = (p0 - o * r, p1 - o * r);
                        b.quad([v(q0, s0), v(q1, s0), v(p1, s0), v(p0, s0)], m(FRAME, pal.frame)); // sill
                        b.quad([v(p0, s1), v(p1, s1), v(q1, s1), v(q0, s1)], m(WALL, pal.wall));
                        b.quad([v(p0, s0), v(p0, s1), v(q0, s1), v(q0, s0)], m(WALL, pal.wall));
                        b.quad([v(q1, s0), v(q1, s1), v(p1, s1), v(p1, s0)], m(WALL, pal.wall));
                    }
                }
            }
        }
        Fac::Garage => {
            for k in 0..n {
                let (a, c) = (poly[k], poly[(k + 1) % n]);
                for f in 0..floors {
                    let y = ys + f as f32 * fh;
                    b.wall(a, c, y, y + 1.1, 0.0, m(GARAGE, pal.wall), None);
                    b.wall(a, c, y + 1.1, y + fh, -2.0, m(DARK, 0), None);
                }
                // columns
                let len = (c - a).length();
                let dir = (c - a) / len;
                let cols = (len / 8.5).round().max(1.0) as usize;
                for j in 0..=cols {
                    let p = a + dir * (len * j as f32 / cols as f32);
                    b.strip(p - dir * 0.3, p + dir * 0.3, ys, y1, -0.6, 0.0, m(GARAGE, pal.wall));
                }
            }
        }
    }
    if roof {
        b.cap(poly, y1, true, m(ROOF, 0));
        // parapet
        for k in 0..n {
            b.strip(poly[k], poly[(k + 1) % n], y1, y1 + 1.0, -0.25, 0.0, m(if matches!(fac, Fac::Curtain { .. }) { FRAME } else { WALL }, if matches!(fac, Fac::Curtain { .. }) { pal.frame } else { pal.wall }));
        }
        // rooftop plant: a few boxes inside
        let core = inset(poly, 3.0);
        let (lo, hi) = core.iter().fold((Vec2::splat(f32::MAX), Vec2::splat(f32::MIN)), |(l, h), p| (l.min(*p), h.max(*p)));
        if hi.x - lo.x > 4.0 && hi.y - lo.y > 4.0 {
            for _ in 0..(2 + rng.pick(4)) {
                let (sx, sz) = (rng.range(1.5, ((hi.x - lo.x) * 0.35).max(1.6)), rng.range(1.5, ((hi.y - lo.y) * 0.35).max(1.6)));
                let cx = rng.range(lo.x + sx * 0.5, hi.x - sx * 0.5);
                let cz = rng.range(lo.y + sz * 0.5, hi.y - sz * 0.5);
                let h = rng.range(1.2, 3.2);
                b.bx(Vec3::new(cx - sx * 0.5, y1, cz - sz * 0.5), Vec3::new(cx + sx * 0.5, y1 + h, cz + sz * 0.5), m(EQUIP, rng.pick(3) as u8), false);
            }
        }
    }
}

/// A house: stucco walls with windows, a tiled hip roof (or flat with a parapet).
fn house(b: &mut B, w: f32, d: f32, storeys: usize, hip: bool, pal: &Pal, rng: &mut Rng) -> f32 {
    let fh = 3.1;
    let poly = footprint(w, d, 0.0, 0);
    let h = storeys as f32 * fh + 0.4;
    volume(b, &poly, 0.0, h, 0.0, fh, Fac::Punched { col: 3.4, w: 1.4, h: 1.6 }, pal, !hip, rng);
    if hip {
        // hip roof with eaves
        let (hx, hz) = (w * 0.5 + 0.6, d * 0.5 + 0.6);
        let rise = 0.45 * hx.min(hz);
        let along_x = w >= d;
        let r = if along_x { (hx - hz).max(0.0) } else { (hz - hx).max(0.0) };
        let (r0, r1) = if along_x { (Vec3::new(-r, h + rise, 0.0), Vec3::new(r, h + rise, 0.0)) } else { (Vec3::new(0.0, h + rise, -r), Vec3::new(0.0, h + rise, r)) };
        let c = [Vec3::new(-hx, h, -hz), Vec3::new(hx, h, -hz), Vec3::new(hx, h, hz), Vec3::new(-hx, h, hz)];
        let tm = m(TILE, rng.pick(3) as u8);
        if along_x {
            b.quad([c[0], r0, r1, c[1]], tm); // north slope
            b.quad([c[2], r1, r0, c[3]], tm);
            b.quad([c[1], r1, r1, c[2]], tm);
            b.quad([c[3], r0, r0, c[0]], tm);
        } else {
            b.quad([c[1], r0, r1, c[2]], tm);
            b.quad([c[3], r1, r0, c[0]], tm);
            b.quad([c[0], r0, r0, c[1]], tm);
            b.quad([c[2], r1, r1, c[3]], tm);
        }
        // soffit
        b.cap(&ccw(c.iter().map(|p| Vec2::new(p.x, p.z)).collect()), h, false, m(WALL, pal.wall));
        h + rise
    } else {
        h + 1.0
    }
}

/// All archetypes, deterministic. About 40 of them (the variant slots are shared with the
/// plants: rt.glsl keeps a variant in 8 bits).
pub fn archetypes() -> (Vec<MeshData>, Vec<Arch>) {
    let mut meshes = Vec::new();
    let mut archs = Vec::new();
    let mut rng = Rng(0x5EED_C17E);
    let push = |b: B, name: String, height: f32, meshes: &mut Vec<MeshData>, archs: &mut Vec<Arch>| {
        meshes.push(b.mesh(&name));
        archs.push(Arch { height });
    };

    // glass office towers: podium, shaft (some with setbacks), crown
    for k in 0..9 {
        let mut b = B::new();
        let pal = Pal { wall: [0, 2, 8][k % 3], glass: [0, 1, 2, 3, 4, 5][k % 6], frame: [0, 1, 2, 3][k % 4] };
        let h = [120.0, 150.0, 185.0, 215.0, 240.0, 105.0, 165.0, 200.0, 135.0][k];
        let (w, d) = ([38.0, 42.0, 36.0, 44.0, 40.0, 34.0, 46.0, 38.0, 40.0][k], [34.0, 36.0, 36.0, 38.0, 40.0, 30.0, 34.0, 38.0, 32.0][k]);
        let (chamfer, round) = [(0.0, 0), (6.0, 0), (10.0, 4), (0.0, 0), (8.0, 0), (12.0, 6), (0.0, 0), (9.0, 3), (5.0, 0)][k];
        let pod_h = [0.0, 22.0, 0.0, 26.0, 18.0, 0.0, 30.0, 0.0, 20.0][k];
        let col = [1.5, 1.6, 1.4, 1.5, 3.0, 1.5, 1.8, 1.5, 2.0][k];
        let fin = [0.35, 0.15, 0.0, 0.5, 0.25, 0.0, 0.4, 0.2, 0.3][k];
        let shaft = footprint(w, d, chamfer, round);
        let mut y0 = 0.0;
        if pod_h > 0.0 {
            let pod = footprint(w + 16.0, d + 14.0, 0.0, 0);
            volume(&mut b, &pod, 0.0, pod_h, 5.5, 3.1, Fac::Garage, &pal, true, &mut rng);
            y0 = pod_h;
        }
        // setbacks: the top part narrower on some
        let tiers = if k % 3 == 1 { 2 } else { 1 };
        let mut poly = shaft.clone();
        let mut lo = y0;
        for t in 0..tiers {
            let top = if t + 1 == tiers { h } else { y0 + (h - y0) * 0.72 };
            volume(&mut b, &poly, lo, top, if lo == 0.0 { 6.0 } else { 0.0 }, 4.0, Fac::Curtain { col, fin }, &pal, true, &mut rng);
            lo = top;
            poly = inset(&poly, 4.0);
        }
        // crown: mechanical floors behind fins, on some a lit frame or a mast
        let crown = inset(&shaft, if tiers > 1 { 6.0 } else { 3.0 });
        volume(&mut b, &crown, h, h + 7.0, 0.0, 3.5, Fac::Curtain { col: 1.2, fin: 0.6 }, &Pal { wall: pal.wall, glass: 6, frame: pal.frame }, true, &mut rng);
        let mut top = h + 8.0;
        if k % 4 == 2 {
            b.bx(Vec3::new(-0.6, top, -0.6), Vec3::new(0.6, top + 30.0, 0.6), m(FRAME, 1), false);
            top += 30.0;
        }
        push(b, format!("Building_Tower_{k}"), top, &mut meshes, &mut archs);
    }
    // residential towers: white slab bands, glass rails, blue-green glass (the Miami look)
    for k in 0..9 {
        let mut b = B::new();
        let pal = Pal { wall: [0, 1, 0, 3, 0, 7, 0, 1, 0][k], glass: [0, 2, 1, 5, 0, 3, 7, 1, 2][k], frame: [2, 2, 1, 0, 2, 1, 2, 2, 0][k] };
        let h = [95.0, 140.0, 180.0, 70.0, 210.0, 120.0, 160.0, 85.0, 190.0][k];
        let (w, d) = ([44.0, 52.0, 40.0, 48.0, 42.0, 56.0, 38.0, 50.0, 46.0][k], [24.0, 26.0, 30.0, 22.0, 34.0, 24.0, 38.0, 26.0, 28.0][k]);
        let (chamfer, round) = [(0.0, 0), (8.0, 4), (0.0, 0), (6.0, 0), (12.0, 5), (0.0, 0), (10.0, 4), (0.0, 0), (9.0, 0)][k];
        let pod_h = [16.0, 20.0, 0.0, 14.0, 24.0, 18.0, 0.0, 12.0, 22.0][k];
        let mut y0 = 0.0;
        if pod_h > 0.0 {
            let pod = footprint(w + 14.0, d + 18.0, 0.0, 0);
            volume(&mut b, &pod, 0.0, pod_h, 5.0, 3.0, if k % 2 == 0 { Fac::Garage } else { Fac::Ribbon { win: 0.4, col: 2.0 } }, &Pal { wall: pal.wall, glass: pal.glass, frame: pal.frame }, true, &mut rng);
            y0 = pod_h;
        }
        let poly = footprint(w, d, chamfer, round);
        volume(&mut b, &poly, y0, h, if y0 == 0.0 { 6.0 } else { 0.0 }, 3.2, Fac::Balcony { depth: [2.0, 2.4, 1.8, 2.2, 2.6, 2.0, 2.4, 1.6, 2.2][k] }, &pal, true, &mut rng);
        // a solid end core on some (stair / lift shaft), standing proud
        if k % 3 == 0 {
            let x = w * 0.5;
            b.bx(Vec3::new(x - 4.0, y0, -5.0), Vec3::new(x + 1.0, h + 4.0, 5.0), m(WALL, pal.wall), false);
        }
        let crown = inset(&poly, 6.0);
        volume(&mut b, &crown, h, h + 5.0, 0.0, 5.0, Fac::Ribbon { win: 0.5, col: 2.0 }, &pal, true, &mut rng);
        push(b, format!("Building_Residential_{k}"), h + 6.0, &mut meshes, &mut archs);
    }
    // office blocks: ribbon windows or curtain walls, 6-16 storeys
    for k in 0..6 {
        let mut b = B::new();
        let pal = Pal { wall: [0, 2, 3, 8, 1, 0][k], glass: [1, 2, 0, 3, 5, 4][k], frame: [1, 0, 2, 3, 1, 0][k] };
        let (w, d, storeys) = [(40.0, 30.0, 8), (55.0, 35.0, 12), (30.0, 30.0, 6), (60.0, 40.0, 16), (45.0, 25.0, 10), (35.0, 35.0, 14)][k];
        let poly = footprint(w, d, if k == 4 { 5.0 } else { 0.0 }, 0);
        let h = 5.0 + storeys as f32 * 3.8;
        let fac = if k % 2 == 0 { Fac::Ribbon { win: 0.55, col: 1.8 } } else { Fac::Curtain { col: 1.8, fin: 0.2 } };
        volume(&mut b, &poly, 0.0, h, 5.0, 3.8, fac, &pal, true, &mut rng);
        push(b, format!("Building_Office_{k}"), h + 4.0, &mut meshes, &mut archs);
    }
    // stucco apartment blocks and hotels: punched windows, some with balcony bands
    for k in 0..6 {
        let mut b = B::new();
        let pal = Pal { wall: [0, 1, 4, 5, 6, 7][k], glass: [0, 2, 1, 5, 0, 3][k], frame: [2, 2, 1, 2, 0, 2][k] };
        let (w, d, storeys) = [(30.0, 18.0, 4), (40.0, 20.0, 6), (24.0, 16.0, 3), (36.0, 24.0, 8), (28.0, 28.0, 5), (50.0, 18.0, 10)][k];
        let poly = footprint(w, d, 0.0, 0);
        let h = 3.0 * storeys as f32 + 1.0;
        let fac = if k % 3 == 2 { Fac::Balcony { depth: 1.8 } } else { Fac::Punched { col: 3.2, w: 1.6, h: 1.7 } };
        volume(&mut b, &poly, 0.0, h, 0.0, 3.0, fac, &pal, true, &mut rng);
        push(b, format!("Building_Apartments_{k}"), h + 4.0, &mut meshes, &mut archs);
    }
    // low-rise shops and warehouses: one or two storeys, flat roofs with plant
    for k in 0..4 {
        let mut b = B::new();
        let pal = Pal { wall: [1, 3, 0, 7][k], glass: [2, 0, 1, 3][k], frame: [2, 0, 1, 2][k] };
        let (w, d, storeys) = [(30.0, 24.0, 1), (42.0, 30.0, 2), (20.0, 16.0, 2), (36.0, 36.0, 1)][k];
        let poly = footprint(w, d, 0.0, 0);
        let h = 4.5 + (storeys - 1) as f32 * 3.6 + 0.5;
        volume(&mut b, &poly, 0.0, h, if storeys == 1 { 4.5 } else { 4.5 }, 3.6, Fac::Punched { col: 3.6, w: 2.2, h: 1.6 }, &pal, true, &mut rng);
        push(b, format!("Building_Shops_{k}"), h + 4.0, &mut meshes, &mut archs);
    }
    // houses
    for k in 0..5 {
        let mut b = B::new();
        let pal = Pal { wall: [0, 1, 4, 5, 6][k], glass: [0, 2, 0, 1, 5][k], frame: [2, 2, 2, 2, 2][k] };
        let (w, d, storeys, hip) = [(15.0, 11.0, 1, true), (14.5, 12.5, 2, true), (13.0, 12.0, 1, true), (14.5, 13.0, 2, false), (14.0, 10.5, 1, false)][k];
        let top = house(&mut b, w, d, storeys, hip, &pal, &mut rng);
        push(b, format!("Building_House_{k}"), top, &mut meshes, &mut archs);
    }
    // parking garages
    for k in 0..2 {
        let mut b = B::new();
        let pal = Pal { wall: [8, 0][k], glass: 0, frame: 0 };
        let (w, d, storeys) = [(50.0, 36.0, 6), (40.0, 30.0, 4)][k];
        let poly = footprint(w, d, 0.0, 0);
        let h = storeys as f32 * 3.1;
        volume(&mut b, &poly, 0.0, h, 0.0, 3.1, Fac::Garage, &pal, true, &mut rng);
        push(b, format!("Building_Garage_{k}"), h + 4.0, &mut meshes, &mut archs);
    }
    // boats (origin at the waterline, bow towards +x): a centre-console runabout, a cabin
    // cruiser, a motor yacht; and a timber dock (from the shore at x = 0 out along +x)
    for (k, &(len, beam, kind)) in [(8.5f32, 2.8f32, 0u32), (13.0, 4.0, 1), (27.0, 6.4, 2)].iter().enumerate() {
        let mut b = B::new();
        let top = boat(&mut b, len, beam, kind, k as u8);
        push(b, format!("Building_Boat_{k}"), top, &mut meshes, &mut archs);
    }
    {
        let mut b = B::new();
        b.bx(Vec3::new(0.0, 0.45, -1.4), Vec3::new(13.0, 0.7, 1.4), m(WOOD, 0), true);
        for s in 0..5 {
            let x = 0.6 + s as f32 * 3.0;
            for z in [-1.25f32, 1.25] {
                b.bx(Vec3::new(x - 0.13, -1.5, z - 0.13), Vec3::new(x + 0.13, 1.1, z + 0.13), m(WOOD, 1), false);
            }
        }
        push(b, "Building_Dock".into(), 1.2, &mut meshes, &mut archs);
    }
    (meshes, archs)
}

/// A boat: hull (white gelcoat, a painted boot stripe, a teak deck), and by `kind` a console
/// with a T-top (0), a cabin with a windscreen (1), or two decks of a motor yacht with dark
/// glazing and a flybridge (2). Returns its height above the waterline.
fn boat(b: &mut B, len: f32, beam: f32, kind: u32, pal: u8) -> f32 {
    let (draft, fb) = (0.25 * beam.sqrt(), 0.32 * beam.sqrt());
    let hull = ccw(vec![Vec2::new(-len * 0.5, -beam * 0.5), Vec2::new(len * 0.2, -beam * 0.5), Vec2::new(len * 0.5, 0.0), Vec2::new(len * 0.2, beam * 0.5), Vec2::new(-len * 0.5, beam * 0.5)]);
    b.prism(&hull, -draft, fb, m(PAINT, 4), m(WOOD, 0), true);
    // gunwale and boot stripe
    for k in 0..hull.len() {
        let (a, c) = (hull[k], hull[(k + 1) % hull.len()]);
        b.strip(a, c, fb - 0.3, fb - 0.12, 0.0, 0.03, m(PAINT, pal % 3));
        b.strip(a, c, fb, fb + 0.08, -0.08, 0.0, m(PAINT, 4));
    }
    let v = |x0: f32, x1: f32, z: f32, y0: f32, y1: f32, mat: u8, b: &mut B| b.bx(Vec3::new(x0, y0, -z), Vec3::new(x1, y1, z), mat, false);
    match kind {
        0 => {
            v(-0.6, 0.5, 0.5, fb, fb + 1.1, m(PAINT, 4), b);
            v(-0.45, 0.45, 0.48, fb + 1.1, fb + 1.4, m(GLASS, 6), b);
            v(-1.2, 0.6, 0.9, fb + 2.1, fb + 2.2, m(PAINT, 4), b);
            for (x, z) in [(-1.1f32, 0.8f32), (-1.1, -0.8), (0.5, 0.8), (0.5, -0.8)] {
                b.bx(Vec3::new(x - 0.03, fb + 1.1, z - 0.03), Vec3::new(x + 0.03, fb + 2.1, z + 0.03), m(FRAME, 1), false);
            }
            fb + 2.2
        }
        1 => {
            v(-len * 0.3, len * 0.12, beam * 0.36, fb, fb + 1.0, m(PAINT, 4), b);
            v(-len * 0.28, len * 0.1, beam * 0.37, fb + 1.0, fb + 1.55, m(GLASS, 6), b);
            v(-len * 0.3, len * 0.12, beam * 0.36, fb + 1.55, fb + 1.7, m(PAINT, 4), b);
            fb + 1.7
        }
        _ => {
            v(-len * 0.36, len * 0.22, beam * 0.42, fb, fb + 0.9, m(PAINT, 4), b);
            v(-len * 0.35, len * 0.21, beam * 0.43, fb + 0.9, fb + 1.9, m(GLASS, 6), b);
            v(-len * 0.38, len * 0.18, beam * 0.44, fb + 1.9, fb + 2.15, m(PAINT, 4), b);
            v(-len * 0.24, len * 0.08, beam * 0.36, fb + 2.15, fb + 3.0, m(GLASS, 6), b);
            v(-len * 0.26, len * 0.06, beam * 0.38, fb + 3.0, fb + 3.2, m(PAINT, 4), b);
            v(-len * 0.12, len * 0.02, beam * 0.3, fb + 3.2, fb + 4.1, m(FRAME, 1), b);
            fb + 4.1
        }
    }
}

/// The archetypes as a species pack: one mesh and one variant (one part, at its origin) each.
pub fn pack() -> (TreePack, Vec<Arch>) {
    let (meshes, archs) = archetypes();
    let trees = meshes
        .iter()
        .enumerate()
        .map(|(i, mesh)| {
            let (lo, hi) = mesh.bounds();
            TreeAsset { name: mesh.name.clone(), bmin: lo, bmax: hi, parts: vec![TreePart { mesh: i as u32, xform: Affine3A::IDENTITY }] }
        })
        .collect();
    (TreePack { meshes, trees }, archs)
}

/// An archetype's class from its name (the variants keep only their names).
pub fn class_of(name: &str) -> Class {
    if name.contains("Boat") {
        Class::Boat
    } else if name.contains("Dock") {
        Class::Dock
    } else if name.contains("Tower") || name.contains("Residential") {
        Class::Tower
    } else if name.contains("Office") || name.contains("Apartments") {
        Class::MidRise
    } else if name.contains("House") {
        Class::House
    } else {
        Class::LowRise
    }
}
