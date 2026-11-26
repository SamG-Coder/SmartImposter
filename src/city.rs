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
pub const CARPAINT: u8 = 14;
pub const ASPHALT: u8 = 15;

/// What an archetype is: the city plan (globe/city.rs) picks by it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Class {
    Tower,
    MidRise,
    LowRise,
    House,
    /// boats (moored at docks and out on the water) and docks: not for lots
    Boat,
    /// boats under way, with their wakes
    Underway,
    Dock,
    /// cars (parking lots, driveways, traffic) and bridge decks: not for lots
    Car,
    Bridge,
    /// elevated highway and ramp decks
    Highway,
    /// towers of their own shapes: one each in the densest cores
    Landmark,
    /// street lights and traffic signals
    Street,
    /// a house's back-yard pool
    Pool,
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
            // small balconies (a slab and a glass rail) under every other window of most
            // walk-ups and apartment blocks
            let balconies = floors >= 3 && w >= 1.4 && h >= 1.5 && rng.f() < 0.65;
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
                        if balconies && f >= 1 && j % 2 == 0 {
                            let (a0, a1) = (p0 - dir * 0.35, p1 + dir * 0.35);
                            b.strip(a0, a1, y - 0.02, y + 0.14, 0.0, 1.0, m(SLAB, 0));
                            b.wall(a0, a1, y + 0.14, y + 1.05, 0.98, m(RAIL, pal.glass), Some((cu * ww, ww, y, 1.0)));
                        }
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
            for _ in 0..(3 + rng.pick(6)) {
                let (sx, sz) = (rng.range(1.5, ((hi.x - lo.x) * 0.35).max(1.6)), rng.range(1.5, ((hi.y - lo.y) * 0.35).max(1.6)));
                let cx = rng.range(lo.x + sx * 0.5, hi.x - sx * 0.5);
                let cz = rng.range(lo.y + sz * 0.5, hi.y - sz * 0.5);
                let h = rng.range(1.2, 3.2);
                b.bx(Vec3::new(cx - sx * 0.5, y1, cz - sz * 0.5), Vec3::new(cx + sx * 0.5, y1 + h, cz + sz * 0.5), m(EQUIP, rng.pick(3) as u8), false);
            }
            let (ax, az) = (hi.x - lo.x, hi.y - lo.y);
            // a water tank on legs
            if ax * az > 200.0 && rng.f() < 0.45 {
                let (cx, cz) = (rng.range(lo.x + 2.0, hi.x - 2.0), rng.range(lo.y + 2.0, hi.y - 2.0));
                let tank: Vec<Vec2> = footprint(2.6, 2.6, 0.75, 0).iter().map(|p| *p + Vec2::new(cx, cz)).collect();
                for (dx, dz) in [(-0.9f32, -0.9f32), (0.9, -0.9), (0.9, 0.9), (-0.9, 0.9)] {
                    b.bx(Vec3::new(cx + dx - 0.08, y1, cz + dz - 0.08), Vec3::new(cx + dx + 0.08, y1 + 2.0, cz + dz + 0.08), m(FRAME, 0), false);
                }
                b.prism(&tank, y1 + 2.0, y1 + 4.6, m(EQUIP, 1), m(EQUIP, 2), true);
            }
            // a roof terrace: a timber deck behind a glass rail, a few planters
            if !matches!(fac, Fac::Curtain { .. }) && ax > 10.0 && az > 8.0 && rng.f() < 0.4 {
                let (dx0, dz0) = (lo.x + 0.5, lo.y + 0.5);
                let (dx1, dz1) = (lo.x + ax * 0.45, lo.y + az * 0.55);
                b.bx(Vec3::new(dx0, y1, dz0), Vec3::new(dx1, y1 + 0.15, dz1), m(WOOD, 0), false);
                let deck = [Vec2::new(dx0, dz0), Vec2::new(dx1, dz0), Vec2::new(dx1, dz1), Vec2::new(dx0, dz1)];
                b.wall(deck[1], deck[2], y1 + 0.15, y1 + 1.15, 0.0, m(RAIL, 0), None);
                b.wall(deck[2], deck[3], y1 + 0.15, y1 + 1.15, 0.0, m(RAIL, 0), None);
                for t in 0..3 {
                    let px = dx0 + (dx1 - dx0) * (0.2 + 0.3 * t as f32);
                    b.bx(Vec3::new(px - 0.5, y1 + 0.15, dz1 - 1.4), Vec3::new(px + 0.5, y1 + 0.8, dz1 - 0.4), m(TILE, 2), false);
                }
                // and on most, a pool in its coping on the other half of the roof
                if ax > 16.0 && rng.f() < 0.7 {
                    let (px0, px1) = (lo.x + ax * 0.55, lo.x + ax * 0.55 + (ax * 0.4).min(12.0));
                    let (pz0, pz1) = (lo.y + 1.0, lo.y + 1.0 + (az * 0.4).min(5.0));
                    b.bx(Vec3::new(px0 - 0.6, y1, pz0 - 0.6), Vec3::new(px1 + 0.6, y1 + 0.3, pz1 + 0.6), m(SLAB, 0), false);
                    b.bx(Vec3::new(px0, y1 + 0.3, pz0), Vec3::new(px1, y1 + 0.32, pz1), m(POOL, 0), false);
                }
            }
        }
    }
}

/// A house: stucco walls with windows, a tiled hip roof (or flat with a parapet).
/// A hip roof over the rectangle +-hx, +-hz at height h, rising `pitch` x the shorter half
/// width, with its soffit. Returns the ridge height.
fn hip_roof(b: &mut B, hx: f32, hz: f32, h: f32, pitch: f32, tm: u8, soffit: u8) -> f32 {
    let rise = pitch * hx.min(hz);
    let along_x = hx >= hz;
    let r = if along_x { hx - hz } else { hz - hx };
    let (r0, r1) = if along_x { (Vec3::new(-r, h + rise, 0.0), Vec3::new(r, h + rise, 0.0)) } else { (Vec3::new(0.0, h + rise, -r), Vec3::new(0.0, h + rise, r)) };
    let c = [Vec3::new(-hx, h, -hz), Vec3::new(hx, h, -hz), Vec3::new(hx, h, hz), Vec3::new(-hx, h, hz)];
    if along_x {
        b.quad([c[0], r0, r1, c[1]], tm);
        b.quad([c[2], r1, r0, c[3]], tm);
        b.quad([c[1], r1, r1, c[2]], tm);
        b.quad([c[3], r0, r0, c[0]], tm);
    } else {
        b.quad([c[1], r0, r1, c[2]], tm);
        b.quad([c[3], r1, r0, c[0]], tm);
        b.quad([c[0], r0, r0, c[1]], tm);
        b.quad([c[2], r1, r1, c[3]], tm);
    }
    b.cap(&ccw(c.iter().map(|p| Vec2::new(p.x, p.z)).collect()), h, false, soffit);
    h + rise
}

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
        let tm = m(TILE, [3u8, 4, 4, 0, 3, 4][rng.pick(6)]);
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

/// All archetypes, deterministic (the variant slots are shared with the plants: 512, rt.glsl
/// plantVariant).
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
        let pal = Pal { wall: [0, 2, 8][k % 3], glass: [4, 1, 2, 5, 4, 0, 2, 4, 3][k], frame: [1, 2, 1, 0, 2, 1, 2, 1, 0][k] };
        // (the fifth is a supertall: the skyline's peak, chosen in the densest cores)
        let h = [120.0, 150.0, 185.0, 215.0, 320.0, 105.0, 165.0, 200.0, 135.0][k];
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
    // landmarks: three towers with silhouettes of their own for the skyline's peaks: a tapering
    // tower stepped back every few floors under a needle spire, a cylinder whose crown is a
    // ring of fins round an open top, a slab with a curved sail crown sweeping above its roof
    {
        let mut b = B::new();
        let pal = Pal { wall: 2, glass: 4, frame: 1 };
        let mut lo = 0.0f32;
        let (mut w, mut d) = (40.0f32, 40.0f32);
        for t in 0..6 {
            let hi = lo + [70.0f32, 55.0, 45.0, 40.0, 30.0, 25.0][t];
            let poly = footprint(w, d, w * 0.22, 0);
            volume(&mut b, &poly, lo, hi, if lo == 0.0 { 7.0 } else { 0.0 }, 4.0, Fac::Curtain { col: 1.5, fin: 0.3 }, &pal, true, &mut rng);
            lo = hi;
            w -= 5.0;
            d -= 5.0;
        }
        b.bx(Vec3::new(-0.7, lo, -0.7), Vec3::new(0.7, lo + 70.0, 0.7), m(FRAME, 2), false);
        push(b, "Building_Landmark_0".into(), lo + 70.0, &mut meshes, &mut archs);
    }
    {
        let mut b = B::new();
        let pal = Pal { wall: 0, glass: 5, frame: 2 };
        let poly = footprint(38.0, 38.0, 19.0, 6);
        volume(&mut b, &poly, 0.0, 250.0, 7.0, 3.4, Fac::Balcony { depth: 2.2 }, &pal, false, &mut rng);
        b.cap(&poly, 250.0, true, m(ROOF, 0));
        // the crown: fins standing round the rim, open between them
        for k in 0..16 {
            let a = k as f32 / 16.0 * std::f32::consts::TAU;
            let (c, s) = (a.cos(), a.sin());
            let (r0, r1) = (17.0f32, 19.5f32);
            let p = |r: f32, t: f32| Vec2::new(c * r - s * t, s * r + c * t);
            let fin = ccw(vec![p(r0, -0.4), p(r1, -0.4), p(r1, 0.4), p(r0, 0.4)]);
            b.prism(&fin, 250.0, 278.0, m(SLAB, 0), m(SLAB, 0), false);
        }
        push(b, "Building_Landmark_1".into(), 278.0, &mut meshes, &mut archs);
    }
    {
        let mut b = B::new();
        let pal = Pal { wall: 0, glass: 1, frame: 2 };
        let poly = footprint(54.0, 26.0, 0.0, 0);
        volume(&mut b, &poly, 0.0, 230.0, 7.0, 3.6, Fac::Curtain { col: 1.6, fin: 0.25 }, &pal, true, &mut rng);
        // the sail: a curved plate rising off one end, leaning back over the roof
        let n = 12;
        for k in 0..n {
            let (t0, t1) = (k as f32 / n as f32, (k + 1) as f32 / n as f32);
            let x = |t: f32| -27.0 + 30.0 * t * t;
            let y = |t: f32| 230.0 + 60.0 * t;
            b.bx(Vec3::new(x(t0).min(x(t1)) - 0.6, y(t0), -13.0), Vec3::new(x(t0).max(x(t1)) + 0.6, y(t1), 13.0), m(GLASS, 4), false);
        }
        push(b, "Building_Landmark_2".into(), 290.0, &mut meshes, &mut archs);
    }
    // residential towers: white slab bands, glass rails, blue-green glass (the Miami look)
    for k in 0..9 {
        let mut b = B::new();
        let pal = Pal { wall: [0, 1, 0, 0, 0, 7, 0, 1, 0][k], glass: [5, 2, 4, 5, 0, 4, 5, 1, 2][k], frame: [2, 2, 2, 2, 2, 1, 2, 2, 2][k] };
        let h = [95.0, 140.0, 180.0, 70.0, 210.0, 120.0, 160.0, 85.0, 290.0][k];
        let (w, d) = ([44.0, 52.0, 40.0, 48.0, 42.0, 56.0, 38.0, 50.0, 46.0][k], [24.0, 26.0, 30.0, 22.0, 34.0, 24.0, 38.0, 26.0, 28.0][k]);
        let (chamfer, round) = [(0.0, 0), (8.0, 4), (0.0, 0), (6.0, 0), (12.0, 5), (0.0, 0), (10.0, 4), (0.0, 0), (9.0, 0)][k];
        let pod_h = [16.0, 20.0, 0.0, 14.0, 24.0, 18.0, 0.0, 12.0, 22.0][k];
        let mut y0 = 0.0;
        if pod_h > 0.0 {
            let pod = footprint(w + 14.0, d + 18.0, 0.0, 0);
            volume(&mut b, &pod, 0.0, pod_h, 5.0, 3.0, if k % 2 == 0 { Fac::Garage } else { Fac::Ribbon { win: 0.4, col: 2.0 } }, &Pal { wall: pal.wall, glass: pal.glass, frame: pal.frame }, true, &mut rng);
            y0 = pod_h;
            // the amenity deck on the podium roof: a pool in its coping, a timber deck
            let z0 = d * 0.5 + 1.5;
            b.bx(Vec3::new(-w * 0.35, pod_h, z0), Vec3::new(w * 0.35, pod_h + 0.12, z0 + 7.0), m(WOOD, 0), false);
            b.bx(Vec3::new(-w * 0.25, pod_h + 0.12, z0 + 1.0), Vec3::new(w * 0.25, pod_h + 0.2, z0 + 6.0), m(SLAB, 0), false);
            b.bx(Vec3::new(-w * 0.25 + 0.5, pod_h + 0.2, z0 + 1.5), Vec3::new(w * 0.25 - 0.5, pod_h + 0.22, z0 + 5.5), m(POOL, 0), false);
        }
        let poly = footprint(w, d, chamfer, round);
        volume(&mut b, &poly, y0, h, if y0 == 0.0 { 6.0 } else { 0.0 }, 3.2, Fac::Balcony { depth: [2.0, 2.4, 1.8, 2.2, 2.6, 2.0, 2.4, 1.6, 2.2][k] }, &pal, true, &mut rng);
        // white concrete fins at the corners, full height: the towers read white, glass between
        for c in [Vec2::new(-w * 0.5, -d * 0.5), Vec2::new(w * 0.5, -d * 0.5), Vec2::new(w * 0.5, d * 0.5), Vec2::new(-w * 0.5, d * 0.5)] {
            if chamfer == 0.0 {
                b.bx(Vec3::new(c.x - 0.6, y0, c.y - 0.6), Vec3::new(c.x + 0.6, h + 1.0, c.y + 0.6), m(SLAB, 0), false);
            }
        }
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

    // ---- more of the low and mid-rise city
    // South Beach Art Deco walk-ups: pastel stucco, rounded street corners, "eyebrow" sun
    // shades over every window band, a fin rising over the entrance
    for k in 0..6 {
        let mut b = B::new();
        let pal = Pal { wall: [4, 6, 5, 0, 1, 4][k], glass: [5, 0, 1, 5, 2, 0][k], frame: 2 };
        let (w, d, storeys) = [(28.0f32, 18.0f32, 3usize), (34.0, 20.0, 4), (24.0, 16.0, 3), (30.0, 22.0, 5), (36.0, 18.0, 3), (26.0, 20.0, 4)][k];
        let poly = footprint(w, d, 5.0, 4);
        let h = 3.3 * storeys as f32 + 0.6;
        volume(&mut b, &poly, 0.0, h, 0.0, 3.3, Fac::Punched { col: 2.6, w: 1.9, h: 1.7 }, &pal, true, &mut rng);
        for f in 1..=storeys {
            let y = f as f32 * 3.3 - 0.75;
            for e in 0..poly.len() {
                // (shallow, in a darker shade of the wall: from above the windows show between them)
                b.strip(poly[e], poly[(e + 1) % poly.len()], y, y + 0.12, 0.0, 0.4, m(SPANDREL, pal.wall));
            }
        }
        // the fin over the entrance on the street side (-z)
        b.bx(Vec3::new(-0.6, 0.0, -d * 0.5 - 0.5), Vec3::new(0.6, h + 4.0, -d * 0.5 + 1.0), m(WALL, pal.wall), false);
        b.bx(Vec3::new(-2.5, h - 0.4, -d * 0.5 - 0.6), Vec3::new(2.5, h + 0.6, -d * 0.5 + 0.4), m(SPANDREL, [1u8, 3, 6, 5, 4, 1][k]), false);
        push(b, format!("Building_Deco_{k}"), h + 4.0, &mut meshes, &mut archs);
    }
    // Mediterranean revival mid-rises: sand and cream stucco, recessed windows; two with tile hip
    // roofs, the rest flat behind a parapet with their rooftop plant (most of the city's
    // mid-rise roofs are flat)
    for k in 0..5 {
        let mut b = B::new();
        let pal = Pal { wall: [7, 1, 3, 5, 0][k], glass: [2, 0, 1, 2, 5][k], frame: 2 };
        let (w, d, storeys) = [(30.0f32, 22.0f32, 6usize), (40.0, 18.0, 8), (26.0, 26.0, 5), (36.0, 24.0, 10), (32.0, 20.0, 7)][k];
        let poly = footprint(w, d, 0.0, 0);
        let h = 3.2 * storeys as f32 + 1.0;
        let hip = k == 0 || k == 3;
        volume(&mut b, &poly, 0.0, h, 4.5, 3.2, Fac::Punched { col: 3.0, w: 1.4, h: 1.8 }, &pal, !hip, &mut rng);
        let top = if hip {
            b.cap(&poly, h, true, m(ROOF, 0));
            hip_roof(&mut b, w * 0.5 + 0.8, d * 0.5 + 0.8, h, 0.32, m(TILE, [4u8, 3, 4, 0, 4][k]), m(WALL, pal.wall))
        } else {
            h + 1.0
        };
        push(b, format!("Building_Med_{k}"), top, &mut meshes, &mut archs);
    }
    // long condo slabs with balconies, 9-15 storeys
    for k in 0..4 {
        let mut b = B::new();
        let pal = Pal { wall: 0, glass: [5, 0, 4, 1][k], frame: 2 };
        let (w, d, storeys) = [(54.0f32, 16.0f32, 12usize), (46.0, 18.0, 9), (60.0, 16.0, 15), (40.0, 20.0, 11)][k];
        let poly = footprint(w, d, 0.0, 0);
        let h = 3.1 * storeys as f32 + 5.0;
        volume(&mut b, &poly, 0.0, h, 5.0, 3.1, Fac::Balcony { depth: 1.8 }, &pal, true, &mut rng);
        push(b, format!("Building_Condo_{k}"), h + 4.0, &mut meshes, &mut archs);
    }
    // L-shaped and courtyard apartment blocks, 3-6 storeys
    for k in 0..4 {
        let mut b = B::new();
        let pal = Pal { wall: [1, 5, 7, 4][k], glass: [0, 2, 5, 1][k], frame: 2 };
        let storeys = [3usize, 4, 5, 6][k];
        let h = 3.0 * storeys as f32 + 0.8;
        let fac = if k % 2 == 0 { Fac::Punched { col: 3.0, w: 1.5, h: 1.6 } } else { Fac::Balcony { depth: 1.6 } };
        let a = ccw(vec![Vec2::new(-20.0, -16.0), Vec2::new(20.0, -16.0), Vec2::new(20.0, -3.0), Vec2::new(-20.0, -3.0)]);
        let c = ccw(vec![Vec2::new(-20.0, -3.0), Vec2::new(-7.0, -3.0), Vec2::new(-7.0, 16.0), Vec2::new(-20.0, 16.0)]);
        volume(&mut b, &a, 0.0, h, 0.0, 3.0, fac, &pal, true, &mut rng);
        volume(&mut b, &c, 0.0, h, 0.0, 3.0, fac, &pal, true, &mut rng);
        if k >= 2 {
            let e = ccw(vec![Vec2::new(7.0, -3.0), Vec2::new(20.0, -3.0), Vec2::new(20.0, 16.0), Vec2::new(7.0, 16.0)]);
            volume(&mut b, &e, 0.0, h, 0.0, 3.0, fac, &pal, true, &mut rng);
        }
        // the courtyard's pool
        b.bx(Vec3::new(-4.0, 0.0, 2.0), Vec3::new(4.0, 0.15, 12.0), m(SLAB, 0), false);
        b.bx(Vec3::new(-3.2, 0.15, 3.0), Vec3::new(3.2, 0.18, 11.0), m(POOL, 0), false);
        push(b, format!("Building_Court_{k}"), h + 1.0, &mut meshes, &mut archs);
    }
    // strip malls, a big-box store, gas stations
    for k in 0..3 {
        let mut b = B::new();
        let pal = Pal { wall: [1, 7, 3][k], glass: [2, 0, 1][k], frame: [1, 0, 2][k] };
        let (w, d) = [(50.0f32, 16.0f32), (44.0, 18.0), (36.0, 14.0)][k];
        let poly = footprint(w, d, 0.0, 0);
        volume(&mut b, &poly, 0.0, 5.5, 5.0, 5.0, Fac::Punched { col: 6.0, w: 4.2, h: 2.6 }, &pal, true, &mut rng);
        // the arcade along the front and a raised sign band
        b.bx(Vec3::new(-w * 0.5, 3.4, -d * 0.5 - 2.5), Vec3::new(w * 0.5, 3.7, -d * 0.5), m(FRAME, pal.frame), false);
        b.bx(Vec3::new(-w * 0.2, 5.5, -d * 0.5 - 0.2), Vec3::new(w * 0.2, 7.2, -d * 0.5 + 0.4), m(SPANDREL, [6u8, 4, 5][k]), false);
        push(b, format!("Building_Strip_{k}"), 7.5, &mut meshes, &mut archs);
    }
    {
        let mut b = B::new();
        let pal = Pal { wall: 2, glass: 2, frame: 1 };
        let poly = footprint(50.0, 38.0, 0.0, 0);
        volume(&mut b, &poly, 0.0, 9.0, 0.0, 9.0, Fac::Ribbon { win: 0.12, col: 8.0 }, &pal, true, &mut rng);
        b.bx(Vec3::new(-8.0, 0.0, -19.5), Vec3::new(8.0, 10.5, -18.5), m(SPANDREL, 1), false);
        push(b, "Building_BigBox".into(), 11.0, &mut meshes, &mut archs);
    }
    {
        let mut b = B::new();
        let pal = Pal { wall: 0, glass: 0, frame: 1 };
        // the canopy over the pumps on six pillars, the shop behind
        b.bx(Vec3::new(-11.0, 4.6, -7.0), Vec3::new(11.0, 5.3, 7.0), m(SPANDREL, 1), true);
        for x in [-8.0f32, 0.0, 8.0] {
            for z in [-3.5f32, 3.5] {
                b.bx(Vec3::new(x - 0.25, 0.0, z - 0.25), Vec3::new(x + 0.25, 4.6, z + 0.25), m(WALL, 0), false);
                b.bx(Vec3::new(x - 0.9, 0.0, z - 0.4), Vec3::new(x + 0.9, 1.6, z + 0.4), m(PAINT, 1), false);
            }
        }
        let shop = footprint(16.0, 10.0, 0.0, 0);
        let moved: Vec<Vec2> = shop.iter().map(|p| *p + Vec2::new(0.0, 14.0)).collect();
        volume(&mut b, &moved, 0.0, 4.5, 4.5, 4.5, Fac::Punched { col: 4.0, w: 3.0, h: 2.4 }, &pal, true, &mut rng);
        push(b, "Building_Gas".into(), 5.5, &mut meshes, &mut archs);
    }
    // more houses: L-shaped with a garage wing, grey and white tile roofs, modern two-storey
    for k in 0..6 {
        let mut b = B::new();
        let pal = Pal { wall: [0, 5, 1, 6, 0, 7][k], glass: [0, 2, 5, 0, 1, 2][k], frame: 2 };
        let (w, d, storeys) = [(13.0f32, 10.0f32, 1usize), (14.0, 11.0, 2), (12.0, 10.0, 1), (15.0, 10.0, 1), (13.0, 12.0, 2), (14.0, 9.0, 1)][k];
        let tile = [3u8, 4, 0, 3, 4, 1][k];
        let main = footprint(w, d, 0.0, 0);
        let h = storeys as f32 * 3.1 + 0.4;
        volume(&mut b, &main, 0.0, h, 0.0, 3.1, Fac::Punched { col: 3.4, w: 1.5, h: 1.6 }, &pal, k == 4, &mut rng);
        let mut top = h + 1.0;
        if k != 4 {
            top = hip_roof(&mut b, w * 0.5 + 0.6, d * 0.5 + 0.6, h, 0.45, m(TILE, tile), m(WALL, pal.wall));
        }
        // the garage wing (one storey) off one end, its door to the street (-z)
        let gw: Vec<Vec2> = footprint(6.5, 7.0, 0.0, 0).iter().map(|p| *p + Vec2::new(w * 0.5 + 3.25, -d * 0.5 + 3.5)).collect();
        volume(&mut b, &gw, 0.0, 3.0, 0.0, 3.0, Fac::Punched { col: 6.5, w: 0.1, h: 0.1 }, &pal, true, &mut rng);
        b.bx(Vec3::new(w * 0.5 + 0.8, 0.1, -d * 0.5 - 0.05), Vec3::new(w * 0.5 + 5.7, 2.4, -d * 0.5 + 0.05), m(FRAME, 2), false);
        push(b, format!("Building_House_L{k}"), top, &mut meshes, &mut archs);
    }
    // boats (origin at the waterline, bow towards +x): a centre-console runabout, a cabin
    // cruiser, a motor yacht; and a timber dock (from the shore at x = 0 out along +x)
    for (k, &(len, beam, kind)) in [(8.5f32, 2.8f32, 0u32), (13.0, 4.0, 1), (27.0, 6.4, 2)].iter().enumerate() {
        let mut b = B::new();
        let top = boat(&mut b, len, beam, kind, k as u8);
        push(b, format!("Building_Boat_{k}"), top, &mut meshes, &mut archs);
    }
    // the same boats underway: a wake of foam behind them (scattered patches, so the water shows
    // between them in the mesh and the impostor alike): the turbulent trail straight back from
    // the stern, widening, and the two arms of the Kelvin wedge (19.5 degrees each side)
    for (k, &(len, beam, kind)) in [(8.5f32, 2.8f32, 0u32), (13.0, 4.0, 1), (27.0, 6.4, 2)].iter().enumerate() {
        let mut b = B::new();
        let top = boat(&mut b, len, beam, kind, k as u8);
        let mut r = Rng(0xA4E5 + k as u64);
        let wake = len * 5.0;
        let foam = m(POOL, 1);
        // (each a streak drawn out along the boat's track, 4-6 x as long as wide, tapered:
        // diamond-shaped, so the trail reads as foam lines, not as scattered tiles)
        let patch = |b: &mut B, x: f32, z: f32, s: f32, rng: &mut Rng| {
            let l = s * rng.range(2.0, 3.0);
            let w = s * 0.45;
            let y = 0.12;
            b.quad([Vec3::new(x - l, y, z), Vec3::new(x, y, z - w), Vec3::new(x + l, y, z), Vec3::new(x, y, z + w)], foam);
        };
        let stern = -len * 0.5;
        for _ in 0..(wake * 2.2) as usize {
            // the trail: dense near the stern, fading and widening behind
            let t = r.f().powf(1.6);
            let x = stern - t * wake;
            let half = beam * (0.45 + 1.6 * t);
            let z = (r.f() * 2.0 - 1.0) * half;
            if r.f() < 1.0 - 0.75 * t {
                patch(&mut b, x, z, r.range(0.25, 0.9) * (1.0 + t), &mut r);
            }
        }
        let tan = 19.5f32.to_radians().tan();
        for side in [-1.0f32, 1.0] {
            for _ in 0..(wake * 1.2) as usize {
                let t = r.f();
                let x = len * 0.35 - t * wake * 0.8;
                let z = side * ((len * 0.35 - x) * tan + beam * 0.4) + (r.f() - 0.5) * (0.6 + 2.0 * t);
                if r.f() < 1.0 - 0.8 * t {
                    patch(&mut b, x, z, r.range(0.2, 0.6), &mut r);
                }
            }
        }
        push(b, format!("Building_Boat_Wake_{k}"), top, &mut meshes, &mut archs);
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
    // a back-yard pool (6 x 8 m with its coping and a strip of deck), placed with its house
    {
        let mut b = B::new();
        b.bx(Vec3::new(-3.6, 0.0, -4.6), Vec3::new(3.6, 0.15, 4.6), m(SLAB, 0), false);
        b.bx(Vec3::new(-3.0, 0.15, -4.0), Vec3::new(3.0, 0.17, 4.0), m(POOL, 0), false);
        b.bx(Vec3::new(-3.6, 0.0, 4.6), Vec3::new(3.6, 0.1, 6.0), m(WOOD, 0), false);
        push(b, "Building_Pool".into(), 0.2, &mut meshes, &mut archs);
    }
    // cars (origin on the ground, front towards +x; painted white, the paint's colour is each
    // car's own: building.rchit / impostor.rchit by its record's seed)
    for (k, &(len, wid, hgt, kind)) in [(4.0f32, 1.75f32, 1.45f32, 0u32), (4.75, 1.82, 1.45, 0), (4.85, 1.95, 1.75, 1), (5.6, 2.0, 1.85, 2), (5.2, 2.0, 2.15, 3), (12.0, 2.55, 3.1, 4)].iter().enumerate() {
        let mut b = B::new();
        car(&mut b, len, wid, hgt, kind);
        push(b, format!("Building_Car_{k}"), hgt, &mut meshes, &mut archs);
    }
    // street furniture (its arm towards +x): a street light (a 9 m pole, an arm of 2.4 m, the
    // head) and a traffic signal (a pole with a 7 m mast arm over the road, three signal heads
    // hanging from it, a pedestrian signal on the pole)
    {
        let mut b = B::new();
        b.bx(Vec3::new(-0.09, 0.0, -0.09), Vec3::new(0.09, 9.0, 0.09), m(FRAME, 1), false);
        b.bx(Vec3::new(0.0, 8.8, -0.05), Vec3::new(2.4, 8.95, 0.05), m(FRAME, 1), false);
        b.bx(Vec3::new(1.9, 8.6, -0.22), Vec3::new(2.8, 8.85, 0.22), m(POOL, 2), false);
        push(b, "Building_StreetLight".into(), 9.0, &mut meshes, &mut archs);
    }
    {
        let mut b = B::new();
        b.bx(Vec3::new(-0.16, 0.0, -0.16), Vec3::new(0.16, 6.6, 0.16), m(FRAME, 0), false);
        b.bx(Vec3::new(0.0, 6.2, -0.1), Vec3::new(7.5, 6.42, 0.1), m(FRAME, 0), false);
        for x in [2.8f32, 5.0, 7.0] {
            b.bx(Vec3::new(x - 0.18, 5.05, -0.18), Vec3::new(x + 0.18, 6.2, 0.18), m(DARK, 0), false);
            b.bx(Vec3::new(x - 0.1, 5.85, -0.2), Vec3::new(x + 0.1, 6.0, -0.18), m(PAINT, 1), false);
        }
        b.bx(Vec3::new(-0.2, 2.4, 0.16), Vec3::new(0.2, 2.9, 0.4), m(DARK, 0), false);
        push(b, "Building_Signal".into(), 6.6, &mut meshes, &mut archs);
    }
    // elevated road decks, 20 m along (+x), deck top at 10 m: the highway (six lanes, a median
    // barrier, two columns on a crossbeam) and a ramp (one lane, one column); the columns run
    // 14 m below the deck's foot, so a deck sunk for a ramp's climb still stands on them
    // (the third: a ramp piece of 6 m, laid every 5 m round the loops' curves)
    // (then each again without its left (-z) or its right (+z) parapet: where a ramp runs
    // alongside, the two decks meet at a painted gore, not across two walls; Highway_3..8)
    let kinds: Vec<(usize, i32)> = (0..3).map(|k| (k, 0)).chain((0..3).flat_map(|k| [(k, -1), (k, 1)])).collect();
    for (n, &(k, open)) in kinds.iter().enumerate() {
        let (width, hl) = [(30.0f32, 10.05f32), (11.0, 10.05), (11.0, 3.1)][k];
        let mut b = B::new();
        let hw = width * 0.5;
        b.bx(Vec3::new(-hl, 8.6, -hw), Vec3::new(hl, 10.0, hw), m(SLAB, 0), true);
        b.quad([Vec3::new(-hl, 10.02, -hw + 0.5), Vec3::new(hl, 10.02, -hw + 0.5), Vec3::new(hl, 10.02, hw - 0.5), Vec3::new(-hl, 10.02, hw - 0.5)], m(ASPHALT, 2 + k.min(1) as u8));
        for (side, z) in [(-1, -hw), (1, hw - 0.45)] {
            if side == open {
                // (the open side: the deck's asphalt runs to its edge)
                let z0 = if side < 0 { -hw } else { hw - 0.5 };
                b.quad([Vec3::new(-hl, 10.02, z0), Vec3::new(hl, 10.02, z0), Vec3::new(hl, 10.02, z0 + 0.5), Vec3::new(-hl, 10.02, z0 + 0.5)], m(ASPHALT, 2 + k.min(1) as u8));
                continue;
            }
            b.bx(Vec3::new(-hl, 10.0, z), Vec3::new(hl, 11.0, z + 0.45), m(SLAB, 0), false);
        }
        if k == 0 {
            b.bx(Vec3::new(-10.05, 10.0, -0.3), Vec3::new(10.05, 10.85, 0.3), m(SLAB, 0), false);
            b.bx(Vec3::new(-1.2, 7.4, -hw + 2.0), Vec3::new(1.2, 8.6, hw - 2.0), m(SLAB, 0), false);
            for z in [-7.0f32, 7.0] {
                b.bx(Vec3::new(-0.9, -14.0, z - 0.9), Vec3::new(0.9, 7.4, z + 0.9), m(SLAB, 0), false);
            }
        } else if k == 1 {
            b.bx(Vec3::new(-0.8, -14.0, -0.8), Vec3::new(0.8, 8.6, 0.8), m(SLAB, 0), false);
        } else {
            b.bx(Vec3::new(-0.6, -14.0, -0.6), Vec3::new(0.6, 8.6, 0.6), m(SLAB, 0), false);
        }
        push(b, format!("Building_Highway_{n}"), 11.0, &mut meshes, &mut archs);
    }
    // bridge decks, 20 m along the street (+x, 21 m long: the pieces of a crossing overlap,
    // and are pitched with its arch, globe/city.rs streets): an arterial's 22 m and a side
    // street's 14 m. A thin deck over a darker soffit and edge girders, thin concrete parapets
    // with a rail, a pier of two columns on a crossbeam reaching down into the water, a lamp
    // post on each side
    for (k, &width) in [22.0f32, 14.0].iter().enumerate() {
        let mut b = B::new();
        let hw = width * 0.5;
        let (x0, x1) = (-10.5f32, 10.5f32);
        b.bx(Vec3::new(x0, 2.3, -hw), Vec3::new(x1, 2.9, hw), m(SLAB, 0), false);
        b.bx(Vec3::new(x0, 1.5, -hw + 0.6), Vec3::new(x1, 2.3, -hw + 1.6), m(GARAGE, 2), false);
        b.bx(Vec3::new(x0, 1.5, hw - 1.6), Vec3::new(x1, 2.3, hw - 0.6), m(GARAGE, 2), false);
        b.cap(&ccw(vec![Vec2::new(x0, -hw), Vec2::new(x1, -hw), Vec2::new(x1, hw), Vec2::new(x0, hw)]), 2.3, false, m(GARAGE, 2));
        b.quad([Vec3::new(x0, 2.92, -hw + 0.6), Vec3::new(x1, 2.92, -hw + 0.6), Vec3::new(x1, 2.92, hw - 0.6), Vec3::new(x0, 2.92, hw - 0.6)], m(ASPHALT, k as u8));
        for zs in [-1.0f32, 1.0] {
            let z = zs * (hw - 0.15);
            b.bx(Vec3::new(x0, 2.9, z - 0.15), Vec3::new(x1, 3.7, z + 0.15), m(SLAB, 0), false);
            b.bx(Vec3::new(x0, 3.85, z - 0.04), Vec3::new(x1, 3.93, z + 0.04), m(FRAME, 1), false);
            // the lamp: a post, an arm out over the road, the head
            let zi = z - zs * 0.1;
            b.bx(Vec3::new(-0.1, 3.7, zi - 0.1), Vec3::new(0.1, 11.5, zi + 0.1), m(FRAME, 1), false);
            b.bx(Vec3::new(-0.06, 11.3, zi - zs * 2.2), Vec3::new(0.06, 11.45, zi), m(FRAME, 1), false);
            b.bx(Vec3::new(-0.3, 11.1, zi - zs * 2.5 - 0.18), Vec3::new(0.3, 11.35, zi - zs * 2.5 + 0.18), m(POOL, 2), false);
        }
        b.bx(Vec3::new(-0.9, 0.9, -hw + 1.2), Vec3::new(0.9, 1.5, hw - 1.2), m(SLAB, 0), false);
        for zs in [-1.0f32, 1.0] {
            let col: Vec<Vec2> = footprint(1.3, 1.3, 0.65, 3).iter().map(|p| *p + Vec2::new(0.0, zs * (hw * 0.5))).collect();
            b.prism(&col, -14.0, 0.9, m(SLAB, 0), m(SLAB, 0), false);
        }
        push(b, format!("Building_Bridge_{k}"), 11.5, &mut meshes, &mut archs);
    }
    (meshes, archs)
}

/// A boat: hull (white gelcoat, a painted boot stripe, a teak deck), and by `kind` a console
/// with a T-top (0), a cabin with a windscreen (1), or two decks of a motor yacht with dark
/// glazing and a flybridge (2). Returns its height above the waterline.
/// A car: the body (a box with its ends rounded off), the glasshouse on it (pillars of paint,
/// windows of dark glass), four wheels. kind: 0 car, 1 SUV, 2 pickup (an open bed), 3 van,
/// 4 bus (glass all along).
fn car(b: &mut B, len: f32, wid: f32, hgt: f32, kind: u32) {
    let (hl, hw) = (len * 0.5, wid * 0.5);
    let sill = 0.32;
    let belt = if kind == 4 { 1.1 } else { hgt * 0.58 };
    let body = ccw(vec![
        Vec2::new(-hl, -hw + 0.12),
        Vec2::new(-hl + 0.15, -hw),
        Vec2::new(hl - 0.3, -hw),
        Vec2::new(hl, -hw + 0.2),
        Vec2::new(hl, hw - 0.2),
        Vec2::new(hl - 0.3, hw),
        Vec2::new(-hl + 0.15, hw),
        Vec2::new(-hl, hw - 0.12),
    ]);
    b.prism(&body, sill, belt, m(CARPAINT, 0), m(CARPAINT, 0), true);
    // the glasshouse: from the windscreen back to the rear window (a pickup's ends at its bed)
    let (x0, x1) = match kind {
        0 => (-hl * 0.55, hl * 0.25),
        1 => (-hl * 0.8, hl * 0.35),
        2 => (-hl * 0.1, hl * 0.4),
        3 => (-hl * 0.95, hl * 0.55),
        _ => (-hl * 0.97, hl * 0.97),
    };
    let top = hgt - 0.05;
    let inset = if kind == 4 { 0.02 } else { 0.12 };
    // tapered: the windscreen and the rear window raked, the sides leaning in (tumblehome), so
    // the glass shows from above as it does on a real car; the roof is paint
    let (rake_f, rake_r, tumble) = match kind {
        0 => (1.0, 0.7, 0.24),
        1 => (0.8, 0.35, 0.2),
        2 => (0.65, 0.1, 0.18),
        3 => (0.95, 0.12, 0.15),
        _ => (0.12, 0.05, 0.02),
    };
    let (zb, zt) = (hw - inset, hw - inset - tumble);
    let (xa, xb) = (x0 + rake_r, x1 - rake_f);
    let v = |x: f32, y: f32, z: f32| Vec3::new(x, y, z);
    let glass = m(GLASS, 6);
    b.quad([v(x1, belt, -zb), v(x1, belt, zb), v(xb, top, zt), v(xb, top, -zt)], glass); // windscreen
    b.quad([v(x0, belt, zb), v(x0, belt, -zb), v(xa, top, -zt), v(xa, top, zt)], glass); // rear window
    b.quad([v(x0, belt, -zb), v(x1, belt, -zb), v(xb, top, -zt), v(xa, top, -zt)], glass); // sides
    b.quad([v(x1, belt, zb), v(x0, belt, zb), v(xa, top, zt), v(xb, top, zt)], glass);
    b.quad([v(xa, top, -zt), v(xb, top, -zt), v(xb, top, zt), v(xa, top, zt)], m(CARPAINT, 0)); // roof
    // pillars between the side windows (B pillar), paint
    let xm = 0.5 * (xa + xb);
    for zs in [-1.0f32, 1.0] {
        b.bx(Vec3::new(xm - 0.07, belt, zs * zb - 0.02), Vec3::new(xm + 0.07, top - 0.02, zs * zb + 0.02), m(CARPAINT, 0), false);
    }
    // headlights and taillights
    let ly = belt - 0.22;
    for zs in [-1.0f32, 1.0] {
        let z = zs * (hw - 0.4);
        b.bx(Vec3::new(hl - 0.05, ly, z - 0.25), Vec3::new(hl + 0.02, ly + 0.14, z + 0.25), m(POOL, 3), false);
        b.bx(Vec3::new(-hl - 0.02, ly, z - 0.22), Vec3::new(-hl + 0.05, ly + 0.16, z + 0.22), m(POOL, 4), false);
    }
    // bumpers, dark, front and back
    b.bx(Vec3::new(hl - 0.1, sill - 0.05, -hw + 0.15), Vec3::new(hl + 0.05, sill + 0.18, hw - 0.15), m(DARK, 0), false);
    b.bx(Vec3::new(-hl - 0.05, sill - 0.05, -hw + 0.15), Vec3::new(-hl + 0.1, sill + 0.18, hw - 0.15), m(DARK, 0), false);
    if kind == 2 {
        // the bed: low walls round an open box
        b.bx(Vec3::new(-hl, belt - 0.05, -hw), Vec3::new(x0 - 0.1, belt + 0.05, hw), m(DARK, 0), false);
    }
    // wheels: tyres standing a little proud of the body, a silver rim on the outer face, a
    // dark arch cut round each in the body side
    let r = if kind == 4 { 0.5 } else { 0.34 };
    for &x in &[-hl * 0.68, hl * 0.62] {
        for zs in [-1.0f32, 1.0] {
            let z = zs * (hw - 0.12);
            b.bx(Vec3::new(x - r, 0.0, z - 0.13), Vec3::new(x + r, 2.0 * r, z + 0.13), m(DARK, 0), false);
            let face = zs * (hw + 0.015);
            b.bx(Vec3::new(x - r * 0.6, r * 0.4, face - 0.01), Vec3::new(x + r * 0.6, r * 1.6, face + 0.01), m(FRAME, 1), false);
            // the arch: a dark band over the tyre on the side
            b.bx(Vec3::new(x - r - 0.08, 2.0 * r - 0.02, zs * hw - 0.012), Vec3::new(x + r + 0.08, 2.0 * r + 0.1, zs * hw + 0.012), m(DARK, 0), false);
        }
    }
    if kind < 4 {
        // door mirrors, a dark trim line round the glasshouse's foot, the grille
        for zs in [-1.0f32, 1.0] {
            b.bx(Vec3::new(x1 - 0.25, belt + 0.02, zs * hw - 0.1), Vec3::new(x1 - 0.05, belt + 0.16, zs * (hw + 0.16)), m(CARPAINT, 0), false);
        }
        b.prism(&g_trim(x0 - 0.02, x1 + 0.02, hw - inset + 0.03), belt - 0.03, belt + 0.03, m(DARK, 0), m(DARK, 0), false);
        b.bx(Vec3::new(hl - 0.03, sill + 0.2, -hw * 0.45), Vec3::new(hl + 0.02, belt - 0.3, hw * 0.45), m(DARK, 0), false);
    }
}

fn g_trim(x0: f32, x1: f32, z: f32) -> Vec<Vec2> {
    ccw(vec![Vec2::new(x0, -z), Vec2::new(x1, -z), Vec2::new(x1, z), Vec2::new(x0, z)])
}

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
    if name.contains("Building_Pool") {
        Class::Pool
    } else if name.contains("StreetLight") || name.contains("Signal") {
        Class::Street
    } else if name.contains("Landmark") {
        Class::Landmark
    } else if name.contains("Highway") {
        Class::Highway
    } else if name.contains("Boat_Wake") {
        Class::Underway
    } else if name.contains("Boat") {
        Class::Boat
    } else if name.contains("Dock") {
        Class::Dock
    } else if name.contains("Car_") {
        Class::Car
    } else if name.contains("Bridge") {
        Class::Bridge
    } else if name.contains("Tower") || name.contains("Residential") {
        Class::Tower
    } else if name.contains("Office") || name.contains("Apartments") || name.contains("Med_") || name.contains("Condo") {
        Class::MidRise
    } else if name.contains("House") {
        Class::House
    } else {
        Class::LowRise
    }
}
