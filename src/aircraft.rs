//! The aircraft of `--test-flight`: the model tools/make_f16.py writes (assets/aircraft/f16c.acm)
//! and its painted skin. The model's frame is the one the renderer flies it in: x right, y up,
//! z aft, origin at the centre of gravity, metres. Its parts are the body, the glass, and each
//! moving surface about its own pivot with its hinge axis.

use crate::asset::MeshData;
use crate::gpu::{Gpu, Image};
use anyhow::{bail, Context, Result};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Affine3A, Quat, Vec3};
use std::path::{Path, PathBuf};

pub struct Part {
    pub name: String,
    pub pivot: Vec3,
    /// hinge axis (unit, model frame); zero for a fixed part
    pub axis: Vec3,
    /// the canopy and the head-up display: seen through (aircraft.rahit)
    pub glass: bool,
}

/// A material as aircraft.rchit reads it (binding 37).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MaterialGpu {
    /// linear colour, opacity
    pub base: [f32; 4],
    /// metal, roughness, 1 = the painted skin's textures, 0
    pub mre: [f32; 4],
    pub emit: [f32; 4],
}

pub struct Model {
    pub parts: Vec<Part>,
    /// one mesh per part, in the part's own frame (about its pivot); then the rocket and the
    /// other things of --battle, in the order of Craft::kind
    pub meshes: Vec<MeshData>,
    pub materials: Vec<MaterialGpu>,
    pub dir: PathBuf,
}

struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        if self.0.len() < n {
            bail!("the aircraft file ends early");
        }
        let (a, b) = self.0.split_at(n);
        self.0 = b;
        Ok(a)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f32s(&mut self, n: usize) -> Result<Vec<f32>> {
        Ok(self.take(4 * n)?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }
    fn text(&mut self) -> Result<String> {
        let n = self.take(1)?[0] as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
}

pub fn load(path: &Path) -> Result<Model> {
    let bytes = std::fs::read(path).with_context(|| format!("{} (write it with tools/make_f16.py)", path.display()))?;
    let mut r = Reader(&bytes);
    if r.take(4)? != b"SIAC" || r.u32()? != 1 {
        bail!("{}: not an aircraft model this build reads", path.display());
    }
    let mut materials = Vec::new();
    for _ in 0..r.u32()? {
        let _name = r.text()?;
        let v = r.f32s(9)?;
        let flags = r.u32()?;
        materials.push(MaterialGpu { base: [v[0], v[1], v[2], v[3]], mre: [v[4], v[5], (flags & 1) as f32, 0.0], emit: [v[6], v[7], v[8], 0.0] });
    }
    let (mut parts, mut meshes) = (Vec::new(), Vec::new());
    for _ in 0..r.u32()? {
        let name = r.text()?;
        let v = r.f32s(6)?;
        let flags = r.u32()?;
        let n = r.u32()? as usize;
        let positions = r.f32s(9 * n)?;
        let normals = r.f32s(9 * n)?;
        let uvs = r.f32s(6 * n)?;
        let materials = r.take((n + 3) & !3)?.to_vec();
        meshes.push(MeshData { name: format!("aircraft {name}"), positions, normals, uvs, indices: (0..3 * n as u32).collect(), materials, grid: 0 });
        parts.push(Part { name, pivot: Vec3::new(v[0], v[1], v[2]), axis: Vec3::new(v[3], v[4], v[5]), glass: flags & 1 != 0 });
    }
    // (after the parts: the rocket of --battle, and the rest of what it draws)
    meshes.push(rocket(&mut materials));
    meshes.extend(extras(&mut materials));
    Ok(Model { parts, meshes, materials, dir: path.parent().unwrap_or(Path::new(".")).to_path_buf() })
}

/// Half the size (a box filter): the roughness and normal maps do not need the colour's detail.
fn halve(w: u32, h: u32, px: &[u8]) -> (u32, u32, Vec<u8>) {
    let (w2, h2) = (w / 2, h / 2);
    let mut out = vec![0u8; (w2 * h2 * 4) as usize];
    for y in 0..h2 as usize {
        for x in 0..w2 as usize {
            for c in 0..4 {
                let at = |dx: usize, dy: usize| px[((2 * y + dy) * w as usize + 2 * x + dx) * 4 + c] as u32;
                out[(y * w2 as usize + x) * 4 + c] = ((at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1) + 2) / 4) as u8;
            }
        }
    }
    (w2, h2, out)
}

/// The skin's three maps (binding 36): colour, occlusion / roughness / metal, normal.
pub fn load_textures(gpu: &Gpu, dir: &Path) -> Result<Vec<Image>> {
    let jobs = [("f16c_albedo.png", vk::Format::R8G8B8A8_SRGB, false), ("f16c_orm.png", vk::Format::R8G8B8A8_UNORM, true), ("f16c_normal.png", vk::Format::R8G8B8A8_UNORM, true)];
    let decoded: Vec<Result<(u32, u32, Vec<u8>)>> = std::thread::scope(|s| {
        let hs: Vec<_> = jobs
            .iter()
            .map(|(name, _, half)| {
                let p = dir.join(name);
                s.spawn(move || {
                    let (w, h, px) = crate::textures::decode_rgba(&p).with_context(|| p.display().to_string())?;
                    Ok(if *half { halve(w, h, &px) } else { (w, h, px) })
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut out = Vec::new();
    for ((name, format, _), d) in jobs.iter().zip(decoded) {
        let (w, h, px) = d?;
        out.push(crate::textures::upload_with_mips(gpu, w, h, &px, *format, name)?);
    }
    Ok(out)
}

/// Where the aircraft is this frame, in the renderer's frame (set by main.rs, drawn by the renderer).
#[derive(Clone, Copy)]
pub struct Pose {
    /// model -> renderer frame
    pub xform: Affine3A,
    /// a point of it this frame -> where that point was last frame (the renderer's frame):
    /// motion vectors
    pub back: Affine3A,
    /// hinge angles (rad): leading-edge flaps (down positive), flaperons and tailplanes right /
    /// left (trailing edge down positive), rudder (trailing edge right positive)
    pub lef: f32,
    pub flaperon: [f32; 2],
    pub tail: [f32; 2],
    pub rudder: f32,
}

impl Pose {
    /// A part's own transform (about its pivot and hinge), model frame.
    pub fn part(&self, p: &Part) -> Affine3A {
        let angle = match p.name.as_str() {
            // (about an axis pointing outboard, a positive turn takes what is aft of the hinge down:
            // trailing edges down; a leading-edge flap is ahead of its hinge)
            "le_flap_R" => -self.lef,
            "le_flap_L" => self.lef,
            "flaperon_R" => self.flaperon[0],
            "flaperon_L" => -self.flaperon[1],
            "tailplane_R" => self.tail[0],
            "tailplane_L" => -self.tail[1],
            "rudder" => self.rudder,
            _ => 0.0,
        };
        let rot = if p.axis.length_squared() > 0.5 { Quat::from_axis_angle(p.axis, angle) } else { Quat::IDENTITY };
        Affine3A::from_rotation_translation(rot, p.pivot)
    }
}

/// One more thing in the air in `--battle` (battle.rs), drawn like the aircraft flown: another
/// aircraft in its squadron's paint (shaders/livery.glsl), or a rocket.
#[derive(Clone, Copy)]
pub struct Craft {
    pub pose: Pose,
    pub livery: u32,
    /// what it is: an aircraft (0), or one of the meshes after the aircraft's parts
    pub kind: u8,
    /// its squadron's colour, on everything of it that is not the aircraft's painted skin
    pub tint: [f32; 3],
}

pub const KIND_JET: u8 = 0;
pub const KIND_ROCKET: u8 = 1;
pub const KIND_DRONE: u8 = 2;
pub const KIND_LAUNCHER: u8 = 3;
pub const KIND_GUN: u8 = 4;
/// the pilot's own, on foot: the gun's body, its barrels (which turn), the parachute
pub const KIND_MINIGUN: u8 = 5;
pub const KIND_BARRELS: u8 = 6;
pub const KIND_CANOPY: u8 = 7;

/// A piece of smoke or fire as fx.rint reads it (shaders/fx.glsl): a tube of Gaussian section
/// from `a` to `b` in the renderer's frame (a ball where they are the same).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FxGpu {
    /// one end, its radius there (m)
    pub a: [f32; 4],
    pub b: [f32; 4],
    /// a smoke's colour or a fire's light; density (1 / m)
    pub colour: [f32; 4],
    /// 0 smoke / 1 fire, a number of its own, where along its trail `a` is (m), how ragged
    pub kind: [f32; 4],
}

/// A rocket for `--battle`: three metres of tube with a pointed nose and four fins, in the
/// aircraft's frame (its nose towards -z). Its two materials are added to the aircraft's.
pub fn rocket(materials: &mut Vec<MaterialGpu>) -> MeshData {
    let first = materials.len() as u8;
    materials.push(MaterialGpu { base: [0.72, 0.73, 0.74, 1.0], mre: [0.0, 0.42, 0.0, 0.0], emit: [0.0; 4] });
    materials.push(MaterialGpu { base: [0.10, 0.11, 0.12, 1.0], mre: [0.6, 0.35, 0.0, 0.0], emit: [0.0; 4] });
    let (mut positions, mut normals, mut mats) = (Vec::new(), Vec::new(), Vec::new());
    let mut tri = |p: [Vec3; 3], n: [Vec3; 3], m: u8| {
        for k in 0..3 {
            positions.extend_from_slice(&p[k].to_array());
            normals.extend_from_slice(&n[k].to_array());
        }
        mats.push(first + m);
    };
    // the body: rings of (z, radius) from the nose's tip to the tail
    const SEG: usize = 12;
    let rings = [(-1.5f32, 0.0f32), (-1.38, 0.035), (-1.2, 0.064), (1.45, 0.064), (1.5, 0.05), (1.5, 0.0)];
    let at = |k: usize, i: usize| {
        let a = i as f32 / SEG as f32 * std::f32::consts::TAU;
        Vec3::new(rings[k].1 * a.cos(), rings[k].1 * a.sin(), rings[k].0)
    };
    let out = |k: usize, i: usize| {
        let a = i as f32 / SEG as f32 * std::f32::consts::TAU;
        let (k0, k1) = (k.saturating_sub(1), (k + 1).min(rings.len() - 1));
        let (dz, dr) = (rings[k1].0 - rings[k0].0, rings[k1].1 - rings[k0].1);
        (Vec3::new(a.cos(), a.sin(), 0.0) * dz.abs().max(1e-3) - Vec3::Z * dr * dz.signum()).normalize_or(Vec3::Z)
    };
    for k in 0..rings.len() - 1 {
        for i in 0..SEG {
            let (a, b, c, d) = (at(k, i), at(k, i + 1), at(k + 1, i + 1), at(k + 1, i));
            let (na, nb, nc, nd) = (out(k, i), out(k, i + 1), out(k + 1, i + 1), out(k + 1, i));
            let m = if k == 0 { 1 } else { 0 };
            tri([a, b, c], [na, nb, nc], m);
            tri([a, c, d], [na, nc, nd], m);
        }
    }
    // four fins at the tail, four small ones behind the nose
    for (z0, z1, span) in [(0.95f32, 1.45f32, 0.26f32), (-0.95, -0.72, 0.15)] {
        for f in 0..4 {
            let a = (f as f32 + 0.5) * std::f32::consts::FRAC_PI_2;
            let (r, n) = (Vec3::new(a.cos(), a.sin(), 0.0), Vec3::new(-a.sin(), a.cos(), 0.0));
            let (p0, p1, p2, p3) = (r * 0.06 + Vec3::Z * z0, r * 0.06 + Vec3::Z * z1, r * span + Vec3::Z * z1, r * span + Vec3::Z * (z0 + 0.6 * (z1 - z0)));
            tri([p0, p1, p2], [n; 3], 1);
            tri([p0, p2, p3], [n; 3], 1);
        }
    }
    let n = mats.len();
    mats.resize((n + 3) & !3, 0);
    MeshData { name: "rocket".into(), positions, normals, uvs: vec![0.0; 6 * n], indices: (0..3 * n as u32).collect(), materials: mats, grid: 0 }
}
/// Triangles being put together into a mesh (the made-up things of --battle).
struct Shape {
    positions: Vec<f32>,
    normals: Vec<f32>,
    mats: Vec<u8>,
}

impl Shape {
    fn new() -> Self {
        Shape { positions: Vec::new(), normals: Vec::new(), mats: Vec::new() }
    }

    fn tri(&mut self, p: [Vec3; 3], n: [Vec3; 3], m: u8) {
        for k in 0..3 {
            self.positions.extend_from_slice(&p[k].to_array());
            self.normals.extend_from_slice(&n[k].to_array());
        }
        self.mats.push(m);
    }

    /// A flat four-sided face.
    fn quad(&mut self, p: [Vec3; 4], m: u8) {
        let n = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or(Vec3::Y);
        self.tri([p[0], p[1], p[2]], [n; 3], m);
        self.tri([p[0], p[2], p[3]], [n; 3], m);
    }

    /// A box of half size `h` about `c`, tipped up by `pitch` (its front, -z, rising) about the
    /// x axis through `pivot`.
    fn boxed(&mut self, c: Vec3, h: Vec3, pivot: Vec3, pitch: f32, m: u8) {
        let turn = Quat::from_rotation_x(pitch);
        let at = |q: Vec3| pivot + turn * (c + q * h - pivot);
        for (a, b, d) in [(Vec3::X, Vec3::Y, Vec3::Z), (Vec3::Y, Vec3::Z, Vec3::X), (Vec3::Z, Vec3::X, Vec3::Y)] {
            for s in [-1.0f32, 1.0] {
                let f = |u: f32, v: f32| at(a * s + b * u + d * v);
                self.quad([f(-1.0, -1.0), f(1.0, -1.0), f(1.0, 1.0), f(-1.0, 1.0)], m);
            }
        }
    }

    /// A round tube from `a` to `b`, its radius `ra` at one end and `rb` at the other, closed.
    fn tube(&mut self, a: Vec3, b: Vec3, ra: f32, rb: f32, seg: usize, m: u8) {
        let u = (b - a).normalize_or(Vec3::Z);
        let v = u.any_orthonormal_vector();
        let w = u.cross(v);
        let ring = |i: usize| {
            let t = i as f32 / seg as f32 * std::f32::consts::TAU;
            v * t.cos() + w * t.sin()
        };
        for i in 0..seg {
            let (n0, n1) = (ring(i), ring(i + 1));
            let (p0, p1, p2, p3) = (a + n0 * ra, a + n1 * ra, b + n1 * rb, b + n0 * rb);
            self.tri([p0, p1, p2], [n0, n1, n1], m);
            self.tri([p0, p2, p3], [n0, n1, n0], m);
            self.tri([a, p1, p0], [-u; 3], m);
            self.tri([b, p3, p2], [u; 3], m);
        }
    }

    fn done(mut self, name: &str) -> MeshData {
        let n = self.mats.len();
        self.mats.resize((n + 3) & !3, 0);
        MeshData { name: name.into(), positions: self.positions, normals: self.normals, uvs: vec![0.0; 6 * n], indices: (0..3 * n as u32).collect(), materials: self.mats, grid: 0 }
    }
}

/// The rest of what `--battle` draws, in the order of Craft::kind from KIND_DRONE: a drone, a
/// rocket launcher and an anti-aircraft gun on their trucks, the gun the pilot holds on foot (its
/// body, then its barrels about the axis they turn on) and the parachute. All made up, all in
/// the aircraft's frame (x right, y up, z aft: the front is -z), metres; their materials are
/// added to the aircraft's. What is painted takes its squadron's colour (Craft::tint).
pub fn extras(materials: &mut Vec<MaterialGpu>) -> Vec<MeshData> {
    let first = materials.len() as u8;
    let flat = |c: [f32; 3], metal: f32, rough: f32| MaterialGpu { base: [c[0], c[1], c[2], 1.0], mre: [metal, rough, 0.0, 0.0], emit: [0.0; 4] };
    materials.extend([
        flat([0.85, 0.85, 0.85], 0.0, 0.6),     // paint (tinted)
        flat([0.035, 0.035, 0.035], 0.2, 0.55), // tyres, glass, muzzles
        flat([0.20, 0.21, 0.23], 0.85, 0.38),   // gun metal
        flat([0.028, 0.028, 0.03], 0.0, 0.5),   // black plastic
        flat([0.55, 0.40, 0.12], 0.9, 0.3),     // brass
        // the canopy's orange and its white: thin cloth, which the daylight comes through
        MaterialGpu { base: [0.85, 0.20, 0.02, 1.0], mre: [0.0, 0.85, 0.0, 0.0], emit: [0.42, 0.095, 0.01, 0.0] },
        MaterialGpu { base: [0.82, 0.82, 0.80, 1.0], mre: [0.0, 0.85, 0.0, 0.0], emit: [0.40, 0.40, 0.385, 0.0] },
    ]);
    let (paint, dark, steel, plastic, brass, orange, white) = (first, first + 1, first + 2, first + 3, first + 4, first + 5, first + 6);
    let v = Vec3::new;
    let o = Vec3::ZERO;

    // ---- a drone: a small flying wing with a pusher propeller, 4.5 m across
    let mut d = Shape::new();
    let (nose, tip_r, tip_l, tail) = (v(0.0, 0.07, -1.7), v(2.25, 0.03, 1.25), v(-2.25, 0.03, 1.25), v(0.0, 0.05, 1.0));
    for side in [1.0f32, -1.0] {
        let up = Vec3::Y * side;
        d.tri([nose + up * 0.04, tip_r, tail + up * 0.04], [up; 3], paint);
        d.tri([nose + up * 0.04, tail + up * 0.04, tip_l], [up; 3], paint);
    }
    d.tube(v(0.0, 0.08, -1.9), v(0.0, 0.10, -0.9), 0.05, 0.22, 10, paint);
    d.tube(v(0.0, 0.10, -0.9), v(0.0, 0.10, 0.9), 0.22, 0.18, 10, paint);
    d.tube(v(0.0, 0.10, 0.9), v(0.0, 0.10, 1.25), 0.18, 0.07, 10, dark);
    d.boxed(v(0.0, 0.10, 1.3), v(0.5, 0.035, 0.012), o, 0.0, dark);
    d.tube(v(0.0, 0.0, -1.45), v(0.0, -0.14, -1.45), 0.09, 0.09, 8, dark); // its eye
    for x in [2.2f32, -2.2] {
        d.quad([v(x, 0.03, 0.55), v(x, 0.03, 1.3), v(x, 0.62, 1.35), v(x, 0.45, 0.95)], paint);
    }
    let drone = d.done("drone");

    // ---- a truck: what both the launcher and the gun stand on
    let truck = |s: &mut Shape| {
        s.boxed(v(0.0, 1.0, 0.3), v(1.25, 0.28, 3.7), o, 0.0, paint);
        s.boxed(v(0.0, 1.85, -2.75), v(1.2, 0.6, 0.85), o, 0.0, paint);
        s.boxed(v(0.0, 2.05, -3.61), v(1.0, 0.25, 0.02), o, 0.0, dark);
        for z in [-2.7f32, -0.6, 1.6, 3.0] {
            for x in [1.12f32, -1.12] {
                s.tube(v(x - 0.2, 0.55, z), v(x + 0.2, 0.55, z), 0.55, 0.55, 12, dark);
            }
        }
    };

    // ---- a rocket launcher: a pack of tubes on the bed of the truck, tipped up
    let mut l = Shape::new();
    truck(&mut l);
    let hinge = v(0.0, 1.5, 3.2);
    l.boxed(v(0.0, 2.0, 0.9), v(1.05, 0.62, 2.5), hinge, 0.62, paint);
    l.boxed(v(0.0, 2.0, -1.62), v(0.95, 0.52, 0.03), hinge, 0.62, dark);
    l.boxed(v(0.0, 1.45, 1.6), v(0.25, 0.55, 0.25), o, 0.0, steel);
    let launcher = l.done("launcher");

    // ---- an anti-aircraft gun: a turret with two long barrels
    let mut g = Shape::new();
    truck(&mut g);
    g.tube(v(0.0, 1.28, 1.2), v(0.0, 1.6, 1.2), 1.15, 1.05, 16, steel);
    g.boxed(v(0.0, 2.15, 1.2), v(0.8, 0.55, 0.95), o, 0.0, paint);
    let trunnion = v(0.0, 2.3, 0.7);
    for x in [0.42f32, -0.42] {
        let turn = Quat::from_rotation_x(0.85);
        let at = |z: f32| trunnion + turn * (v(x, 2.3, z) - trunnion);
        g.tube(at(0.6), at(-3.2), 0.085, 0.07, 10, steel);
        g.tube(at(-3.2), at(-3.6), 0.11, 0.11, 10, dark);
    }
    let gun = g.done("anti-aircraft gun");

    // ---- the gun the pilot holds on foot: six barrels that turn. Its body:
    let mut b = Shape::new();
    // the housing the barrels turn in, with its bands
    b.tube(v(0.0, 0.0, -0.27), v(0.0, 0.0, -0.05), 0.066, 0.070, 24, steel);
    for z in [-0.26f32, -0.19, -0.12] {
        b.tube(v(0.0, 0.0, z), v(0.0, 0.0, z + 0.022), 0.074, 0.074, 24, plastic);
    }
    // the receiver behind it, the motor on its right shoulder, the cover plate with its bolts
    b.boxed(v(0.0, 0.0, 0.05), v(0.062, 0.066, 0.11), o, 0.0, steel);
    b.tube(v(0.0, 0.0, -0.06), v(0.0, 0.0, 0.17), 0.058, 0.058, 20, steel);
    b.tube(v(0.062, 0.062, -0.02), v(0.062, 0.062, 0.20), 0.034, 0.034, 16, plastic);
    b.tube(v(0.062, 0.062, 0.20), v(0.062, 0.062, 0.23), 0.026, 0.02, 12, steel);
    b.boxed(v(-0.064, 0.0, 0.05), v(0.004, 0.045, 0.08), o, 0.0, plastic);
    for (y, z) in [(0.035f32, -0.02f32), (0.035, 0.12), (-0.035, -0.02), (-0.035, 0.12)] {
        b.tube(v(-0.066, y, z), v(-0.072, y, z), 0.006, 0.006, 8, steel);
    }
    // the carrying handle over the top
    b.tube(v(0.0, 0.066, -0.10), v(0.0, 0.125, -0.08), 0.008, 0.008, 8, steel);
    b.tube(v(0.0, 0.066, 0.12), v(0.0, 0.125, 0.10), 0.008, 0.008, 8, steel);
    b.tube(v(0.0, 0.125, -0.085), v(0.0, 0.125, 0.105), 0.012, 0.012, 10, plastic);
    // the feeder on the right, and the belt of rounds coming up into it from below
    b.boxed(v(0.085, -0.01, 0.03), v(0.024, 0.05, 0.07), o, 0.0, steel);
    for k in 0..9 {
        let t = k as f32 / 8.0;
        let at = v(0.10 + 0.13 * t, -0.07 - 0.36 * t * t, 0.03 + 0.05 * t);
        b.tube(at + v(0.0, 0.0, -0.045), at + v(0.0, 0.0, 0.03), 0.0085, 0.0085, 8, brass);
        b.tube(at + v(0.0, 0.0, -0.07), at + v(0.0, 0.0, -0.045), 0.003, 0.0085, 8, steel);
        b.boxed(at + v(0.0, 0.0, -0.005), v(0.011, 0.004, 0.012), at, 0.0, plastic);
    }
    // the two grips at the back, on their yoke, and the trigger bar between them
    b.tube(v(0.0, 0.0, 0.16), v(0.0, 0.0, 0.215), 0.03, 0.022, 12, steel);
    b.tube(v(-0.085, 0.0, 0.215), v(0.085, 0.0, 0.215), 0.012, 0.012, 10, steel);
    for x in [0.085f32, -0.085] {
        b.tube(v(x, 0.06, 0.215), v(x, -0.07, 0.235), 0.019, 0.021, 12, plastic);
    }
    b.tube(v(-0.085, 0.045, 0.24), v(0.085, 0.045, 0.24), 0.007, 0.007, 8, steel);
    let body = b.done("minigun");
    // and its barrels, about the z axis
    let mut r = Shape::new();
    for k in 0..6 {
        let a = k as f32 / 6.0 * std::f32::consts::TAU;
        let c = v(0.040 * a.cos(), 0.040 * a.sin(), 0.0);
        r.tube(c + v(0.0, 0.0, -0.25), c + v(0.0, 0.0, -0.97), 0.0125, 0.0115, 12, steel);
        r.tube(c + v(0.0, 0.0, -0.97), c + v(0.0, 0.0, -0.975), 0.0075, 0.0075, 8, dark);
    }
    for z in [-0.40f32, -0.66, -0.93] {
        r.tube(v(0.0, 0.0, z), v(0.0, 0.0, z - 0.018), 0.060, 0.060, 24, steel); // the clamps
        r.tube(v(0.0, 0.0, z - 0.018), v(0.0, 0.0, z - 0.024), 0.060, 0.052, 24, plastic);
    }
    r.tube(v(0.0, 0.0, -0.25), v(0.0, 0.0, -0.94), 0.017, 0.017, 10, steel); // the spindle
    let barrels = r.done("minigun barrels");

    // ---- the parachute: a round canopy in orange and white gores, 8 m across, its rim 6 m
    // above the eye (the origin), and the lines down from it
    let mut c = Shape::new();
    const GORES: usize = 20;
    const RINGS: usize = 7;
    let dome = |i: usize, k: usize| {
        let a = i as f32 / GORES as f32 * std::f32::consts::TAU;
        let e = k as f32 / RINGS as f32 * 1.42; // from the rim up to the vent
        v(4.0 * e.cos() * a.cos(), 6.0 + 2.6 * e.sin(), 4.0 * e.cos() * a.sin())
    };
    let out = |p: Vec3| v(p.x / 16.0, (p.y - 6.0) / 6.76, p.z / 16.0).normalize_or(Vec3::Y);
    for i in 0..GORES {
        let m = if i % 2 == 0 { orange } else { white };
        for k in 0..RINGS {
            let (p0, p1, p2, p3) = (dome(i, k), dome(i + 1, k), dome(i + 1, k + 1), dome(i, k + 1));
            c.tri([p0, p1, p2], [out(p0), out(p1), out(p2)], m);
            c.tri([p0, p2, p3], [out(p0), out(p2), out(p3)], m);
        }
        // the line from the rim to the risers, well above the shoulders
        let rim = dome(i, 0);
        let riser = v(if rim.x > 0.0 { 0.32 } else { -0.32 }, 0.95, 0.12);
        let w = v(-rim.z, 0.0, rim.x).normalize_or(Vec3::X) * 0.004;
        c.quad([rim - w, rim + w, riser + w, riser - w], white);
    }
    for x in [0.32f32, -0.32] {
        c.boxed(v(x, 0.72, 0.12), v(0.011, 0.23, 0.002), o, 0.0, plastic); // the risers
    }
    let canopy = c.done("parachute");

    vec![drone, launcher, gun, body, barrels, canopy]
}
