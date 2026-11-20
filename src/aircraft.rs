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
    /// one mesh per part, in the part's own frame (about its pivot); then the rocket
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
    // (after the parts: the rocket of --battle)
    meshes.push(rocket(&mut materials));
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
    pub rocket: bool,
}

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
