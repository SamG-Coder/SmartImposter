//! Loader for `.mtree` files produced by `tools/usd_to_mtree.py`.

use anyhow::{bail, Context, Result};
use glam::{Affine3A, Vec3, Vec3A};

pub struct MeshData {
    pub name: String,
    pub positions: Vec<f32>, // xyz
    pub normals: Vec<f32>,   // xyz
    pub uvs: Vec<f32>,       // uv, image space (top-left origin)
    pub indices: Vec<u32>,   // 3 per triangle
    pub materials: Vec<u8>,  // 1 per triangle, padded to a multiple of 4
    /// A regular grid of `grid` x `grid` vertices (the terrain: world.rs World::mesh), two
    /// triangles per cell in rows, all of one material: the shaders work the triangles out
    /// (rt.glsl meshTri), so the renderer keeps no index or material buffer. 0 = any mesh.
    pub grid: u32,
}

impl MeshData {
    pub fn vertex_count(&self) -> u32 {
        (self.positions.len() / 3) as u32
    }
    pub fn tri_count(&self) -> u32 {
        (self.indices.len() / 3) as u32
    }
    /// The box of the vertices (min, max).
    pub fn bounds(&self) -> (Vec3, Vec3) {
        self.positions
            .chunks_exact(3)
            .fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(lo, hi), v| (lo.min(Vec3::new(v[0], v[1], v[2])), hi.max(Vec3::new(v[0], v[1], v[2]))))
    }
}

pub struct TreePart {
    pub mesh: u32,
    pub xform: Affine3A,
}

pub struct TreeAsset {
    pub name: String,
    pub bmin: Vec3,
    pub bmax: Vec3,
    pub parts: Vec<TreePart>,
}

impl TreeAsset {
    pub fn expanded_tris(&self, meshes: &[MeshData]) -> u64 {
        self.parts.iter().map(|p| meshes[p.mesh as usize].tri_count() as u64).sum()
    }

    /// How far the plant's geometry reaches from the centre of its bounding box: (the largest
    /// horizontal distance from the vertical axis through it, the largest distance). Seen from
    /// any direction, the plant is at most the first wide and twice the second tall around the
    /// centre: the impostor frame it needs (renderer.rs frame_res). `bounds`: every mesh's box
    /// (MeshData::bounds).
    pub fn reach(&self, meshes: &[MeshData], bounds: &[(Vec3, Vec3)]) -> (f32, f32) {
        let c = (self.bmin + self.bmax) * 0.5;
        let across = |p: Vec3| (p.x - c.x) * (p.x - c.x) + (p.z - c.z) * (p.z - c.z);
        let all = |p: Vec3| (p - c).length_squared();
        let corners = |p: &TreePart| -> [Vec3; 8] {
            let (lo, hi) = bounds[p.mesh as usize];
            std::array::from_fn(|k| p.xform.transform_point3(Vec3::new(if k & 1 == 0 { lo.x } else { hi.x }, if k & 2 == 0 { lo.y } else { hi.y }, if k & 4 == 0 { lo.z } else { hi.z })))
        };
        // (a part's box bounds it from above: its vertices are only walked while that could still
        // beat the best found, the outermost twigs)
        let mut best = (0.0f32, 0.0f32);
        let mut parts: Vec<(f32, f32, &TreePart)> =
            self.parts.iter().map(|p| (corners(p).into_iter().map(across).fold(0.0f32, f32::max), corners(p).into_iter().map(all).fold(0.0f32, f32::max), p)).collect();
        parts.sort_by(|a, b| (b.0.max(b.1)).total_cmp(&a.0.max(a.1)));
        for (ub_across, ub_all, p) in parts {
            if ub_across <= best.0 && ub_all <= best.1 {
                continue;
            }
            for v in meshes[p.mesh as usize].positions.chunks_exact(3) {
                let q = p.xform.transform_point3(Vec3::new(v[0], v[1], v[2]));
                best = (best.0.max(across(q)), best.1.max(all(q)));
            }
        }
        (best.0.sqrt(), best.1.sqrt())
    }
}

pub struct TreePack {
    pub meshes: Vec<MeshData>,
    pub trees: Vec<TreeAsset>,
}

struct Reader<'a> {
    d: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.p + n > self.d.len() {
            bail!("truncated file");
        }
        let s = &self.d[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u32()? as usize;
        Ok(String::from_utf8_lossy(self.bytes(n)?).into_owned())
    }
    fn f32s(&mut self, n: usize) -> Result<Vec<f32>> {
        Ok(bytemuck::pod_collect_to_vec(self.bytes(n * 4)?))
    }
    fn u32s(&mut self, n: usize) -> Result<Vec<u32>> {
        Ok(bytemuck::pod_collect_to_vec(self.bytes(n * 4)?))
    }
}

pub fn load(path: &str) -> Result<TreePack> {
    let data = std::fs::read(path).with_context(|| format!("reading {path}"))?;
    let mut r = Reader { d: &data, p: 0 };
    if r.bytes(4)? != b"MTRE" {
        bail!("{path}: not an .mtree file");
    }
    let version = r.u32()?;
    if version != 1 && version != 2 {
        bail!("{path}: unsupported version {version}");
    }
    let mesh_count = r.u32()? as usize;
    let mut meshes = Vec::with_capacity(mesh_count);
    for _ in 0..mesh_count {
        let name = r.string()?;
        let v = r.u32()? as usize;
        let t = r.u32()? as usize;
        let positions = r.f32s(v * 3)?;
        let normals = r.f32s(v * 3)?;
        let uvs = if version >= 2 { r.f32s(v * 2)? } else { vec![0.0; v * 2] };
        let indices = r.u32s(t * 3)?;
        let padded = t.div_ceil(4) * 4;
        let materials = r.bytes(padded)?.to_vec();
        meshes.push(MeshData { name, positions, normals, uvs, indices, materials, grid: 0 });
    }
    let tree_count = r.u32()? as usize;
    let mut trees = Vec::with_capacity(tree_count);
    for _ in 0..tree_count {
        let name = r.string()?;
        let b = r.f32s(6)?;
        let n = r.u32()? as usize;
        let mut parts = Vec::with_capacity(n);
        for _ in 0..n {
            let mesh = r.u32()?;
            let m = r.f32s(12)?;
            let xform = Affine3A::from_cols(
                Vec3A::new(m[0], m[4], m[8]),
                Vec3A::new(m[1], m[5], m[9]),
                Vec3A::new(m[2], m[6], m[10]),
                Vec3A::new(m[3], m[7], m[11]),
            );
            parts.push(TreePart { mesh, xform });
        }
        trees.push(TreeAsset { name, bmin: Vec3::new(b[0], b[1], b[2]), bmax: Vec3::new(b[3], b[4], b[5]), parts });
    }
    Ok(TreePack { meshes, trees })
}
