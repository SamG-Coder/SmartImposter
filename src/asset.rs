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
}

impl MeshData {
    pub fn vertex_count(&self) -> u32 {
        (self.positions.len() / 3) as u32
    }
    pub fn tri_count(&self) -> u32 {
        (self.indices.len() / 3) as u32
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
        let padded = (t + 3) / 4 * 4;
        let materials = r.bytes(padded)?.to_vec();
        meshes.push(MeshData { name, positions, normals, uvs, indices, materials });
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
