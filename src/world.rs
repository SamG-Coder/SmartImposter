//! Procedural terrain and forest layout.

use crate::asset::{MeshData, TreeAsset};
use glam::{Affine3A, Quat, Vec3};

pub const MAT_TERRAIN: u8 = 2;

pub struct TreeInstance {
    pub xform: Affine3A,
    pub center: Vec3, // world-space bounding sphere centre, used for LOD distance
    pub variant: u8,
    pub seed: u16, // 13 bits, never 0 (0 = untinted, used by the baker)
    pub color: u8, // 0..6: impostor instance-mask colour, differs between overlapping neighbours
}

pub struct World {
    pub terrain: MeshData,
    pub trees: Vec<TreeInstance>,
    pub size: f32,
    cells: usize,
    heights: Vec<f32>,
}

fn hash(ix: i32, iz: i32, seed: u32) -> f32 {
    let mut h = (ix as u32).wrapping_mul(0x8da6_b343) ^ (iz as u32).wrapping_mul(0xd816_3841) ^ seed.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^= h >> 15;
    h as f32 / u32::MAX as f32
}

fn value_noise(x: f32, z: f32, seed: u32) -> f32 {
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

/// Fractal noise in roughly [-1, 1].
fn fbm(x: f32, z: f32, octaves: u32, seed: u32) -> f32 {
    let (mut sum, mut amp, mut freq, mut norm) = (0.0, 1.0, 1.0, 0.0);
    for o in 0..octaves {
        sum += (value_noise(x * freq, z * freq, seed + o * 101) * 2.0 - 1.0) * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.03;
    }
    sum / norm
}

pub fn terrain_height(x: f32, z: f32) -> f32 {
    let r = (x * x + z * z).sqrt();
    let hills = 55.0 * fbm(x / 900.0, z / 900.0, 5, 11) + 9.0 * fbm(x / 160.0, z / 160.0, 4, 23);
    // Flatten the start area a little so the clearing reads well.
    let flatten = (r / 120.0).clamp(0.0, 1.0);
    hills * (0.35 + 0.65 * flatten)
}

fn forest_density(x: f32, z: f32) -> f32 {
    let r = (x * x + z * z).sqrt();
    let clearing = ((r - 24.0) / 10.0).clamp(0.0, 1.0);
    let patches = fbm(x / 260.0, z / 260.0, 4, 7);
    let mask = ((patches + 0.25) / 0.2).clamp(0.0, 1.0);
    clearing * mask
}

impl World {
    /// Flat test scene: one tree of each variant in a row along X, 16 m apart, at z = 0.
    pub fn gallery(trees: &[TreeAsset]) -> Self {
        let mut w = Self::build(trees, 3072.0, 256, 0.0, 0.0, true);
        let n = trees.len();
        for (v, t) in trees.iter().enumerate() {
            let x = (v as f32 - (n as f32 - 1.0) * 0.5) * 16.0;
            let xform = Affine3A::from_translation(Vec3::new(x, 0.0, 0.0));
            let center = xform.transform_point3((t.bmin + t.bmax) * 0.5);
            w.trees.push(TreeInstance { xform, center, variant: v as u8, seed: 0, color: (v % 7) as u8 });
        }
        w
    }

    pub fn generate(trees: &[TreeAsset], size: f32, cells: usize, spacing: f32, radius: f32) -> Self {
        Self::build(trees, size, cells, spacing, radius, false)
    }

    fn build(trees: &[TreeAsset], size: f32, cells: usize, spacing: f32, radius: f32, flat: bool) -> Self {
        let n = cells + 1;
        let step = size / cells as f32;
        let half = size * 0.5;
        let mut heights = vec![0.0f32; n * n];
        if !flat {
            for j in 0..n {
                for i in 0..n {
                    heights[j * n + i] = terrain_height(i as f32 * step - half, j as f32 * step - half);
                }
            }
        }
        let mut positions = Vec::with_capacity(n * n * 3);
        let mut normals = Vec::with_capacity(n * n * 3);
        for j in 0..n {
            for i in 0..n {
                let h = heights[j * n + i];
                positions.extend_from_slice(&[i as f32 * step - half, h, j as f32 * step - half]);
                let hl = heights[j * n + i.saturating_sub(1)];
                let hr = heights[j * n + (i + 1).min(n - 1)];
                let hd = heights[j.saturating_sub(1) * n + i];
                let hu = heights[(j + 1).min(n - 1) * n + i];
                let nrm = Vec3::new(hl - hr, 2.0 * step, hd - hu).normalize();
                normals.extend_from_slice(&[nrm.x, nrm.y, nrm.z]);
            }
        }
        let mut indices = Vec::with_capacity(cells * cells * 6);
        for j in 0..cells {
            for i in 0..cells {
                let a = (j * n + i) as u32;
                let b = a + 1;
                let c = a + n as u32;
                let d = c + 1;
                indices.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }
        let tri_count = indices.len() / 3;
        let materials = vec![MAT_TERRAIN; (tri_count + 3) / 4 * 4];
        let uvs = vec![0.0; n * n * 2];
        let terrain = MeshData { name: "terrain".into(), positions, normals, uvs, indices, materials };

        let mut world = World { terrain, trees: Vec::new(), size, cells, heights };

        // Jittered grid scatter, thinned by the forest mask.
        let weights = [0.22f32, 0.28, 0.30, 0.20];
        if flat || spacing <= 0.0 {
            return world;
        }
        let count = (radius * 2.0 / spacing) as i32;
        for gz in -count / 2..count / 2 {
            for gx in -count / 2..count / 2 {
                let jx = hash(gx, gz, 1) - 0.5;
                let jz = hash(gx, gz, 2) - 0.5;
                let x = (gx as f32 + 0.5 + jx * 0.9) * spacing;
                let z = (gz as f32 + 0.5 + jz * 0.9) * spacing;
                if x * x + z * z > radius * radius {
                    continue;
                }
                if hash(gx, gz, 3) > forest_density(x, z) {
                    continue;
                }
                let pick = hash(gx, gz, 4);
                let mut acc = 0.0;
                let mut variant = 0usize;
                for (k, w) in weights.iter().enumerate().take(trees.len()) {
                    acc += w;
                    variant = k;
                    if pick < acc {
                        break;
                    }
                }
                let yaw = hash(gx, gz, 5) * std::f32::consts::TAU;
                let scale = 0.82 + 0.36 * hash(gx, gz, 6);
                let y = world.height_at(x, z) - 0.15;
                let xform = Affine3A::from_scale_rotation_translation(Vec3::splat(scale), Quat::from_rotation_y(yaw), Vec3::new(x, y, z));
                let t = &trees[variant];
                let center = xform.transform_point3((t.bmin + t.bmax) * 0.5);
                let seed = 1 + (hash(gx, gz, 7) * 8190.0) as u16;
                // Lattice colouring (gx + 3 gz) mod 7: the nearest same-coloured cells are at offsets
                // like (1,2) or (3,-1), i.e. >= 15 m apart, so overlapping crowns never share a colour.
                let color = (gx + 3 * gz).rem_euclid(7) as u8;
                world.trees.push(TreeInstance { xform, center, variant: variant as u8, seed, color });
            }
        }
        world
    }

    /// Height of the triangulated terrain surface (matches the mesh exactly).
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        let n = self.cells + 1;
        let step = self.size / self.cells as f32;
        let fx = ((x + self.size * 0.5) / step).clamp(0.0, self.cells as f32 - 1e-3);
        let fz = ((z + self.size * 0.5) / step).clamp(0.0, self.cells as f32 - 1e-3);
        let (i, j) = (fx.floor() as usize, fz.floor() as usize);
        let (u, v) = (fx - i as f32, fz - j as f32);
        let h = |a: usize, b: usize| self.heights[b * n + a];
        // Triangles per cell: (a, c, b) and (b, c, d) split along the b-c diagonal (u + v = 1).
        if u + v <= 1.0 {
            h(i, j) + (h(i + 1, j) - h(i, j)) * u + (h(i, j + 1) - h(i, j)) * v
        } else {
            h(i + 1, j + 1) + (h(i, j + 1) - h(i + 1, j + 1)) * (1.0 - u) + (h(i + 1, j) - h(i + 1, j + 1)) * (1.0 - v)
        }
    }
}
