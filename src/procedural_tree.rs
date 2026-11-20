//! Grow a broadleaf from measured limb axes and shoot distributions. Only the offline fitter
//! reads scan triangles. The runtime builds bark and four shoot prototypes once, then regrows
//! a seeded crown by changing instance transforms, reusing the same mesh buffers and BLAS.
use crate::asset::{MeshData, TreeAsset, TreePack, TreePart};
use anyhow::{ensure, Context, Result};
use glam::{Affine3A, Mat3, Quat, Vec3};
use std::f32::consts::TAU;
use std::path::Path;

const SHOOT_TYPES: usize = 4;

struct Reader<'a>(&'a [u8]);
impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        let (head, tail) = self.0.split_at_checked(n).context("truncated tree recipe")?;
        self.0 = tail;
        Ok(head)
    }
    fn word(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into()?))
    }
    fn float(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into()?))
    }
}

struct Ring {
    at: Vec3,
    radius: f32,
}
struct Buds {
    a: Vec3,
    b: Vec3,
    count: usize,
    direction: Vec3,
    agreement: f32,
}

pub struct Recipe {
    pub name: String,
    pub palette: [[u8; 3]; 2],
    pub reference_height: f32,
    axes: Vec<Vec<Ring>>,
    buds: Vec<Buds>,
    count: usize,
    length: f32,
    area: f32,
}

struct Rng(u32);
impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(747796405).wrapping_add(2891336453);
        let x = ((self.0 >> ((self.0 >> 28) + 4)) ^ self.0).wrapping_mul(277803737);
        (((x >> 22) ^ x) >> 8) as f32 / 16777216.0
    }
    fn signed(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }
}

impl Recipe {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("reading tree recipe {}", path.display()))?;
        let mut r = Reader(&bytes);
        ensure!(r.take(4)? == b"SGT2", "unsupported tree recipe");
        let n = r.word()? as usize;
        let name = std::str::from_utf8(r.take(n)?)?.to_owned();
        let palette = [r.take(3)?.try_into()?, r.take(3)?.try_into()?];
        let reference_height = r.float()?;
        ensure!(
            reference_height.is_finite() && reference_height > 0.0,
            "invalid reference height"
        );
        let (length, area) = (r.float()?, r.float()?);
        ensure!(
            (0.015..=20.0).contains(&length) && (0.00001..=100.0).contains(&area),
            "invalid shoot dimensions"
        );
        let n = u32::from_le_bytes(r.take(4)?.try_into()?) as usize;
        ensure!((1..=65535).contains(&n), "invalid axis count");
        let mut axes = Vec::with_capacity(n);
        let mut buds = Vec::new();
        let mut count = 0;
        for _ in 0..n {
            let nr = r.word()? as usize;
            ensure!(nr >= 2, "axis needs two sections");
            let mut rings = Vec::with_capacity(nr);
            for _ in 0..nr {
                let at = Vec3::new(r.word()? as i16 as f32, r.word()? as i16 as f32, r.word()? as i16 as f32) * 0.01;
                rings.push(Ring {
                    at,
                    radius: (r.word()? as f32 * 0.001).max(0.001),
                });
            }
            for seg in rings.windows(2) {
                let k = r.word()? as usize;
                if k == 0 {
                    continue;
                }
                count += k;
                ensure!(count <= 100_000, "too many shoots for one tree");
                let p = r.take(5)?;
                ensure!(p[0] <= p[1], "reversed growth interval");
                let mean = Vec3::new(p[2] as i8 as f32, p[3] as i8 as f32, p[4] as i8 as f32) / 127.0;
                let along = |t| seg[0].at.lerp(seg[1].at, t as f32 / 255.0);
                buds.push(Buds {
                    a: along(p[0]),
                    b: along(p[1]),
                    count: k,
                    direction: mean.try_normalize().unwrap_or(Vec3::Y),
                    agreement: mean.length().min(1.0),
                });
            }
            axes.push(rings);
        }
        ensure!(r.0.is_empty() && count > 0, "invalid end of tree recipe");
        Ok(Self {
            name,
            palette,
            reference_height,
            axes,
            buds,
            count,
            length,
            area,
        })
    }

    pub fn grow(&self, seed: u32, parts: &mut Vec<TreePart>) {
        let mut rng = Rng(seed);
        // Whole-plant variation preserves the connections between limbs. Seed zero uses the
        // measured proportions; the finer shoots vary within each measured growth cone.
        let form = if seed == 0 {
            Affine3A::IDENTITY
        } else {
            Affine3A::from_mat3(Mat3::from_cols(
                Vec3::X * (1.0 + 0.10 * rng.signed()),
                Vec3::new(0.025 * rng.signed(), 1.0 + 0.06 * rng.signed(), 0.025 * rng.signed()),
                Vec3::Z * (1.0 + 0.10 * rng.signed()),
            ))
        };
        parts.clear();
        parts.reserve(self.count + 1);
        parts.push(TreePart { mesh: 0, xform: form });
        for bud in &self.buds {
            let (x, z) = bud.direction.any_orthonormal_pair();
            for i in 0..bud.count {
                let at = bud.a.lerp(bud.b, (i as f32 + rng.unit()) / bud.count as f32);
                let (s, c) = (TAU * rng.unit()).sin_cos();
                // A uniform cone of half-angle theta has mean cosine (1+cos(theta))/2.
                let cos = 1.0 - rng.unit() * (2.0 * (1.0 - bud.agreement)).max(0.065);
                let dir = bud.direction * cos + (x * c + z * s) * (1.0 - cos * cos).max(0.0).sqrt();
                let rot = Quat::from_rotation_arc(Vec3::Y, dir.normalize()) * Quat::from_rotation_y(TAU * rng.unit());
                let scale = Vec3::splat(1.0 + 0.14 * rng.signed());
                let mesh = 1 + (rng.unit() * SHOOT_TYPES as f32) as u32;
                parts.push(TreePart {
                    mesh,
                    xform: form * Affine3A::from_scale_rotation_translation(scale, rot, at),
                });
            }
        }
    }

    pub fn build(&self, seed: u32) -> TreePack {
        let mut bark = Builder::new("procedural bark");
        for rings in &self.axes {
            let thick = rings.iter().map(|r| r.radius).fold(0.0, f32::max);
            bark.tube(
                rings,
                if thick > 0.09 {
                    10
                } else if thick > 0.026 {
                    6
                } else {
                    4
                },
            );
        }
        let mut meshes = vec![bark.finish()];
        meshes.extend((0..SHOOT_TYPES).map(|i| shoot(i as u32, self.length, self.area)));
        let mut parts = Vec::new();
        self.grow(seed, &mut parts);
        // A seed-independent envelope lets existing world AABBs survive a live regrowth.
        let reach = meshes[1..]
            .iter()
            .flat_map(|m| m.positions.chunks_exact(3))
            .map(|p| Vec3::from_slice(p).length())
            .fold(0.0, f32::max)
            * 1.14;
        let (mut lo, mut hi) = meshes[0].bounds();
        for bud in &self.buds {
            lo = lo.min(bud.a.min(bud.b) - reach);
            hi = hi.max(bud.a.max(bud.b) + reach);
        }
        let ext = lo.abs().max(hi.abs());
        let pad = ext * Vec3::new(0.10, 0.06, 0.10) + ext.y * Vec3::new(0.025, 0.0, 0.025) + Vec3::splat(0.002);
        TreePack {
            meshes,
            trees: vec![TreeAsset {
                name: self.name.clone(),
                bmin: lo - pad,
                bmax: hi + pad,
                parts,
            }],
        }
    }
}

struct Builder(MeshData);
impl Builder {
    fn new(name: &str) -> Self {
        Self(MeshData {
            name: name.into(),
            positions: vec![],
            normals: vec![],
            uvs: vec![],
            indices: vec![],
            materials: vec![],
            grid: 0,
        })
    }
    fn vert(&mut self, p: Vec3, n: Vec3, uv: [f32; 2]) -> u32 {
        let id = self.0.vertex_count();
        self.0.positions.extend(p.to_array());
        self.0.normals.extend(n.to_array());
        self.0.uvs.extend(uv);
        id
    }
    fn tri(&mut self, a: u32, b: u32, c: u32, mat: u8) {
        self.0.indices.extend([a, b, c]);
        self.0.materials.push(mat);
    }
    fn tube(&mut self, rings: &[Ring], sides: usize) {
        let first = self.0.vertex_count();
        let mut side = (rings.last().unwrap().at - rings[0].at)
            .try_normalize()
            .unwrap_or(Vec3::Y)
            .any_orthonormal_vector();
        for (i, r) in rings.iter().enumerate() {
            let axis = (rings[(i + 1).min(rings.len() - 1)].at - rings[i.saturating_sub(1)].at)
                .try_normalize()
                .unwrap_or(Vec3::Y);
            side = (side - axis * axis.dot(side))
                .try_normalize()
                .unwrap_or_else(|| axis.any_orthonormal_vector());
            let other = axis.cross(side);
            for j in 0..sides {
                let (s, c) = (TAU * j as f32 / sides as f32).sin_cos();
                let n = c * side + s * other;
                self.vert(r.at + r.radius * n, n, [j as f32 / sides as f32, i as f32]);
                if i > 0 {
                    let a = first + ((i - 1) * sides + j) as u32;
                    let b = first + ((i - 1) * sides + (j + 1) % sides) as u32;
                    self.tri(a, b, a + sides as u32, 0);
                    self.tri(b, b + sides as u32, a + sides as u32, 0);
                }
            }
        }
    }
    fn stem(&mut self, a: Vec3, b: Vec3, r: f32) {
        self.tube(&[Ring { at: a, radius: r }, Ring { at: b, radius: r * 0.32 }], 3);
    }
    fn leaf(&mut self, at: Vec3, dir: Vec3, length: f32, mat: u8) {
        let right = Vec3::Y.cross(dir).normalize();
        let up = dir.cross(right);
        let first = self.0.vertex_count();
        // Pointed blade with broad shoulders, a raised midrib and a downturned tip.
        for (along, width, lift) in [
            (0., 0., 0.),
            (0.30, 0.84, -0.02),
            (0.67, 1., -0.03),
            (1., 0., -0.10),
            (0.67, -1., -0.03),
            (0.30, -0.84, -0.02),
            (0.48, 0., 0.075),
        ] {
            let p = at + length * (dir * along + right * (width * 0.39) + up * lift);
            self.vert(p, (up + right * (width * 0.28)).normalize(), [0.5 + width * 0.48, along]);
        }
        for i in 0..6 {
            self.tri(first + 6, first + (i + 1) % 6, first + i, mat);
        }
    }
    fn finish(mut self) -> MeshData {
        self.0.materials.resize(self.0.materials.len().next_multiple_of(4), 0);
        self.0
    }
}

fn shoot(id: u32, length: f32, area: f32) -> MeshData {
    let mut m = Builder::new("procedural leafy shoot");
    let mut rng = Rng(31847 + id * 1999);
    let mut prev = Vec3::ZERO;
    // Scan shoot surfaces include petioles and folded blades. Reserve part of their area
    // for those, otherwise flattened procedural blades close the gaps in the crown.
    let leaf_size = (0.86 * area / (8.0 * 6.0 * 0.53)).sqrt();
    for i in 0..8 {
        let t = (i as f32 + 1.0) / 8.5;
        let stem = Vec3::new(0.11 * t * t, t, 0.075 * (4.0 * t).sin() * t) * length;
        m.stem(prev, stem, 0.006 * length * (1.2 - 0.6 * t));
        prev = stem;
        let angle = i as f32 * 2.39996 + 0.45 * rng.signed();
        let (s, c) = angle.sin_cos();
        let outward = Vec3::new(c, 0.0, s);
        let across = Vec3::new(-s, 0.0, c);
        let tip = stem
            + length * (outward * (0.23 + 0.13 * (std::f32::consts::PI * t).sin()) + Vec3::Y * (0.10 + 0.10 * rng.unit()));
        m.stem(stem, tip, length * 0.0034);
        for k in 0..6 {
            let u = (k as f32 + 0.6) / 6.4;
            let side = if k % 2 == 0 { 1.0 } else { -1.0 };
            let dir = (outward * 0.65 + side * across * (0.5 + 0.45 * rng.unit()) + Vec3::Y * (0.55 * rng.signed() - 0.10))
                .normalize();
            let size = leaf_size * (0.82 + 0.34 * rng.unit());
            let mat = 1 | ((rng.unit() * 64.0) as u8) << 2;
            m.leaf(stem.lerp(tip, u), dir, size, mat);
        }
    }
    m.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Live regeneration reuses the original AABB, images and mesh BLAS. Check all three
    // contracts using a small tree that can exercise every vertex of every placed shoot.
    #[test]
    fn regrowth_is_repeatable_and_stays_in_the_initial_bounds() {
        let tip = Vec3::new(0.4, 3.0, -0.2);
        let recipe = Recipe {
            name: "test tree".into(),
            palette: [[70, 90, 40], [80, 70, 50]],
            reference_height: 3.4,
            length: 0.75,
            area: 0.3,
            count: 9,
            axes: vec![vec![
                Ring {
                    at: Vec3::ZERO,
                    radius: 0.18,
                },
                Ring { at: tip, radius: 0.02 },
            ]],
            buds: vec![Buds {
                a: tip * 0.6,
                b: tip,
                count: 9,
                direction: Vec3::X,
                agreement: 0.55,
            }],
        };
        let mut pack = recipe.build(0);
        let tree = &mut pack.trees[0];
        let snapshot = |parts: &[TreePart]| parts.iter().map(|p| (p.mesh, p.xform.to_cols_array())).collect::<Vec<_>>();
        let first = snapshot(&tree.parts);
        let storage = tree.parts.as_ptr();
        for seed in [7, 1, 52, u32::MAX, 0] {
            recipe.grow(seed, &mut tree.parts);
            assert_eq!(storage, tree.parts.as_ptr());
            assert_eq!(snapshot(&tree.parts) == first, seed == 0);
            for p in &tree.parts {
                let m = &pack.meshes[p.mesh as usize];
                for v in m.positions.chunks_exact(3) {
                    let at = p.xform.transform_point3(Vec3::from_slice(v));
                    assert!(at.cmpge(tree.bmin).all() && at.cmple(tree.bmax).all());
                }
                assert_eq!(m.uvs.len() / 2, m.vertex_count() as usize);
                assert_eq!(m.positions.len(), m.normals.len());
                assert!(m.indices.iter().all(|i| *i < m.vertex_count()));
                assert_eq!(m.materials.len(), (m.tri_count() as usize).next_multiple_of(4));
            }
        }
    }
}
