//! Procedural terrain and forest layout.

use crate::asset::{MeshData, TreeAsset};
use crate::catalog::{Catalog, Kind};
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
    let mask = ((patches + 0.42) / 0.22).clamp(0.0, 1.0);
    clearing * mask
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Biome fields at a point, all in [0, 1].
struct Biome {
    /// forest canopy density (0 = meadow, 1 = closed forest)
    dens: f32,
    /// soil moisture: low ground and wet patches (alder, ferns)
    wet: f32,
    /// forest edge: peaks where the canopy density crosses ~0.5
    edge: f32,
    /// large beech-dominated stands
    beech: f32,
    /// aspen groves (clonal clusters)
    aspen: f32,
}

fn biome(x: f32, z: f32) -> Biome {
    let dens = forest_density(x, z);
    let h = terrain_height(x, z);
    let wet = (0.45 + 0.6 * fbm(x / 380.0, z / 380.0, 4, 31) - h / 70.0).clamp(0.0, 1.0);
    let edge = smoothstep(0.05, 0.45, dens) * (1.0 - smoothstep(0.55, 0.95, dens));
    let beech = smoothstep(-0.15, 0.3, fbm(x / 700.0, z / 700.0, 3, 41));
    let aspen = smoothstep(0.12, 0.42, fbm(x / 170.0, z / 170.0, 3, 53));
    Biome { dens, wet, edge, beech, aspen }
}

/// Weighted choice among (weight, variant list); variants of a species are equally likely.
fn pick(groups: &[(f32, &[usize])], r1: f32, r2: f32) -> Option<usize> {
    let live = || groups.iter().filter(|g| !g.1.is_empty() && g.0 > 0.0);
    let total: f32 = live().map(|g| g.0).sum();
    if total <= 0.0 {
        return None;
    }
    let mut acc = 0.0;
    for (w, vs) in live() {
        acc += w / total;
        if r1 <= acc {
            return Some(vs[((r2 * vs.len() as f32) as usize).min(vs.len() - 1)]);
        }
    }
    live().last().map(|g| g.1[0])
}

impl World {
    /// Flat test scene: every variant on a flat plane, one row per species (rows spaced by the
    /// largest crown along -Z), variants side by side along X.
    pub fn gallery(trees: &[TreeAsset], cat: &Catalog) -> Self {
        let mut w = Self::build(3072.0, 256, true);
        let mut z = 0.0f32;
        for sp in &cat.species {
            let vs: Vec<usize> = sp.variants.clone().collect();
            let width: f32 = vs.iter().map(|&v| trees[v].bmax.x - trees[v].bmin.x + 1.5).sum();
            let depth = vs.iter().map(|&v| trees[v].bmax.z - trees[v].bmin.z).fold(0.0f32, f32::max);
            let mut x = -width * 0.5;
            println!("gallery row {:<24} z {:7.1} .. {:7.1}, x {:6.1} .. {:6.1}", sp.name, z, z - depth, -width * 0.5, width * 0.5);
            for &v in &vs {
                let t = &trees[v];
                let tw = t.bmax.x - t.bmin.x + 1.5;
                let pos = Vec3::new(x + tw * 0.5 - (t.bmin.x + t.bmax.x) * 0.5, 0.0, z - depth * 0.5 - (t.bmin.z + t.bmax.z) * 0.5);
                let xform = Affine3A::from_translation(pos);
                let center = xform.transform_point3((t.bmin + t.bmax) * 0.5);
                w.trees.push(TreeInstance { xform, center, variant: v as u8, seed: 0, color: (v % 7) as u8 });
                x += tw;
            }
            z -= depth + 2.0;
        }
        w
    }

    /// Terrain plus three scattered layers: canopy trees (`spacing`), shrubs and ferns. Species
    /// follow biome fields: beech stands, alder on wet low ground, poplar and hazel along forest
    /// edges, aspen groves, heather in open dry ground, huckleberry at edges, bracken under canopy.
    pub fn generate(trees: &[TreeAsset], cat: &Catalog, size: f32, cells: usize, spacing: f32, radius: f32) -> Self {
        let mut world = Self::build(size, cells, false);
        if spacing <= 0.0 {
            return world;
        }
        let beech = cat.variants_of("Beech");
        let alder = cat.variants_of("Alder");
        let poplar = cat.variants_of("Poplar");
        let aspen = cat.variants_of("Aspen");
        let hazel = cat.variants_of("Hazel");
        let heather = cat.variants_of("Heather");
        let huck = cat.variants_of("Huckleberry");
        let fern = cat.variants_of("Fern");
        // Anything not covered by the biome rules (e.g. a newly added species) is spread evenly.
        let known: Vec<usize> =
            [&beech, &alder, &poplar, &aspen, &hazel, &heather, &huck, &fern].iter().flat_map(|v| v.iter().copied()).collect();
        let other_trees: Vec<usize> = (0..trees.len()).filter(|v| !known.contains(v) && cat.variants[*v].kind == Kind::Tree).collect();
        let other_small: Vec<usize> = (0..trees.len()).filter(|v| !known.contains(v) && cat.variants[*v].kind != Kind::Tree).collect();

        let under_radius = radius.min(900.0);
        // (layer, grid spacing, radius)
        let layers = [(0u32, spacing, radius), (1, 3.4, under_radius), (2, 2.1, under_radius)];
        let mut counts = [0usize; 3];
        for (layer, step, rad) in layers {
            let count = (rad * 2.0 / step) as i32;
            let hs = |gx: i32, gz: i32, k: u32| hash(gx, gz, k + layer * 100);
            for gz in -count / 2..count / 2 {
                for gx in -count / 2..count / 2 {
                    let x = (gx as f32 + 0.5 + (hs(gx, gz, 1) - 0.5) * 0.9) * step;
                    let z = (gz as f32 + 0.5 + (hs(gx, gz, 2) - 0.5) * 0.9) * step;
                    if x * x + z * z > rad * rad {
                        continue;
                    }
                    let b = biome(x, z);
                    let (r_place, r1, r2) = (hs(gx, gz, 3), hs(gx, gz, 4), hs(gx, gz, 5));
                    let choice = match layer {
                        0 => {
                            // Canopy: forest mask, plus the odd lone tree out in the meadows.
                            if r_place > b.dens.max(0.025) {
                                continue;
                            }
                            pick(
                                &[
                                    (0.35 + 2.6 * b.beech * (1.0 - b.wet), &beech),
                                    (3.2 * b.wet * b.wet, &alder),
                                    (0.08 + 1.4 * b.edge * (0.4 + b.wet), &poplar),
                                    (0.08 + 2.4 * b.aspen, &aspen),
                                    (0.25 + 1.1 * b.edge, &hazel),
                                    (0.5, &other_trees),
                                ],
                                r1,
                                r2,
                            )
                        }
                        1 => {
                            let open = 1.0 - b.dens;
                            let clump = smoothstep(-0.15, 0.35, fbm(x / 28.0, z / 28.0, 3, 61));
                            let w_heather = 1.6 * open * open * (1.0 - b.wet) * clump;
                            let w_huck = 0.9 * b.edge + 0.25 * b.dens * clump;
                            let w_other = if other_small.is_empty() { 0.0 } else { 0.2 };
                            if r_place > (w_heather + w_huck + w_other).min(0.85) {
                                continue;
                            }
                            pick(&[(w_heather, &heather), (w_huck, &huck), (w_other, &other_small)], r1, r2)
                        }
                        _ => {
                            let clump = smoothstep(-0.25, 0.3, fbm(x / 22.0, z / 22.0, 3, 71));
                            let w = b.dens.powf(1.3) * (0.35 + 0.65 * b.wet) * clump + 0.06 * b.edge;
                            if r_place > (1.4 * w).min(0.85) {
                                continue;
                            }
                            pick(&[(1.0, &fern)], r1, r2)
                        }
                    };
                    let Some(variant) = choice else { continue };
                    let yaw = hs(gx, gz, 6) * std::f32::consts::TAU;
                    let scale = if layer == 0 { 0.8 + 0.4 * hs(gx, gz, 7) } else { 0.7 + 0.6 * hs(gx, gz, 7) };
                    let y = world.height_at(x, z) - if layer == 0 { 0.15 } else { 0.03 };
                    let xform = Affine3A::from_scale_rotation_translation(Vec3::splat(scale), Quat::from_rotation_y(yaw), Vec3::new(x, y, z));
                    let t = &trees[variant];
                    let center = xform.transform_point3((t.bmin + t.bmax) * 0.5);
                    let seed = 1 + (hs(gx, gz, 8) * 8190.0) as u16;
                    // Lattice colouring (gx + 3 gz) mod 7 within a layer; layers are offset.
                    let color = (gx + 3 * gz + 2 * layer as i32).rem_euclid(7) as u8;
                    world.trees.push(TreeInstance { xform, center, variant: variant as u8, seed, color });
                    counts[layer as usize] += 1;
                }
            }
        }
        println!("Scattered {} trees, {} shrubs, {} ferns/plants", counts[0], counts[1], counts[2]);
        let mut per = vec![0usize; trees.len()];
        for t in &world.trees {
            per[t.variant as usize] += 1;
        }
        for sp in &cat.species {
            let n: usize = sp.variants.clone().map(|v| per[v]).sum();
            println!("  {:<24} {:>7}", sp.name, n);
        }
        world
    }

    fn build(size: f32, cells: usize, flat: bool) -> Self {
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

        World { terrain, trees: Vec::new(), size, cells, heights }
    }

    /// Ground layer mask over the whole terrain (res x res RGBA8): R canopy cover (splatted from
    /// the actual tree crowns), G moisture, B dryness (open sunny ground), A understory density.
    /// The terrain shader blends the scanned ground materials with it.
    pub fn ground_mask(&self, trees: &[TreeAsset], cat: &Catalog, res: usize) -> Vec<u8> {
        let cell = self.size / res as f32;
        let half = self.size * 0.5;
        let mut cover = vec![0.0f32; res * res];
        let mut under = vec![0.0f32; res * res];
        for t in &self.trees {
            let v = t.variant as usize;
            let a = &trees[v];
            let scale = t.xform.matrix3.x_axis.length();
            let tree = cat.variants[v].kind == Kind::Tree && cat.variants[v].height >= 4.0;
            let rc = 0.5 * (a.bmax.x - a.bmin.x).max(a.bmax.z - a.bmin.z) * scale * if tree { 1.0 } else { 1.6 };
            let (cx, cz) = (t.center.x, t.center.z);
            let r = rc * 1.2 + cell;
            let (i0, i1) = (((cx - r + half) / cell).floor().max(0.0) as usize, (((cx + r + half) / cell).ceil() as usize).min(res - 1));
            let (j0, j1) = (((cz - r + half) / cell).floor().max(0.0) as usize, (((cz + r + half) / cell).ceil() as usize).min(res - 1));
            for j in j0..=j1 {
                for i in i0..=i1 {
                    let x = (i as f32 + 0.5) * cell - half;
                    let z = (j as f32 + 0.5) * cell - half;
                    let d = ((x - cx).powi(2) + (z - cz).powi(2)).sqrt();
                    let f = 1.0 - smoothstep(0.5 * rc, 1.2 * rc, d);
                    let dst = if tree { &mut cover } else { &mut under };
                    let k = j * res + i;
                    dst[k] = 1.0 - (1.0 - dst[k]) * (1.0 - f * if tree { 0.9 } else { 0.5 });
                }
            }
        }
        // soften the splats
        let blur = |src: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0f32; res * res];
            for j in 0..res {
                for i in 0..res {
                    let mut s = 0.0;
                    for dj in -1i32..=1 {
                        for di in -1i32..=1 {
                            let ii = (i as i32 + di).clamp(0, res as i32 - 1) as usize;
                            let jj = (j as i32 + dj).clamp(0, res as i32 - 1) as usize;
                            s += src[jj * res + ii];
                        }
                    }
                    out[j * res + i] = s / 9.0;
                }
            }
            out
        };
        // three passes: a lone crown fades to partial cover, closed canopy stays ~1
        let cover = blur(&blur(&blur(&cover)));
        let under = blur(&under);
        let mut px = vec![0u8; res * res * 4];
        for j in 0..res {
            for i in 0..res {
                let x = (i as f32 + 0.5) * cell - half;
                let z = (j as f32 + 0.5) * cell - half;
                let h = self.height_at(x, z);
                let wet = (0.45 + 0.6 * fbm(x / 380.0, z / 380.0, 4, 31) - h / 70.0).clamp(0.0, 1.0);
                let dry = smoothstep(-0.1, 0.45, fbm(x / 140.0, z / 140.0, 3, 83) + 0.5 * (0.5 - wet));
                let k = j * res + i;
                let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                px[k * 4..k * 4 + 4].copy_from_slice(&[q(cover[k]), q(wet), q(dry), q(under[k])]);
            }
        }
        px
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
