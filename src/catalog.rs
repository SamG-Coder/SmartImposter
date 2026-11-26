//! Plant catalog: every species pack in `assets/species` (written by tools/megaplant_to_mtree.py)
//! merged into one `TreePack`, plus per-species materials and per-variant LOD parameters.
//! Also finds the scanned ground materials.

use crate::asset::{self, TreePack};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Tree,
    Shrub,
    Plant,
    /// Megascans grass patches (alpha-tested cards, tools/grass_to_mtree.py)
    Grass,
    /// Opaque scans without foliage: rocks, logs, snags, cacti (tools/fab_to_mtree.py solid)
    Rock,
    /// Procedural building archetypes (city.rs): a facade material of their own (tri.rchit),
    /// meshes much further out than plants, never thinned with distance
    Building,
}

pub struct Species {
    pub name: String,
    pub kind: Kind,
    /// foliage CA, foliage NT, bark C, bark NAH (None = placeholder)
    pub textures: [Option<PathBuf>; 4],
    pub leaf_tint: [f32; 3],
    pub bark_tint: [f32; 3],
    pub trans_tint: [f32; 3],
    /// Global variant indices of this species.
    pub variants: std::ops::Range<usize>,
}

pub struct Variant {
    pub species: usize,
    pub kind: Kind,
    pub height: f32,
    /// LOD distances (mesh <-> impostor) are multiplied by this: small plants switch closer.
    pub lod_scale: f32,
    /// Draw distance in metres (0 = unlimited).
    pub cull: f32,
    /// Impostor texel size multiplier (grass patches are wide and low: coarser atlases).
    pub texel_scale: f32,
}

/// A scanned ground surface: albedo, normal (DirectX), occlusion/roughness/metal.
pub struct GroundMaterial {
    pub albedo: PathBuf,
    pub normal: PathBuf,
    pub orm: PathBuf,
}

pub struct Catalog {
    pub species: Vec<Species>,
    pub variants: Vec<Variant>,
    /// Texture set (= species index) of every mesh in the merged pack.
    pub mesh_species: Vec<u32>,
    /// Meshes whose leaf texture alpha is an opacity mask (cards): alpha-tested in tri.rahit.
    pub mesh_alpha: Vec<bool>,
    /// Ground layers in shader order (GROUND_LAYERS); None = not downloaded.
    pub ground: Vec<Option<GroundMaterial>>,
}

impl Catalog {
    /// Is this mesh one of the procedural buildings (city.rs)?
    pub fn is_building_mesh(&self, mesh: usize) -> bool {
        self.mesh_species.get(mesh).and_then(|&s| self.species.get(s as usize)).is_some_and(|s| s.kind == Kind::Building)
    }

    /// Variants of the species whose name contains `key`.
    pub fn variants_of(&self, key: &str) -> Vec<usize> {
        self.species.iter().filter(|s| s.name.contains(key)).flat_map(|s| s.variants.clone()).collect()
    }
}

fn lod_for(height: f32) -> (f32, f32) {
    let lod_scale = (height / 16.0).clamp(0.25, 1.2);
    let cull = if height < 2.0 {
        120.0 + 60.0 * height
    } else if height < 6.0 {
        450.0
    } else {
        0.0
    };
    (lod_scale, cull)
}

fn f3(v: &serde_json::Value, default: [f32; 3]) -> [f32; 3] {
    match v.as_array() {
        Some(a) if a.len() >= 3 => [0, 1, 2].map(|i| a[i].as_f64().unwrap_or(default[i] as f64) as f32),
        _ => default,
    }
}

/// Load every `<dir>/*.json` + `.mtree` species pack. Falls back to a single `fallback` mtree
/// (with textures from `fallback_tex`) when the directory has none.
pub fn load(dir: &str, fallback: &str, fallback_tex: &str, only: Option<&str>) -> Result<(TreePack, Catalog)> {
    let mut jsons: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "json")).collect())
        .unwrap_or_default();
    jsons.sort();
    if let Some(only) = only {
        jsons.retain(|p| only.split(',').any(|k| p.file_stem().unwrap().to_string_lossy().contains(k)));
    }
    let mut pack = TreePack { meshes: Vec::new(), trees: Vec::new() };
    let mut cat = Catalog { species: Vec::new(), variants: Vec::new(), mesh_species: Vec::new(), mesh_alpha: Vec::new(), ground: find_ground() };

    if jsons.is_empty() {
        // Legacy single-species scene.
        let p = asset::load(fallback)?;
        let tex = |suffix: &str, foliage: bool| crate::textures::find(Path::new(fallback_tex), suffix, foliage);
        cat.species.push(Species {
            name: Path::new(fallback).file_stem().unwrap().to_string_lossy().into_owned(),
            kind: Kind::Tree,
            textures: [tex("_CA.png", true), tex("_NT.png", true), tex("_C.png", false), tex("_NAH.png", false)],
            leaf_tint: [0.5; 3],
            bark_tint: [0.5; 3],
            trans_tint: [0.5; 3],
            variants: 0..p.trees.len(),
        });
        for t in &p.trees {
            let (lod_scale, cull) = lod_for(t.bmax.y - t.bmin.y);
            cat.variants.push(Variant { species: 0, kind: Kind::Tree, height: t.bmax.y - t.bmin.y, lod_scale, cull, texel_scale: 1.0 });
        }
        cat.mesh_species = vec![0; p.meshes.len()];
        cat.mesh_alpha = vec![false; p.meshes.len()];
        return Ok((p, cat));
    }

    for json in &jsons {
        let text = std::fs::read_to_string(json).with_context(|| format!("reading {}", json.display()))?;
        let j: serde_json::Value = serde_json::from_str(&text).with_context(|| format!("parsing {}", json.display()))?;
        let si = cat.species.len();
        let p = asset::load(&json.with_extension("mtree").to_string_lossy())?;
        let kind = match j["kind"].as_str().unwrap_or("tree") {
            "shrub" => Kind::Shrub,
            "plant" => Kind::Plant,
            "grass" => Kind::Grass,
            "rock" => Kind::Rock,
            _ => Kind::Tree,
        };
        let alpha = j["alpha"].as_bool().unwrap_or(false);
        let base = json.parent().unwrap_or(Path::new("."));
        let tex = |key: &str| j["textures"][key].as_str().map(|s| base.join(s.replace('\\', "/")));
        let mesh_base = pack.meshes.len() as u32;
        let first = pack.trees.len();
        for mut t in p.trees {
            for part in &mut t.parts {
                part.mesh += mesh_base;
            }
            let height = t.bmax.y - t.bmin.y;
            let (mut lod_scale, cull) = lod_for(height);
            if kind == Kind::Rock {
                // rocks and logs lie flat: their size, not their height, sets the impostor detail
                let size = (t.bmax - t.bmin).max_element();
                lod_scale = lod_for(size).0;
            }
            let texel_scale = if kind == Kind::Grass { 1.6 } else { 1.0 };
            if kind == Kind::Grass {
                // Blades are thinner than an atlas texel: keep patches as real cards to ~20 m,
                // where the blades are sub-pixel and the impostor reads as the same sward.
                lod_scale = lod_scale.max(0.4);
            }
            cat.variants.push(Variant { species: si, kind, height, lod_scale, cull, texel_scale });
            pack.trees.push(t);
        }
        cat.mesh_species.extend(std::iter::repeat_n(si as u32, p.meshes.len()));
        cat.mesh_alpha.extend(std::iter::repeat_n(alpha, p.meshes.len()));
        pack.meshes.extend(p.meshes);
        cat.species.push(Species {
            name: j["species"].as_str().unwrap_or("?").to_string(),
            kind,
            textures: [tex("foliage_ca"), tex("foliage_nt"), tex("bark_c"), tex("bark_nah")],
            leaf_tint: f3(&j["leaf_tint"], [0.5; 3]),
            bark_tint: f3(&j["bark_tint"], [0.5; 3]),
            trans_tint: f3(&j["translucency_tint"], [0.5; 3]),
            variants: first..pack.trees.len(),
        });
    }
    Ok((pack, cat))
}

/// Tall palms for the cities, built from what the library has: the Megascans palms are young
/// plants of 1-2 m, so each tall palm is a procedural trunk (7-14 m, tapering, some leaning, with
/// a crownshaft swelling under the fronds; the smooth grey bark of the beech scan) carrying one
/// of those palms scaled up as its crown. Their meshes and textures are the palms' own; only the
/// trunks are new. One species of four variants, grown like trees (globe/city.rs). Nothing is
/// added when the palms are not downloaded.
pub fn add_city_palms(pack: &mut TreePack, cat: &mut Catalog) {
    let find = |cat: &Catalog, k: &str| cat.species.iter().position(|s| s.name.contains(k) && !s.variants.is_empty());
    let bark = find(cat, "European_Beech").or_else(|| find(cat, "Silver_Birch"));
    // (crown species, its variant, trunk height, crown scale, lean m)
    let plan: [(&str, usize, f32, f32, f32); 4] =
        [("Alexandra_Palm", 0, 11.0, 3.6, 0.0), ("Alexandra_Palm", 2, 14.0, 3.4, 0.6), ("Tropical_Palm", 0, 8.5, 3.3, 1.8), ("Chinese_Fan_Palm", 1, 7.0, 3.2, 0.3)];
    let si = cat.species.len();
    let first = pack.trees.len();
    let mut trans = [0.5f32; 3];
    for (k, &(crown_sp, cv, th, cs, lean)) in plan.iter().enumerate() {
        let Some(s) = find(cat, crown_sp) else { continue };
        let Some(src) = s_variant(cat, s, cv) else { continue };
        trans = cat.species[s].trans_tint;
        // the trunk: 12 sides, 24 rings, bent towards +x by `lean` at the top
        let (sides, rings) = (12usize, 24usize);
        let mut m = crate::asset::MeshData { name: format!("City_Palm_Trunk_{k}"), positions: Vec::new(), normals: Vec::new(), uvs: Vec::new(), indices: Vec::new(), materials: Vec::new(), grid: 0 };
        let centre = |t: f32| glam::Vec3::new(lean * t * t, th * t, 0.0);
        for r in 0..=rings {
            let t = r as f32 / rings as f32;
            // root flare, taper, the crownshaft swelling in the top tenth
            let rad = 0.24 - 0.07 * t + 0.12 * (1.0 - t).powi(8) + 0.06 * ((t - 0.9) / 0.1).clamp(0.0, 1.0);
            let c = centre(t);
            for a in 0..=sides {
                let ang = a as f32 / sides as f32 * std::f32::consts::TAU;
                let n = glam::Vec3::new(ang.cos(), 0.0, ang.sin());
                m.positions.extend_from_slice(&(c + n * rad).to_array());
                m.normals.extend_from_slice(&n.to_array());
                m.uvs.extend_from_slice(&[a as f32 / sides as f32 * 0.6, t * th / 1.6]);
            }
        }
        for r in 0..rings as u32 {
            for a in 0..sides as u32 {
                let (i0, i1) = (r * (sides as u32 + 1) + a, (r + 1) * (sides as u32 + 1) + a);
                m.indices.extend_from_slice(&[i0, i1, i0 + 1, i0 + 1, i1, i1 + 1]);
            }
        }
        m.materials = vec![0u8; (m.indices.len() / 3).div_ceil(4) * 4];
        let trunk = pack.meshes.len() as u32;
        let (tlo, thi) = m.bounds();
        pack.meshes.push(m);
        cat.mesh_species.push(bark.unwrap_or(s) as u32);
        cat.mesh_alpha.push(false);
        // the crown on top: the young palm scaled, its base a little inside the crownshaft
        let src_t = &pack.trees[src];
        let top = centre(1.0);
        let lift = glam::Affine3A::from_translation(glam::Vec3::new(top.x, top.y - 0.4 - src_t.bmin.y * cs, top.z)) * glam::Affine3A::from_scale(glam::Vec3::splat(cs));
        let mut parts = vec![crate::asset::TreePart { mesh: trunk, xform: glam::Affine3A::IDENTITY }];
        parts.extend(src_t.parts.iter().map(|p| crate::asset::TreePart { mesh: p.mesh, xform: lift * p.xform }));
        let (mut lo, mut hi) = (tlo, thi);
        for c in 0..8 {
            let q = glam::Vec3::new(
                if c & 1 == 0 { src_t.bmin.x } else { src_t.bmax.x },
                if c & 2 == 0 { src_t.bmin.y } else { src_t.bmax.y },
                if c & 4 == 0 { src_t.bmin.z } else { src_t.bmax.z },
            );
            let w = lift.transform_point3(q);
            lo = lo.min(w);
            hi = hi.max(w);
        }
        let height = hi.y - lo.y;
        let (lod_scale, cull) = lod_for(height);
        cat.variants.push(Variant { species: si, kind: Kind::Tree, height, lod_scale, cull, texel_scale: 1.0 });
        pack.trees.push(crate::asset::TreeAsset { name: format!("City_Palm_{k}"), bmin: lo, bmax: hi, parts });
    }
    if pack.trees.len() > first {
        cat.species.push(Species {
            name: "City_Palm".into(),
            kind: Kind::Tree,
            textures: [None, None, None, None],
            leaf_tint: [0.5; 3],
            bark_tint: [0.5; 3],
            trans_tint: trans,
            variants: first..pack.trees.len(),
        });
    }
}

/// Global variant index of species `s`'s `k`-th variant (the last if it has fewer).
fn s_variant(cat: &Catalog, s: usize, k: usize) -> Option<usize> {
    let r = cat.species[s].variants.clone();
    (!r.is_empty()).then(|| (r.start + k).min(r.end - 1))
}

/// Append the procedural building archetypes (city.rs) as one more species, after the plants.
pub fn add_buildings(pack: &mut TreePack, cat: &mut Catalog) {
    let (p, archs) = crate::city::pack();
    let si = cat.species.len();
    let mesh_base = pack.meshes.len() as u32;
    let first = pack.trees.len();
    for (mut t, a) in p.trees.into_iter().zip(&archs) {
        for part in &mut t.parts {
            part.mesh += mesh_base;
        }
        // A building's mesh costs one instance, a plant's hundreds: the mesh band goes out to
        // ~44 m x the LOD scale, 350 m for the tallest towers, where the impostor's texels are
        // below a pixel and a half at 1440p.
        let lod_scale = (a.height / 25.0).clamp(5.0, 10.0);
        cat.variants.push(Variant { species: si, kind: Kind::Building, height: a.height, lod_scale, cull: 0.0, texel_scale: 1.0 });
        pack.trees.push(t);
    }
    cat.mesh_species.extend(std::iter::repeat_n(si as u32, p.meshes.len()));
    cat.mesh_alpha.extend(std::iter::repeat_n(false, p.meshes.len()));
    pack.meshes.extend(p.meshes);
    cat.species.push(Species {
        name: "Buildings".into(),
        kind: Kind::Building,
        textures: [None, None, None, None],
        leaf_tint: [0.5; 3],
        bark_tint: [0.5; 3],
        trans_tint: [0.0; 3],
        variants: first..pack.trees.len(),
    });
}

/// Ground layers in shader order (tri.rchit): Fab library folder name prefix of each Megascans
/// surface (`<Name>-<first 8 hex digits of the Fab listing id>`).
pub const GROUND_LAYERS: [&str; 26] = [
    "Forest_Floor-cf8af196",     // (a second Forest_Floor scan exists: pin this one)
    "Mossy_Grass",
    "Uncut_Grass",
    "Grass_Dried",
    "Rock_Cliff-e303637d",       // https://www.fab.com/listings/e303637d-ac85-4b3b-8abd-856191b2e251
    "Rock_Cliff-22762313",       // https://www.fab.com/listings/22762313-071f-4017-a721-b29c5a2f1a87
    "Snow_Cliff",                // https://www.fab.com/listings/3e6971d7-95c0-4d0b-a9c0-0430758de77f
    "Icelandic_Gravel-0524d1d9", // https://www.fab.com/listings/0524d1d9-76e1-4c43-be37-88950c7b878a
    "Mossy_Rock-26a6b65d",       // https://www.fab.com/listings/26a6b65d-be1f-453e-94fd-183a5986b507
    "Mossy_Riverbed-ecca0e0f",   // https://www.fab.com/listings/ecca0e0f-0ae5-4f4c-8db0-49537d799253
    "Pebbles_Round-60ac923a",    // https://www.fab.com/listings/60ac923a-b915-40b0-92af-9748ba20c453
    "Wet_Grass_Ground-138dc11e", // https://www.fab.com/listings/138dc11e-bdbd-41cd-8c63-6e16846d0b72
    // the globe's biomes (globe.rchit groundMaterial); the forest scene blends 12 slots, which a
    // --biome preset maps onto any of these (biome.rs)
    "Beach_Sand",            // 12 sea coasts
    "Sand-",                 // 13 dunes, sandy desert
    "Arid_Soil_Ground",      // 14 dry desert and Mediterranean soil
    "Soil_Clay",             // 15 tropical red soil
    "Tundra_Rocky_Ground",   // 16 tundra
    "Lichened_Rocky_Ground", // 17 tundra, boreal outcrops
    "Mossy_Forest_Floor",    // 18 boreal forest floor
    "Icelandic_Glacier",     // 19 glaciers and ice caps
    "Soil_Mud",              // 20 wetlands, mangroves, tidal flats
    "Salt_Covered_Mud",      // 21 salt pans
    "Ploughed_Soil",         // 22 cropland
    "Icelandic_Lava_Sand",   // 23 volcanic sand
    "Sparse_Grassy_Soil",    // 24 steppe
    "Dried_Grass-57a98bb5",  // 25 savanna, dry grassland
];

/// Ground surfaces from the Fab library cache (Quixel scans exported as glTF with UE textures).
/// Override the root with SI_GROUND_DIR. Missing layers are None (procedural stand-in).
pub fn find_ground() -> Vec<Option<GroundMaterial>> {
    // Fab library cache: SI_GROUND_DIR at run time, else at build time (.cargo/config.toml [env]).
    let root = std::env::var("SI_GROUND_DIR").unwrap_or_else(|_| option_env!("SI_GROUND_DIR").unwrap_or("assets/ground").into());
    let mut out = Vec::new();
    for name in GROUND_LAYERS {
        let Some(dir) = std::fs::read_dir(&root).ok().and_then(|rd| {
            rd.filter_map(|e| e.ok()).map(|e| e.path()).find(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with(name)))
        }) else {
            out.push(None);
            continue;
        };
        let mut files = Vec::new();
        collect_pngs(&dir, &mut files, 0);
        let pick = |suffix: &str| files.iter().find(|p| p.to_string_lossy().ends_with(suffix)).cloned();
        match (pick("_B.png"), pick("_N.png"), pick("_ORM.png")) {
            (Some(albedo), Some(normal), Some(orm)) => out.push(Some(GroundMaterial { albedo, normal, orm })),
            _ => out.push(None),
        }
    }
    out
}

fn collect_pngs(dir: &Path, out: &mut Vec<PathBuf>, depth: u32) {
    if depth > 6 {
        return;
    }
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.filter_map(|e| e.ok()) {
            let p = e.path();
            if p.is_dir() {
                collect_pngs(&p, out, depth + 1);
            } else if p.extension().is_some_and(|x| x == "png") {
                out.push(p);
            }
        }
    }
}
