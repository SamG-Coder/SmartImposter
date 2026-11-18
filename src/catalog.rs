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
    pub name: String,
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
            let texel_scale = if kind == Kind::Grass { 1.6 } else { 1.0 };
            if kind == Kind::Grass {
                // Blades are thinner than an atlas texel: keep patches as real cards to ~20 m,
                // where the blades are sub-pixel and the impostor reads as the same sward.
                lod_scale = lod_scale.max(0.4);
            }
            cat.variants.push(Variant { species: si, kind, height, lod_scale, cull, texel_scale });
            pack.trees.push(t);
        }
        cat.mesh_species.extend(std::iter::repeat(si as u32).take(p.meshes.len()));
        cat.mesh_alpha.extend(std::iter::repeat(alpha).take(p.meshes.len()));
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

/// Ground layers in shader order (tri.rchit): Fab library folder name prefix of each Megascans
/// surface (`<Name>-<first 8 hex digits of the Fab listing id>`).
pub const GROUND_LAYERS: [&str; 12] = [
    "Forest_Floor",
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
            (Some(albedo), Some(normal), Some(orm)) => out.push(Some(GroundMaterial { name: name.to_string(), albedo, normal, orm })),
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
