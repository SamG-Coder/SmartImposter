//! Biomes for the globe: RESOLVE Ecoregions 2017 rasterised by tools/bake_biomes.py
//! (assets/earth/ecoregions/eco_grid.bin, ~3.7 km cells, + eco_table.json). The biome picks which
//! species grow (forest.rs BIOME_TREES) and which ground scans cover the land (globe.rchit).
//!
//! Biome numbers (RESOLVE): 1 tropical moist broadleaf, 2 tropical dry broadleaf, 3 tropical
//! conifer, 4 temperate broadleaf and mixed, 5 temperate conifer, 6 boreal / taiga, 7 tropical
//! grassland and savanna, 8 temperate grassland, 9 flooded grassland, 10 montane grassland,
//! 11 tundra, 12 Mediterranean, 13 desert and xeric shrubland, 14 mangroves. 0: none (sea, rock
//! and ice).

use anyhow::{bail, Context, Result};
use std::io::Read;
use std::path::Path;

pub const TROPICAL_MOIST: u8 = 1;
pub const TROPICAL_DRY: u8 = 2;
pub const TROPICAL_CONIFER: u8 = 3;
pub const TEMPERATE_BROADLEAF: u8 = 4;
pub const TEMPERATE_CONIFER: u8 = 5;
pub const BOREAL: u8 = 6;
pub const SAVANNA: u8 = 7;
pub const TEMPERATE_GRASSLAND: u8 = 8;
pub const FLOODED: u8 = 9;
pub const MONTANE: u8 = 10;
pub const TUNDRA: u8 = 11;
pub const MEDITERRANEAN: u8 = 12;
pub const DESERT: u8 = 13;
pub const MANGROVES: u8 = 14;

pub struct Biomes {
    w: usize,
    h: usize,
    /// ECO_ID per cell, row 0 at 90 N
    grid: Vec<u16>,
    /// ECO_ID -> biome number
    biome_of: Vec<u8>,
    /// ECO_ID -> realm (Palearctic, Nearctic, Neotropic, Afrotropic, Indomalayan, Australasia,
    /// Oceania, Antarctica) as a small index
    realm_of: Vec<u8>,
}

pub const REALMS: [&str; 8] = ["Palearctic", "Nearctic", "Neotropic", "Afrotropic", "Indomalayan", "Australasia", "Oceania", "Antarctica"];

impl Biomes {
    pub fn load(dir: &Path) -> Result<Option<Self>> {
        let grid_path = dir.join("eco_grid.bin");
        if !grid_path.exists() {
            return Ok(None);
        }
        let b = std::fs::read(&grid_path)?;
        if b.len() < 16 || &b[..8] != b"SIECO1\0\0" {
            bail!("{}: not an ecoregion grid (tools/bake_biomes.py)", grid_path.display());
        }
        let w = u32::from_le_bytes(b[8..12].try_into()?) as usize;
        let h = u32::from_le_bytes(b[12..16].try_into()?) as usize;
        let mut raw = Vec::with_capacity(w * h * 2);
        flate2::read::ZlibDecoder::new(&b[16..]).read_to_end(&mut raw)?;
        if raw.len() != w * h * 2 {
            bail!("{}: truncated", grid_path.display());
        }
        let grid: Vec<u16> = bytemuck::pod_collect_to_vec(&raw);
        let table: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("eco_table.json")).context("eco_table.json")?)?;
        let (mut biome_of, mut realm_of) = (vec![0u8; 65536], vec![255u8; 65536]);
        if let Some(m) = table.as_object() {
            for (k, v) in m {
                let Ok(id) = k.parse::<usize>() else { continue };
                biome_of[id] = v.get(0).and_then(|x| x.as_u64()).unwrap_or(0) as u8;
                let realm = v.get(1).and_then(|x| x.as_str()).unwrap_or("");
                realm_of[id] = REALMS.iter().position(|r| *r == realm).map(|i| i as u8).unwrap_or(255);
            }
        }
        println!("Biomes: RESOLVE ecoregions {}x{} grid", w, h);
        Ok(Some(Biomes { w, h, grid, biome_of, realm_of }))
    }

    fn eco(&self, lat: f64, lon: f64) -> u16 {
        let x = (((lon + 180.0).rem_euclid(360.0)) / 360.0 * self.w as f64) as usize;
        let y = ((90.0 - lat) / 180.0 * self.h as f64) as usize;
        self.grid[y.min(self.h - 1) * self.w + x.min(self.w - 1)]
    }

    /// Biome number at lat / lon (0 = none).
    pub fn biome(&self, lat: f64, lon: f64) -> u8 {
        self.biome_of[self.eco(lat, lon) as usize]
    }

    /// Biogeographic realm index into REALMS (255 = none).
    pub fn realm(&self, lat: f64, lon: f64) -> u8 {
        self.realm_of[self.eco(lat, lon) as usize]
    }
}
