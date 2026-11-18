//! Biome presets for the generated (eroded) terrain: `--biome boreal|rainforest|savanna|...`.
//!
//! The forest scene's terrain comes from the erosion simulation (erosion.rs) whatever the biome;
//! a preset decides what grows on it and how the ground looks:
//!
//! * the canopy: the seed-dispersal succession (world.rs grow_forest) works with six ecological
//!   roles (shade-tolerant climax trees, trees of wet ground, edge pioneers, clonal groves, an
//!   understory tree, and others); a preset fills the roles with its own species, and scales how
//!   much land is forest and how far apart the trees stand (open savanna, oasis palms only along
//!   the rivers);
//! * the understory: shrubs, ferns / forest plants and grasses, each with the habitat it grows in
//!   (open ground, forest edges, under the canopy; moist, dry or wet grass), boulder fields and
//!   deadwood;
//! * the ground: the forest scene blends 12 ground slots (forest floor, moss, grass, dry grass,
//!   the mountain rock / snow / scree layers, river mud, pebbles, wet grass); a preset maps each
//!   slot to any of the 26 scanned surfaces (sand, arid soil, red clay, tundra, lichen, boreal
//!   moss...), see tri.rchit groundMaterial;
//! * the climate: plants' dryness tint, how dry and wet the ground is, an altitude shift of the
//!   treeline / scree / snow zones, and a default sun and haze.

use std::sync::OnceLock;

/// Where a shrub or forest plant grows.
#[derive(Clone, Copy, PartialEq)]
pub enum Habitat {
    /// open ground, more where it is dry (heather, desert shrubs)
    Open,
    /// forest edges and gaps
    Edge,
    /// under the canopy
    Under,
}

/// Which grass patches go where.
#[derive(Clone, Copy, PartialEq)]
pub enum Grass {
    Moist,
    Dry,
    Wet,
    Tall,
}

pub struct Preset {
    pub name: &'static str,
    pub title: &'static str,
    /// species (catalog name substrings) per succession role: shade-tolerant climax, wet ground,
    /// edge pioneer, clonal groves, understory tree, other
    pub roles: [&'static [&'static str]; 6],
    /// how much of the dry land is forest (1 = the temperate default) and how much of the river
    /// banks (riparian strips, oases)
    pub forest: f32,
    pub riparian: f32,
    /// how close to the water the riparian strip starts (bank proximity 0..1: 0.25 a broad
    /// floodplain belt, 0.75 a ribbon at the water's edge)
    pub riparian_from: f32,
    /// minimum spacing between trees x this (open woodland, savanna)
    pub spacing: f32,
    /// tree size x this (dwarf birches and willows in the tundra)
    pub tree_scale: f32,
    /// shrubs (step 3.4 m) and forest plants (step 2.1 m): name, weight, habitat
    pub shrubs: &'static [(&'static str, f32, Habitat)],
    pub plants: &'static [(&'static str, f32, Habitat)],
    /// how much of the shrub and plant layers is filled (1 = default)
    pub shrub_cover: f32,
    pub plant_cover: f32,
    /// grass patches (step 2.3 m) and how much of the open ground they cover (1 = meadow)
    pub grass: &'static [(&'static str, f32, Grass)],
    pub grass_cover: f32,
    /// boulder fields and deadwood (logs, snags): 0 none, 1 plenty
    pub rocks: f32,
    pub deadwood: f32,
    /// the 12 ground slots of tri.rchit -> scanned surfaces (catalog.rs GROUND_LAYERS indices)
    pub ground: [u8; 12],
    /// plant dryness tint 0..15 (rt.glsl plantDry)
    pub dry_tint: u8,
    /// ground dryness added (0..1) and moisture scale
    pub dry_bias: f32,
    pub wet_scale: f32,
    /// the treeline, alpine meadow, scree and snow zones move up by this many metres (as a real
    /// place's latitude shift, world.rs Site::alt_shift; negative: down, as in the tundra)
    pub alt_shift: f32,
    /// default sun (azimuth, elevation in degrees) and aerial haze, when not given on the command line
    pub sun: [f32; 2],
    pub haze: f32,
    /// camera exposure x this (bright sand and dry grass want less than a dark forest)
    pub exposure: f32,
}

// catalog.rs GROUND_LAYERS indices
const FLOOR: u8 = 0;
const MOSS: u8 = 1;
const GRASS: u8 = 2;
const DRY: u8 = 3;
const MUD: u8 = 9;
const PEBBLES: u8 = 10;
const WET_GRASS: u8 = 11;
const SAND: u8 = 13;
const ARID: u8 = 14;
const CLAY: u8 = 15;
const TUNDRA: u8 = 16;
const LICHEN: u8 = 17;
const BOREAL_FLOOR: u8 = 18;
const SOIL_MUD: u8 = 20;
const SALT: u8 = 21;
const STEPPE: u8 = 24;
const DRY_GRASS: u8 = 25;
/// the mountain layers (rock cliff, dark rock, snow, scree, lichen rock) stay themselves
const fn ground(floor: u8, moss: u8, grass: u8, dry: u8, mud: u8, pebbles: u8, wet: u8) -> [u8; 12] {
    [floor, moss, grass, dry, 4, 5, 6, 7, 8, mud, pebbles, wet]
}

use Grass::*;
use Habitat::*;

pub const PRESETS: &[Preset] = &[
    Preset {
        name: "temperate",
        title: "Temperate broadleaf forest",
        roles: [&["European_Beech"], &["Black_Alder"], &["Black_Poplar"], &["European_Aspen"], &["Common_Hazel"], &[]],
        forest: 1.0,
        riparian: 1.0,
        riparian_from: 0.25,
        spacing: 1.0,
        tree_scale: 1.0,
        shrubs: &[("Heather", 1.6, Open), ("Huckleberry", 0.9, Edge)],
        plants: &[("Bracken", 1.0, Under)],
        shrub_cover: 1.0,
        plant_cover: 1.0,
        grass: &[
            ("Wild_Grass", 1.0, Moist),
            ("Grass_Clumps", 1.0, Moist),
            ("Dry_Grass", 1.0, Dry),
            ("Cotton_Grass", 1.0, Wet),
            ("Tall_Grass", 1.0, Tall),
            ("Tufted_Grass", 0.7, Tall),
        ],
        grass_cover: 1.0,
        rocks: 0.15,
        deadwood: 0.3,
        ground: ground(FLOOR, MOSS, GRASS, DRY, MUD, PEBBLES, WET_GRASS),
        dry_tint: 0,
        dry_bias: 0.0,
        wet_scale: 1.0,
        alt_shift: 0.0,
        sun: [35.0, 38.0],
        haze: 1.0,
        exposure: 1.0,
    },
    Preset {
        name: "boreal",
        title: "Boreal forest (taiga)",
        roles: [&["Norway_Spruce"], &["Goat_Willow"], &["Silver_Birch"], &["European_Aspen"], &["Rowan"], &["Baltic_Pine"]],
        forest: 1.15,
        riparian: 0.8,
        riparian_from: 0.25,
        spacing: 1.0,
        tree_scale: 1.0,
        shrubs: &[("Heather", 1.4, Open), ("Heather", 0.5, Edge)],
        plants: &[("Bilberry", 1.6, Under), ("Bracken", 0.35, Under), ("Bilberry", 0.5, Open)],
        shrub_cover: 1.0,
        plant_cover: 1.4,
        grass: &[("Tufted_Grass", 1.0, Moist), ("Wild_Grass", 0.5, Moist), ("Tundra_Grass", 0.8, Dry), ("Cotton_Grass", 1.6, Wet), ("Tall_Grass", 0.4, Tall)],
        grass_cover: 0.85,
        rocks: 0.5,
        deadwood: 1.0,
        ground: ground(BOREAL_FLOOR, MOSS, GRASS, LICHEN, SOIL_MUD, PEBBLES, WET_GRASS),
        dry_tint: 0,
        dry_bias: -0.05,
        wet_scale: 1.15,
        alt_shift: -550.0,
        sun: [200.0, 24.0],
        haze: 0.8,
        exposure: 1.0,
    },
    Preset {
        name: "rainforest",
        title: "Tropical rainforest",
        roles: [&["English_Oak", "European_Beech", "Norway_Maple"], &["Tropical_Palm", "Black_Alder"], &["Giant_Bamboo"], &["Chinese_Fan_Palm", "Elder"], &["Common_Hazel"], &["Tropical_Palm"]],
        forest: 1.5,
        riparian: 1.2,
        riparian_from: 0.25,
        spacing: 0.9,
        tree_scale: 1.0,
        shrubs: &[
            ("Alexandra_Palm", 1.0, Under),
            ("Lady_Palm", 1.0, Under),
            ("Cat_Palm", 0.8, Under),
            ("Chinese_Fan_Palm", 0.6, Edge),
            ("Tropical_Palm", 1.2, Edge),
        ],
        plants: &[("Leather_Fern", 1.2, Under), ("Xanadu", 1.0, Under), ("Zamia_Palm", 0.6, Edge), ("Bracken", 0.3, Under)],
        shrub_cover: 1.6,
        plant_cover: 1.6,
        grass: &[("Tall_Grass", 1.0, Tall), ("Ribbon_Grass", 0.8, Moist), ("Giant_Reed", 1.4, Wet), ("Wild_Grass", 0.5, Moist)],
        grass_cover: 0.8,
        rocks: 0.1,
        deadwood: 0.4,
        ground: ground(FLOOR, MOSS, GRASS, CLAY, SOIL_MUD, PEBBLES, WET_GRASS),
        dry_tint: 0,
        dry_bias: -0.25,
        wet_scale: 1.4,
        alt_shift: 900.0,
        sun: [80.0, 55.0],
        haze: 1.25,
        exposure: 1.25,
    },
    Preset {
        name: "savanna",
        title: "Savanna",
        roles: [&["English_Oak"], &["Black_Poplar", "Goat_Willow"], &["Aleppo_Pine"], &["Tropical_Palm"], &[], &[]],
        forest: 0.42,
        riparian: 1.0,
        riparian_from: 0.25,
        spacing: 3.2,
        tree_scale: 1.0,
        shrubs: &[("Greasewood", 0.8, Open), ("Dead_Shrubs", 0.6, Open), ("Spineless_Yucca", 0.2, Open)],
        plants: &[],
        shrub_cover: 0.12,
        plant_cover: 0.0,
        grass: &[("Dry_Grass", 1.8, Dry), ("Dry_Grass", 0.8, Moist), ("Tall_Grass", 1.0, Tall), ("Wheat_Grass", 0.8, Dry), ("Giant_Reed", 1.0, Wet)],
        grass_cover: 1.1,
        rocks: 0.35,
        deadwood: 0.15,
        ground: ground(ARID, STEPPE, DRY_GRASS, DRY_GRASS, SOIL_MUD, PEBBLES, GRASS),
        dry_tint: 8,
        dry_bias: 0.35,
        wet_scale: 0.7,
        alt_shift: 1500.0,
        sun: [255.0, 22.0],
        haze: 1.4,
        exposure: 0.55,
    },
    Preset {
        name: "desert",
        title: "Desert with river oases",
        // a ribbon of willows at the water (as the tugai woods of the Central Asian deserts),
        // palms at its edges
        roles: [&[], &["Goat_Willow"], &[], &[], &[], &[]],
        forest: 0.0,
        riparian: 1.2,
        riparian_from: 0.72,
        spacing: 1.3,
        tree_scale: 1.0,
        shrubs: &[
            ("Greasewood", 1.0, Open),
            ("Dead_Shrubs", 1.0, Open),
            ("Spineless_Yucca", 0.5, Open),
            ("Cacti", 0.8, Open),
            ("Tropical_Palm", 6.0, Edge),
            ("Chinese_Fan_Palm", 4.0, Edge),
        ],
        plants: &[],
        shrub_cover: 0.07,
        plant_cover: 0.0,
        grass: &[("Dry_Grass", 1.0, Dry), ("Marram_Grass", 0.6, Dry), ("Giant_Reed", 1.2, Wet)],
        grass_cover: 0.18,
        rocks: 0.3,
        deadwood: 0.0,
        ground: ground(ARID, STEPPE, ARID, SAND, SALT, PEBBLES, STEPPE),
        dry_tint: 15,
        dry_bias: 0.7,
        wet_scale: 0.35,
        alt_shift: 3000.0,
        sun: [240.0, 20.0],
        haze: 1.5,
        exposure: 0.42,
    },
    Preset {
        name: "tundra",
        title: "Arctic tundra",
        roles: [&[], &["Goat_Willow"], &["Silver_Birch"], &[], &[], &[]],
        forest: 0.3,
        riparian: 0.6,
        riparian_from: 0.25,
        spacing: 2.0,
        tree_scale: 0.55,
        shrubs: &[("Heather", 1.5, Open), ("Dead_Shrubs", 0.3, Open)],
        plants: &[("Bilberry", 1.4, Open), ("Bilberry", 0.6, Under)],
        shrub_cover: 0.9,
        plant_cover: 0.6,
        grass: &[("Tundra_Grass", 1.6, Moist), ("Tundra_Grass", 1.0, Dry), ("Cotton_Grass", 2.0, Wet), ("Tufted_Grass", 0.5, Tall)],
        grass_cover: 0.95,
        rocks: 1.0,
        deadwood: 0.0,
        ground: ground(TUNDRA, LICHEN, MOSS, TUNDRA, SOIL_MUD, PEBBLES, WET_GRASS),
        dry_tint: 3,
        dry_bias: 0.1,
        wet_scale: 1.1,
        alt_shift: -950.0,
        sun: [180.0, 12.0],
        haze: 0.7,
        exposure: 0.85,
    },
    Preset {
        name: "mediterranean",
        title: "Mediterranean woodland",
        roles: [&["Aleppo_Pine"], &["Black_Poplar"], &["English_Oak"], &[], &["Elder"], &["Baltic_Pine"]],
        forest: 0.75,
        riparian: 1.0,
        riparian_from: 0.25,
        spacing: 1.5,
        tree_scale: 1.0,
        shrubs: &[("Heather", 1.0, Open), ("Greasewood", 0.8, Open), ("European_Spindle", 0.6, Edge), ("Huckleberry", 0.4, Edge)],
        plants: &[("Bracken", 0.3, Under)],
        shrub_cover: 1.2,
        plant_cover: 0.4,
        grass: &[("Dry_Grass", 1.6, Dry), ("Wild_Grass", 0.6, Moist), ("Field_Poppy", 0.3, Moist), ("Tall_Grass", 0.6, Tall), ("Giant_Reed", 0.8, Wet)],
        grass_cover: 0.9,
        rocks: 0.5,
        deadwood: 0.1,
        ground: ground(ARID, STEPPE, DRY_GRASS, ARID, SOIL_MUD, PEBBLES, GRASS),
        dry_tint: 9,
        dry_bias: 0.4,
        wet_scale: 0.75,
        alt_shift: 700.0,
        sun: [250.0, 28.0],
        haze: 1.2,
        exposure: 0.65,
    },
    Preset {
        name: "alpine",
        title: "Alpine conifer forest",
        roles: [&["Norway_Spruce"], &["Black_Alder", "Goat_Willow"], &["Baltic_Pine"], &[], &["Rowan"], &[]],
        forest: 1.1,
        riparian: 0.9,
        riparian_from: 0.25,
        spacing: 1.0,
        tree_scale: 1.0,
        shrubs: &[("Heather", 1.2, Open), ("Huckleberry", 0.4, Edge)],
        plants: &[("Bilberry", 1.0, Under), ("Bracken", 0.5, Under), ("Wood_Anemone", 0.4, Edge)],
        shrub_cover: 1.0,
        plant_cover: 1.1,
        grass: &[("Wild_Grass", 1.0, Moist), ("Grass_Clumps", 0.8, Moist), ("Leafy_Dandelions", 0.25, Moist), ("Dry_Grass", 0.6, Dry), ("Cotton_Grass", 1.0, Wet), ("Tufted_Grass", 0.8, Tall)],
        grass_cover: 1.0,
        rocks: 0.6,
        deadwood: 0.6,
        ground: ground(FLOOR, MOSS, GRASS, DRY, MUD, PEBBLES, WET_GRASS),
        dry_tint: 0,
        dry_bias: 0.0,
        wet_scale: 1.0,
        alt_shift: -450.0,
        sun: [150.0, 30.0],
        haze: 0.85,
        exposure: 0.95,
    },
];

static CURRENT: OnceLock<&'static Preset> = OnceLock::new();

/// The preset of this run (the temperate default unless `--biome` chose another).
pub fn current() -> &'static Preset {
    CURRENT.get().copied().unwrap_or(&PRESETS[0])
}

pub fn find(name: &str) -> Option<&'static Preset> {
    let n = name.to_ascii_lowercase();
    PRESETS.iter().find(|p| p.name == n)
}

pub fn names() -> String {
    PRESETS.iter().map(|p| p.name).collect::<Vec<_>>().join(", ")
}

pub fn set(p: &'static Preset) {
    let _ = CURRENT.set(p);
}

impl Preset {
    /// The ground slot map for the shaders: 12 scan indices, one byte each (Globals.biome).
    pub fn ground_packed(&self) -> [u32; 4] {
        let mut out = [0u32; 4];
        for (i, s) in self.ground.iter().enumerate() {
            out[i / 4] |= (*s as u32) << (8 * (i % 4));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_are_well_formed() {
        assert_eq!(PRESETS[0].name, "temperate");
        let mut names: Vec<&str> = PRESETS.iter().map(|p| p.name).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), PRESETS.len(), "duplicate preset names");
        for p in PRESETS {
            assert!(p.ground.iter().all(|g| (*g as usize) < crate::catalog::GROUND_LAYERS.len()), "{}", p.name);
            assert!(p.dry_tint <= 15, "{}", p.name);
            assert!(p.roles.iter().any(|r| !r.is_empty()), "{} grows no trees", p.name);
        }
        // the identity map for the default
        let g = PRESETS[0].ground_packed();
        assert_eq!(g[0] & 0xFF, 0);
        assert_eq!((g[2] >> 24) & 0xFF, 11);
        assert_eq!(find("Desert").map(|p| p.name), Some("desert"));
    }
}
