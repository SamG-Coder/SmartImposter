# SmartImposter Earth: a 1:1 globe

Goal: the whole planet at true scale. You can stand in a forest, climb to 10 km or fly out to
space, and everything you see is the real thing: real elevation, real rivers and coastlines, and
real plants, placed by the same growth rules we already use. No painted forests and no
backdrops. A tree you see 50 km away exists in the world and is still there when you fly to it.

This document covers four things: what has to be built, which global data it comes from, how far
you can see (which sets the rendering budget), and every asset needed per biome.

---

## 1. The numbers that drive the design

| Quantity | Value |
|---|---|
| Earth radius (mean) | 6,371 km |
| Surface | 510 million km² (land 149 million km², 29%) |
| Trees on Earth | ~3 trillion (Crowther et al. 2015) |
| Cube face edge on the sphere | ~10,000 km |
| float32 precision at one Earth radius | ~0.5 m, so GPU positions must be camera-relative |

How far you can see (horizon distance, √(2Rh)):

| Eye height | Horizon | Visible area |
|---|---|---|
| 1.7 m (standing) | 4.7 km | 70 km² |
| 100 m (tree tops / hill) | 36 km | 4,000 km² |
| 1 km (mountain / low flight) | 113 km | 40,000 km² |
| 10 km (airliner) | 357 km | 400,000 km² |
| 400 km (orbit) | 2,300 km | about 1/25 of the planet |

Mountains are visible beyond the horizon: a 3 km peak shows up from about 200 km away.

At 10 km altitude, 400,000 km² of land can be in view: up to tens of billions of trees.
Rendering every one of them individually at that range is impossible on any GPU. Showing them
honestly is still possible, because a 20 m crown 50 km away covers less than a pixel. That is
what hierarchical LOD (section 3.5) is for.

---

## 2. Global data (all free)

| Layer | Dataset | Resolution | Size (approx.) | Licence |
|---|---|---|---|---|
| Land elevation | Copernicus DEM GLO-30 | 30 m | ~26,000 1°×1° tiles, roughly 90–100 GB compressed (third-party estimate) | Copernicus, free incl. commercial (a few countries' tiles withheld) |
| Elevation + seafloor (global base) | NOAA ETOPO 2022 | 15″ (~450 m), 30″, 60″ | 60″ global grid ≈ 0.9 GB raw float32; 15″ in 288 tiles | Public domain |
| Land cover | ESA WorldCover 2021 v200 | 10 m, 11 classes | ~124 GB | CC BY 4.0 |
| Biomes / ecoregions | RESOLVE Ecoregions 2017 | vector, 846 ecoregions, 14 biomes | ~150 MB | CC BY 4.0 |
| Canopy height | Lang et al. 2023 (ETH, GEDI + Sentinel-2) | 10 m | very large (global 10 m raster), regional subsets possible | CC BY 4.0 |
| Canopy height (fine) | Meta / WRI 2024 | 1 m | very large; use for regions | free incl. commercial |
| Rivers | HydroRIVERS | vector, 8.5M reaches, catchments ≥ 10 km² | ~1–2 GB | free incl. commercial |
| Lakes | HydroLAKES | vector, 1.4M lakes ≥ 10 ha | <1 GB | free incl. commercial |
| Climate (species choice) | WorldClim 2.1 / Köppen-Geiger (Beck 2023) | ~1 km | a few GB | CC BY |
| Glaciers | Randolph Glacier Inventory 7 | vector | <1 GB | CC BY |
| Coastline | OpenStreetMap land polygons | vector | ~1 GB | ODbL |
| Optional: soil | SoilGrids 250 m | 250 m | large | CC BY 4.0 |

WorldCover classes: tree cover, shrubland, grassland, cropland, built-up, bare / sparse
vegetation, snow and ice, permanent water, herbaceous wetland, mangroves, moss and lichen.

What each layer drives:
- **Elevation:** the terrain shape. 30 m is the real measurement; finer detail (1–3 m) comes
  from our erosion and relief generator, constrained to match the 30 m data.
- **Rivers and lakes:** carved into the terrain from HydroRIVERS / HydroLAKES vectors. That gives
  real courses with widths from discharge. Below the 10 km² catchment limit, our own drainage
  extraction adds the small streams.
- **Land cover:** where forest, shrub, grass, crop, bare ground, wetland and ice are, at 10 m.
- **Canopy height:** how tall the trees are, which sets stand age and variant size in the
  growth sim.
- **Biome, ecoregion and climate:** which species grow there and in what mix (section 5).
- **Slope, aspect and wetness:** the local mix within a stand. These come from our existing
  succession and ground logic.

Raw data for the whole planet is a few hundred GB. It is converted once, offline, into our own
tile pyramid (section 3.3). The converted planet is roughly the same size, streamed from SSD.
**Start with one region** (e.g. the Alps or Scandinavia, a few GB) plus ETOPO for the rest of the
globe, then grow.

---

## 3. Engine architecture

### 3.1 Coordinates and precision
- World positions in **float64, Earth-centred (ECEF)** on the CPU.
- Every tile has a float64 anchor. The GPU only ever sees **camera-relative float32**: TLAS
  instance transforms are rebuilt each frame relative to the camera (we already rebuild the TLAS
  every frame).
- Local "up" is the sphere normal, so gravity, the sun and the sky all work from any point.

### 3.2 Globe terrain: cube-sphere quadtree
- 6 cube faces, each a quadtree. Map cube→sphere with an equal-area-ish warp so tiles stay
  similar in size.
- A tile is a 65×65 vertex grid. Level 0 is a whole face (~10,000 km); level 17 is ~76 m tiles
  at ~1.2 m vertex spacing.
- Selection by screen-space error: refine a tile while its geometric error exceeds ~1 pixel.
  About 1,000–2,000 tiles are active at once. Skirts or stitched edges prevent cracks.
- Each tile has its **own BLAS**, built when it streams in (on async compute) and kept in an
  LRU cache. The TLAS holds all active tiles.
- This replaces today's fixed ±60 km mesh centred on the start. It removes the "smooth coarse
  mountains when you fly away" problem, because detail always follows the camera.

### 3.3 Offline baker: data → tile pyramid
A command-line tool (Rust) that:
1. Reads the source rasters (COG GeoTIFF) and vectors.
2. Reprojects into cube-face tiles at each level.
3. Writes per tile:
   - height (int16 / f16, plus min/max)
   - land cover (u8)
   - canopy height (u8)
   - biome / ecoregion id (u16)
   - climate indices (u8 ×2)
   - river / lake / coast vector segments clipped to the tile
4. Compresses everything (zstd) into one pack file per face and level range, with an index.

The engine streams tiles from that pack on background threads.

### 3.4 Per-tile generation (deterministic, on the fly)
Everything finer than the data is generated from a seed = hash(tile id), so it is identical
every time you come back:
- **Terrain detail:**
  - erosion and relief amplification of the 30 m DEM down to ~1 m;
  - river channels, banks and floodplains carved along the vector rivers (today's river code);
  - lake shores.
- **Ground materials:** the biome's surface set (section 5), blended by slope, wetness, aspect
  and canopy cover (today's ground logic, extended per biome).
- **Plants:** today's succession sim, run per tile.
  - Inputs: land-cover class, canopy height, biome and climate, slope and wetness.
  - Process: founder lattice, seed dispersal, light competition, mortality and gaps.
  - Seams: each tile includes a margin from its neighbours, so stands don't line up on tile
    edges.
  - Cost: our sim grows ~6M trees over 200 km² in ~20 s on the CPU. A 1 km² tile (~20–60k trees)
    takes a few ms, so tiles can be grown as you move.

### 3.5 Seeing every tree at every distance: hierarchical LOD
The forest uses the same quadtree as the terrain:

| Distance | Representation | Built from |
|---|---|---|
| 0–60 m | Full meshes (today) | the tree's mesh |
| 60 m – ~3 km | Per-tree parallax impostors, one AABB per tree in cluster BLASes (today) | the tree's impostor |
| ~3 km – horizon | **Tile canopy proxies**: per quadtree tile, a canopy-top height field, crown colour, normal and gap mask, ray-marched like our impostors | **rasterised from that tile's actual generated trees** (positions, crowns, species) |

Each parent tile's proxy is baked by downsampling its children's proxies. The tree you see
40 km away is therefore the same tree, in the same place, with the same crown shape and colour,
summed into fewer pixels. This is the honest version of distance rendering: no noise patterns
and no invented crowns. Microsoft Flight Simulator 2024 likewise drives vegetation from
land-class data with ~30 modelled biomes.

Budget: ~15–30M individual tree impostors near the camera (we render ~6M today at 4–9 ms on a
5080), plus a few thousand canopy-proxy tiles to the horizon.

### 3.6 Water
- **Rivers:** surfaces from the vector network.
  - Shading: refraction to the riverbed, depth-based absorption and scattering.
  - Flow: normal maps scrolled along the flow direction (flow mapping). Speed comes from slope
    and discharge.
  - Edges: foam and wet banks at the waterline.
  - This is how Unreal's Water system does rivers: spline-based water bodies carry width,
    depth and velocity, carve the landscape, and the material uses the velocity for flow.
- **Lakes:** flat surfaces at the HydroLAKES level.
- **Ocean:** sea level on the sphere.
  - Waves: FFT wave spectrum near the camera, fading to a spectrum normal map far away.
  - Depth: bathymetry from ETOPO sets water colour and shoreline breaking.

### 3.7 Sky and atmosphere
Physically based atmosphere (Hillaire 2020 / Bruneton) that works from the ground to orbit,
replacing today's flat haze. Volumetric clouds come later.

---

## 4. Milestones

| # | Milestone | Result |
|---|---|---|
| 1 | Globe skeleton | Cube-sphere quadtree terrain from ETOPO 60″, float64 camera, tile streaming and per-tile BLAS, atmosphere. Fly from orbit to the ground. |
| 2 | Real elevation | GLO-30 for a first region (then global); detail amplification; rivers and lakes from HydroSHEDS; coastlines. |
| 3 | Land classification | Baker adds WorldCover, RESOLVE biomes, Köppen and canopy height; per-biome ground materials. |
| 4 | Streaming forests | Per-tile deterministic growth from the data, individual impostors near the camera, canopy-proxy HLOD to the horizon. |
| 5 | Plant library | Plant functional types for every biome from free Quixel only (section 5); converter for Megascans glTF plants; per-ecoregion tint and scale for stand-ins. |
| 6 | Water | Rivers and lakes with refraction, flow and foam; then the ocean. |
| 7 | Polish | Seasons (leaf colour and snow from latitude and date), clouds, night sky. |

Milestone 1 needs only ETOPO (a single download, under 1 GB) and no new assets.

---

## 5. Assets for every biome

**Approach:** Earth has ~60,000 tree species. Nobody models them all, Flight Simulator included.
We use **plant functional types**, about 4–8 archetype species per biome, each with several age
variants, tinted and scaled per ecoregion. The species mix comes from climate and ecoregion.

**Rule: free Quixel (Megascans / Megaplants) only.** Where Quixel has no scan of a species, the
closest free Quixel plant stands in for it, retinted and rescaled per ecoregion (→ below). That
gives about 50–60 Quixel assets for the whole planet.

Legend:
- ✓ = free Quixel asset confirmed on Fab (searched Nov 2026).
- → X = no Quixel scan of this species; free Quixel asset X stands in.

Search links go to Fab.

### Already in the project
European Beech, Black Alder, Black Poplar, European Aspen, Common Hazel, Bracken Fern, Heather,
Huckleberry Oak, and 7 grasses. Ground: Forest Floor, Mossy Grass, Uncut Grass, Dried Grass, two
Rock Cliffs, Snow Cliff, Icelandic Gravel, Mossy Rock, Mossy Riverbed, Round Pebbles, Wet Grass
Ground.

### 5.1 Temperate broadleaf and mixed forests (Europe, E. USA, E. Asia)
- **Canopy:**
  - Beech, Oak ✓ (English Oak), Hornbeam ✓ (European Hornbeam)
  - Silver Birch ✓, Black Poplar, Goat Willow ✓, Norway Maple
  - E. Asia: Ginkgo ✓, Yoshino Cherry ✓
- **Understory:** Hazel, Elder ✓, European Spindle ✓, saplings ✓ (Beech, Norway Maple, Rowan,
  Aspen)
- **Ground cover:** Bilberry ✓, Wood Anemone ✓, Lady Fern, Wood Sorrel, Bramble, Ivy
- **Surfaces:** forest floor / leaf litter, moss, mud, forest path
- **Debris:** fallen logs, stumps, branches, mushrooms

### 5.2 Temperate conifer forests (Alps, Rockies, Pacific NW, Japan)
- **Canopy:**
  - Norway Spruce ✓, Baltic (Scots) Pine ✓, Dead Pine ✓
  - Silver Fir, Douglas Fir → Norway Spruce (taller, darker)
  - European Larch → Baltic Pine (lighter green, gold in autumn)
  - Western Red Cedar, Coast Redwood → Norway Spruce scaled up, with the Japanese Cedar trunk scan
  - Japanese Cedar ✓ (trunk scan only)
- **Understory:** spruce and pine saplings ✓, juniper → Utah Juniper ✓, rhododendron → European Spindle ✓
- **Ground:** Lingonberry ✓, Bilberry ✓, mosses, needle litter
- **Surfaces:** needle forest floor, mossy forest floor ✓, rocky ground

### 5.3 Boreal forest / taiga (Canada, Scandinavia, Siberia)
- **Canopy:**
  - Norway Spruce ✓, Scots Pine ✓, Silver Birch ✓, Aspen
  - Black / White Spruce → Norway Spruce (narrower, smaller), Siberian Larch → Baltic Pine
- **Ground:** Lingonberry ✓, Bilberry ✓, Labrador tea, crowberry, feather moss, reindeer
  lichen
- **Surfaces:** boreal moss, lichen ground ✓ (Lichened Rocky Ground), bog / peat, Nordic forest
  floor
- **Debris:** Nordic forest logs and uprooted trees ✓

### 5.4 Tundra (Arctic)
- **Plants:**
  - Tundra Grass ✓, cotton grass (have)
  - dwarf birch, arctic willow → Goat Willow and Silver Birch saplings ✓ scaled down, Heather (have)
  - mosses, lichens
- **Surfaces:** Tundra Rocky Ground ✓, Icelandic Rocky Ground ✓, Lichened Rocky Ground ✓,
  permafrost polygons, snow
- **Rocks:** Tundra Lichened Rock and Boulder ✓, Tundra Mossy Boulder ✓, Massive Tundra Rock
  Formation ✓

### 5.5 Montane grasslands, shrublands and alpine zones
- **Plants:**
  - alpine grasses (Tundra Grass ✓), dwarf pine (Pinus mugo) → Baltic Pine Saplings ✓, juniper → Utah Juniper ✓
  - Tibetan / Andean puna grass → Tufted Grass (have)
- **Surfaces:** alpine meadow, scree, rock cliffs (have), snow and ice (have Snow Cliff), glacier
  ice (search "glacier ice" / "ice" surfaces)

### 5.6 Temperate grasslands, savannas and shrublands (steppe, prairie, pampas)
- **Plants:**
  - Wheat Grass ✓, Wild Grass ✓, Tall Grass ✓, Tufted Grass ✓, Ribbon Grass ✓
  - sagebrush → Greasewood ✓ (Megaplants), wildflowers ✓ (Field Poppy, Dandelions)
- **Trees (sparse):** Bur Oak → English Oak ✓, poplars along rivers
- **Surfaces:** dry grassland soil, steppe dirt, chernozem cropland

### 5.7 Mediterranean forests, woodlands and scrub
- **Canopy:** Aleppo Pine ✓, Holm / Cork Oak → English Oak ✓ (dark, small), Olive → English Oak
  young variant (grey-green), Stone Pine → Aleppo Pine, Cypress → Utah Juniper ✓ (tall, narrow)
- **Shrubs (maquis / chaparral):** lavender, rosemary, rockrose → Heather and Huckleberry Oak (have),
  Greasewood ✓
- **Surfaces:** dry red soil, limestone rock, dry grass, gravel

### 5.8 Deserts and xeric shrublands
- **Plants:**
  - Desert American cacti ✓ (many: branch, cluster, small and medium)
  - Desert Acacia ✓, Utah Juniper ✓, Greasewood ✓
  - Joshua tree → Utah Juniper / branch cactus, creosote bush → Greasewood ✓, date palm → Tropical Palm ✓
- **Surfaces:** sand dunes, desert pavement, cracked clay / playa, salt flats, red sandstone,
  Desert Western rocks ✓ (Desert Western Ledge Rock)
- **Coastal dunes:** Marram Grass ✓

### 5.9 Tropical and subtropical grasslands and savannas (Africa, Cerrado, N. Australia)
- **Plants:**
  - Savanna Dry Grass ✓ (several), Desert Acacia ✓ (umbrella acacia stand-in)
  - baobab, African acacia → Desert Acacia ✓ (baobab is the weakest match)
  - eucalyptus → Silver Birch ✓ scaled up (pale trunk, blue-grey tint), termite mounds → Desert
    Western rocks ✓ (red tint)
- **Surfaces:** red laterite soil, dry savanna ground, burnt ground

### 5.10 Tropical and subtropical moist broadleaf forests (rainforest)
- **Canopy:**
  - emergent and canopy trees (kapok, dipterocarps, ficus) → Black Poplar ✓ and Ginkgo ✓ crowns,
    tinted deep green, with the Buttress Root ✓ and Ficus Microcarpa trunk ✓ scans at the base.
    The weakest biome: rainforest will look the least accurate.
  - Giant Bamboo ✓ (Megaplants)
- **Palms:** Alexandra Palm ✓, Tropical Palm ✓, Chinese Fan Palm ✓, Chinese Windmill Palm ✓,
  Broadleaf Lady Palm ✓, Cat Palm ✓, Zamia Palm ✓
- **Understory:** Xanadu ✓, tree ferns → Zamia / Cat Palm ✓, banana and heliconia → Broadleaf
  Lady Palm ✓ and Xanadu ✓, lianas and vines (Forest Japanese Tree Vine ✓)
- **Surfaces:** tropical leaf litter ✓ (Palm Leaves Dead, Damaged Palm Leaves), red clay, mossy
  roots

### 5.11 Tropical and subtropical dry broadleaf and coniferous forests
- **Trees:** teak, sal → English Oak / European Hornbeam ✓ (dry-season tint), Caribbean /
  Mexican pines → Baltic Pine ✓, bamboo ✓
- **Surfaces:** dry tropical leaf litter, laterite

### 5.12 Flooded grasslands and wetlands (Pantanal, Okavango, Sudd)
- **Plants:**
  - Common Bulrush ✓ (Megaplants), Dry Common Reed ✓, papyrus → Common Bulrush
  - water lilies ✓ (Rembrandt, Carolana), Water Cabbage ✓
- **Surfaces:** Swamp Water ✓, mud, wet grass (have)

### 5.13 Mangroves
- **Trees:** red and black mangroves → Goat Willow / Black Alder ✓ crowns on the Buttress Root ✓
  scan, standing in water (a rough match)
- **Surfaces:** tidal mud flats, Marine Couch ✓ (coastal grass)

### 5.14 Lakes, rivers, coasts and oceans
- **River:** River Debris ✓, River Bank Rocks ✓, pebbles (have), riverbed (have)
- **Lakes:** reeds, lilies (above), shore gravel and sand
- **Coasts:** beach sand, shingle, wet sand, sea cliffs, kelp and wrack (search Quixel "seaweed"), Marram Grass ✓
- **Ocean:** procedural (FFT waves); no textures needed

### 5.15 Ice, snow and bare land
- Snow Cliff (have), glacier ice and old snow (search Quixel "ice" / "snow" surfaces), crevasses
  (procedural)
- volcanic rock, lava fields, black sand (Iceland ✓: Icelandic Rock and Rocky Ground)

### 5.16 Human land (WorldCover cropland and built-up)
- **Cropland:** wheat ✓ (Wheat Grass), maize → Tall Grass (have), rice paddies → Wheat Grass in
  shallow water, plough furrow soil, field edges
  and hedgerows (use the temperate shrubs)
- **Built-up:** out of scope for now. Render as cleared ground or flag it; cities are a separate
  project (OpenStreetMap buildings).

### Summary
- **Free Quixel covers well:** temperate, boreal, tundra, alpine, Mediterranean pine, desert,
  grasslands, savanna grass, palms, bamboo, wetlands, river and coast edges.
- **Covered by stand-ins:** firs, larch, redwood, cedars, Mediterranean oaks and olive,
  eucalyptus, acacias and baobab. Close in silhouette once retinted and rescaled.
- **Weakest:** tropical rainforest canopy and mangroves. Quixel has no scans of those trees, so
  the stand-ins are approximate. Revisit if Quixel adds them.

---

## Sources
- [Copernicus DEM on AWS](https://registry.opendata.aws/copernicus-dem/) and
  [readme](https://copernicus-dem-30m.s3.amazonaws.com/readme.html); size estimate from a
  [derivative's README](https://huggingface.co/datasets/aliasfox/copernicus-glo30-chunks/blob/main/README.md)
- [NOAA ETOPO 2022](https://www.ncei.noaa.gov/products/etopo-global-relief-model)
- [ESA WorldCover 2021 v200 (Zenodo)](https://zenodo.org/records/7254221)
- [RESOLVE Ecoregions 2017](https://www.geographyrealm.com/terrestrial-ecoregions-gis-data/)
- [HydroRIVERS](https://hydrosheds.org/page/hydrorivers),
  [HydroSHEDS products](https://hydrosheds.org/products)
- [Lang et al. 2023, global 10 m canopy height](https://research-collection.ethz.ch/handle/20.500.11850/636572),
  [model code](https://github.com/langnico/global-canopy-height-model)
- [Meta / WRI 1 m canopy height](https://sustainability.atmeta.com/?p=5573)
- Cube-sphere planets: [acko.net "Making Worlds"](https://acko.net/blog/making-worlds-1-of-spheres-and-cubes/),
  [planetary terrain and precision](https://ralith.com/blog/planetary-terrain/),
  [Cesium quantized-mesh](https://cesium.com/blog/2015/12/18/terrain-quantization/)
- [MSFS 2024: procedural vegetation, 30 biomes](https://pcgamer.com/games/sim/more-details-on-ms-flight-simulator-2024-including-full-3d-landscapes-in-30-biomes)
- Fab availability: searches on [fab.com](https://www.fab.com/search?q=megaplants), Nov 2026
