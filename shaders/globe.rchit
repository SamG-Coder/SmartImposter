#version 460
#extension GL_GOOGLE_include_directive : require
#include "globe.glsl"

// Terrain tiles: geometry, land cover and the ground material (tiles.rs vertex: position (3),
// normal (3), elevation, land cover (3 x unorm4x8), texture coordinates in metres (2)).
layout(set = 0, binding = 30) uniform GG { GlobeGlobals gg; };
layout(set = 0, binding = 3) uniform GlobalsUBO { Globals g; };
#include "roads.glsl"
layout(set = 0, binding = 31, scalar) readonly buffer TileTable { uint64_t vtx[]; } table;
layout(buffer_reference, scalar) readonly buffer Verts { float v[]; };
layout(buffer_reference, scalar) readonly buffer Indices { uint i[]; };
layout(set = 0, binding = 32) uniform sampler2D elevTex; // ETOPO 2022 60", R16_SNORM metres / 32767

layout(location = 0) rayPayloadInEXT HitPayload pl;
hitAttributeEXT vec2 bary;

// Elevation (m) at a direction from the planet centre, at a mip for the pixel footprint (m).
const float TEXEL_M = 1853.0; // one arc-minute on the equator
vec2 elevUV(vec3 up) {
    float lat = asin(clamp(up.z, -1.0, 1.0));
    float lon = atan(up.y, up.x);
    return vec2(lon / (2.0 * PI) + 0.5, 0.5 - lat / PI);
}
float elevAt(vec2 uv, float lod) { return textureLod(elevTex, uv, lod).r * 32767.0; }

// ---- ground: Quixel scans (catalog.rs GROUND_LAYERS) blended by land cover, hex-tiled
layout(set = 0, binding = 16) uniform sampler2D groundTex[MAX_GROUND_TEX]; // the renderer's ground scans (3 per layer)
const float GT = 2.048; // metres per texture repeat (divides tiles.rs TEX_PERIOD)
// The scans' mean albedos differ ~10x: bring them to plausible ground albedos (as tri.rchit).
// (layers 12-25: measured mean linear albedo -> beach sand 0.36, sand 0.34, arid soil 0.24, red
// clay 0.2, tundra 0.1, lichen 0.12, boreal floor 0.1, glacier 0.55, mud 0.11, salt 0.45,
// ploughed 0.15, lava sand 0.07, steppe soil 0.17, dry grass 0.24; very dark scans capped)
const int LAYERS = 26;
const int ASPHALT = 26; // (the roads' scan: a layer of its own, not blended by land cover)
const float GAIN[LAYERS] = float[LAYERS](0.7, 2.4, 0.68, 0.5, 1.0, 1.0, 0.75, 1.0, 1.0, 1.0, 0.8, 1.0,
                                         1.02, 1.99, 4.0, 2.27, 3.3, 0.85, 0.58, 4.09, 3.69, 2.65, 2.62, 3.97, 3.44, 4.02);

struct Hex { vec2 uv[3]; mat2 rot[3]; vec3 w; };
Hex hexTile(vec2 uv) {
    Hex h;
    vec2 st = uv * 1.6;
    vec2 sk = vec2(st.x, -0.57735027 * st.x + 1.15470054 * st.y);
    vec2 base = floor(sk);
    vec3 t = vec3(fract(sk), 0.0);
    t.z = 1.0 - t.x - t.y;
    float s = step(0.0, -t.z), s2 = 2.0 * s - 1.0;
    h.w = vec3(-t.z * s2, s - t.y * s2, s - t.x * s2);
    vec2 v[3] = vec2[3](base + vec2(s, s), base + vec2(s, 1.0 - s), base + vec2(1.0 - s, s));
    for (int i = 0; i < 3; ++i) {
        uint r = hash2(uvec2(ivec2(v[i]) + 32768));
        float ang = float(r & 0xFFFFu) * (6.2831853 / 65536.0);
        vec2 off = vec2(float((r >> 16) & 0xFFu), float(r >> 24)) / 256.0;
        float c = cos(ang), sn = sin(ang);
        mat2 Rm = mat2(c, sn, -sn, c);
        h.uv[i] = Rm * uv + off * 7.0;
        h.rot[i] = transpose(Rm);
    }
    return h;
}

float fbmN(vec2 p, int octaves) {
    float s = 0.0, a = 0.5, norm = 0.0;
    for (int i = 0; i < octaves; ++i) {
        s += a * vnoise(p);
        norm += a;
        p = mat2(0.8, 0.6, -0.6, 0.8) * p * 2.03 + vec2(7.3, 1.9); // (turned: no grid-aligned streaks)
        a *= 0.5;
    }
    return s / norm;
}

// ---- shoreline field (tiles.rs water_field) and water level grid (water_levels), stored after
// each tile's vertices: cell size, LG x LG levels, FIELD x FIELD bytes
const uint GRID = 33u;                          // tiles.rs GRID
const uint NVERT = GRID * GRID + 4u * GRID;     // surface grid + skirts
const uint FIELD = 64u;                         // tiles.rs FIELD
const uint LG = 9u;                             // tiles.rs LEVEL_GRID
const uint LEVELS_AT = NVERT * 12u + 1u;
const uint FIELD_AT = LEVELS_AT + LG * LG;

// Tile coordinates (0..1) of vertex v: the surface grid, or for a skirt vertex its edge vertex
// (tiles.rs index_buffer / build_tile order).
vec2 vertUV(uint v) {
    uint i, j;
    if (v < GRID * GRID) {
        i = v % GRID;
        j = v / GRID;
    } else {
        uint s = v - GRID * GRID, side = s / GRID, k = s % GRID;
        if (side == 0u) { i = k; j = 0u; }
        else if (side == 1u) { i = GRID - 1u; j = k; }
        else if (side == 2u) { i = GRID - 1u - k; j = GRID - 1u; }
        else { i = 0u; j = GRID - 1u - k; }
    }
    return vec2(i, j) / float(GRID - 1u);
}

float fieldAt(Verts V, ivec2 c) {
    c = clamp(c, ivec2(0), ivec2(int(FIELD) - 1));
    uint idx = uint(c.y) * FIELD + uint(c.x);
    uint word = floatBitsToUint(V.v[FIELD_AT + idx / 4u]);
    return float((word >> (8u * (idx % 4u))) & 255u);
}

// Signed distance to the shore (m; positive on land, negative in water), bilinear between the
// field's cells so the shoreline is a smooth curve at any distance.
float shoreDist(Verts V, vec2 uv) {
    float cellM = V.v[NVERT * 12u];
    vec2 f = uv * float(FIELD) - 0.5;
    ivec2 c0 = ivec2(floor(f));
    vec2 t = f - vec2(c0);
    float a = mix(fieldAt(V, c0), fieldAt(V, c0 + ivec2(1, 0)), t.x);
    float b = mix(fieldAt(V, c0 + ivec2(0, 1)), fieldAt(V, c0 + ivec2(1, 1)), t.x);
    return (mix(a, b, t.y) - 128.0) / 8.0 * cellM;
}

// The water surface height (m) at tile coordinates uv: bilinear between the level grid nodes.
// < -1e6: no water near; > 1e6: unknown (ETOPO heights), the field alone decides.
float waterLevel(Verts V, vec2 uv) {
    vec2 f = clamp(uv, 0.0, 1.0) * float(LG - 1u);
    ivec2 c0 = min(ivec2(floor(f)), ivec2(int(LG) - 2));
    vec2 t = f - vec2(c0);
    uint i = uint(c0.y) * LG + uint(c0.x);
    float a = V.v[LEVELS_AT + i], b = V.v[LEVELS_AT + i + 1u];
    float c = V.v[LEVELS_AT + i + LG], d = V.v[LEVELS_AT + i + LG + 1u];
    float lo = min(min(a, b), min(c, d)), hi = max(max(a, b), max(c, d));
    if (hi > 1e6) return 2e7;
    if (lo < -1e6) return hi < -1e6 ? -2e7 : hi; // at the edge of the water: the known level
    return mix(mix(a, b, t.x), mix(c, d, t.x), t.y);
}

// A smooth step whose edge is broken by Earth-fixed noise: land cover boundaries become organic
// lines (field and forest edges, clumps) instead of the vertex grid's diamonds.
float organic(float a, float n, float sharp) {
    float e = a + (n - 0.5) * 0.7;
    return mix(a, smoothstep(0.5 - sharp, 0.5 + sharp, e), 0.75);
}

// Ground albedo, ambient occlusion and normal from the land cover fractions.
// `bed` (0..1): the bottom of a lake, river or the sea; `shore`: the strip of land at the water.
// `biome`: RESOLVE biome number (biomes.rs); `sea`: the water at the shore / over the bed is sea.
// `coast` (0..1): within a beach's width of the water (only there is bare ground a beach).
// `floorM`: the sea floor's depth there (m, ETOPO): the bed of a pond or a sheltered bay is mud.
vec3 groundMaterial(vec4 A, vec4 B, vec2 C, vec2 uvm, float elev, vec3 up, inout vec3 n, float footprint, float bed, float shore, float coast, float floorM, int biome, bool sea, out float ao) {
    float slope = 1.0 - max(dot(n, up), 0.0);
    vec2 q = uvm;
    float desert = biome == 13 ? 1.0 : 0.0;
    float hot = (biome == 1 || biome == 2 || biome == 7) ? 1.0 : 0.0;
    float med = biome == 12 ? 1.0 : 0.0;
    float arctic = biome == 11 ? 1.0 : 0.0;
    float dryLand = max(max(desert, med), biome == 7 || biome == 8 ? 0.7 : 0.0);
    float nA = fbmN(q * 0.05, 3), nB = fbmN(q * 0.008 + 11.0, 3);
    // organic class edges: each class's fraction gets its own Earth-fixed noise (clumps of ~20 m
    // on outlines of ~150 m), sharpened, then renormalised
    float sharp = clamp(footprint * 0.02, 0.12, 0.5);
    float tree = A.x, shrub = A.y, grass = A.z, crop = A.w, built = B.x, bare = B.y, snow = B.z, wet = C.x, moss = C.y;
    {
        // every class (built and crop too): a class left out keeps its smooth ramp while the others
        // sharpen, and shows as a thin band of its own along every edge
        float total = tree + shrub + grass + crop + built + bare + wet + moss;
        float nT = fbmN(q * 0.0065 + 3.1, 3) * 0.6 + fbmN(q * 0.045 + 8.3, 2) * 0.4;
        float nS = fbmN(q * 0.008 + 17.0, 3) * 0.6 + fbmN(q * 0.06 + 2.0, 2) * 0.4;
        float nG = fbmN(q * 0.007 + 23.0, 3) * 0.6 + fbmN(q * 0.05 + 6.0, 2) * 0.4;
        float nW = fbmN(q * 0.01 + 31.0, 3) * 0.6 + fbmN(q * 0.08 + 4.0, 2) * 0.4;
        float nU = fbmN(q * 0.009 + 43.0, 3) * 0.6 + fbmN(q * 0.07 + 1.0, 2) * 0.4;
        tree = organic(tree, nT, sharp);
        shrub = organic(shrub, nS, sharp);
        grass = organic(grass, nG, sharp);
        crop = organic(crop, 1.0 - nT, sharp);
        built = organic(built, nU, sharp);
        bare = organic(bare, 1.0 - nG, sharp);
        wet = organic(wet, nW, sharp);
        moss = organic(moss, 1.0 - nW, sharp);
        float sum = tree + shrub + grass + crop + built + bare + wet + moss;
        if (sum < 0.05) {
            // every class sharpened away here: keep the smooth fractions
            tree = A.x; shrub = A.y; grass = A.z; crop = A.w; built = B.x; bare = B.y; wet = C.x; moss = C.y;
            sum = total;
        }
        float k = total / max(sum, 1e-3);
        tree *= k; shrub *= k; grass *= k; crop *= k; built *= k; bare *= k; wet *= k; moss *= k;
    }
    // cliffs: rock wherever it is too steep for soil, whatever the land cover says
    float cliff = smoothstep(0.38, 0.55, slope + (nA - 0.5) * 0.15);
    float rockBare = bare * smoothstep(0.1, 0.3, slope + (nA - 0.5) * 0.2);
    float high = smoothstep(1900.0, 2600.0, elev);
    float w[LAYERS];
    for (int l = 12; l < LAYERS; ++l) w[l] = 0.0;
    w[0] = tree * 0.75 + shrub * 0.3;                                             // forest floor
    w[1] = tree * 0.25 + grass * (0.35 - 0.2 * high) + shrub * 0.3 + wet * 0.2;   // mossy grass
    w[2] = grass * (0.65 - 0.3 * high) + crop * 0.45;                             // grass
    w[3] = crop * 0.55 + shrub * 0.4 + grass * 0.5 * high;                        // dried grass
    w[4] = (rockBare + cliff) * nB;                                               // rock (light)
    w[5] = (rockBare + cliff) * (1.0 - nB);                                       // rock (dark)
    w[6] = snow;                                                                  // snow / ice
    // flat bare ground in the mountains is scree: rock debris and grey gravel
    // (only up in the mountains: lowland bare ground is the biome's soil, sand or gravel below)
    float scree = 0.5 * max(smoothstep(1400.0, 2400.0, elev), smoothstep(0.08, 0.2, slope)) * (1.0 - 0.6 * desert);
    w[4] += (bare - rockBare) * scree * nB;
    w[5] += (bare - rockBare) * scree * (1.0 - nB);
    // built-up land (no buildings yet): the local soil with gravel, and gardens where it is not dry
    // land, in low contrast; never dark blotches
    float nb = fbmN(q * 0.035 + 61.0, 3);
    w[7] = (bare - rockBare) * (1.0 - scree);                                     // gravel
    // (as the grassland beside it, a little drier and more worn: without its buildings a town
    // must not stand out of the land as blotches of another colour)
    w[2] += built * 0.55 * (1.0 - dryLand);
    w[1] += built * 0.15 * (1.0 - dryLand);
    w[3] += built * 0.22;
    w[8] = moss * 0.25;                                                           // lichen rock
    w[9] = 0.0;
    w[10] = 0.0;
    w[11] = wet * 0.8;                                                            // wet grass
    // meadows are not one grass: patches of mossy, green and dry grass on their own noise fields
    {
        float m1 = fbmN(q * 0.03 + 41.0, 3), m2 = fbmN(q * 0.012 + 47.0, 3), m3 = fbmN(q * 0.022 + 53.0, 3);
        w[1] *= 0.35 + 1.3 * smoothstep(0.3, 0.7, m1);
        w[2] *= 0.45 + 1.1 * smoothstep(0.25, 0.75, m2);
        w[3] *= 0.25 + 1.5 * smoothstep(0.4, 0.75, m3) * (1.0 - wet);
    }
    for (int l = 0; l < 12; ++l) if (l != 4 && l != 5 && l != 6) w[l] *= 1.0 - cliff;
    // ---- the biome's own surfaces (biomes.rs numbers; layers 12-25, catalog.rs GROUND_LAYERS)
    float soil = 1.0 - cliff;
    // cropland: no invented field layout (it never matches the real fields): the crop fraction
    // from the land cover data as a soft mix of ploughed soil, green and ripening crop
    if (crop > 0.01) {
        float cw = crop * soil;
        float c1 = fbmN(q * 0.004 + 71.0, 3), c2 = fbmN(q * 0.0025 + 77.0, 3);
        int ripe = biome == 8 || biome == 13 ? 25 : 3;
        w[2] = max(w[2] - crop * 0.45, 0.0);
        w[3] = max(w[3] - crop * 0.55, 0.0);
        w[22] += cw * (0.25 + 0.3 * smoothstep(0.45, 0.7, c1));
        w[2] += cw * (0.3 + 0.3 * smoothstep(0.55, 0.3, c2));
        w[ripe] += cw * (0.2 + 0.3 * smoothstep(0.45, 0.7, c2));
    }
    // forest floor: moss under boreal and mountain conifers, red soil in the tropics
    float conifer = biome == 6 ? 0.7 : biome == 5 ? 0.4 : 0.0;
    w[18] = w[0] * conifer;
    float tropic = (biome == 1 || biome == 2 || biome == 3 || biome == 14) ? 0.5 : 0.0;
    w[15] = w[0] * tropic;
    w[0] *= 1.0 - conifer - tropic;
    // grass: dry savanna grass and red soil, steppe soil, dry Mediterranean and desert grass
    float savanna = biome == 7 ? 0.7 : biome == 12 ? 0.5 : biome == 13 ? 0.6 : 0.0;
    float steppe = biome == 8 ? 0.45 : biome == 13 ? 0.3 : 0.0;
    float g = w[1] + w[2] + w[3];
    w[25] += g * savanna;
    w[24] += g * steppe;
    w[15] += biome == 7 ? g * 0.15 : 0.0;
    float keep = max(1.0 - savanna - steppe - (biome == 7 ? 0.15 : 0.0), 0.0);
    w[1] *= keep; w[2] *= keep; w[3] *= keep;
    // flat bare ground: sand and arid soil in deserts, red soil in the tropics, arid soil in
    // Mediterranean scrub, tundra stones and lichen in the Arctic (the mountains keep their scree)
    float bareFlat = w[7];
    // bare ground along the sea is a beach: sand, whatever the biome. Only by the water: low land
    // inland (all of south Florida is under 4 m) is not a beach, and sand there made white blots.
    if (sea && bed == 0.0) {
        float bch = bareFlat * coast;
        w[12] += bch * 0.85;
        w[7] -= bch * 0.85;
        bareFlat -= bch;
    }
    // wet lowlands of the tropics and subtropics (the Everglades, mangrove swamps): bare ground is
    // grey marl and mud, not gravel
    float wetTrop = (biome == 9 && abs(degrees(asin(clamp(normalize(mat3(gl_WorldToObjectEXT) * up).z, -1.0, 1.0)))) < 31.0) || biome == 14 ? 1.0 : 0.0;
    w[20] += bareFlat * wetTrop * 0.65;
    w[13] = bareFlat * desert * (0.55 + 0.3 * nB);
    w[14] = bareFlat * (desert * (0.45 - 0.3 * nB) + hot * 0.4 + med * 0.8);
    w[15] += bareFlat * hot * 0.5;
    // (tundra: sedge, moss and dwarf shrubs over most of the ground, lichen and stones in patches)
    w[16] = bareFlat * arctic * 0.55 + moss * 0.08;
    w[17] = bareFlat * arctic * 0.35 + moss * 0.18;                               // lichen
    // (the "mossy forest floor" scan is grey-brown needles: the moss and sedge are the olive
    // mossy and uncut grass scans)
    w[1] += moss * 0.5;                                                           // moss, sedge
    w[2] += moss * 0.25;                                                          // sedge
    w[3] += moss * 0.1;
    w[7] *= max(1.0 - desert - hot * 0.9 - med * 0.8 - arctic * 0.9 - wetTrop * 0.65, 0.0);
    w[8] *= 1.0 - 0.6 * arctic;
    // towns: grey gravel (never the biome's red earth or sand), on dry land half the local soil
    // (towns without their buildings yet: yards of worn earth and grass; a little gravel)
    w[7] += built * 0.03 * (1.0 - 0.5 * dryLand) * soil;
    w[22] += built * 0.07 * soil;
    // wetlands and mangroves: mud
    float marsh = (biome == 9 || biome == 14) ? 0.6 : 0.25;
    w[20] = w[11] * marsh;
    w[11] *= 1.0 - marsh;
    // ice sheets and glaciers on gentle slopes; snow on steep ones
    float sheet = snow * smoothstep(0.25, 0.1, slope);
    w[19] = sheet * 0.8;
    w[6] -= sheet * 0.8;
    // Open ground is not one sward: stretches where the grass has dried, in soft drifts of
    // 60-200 m (no invented bare patches: bare ground is where the land cover data has it).
    {
        float dryP = smoothstep(0.35, 0.85, fbmN(q * 0.011 + 59.0, 4)) * 0.3;
        float moved = (w[1] + w[2]) * dryP;
        w[1] *= 1.0 - dryP; w[2] *= 1.0 - dryP;
        w[3] += moved;
    }
    // Under water: silt and moss on lake and river beds; under the sea sand, but dark mud and
    // seagrass in ponds and sheltered shallows. Along the shore: sand (sea) or pebbles (lakes)
    // only where the land cover has bare ground at the water, a real beach; a shore of grass,
    // marsh or mangrove runs into the water as wet ground and mud, with no white rim.
    float sandy = smoothstep(0.06, 0.3, B.y);
    float beach = shore * sandy, muddy = shore * (1.0 - sandy);
    for (int l = 0; l < LAYERS; ++l) w[l] *= (1.0 - bed) * (1.0 - beach) * (1.0 - 0.55 * muddy);
    w[20] += 0.55 * muddy;
    if (sea) {
        float mudBed = 1.0 - smoothstep(1.0, 6.0, floorM);
        w[12] += beach + bed * (1.0 - 0.8 * mudBed);                              // sand
        w[20] += bed * 0.8 * mudBed;                                              // mud
    } else {
        w[9] += bed * (0.45 + 0.4 * nA);                                          // riverbed
        w[10] += bed * (0.55 - 0.4 * nA) + beach;                                 // pebbles
    }
    // the three strongest layers
    int L[3] = int[3](0, 0, 0);
    float W[3] = float[3](-1.0, -1.0, -1.0);
    for (int l = 0; l < LAYERS; ++l) {
        float v = w[l];
        if (v > W[0]) { W[2] = W[1]; L[2] = L[1]; W[1] = W[0]; L[1] = L[0]; W[0] = v; L[0] = l; }
        else if (v > W[1]) { W[2] = W[1]; L[2] = L[1]; W[1] = v; L[1] = l; }
        else if (v > W[2]) { W[2] = v; L[2] = l; }
    }
    if (W[0] <= 0.0) { W[0] = 1.0; L[0] = 2; } // nothing left: grass, never black
    float sum = max(W[0], 0.0) + max(W[1], 0.0) + max(W[2], 0.0);
    vec3 alb = vec3(0.0);
    vec2 nxy = vec2(0.0);
    ao = 0.0;
    uint mask = floatBitsToUint(gg.params.w); // a bit mask stored as float bits (26 layers)
    Hex hx = hexTile(q / GT);
    bool near = footprint < 4.0;
    // A scan is 2 m of ground: seen from more than a few metres up its detail is smaller than a
    // pixel and the land was a flat tint. So each scan is laid a second time, 5.3 times larger
    // (11 m, turned, and bent by noise so that its repeat never lines up): its light and dark,
    // its stones and tufts, read as clumps and patches of 1-5 m from the air, and it modulates the
    // fine laying (colour over the scan's own mean). The two together hold from the ground to a
    // few hundred metres; beyond, the noise fields below.
    const float MACRO = 5.3;
    vec2 qm = mat2(0.8, 0.6, -0.6, 0.8) * (q + (vec2(fbmN(q * 0.06 + 5.0, 2), fbmN(q * 0.06 + 31.0, 2)) - 0.5) * 9.0) / (GT * MACRO);
    // Where two materials meet, the higher (brighter, less occluded) parts of each stand through
    // the other: grass tufts over soil, stones through sand; not a cross-fade (near only: from far
    // a pixel holds both).
    float sharpP = mix(4.0, 1.0, smoothstep(0.15, 2.5, footprint));
    vec3 cols[3];
    vec2 nrms[3];
    float aos[3], wts[3];
    float wsum = 0.0;
    for (int k = 0; k < 3; ++k) {
        cols[k] = vec3(0.0); nrms[k] = vec2(0.0); aos[k] = 1.0; wts[k] = 0.0;
        if (W[k] <= 0.0) continue;
        float wk = W[k] / max(sum, 1e-4);
        int l = L[k];
        vec3 c;
        vec2 nn = vec2(0.0);
        float o = 1.0, rel = 1.0;
        if ((mask & (1u << uint(l))) == 0u) {
            c = l == 6 ? vec3(0.8) : (l == 4 || l == 5) ? vec3(0.15, 0.14, 0.12) : vec3(0.08, 0.09, 0.05);
        } else {
            float texel = GT / float(textureSize(groundTex[l * 3], 0).x);
            float lod = max(log2(footprint / texel), 0.0);
            if (near) {
                c = vec3(0.0);
                for (int i = 0; i < 3; ++i) {
                    c += textureLod(groundTex[l * 3], hx.uv[i], lod).rgb * hx.w[i];
                    nn += (hx.rot[i] * (textureLod(groundTex[l * 3 + 1], hx.uv[i], lod).rg * 2.0 - 1.0)) * hx.w[i];
                }
                o = textureLod(groundTex[l * 3 + 2], hx.uv[0], lod).r;
            } else {
                c = textureLod(groundTex[l * 3], q / GT, lod).rgb;
            }
            vec3 mean = max(textureLod(groundTex[l * 3], vec2(0.5), 20.0).rgb, vec3(0.004));
            vec3 macro = textureLod(groundTex[l * 3], qm, max(log2(footprint / (texel * MACRO)), 0.0)).rgb / mean;
            // (its light and dark, its colour shifts at half; blade scans gently: a grass blade
            // five times its size is straw laid with a brush, while soil, sand and stones only
            // gain clods and patches)
            float ml = dot(macro, vec3(0.3, 0.59, 0.11));
            bool blades = l == 1 || l == 2 || l == 3 || l == 11 || l == 25;
            c *= clamp(mix(vec3(1.0), mix(vec3(ml), macro, 0.5), blades ? 0.35 : 1.2), 0.5, 1.8);
            rel = dot(c, vec3(0.3, 0.59, 0.11)) / dot(mean, vec3(0.3, 0.59, 0.11));
            c *= GAIN[l];
            if (l == 7) c = mix(c, vec3(dot(c, vec3(0.33))), 0.7) * mix(1.1, 0.62, built); // grey the reddish gravel scan (towns: worn, darker)
            // the arid soil scan is a saturated brick red: dusty red-brown earth instead
            if (l == 14) c = mix(c, vec3(dot(c, vec3(0.3, 0.59, 0.11))) * vec3(1.25, 0.95, 0.7), 0.55) * 0.85;
            // (mud: a warm grey-brown, not slate)
            if (l == 20) c = mix(c, vec3(dot(c, vec3(0.3, 0.59, 0.11))) * vec3(1.14, 0.98, 0.8), 0.6) * 0.62;
            // the ploughed-soil scan as plain bare earth: grey-brown
            if (l == 22) c = mix(c, vec3(dot(c, vec3(0.3, 0.59, 0.11))) * vec3(1.12, 0.98, 0.82), 0.4) * 0.5;
            // living ground is never one green: yellower and drier, bluer and lusher, in drifts
            // of ~15 m and ~120 m (grasses, moss, marsh, the forest floor's herbs)
            if (l <= 3 || l == 11 || l == 18 || l == 24 || l == 25) {
                float hv = fbmN(q * 0.07 + float(l) * 3.1, 3) * 0.55 + fbmN(q * 0.0085 + float(l) * 7.7, 3) * 0.45;
                c *= mix(vec3(1.08, 1.02, 0.86), vec3(0.93, 0.99, 1.02), smoothstep(0.2, 0.8, hv));
            }
        }
        cols[k] = c; nrms[k] = nn; aos[k] = o;
        wts[k] = pow(wk * (0.35 + 0.65 * clamp(rel * (0.6 + 0.4 * o), 0.0, 2.0)), sharpP);
        wsum += wts[k];
    }
    for (int k = 0; k < 3; ++k) {
        float wk = wts[k] / max(wsum, 1e-9);
        alb += cols[k] * wk;
        nxy += nrms[k] * wk;
        ao += aos[k] * wk;
    }
    // Large-scale variation, so the land is not one flat colour from high up either: lighter and
    // darker over ~20 m and ~125 m (growth, moisture), and broad drifts of ~700 m. Gently: strong
    // brightness noise reads as stains.
    float nC = fbmN(q * 0.0014 + 29.0, 3);
    alb *= mix(0.87, 1.12, nB) * mix(0.93, 1.07, nA) * mix(0.94, 1.06, nC);
    ao = mix(1.0, ao, 0.8);
    // east in Earth axes, into the camera frame the instance is placed in
    vec3 upE = normalize(mat3(gl_WorldToObjectEXT) * up);
    vec3 T = normalize(mat3(gl_ObjectToWorldEXT) * normalize(cross(vec3(0.0, 0.0, 1.0), upE) + vec3(1e-4, 0.0, 0.0)));
    vec3 Bt = cross(n, T);
    T = cross(Bt, n);
    n = normalize(T * nxy.x + Bt * nxy.y + n * sqrt(max(1.0 - dot(nxy, nxy), 0.05)));
    return alb;
}


// One ground scan's colour at uvm (metres), as groundMaterial reads it (hex-tiled near, plain far).
vec3 scanColor(int l, vec2 uvm, float footprint, vec3 fallback) {
    if ((floatBitsToUint(gg.params.w) & (1u << uint(l))) == 0u) return fallback;
    float texel = GT / float(textureSize(groundTex[l * 3], 0).x);
    float lod = max(log2(footprint / texel), 0.0);
    vec3 c;
    if (footprint < 4.0) {
        Hex hx = hexTile(uvm / GT);
        c = vec3(0.0);
        for (int i = 0; i < 3; ++i) c += textureLod(groundTex[l * 3], hx.uv[i], lod).rgb * hx.w[i];
    } else {
        c = textureLod(groundTex[l * 3], uvm / GT, lod).rgb;
    }
    return c * GAIN[l];
}

// Land colour until the land-cover data arrives (milestone 3: ESA WorldCover + RESOLVE biomes):
// a neutral stand-in from latitude and elevation only, so the relief can be judged.
vec3 landAlbedo(float lat, float elev, float slope) {
    float a = abs(lat);
    vec3 c = vec3(0.06, 0.08, 0.045);
    c = mix(c, vec3(0.24, 0.20, 0.14), smoothstep(14.0, 22.0, a) * (1.0 - smoothstep(32.0, 40.0, a)));
    c = mix(c, vec3(0.05, 0.065, 0.04), smoothstep(52.0, 58.0, a));
    c = mix(c, vec3(0.12, 0.11, 0.09), smoothstep(64.0, 70.0, a));
    c = mix(c, vec3(0.14, 0.13, 0.12), smoothstep(0.2, 0.45, slope));
    // ice sheets (Antarctica, Greenland) and snow on the high mountains
    float snowline = mix(5800.0, 600.0, smoothstep(30.0, 70.0, a));
    float ice = max(smoothstep(snowline, snowline + 500.0, elev) * (1.0 - smoothstep(0.35, 0.6, slope)),
                    smoothstep(62.0, 70.0, a) * smoothstep(800.0, 1500.0, elev));
    ice = max(ice, smoothstep(-64.0, -66.0, lat) * step(1.0, elev));
    return mix(c, vec3(0.78, 0.8, 0.84), ice);
}

void main() {
    Verts V = Verts(table.vtx[gl_InstanceCustomIndexEXT]);
    Indices I = Indices(packUint2x32(gg.addr.xy));
    uint k = gl_PrimitiveID * 3u;
    uvec3 tri = uvec3(I.i[k], I.i[k + 1u], I.i[k + 2u]);
    vec3 w = vec3(1.0 - bary.x - bary.y, bary.x, bary.y);
    vec3 n = vec3(0.0);
    float elev = 0.0;
    vec4 A = vec4(0.0), B = vec4(0.0), C = vec4(0.0);
    vec2 uvm = vec2(0.0);
    float bw = -1.0, biomeF = 0.0;
    for (int c = 0; c < 3; ++c) {
        uint b = tri[c] * 12u;
        // the biome is a category: the nearest vertex's, not a blend
        if (w[c] > bw) { bw = w[c]; biomeF = unpackUnorm4x8(floatBitsToUint(V.v[b + 9u])).w * 255.0; }
        n += vec3(V.v[b + 3u], V.v[b + 4u], V.v[b + 5u]) * w[c];
        elev += V.v[b + 6u] * w[c];
        A += unpackUnorm4x8(floatBitsToUint(V.v[b + 7u])) * w[c];
        B += unpackUnorm4x8(floatBitsToUint(V.v[b + 8u])) * w[c];
        C += unpackUnorm4x8(floatBitsToUint(V.v[b + 9u])) * w[c];
        uvm += vec2(V.v[b + 10u], V.v[b + 11u]) * w[c];
    }
    vec2 tuv = vertUV(tri.x) * w.x + vertUV(tri.y) * w.y + vertUV(tri.z) * w.z;
    float cellM = V.v[NVERT * 12u];
    float sd = shoreDist(V, tuv);
    // the mapped shoreline is 10 m pixels made smooth: break it with Earth-fixed noise
    sd += ((fbmN(uvm * 0.07, 3) - 0.5) * 4.0 + (fbmN(uvm * 0.4 + 5.0, 2) - 0.5) * 1.2) * min(1.0, 8.0 / max(cellM, 1.0));
    float level = waterLevel(V, tuv);
    bool hasLevel = abs(level) < 1e6;
    // tile-local (Earth axes) -> world (the camera frame: the instance rotates)
    n = normalize(mat3(gl_ObjectToWorldEXT) * n);
    float t = gl_HitTEXT;
    vec3 p = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * t;
    vec3 up = normalize(p - gg.planet.xyz);
    if (dot(n, up) < 0.0) n = -n;
    vec3 ng = n;
    // `up` is in the camera frame (east / up / south at the camera); latitude, the elevation grid
    // and east / north need Earth axes: the instance's inverse rotation takes it there.
    vec3 upE = normalize(mat3(gl_WorldToObjectEXT) * up);
    float lat = degrees(asin(clamp(upE.z, -1.0, 1.0)));
    bool cover = C.z > 0.5;
    // Water: the field is the distance to the drawn water (tiles.rs water_field: mapped water
    // whose ground lies below the level), so it decides alone; the surface is flat at the level.
    float above = hasLevel ? elev - level : 0.0; // height of the ground over the water level
    bool water = sd < 0.0;
    bool lake = water && (hasLevel ? level > 0.5 : elev > 0.5); // a water surface above sea level
    // Where the tile mesh is coarser than the data (far away), coastline and relief come from the
    // elevation grid per pixel instead (geometry cells ~ t / 160 m, see tiles.rs SPLIT_K).
    float footprint = t * gg.misc.x;
    float cell = t / 160.0;
    float wTex = smoothstep(0.7, 2.0, cell / TEXEL_M);
    if (wTex > 0.0) {
        float coslat = max(sqrt(1.0 - upE.z * upE.z), 0.02);
        float lod = max(log2(footprint / (TEXEL_M * coslat)), 0.0);
        vec2 uv = elevUV(upE);
        vec2 texel = vec2(1.0 / 21600.0, 1.0 / 10800.0) * exp2(lod);
        float h0 = elevAt(uv, lod);
        float hE = elevAt(uv + vec2(texel.x, 0.0), lod), hW = elevAt(uv - vec2(texel.x, 0.0), lod);
        float hN = elevAt(uv - vec2(0.0, texel.y), lod), hS = elevAt(uv + vec2(0.0, texel.y), lod);
        vec3 east = normalize(mat3(gl_ObjectToWorldEXT) * cross(vec3(0.0, 0.0, 1.0), upE));
        vec3 north = cross(up, east);
        float dx = 2.0 * texel.x * 2.0 * PI * R_GROUND * coslat;
        float dy = 2.0 * texel.y * PI * R_GROUND;
        vec3 nt = normalize(up - east * max(hE, 0.0) / dx + east * max(hW, 0.0) / dx - north * max(hN, 0.0) / dy + north * max(hS, 0.0) / dy);
        n = normalize(mix(n, nt, wTex));
        if (wTex > 0.5 && !cover) water = h0 < 0.0;
        elev = mix(elev, h0, wTex);
    }
    float ao = 1.0;
    vec3 alb;
    // Water depth: the bed shelves off the shore, then (sea) the sea floor from ETOPO. The field
    // only reaches +-16 cells (38 m on the finest tiles), so the shelf reaches full visual depth
    // within that: tiles of every size agree, no seams. main.rgen shades the water over this bed.
    float depth = 0.0;
    if (water) {
        float d = -sd;
        float floorE = max(-elevAt(elevUV(upE), 0.0), 0.0);
        // (both reach full depth within 30 m: the finest tiles' field reaches 37 m, so tiles of
        // every size agree, no seams)
        depth = lake ? min(max(d, 0.0) * 0.6, 18.0) : max(min(max(d, 0.0) * 0.4, 12.0), floorE * smoothstep(15.0, 30.0, d));
        depth = max(max(depth, -above), 0.02); // (the shallows fade to the shore that is seen)
    }
    float nW0 = fbmN(uvm * 0.3 + 71.0, 2); // breaks up the wet strip
    // the strip along the water: pebbles, and wet (darker) right at the waterline
    float band = max(2.5 * (0.4 + 1.2 * nW0), footprint * 1.5);
    // (inside the mapped water but above the level: by the height over the water)
    float shore = water ? 0.0 : smoothstep(band, band * 0.25, sd);
    int biome = int(biomeF + 0.5);
    // the water at this shore / over this bed is the sea: low land with the ocean within ~15 km
    // (ETOPO blurred: below sea level there). Low land alone is not enough: all of south Florida
    // is under 4 m, and every pond in the Everglades got a beach.
    bool seaSide = elev < 4.0 && elevAt(elevUV(upE), 3.0) < 0.5;
    if (water) {
        n = up;
        alb = groundMaterial(A, B, C.xy, uvm, elev, up, n, footprint, 1.0, 0.0, 1.0, max(-elevAt(elevUV(upE), 0.0), 0.0), biome, !lake, ao);
    } else if (cover) {
        float coast = smoothstep(28.0 + 12.0 * nW0, 8.0, sd);
        alb = groundMaterial(A, B, C.xy, uvm, elev, up, n, footprint, 0.0, shore, coast, 0.0, biome, seaSide, ao);
        // wet: right at the mapped water, and just above the water level along the contour; on a
        // sea beach the swash keeps a wider band of sand dark and wet
        float damp = max(smoothstep(1.2 + 0.6 * nW0, 0.0, sd), seaSide ? 0.75 * smoothstep(4.5 + 3.0 * nW0, 0.6, sd) : 0.0);
        alb *= mix(1.0, 0.55, damp);
        if (seaSide) {
            // the wrack line: seaweed and driftwood left at the last high tide, a broken dark band
            // a few metres up the beach (not where the trees stand)
            float at = 7.5 + 4.0 * (fbmN(uvm * 0.05 + 9.0, 2) - 0.5);
            float wr = smoothstep(1.4, 0.2, abs(sd - at)) * smoothstep(0.45, 0.7, fbmN(uvm * 0.7 + 3.0, 2)) * (1.0 - A.x);
            alb = mix(alb, alb * vec3(0.42, 0.38, 0.30), wr * 0.85);
        }
    } else {
        alb = landAlbedo(lat, elev, 1.0 - max(dot(n, up), 0.0));
    }
    // Trees thin out with distance (forest.rs: beyond NEAR a tile k levels coarser keeps 1 / 4^k of
    // them, beyond the horizon range none). Their crowns are then a pixel or less: the pixel shows
    // the average of the trees that were dropped, the canopy colour by the tree cover fraction
    // (closed stands cover ~85% of the ground), darker for the shade between the crowns.
    if (!water && cover) {
        float kept = 1.0;
        if (gg.addr.z != 0u) {
            kept = exp2(-2.0 * thinLevels(t)); // as impostor.rint thins the trees
        } else if (t > gg.misc.y) {
            // stepped: each tree tile keeps 1 / 4^k of its trees (levels coarser than the finest)
            float k = max(floor(log2(t / gg.misc.z)), 1.0);
            kept = exp2(-2.0 * k);
        }
        if (t > gg.misc.w) kept = 0.0;
        float canopy = A.x * 0.85 * (1.0 - kept);
        alb = mix(alb, vec3(0.040, 0.062, 0.024), canopy);
        ao = mix(ao, 0.55, canopy);
    }
    // Roads (roads.glsl): painted on the ground where OpenStreetMap has one. A road wins over
    // mapped water (causeways, and bridges as far as the ground reaches): the 10 m land cover
    // often has water where a narrow fill carries a road.
    if (roadsOn() && footprint < 60.0) {
        RoadHit rh;
        vec2 rq = roadXY(p);
        if (roadAt(rq, 3.5, rh)) {
            vec3 asphalt = vec3(0.06);
            if ((floatBitsToUint(gg.params.w) & (1u << uint(ASPHALT))) != 0u) {
                float texel = GT / float(textureSize(groundTex[ASPHALT * 3], 0).x);
                float lod = max(log2(footprint / texel), 0.0);
                if (footprint < 4.0) {
                    Hex hx = hexTile(uvm / GT);
                    asphalt = vec3(0.0);
                    for (int i = 0; i < 3; ++i) asphalt += textureLod(groundTex[ASPHALT * 3], hx.uv[i], lod).rgb * hx.w[i];
                } else {
                    asphalt = textureLod(groundTex[ASPHALT * 3], uvm / GT, lod).rgb;
                }
                // (the scan's texture at a worn asphalt's reflectance, ~6%: its own mean, the coarsest
                // mip, taken out)
                vec3 mean = textureLod(groundTex[ASPHALT * 3], vec2(0.5), 20.0).rgb;
                asphalt *= 0.06 / max(dot(mean, vec3(0.2126, 0.7152, 0.0722)), 0.02);
            }
            float rcov, rpaint;
            // (the centre line is yellow in the Americas, white elsewhere)
            float lonE = degrees(atan(upE.y, upE.x));
            vec3 rc = roadSurface(rh, rq, alb, asphalt, footprint, lonE > -170.0 && lonE < -30.0, rcov, rpaint);
            if (water && rcov > 0.5) {
                water = false;
                lake = false;
                depth = 0.0;
                rcov = 1.0;
            }
            if (!water) {
                // the verge: a strip of worn, packed ground along the seal (1-2 m), so a road has
                // an edge against whatever it crosses
                // The roadside, from the ground scans: a shoulder of crushed gravel against the
                // seal, then packed dirt worn bare by tyres and mowing, of the place's own soil
                // (sand on a coast, red clay in the tropics, pale arid soil where it is dry, dark
                // mud where it is wet), in stretches of one or the other; grass creeps back in from
                // outside in tongues, and the width changes along the road.
                float out_ = -rh.edge; // metres outside the seal
                if (out_ > 0.0 && footprint < 8.0) {
                    float wob = fbmN(rq * 0.11 + 3.0, 2);
                    float shoulderW = 0.5 + 0.9 * wob, dirtW = shoulderW + 0.5 + 1.6 * fbmN(rq * 0.07 + 9.0, 2);
                    vec3 gravel = scanColor(7, uvm, footprint, vec3(0.16));
                    gravel = mix(gravel, vec3(dot(gravel, vec3(0.33))), 0.6) * 0.8;
                    int soilA = biome == 13 || biome == 12 ? 14 : (biome == 1 || biome == 2 || biome == 7) ? 15 : (biome == 9 || biome == 14) ? 20 : 14;
                    int soilB = seaSide ? 12 : (biome == 13 ? 13 : 20);
                    vec3 dirtA = scanColor(soilA, uvm, footprint, vec3(0.13, 0.10, 0.07));
                    if (soilA == 14) dirtA = mix(dirtA, vec3(dot(dirtA, vec3(0.3, 0.59, 0.11))) * vec3(1.2, 0.97, 0.75), 0.55) * 0.8;
                    vec3 dirtB = scanColor(soilB, uvm, footprint, vec3(0.20, 0.17, 0.12));
                    // (a roadside's sand is trodden and dirty, not a beach's; its soil greyer than a field's)
                    if (soilB == 12 || soilB == 13) dirtB = mix(dirtB, vec3(dot(dirtB, vec3(0.33))) * vec3(1.04, 1.0, 0.92), 0.4) * 0.5;
                    dirtA = mix(dirtA, vec3(dot(dirtA, vec3(0.3, 0.59, 0.11))), 0.35) * 0.85;
                    vec3 dirt = mix(dirtA, dirtB, smoothstep(0.4, 0.6, fbmN(rq * 0.025 + 31.0, 3)));
                    dirt *= mix(0.75, 1.1, fbmN(rq * 0.9, 2));
                    // (tyre marks pulled off the edge, darker and damp in places)
                    dirt *= 1.0 - 0.3 * smoothstep(0.55, 0.8, fbmN(vec2(rh.along * 0.15, out_ * 1.5) + 5.0, 2));
                    float ragged = (fbmN(rq * 0.6 + 2.0, 2) - 0.5) * 0.9;
                    float inShoulder = 1.0 - smoothstep(shoulderW - 0.25, shoulderW + 0.15, out_ + ragged * 0.4);
                    float inDirt = 1.0 - smoothstep(dirtW - 0.6, dirtW + 0.3, out_ + ragged);
                    vec3 side = mix(dirt, gravel, inShoulder);
                    alb = mix(alb, side, max(inDirt, inShoulder) * (1.0 - rcov) * smoothstep(8.0, 3.0, footprint));
                }
                alb = mix(alb, rc, rcov);
                n = normalize(mix(n, up, 0.85 * rcov));
                ao = mix(ao, 1.0, rcov);
            }
        }
    }
    uint dbg = uint(gg.params.z);
    if (dbg == 2u) alb = vec3(fract(elev / 500.0), clamp(elev / 5000.0, 0.0, 1.0), float(water));
    if (dbg == 3u) alb = A.x * vec3(0.0, 0.4, 0.0) + A.z * vec3(0.5, 1.0, 0.2) + A.w * vec3(1.0, 0.9, 0.3) + B.x * vec3(1.0, 0.0, 0.0)
                         + B.y * vec3(0.6) + B.z * vec3(1.0) + B.w * vec3(0.0, 0.3, 1.0) + A.y * vec3(0.6, 0.5, 0.0) + C.x * vec3(0.0, 0.8, 0.8);
    if (dbg == 3u && water) alb = sd < 0.0 ? (hasLevel ? vec3(1.0, 0.0, 1.0) : vec3(0.5, 0.0, 1.0)) : vec3(0.0, 0.0, 0.0);
    pl.albedo = alb;
    pl.t = t;
    pl.n = n;
    pl.ng = ng;
    pl.flags = MAT_TERRAIN | (water ? FLAG_WATER : 0u) | (water && !lake ? FLAG_SEA : 0u);
    pl.ao = ao;
    pl.sunVis = 1.0;
    pl.shadowMask = 0xFFu;
    pl.trans = vec3(0.0);
    pl.wdepth = depth;
    if (water) {
        // Earth-fixed water coordinates for main.rgen's ripples and foam: the texture coordinates
        // (m), the shore distance, the direction of the first coordinate, the surface height
        vec3 P[3];
        vec2 U[3];
        for (int c = 0; c < 3; ++c) {
            uint b = tri[c] * 12u;
            P[c] = vec3(V.v[b], V.v[b + 1u], V.v[b + 2u]);
            U[c] = vec2(V.v[b + 10u], V.v[b + 11u]);
        }
        vec3 e1 = P[1] - P[0], e2 = P[2] - P[0];
        vec2 d1 = U[1] - U[0], d2 = U[2] - U[0];
        vec3 Tu = mat3(gl_ObjectToWorldEXT) * (e1 * d2.y - e2 * d1.y);
        Tu -= up * dot(Tu, up);
        float tl = length(Tu);
        pl.wtan = tl > 1e-12 ? Tu / tl : normalize(cross(up, vec3(0.0, 0.0, 1.0)) + vec3(1e-4, 0.0, 0.0));
        float det = d1.x * d2.y - d2.x * d1.y;
        if (det < 0.0) pl.wtan = -pl.wtan;
        // the second coordinate runs along +-cross(up, wtan): length 2 marks the minus
        vec3 Tv = mat3(gl_ObjectToWorldEXT) * (e2 * d1.x - e1 * d2.x) * sign(det);
        if (dot(cross(up, pl.wtan), Tv) < 0.0) pl.wtan *= 2.0;
        pl.trans = vec3(uvm, sd);
        pl.wlevel = max(-above, 0.0);
        pl.wfloor = max(-elevAt(elevUV(upE), 0.0), 0.0);
    }
}
