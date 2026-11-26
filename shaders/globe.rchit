#version 460
#extension GL_GOOGLE_include_directive : require
#include "globe.glsl"

// Terrain tiles: geometry, land cover and the ground material (tiles.rs vertex: position (3),
// normal (3), elevation, land cover (3 x unorm4x8), texture coordinates in metres (2)).
layout(set = 0, binding = 30) uniform GG { GlobeGlobals gg; };
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
        p = p * 2.03 + vec2(7.3, 1.9);
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
vec3 groundMaterial(vec4 A, vec4 B, vec2 C, vec2 uvm, float elev, vec3 up, inout vec3 n, float footprint, float bed, float shore, int biome, bool sea, out float ao) {
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
    w[2] += built * 0.3 * (1.0 - dryLand) * smoothstep(0.55, 0.35, nb);
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
    // bare ground at sea level is a beach: sand, whatever the biome
    if (sea && bed == 0.0) { w[12] += bareFlat * 0.85; w[7] = bareFlat * 0.15; bareFlat = 0.0; }
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
    w[7] *= max(1.0 - desert - hot * 0.9 - med * 0.8 - arctic * 0.9, 0.0);
    w[8] *= 1.0 - 0.6 * arctic;
    // towns: grey gravel (never the biome's red earth or sand), on dry land half the local soil
    w[7] += built * 0.55 * (1.0 - 0.5 * dryLand) * soil;
    // wetlands and mangroves: mud
    float marsh = (biome == 9 || biome == 14) ? 0.6 : 0.25;
    w[20] = w[11] * marsh;
    w[11] *= 1.0 - marsh;
    // ice sheets and glaciers on gentle slopes; snow on steep ones
    float sheet = snow * smoothstep(0.25, 0.1, slope);
    w[19] = sheet * 0.8;
    w[6] -= sheet * 0.8;
    // under water: silt and moss on lake and river beds, sand under the sea, pebbles in the
    // shallows; along the shore pebbles (lakes) or beach sand (sea)
    for (int l = 0; l < LAYERS; ++l) w[l] *= (1.0 - bed) * (1.0 - shore);
    if (sea) {
        w[12] += bed + shore;                                                     // sand
    } else {
        w[9] += bed * (0.45 + 0.4 * nA);                                          // riverbed
        w[10] += bed * (0.55 - 0.4 * nA) + shore;                                 // pebbles
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
    for (int k = 0; k < 3; ++k) {
        if (W[k] <= 0.0) continue;
        float wk = W[k] / max(sum, 1e-4);
        int l = L[k];
        vec3 c;
        vec2 nn = vec2(0.0);
        float o = 1.0;
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
            c *= GAIN[l];
            if (l == 7) c = mix(c, vec3(dot(c, vec3(0.33))), 0.7) * 1.1; // grey the reddish gravel scan
            // the arid soil scan is a saturated brick red: dusty red-brown earth instead
            if (l == 14) c = mix(c, vec3(dot(c, vec3(0.3, 0.59, 0.11))) * vec3(1.25, 0.95, 0.7), 0.55) * 0.85;
        }
        alb += c * wk;
        nxy += nn * wk;
        ao += o * wk;
    }
    // large-scale variation so meadows and forests are not one flat colour from a distance
    alb *= mix(0.8, 1.15, nB) * mix(0.9, 1.1, nA);
    ao = mix(1.0, ao, 0.8);
    // east in Earth axes, into the camera frame the instance is placed in
    vec3 upE = normalize(mat3(gl_WorldToObjectEXT) * up);
    vec3 T = normalize(mat3(gl_ObjectToWorldEXT) * normalize(cross(vec3(0.0, 0.0, 1.0), upE) + vec3(1e-4, 0.0, 0.0)));
    vec3 Bt = cross(n, T);
    T = cross(Bt, n);
    n = normalize(T * nxy.x + Bt * nxy.y + n * sqrt(max(1.0 - dot(nxy, nxy), 0.05)));
    return alb;
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

// ---- the city's ground (globe/city.rs): the plan's streets, zones and lots on the city's
// tangent plane, worked out with the same integer hashes as the CPU plan (keep in step with
// city.rs PLAN). Arterials (every 6th street north-south, every 5th east-west) are 22 m with
// double yellow centre lines, dashed lanes and zebra crossings; the other streets are 14 m, two
// lanes and no paint. Between them the lots by zone: houses in two rows facing their streets
// (lawn, driveway, a pool in some back yards, hedges), shops on asphalt parking with stall
// lines, office and apartment blocks on parking or plazas, towers on plazas with planters.
// Returns how much of the ground the city covers here (0 outside it, on water, wetland or at
// the shore); `alb` takes its surface.
uint cHash(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
}
uint cBh(int i, int j, uint s) { return cHash(uint(i) * 0x9E3779B1u ^ cHash(uint(j) + s * 0x85EBCA6Bu)); }
float cBu(int i, int j, uint s) { return float(cBh(i, j, s) >> 8) / 16777216.0; }
int cMod(int i, int m) { return i - m * int(floor(float(i) / float(m))); }
bool cArtX(int i) { return cMod(i, 6) == 0; }
bool cArtY(int j) { return cMod(j, 5) == 0; }
float cHalf(bool a) { return a ? 11.0 : 7.0; }
float cLineX(int i) { return float(i) * gg.cityP.x + (cBu(i, 0, 40u) - 0.5) * 24.0; }
float cLineY(int j) { return float(j) * gg.cityP.y + (cBu(0, j, 41u) - 0.5) * 20.0; }
float cSide(bool a) { return a ? 4.0 : 3.0; }
float cDensity(vec2 c) {
    float d = 0.0;
    for (int k = 0; k < int(gg.cityP.w); ++k) {
        vec4 core = gg.cityCores[k];
        vec2 q = c - core.xy;
        d = max(d, core.z * exp(-dot(q, q) / (core.w * core.w)));
    }
    return d;
}
// a filtered box: 1 inside [lo, hi] (per axis), soft over the pixel footprint
float cBox(vec2 p, vec2 lo, vec2 hi, float fw) {
    vec2 a = smoothstep(lo - fw, lo + fw, p) * (1.0 - smoothstep(hi - fw, hi + fw, p));
    return a.x * a.y;
}

// ---- the plan bends (globe/city.rs warp, to_plan): the grid is straight in the plan, curved in
// the city's plane
vec2 cWarp(vec2 q) {
    return vec2(50.0 * sin(q.y / 330.0 + 1.3) + 18.0 * sin(q.y / 140.0 + q.x / 500.0 + 0.4),
                40.0 * sin(q.x / 380.0 + 2.1) + 15.0 * sin(q.x / 150.0 + q.y / 520.0 + 2.9));
}
vec2 cToPlan(vec2 c) {
    vec2 q = c;
    for (int k = 0; k < 6; ++k) q = c - cWarp(q);
    return q;
}
// the roads of their own (globe/city.rs WAYS; 0 avenue, 1 highway, 2 ramp), in the city's plane
const int C_WAYS = 7;
const int C_WAY_FIRST[8] = int[8](0, 10, 15, 20, 25, 30, 35, 41);
const int C_WAY_KIND[7] = int[7](1, 1, 2, 2, 0, 0, 0);
const float C_WAY_HALF[7] = float[7](15.0, 13.0, 5.5, 5.5, 14.0, 14.0, 14.0);
const vec2 C_WAY_P[41] = vec2[41](vec2(-3350.0, -7000.0), vec2(-3250.0, -5000.0), vec2(-3150.0, -3000.0), vec2(-3050.0, -1500.0), vec2(-3000.0, -200.0), vec2(-3080.0, 1200.0), vec2(-3250.0, 2800.0), vec2(-3450.0, 4500.0), vec2(-3600.0, 6500.0), vec2(-3700.0, 9000.0), vec2(-3010.0, 700.0), vec2(-2700.0, 790.0), vec2(-2300.0, 860.0), vec2(-1900.0, 890.0), vec2(-1450.0, 880.0), vec2(-3018.0, -1300.0), vec2(-3013.0, -1450.0), vec2(-2995.0, -1600.0), vec2(-2965.0, -1750.0), vec2(-2930.0, -1900.0), vec2(-3277.0, 2800.0), vec2(-3300.0, 2950.0), vec2(-3345.0, 3100.0), vec2(-3400.0, 3250.0), vec2(-3460.0, 3400.0), vec2(-7500.0, -6500.0), vec2(-5600.0, -4400.0), vec2(-4200.0, -2900.0), vec2(-3300.0, -1900.0), vec2(-2500.0, -1100.0), vec2(-1800.0, 2600.0), vec2(-3000.0, 3600.0), vec2(-4400.0, 4800.0), vec2(-6000.0, 6300.0), vec2(-7500.0, 7800.0), vec2(4300.0, -1500.0), vec2(4150.0, 0.0), vec2(4250.0, 1500.0), vec2(4500.0, 3000.0), vec2(4800.0, 4500.0), vec2(5000.0, 6000.0));
// The nearest road to c: its kind, half width, signed distance across, distance along.
// Returns false if none is within its half width + pad.
bool cWay(vec2 c, float pad, out int kind, out float wHalf, out float d, out float s) {
    float best = 1e9;
    kind = -1;
    for (int w = 0; w < C_WAYS; ++w) {
        float hw = C_WAY_HALF[w] + pad, acc = 0.0;
        for (int p = C_WAY_FIRST[w]; p < C_WAY_FIRST[w + 1] - 1; ++p) {
            vec2 a = C_WAY_P[p], b = C_WAY_P[p + 1];
            vec2 ab = b - a;
            float l = length(ab);
            if (any(lessThan(c, min(a, b) - hw)) || any(greaterThan(c, max(a, b) + hw))) { acc += l; continue; }
            float t = clamp(dot(c - a, ab) / (l * l), 0.0, 1.0);
            vec2 e = c - a - ab * t;
            float dist = length(e);
            if (dist < hw && dist < best) {
                best = dist;
                kind = C_WAY_KIND[w];
                wHalf = C_WAY_HALF[w];
                d = (ab.x * e.y - ab.y * e.x) / l;
                s = acc + t * l;
            }
            acc += l;
        }
    }
    return kind >= 0;
}

// Is p inside the city's plan?
bool cityNear(vec3 p) {
    if (gg.cityP.z <= 0.0) return false;
    vec3 v = p - gg.cityO.xyz;
    vec2 c = vec2(dot(v, gg.cityE.xyz), dot(v, gg.cityN.xyz));
    return dot(c, c) < gg.cityP.z * gg.cityP.z;
}

float cityGround(vec3 p, float footprint, float built, float waterF, float wetF, out vec3 alb) {
    alb = vec3(0.0);
    if (gg.cityP.z <= 0.0) return 0.0;
    vec3 v = p - gg.cityO.xyz;
    vec2 cw = vec2(dot(v, gg.cityE.xyz), dot(v, gg.cityN.xyz)); // the city's plane
    float r = length(cw);
    if (r > gg.cityP.z) return 0.0;
    vec2 c = cToPlan(cw); // the plan (the grid's own coordinates)
    float BX = gg.cityP.x, BY = gg.cityP.y;
    int i = int(floor(c.x / BX)), j = int(floor(c.y / BY));
    if (c.x < cLineX(i)) i -= 1; else if (c.x >= cLineX(i + 1)) i += 1;
    if (c.y < cLineY(j)) j -= 1; else if (c.y >= cLineY(j + 1)) j += 1;
    float lx0 = cLineX(i), lx1 = cLineX(i + 1), ly0 = cLineY(j), ly1 = cLineY(j + 1);
    vec2 f = c - vec2(lx0, ly0);
    float bw = lx1 - lx0, bh = ly1 - ly0;
    // the nearest street lines and how far from them
    int li = f.x < 0.5 * bw ? i : i + 1, lj = f.y < 0.5 * bh ? j : j + 1;
    vec2 e = vec2(min(f.x, bw - f.x), min(f.y, bh - f.y));
    bool ax = cArtX(li), ay = cArtY(lj);
    vec2 hw = vec2(cHalf(ax), cHalf(ay));
    vec2 road = hw - vec2(cSide(ax), cSide(ay));
    float fw = max(footprint, 0.03);
    float inRX = 1.0 - smoothstep(road.x - fw, road.x + fw, e.x);
    float inRY = 1.0 - smoothstep(road.y - fw, road.y + fw, e.y);
    float inSX = 1.0 - smoothstep(hw.x - fw, hw.x + fw, e.x);
    float inSY = 1.0 - smoothstep(hw.y - fw, hw.y + fw, e.y);
    float roadW = max(inRX, inRY), streetW = max(inSX, inSY);
    float n1 = fbmN(c * 0.12, 3), n2 = fbmN(c * 0.9 + 7.0, 2), n3 = fbmN(c * 0.02 + 3.0, 2);
    float shown = 1.0 - smoothstep(0.25, 0.8, fw); // can a pixel show paint and joints?

    // ---- streets
    vec3 asphalt = vec3(0.075, 0.075, 0.078) * mix(0.8, 1.2, n1) * mix(0.92, 1.08, n2) * mix(0.85, 1.12, n3);
    {
        float only = 1.0 - min(inRX, inRY); // not in the crossing itself
        float yel = 0.0, wht = 0.0;
        if (ax) {
            yel = max(yel, (1.0 - smoothstep(0.06, 0.06 + fw, abs(e.x - 0.2))) * inRX * only);
            wht = max(wht, (1.0 - smoothstep(0.06, 0.06 + fw, abs(e.x - road.x * 0.5))) * step(fract(c.y / 9.0), 0.35) * inRX * only);
        }
        if (ay) {
            yel = max(yel, (1.0 - smoothstep(0.06, 0.06 + fw, abs(e.y - 0.2))) * inRY * only);
            wht = max(wht, (1.0 - smoothstep(0.06, 0.06 + fw, abs(e.y - road.y * 0.5))) * step(fract(c.x / 9.0), 0.35) * inRY * only);
        }
        // zebra crossings where an arterial is involved; stop bars on the side streets
        if (ax || ay) {
            wht = max(wht, inRX * step(road.y + 0.6, e.y) * step(e.y, road.y + 3.4) * step(fract(e.x / 1.0), 0.5));
            wht = max(wht, inRY * step(road.x + 0.6, e.x) * step(e.x, road.x + 3.4) * step(fract(e.y / 1.0), 0.5));
        }
        float amt = max(yel, wht) * shown * mix(0.7, 1.0, n2);
        asphalt = mix(asphalt, yel > wht ? vec3(0.55, 0.42, 0.10) : vec3(0.6, 0.6, 0.58), amt);
        asphalt += vec3(0.012) * (1.0 - shown) * (ax || ay ? 1.0 : 0.0);
    }
    vec2 sj = abs(fract(c / 1.5) - 0.5);
    float joint = (1.0 - smoothstep(0.46, 0.5, max(sj.x, sj.y))) * shown;
    vec3 walk = vec3(0.21, 0.205, 0.195) * mix(0.8, 1.1, n1) * mix(1.0, 0.86, 1.0 - joint);
    float curb = max((1.0 - smoothstep(0.0, 0.25 + fw, abs(e.x - road.x))) * inSX, (1.0 - smoothstep(0.0, 0.25 + fw, abs(e.y - road.y))) * inSY);
    walk = mix(walk, vec3(0.30, 0.295, 0.28), curb * shown);
    // a strip of grass between the kerb and the walk on the residential streets
    float verge = (!ax ? (1.0 - smoothstep(road.x + 1.4, road.x + 1.4 + fw, e.x)) * inSX : 0.0) + (!ay ? (1.0 - smoothstep(road.y + 1.4, road.y + 1.4 + fw, e.y)) * inSY : 0.0);
    // lawn: St Augustine grass, patchy, browner where it dries out, darker in tufts
    float nl = fbmN(c * 0.3 + 5.0, 3), nd = fbmN(c * 0.07 + 13.0, 3);
    vec3 lawnC = mix(vec3(0.062, 0.085, 0.030), vec3(0.095, 0.098, 0.050), 0.6 * smoothstep(0.62, 0.9, nd)) * mix(0.72, 1.18, nl) * mix(0.88, 1.08, fbmN(c * 2.1 + 9.0, 2));
    walk = mix(walk, lawnC, clamp(verge, 0.0, 1.0) * (1.0 - curb));

    // ---- the block's lots
    float hx0 = cHalf(cArtX(i)), hx1 = cHalf(cArtX(i + 1)), hy0 = cHalf(cArtY(j)), hy1 = cHalf(cArtY(j + 1));
    vec2 b = f - vec2(hx0, hy0);
    float W = bw - hx0 - hx1, H = bh - hy0 - hy1;
    vec2 bc = vec2(lx0 + hx0 + 0.5 * W, ly0 + hy0 + 0.5 * H);
    float dz = cDensity(bc) + (cBu(i, j, 1u) - 0.5) * 0.12;
    int zone = dz > 0.55 ? 3 : dz > 0.2 ? 2 : dz > 0.05 ? 1 : 0;
    if (zone == 0 && (cArtX(i) || cArtX(i + 1) || cArtY(j) || cArtY(j + 1)) && cBu(i, j, 6u) < 0.6) zone = 1;
    vec3 pave = vec3(0.15, 0.145, 0.135) * mix(0.82, 1.1, n1);
    vec2 tj = abs(fract(c / vec2(1.2, 0.6)) - 0.5);
    vec3 plaza = vec3(0.175, 0.165, 0.15) * mix(0.82, 1.1, n1) * mix(0.9, 1.06, n3) * mix(1.0, 0.9, (1.0 - smoothstep(0.44, 0.5, max(tj.x, tj.y))) * shown);
    vec3 lot;
    if (zone == 0) {
        float n = max(floor(W / 21.0), 1.0);
        float lw = W / n;
        bool south = b.y < 0.5 * H;
        float dv = south ? b.y : H - b.y; // back from the street
        float l = clamp(floor(b.x / lw), 0.0, n - 1.0);
        float u = b.x - l * lw;
        uint lotN = uint((south ? 0.0 : n) + l);
        bool right = cBu(i, j, 30u + lotN) < 0.5;
        vec2 q = vec2(u, dv);
        lot = lawnC * mix(0.85, 1.15, cBu(i, j, 700u + lotN));
        // driveway from the street to the carport, and the front path
        float drive = right ? cBox(q, vec2(lw - 4.6, -1.0), vec2(lw - 1.4, 20.0), fw) : cBox(q, vec2(1.4, -1.0), vec2(4.6, 20.0), fw);
        lot = mix(lot, vec3(0.20, 0.195, 0.185) * mix(0.8, 1.05, n2), drive);
        lot = mix(lot, vec3(0.22, 0.21, 0.19), cBox(q, vec2(0.5 * lw - 0.6, -1.0), vec2(0.5 * lw + 0.6, 6.5), fw));
        // the back yard: a patio and, in some, a pool with its coping
        float back = 6.0 + 17.5;
        lot = mix(lot, vec3(0.23, 0.215, 0.195), cBox(q, vec2(2.0, back), vec2(lw - 2.0, back + 2.5), fw));
        if (cBu(i, j, 600u + lotN) < 0.5) {
            float px = 2.5 + (lw - 12.0) * cBu(i, j, 620u + lotN);
            vec2 lo = vec2(px, back + 3.5), hi = vec2(px + 4.5 + 3.0 * cBu(i, j, 640u + lotN), back + 3.5 + 7.5);
            lot = mix(lot, vec3(0.42, 0.40, 0.37), cBox(q, lo - 0.6, hi + 0.6, fw));
            vec2 pq = (q - lo) / max(hi - lo, vec2(1.0));
            vec3 water = mix(vec3(0.05, 0.30, 0.34), vec3(0.10, 0.42, 0.46), smoothstep(0.0, 1.0, pq.y)) * mix(0.9, 1.1, n2);
            lot = mix(lot, water, cBox(q, lo, hi, fw));
        }
        // hedges along the lot lines
        float hedge = max(1.0 - smoothstep(0.5, 0.5 + fw, min(u, lw - u)), 1.0 - smoothstep(0.6, 0.6 + fw, 0.5 * H - dv));
        lot = mix(lot, vec3(0.035, 0.06, 0.02) * mix(0.8, 1.2, n2), hedge * step(6.0, dv));
    } else if (zone == 1 || (zone == 2 && cBu(i, j, 3u) < 0.3)) {
        // asphalt parking: bays of 2.6 m stalls either side of 7 m aisles, rows every 18 m,
        // a planted strip round the edge
        float rowV = mod(b.y - 2.0, 18.0);
        float bay = step(rowV, 5.5) + step(12.5, rowV);
        float lineS = (1.0 - smoothstep(0.06, 0.06 + fw, min(mod(b.x, 2.6), 2.6 - mod(b.x, 2.6)))) * bay;
        lot = vec3(0.085, 0.085, 0.088) * mix(0.8, 1.2, n1) * mix(0.9, 1.1, n3);
        lot = mix(lot, vec3(0.55, 0.55, 0.52), lineS * shown * 0.85);
        float edge = 1.0 - cBox(b, vec2(2.2), vec2(W, H) - 2.2, fw);
        lot = mix(lot, lawnC, edge);
    } else {
        // plazas: stone paving, planters, lawn panels
        lot = plaza;
        vec2 pm = mod(b, vec2(14.0, 11.0));
        float planter = cBox(pm, vec2(1.0, 1.0), vec2(5.0, 2.6), fw) * step(0.4, cBu(i, j, 5u));
        lot = mix(lot, lawnC * 0.8, planter);
        float edge = 1.0 - cBox(b, vec2(1.6), vec2(W, H) - 1.6, fw);
        lot = mix(lot, lawnC, edge * 0.8);
        if (zone == 2) lot = mix(lot, pave, 0.4);
    }
    vec3 planned = mix(lot, mix(walk, asphalt, roadW), streetW);
    // where the land cover is not built up (parks, grounds, open land in the city): kept lawn
    // under scattered shade, with paths, never the bare native soil showing through
    float shade = smoothstep(0.55, 0.75, fbmN(c * 0.06 + 31.0, 3) * 0.7 + fbmN(c * 0.4 + 2.0, 2) * 0.3);
    vec3 park = mix(lawnC * 1.05, lawnC * vec3(0.6, 0.72, 0.55), shade);
    vec2 pp = abs(fract(c / vec2(37.0, 29.0)) - 0.5);
    park = mix(park, vec3(0.24, 0.225, 0.2), (1.0 - smoothstep(0.012, 0.012 + fw / 37.0, min(pp.x, pp.y))) * shown * 0.6);
    // wetland in the city: mangrove and dense shrub, a deep, uneven green
    vec3 mangrove = vec3(0.028, 0.05, 0.02) * mix(0.7, 1.3, fbmN(c * 0.25 + 77.0, 3));
    park = mix(park, mangrove, smoothstep(0.15, 0.45, wetF));
    alb = mix(park, planned, smoothstep(0.08, 0.3, built) * (1.0 - smoothstep(0.2, 0.5, wetF)));
    // the roads of their own: an avenue's asphalt, lane lines and planted median; the ground
    // under an elevated road is a mown verge
    int wk;
    float wh, wd, ws;
    if (cWay(cw, 4.0, wk, wh, wd, ws)) {
        float ad = abs(wd);
        vec3 road;
        float cover;
        if (wk == 0) {
            vec3 asph = vec3(0.075, 0.075, 0.078) * mix(0.82, 1.18, n1) * mix(0.9, 1.1, n3);
            float lane = (1.0 - smoothstep(0.07, 0.07 + fw, abs(ad - 4.9))) * step(fract(ws / 9.0), 0.35)
                       + (1.0 - smoothstep(0.07, 0.07 + fw, abs(ad - (wh - 3.9))));
            road = mix(asph, vec3(0.6, 0.6, 0.58), clamp(lane, 0.0, 1.0) * shown);
            road = mix(road, lawnC, 1.0 - smoothstep(1.5, 1.5 + fw, ad));                       // median
            road = mix(road, vec3(0.30, 0.29, 0.27), smoothstep(1.4, 1.4 + fw, ad) * (1.0 - smoothstep(1.7, 1.7 + fw, ad)) * shown); // its kerb
            road = mix(road, walk, smoothstep(wh - 3.5, wh - 3.5 + fw, ad));                     // sidewalks
            cover = 1.0 - smoothstep(wh, wh + fw, ad);
        } else {
            road = lawnC * 0.9;
            cover = 1.0 - smoothstep(wh + 3.0, wh + 3.0 + fw, ad);
        }
        alb = mix(alb, road, cover);
    }
    // the city keeps off water and wetland; it fades in with the built-up land and out at its rim
    float urban = 1.0 - smoothstep(0.85, 1.0, r / gg.cityP.z);
    // (land by the shore field is land, right up to the water: no fade by the water fraction,
    // which let the native mud show along every shore)
    return urban;
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
    // in a city the land cover's ponds above sea level (most are mapping noise in built-up
    // land: dark puddles among the blocks) are land; the bay, the river and the canals stay
    if (lake && cityNear(p)) {
        water = false;
        lake = false;
        sd = 60.0;
    }
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
    // in a city the water is held by seawalls (all but sand beaches and mangroves): deep right
    // up to the wall
    bool seawall = water && cityNear(p) && (B.y < 0.6 || B.x > 0.08);
    if (seawall) depth = max(depth, 3.5);
    float nW0 = fbmN(uvm * 0.3 + 71.0, 2); // breaks up the wet strip
    // the strip along the water: pebbles, and wet (darker) right at the waterline
    float band = max(2.5 * (0.4 + 1.2 * nW0), footprint * 1.5);
    // (inside the mapped water but above the level: by the height over the water)
    float shore = water ? 0.0 : smoothstep(band, band * 0.25, sd);
    int biome = int(biomeF + 0.5);
    bool seaSide = elev < 4.0; // the water at this shore / over this bed is the sea
    if (water) {
        n = up;
        alb = groundMaterial(A, B, C.xy, uvm, elev, up, n, footprint, 1.0, 0.0, biome, !lake, ao);
    } else if (cover || cityNear(p)) {
        // (in a city also where the land cover has no data: a former pond, the plan covers it)
        vec3 cityAlb;
        float urban = cityGround(p, footprint, B.x, B.w, C.x, cityAlb);
        // A built-up shore in a city is a seawall, not a beach: the city's ground runs to a
        // concrete cap with a dark, wet face at the edge (main.rgen: no lapping foam there, the
        // water is deep at once). Natural shores (beach sand, mangroves) keep their strip.
        float wall = cityNear(p) ? max(1.0 - smoothstep(0.55, 0.8, B.y), smoothstep(0.06, 0.18, B.x)) : 0.0;
        alb = groundMaterial(A, B, C.xy, uvm, elev, up, n, footprint, 0.0, shore * (1.0 - wall), biome, seaSide, ao);
        urban *= 1.0 - shore * (1.0 - wall);
        if (urban > 0.0) {
            alb = mix(alb, cityAlb, urban);
            n = normalize(mix(n, up, urban));
            ao = mix(ao, 1.0, urban);
        }
        if (wall > 0.0) {
            // (a thin line: it fades out once it is under a pixel, never a bright outline)
            vec3 cap = vec3(0.23, 0.225, 0.21) * mix(0.85, 1.08, nW0);
            float thin = 1.0 - smoothstep(0.4, 1.2, footprint);
            alb = mix(alb, cap, smoothstep(0.9 + footprint, 0.7, sd) * wall * thin);
            alb = mix(alb, vec3(0.05, 0.055, 0.05), smoothstep(0.3 + footprint, 0.05, sd) * wall * thin);
        }
        // wet: right at the mapped water, and just above the water level along the contour
        float damp = smoothstep(1.2 + 0.6 * nW0, 0.0, sd) * (1.0 - wall);
        alb *= mix(1.0, 0.55, damp);
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
    uint dbg = uint(gg.params.z);
    if (dbg == 2u) alb = vec3(fract(elev / 500.0), clamp(elev / 5000.0, 0.0, 1.0), float(water));
    if (dbg == 3u) alb = A.x * vec3(0.0, 0.4, 0.0) + A.z * vec3(0.5, 1.0, 0.2) + A.w * vec3(1.0, 0.9, 0.3) + B.x * vec3(1.0, 0.0, 0.0)
                         + B.y * vec3(0.6) + B.z * vec3(1.0) + B.w * vec3(0.0, 0.3, 1.0) + A.y * vec3(0.6, 0.5, 0.0) + C.x * vec3(0.0, 0.8, 0.8);
    if (dbg == 3u && water) alb = sd < 0.0 ? (hasLevel ? vec3(1.0, 0.0, 1.0) : vec3(0.5, 0.0, 1.0)) : vec3(0.0, 0.0, 0.0);
    pl.albedo = alb;
    pl.t = t;
    pl.n = n;
    pl.ng = ng;
    pl.flags = MAT_TERRAIN | (water ? FLAG_WATER : 0u) | (water && !lake ? FLAG_SEA : 0u) | (seawall ? FLAG_SEAWALL : 0u);
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
