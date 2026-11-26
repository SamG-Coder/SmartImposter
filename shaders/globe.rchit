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
const int C_WAYS = 12;
const int C_WAY_FIRST[13] = int[13](0, 399, 439, 590, 746, 929, 1122, 1310, 1714, 1769, 1824, 1879, 1934);
const int C_WAY_KIND[12] = int[12](1, 1, 2, 2, 0, 0, 0, 2, 2, 2, 2, 2);
const float C_WAY_HALF[12] = float[12](15.0, 13.0, 5.5, 5.5, 14.0, 14.0, 14.0, 6.0, 5.0, 5.0, 5.0, 5.0);
const vec2 C_WAY_P[1934] = vec2[1934](vec2(-3350.0, -7000.0), vec2(-3349.0, -6979.2), vec2(-3347.8, -6956.9), vec2(-3346.7, -6933.0), vec2(-3345.4, -6907.7), vec2(-3344.1, -6881.0), vec2(-3342.6, -6852.9), vec2(-3341.2, -6823.5), vec2(-3339.6, -6792.9), vec2(-3338.1, -6761.0), vec2(-3336.4, -6728.0), vec2(-3334.7, -6693.8), vec2(-3332.9, -6658.6), vec2(-3331.1, -6622.4), vec2(-3329.3, -6585.2), vec2(-3327.3, -6547.0), vec2(-3325.4, -6508.0), vec2(-3323.4, -6468.1), vec2(-3321.4, -6427.5), vec2(-3319.3, -6386.1), vec2(-3317.2, -6344.0), vec2(-3315.1, -6301.3), vec2(-3312.9, -6258.0), vec2(-3310.7, -6214.1), vec2(-3308.5, -6169.8), vec2(-3306.2, -6125.0), vec2(-3304.0, -6079.8), vec2(-3301.7, -6034.3), vec2(-3299.4, -5988.4), vec2(-3297.1, -5942.3), vec2(-3294.8, -5896.0), vec2(-3292.5, -5849.5), vec2(-3290.1, -5802.9), vec2(-3287.8, -5756.3), vec2(-3285.5, -5709.6), vec2(-3283.2, -5663.0), vec2(-3280.8, -5616.4), vec2(-3278.5, -5570.0), vec2(-3276.2, -5523.8), vec2(-3273.9, -5477.8), vec2(-3271.6, -5432.0), vec2(-3269.3, -5386.6), vec2(-3267.1, -5341.5), vec2(-3264.8, -5296.9), vec2(-3262.6, -5252.7), vec2(-3260.4, -5209.0), vec2(-3258.3, -5165.9), vec2(-3256.2, -5123.4), vec2(-3254.1, -5081.5), vec2(-3252.0, -5040.4), vec2(-3250.0, -5000.0), vec2(-3248.0, -4959.9), vec2(-3246.0, -4919.6), vec2(-3244.0, -4879.2), vec2(-3242.0, -4838.5), vec2(-3240.0, -4797.8), vec2(-3238.0, -4756.8), vec2(-3236.0, -4715.8), vec2(-3234.0, -4674.6), vec2(-3232.0, -4633.4), vec2(-3230.0, -4592.0), vec2(-3228.0, -4550.6), vec2(-3226.0, -4509.1), vec2(-3224.0, -4467.5), vec2(-3222.0, -4425.9), vec2(-3220.0, -4384.2), vec2(-3218.0, -4342.6), vec2(-3216.0, -4300.9), vec2(-3214.0, -4259.3), vec2(-3212.0, -4217.6), vec2(-3210.0, -4176.0), vec2(-3208.0, -4134.4), vec2(-3206.0, -4092.9), vec2(-3204.0, -4051.4), vec2(-3202.0, -4010.0), vec2(-3200.0, -3968.8), vec2(-3198.0, -3927.6), vec2(-3196.0, -3886.5), vec2(-3194.0, -3845.5), vec2(-3192.0, -3804.7), vec2(-3190.0, -3764.0), vec2(-3188.0, -3723.5), vec2(-3186.0, -3683.1), vec2(-3184.0, -3643.0), vec2(-3182.0, -3603.0), vec2(-3180.0, -3563.2), vec2(-3178.0, -3523.7), vec2(-3176.0, -3484.4), vec2(-3174.0, -3445.3), vec2(-3172.0, -3406.5), vec2(-3170.0, -3368.0), vec2(-3168.0, -3329.7), vec2(-3166.0, -3291.8), vec2(-3164.0, -3254.1), vec2(-3162.0, -3216.8), vec2(-3160.0, -3179.8), vec2(-3158.0, -3143.1), vec2(-3156.0, -3106.7), vec2(-3154.0, -3070.8), vec2(-3152.0, -3035.2), vec2(-3150.0, -3000.0), vec2(-3147.3, -2953.0), vec2(-3144.5, -2906.6), vec2(-3141.7, -2860.7), vec2(-3138.9, -2815.3), vec2(-3136.1, -2770.4), vec2(-3133.2, -2726.1), vec2(-3130.4, -2682.2), vec2(-3127.5, -2638.8), vec2(-3124.6, -2595.8), vec2(-3121.6, -2553.3), vec2(-3118.7, -2511.1), vec2(-3115.8, -2469.4), vec2(-3112.9, -2428.0), vec2(-3109.9, -2387.0), vec2(-3107.0, -2346.3), vec2(-3104.1, -2305.9), vec2(-3101.2, -2265.8), vec2(-3098.3, -2226.0), vec2(-3095.4, -2186.5), vec2(-3092.6, -2147.2), vec2(-3089.8, -2108.2), vec2(-3087.0, -2069.3), vec2(-3084.2, -2030.7), vec2(-3081.4, -1992.2), vec2(-3078.7, -1953.9), vec2(-3076.1, -1915.7), vec2(-3073.4, -1877.7), vec2(-3070.8, -1839.7), vec2(-3068.3, -1801.9), vec2(-3065.8, -1764.1), vec2(-3063.4, -1726.4), vec2(-3061.0, -1688.6), vec2(-3058.7, -1651.0), vec2(-3056.4, -1613.3), vec2(-3054.2, -1575.6), vec2(-3052.1, -1537.8), vec2(-3050.0, -1500.0), vec2(-3047.6, -1456.5), vec2(-3045.3, -1413.4), vec2(-3042.9, -1370.8), vec2(-3040.5, -1328.6), vec2(-3038.1, -1286.8), vec2(-3035.7, -1245.3), vec2(-3033.3, -1204.1), vec2(-3030.9, -1163.3), vec2(-3028.6, -1122.7), vec2(-3026.3, -1082.3), vec2(-3024.1, -1042.2), vec2(-3021.9, -1002.2), vec2(-3019.7, -962.5), vec2(-3017.7, -922.8), vec2(-3015.7, -883.2), vec2(-3013.8, -843.8), vec2(-3011.9, -804.3), vec2(-3010.2, -764.9), vec2(-3008.6, -725.5), vec2(-3007.0, -686.0), vec2(-3005.6, -646.5), vec2(-3004.3, -606.9), vec2(-3003.2, -567.2), vec2(-3002.2, -527.3), vec2(-3001.3, -487.3), vec2(-3000.6, -447.1), vec2(-3000.1, -406.6), vec2(-2999.7, -365.9), vec2(-2999.5, -324.9), vec2(-2999.5, -283.6), vec2(-2999.6, -242.0), vec2(-3000.0, -200.0), vec2(-3000.5, -161.4), vec2(-3001.1, -122.8), vec2(-3001.9, -84.3), vec2(-3002.8, -45.6), vec2(-3003.8, -7.0), vec2(-3005.0, 31.7), vec2(-3006.2, 70.4), vec2(-3007.6, 109.2), vec2(-3009.1, 148.0), vec2(-3010.8, 186.9), vec2(-3012.5, 225.8), vec2(-3014.3, 264.9), vec2(-3016.3, 304.0), vec2(-3018.3, 343.2), vec2(-3020.5, 382.5), vec2(-3022.7, 421.9), vec2(-3025.0, 461.4), vec2(-3027.5, 501.1), vec2(-3030.0, 540.9), vec2(-3032.6, 580.8), vec2(-3035.3, 620.8), vec2(-3038.0, 661.0), vec2(-3040.9, 701.3), vec2(-3043.8, 741.8), vec2(-3046.8, 782.5), vec2(-3049.9, 823.4), vec2(-3053.0, 864.4), vec2(-3056.2, 905.6), vec2(-3059.4, 947.0), vec2(-3062.7, 988.6), vec2(-3066.1, 1030.5), vec2(-3069.5, 1072.5), vec2(-3072.9, 1114.8), vec2(-3076.4, 1157.3), vec2(-3080.0, 1200.0), vec2(-3083.2, 1237.6), vec2(-3086.4, 1275.4), vec2(-3089.8, 1313.3), vec2(-3093.2, 1351.5), vec2(-3096.7, 1389.7), vec2(-3100.3, 1428.2), vec2(-3104.0, 1466.8), vec2(-3107.8, 1505.6), vec2(-3111.6, 1544.5), vec2(-3115.5, 1583.6), vec2(-3119.4, 1622.8), vec2(-3123.4, 1662.2), vec2(-3127.5, 1701.6), vec2(-3131.7, 1741.2), vec2(-3135.8, 1781.0), vec2(-3140.1, 1820.8), vec2(-3144.4, 1860.8), vec2(-3148.7, 1900.8), vec2(-3153.1, 1941.0), vec2(-3157.5, 1981.2), vec2(-3162.0, 2021.6), vec2(-3166.4, 2062.1), vec2(-3171.0, 2102.6), vec2(-3175.5, 2143.2), vec2(-3180.1, 2183.9), vec2(-3184.7, 2224.6), vec2(-3189.3, 2265.5), vec2(-3194.0, 2306.3), vec2(-3198.6, 2347.3), vec2(-3203.3, 2388.3), vec2(-3208.0, 2429.3), vec2(-3212.6, 2470.4), vec2(-3217.3, 2511.5), vec2(-3222.0, 2552.7), vec2(-3226.7, 2593.8), vec2(-3231.4, 2635.1), vec2(-3236.1, 2676.3), vec2(-3240.7, 2717.5), vec2(-3245.4, 2758.8), vec2(-3250.0, 2800.0), vec2(-3254.4, 2839.3), vec2(-3258.9, 2878.5), vec2(-3263.5, 2917.6), vec2(-3268.1, 2956.8), vec2(-3272.7, 2995.9), vec2(-3277.4, 3035.0), vec2(-3282.2, 3074.1), vec2(-3287.0, 3113.2), vec2(-3291.8, 3152.3), vec2(-3296.6, 3191.4), vec2(-3301.5, 3230.5), vec2(-3306.4, 3269.7), vec2(-3311.3, 3308.9), vec2(-3316.3, 3348.1), vec2(-3321.3, 3387.5), vec2(-3326.2, 3426.8), vec2(-3331.2, 3466.3), vec2(-3336.2, 3505.8), vec2(-3341.2, 3545.5), vec2(-3346.2, 3585.2), vec2(-3351.2, 3625.0), vec2(-3356.2, 3664.9), vec2(-3361.2, 3705.0), vec2(-3366.2, 3745.2), vec2(-3371.2, 3785.5), vec2(-3376.1, 3826.0), vec2(-3381.0, 3866.6), vec2(-3385.9, 3907.4), vec2(-3390.8, 3948.4), vec2(-3395.6, 3989.5), vec2(-3400.4, 4030.8), vec2(-3405.2, 4072.3), vec2(-3409.9, 4114.1), vec2(-3414.6, 4156.0), vec2(-3419.2, 4198.1), vec2(-3423.8, 4240.5), vec2(-3428.3, 4283.1), vec2(-3432.8, 4326.0), vec2(-3437.2, 4369.1), vec2(-3441.5, 4412.5), vec2(-3445.8, 4456.1), vec2(-3450.0, 4500.0), vec2(-3453.5, 4537.0), vec2(-3457.0, 4574.1), vec2(-3460.4, 4611.2), vec2(-3463.8, 4648.4), vec2(-3467.2, 4685.6), vec2(-3470.6, 4722.9), vec2(-3474.0, 4760.3), vec2(-3477.4, 4797.7), vec2(-3480.7, 4835.2), vec2(-3484.0, 4872.8), vec2(-3487.3, 4910.5), vec2(-3490.6, 4948.3), vec2(-3493.8, 4986.1), vec2(-3497.0, 5024.1), vec2(-3500.2, 5062.2), vec2(-3503.4, 5100.4), vec2(-3506.6, 5138.7), vec2(-3509.8, 5177.1), vec2(-3512.9, 5215.7), vec2(-3516.0, 5254.4), vec2(-3519.1, 5293.2), vec2(-3522.2, 5332.2), vec2(-3525.2, 5371.3), vec2(-3528.2, 5410.6), vec2(-3531.2, 5450.0), vec2(-3534.2, 5489.6), vec2(-3537.2, 5529.3), vec2(-3540.2, 5569.2), vec2(-3543.1, 5609.3), vec2(-3546.0, 5649.6), vec2(-3548.9, 5690.1), vec2(-3551.8, 5730.7), vec2(-3554.6, 5771.5), vec2(-3557.4, 5812.6), vec2(-3560.2, 5853.8), vec2(-3563.0, 5895.2), vec2(-3565.8, 5936.9), vec2(-3568.6, 5978.8), vec2(-3571.3, 6020.9), vec2(-3574.0, 6063.2), vec2(-3576.7, 6105.8), vec2(-3579.4, 6148.6), vec2(-3582.0, 6191.6), vec2(-3584.6, 6234.9), vec2(-3587.2, 6278.4), vec2(-3589.8, 6322.2), vec2(-3592.4, 6366.2), vec2(-3595.0, 6410.6), vec2(-3597.5, 6455.1), vec2(-3600.0, 6500.0), vec2(-3602.0, 6536.7), vec2(-3604.0, 6574.4), vec2(-3606.0, 6612.8), vec2(-3608.1, 6652.0), vec2(-3610.1, 6692.0), vec2(-3612.1, 6732.8), vec2(-3614.1, 6774.2), vec2(-3616.1, 6816.2), vec2(-3618.1, 6858.9), vec2(-3620.1, 6902.1), vec2(-3622.0, 6945.9), vec2(-3624.0, 6990.2), vec2(-3626.0, 7034.9), vec2(-3627.9, 7080.0), vec2(-3629.9, 7125.5), vec2(-3631.8, 7171.4), vec2(-3633.8, 7217.6), vec2(-3635.7, 7264.0), vec2(-3637.6, 7310.7), vec2(-3639.5, 7357.6), vec2(-3641.4, 7404.6), vec2(-3643.2, 7451.7), vec2(-3645.1, 7498.9), vec2(-3646.9, 7546.2), vec2(-3648.8, 7593.5), vec2(-3650.6, 7640.7), vec2(-3652.4, 7687.8), vec2(-3654.1, 7734.9), vec2(-3655.9, 7781.8), vec2(-3657.7, 7828.5), vec2(-3659.4, 7875.0), vec2(-3661.1, 7921.2), vec2(-3662.8, 7967.2), vec2(-3664.4, 8012.8), vec2(-3666.1, 8058.0), vec2(-3667.7, 8102.8), vec2(-3669.3, 8147.2), vec2(-3670.9, 8191.1), vec2(-3672.4, 8234.4), vec2(-3673.9, 8277.2), vec2(-3675.4, 8319.4), vec2(-3676.9, 8361.0), vec2(-3678.4, 8401.8), vec2(-3679.8, 8442.0), vec2(-3681.2, 8481.4), vec2(-3682.5, 8520.1), vec2(-3683.9, 8557.9), vec2(-3685.2, 8594.8), vec2(-3686.4, 8630.8), vec2(-3687.7, 8665.9), vec2(-3688.9, 8700.0), vec2(-3690.1, 8733.1), vec2(-3691.2, 8765.2), vec2(-3692.4, 8796.1), vec2(-3693.4, 8826.0), vec2(-3694.5, 8854.6), vec2(-3695.5, 8882.1), vec2(-3696.5, 8908.3), vec2(-3697.4, 8933.2), vec2(-3698.3, 8956.9), vec2(-3699.2, 8979.1), vec2(-3700.0, 9000.0), vec2(-3010.0, 700.0), vec2(-2986.7, 707.1), vec2(-2956.4, 716.6), vec2(-2920.4, 728.0), vec2(-2880.0, 740.6), vec2(-2836.5, 753.8), vec2(-2791.1, 766.8), vec2(-2745.2, 779.1), vec2(-2700.0, 790.0), vec2(-2663.6, 798.0), vec2(-2625.8, 805.9), vec2(-2586.6, 813.7), vec2(-2546.5, 821.4), vec2(-2505.6, 828.8), vec2(-2464.3, 835.8), vec2(-2422.8, 842.6), vec2(-2381.4, 848.9), vec2(-2340.4, 854.7), vec2(-2300.0, 860.0), vec2(-2260.2, 864.8), vec2(-2220.8, 869.2), vec2(-2181.6, 873.2), vec2(-2142.4, 876.8), vec2(-2103.1, 880.0), vec2(-2063.6, 882.8), vec2(-2023.7, 885.2), vec2(-1983.2, 887.2), vec2(-1942.0, 888.8), vec2(-1900.0, 890.0), vec2(-1859.3, 890.6), vec2(-1815.1, 890.5), vec2(-1768.7, 889.9), vec2(-1721.1, 888.9), vec2(-1673.5, 887.6), vec2(-1626.9, 886.1), vec2(-1582.6, 884.6), vec2(-1541.6, 883.1), vec2(-1505.1, 881.8), vec2(-1474.2, 880.7), vec2(-1450.0, 880.0), vec2(-3018.0, -1300.0), vec2(-3017.9, -1302.1), vec2(-3017.9, -1304.5), vec2(-3017.8, -1307.0), vec2(-3017.7, -1309.8), vec2(-3017.7, -1312.7), vec2(-3017.6, -1315.8), vec2(-3017.6, -1319.1), vec2(-3017.5, -1322.5), vec2(-3017.4, -1326.0), vec2(-3017.4, -1329.7), vec2(-3017.3, -1333.6), vec2(-3017.2, -1337.5), vec2(-3017.1, -1341.6), vec2(-3017.1, -1345.8), vec2(-3017.0, -1350.1), vec2(-3016.9, -1354.4), vec2(-3016.8, -1358.9), vec2(-3016.7, -1363.4), vec2(-3016.6, -1367.9), vec2(-3016.5, -1372.5), vec2(-3016.3, -1377.2), vec2(-3016.2, -1381.9), vec2(-3016.1, -1386.6), vec2(-3015.9, -1391.3), vec2(-3015.8, -1396.0), vec2(-3015.6, -1400.7), vec2(-3015.4, -1405.5), vec2(-3015.2, -1410.2), vec2(-3015.0, -1414.8), vec2(-3014.8, -1419.4), vec2(-3014.6, -1424.0), vec2(-3014.4, -1428.5), vec2(-3014.1, -1433.0), vec2(-3013.9, -1437.4), vec2(-3013.6, -1441.7), vec2(-3013.3, -1445.9), vec2(-3013.0, -1450.0), vec2(-3012.7, -1454.1), vec2(-3012.4, -1458.1), vec2(-3012.0, -1462.2), vec2(-3011.7, -1466.2), vec2(-3011.3, -1470.3), vec2(-3011.0, -1474.3), vec2(-3010.6, -1478.4), vec2(-3010.2, -1482.4), vec2(-3009.8, -1486.5), vec2(-3009.4, -1490.5), vec2(-3009.0, -1494.6), vec2(-3008.6, -1498.6), vec2(-3008.1, -1502.7), vec2(-3007.7, -1506.8), vec2(-3007.2, -1510.8), vec2(-3006.8, -1514.9), vec2(-3006.3, -1518.9), vec2(-3005.8, -1523.0), vec2(-3005.3, -1527.0), vec2(-3004.8, -1531.1), vec2(-3004.3, -1535.1), vec2(-3003.8, -1539.2), vec2(-3003.3, -1543.2), vec2(-3002.7, -1547.3), vec2(-3002.2, -1551.4), vec2(-3001.6, -1555.4), vec2(-3001.1, -1559.5), vec2(-3000.5, -1563.5), vec2(-2999.9, -1567.6), vec2(-2999.3, -1571.6), vec2(-2998.7, -1575.7), vec2(-2998.1, -1579.7), vec2(-2997.5, -1583.8), vec2(-2996.9, -1587.8), vec2(-2996.3, -1591.9), vec2(-2995.6, -1595.9), vec2(-2995.0, -1600.0), vec2(-2994.4, -1603.9), vec2(-2993.7, -1607.9), vec2(-2993.0, -1611.8), vec2(-2992.4, -1615.8), vec2(-2991.7, -1619.7), vec2(-2991.0, -1623.7), vec2(-2990.3, -1627.6), vec2(-2989.6, -1631.6), vec2(-2988.8, -1635.5), vec2(-2988.1, -1639.5), vec2(-2987.3, -1643.4), vec2(-2986.6, -1647.4), vec2(-2985.8, -1651.3), vec2(-2985.0, -1655.3), vec2(-2984.3, -1659.2), vec2(-2983.5, -1663.2), vec2(-2982.7, -1667.1), vec2(-2981.9, -1671.1), vec2(-2981.1, -1675.0), vec2(-2980.2, -1678.9), vec2(-2979.4, -1682.9), vec2(-2978.6, -1686.8), vec2(-2977.8, -1690.8), vec2(-2976.9, -1694.7), vec2(-2976.1, -1698.7), vec2(-2975.3, -1702.6), vec2(-2974.4, -1706.6), vec2(-2973.6, -1710.5), vec2(-2972.7, -1714.5), vec2(-2971.9, -1718.4), vec2(-2971.0, -1722.4), vec2(-2970.1, -1726.3), vec2(-2969.3, -1730.3), vec2(-2968.4, -1734.2), vec2(-2967.6, -1738.2), vec2(-2966.7, -1742.1), vec2(-2965.9, -1746.1), vec2(-2965.0, -1750.0), vec2(-2964.1, -1754.0), vec2(-2963.2, -1758.1), vec2(-2962.3, -1762.3), vec2(-2961.4, -1766.5), vec2(-2960.4, -1770.9), vec2(-2959.4, -1775.3), vec2(-2958.4, -1779.7), vec2(-2957.3, -1784.2), vec2(-2956.3, -1788.7), vec2(-2955.3, -1793.3), vec2(-2954.2, -1797.9), vec2(-2953.1, -1802.5), vec2(-2952.0, -1807.1), vec2(-2951.0, -1811.7), vec2(-2949.9, -1816.3), vec2(-2948.8, -1820.9), vec2(-2947.7, -1825.4), vec2(-2946.7, -1829.9), vec2(-2945.6, -1834.4), vec2(-2944.6, -1838.8), vec2(-2943.5, -1843.1), vec2(-2942.5, -1847.4), vec2(-2941.5, -1851.6), vec2(-2940.5, -1855.8), vec2(-2939.6, -1859.8), vec2(-2938.6, -1863.7), vec2(-2937.7, -1867.5), vec2(-2936.8, -1871.2), vec2(-2936.0, -1874.8), vec2(-2935.2, -1878.3), vec2(-2934.4, -1881.6), vec2(-2933.6, -1884.7), vec2(-2932.9, -1887.7), vec2(-2932.2, -1890.5), vec2(-2931.6, -1893.2), vec2(-2931.0, -1895.6), vec2(-2930.5, -1897.9), vec2(-2930.0, -1900.0), vec2(-3277.0, 2800.0), vec2(-3277.3, 2802.1), vec2(-3277.7, 2804.5), vec2(-3278.0, 2807.0), vec2(-3278.4, 2809.8), vec2(-3278.8, 2812.7), vec2(-3279.2, 2815.8), vec2(-3279.6, 2819.1), vec2(-3280.0, 2822.5), vec2(-3280.5, 2826.0), vec2(-3281.0, 2829.7), vec2(-3281.5, 2833.6), vec2(-3282.0, 2837.5), vec2(-3282.5, 2841.6), vec2(-3283.0, 2845.8), vec2(-3283.6, 2850.1), vec2(-3284.2, 2854.4), vec2(-3284.8, 2858.9), vec2(-3285.4, 2863.4), vec2(-3286.0, 2867.9), vec2(-3286.6, 2872.5), vec2(-3287.3, 2877.2), vec2(-3288.0, 2881.9), vec2(-3288.7, 2886.6), vec2(-3289.4, 2891.3), vec2(-3290.1, 2896.0), vec2(-3290.8, 2900.7), vec2(-3291.6, 2905.5), vec2(-3292.4, 2910.2), vec2(-3293.1, 2914.8), vec2(-3293.9, 2919.4), vec2(-3294.8, 2924.0), vec2(-3295.6, 2928.5), vec2(-3296.4, 2933.0), vec2(-3297.3, 2937.4), vec2(-3298.2, 2941.7), vec2(-3299.1, 2945.9), vec2(-3300.0, 2950.0), vec2(-3300.9, 2953.8), vec2(-3301.8, 2957.7), vec2(-3302.7, 2961.5), vec2(-3303.7, 2965.4), vec2(-3304.6, 2969.2), vec2(-3305.6, 2973.1), vec2(-3306.6, 2976.9), vec2(-3307.6, 2980.8), vec2(-3308.7, 2984.6), vec2(-3309.7, 2988.5), vec2(-3310.8, 2992.3), vec2(-3311.9, 2996.2), vec2(-3313.0, 3000.0), vec2(-3314.1, 3003.8), vec2(-3315.3, 3007.7), vec2(-3316.4, 3011.5), vec2(-3317.6, 3015.4), vec2(-3318.7, 3019.2), vec2(-3319.9, 3023.1), vec2(-3321.1, 3026.9), vec2(-3322.3, 3030.8), vec2(-3323.5, 3034.6), vec2(-3324.7, 3038.5), vec2(-3326.0, 3042.3), vec2(-3327.2, 3046.2), vec2(-3328.4, 3050.0), vec2(-3329.7, 3053.8), vec2(-3331.0, 3057.7), vec2(-3332.2, 3061.5), vec2(-3333.5, 3065.4), vec2(-3334.8, 3069.2), vec2(-3336.0, 3073.1), vec2(-3337.3, 3076.9), vec2(-3338.6, 3080.8), vec2(-3339.9, 3084.6), vec2(-3341.2, 3088.5), vec2(-3342.4, 3092.3), vec2(-3343.7, 3096.2), vec2(-3345.0, 3100.0), vec2(-3346.3, 3103.8), vec2(-3347.6, 3107.7), vec2(-3348.9, 3111.5), vec2(-3350.2, 3115.4), vec2(-3351.5, 3119.2), vec2(-3352.9, 3123.1), vec2(-3354.2, 3126.9), vec2(-3355.6, 3130.8), vec2(-3356.9, 3134.6), vec2(-3358.3, 3138.5), vec2(-3359.6, 3142.3), vec2(-3361.0, 3146.2), vec2(-3362.4, 3150.0), vec2(-3363.8, 3153.8), vec2(-3365.2, 3157.7), vec2(-3366.6, 3161.5), vec2(-3368.0, 3165.4), vec2(-3369.4, 3169.2), vec2(-3370.8, 3173.1), vec2(-3372.3, 3176.9), vec2(-3373.7, 3180.8), vec2(-3375.1, 3184.6), vec2(-3376.6, 3188.5), vec2(-3378.0, 3192.3), vec2(-3379.5, 3196.2), vec2(-3380.9, 3200.0), vec2(-3382.4, 3203.8), vec2(-3383.8, 3207.7), vec2(-3385.3, 3211.5), vec2(-3386.8, 3215.4), vec2(-3388.2, 3219.2), vec2(-3389.7, 3223.1), vec2(-3391.2, 3226.9), vec2(-3392.6, 3230.8), vec2(-3394.1, 3234.6), vec2(-3395.6, 3238.5), vec2(-3397.1, 3242.3), vec2(-3398.5, 3246.2), vec2(-3400.0, 3250.0), vec2(-3401.5, 3253.8), vec2(-3403.0, 3257.7), vec2(-3404.5, 3261.6), vec2(-3406.1, 3265.7), vec2(-3407.7, 3269.8), vec2(-3409.3, 3273.9), vec2(-3411.0, 3278.1), vec2(-3412.6, 3282.4), vec2(-3414.3, 3286.7), vec2(-3416.1, 3291.0), vec2(-3417.8, 3295.4), vec2(-3419.5, 3299.7), vec2(-3421.3, 3304.1), vec2(-3423.0, 3308.5), vec2(-3424.8, 3312.8), vec2(-3426.5, 3317.2), vec2(-3428.3, 3321.5), vec2(-3430.0, 3325.9), vec2(-3431.7, 3330.1), vec2(-3433.4, 3334.4), vec2(-3435.1, 3338.6), vec2(-3436.8, 3342.7), vec2(-3438.5, 3346.8), vec2(-3440.1, 3350.8), vec2(-3441.7, 3354.7), vec2(-3443.2, 3358.6), vec2(-3444.8, 3362.4), vec2(-3446.3, 3366.0), vec2(-3447.7, 3369.6), vec2(-3449.1, 3373.0), vec2(-3450.5, 3376.4), vec2(-3451.8, 3379.6), vec2(-3453.0, 3382.7), vec2(-3454.2, 3385.6), vec2(-3455.3, 3388.4), vec2(-3456.4, 3391.1), vec2(-3457.4, 3393.6), vec2(-3458.3, 3395.9), vec2(-3459.2, 3398.0), vec2(-3460.0, 3400.0), vec2(-7500.0, -6500.0), vec2(-7486.0, -6484.5), vec2(-7471.1, -6468.1), vec2(-7455.4, -6450.7), vec2(-7438.9, -6432.4), vec2(-7421.6, -6413.2), vec2(-7403.5, -6393.2), vec2(-7384.7, -6372.4), vec2(-7365.1, -6350.7), vec2(-7344.9, -6328.2), vec2(-7323.9, -6305.0), vec2(-7302.3, -6281.0), vec2(-7280.0, -6256.3), vec2(-7257.1, -6230.9), vec2(-7233.6, -6204.8), vec2(-7209.5, -6178.1), vec2(-7184.9, -6150.7), vec2(-7159.7, -6122.8), vec2(-7134.0, -6094.3), vec2(-7107.7, -6065.2), vec2(-7081.0, -6035.6), vec2(-7053.9, -6005.4), vec2(-7026.3, -5974.8), vec2(-6998.3, -5943.8), vec2(-6969.9, -5912.3), vec2(-6941.1, -5880.4), vec2(-6912.0, -5848.1), vec2(-6882.6, -5815.4), vec2(-6852.8, -5782.4), vec2(-6822.7, -5749.1), vec2(-6792.4, -5715.5), vec2(-6761.8, -5681.6), vec2(-6731.0, -5647.4), vec2(-6700.0, -5613.1), vec2(-6668.9, -5578.5), vec2(-6637.5, -5543.8), vec2(-6606.0, -5508.9), vec2(-6574.4, -5473.8), vec2(-6542.7, -5438.7), vec2(-6510.9, -5403.5), vec2(-6479.0, -5368.2), vec2(-6447.1, -5332.9), vec2(-6415.2, -5297.6), vec2(-6383.3, -5262.3), vec2(-6351.4, -5227.0), vec2(-6319.6, -5191.8), vec2(-6287.8, -5156.7), vec2(-6256.1, -5121.7), vec2(-6224.5, -5086.8), vec2(-6193.1, -5052.1), vec2(-6161.8, -5017.5), vec2(-6130.7, -4983.1), vec2(-6099.8, -4949.0), vec2(-6069.0, -4915.1), vec2(-6038.6, -4881.5), vec2(-6008.3, -4848.2), vec2(-5978.4, -4815.2), vec2(-5948.8, -4782.5), vec2(-5919.4, -4750.3), vec2(-5890.4, -4718.4), vec2(-5861.8, -4686.9), vec2(-5833.6, -4655.8), vec2(-5805.7, -4625.3), vec2(-5778.3, -4595.1), vec2(-5751.3, -4565.6), vec2(-5724.8, -4536.5), vec2(-5698.8, -4508.0), vec2(-5673.3, -4480.1), vec2(-5648.3, -4452.7), vec2(-5623.9, -4426.0), vec2(-5600.0, -4400.0), vec2(-5567.7, -4364.8), vec2(-5535.7, -4329.9), vec2(-5503.8, -4295.3), vec2(-5472.1, -4261.0), vec2(-5440.6, -4226.8), vec2(-5409.3, -4193.0), vec2(-5378.2, -4159.4), vec2(-5347.3, -4126.1), vec2(-5316.6, -4093.0), vec2(-5286.1, -4060.1), vec2(-5255.7, -4027.5), vec2(-5225.6, -3995.2), vec2(-5195.7, -3963.1), vec2(-5165.9, -3931.2), vec2(-5136.3, -3899.6), vec2(-5107.0, -3868.2), vec2(-5077.8, -3837.0), vec2(-5048.8, -3806.1), vec2(-5020.0, -3775.4), vec2(-4991.4, -3744.9), vec2(-4963.0, -3714.7), vec2(-4934.8, -3684.6), vec2(-4906.7, -3654.8), vec2(-4878.9, -3625.2), vec2(-4851.2, -3595.9), vec2(-4823.8, -3566.7), vec2(-4796.5, -3537.7), vec2(-4769.5, -3509.0), vec2(-4742.6, -3480.4), vec2(-4715.9, -3452.1), vec2(-4689.4, -3424.0), vec2(-4663.1, -3396.0), vec2(-4637.0, -3368.3), vec2(-4611.1, -3340.7), vec2(-4585.4, -3313.4), vec2(-4559.9, -3286.2), vec2(-4534.5, -3259.2), vec2(-4509.4, -3232.5), vec2(-4484.4, -3205.8), vec2(-4459.7, -3179.4), vec2(-4435.1, -3153.2), vec2(-4410.7, -3127.1), vec2(-4386.5, -3101.2), vec2(-4362.6, -3075.5), vec2(-4338.8, -3049.9), vec2(-4315.1, -3024.5), vec2(-4291.7, -2999.3), vec2(-4268.5, -2974.2), vec2(-4245.5, -2949.3), vec2(-4222.6, -2924.6), vec2(-4200.0, -2900.0), vec2(-4165.6, -2862.5), vec2(-4131.9, -2825.7), vec2(-4099.0, -2789.6), vec2(-4066.9, -2754.1), vec2(-4035.4, -2719.3), vec2(-4004.6, -2685.0), vec2(-3974.4, -2651.4), vec2(-3944.8, -2618.3), vec2(-3915.8, -2585.8), vec2(-3887.3, -2553.8), vec2(-3859.3, -2522.2), vec2(-3831.7, -2491.1), vec2(-3804.6, -2460.5), vec2(-3777.8, -2430.2), vec2(-3751.5, -2400.4), vec2(-3725.4, -2370.9), vec2(-3699.7, -2341.7), vec2(-3674.2, -2312.8), vec2(-3648.9, -2284.3), vec2(-3623.8, -2256.0), vec2(-3598.9, -2227.9), vec2(-3574.1, -2200.0), vec2(-3549.4, -2172.3), vec2(-3524.7, -2144.8), vec2(-3500.1, -2117.4), vec2(-3475.5, -2090.1), vec2(-3450.8, -2062.9), vec2(-3426.0, -2035.7), vec2(-3401.2, -2008.6), vec2(-3376.2, -1981.5), vec2(-3351.0, -1954.4), vec2(-3325.6, -1927.2), vec2(-3300.0, -1900.0), vec2(-3269.3, -1867.6), vec2(-3237.9, -1834.8), vec2(-3205.9, -1801.6), vec2(-3173.5, -1768.2), vec2(-3140.6, -1734.6), vec2(-3107.5, -1700.9), vec2(-3074.2, -1667.2), vec2(-3040.8, -1633.5), vec2(-3007.4, -1600.0), vec2(-2974.1, -1566.7), vec2(-2941.0, -1533.8), vec2(-2908.2, -1501.2), vec2(-2875.7, -1469.1), vec2(-2843.8, -1437.5), vec2(-2812.4, -1406.6), vec2(-2781.6, -1376.4), vec2(-2751.7, -1347.0), vec2(-2722.6, -1318.5), vec2(-2694.4, -1290.9), vec2(-2667.3, -1264.4), vec2(-2641.4, -1239.1), vec2(-2616.7, -1214.9), vec2(-2593.4, -1192.0), vec2(-2571.4, -1170.6), vec2(-2551.0, -1150.5), vec2(-2532.3, -1132.0), vec2(-2515.2, -1115.2), vec2(-2500.0, -1100.0), vec2(-1800.0, 2600.0), vec2(-1816.1, 2613.4), vec2(-1833.6, 2628.0), vec2(-1852.4, 2643.6), vec2(-1872.6, 2660.3), vec2(-1893.9, 2678.1), vec2(-1916.5, 2696.8), vec2(-1940.2, 2716.4), vec2(-1965.0, 2737.0), vec2(-1990.9, 2758.4), vec2(-2017.7, 2780.6), vec2(-2045.5, 2803.6), vec2(-2074.2, 2827.4), vec2(-2103.7, 2851.9), vec2(-2134.0, 2877.0), vec2(-2165.0, 2902.7), vec2(-2196.8, 2929.0), vec2(-2229.1, 2955.8), vec2(-2262.1, 2983.2), vec2(-2295.6, 3010.9), vec2(-2329.5, 3039.2), vec2(-2364.0, 3067.7), vec2(-2398.7, 3096.6), vec2(-2433.9, 3125.8), vec2(-2469.3, 3155.3), vec2(-2504.9, 3185.0), vec2(-2540.7, 3214.8), vec2(-2576.7, 3244.8), vec2(-2612.7, 3274.9), vec2(-2648.8, 3305.0), vec2(-2684.8, 3335.1), vec2(-2720.8, 3365.2), vec2(-2756.7, 3395.2), vec2(-2792.4, 3425.1), vec2(-2827.8, 3454.9), vec2(-2863.0, 3484.5), vec2(-2897.9, 3513.8), vec2(-2932.3, 3542.9), vec2(-2966.4, 3571.6), vec2(-3000.0, 3600.0), vec2(-3028.3, 3623.9), vec2(-3056.7, 3647.9), vec2(-3085.2, 3672.0), vec2(-3113.8, 3696.1), vec2(-3142.5, 3720.2), vec2(-3171.3, 3744.4), vec2(-3200.1, 3768.7), vec2(-3229.1, 3793.1), vec2(-3258.2, 3817.5), vec2(-3287.3, 3842.0), vec2(-3316.6, 3866.6), vec2(-3345.9, 3891.2), vec2(-3375.4, 3916.0), vec2(-3404.9, 3940.8), vec2(-3434.5, 3965.7), vec2(-3464.3, 3990.8), vec2(-3494.1, 4015.9), vec2(-3524.0, 4041.1), vec2(-3554.0, 4066.4), vec2(-3584.1, 4091.8), vec2(-3614.3, 4117.4), vec2(-3644.6, 4143.0), vec2(-3675.0, 4168.8), vec2(-3705.5, 4194.6), vec2(-3736.1, 4220.6), vec2(-3766.7, 4246.7), vec2(-3797.5, 4273.0), vec2(-3828.4, 4299.4), vec2(-3859.3, 4325.9), vec2(-3890.4, 4352.5), vec2(-3921.5, 4379.3), vec2(-3952.7, 4406.2), vec2(-3984.1, 4433.3), vec2(-4015.5, 4460.5), vec2(-4047.0, 4487.9), vec2(-4078.6, 4515.5), vec2(-4110.3, 4543.2), vec2(-4142.2, 4571.0), vec2(-4174.1, 4599.0), vec2(-4206.0, 4627.2), vec2(-4238.1, 4655.6), vec2(-4270.3, 4684.1), vec2(-4302.6, 4712.8), vec2(-4335.0, 4741.7), vec2(-4367.4, 4770.7), vec2(-4400.0, 4800.0), vec2(-4427.9, 4825.1), vec2(-4455.9, 4850.4), vec2(-4484.1, 4875.9), vec2(-4512.4, 4901.6), vec2(-4540.9, 4927.5), vec2(-4569.5, 4953.5), vec2(-4598.3, 4979.7), vec2(-4627.2, 5006.1), vec2(-4656.2, 5032.6), vec2(-4685.4, 5059.3), vec2(-4714.7, 5086.2), vec2(-4744.0, 5113.2), vec2(-4773.5, 5140.3), vec2(-4803.1, 5167.6), vec2(-4832.7, 5194.9), vec2(-4862.5, 5222.4), vec2(-4892.3, 5250.1), vec2(-4922.2, 5277.8), vec2(-4952.2, 5305.6), vec2(-4982.2, 5333.5), vec2(-5012.3, 5361.5), vec2(-5042.5, 5389.7), vec2(-5072.7, 5417.8), vec2(-5102.9, 5446.1), vec2(-5133.1, 5474.4), vec2(-5163.4, 5502.8), vec2(-5193.8, 5531.2), vec2(-5224.1, 5559.7), vec2(-5254.4, 5588.3), vec2(-5284.8, 5616.9), vec2(-5315.1, 5645.5), vec2(-5345.5, 5674.1), vec2(-5375.8, 5702.8), vec2(-5406.1, 5731.5), vec2(-5436.4, 5760.2), vec2(-5466.7, 5788.9), vec2(-5496.9, 5817.6), vec2(-5527.1, 5846.3), vec2(-5557.2, 5875.0), vec2(-5587.3, 5903.6), vec2(-5617.4, 5932.3), vec2(-5647.3, 5960.9), vec2(-5677.2, 5989.5), vec2(-5707.1, 6018.0), vec2(-5736.8, 6046.5), vec2(-5766.5, 6075.0), vec2(-5796.0, 6103.4), vec2(-5825.5, 6131.7), vec2(-5854.9, 6159.9), vec2(-5884.1, 6188.1), vec2(-5913.3, 6216.2), vec2(-5942.3, 6244.2), vec2(-5971.2, 6272.2), vec2(-6000.0, 6300.0), vec2(-6029.5, 6328.6), vec2(-6059.4, 6357.6), vec2(-6089.7, 6387.2), vec2(-6120.4, 6417.2), vec2(-6151.4, 6447.6), vec2(-6182.8, 6478.3), vec2(-6214.4, 6509.5), vec2(-6246.4, 6540.9), vec2(-6278.5, 6572.7), vec2(-6310.9, 6604.7), vec2(-6343.4, 6636.9), vec2(-6376.1, 6669.4), vec2(-6409.0, 6702.0), vec2(-6441.9, 6734.7), vec2(-6474.9, 6767.6), vec2(-6507.9, 6800.5), vec2(-6540.9, 6833.5), vec2(-6574.0, 6866.6), vec2(-6606.9, 6899.6), vec2(-6639.9, 6932.5), vec2(-6672.7, 6965.4), vec2(-6705.3, 6998.2), vec2(-6737.8, 7030.9), vec2(-6770.2, 7063.4), vec2(-6802.3, 7095.7), vec2(-6834.2, 7127.8), vec2(-6865.8, 7159.6), vec2(-6897.1, 7191.2), vec2(-6928.0, 7222.4), vec2(-6958.7, 7253.3), vec2(-6988.9, 7283.9), vec2(-7018.7, 7314.0), vec2(-7048.1, 7343.7), vec2(-7077.0, 7372.9), vec2(-7105.5, 7401.6), vec2(-7133.4, 7429.9), vec2(-7160.7, 7457.5), vec2(-7187.5, 7484.6), vec2(-7213.6, 7511.0), vec2(-7239.1, 7536.9), vec2(-7264.0, 7562.0), vec2(-7288.1, 7586.4), vec2(-7311.6, 7610.1), vec2(-7334.3, 7633.1), vec2(-7356.2, 7655.2), vec2(-7377.3, 7676.5), vec2(-7397.5, 7697.0), vec2(-7416.9, 7716.5), vec2(-7435.4, 7735.2), vec2(-7453.0, 7752.9), vec2(-7469.7, 7769.6), vec2(-7485.3, 7785.3), vec2(-7500.0, 7800.0), vec2(4300.0, -1500.0), vec2(4297.8, -1478.6), vec2(4295.2, -1455.2), vec2(4292.2, -1429.7), vec2(4288.9, -1402.3), vec2(4285.3, -1373.1), vec2(4281.5, -1342.1), vec2(4277.3, -1309.5), vec2(4272.9, -1275.3), vec2(4268.4, -1239.6), vec2(4263.6, -1202.5), vec2(4258.7, -1164.2), vec2(4253.6, -1124.6), vec2(4248.4, -1083.8), vec2(4243.1, -1042.1), vec2(4237.7, -999.4), vec2(4232.3, -955.8), vec2(4226.9, -911.5), vec2(4221.5, -866.5), vec2(4216.1, -820.9), vec2(4210.7, -774.8), vec2(4205.4, -728.2), vec2(4200.2, -681.4), vec2(4195.2, -634.3), vec2(4190.2, -587.1), vec2(4185.5, -539.8), vec2(4180.9, -492.5), vec2(4176.5, -445.4), vec2(4172.4, -398.4), vec2(4168.6, -351.8), vec2(4165.0, -305.5), vec2(4161.7, -259.8), vec2(4158.8, -214.5), vec2(4156.2, -170.0), vec2(4154.1, -126.2), vec2(4152.3, -83.2), vec2(4150.9, -41.1), vec2(4150.0, 0.0), vec2(4149.5, 40.5), vec2(4149.2, 81.1), vec2(4149.1, 121.6), vec2(4149.3, 162.2), vec2(4149.7, 202.7), vec2(4150.3, 243.2), vec2(4151.2, 283.8), vec2(4152.3, 324.3), vec2(4153.6, 364.9), vec2(4155.0, 405.4), vec2(4156.7, 445.9), vec2(4158.6, 486.5), vec2(4160.7, 527.0), vec2(4162.9, 567.6), vec2(4165.3, 608.1), vec2(4167.9, 648.6), vec2(4170.6, 689.2), vec2(4173.5, 729.7), vec2(4176.5, 770.3), vec2(4179.7, 810.8), vec2(4183.0, 851.4), vec2(4186.5, 891.9), vec2(4190.1, 932.4), vec2(4193.8, 973.0), vec2(4197.6, 1013.5), vec2(4201.5, 1054.1), vec2(4205.5, 1094.6), vec2(4209.6, 1135.1), vec2(4213.8, 1175.7), vec2(4218.1, 1216.2), vec2(4222.5, 1256.8), vec2(4226.9, 1297.3), vec2(4231.4, 1337.8), vec2(4236.0, 1378.4), vec2(4240.6, 1418.9), vec2(4245.3, 1459.5), vec2(4250.0, 1500.0), vec2(4254.7, 1539.5), vec2(4259.5, 1578.9), vec2(4264.6, 1618.4), vec2(4269.7, 1657.9), vec2(4275.1, 1697.4), vec2(4280.6, 1736.8), vec2(4286.2, 1776.3), vec2(4291.9, 1815.8), vec2(4297.8, 1855.3), vec2(4303.8, 1894.7), vec2(4309.9, 1934.2), vec2(4316.2, 1973.7), vec2(4322.5, 2013.2), vec2(4328.9, 2052.6), vec2(4335.5, 2092.1), vec2(4342.1, 2131.6), vec2(4348.8, 2171.1), vec2(4355.6, 2210.5), vec2(4362.5, 2250.0), vec2(4369.4, 2289.5), vec2(4376.4, 2328.9), vec2(4383.5, 2368.4), vec2(4390.6, 2407.9), vec2(4397.8, 2447.4), vec2(4405.0, 2486.8), vec2(4412.2, 2526.3), vec2(4419.5, 2565.8), vec2(4426.8, 2605.3), vec2(4434.1, 2644.7), vec2(4441.5, 2684.2), vec2(4448.8, 2723.7), vec2(4456.2, 2763.2), vec2(4463.5, 2802.6), vec2(4470.8, 2842.1), vec2(4478.2, 2881.6), vec2(4485.5, 2921.1), vec2(4492.7, 2960.5), vec2(4500.0, 3000.0), vec2(4507.3, 3039.5), vec2(4514.7, 3078.9), vec2(4522.3, 3118.4), vec2(4530.0, 3157.9), vec2(4537.7, 3197.4), vec2(4545.6, 3236.8), vec2(4553.6, 3276.3), vec2(4561.6, 3315.8), vec2(4569.7, 3355.3), vec2(4577.9, 3394.7), vec2(4586.2, 3434.2), vec2(4594.5, 3473.7), vec2(4602.8, 3513.2), vec2(4611.1, 3552.6), vec2(4619.5, 3592.1), vec2(4627.9, 3631.6), vec2(4636.3, 3671.1), vec2(4644.7, 3710.5), vec2(4653.1, 3750.0), vec2(4661.5, 3789.5), vec2(4669.9, 3828.9), vec2(4678.2, 3868.4), vec2(4686.5, 3907.9), vec2(4694.7, 3947.4), vec2(4702.8, 3986.8), vec2(4710.9, 4026.3), vec2(4719.0, 4065.8), vec2(4726.9, 4105.3), vec2(4734.8, 4144.7), vec2(4742.5, 4184.2), vec2(4750.2, 4223.7), vec2(4757.7, 4263.2), vec2(4765.1, 4302.6), vec2(4772.4, 4342.1), vec2(4779.5, 4381.6), vec2(4786.5, 4421.1), vec2(4793.3, 4460.5), vec2(4800.0, 4500.0), vec2(4806.8, 4541.1), vec2(4813.5, 4583.2), vec2(4820.2, 4626.2), vec2(4827.0, 4670.0), vec2(4833.7, 4714.5), vec2(4840.3, 4759.8), vec2(4847.0, 4805.5), vec2(4853.5, 4851.8), vec2(4860.1, 4898.4), vec2(4866.6, 4945.4), vec2(4873.0, 4992.5), vec2(4879.4, 5039.8), vec2(4885.7, 5087.1), vec2(4891.9, 5134.3), vec2(4898.0, 5181.4), vec2(4904.1, 5228.2), vec2(4910.0, 5274.8), vec2(4915.9, 5320.9), vec2(4921.6, 5366.5), vec2(4927.2, 5411.5), vec2(4932.8, 5455.8), vec2(4938.1, 5499.4), vec2(4943.4, 5542.1), vec2(4948.5, 5583.8), vec2(4953.5, 5624.6), vec2(4958.3, 5664.2), vec2(4963.0, 5702.5), vec2(4967.5, 5739.6), vec2(4971.9, 5775.3), vec2(4976.1, 5809.5), vec2(4980.1, 5842.1), vec2(4983.9, 5873.1), vec2(4987.5, 5902.3), vec2(4990.9, 5929.7), vec2(4994.2, 5955.2), vec2(4997.2, 5978.6), vec2(5000.0, 6000.0), vec2(-3020.0, -400.0), vec2(-3019.8, -397.9), vec2(-3019.5, -395.7), vec2(-3019.2, -393.4), vec2(-3019.0, -390.9), vec2(-3018.7, -388.4), vec2(-3018.4, -385.7), vec2(-3018.1, -382.9), vec2(-3017.8, -380.0), vec2(-3017.5, -377.0), vec2(-3017.2, -373.9), vec2(-3016.9, -370.7), vec2(-3016.5, -367.3), vec2(-3016.2, -364.0), vec2(-3015.8, -360.5), vec2(-3015.5, -356.9), vec2(-3015.1, -353.2), vec2(-3014.8, -349.5), vec2(-3014.4, -345.7), vec2(-3014.0, -341.8), vec2(-3013.6, -337.9), vec2(-3013.2, -333.8), vec2(-3012.8, -329.8), vec2(-3012.4, -325.6), vec2(-3011.9, -321.4), vec2(-3011.5, -317.1), vec2(-3011.1, -312.8), vec2(-3010.6, -308.5), vec2(-3010.2, -304.1), vec2(-3009.7, -299.6), vec2(-3009.2, -295.1), vec2(-3008.8, -290.6), vec2(-3008.3, -286.1), vec2(-3007.8, -281.5), vec2(-3007.3, -276.9), vec2(-3006.8, -272.3), vec2(-3006.2, -267.6), vec2(-3005.7, -262.9), vec2(-3005.2, -258.3), vec2(-3004.6, -253.6), vec2(-3004.1, -248.9), vec2(-3003.5, -244.2), vec2(-3003.0, -239.5), vec2(-3002.4, -234.8), vec2(-3001.8, -230.1), vec2(-3001.2, -225.4), vec2(-3000.6, -220.7), vec2(-3000.0, -216.0), vec2(-2999.4, -211.4), vec2(-2998.8, -206.8), vec2(-2998.1, -202.2), vec2(-2997.5, -197.6), vec2(-2996.9, -193.0), vec2(-2996.2, -188.5), vec2(-2995.6, -184.1), vec2(-2994.9, -179.6), vec2(-2994.2, -175.3), vec2(-2993.5, -170.9), vec2(-2992.8, -166.6), vec2(-2992.1, -162.4), vec2(-2991.4, -158.2), vec2(-2990.7, -154.1), vec2(-2990.0, -150.0), vec2(-2989.3, -146.1), vec2(-2988.6, -142.2), vec2(-2987.9, -138.3), vec2(-2987.1, -134.3), vec2(-2986.4, -130.4), vec2(-2985.7, -126.5), vec2(-2985.0, -122.5), vec2(-2984.2, -118.6), vec2(-2983.5, -114.7), vec2(-2982.7, -110.7), vec2(-2981.9, -106.8), vec2(-2981.2, -102.8), vec2(-2980.4, -98.9), vec2(-2979.6, -94.9), vec2(-2978.8, -91.0), vec2(-2978.0, -87.0), vec2(-2977.2, -83.1), vec2(-2976.4, -79.1), vec2(-2975.6, -75.2), vec2(-2974.8, -71.2), vec2(-2974.0, -67.2), vec2(-2973.1, -63.3), vec2(-2972.3, -59.3), vec2(-2971.5, -55.4), vec2(-2970.6, -51.4), vec2(-2969.7, -47.5), vec2(-2968.9, -43.5), vec2(-2968.0, -39.5), vec2(-2967.1, -35.6), vec2(-2966.2, -31.6), vec2(-2965.3, -27.7), vec2(-2964.4, -23.8), vec2(-2963.5, -19.8), vec2(-2962.5, -15.9), vec2(-2961.6, -11.9), vec2(-2960.6, -8.0), vec2(-2959.7, -4.1), vec2(-2958.7, -0.1), vec2(-2957.7, 3.8), vec2(-2956.7, 7.7), vec2(-2955.8, 11.6), vec2(-2954.7, 15.5), vec2(-2953.7, 19.4), vec2(-2952.7, 23.4), vec2(-2951.7, 27.2), vec2(-2950.6, 31.1), vec2(-2949.6, 35.0), vec2(-2948.5, 38.9), vec2(-2947.4, 42.8), vec2(-2946.4, 46.6), vec2(-2945.3, 50.5), vec2(-2944.2, 54.4), vec2(-2943.0, 58.2), vec2(-2941.9, 62.0), vec2(-2940.8, 65.9), vec2(-2939.6, 69.7), vec2(-2938.5, 73.5), vec2(-2937.3, 77.3), vec2(-2936.1, 81.1), vec2(-2934.9, 84.9), vec2(-2933.7, 88.7), vec2(-2932.5, 92.5), vec2(-2931.2, 96.2), vec2(-2930.0, 100.0), vec2(-2928.7, 103.9), vec2(-2927.4, 107.7), vec2(-2926.1, 111.6), vec2(-2924.8, 115.5), vec2(-2923.4, 119.3), vec2(-2922.1, 123.2), vec2(-2920.7, 127.1), vec2(-2919.3, 130.9), vec2(-2918.0, 134.8), vec2(-2916.6, 138.7), vec2(-2915.2, 142.5), vec2(-2913.8, 146.4), vec2(-2912.3, 150.2), vec2(-2910.9, 154.1), vec2(-2909.5, 157.9), vec2(-2908.0, 161.8), vec2(-2906.6, 165.6), vec2(-2905.1, 169.4), vec2(-2903.6, 173.3), vec2(-2902.1, 177.1), vec2(-2900.6, 180.9), vec2(-2899.1, 184.7), vec2(-2897.6, 188.5), vec2(-2896.0, 192.3), vec2(-2894.5, 196.1), vec2(-2892.9, 199.9), vec2(-2891.4, 203.7), vec2(-2889.8, 207.5), vec2(-2888.2, 211.2), vec2(-2886.6, 215.0), vec2(-2885.0, 218.8), vec2(-2883.4, 222.5), vec2(-2881.8, 226.2), vec2(-2880.1, 230.0), vec2(-2878.5, 233.7), vec2(-2876.8, 237.4), vec2(-2875.1, 241.1), vec2(-2873.5, 244.8), vec2(-2871.8, 248.5), vec2(-2870.1, 252.2), vec2(-2868.4, 255.8), vec2(-2866.6, 259.5), vec2(-2864.9, 263.1), vec2(-2863.2, 266.7), vec2(-2861.4, 270.4), vec2(-2859.6, 274.0), vec2(-2857.9, 277.6), vec2(-2856.1, 281.2), vec2(-2854.3, 284.7), vec2(-2852.5, 288.3), vec2(-2850.7, 291.9), vec2(-2848.8, 295.4), vec2(-2847.0, 298.9), vec2(-2845.2, 302.4), vec2(-2843.3, 305.9), vec2(-2841.4, 309.4), vec2(-2839.5, 312.9), vec2(-2837.7, 316.3), vec2(-2835.8, 319.8), vec2(-2833.9, 323.2), vec2(-2831.9, 326.6), vec2(-2830.0, 330.0), vec2(-2828.0, 333.6), vec2(-2825.9, 337.1), vec2(-2823.8, 340.6), vec2(-2821.8, 344.2), vec2(-2819.7, 347.7), vec2(-2817.5, 351.2), vec2(-2815.4, 354.7), vec2(-2813.3, 358.2), vec2(-2811.1, 361.7), vec2(-2809.0, 365.1), vec2(-2806.8, 368.6), vec2(-2804.6, 372.0), vec2(-2802.4, 375.5), vec2(-2800.2, 378.9), vec2(-2798.0, 382.3), vec2(-2795.7, 385.7), vec2(-2793.5, 389.1), vec2(-2791.2, 392.5), vec2(-2788.9, 395.9), vec2(-2786.6, 399.3), vec2(-2784.3, 402.6), vec2(-2782.0, 406.0), vec2(-2779.7, 409.3), vec2(-2777.4, 412.6), vec2(-2775.0, 415.9), vec2(-2772.7, 419.2), vec2(-2770.3, 422.5), vec2(-2768.0, 425.7), vec2(-2765.6, 429.0), vec2(-2763.2, 432.2), vec2(-2760.8, 435.5), vec2(-2758.4, 438.7), vec2(-2755.9, 441.9), vec2(-2753.5, 445.1), vec2(-2751.1, 448.3), vec2(-2748.6, 451.4), vec2(-2746.1, 454.6), vec2(-2743.7, 457.7), vec2(-2741.2, 460.8), vec2(-2738.7, 463.9), vec2(-2736.2, 467.0), vec2(-2733.7, 470.1), vec2(-2731.2, 473.1), vec2(-2728.7, 476.2), vec2(-2726.1, 479.2), vec2(-2723.6, 482.2), vec2(-2721.1, 485.2), vec2(-2718.5, 488.2), vec2(-2716.0, 491.2), vec2(-2713.4, 494.2), vec2(-2710.8, 497.1), vec2(-2708.2, 500.0), vec2(-2705.7, 502.9), vec2(-2703.1, 505.8), vec2(-2700.5, 508.7), vec2(-2697.9, 511.5), vec2(-2695.2, 514.4), vec2(-2692.6, 517.2), vec2(-2690.0, 520.0), vec2(-2687.2, 523.0), vec2(-2684.3, 526.0), vec2(-2681.5, 528.9), vec2(-2678.6, 531.8), vec2(-2675.7, 534.8), vec2(-2672.9, 537.6), vec2(-2670.0, 540.5), vec2(-2667.0, 543.4), vec2(-2664.1, 546.2), vec2(-2661.2, 549.0), vec2(-2658.2, 551.8), vec2(-2655.3, 554.6), vec2(-2652.3, 557.4), vec2(-2649.3, 560.1), vec2(-2646.3, 562.9), vec2(-2643.3, 565.6), vec2(-2640.3, 568.3), vec2(-2637.3, 571.0), vec2(-2634.3, 573.6), vec2(-2631.2, 576.3), vec2(-2628.2, 578.9), vec2(-2625.1, 581.5), vec2(-2622.0, 584.1), vec2(-2619.0, 586.7), vec2(-2615.9, 589.3), vec2(-2612.8, 591.8), vec2(-2609.7, 594.4), vec2(-2606.6, 596.9), vec2(-2603.4, 599.4), vec2(-2600.3, 601.9), vec2(-2597.2, 604.4), vec2(-2594.0, 606.8), vec2(-2590.9, 609.3), vec2(-2587.7, 611.7), vec2(-2584.6, 614.1), vec2(-2581.4, 616.5), vec2(-2578.2, 618.9), vec2(-2575.0, 621.3), vec2(-2571.8, 623.7), vec2(-2568.6, 626.1), vec2(-2565.4, 628.4), vec2(-2562.2, 630.7), vec2(-2559.0, 633.1), vec2(-2555.8, 635.4), vec2(-2552.5, 637.7), vec2(-2549.3, 639.9), vec2(-2546.1, 642.2), vec2(-2542.8, 644.5), vec2(-2539.6, 646.7), vec2(-2536.3, 649.0), vec2(-2533.1, 651.2), vec2(-2529.8, 653.4), vec2(-2526.5, 655.6), vec2(-2523.3, 657.8), vec2(-2520.0, 660.0), vec2(-2516.6, 662.3), vec2(-2513.2, 664.5), vec2(-2509.7, 666.7), vec2(-2506.3, 668.9), vec2(-2502.8, 671.1), vec2(-2499.3, 673.3), vec2(-2495.8, 675.4), vec2(-2492.3, 677.6), vec2(-2488.8, 679.7), vec2(-2485.2, 681.8), vec2(-2481.7, 683.9), vec2(-2478.1, 685.9), vec2(-2474.6, 688.0), vec2(-2471.0, 690.0), vec2(-2467.4, 692.1), vec2(-2463.8, 694.1), vec2(-2460.2, 696.1), vec2(-2456.6, 698.1), vec2(-2452.9, 700.0), vec2(-2449.3, 702.0), vec2(-2445.7, 703.9), vec2(-2442.0, 705.9), vec2(-2438.4, 707.8), vec2(-2434.8, 709.7), vec2(-2431.1, 711.6), vec2(-2427.5, 713.4), vec2(-2423.8, 715.3), vec2(-2420.1, 717.2), vec2(-2416.5, 719.0), vec2(-2412.8, 720.8), vec2(-2409.2, 722.6), vec2(-2405.5, 724.4), vec2(-2401.9, 726.2), vec2(-2398.2, 728.0), vec2(-2394.5, 729.8), vec2(-2390.9, 731.5), vec2(-2387.3, 733.3), vec2(-2383.6, 735.0), vec2(-2380.0, 736.8), vec2(-2376.4, 738.5), vec2(-2372.7, 740.2), vec2(-2369.1, 741.9), vec2(-2365.5, 743.6), vec2(-2361.9, 745.3), vec2(-2358.3, 746.9), vec2(-2354.7, 748.6), vec2(-2351.2, 750.2), vec2(-2347.6, 751.9), vec2(-2344.1, 753.5), vec2(-2340.5, 755.2), vec2(-2337.0, 756.8), vec2(-2333.5, 758.4), vec2(-2330.0, 760.0), vec2(-2326.1, 761.8), vec2(-2322.2, 763.5), vec2(-2318.1, 765.3), vec2(-2314.1, 767.1), vec2(-2310.0, 768.9), vec2(-2305.8, 770.7), vec2(-2301.6, 772.4), vec2(-2297.3, 774.2), vec2(-2293.1, 776.0), vec2(-2288.8, 777.7), vec2(-2284.4, 779.5), vec2(-2280.1, 781.2), vec2(-2275.7, 783.0), vec2(-2271.3, 784.7), vec2(-2267.0, 786.4), vec2(-2262.6, 788.1), vec2(-2258.2, 789.8), vec2(-2253.9, 791.5), vec2(-2249.5, 793.2), vec2(-2245.2, 794.8), vec2(-2240.9, 796.5), vec2(-2236.6, 798.1), vec2(-2232.3, 799.7), vec2(-2228.1, 801.2), vec2(-2224.0, 802.8), vec2(-2219.8, 804.3), vec2(-2215.8, 805.8), vec2(-2211.7, 807.3), vec2(-2207.8, 808.8), vec2(-2203.9, 810.2), vec2(-2200.0, 811.6), vec2(-2196.3, 813.0), vec2(-2192.6, 814.3), vec2(-2189.0, 815.6), vec2(-2185.5, 816.9), vec2(-2182.1, 818.1), vec2(-2178.8, 819.3), vec2(-2175.6, 820.5), vec2(-2172.5, 821.6), vec2(-2169.5, 822.7), vec2(-2166.6, 823.8), vec2(-2163.8, 824.8), vec2(-2161.2, 825.8), vec2(-2158.7, 826.7), vec2(-2156.3, 827.6), vec2(-2154.0, 828.4), vec2(-2152.0, 829.2), vec2(-2150.0, 830.0), vec2(-3060.3, -1950.7), vec2(-3060.0, -1946.3), vec2(-3059.7, -1940.7), vec2(-3059.3, -1935.7), vec2(-3059.0, -1930.7), vec2(-3058.7, -1925.7), vec2(-3058.3, -1920.8), vec2(-3058.0, -1915.8), vec2(-3057.7, -1910.8), vec2(-3057.3, -1905.8), vec2(-3057.0, -1900.8), vec2(-3056.7, -1895.8), vec2(-3056.3, -1890.8), vec2(-3056.0, -1885.8), vec2(-3055.7, -1880.8), vec2(-3055.3, -1875.9), vec2(-3055.0, -1870.9), vec2(-3054.7, -1865.9), vec2(-3054.3, -1860.9), vec2(-3054.0, -1855.9), vec2(-3053.7, -1850.9), vec2(-3053.4, -1845.9), vec2(-3053.0, -1840.9), vec2(-3052.7, -1835.9), vec2(-3052.4, -1831.0), vec2(-3052.0, -1826.0), vec2(-3051.7, -1821.0), vec2(-3051.4, -1816.0), vec2(-3051.0, -1811.0), vec2(-3050.7, -1806.0), vec2(-3050.4, -1801.0), vec2(-3050.0, -1796.0), vec2(-3049.7, -1791.0), vec2(-3049.4, -1786.1), vec2(-3049.0, -1781.1), vec2(-3048.7, -1776.1), vec2(-3048.4, -1771.1), vec2(-3048.0, -1766.1), vec2(-3047.7, -1761.1), vec2(-3047.4, -1756.1), vec2(-3047.0, -1751.1), vec2(-3046.7, -1746.1), vec2(-3046.4, -1741.2), vec2(-3046.0, -1736.2), vec2(-3045.7, -1731.2), vec2(-3045.4, -1726.2), vec2(-3045.0, -1721.2), vec2(-3044.7, -1716.2), vec2(-3044.4, -1711.2), vec2(-3044.0, -1706.2), vec2(-3043.7, -1701.2), vec2(-3043.4, -1696.3), vec2(-3043.0, -1691.3), vec2(-3042.7, -1685.7), vec2(-3042.4, -1681.3), vec2(-3039.7, -1641.4), vec2(-3039.4, -1637.0), vec2(-3039.0, -1631.4), vec2(-3038.7, -1626.4), vec2(-3038.4, -1621.4), vec2(-3038.1, -1616.4), vec2(-3037.7, -1611.4), vec2(-3037.4, -1606.5), vec2(-3037.1, -1601.5), vec2(-3036.7, -1596.5), vec2(-3036.4, -1591.5), vec2(-3036.1, -1586.5), vec2(-3035.7, -1581.5), vec2(-3035.4, -1576.5), vec2(-3035.1, -1571.5), vec2(-3034.7, -1566.5), vec2(-3034.4, -1561.6), vec2(-3034.1, -1556.6), vec2(-3033.7, -1551.6), vec2(-3033.4, -1546.6), vec2(-3033.1, -1541.6), vec2(-3032.7, -1536.6), vec2(-3032.4, -1531.6), vec2(-3032.1, -1526.6), vec2(-3031.7, -1521.6), vec2(-3031.4, -1516.7), vec2(-3031.1, -1511.7), vec2(-3030.7, -1506.7), vec2(-3030.4, -1501.7), vec2(-3030.1, -1496.7), vec2(-3029.7, -1491.7), vec2(-3029.4, -1486.7), vec2(-3029.1, -1481.7), vec2(-3028.7, -1476.7), vec2(-3028.4, -1471.8), vec2(-3028.1, -1466.8), vec2(-3027.7, -1461.8), vec2(-3027.4, -1456.8), vec2(-3027.1, -1451.8), vec2(-3026.7, -1446.8), vec2(-3026.4, -1441.8), vec2(-3026.1, -1436.8), vec2(-3025.7, -1431.8), vec2(-3025.4, -1426.9), vec2(-3025.1, -1421.9), vec2(-3024.7, -1416.9), vec2(-3024.4, -1411.9), vec2(-3024.1, -1406.9), vec2(-3023.7, -1401.9), vec2(-3023.4, -1396.9), vec2(-3023.1, -1391.9), vec2(-3022.8, -1386.9), vec2(-3022.4, -1382.0), vec2(-3022.0, -1376.3), vec2(-3021.8, -1372.0), vec2(-3061.7, -1369.3), vec2(-3062.0, -1373.7), vec2(-3062.3, -1379.3), vec2(-3062.7, -1384.3), vec2(-3063.0, -1389.3), vec2(-3063.3, -1394.3), vec2(-3063.7, -1399.2), vec2(-3064.0, -1404.2), vec2(-3064.3, -1409.2), vec2(-3064.7, -1414.2), vec2(-3065.0, -1419.2), vec2(-3065.3, -1424.2), vec2(-3065.7, -1429.2), vec2(-3066.0, -1434.2), vec2(-3066.3, -1439.2), vec2(-3066.7, -1444.1), vec2(-3067.0, -1449.1), vec2(-3067.3, -1454.1), vec2(-3067.7, -1459.1), vec2(-3068.0, -1464.1), vec2(-3068.3, -1469.1), vec2(-3068.6, -1474.1), vec2(-3069.0, -1479.1), vec2(-3069.3, -1484.1), vec2(-3069.6, -1489.0), vec2(-3070.0, -1494.0), vec2(-3070.3, -1499.0), vec2(-3070.6, -1504.0), vec2(-3071.0, -1509.0), vec2(-3071.3, -1514.0), vec2(-3071.6, -1519.0), vec2(-3072.0, -1524.0), vec2(-3072.3, -1529.0), vec2(-3072.6, -1533.9), vec2(-3073.0, -1538.9), vec2(-3073.3, -1543.9), vec2(-3073.6, -1548.9), vec2(-3074.0, -1553.9), vec2(-3074.3, -1558.9), vec2(-3074.6, -1563.9), vec2(-3075.0, -1568.9), vec2(-3075.3, -1573.9), vec2(-3075.6, -1578.8), vec2(-3076.0, -1583.8), vec2(-3076.3, -1588.8), vec2(-3076.6, -1593.8), vec2(-3077.0, -1598.8), vec2(-3077.3, -1603.8), vec2(-3077.6, -1608.8), vec2(-3078.0, -1613.8), vec2(-3078.3, -1618.8), vec2(-3078.6, -1623.7), vec2(-3079.0, -1628.7), vec2(-3079.3, -1634.3), vec2(-3079.6, -1638.7), vec2(-3082.3, -1678.6), vec2(-3082.6, -1683.0), vec2(-3083.0, -1688.6), vec2(-3083.3, -1693.6), vec2(-3083.6, -1698.6), vec2(-3083.9, -1703.6), vec2(-3084.3, -1708.6), vec2(-3084.6, -1713.5), vec2(-3084.9, -1718.5), vec2(-3085.3, -1723.5), vec2(-3085.6, -1728.5), vec2(-3085.9, -1733.5), vec2(-3086.3, -1738.5), vec2(-3086.6, -1743.5), vec2(-3086.9, -1748.5), vec2(-3087.3, -1753.5), vec2(-3087.6, -1758.4), vec2(-3087.9, -1763.4), vec2(-3088.3, -1768.4), vec2(-3088.6, -1773.4), vec2(-3088.9, -1778.4), vec2(-3089.3, -1783.4), vec2(-3089.6, -1788.4), vec2(-3089.9, -1793.4), vec2(-3090.3, -1798.4), vec2(-3090.6, -1803.3), vec2(-3090.9, -1808.3), vec2(-3091.3, -1813.3), vec2(-3091.6, -1818.3), vec2(-3091.9, -1823.3), vec2(-3092.3, -1828.3), vec2(-3092.6, -1833.3), vec2(-3092.9, -1838.3), vec2(-3093.3, -1843.3), vec2(-3093.6, -1848.2), vec2(-3093.9, -1853.2), vec2(-3094.3, -1858.2), vec2(-3094.6, -1863.2), vec2(-3094.9, -1868.2), vec2(-3095.3, -1873.2), vec2(-3095.6, -1878.2), vec2(-3095.9, -1883.2), vec2(-3096.3, -1888.2), vec2(-3096.6, -1893.1), vec2(-3096.9, -1898.1), vec2(-3097.3, -1903.1), vec2(-3097.6, -1908.1), vec2(-3097.9, -1913.1), vec2(-3098.3, -1918.1), vec2(-3098.6, -1923.1), vec2(-3098.9, -1928.1), vec2(-3099.2, -1933.1), vec2(-3099.6, -1938.0), vec2(-3100.0, -1943.7), vec2(-3100.2, -1948.0));
const vec4 C_WAY_BOX[12] = vec4[12](vec4(-3700.0, -7000.0, -2999.5, 9000.0), vec4(-3010.0, 700.0, -1450.0, 890.6), vec4(-3018.0, -1900.0, -2930.0, -1300.0), vec4(-3460.0, 2800.0, -3277.0, 3400.0), vec4(-7500.0, -6500.0, -2500.0, -1100.0), vec4(-7500.0, 2600.0, -1800.0, 7800.0), vec4(4149.1, -1500.0, 5000.0, 6000.0), vec4(-3020.0, -400.0, -2150.0, 830.0), vec4(-3060.3, -1950.7, -3042.4, -1681.3), vec4(-3039.7, -1641.4, -3021.8, -1372.0), vec4(-3079.6, -1638.7, -3061.7, -1369.3), vec4(-3100.2, -1948.0, -3082.3, -1678.6));
// The nearest road to c: its kind, half width, signed distance across, distance along.
// Returns false if none is within its half width + pad.
bool cWay(vec2 c, float pad, out int kind, out float wHalf, out float d, out float s) {
    float best = 1e9;
    kind = -1;
    for (int w = 0; w < C_WAYS; ++w) {
        float hw = C_WAY_HALF[w] + pad, acc = 0.0;
        // (the whole road's box first: most are far away)
        if (any(lessThan(c, C_WAY_BOX[w].xy - hw)) || any(greaterThan(c, C_WAY_BOX[w].zw + hw))) continue;
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

// What a city's shore is here, in stretches of a few hundred metres (Earth-fixed): 0 a
// seawall, 1 riprap (rock armour), 2 a mangrove fringe, 3 a small sand beach. The land side
// (below) and the water side (seawall: deep, no foam; the others shelve and lap) agree.
int cityEdge(vec2 uvm) {
    float e = fbmN(uvm * 0.004 + 211.0, 2) + 0.15 * fbmN(uvm * 0.02 + 5.0, 2) - 0.075;
    return e < 0.47 ? 0 : e < 0.58 ? 1 : e < 0.7 ? 2 : 3;
}

// The street lamps' light on the ground (globe/city.rs streets puts the lamps: every 30 m
// from 6 m past the crossing, alternate sides, the head 2.4 m in from the kerb, 8.7 m up):
// relative irradiance at plan point c from the lamps of the street through line `line` at
// across-offset `d` (signed, from the line) and along-distance `s` from the block's start line,
// whose crossings take h0 at the start and h1 at the end of a block `len` long.
float cLampPool(float d, float s, float len, float h0, float h1, float kerb) {
    float n = floor((s - h0 - 6.0 + 15.0) / 30.0);
    float acc = 0.0;
    for (float k = n - 1.0; k <= n + 1.0; k += 1.0) {
        float sn = h0 + 6.0 + 30.0 * k;
        if (k < 0.0 || sn >= len - h1 - 4.0) continue;
        float side = mod(k, 2.0) < 0.5 ? -1.0 : 1.0;
        vec2 q = vec2(d - side * (kerb - 2.4), s - sn);
        float r2 = dot(q, q) + 8.7 * 8.7;
        // cosine-weighted inverse square, ~1 under the head; the fixture's cut-off keeps it to
        // a pool of ~16 m
        acc += 8.7 / (r2 * sqrt(r2)) * 75.0 * smoothstep(16.0, 5.0, length(q));
    }
    return acc;
}
vec3 cityLamp = vec3(0.0); // set by cityGround: the lamps' light here (main.rgen adds it at dusk)

float cityGround(vec3 p, float footprint, float built, float waterF, float wetF, float sd, out vec3 alb) {
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
    int zone = dz > 0.55 ? 3 : dz > 0.2 ? 2 : dz > 0.08 ? 1 : 0;
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
        // (the back-yard pools are placed with their houses: globe/city.rs block)
        // hedges along the lot lines
        float hedge = max(1.0 - smoothstep(0.5, 0.5 + fw, min(u, lw - u)), 1.0 - smoothstep(0.6, 0.6 + fw, 0.5 * H - dv));
        lot = mix(lot, vec3(0.035, 0.06, 0.02) * mix(0.8, 1.2, n2), hedge * step(6.0, dv));
    } else if ((zone == 1 && cBu(i, j, 8u) < 0.08) || (zone == 2 && cBu(i, j, 3u) < 0.04)) {
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
        // plazas: lawn panels between paved paths 4 m wide every 18 x 15 m (globe/city.rs
        // lot_trees puts a shade tree on most panels), planting beds along the edges,
        // varied paving under the paths
        vec2 pm = mod(b, vec2(18.0, 15.0));
        float panel = cBox(pm, vec2(4.0, 4.0), vec2(18.0, 15.0), fw);
        vec3 bed = vec3(0.04, 0.065, 0.022) * mix(0.7, 1.3, fbmN(c * 0.8 + 3.0, 2));
        vec3 grass = mix(lawnC, bed, smoothstep(0.55, 0.75, fbmN(c * 0.15 + 21.0, 2)));
        lot = mix(plaza, grass, panel);
        float edge = 1.0 - cBox(b, vec2(2.0), vec2(W, H) - 2.0, fw);
        lot = mix(lot, bed, edge * 0.9);
        if (zone == 2) lot = mix(lot, pave, 0.25 * (1.0 - panel));
    }
    // the lots' last 20 m or so to the water are a waterfront lawn, never a plaza or car park
    // cut off by the shore
    lot = mix(lot, lawnC * mix(0.85, 1.1, n2), 1.0 - smoothstep(16.0 + 6.0 * n1, 24.0 + 6.0 * n1, sd));
    // along every shore of the city's lots: a paved promenade at the seawall and a lawn
    // strip, 10 m in all, before any lot (globe/city.rs keeps buildings and cars back)
    {
        vec3 prom = vec3(0.20, 0.19, 0.175) * mix(0.85, 1.08, n1);
        float band = 1.0 - smoothstep(9.5, 10.0 + fw, sd);
        lot = mix(lot, mix(prom, lawnC, smoothstep(1.2, 1.2 + fw, sd)), band * step(0.0, sd));
    }
    // the canopy of the city's trees as the ground sees it from afar (where a crown is a pixel
    // or less): dark green clumps of ~20-60 m over the lots, as thick as the plan grows them
    {
        float cm = smoothstep(0.42, 0.62, fbmN(c * 0.03 + 91.0, 3) * 0.7 + fbmN(c * 0.11 + 17.0, 2) * 0.3);
        float far = smoothstep(0.3, 1.2, fw);
        vec3 crown = vec3(0.022, 0.040, 0.016) * mix(0.75, 1.25, fbmN(c * 0.4 + 5.0, 2));
        lot = mix(lot, crown, cm * far * (zone == 0 ? 0.8 : 0.55));
    }
    vec3 planned = mix(lot, mix(walk, asphalt, roadW), streetW);
    {
        // the lamps of the nearest north-south and east-west streets
        float ky = cHalf(ax) - cSide(ax) + 0.6, kx = cHalf(ay) - cSide(ay) + 0.6;
        float lampNS = cLampPool(f.x - (li == i ? 0.0 : bw), f.y, bh, cHalf(cArtY(j)), cHalf(cArtY(j + 1)), ky);
        float lampEW = cLampPool(f.y - (lj == j ? 0.0 : bh), f.x, bw, cHalf(cArtX(i)), cHalf(cArtX(i + 1)), kx);
        cityLamp = vec3(1.0, 0.78, 0.5) * (lampNS + lampEW);
    }
    // where the land cover is not built up (parks, grounds, open land in the city): kept lawn
    // under scattered shade, with paths, never the bare native soil showing through
    float shade = smoothstep(0.55, 0.75, fbmN(c * 0.06 + 31.0, 3) * 0.7 + fbmN(c * 0.4 + 2.0, 2) * 0.3);
    vec3 park = mix(lawnC * 1.05, lawnC * vec3(0.6, 0.72, 0.55), shade);
    vec2 pp = abs(fract(c / vec2(37.0, 29.0)) - 0.5);
    park = mix(park, vec3(0.24, 0.225, 0.2), (1.0 - smoothstep(0.012, 0.012 + fw / 37.0, min(pp.x, pp.y))) * shown * 0.6);
    // wetland in the city: mangrove and dense shrub, a deep, uneven green
    vec3 mangrove = vec3(0.028, 0.05, 0.02) * mix(0.7, 1.3, fbmN(c * 0.25 + 77.0, 3));
    park = mix(park, mangrove, smoothstep(0.15, 0.45, wetF));
    // (the streets run on through parks and wetland: no street, and no bridge, ends in a lawn)
    alb = mix(park, planned, max(smoothstep(0.08, 0.3, built) * (1.0 - smoothstep(0.2, 0.5, wetF)), streetW));
    // the waterfront drive (globe/city.rs WATERFRONT): along the shore of the city's blocks
    // other than houses, following the shoreline field: a paved promenade at the seawall, a
    // planted verge, two lanes with a double yellow line and white edge lines, a sidewalk; the
    // streets of the grid run into it
    // (the drive's lanes lie well inland, D m out from the water, on the low-passed distance:
    // a level set that far in is smooth, and the promenade and a planted verge take up the
    // shore's wiggles; the grid's streets, and with them the bridges' approaches, run through
    // the verge to the drive)
    const float D = 16.0;
    if (zone >= 2 && sd < D + 16.0 && sd > 0.0) {
        vec3 prom = vec3(0.20, 0.19, 0.175) * mix(0.85, 1.08, n1) * mix(1.0, 0.88, (1.0 - smoothstep(0.44, 0.5, max(sj.x, sj.y))) * shown);
        vec3 drive = vec3(0.075, 0.075, 0.078) * mix(0.82, 1.18, n1) * mix(0.9, 1.1, n3);
        float yel = (1.0 - smoothstep(0.07, 0.07 + fw, abs(sd - D - 6.0))) + (1.0 - smoothstep(0.07, 0.07 + fw, abs(sd - D - 6.4)));
        float wht = (1.0 - smoothstep(0.07, 0.07 + fw, abs(sd - D - 0.5))) + (1.0 - smoothstep(0.07, 0.07 + fw, abs(sd - D - 11.9)));
        drive = mix(drive, vec3(0.55, 0.42, 0.10), clamp(yel, 0.0, 1.0) * shown);
        drive = mix(drive, vec3(0.6, 0.6, 0.58), clamp(wht, 0.0, 1.0) * shown);
        vec3 wf = prom;
        wf = mix(wf, lawnC, smoothstep(1.5, 1.5 + fw, sd));
        wf = mix(wf, vec3(0.30, 0.29, 0.27), smoothstep(D - 0.3, D - 0.3 + fw, sd) * shown);
        wf = mix(wf, drive, smoothstep(D, D + fw, sd));
        wf = mix(wf, walk, smoothstep(D + 12.4, D + 12.4 + fw, sd));
        float cover = 1.0 - smoothstep(D + 15.5, D + 15.5 + fw, sd);
        cover *= 1.0 - roadW * (1.0 - smoothstep(D - 1.0, D, sd)); // the streets run through
        alb = mix(alb, wf, cover);
    }
    // inside the interchanges: landscaped ground (globe/city.rs CLEAR)
    {
        const vec3 CLEAR[1] = vec3[1](vec3(-3061.0, -1660.0, 0.0));
        for (int q = 0; q < 1; ++q) {
            float dq = length(cw - CLEAR[q].xy);
            float inC = 1.0 - smoothstep(CLEAR[q].z - 6.0, CLEAR[q].z, dq);
            inC *= 1.0 - streetW; // (the streets run on under the interchange)
            vec3 land = mix(lawnC, vec3(0.04, 0.065, 0.022), smoothstep(0.5, 0.7, fbmN(c * 0.05 + 77.0, 3)));
            alb = mix(alb, land, inC);
        }
    }
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
            // where a grid street crosses: no lane paint in the junction, a zebra crossing
            // across the avenue just outside the street's kerbs, a stop bar before it
            float junction = max(inRX, inRY);
            lane *= 1.0 - junction;
            float zebraBand = max(inSX * (1.0 - inRX), inSY * (1.0 - inRY));
            float zebra = zebraBand * step(fract(wd / 1.2), 0.5) * step(ad, wh - 3.6);
            road = mix(asph, vec3(0.6, 0.6, 0.58), clamp(max(lane, zebra), 0.0, 1.0) * shown);
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
    // the city's waterfront drive follows a low-passed shore: the field (before its noise)
    // averaged over a ring of 30 m round the point
    float sdSmooth = sd;
    if (cityNear(gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT)) {
        // (two rings, 30 m and 60 m: the drive's lines are level sets of this, smooth curves)
        float acc = sd;
        for (int ring = 1; ring <= 2; ++ring) {
            float ru = 30.0 * float(ring) / max(cellM * float(FIELD), 1.0);
            for (int k = 0; k < 12; ++k) {
                float a = float(k) * 0.5235988 + float(ring) * 0.26;
                acc += shoreDist(V, clamp(tuv + vec2(cos(a), sin(a)) * ru, 0.0, 1.0));
            }
        }
        sdSmooth = acc / 25.0;
    }
    // the mapped shoreline is 10 m pixels made smooth: break it with Earth-fixed noise
    bool inCity = cityNear(gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT);
    if (inCity) {
        // a city's shore is built, a smooth line: the field averaged over an 8 m ring (the 10 m
        // land cover pixels showed through as a sawtooth), no noise
        float ru = 8.0 / max(cellM * float(FIELD), 1.0), acc = sd;
        for (int k = 0; k < 8; ++k) {
            float a = float(k) * 0.7853982;
            acc += shoreDist(V, clamp(tuv + vec2(cos(a), sin(a)) * ru, 0.0, 1.0));
        }
        sd = acc / 9.0;
    } else
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
    bool seawall = water && cityNear(p) && (B.y < 0.6 || B.x > 0.08) && cityEdge(uvm) <= 1;
    if (seawall) depth = max(depth, 3.5);
    // (and everywhere in a city but its beaches: no bright shallow halo round every island)
    // (a beach's shallows reach 12 m out, no more: the bay and the canals beyond are deep and
    // murky, main.rgen)
    if (water && cityNear(p) && (cityEdge(uvm) != 3 || sd < -12.0)) depth = max(depth, 6.0);
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
        float urban = cityGround(p, footprint, min(B.x + 0.5 * (A.x + A.z), 1.0), B.w, C.x, sdSmooth, cityAlb);
        // A built-up shore in a city is a seawall, not a beach: the city's ground runs to a
        // concrete cap with a dark, wet face at the edge (main.rgen: no lapping foam there, the
        // water is deep at once). Natural shores (beach sand, mangroves) keep their strip.
        float wall = cityNear(p) ? max(1.0 - smoothstep(0.55, 0.8, B.y), smoothstep(0.06, 0.18, B.x)) : 0.0;
        int edge = cityNear(p) ? cityEdge(uvm) : 0;
        alb = groundMaterial(A, B, C.xy, uvm, elev, up, n, footprint, 0.0, shore * (1.0 - wall), biome, seaSide, ao);
        urban *= 1.0 - shore * (1.0 - wall);
        if (urban > 0.0) {
            alb = mix(alb, cityAlb, urban);
            n = normalize(mix(n, up, urban));
            ao = mix(ao, 1.0, urban);
        }
        if (wall > 0.0 && edge == 1) {
            // riprap: a band of armour stone down to the water, boulders of a metre or so,
            // each its own shade, dark gaps between them, wet and darker at the waterline
            vec2 rq = uvm * 0.9;
            vec2 rc = floor(rq);
            float bh = fract(sin(dot(rc, vec2(12.9898, 78.233))) * 43758.5453);
            vec2 rf = fract(rq) - 0.5;
            float stone = smoothstep(0.5, 0.32, max(abs(rf.x), abs(rf.y)) + 0.12 * vnoise(uvm * 3.1));
            vec3 rock = mix(vec3(0.20, 0.19, 0.17), vec3(0.36, 0.34, 0.30), bh) * mix(0.35, 1.0, mix(0.6, stone, 1.0 - smoothstep(0.3, 1.0, footprint)));
            rock *= mix(0.55, 1.0, smoothstep(0.0, 1.2, sd));
            float band = smoothstep(3.2 + footprint, 2.6, sd) * wall;
            alb = mix(alb, rock, band);
            ao = mix(ao, 0.75, band);
        } else if (edge == 2) {
            // a mangrove fringe: dense, dark, uneven green to the water
            float band = smoothstep(7.0 + 3.0 * fbmN(uvm * 0.15, 2), 4.0, sd);
            vec3 mg = vec3(0.022, 0.042, 0.016) * mix(0.6, 1.4, fbmN(uvm * 0.4 + 9.0, 3));
            alb = mix(alb, mg, band);
            ao = mix(ao, 0.6, band);
        } else if (edge == 3) {
            // a small beach: pale sand from the drive down to the water, wet and darker at the
            // waterline
            float band = smoothstep(9.0, 6.0, sd);
            vec3 sand = vec3(0.46, 0.41, 0.32) * mix(0.9, 1.06, fbmN(uvm * 0.3 + 3.0, 2));
            sand *= mix(0.62, 1.0, smoothstep(0.0, 1.5, sd));
            alb = mix(alb, sand, band);
        } else if (wall > 0.0) {
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
    pl.wtan = water ? vec3(0.0) : cityLamp; // (land: the street lamps' light, main.rgen)
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
