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
// clay 0.2, tundra 0.18, lichen 0.2, boreal floor 0.1, glacier 0.55, mud 0.11, salt 0.45,
// ploughed 0.15, lava sand 0.07, steppe soil 0.17, dry grass 0.24; very dark scans capped)
const int LAYERS = 26;
const float GAIN[LAYERS] = float[LAYERS](0.7, 2.4, 0.68, 0.5, 1.0, 1.0, 0.75, 1.0, 1.0, 1.0, 0.8, 1.0,
                                         1.02, 1.99, 4.0, 2.27, 6.0, 1.41, 0.58, 4.09, 3.69, 2.65, 2.62, 3.97, 3.44, 4.02);

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
    float ptint = 1.0, pw = 0.0; // crop parcels: brightness of this parcel, its weight
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
    w[7] = (bare - rockBare) * (1.0 - scree) + built * 0.55 * (1.0 - 0.5 * dryLand);  // gravel
    w[2] += built * 0.3 * (1.0 - dryLand) * smoothstep(0.55, 0.35, nb);
    w[8] = moss;                                                                  // lichen rock
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
    // cropland: parcels on a rotated, jittered grid (one orientation and size per ~2 km region),
    // each ploughed, green, ripe or stubble, with grass margins along the edges
    if (crop > 0.01) {
        vec2 rg = floor(q / 1900.0);
        uint hr = hash2(uvec2(ivec2(rg) + 40000));
        float ang = float(hr & 0xFFFFu) * (6.2831853 / 65536.0);
        vec2 size = vec2(110.0 + float((hr >> 16) & 0xFFu) * 0.6, 40.0 + float(hr >> 24) * 0.2);
        vec2 r = mat2(cos(ang), sin(ang), -sin(ang), cos(ang)) * q;
        // gently bent furrow lines: the parcel edges follow the land a little
        r += vec2(fbmN(q * 0.004 + 5.0, 2) - 0.5, fbmN(q * 0.004 + 9.0, 2) - 0.5) * 40.0;
        float row = floor(r.y / size.y);
        r.x += float(hash2(uvec2(int(row) + 70000, hr)) & 0xFFFFu) / 65536.0 * size.x;
        vec2 cellId = vec2(floor(r.x / size.x), row);
        vec2 f = fract(r / size) * size;
        float edge = min(min(f.x, size.x - f.x), min(f.y, size.y - f.y));
        float margin = smoothstep(2.5, 0.8, edge + (nA - 0.5) * 1.5);
        uint hc = hash2(uvec2(ivec2(cellId) + 50000) ^ uvec2(hr));
        float st = float(hc & 0xFFu) / 255.0;
        int layer = st < 0.25 ? 22 : st < 0.62 ? 2 : st < 0.82 ? (biome == 8 || biome == 13 ? 25 : 3) : 24;
        float cw = crop * soil;
        w[2] = max(w[2] - crop * 0.45, 0.0);
        w[3] = max(w[3] - crop * 0.55, 0.0);
        w[layer] += cw * (1.0 - margin);
        ptint = 0.78 + 0.36 * float((hc >> 8) & 0xFFu) / 255.0;
        pw = cw * (1.0 - margin);
        w[1] += cw * margin * 0.6;
        w[2] += cw * margin * 0.4;
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
    float bareFlat = w[7] + built * 0.45 * dryLand; // dry towns: on the local soil
    // bare ground at sea level is a beach: sand, whatever the biome
    if (sea && bed == 0.0) { w[12] += bareFlat * 0.85; w[7] = bareFlat * 0.15; bareFlat = 0.0; }
    w[13] = bareFlat * desert * (0.55 + 0.3 * nB);
    w[14] = bareFlat * (desert * (0.45 - 0.3 * nB) + hot * 0.4 + med * 0.8);
    w[15] += bareFlat * hot * 0.5;
    w[16] = bareFlat * arctic * 0.55 + moss * arctic * 0.3;
    w[17] = bareFlat * arctic * 0.35 + moss * (0.4 + 0.3 * arctic);
    w[7] *= max(1.0 - desert - hot * 0.9 - med * 0.8 - arctic * 0.9, 0.0);
    w[8] *= 1.0 - 0.6 * arctic;
    w[7] += built * 0.45 * soil; // built-up land keeps its gravel in every biome
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
        }
        alb += c * wk;
        nxy += nn * wk;
        ao += o * wk;
    }
    // large-scale variation so meadows and forests are not one flat colour from a distance
    alb *= mix(0.8, 1.15, nB) * mix(0.9, 1.1, nA);
    alb *= mix(1.0, ptint, clamp(pw, 0.0, 1.0));
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
    bool seaSide = elev < 4.0; // the water at this shore / over this bed is the sea
    if (water) {
        n = up;
        alb = groundMaterial(A, B, C.xy, uvm, elev, up, n, footprint, 1.0, 0.0, biome, !lake, ao);
    } else if (cover) {
        alb = groundMaterial(A, B, C.xy, uvm, elev, up, n, footprint, 0.0, shore, biome, seaSide, ao);
        // wet: right at the mapped water, and just above the water level along the contour
        float damp = smoothstep(1.2 + 0.6 * nW0, 0.0, sd);
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
        if (t > gg.misc.y) {
            float k = max(floor(log2(t / gg.misc.z)), 1.0); // levels coarser than the finest
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
    }
}
