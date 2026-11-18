#version 460
#extension GL_GOOGLE_include_directive : require
#include "rt.glsl"

// Closest hit for real triangle geometry: tree trunks, twigs (bark + leaves) and terrain.
// Leaves and bark use the Megaplant's own textures (exported from Unreal, see tools/ue_export.py):
//   0 foliage CA  (sRGB colour, leaf alpha)     1 foliage NT (normal RG, translucency B)
//   2 bark C      (sRGB colour)                 3 bark NAH   (normal RG, AO B, height A)
layout(location = 0) rayPayloadInEXT HitPayload pl;
hitAttributeEXT vec2 bary;

layout(set = 0, binding = 3) uniform GlobalsUBO { Globals g; };
layout(set = 0, binding = 4, scalar) readonly buffer Meshes { MeshInfo meshes[]; };
// 4 textures per species (texSet): foliage CA, foliage NT, bark C, bark NAH.
layout(set = 0, binding = 13) uniform sampler2D matTex[4 * MAX_SPECIES];
layout(set = 0, binding = 15, scalar) readonly buffer Species { SpeciesInfo species[]; };

// Unreal material tints are around 0.5 = neutral; applied at reduced strength.
vec3 ueTint(vec3 t) { return mix(vec3(1.0), clamp(t * 2.0, 0.0, 2.0), 0.6); }

// Scanned ground layers (binding 16, 3 textures each: albedo sRGB, normal, ORM):
//   0 forest floor  1 mossy grass  2 uncut grass  3 dried grass  4 rock cliff  5 dark rock cliff
//   6 snow  7 scree  8 lichen rock  9 river mud  10 river pebbles  11 wet grass (catalog.rs
//   GROUND_LAYERS).  Slot MAX_GROUND_TEX-1 = near
//   biome mask (R canopy cover, G moisture, B dryness, A understory, World::ground_mask), slot
//   MAX_GROUND_TEX-2 = far mask over the whole terrain (R forest site density, World::far_mask).
layout(set = 0, binding = 16) uniform sampler2D groundTex[MAX_GROUND_TEX];
const float GROUND_TILE = 2.2; // metres per texture repeat
// The scans' mean albedos differ ~10x (moss 0.02 .. dried grass 0.19); bring them to 0.05..0.1.
// Mountain scans: neutral until calibrated.
const float GROUND_GAIN[12] = float[12](0.7, 2.4, 0.68, 0.5, 1.0, 1.0, 0.75, 1.0, 1.0, 1.0, 0.8, 1.0);

// Fractal value noise in [0, 1].
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

// Hex-tile stochastic texturing (Mikkelsen 2022, "Practical Real-Time Hex-Tiling"): the plane is
// split into hexagons; each texture lookup blends 3 samples, one per surrounding hex vertex, each
// with its own random rotation and offset. Removes visible repetition without a second texture.
struct HexTile {
    vec2 uv[3];
    mat2 rot[3]; // texture-space -> world-tangent rotation for normal maps (transpose of the uv rotation)
    vec3 w;      // barycentric weights within the hex triangle
};

HexTile hexTile(vec2 uv) {
    HexTile h;
    vec2 st = uv * 1.6; // hex cells a bit smaller than one texture repeat
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
        mat2 R = mat2(c, sn, -sn, c);
        h.uv[i] = R * uv + off * 7.0;
        h.rot[i] = transpose(R);
    }
    return h;
}

// Contrast-preserving blend weights: sharpened barycentrics favouring the brighter sample, so the
// 3-way blend does not wash the texture out.
vec3 hexWeights(vec3 w, vec3 lum) {
    vec3 W = w * w * w * (0.35 + lum);
    W = W * W;
    return W / max(W.x + W.y + W.z, 1e-6);
}

// Blended ground material at world position p (footprint = ray-cone width at the hit).
// Layers 0-3: forest floor, mossy grass, uncut grass, dried grass (biome masks). Layers 4-8, the
// mountains (by slope and true altitude): rock cliff, dark rock cliff, snow, scree, lichen rock.
// A layer whose scan is missing (bit not set in the layer mask) uses a procedural stand-in.
#define GROUND_LAYERS 12
void groundMaterial(vec3 p, inout vec3 n, float footprint, uint layerMask, float bank, out vec3 albedo, out float ao) {
    vec4 mkF = textureLod(groundTex[MAX_GROUND_TEX - 2], (p.xz + TERRAIN_HALF) / (2.0 * TERRAIN_HALF), 0.0);
    vec4 mkN = textureLod(groundTex[MAX_GROUND_TEX - 1], (p.xz + NEAR_HALF) / (2.0 * NEAR_HALF), 0.0);
    float nearW = 1.0 - smoothstep(NEAR_HALF - 300.0, NEAR_HALF - 20.0, max(abs(p.x), abs(p.z)));
    // Far mask: forest site density stands in for canopy cover, at partial strength (the tiles' trees
    // have gaps: mossy ground between them, leaf litter only under the densest stands).
    // Far away the dry-grass layer is toned down: at a distance it dominated as a brown wash.
    vec4 mk = mix(vec4(mkF.r, mkF.g, mkF.b, 0.0), mkN, nearW);
    float cover = mk.r, wet = mk.g, dry = mk.b, under = mk.a;
    // Noise at three scales: ~3 m (ragged edges, small patches), ~16 m (patches), ~80 m (regions).
    float nA = fbmN(p.xz * 0.35, 3), nB = fbmN(p.xz * 0.06 + 17.0, 3), nC = fbmN(p.xz * 0.012 + 41.0, 3);
    float slope = 1.0 - n.y;

    // ---- mountains: true altitude (the terrain carries the Earth's curvature)
    // true altitude, shifted so the zones fit a real place's latitude (world.rs Site::alt_shift)
    float alt = p.y + dot(p.xz, p.xz) / (2.0 * 6371000.0) - g.extra.z;
    float nR = fbmN(p.xz * 0.004 + 7.0, 3);
    // Altitude zones (world.rs treeline ~1250-1450 m): forest and meadows below, alpine meadow,
    // scree and rock above, snow on the summits. Rock where it is too steep for soil to stay
    // (~45 degrees in the lowlands, ~37 in the alpine zone), so outcrops follow the landform.
    float alpine = smoothstep(1250.0, 1450.0, alt + (nR - 0.5) * 160.0);
    float rockSlope = mix(0.3, 0.2, alpine);
    float rockAll = clamp(smoothstep(rockSlope, rockSlope + 0.1, slope + (nR - 0.5) * 0.12 + (nB - 0.5) * 0.06) + 0.7 * smoothstep(1600.0, 1800.0, alt + (nR - 0.5) * 220.0), 0.0, 1.0);
    float snow = smoothstep(1650.0, 1780.0, alt + (nR - 0.5) * 200.0) * (1.0 - smoothstep(0.45, 0.7, slope));
    // scree: debris fans below the rock in the alpine zone, in tongues down the slope
    float scree = alpine * smoothstep(0.1, 0.18, slope) * (1.0 - smoothstep(0.28, 0.38, slope)) * smoothstep(0.35, 0.6, nB + (nA - 0.5) * 0.3) * (1.0 - snow);
    float lichenF = (1.0 - smoothstep(0.45, 0.7, slope)) * (1.0 - smoothstep(1500.0, 1750.0, alt));
    float darkF = smoothstep(0.45, 0.65, fbmN(p.xz * 0.0015 + 3.0, 2));
    float rockW = rockAll * (1.0 - snow);
    float wm[5] = float[5](rockW * (1.0 - lichenF) * (1.0 - darkF), rockW * (1.0 - lichenF) * darkF, snow, scree * (1.0 - rockAll), rockW * lichenF);
    float mtn = clamp(wm[0] + wm[1] + wm[2] + wm[3] + wm[4], 0.0, 1.0);

    // ---- lowland layers (biome masks)
    // Leaf litter under closed canopy and on steep banks, with ragged edges, plus scattered litter
    // and bare patches in the grass near trees.
    float wFloor = smoothstep(0.4, 0.75, cover + 0.2 * under + (nA - 0.5) * 0.45 + (nB - 0.5) * 0.2);
    wFloor = max(wFloor, smoothstep(0.65, 0.8, nA) * smoothstep(0.2, 0.5, cover) * 0.7);
    float open = 1.0 - wFloor;
    float wetN = wet + (nB - 0.5) * 0.6 + (nA - 0.5) * 0.3;
    // Mossy, darker-green patches also in dry meadows (hollows, old clumps): ~10-30 m scale.
    float nD = fbmN(p.xz * 0.045 + 93.0, 3);
    float wMoss = (smoothstep(0.42, 0.78, wetN) + 0.7 * smoothstep(0.15, 0.6, cover + (nA - 0.5) * 0.3) + 0.8 * smoothstep(0.6, 0.78, nD + (nA - 0.5) * 0.25)) * (open + 0.4 * wFloor);
    // Sun-facing slopes dry out (more insolation), shaded ones stay lush.
    vec2 sunH = g.sunDir.xz / max(length(g.sunDir.xz), 1e-3);
    float sunny = dot(n.xz, sunH);
    float dryN = dry + 1.2 * sunny + (nC - 0.5) * 0.2 + (nB - 0.5) * 0.3 + (nA - 0.5) * 0.25;
    float wDry = open * smoothstep(0.45, 0.8, dryN) * (1.0 - 0.7 * wet);
    float wGrass = open * (1.0 - 0.6 * smoothstep(0.4, 0.8, wetN)) * (1.0 - smoothstep(0.35, 0.75, dryN));
    // ---- river banks: pebble bars at the water, muddy riverbed, then wet grass on the floodplain
    float bN = bank + (nA - 0.5) * 0.12 + (nB - 0.5) * 0.1;
    float wPeb = smoothstep(0.9, 0.97, bN);
    float wMud = smoothstep(0.66, 0.86, bN) * (1.0 - wPeb) * smoothstep(0.35, 0.65, nB + 0.2);
    float wWetG = smoothstep(0.2, 0.55, bN) * (1.0 - wPeb) * (1.0 - wMud);
    float riv = clamp(wPeb + wMud + wWetG, 0.0, 1.0) * (1.0 - mtn);
    float low = (1.0 - mtn) * (1.0 - riv);
    float w[GROUND_LAYERS] = float[GROUND_LAYERS](wFloor * low, wMoss * low, wGrass * low, wDry * low, wm[0], wm[1], wm[2], wm[3], wm[4],
                                                  wMud * (1.0 - mtn), wPeb * (1.0 - mtn), wWetG * (1.0 - mtn));
    for (int l = 0; l < GROUND_LAYERS; ++l) w[l] += 1e-4;

    // Mip from the ray cone, widened at grazing angles (geometric mean of the footprint axes).
    vec3 V = -gl_WorldRayDirectionEXT;
    float cosv = max(abs(dot(n, V)), 0.03);
    float texel = GROUND_TILE / float(textureSize(groundTex[0], 0).x);
    float lod = max(log2(footprint / texel) - 0.5 * log2(cosv), 0.0);
    HexTile hx = hexTile(p.xz / GROUND_TILE);
    // Tangent frame of the ground (u along +X, v along +Z), for the triplanar rock normals.
    vec3 Tg = normalize(vec3(1.0, 0.0, 0.0) - n * n.x);
    vec3 Bg = cross(Tg, n);
    vec3 tw = pow(abs(n), vec3(4.0));
    tw /= tw.x + tw.y + tw.z;

    vec3 alb[GROUND_LAYERS];
    vec2 nrm[GROUND_LAYERS];
    float occ[GROUND_LAYERS];
    float h[GROUND_LAYERS];
    for (int l = 0; l < GROUND_LAYERS; ++l) {
        alb[l] = vec3(0.0); nrm[l] = vec2(0.0); occ[l] = 1.0; h[l] = 0.0;
        if (w[l] < 0.02) continue;
        // Missing river scans borrow a lowland scan: mud <- forest floor (darker), wet grass <- mossy grass.
        int src = l;
        vec3 tint = vec3(1.0);
        if ((layerMask & (1u << uint(l))) == 0u && (l == 9 || l == 11) && (layerMask & 3u) == 3u) {
            src = l == 9 ? 0 : 1;
            tint = l == 9 ? vec3(0.5, 0.48, 0.42) : vec3(0.9, 1.05, 0.85);
        }
        if (src == l && (layerMask & (1u << uint(l))) == 0u) {
            // procedural stand-in (mountain layers without a scan)
            float m = mix(0.7, 1.3, fbmN(p.xz * 0.08 + float(l) * 13.0, 3)) * mix(0.85, 1.15, nA);
            vec3 c = l == 4 ? vec3(0.20, 0.19, 0.17) : l == 5 ? vec3(0.09, 0.085, 0.08) : l == 6 ? vec3(0.85, 0.87, 0.90)
                   : l == 7 ? vec3(0.13, 0.125, 0.12) : l == 8 ? vec3(0.16, 0.17, 0.13) : l == 9 ? vec3(0.055, 0.05, 0.036)
                   : l == 10 ? vec3(0.11, 0.10, 0.085) : vec3(0.05, 0.075, 0.028);
            alb[l] = c * (l == 6 ? mix(0.95, 1.05, m - 0.5) : m);
            h[l] = dot(alb[l], vec3(0.3, 0.5, 0.2)) * 3.0 + 0.5;
            continue;
        }
        if (l == 4 || l == 5 || l == 8) {
            // Rock: triplanar projection (a top-down projection smears down cliff faces), larger
            // repeat, plus a macro sample at 1/6 scale and horizontal strata so faces read as rock
            // from a distance.
            const float RT = 5.0;
            float lodR = max(log2(footprint / (RT / float(textureSize(groundTex[0], 0).x))) - 0.5 * log2(cosv), 0.0);
            vec3 ra = vec3(0.0), off = vec3(0.0);
            float ro = 0.0;
            for (int ax = 0; ax < 3; ++ax) {
                if (tw[ax] < 0.02) continue;
                vec2 uv = (ax == 0 ? p.zy : ax == 1 ? p.xz : p.xy) / RT;
                vec3 c0 = textureLod(groundTex[src * 3], uv, lodR).rgb;
                vec3 c1 = textureLod(groundTex[src * 3], uv * 0.17 + 0.37, max(lodR - 2.58, 0.0)).rgb;
                ra += c0 * mix(vec3(1.0), 2.0 * c1 / max(c0 + c1, vec3(0.02)), 0.5) * tw[ax];
                vec2 nt = textureLod(groundTex[src * 3 + 1], uv, lodR).rg * 2.0 - 1.0;
                off += (ax == 0 ? vec3(0.0, nt.y, nt.x) : ax == 1 ? vec3(nt.x, 0.0, nt.y) : vec3(nt.x, nt.y, 0.0)) * tw[ax];
                ro += textureLod(groundTex[src * 3 + 2], uv, lodR).r * tw[ax];
            }
            float strata = 0.82 + 0.3 * vnoise(vec2(p.y * 0.22 + 0.5 * vnoise(p.xz * 0.01), (p.x + p.z) * 0.004));
            alb[l] = ra * tint * strata * mix(0.8, 1.15, fbmN(p.xz * 0.03 + p.y * 0.02, 2)) * GROUND_GAIN[src];
            nrm[l] = vec2(dot(off, Tg), dot(off, Bg));
            occ[l] = ro;
            h[l] = dot(alb[l], vec3(0.3, 0.5, 0.2)) * 3.0 + occ[l] * 0.5;
            continue;
        }
        vec3 a[3];
        vec3 lum;
        for (int i = 0; i < 3; ++i) {
            a[i] = textureLod(groundTex[src * 3], hx.uv[i], lod).rgb * tint;
            lum[i] = dot(a[i], vec3(0.3, 0.5, 0.2));
        }
        vec3 hw3 = hexWeights(hx.w, lum);
        alb[l] = a[0] * hw3.x + a[1] * hw3.y + a[2] * hw3.z;
        occ[l] = 0.0;
        for (int i = 0; i < 3; ++i) {
            if (hw3[i] < 0.01) continue;
            vec2 nt = textureLod(groundTex[src * 3 + 1], hx.uv[i], lod).rg * 2.0 - 1.0;
            nrm[l] += (hx.rot[i] * nt) * hw3[i];
            occ[l] += textureLod(groundTex[src * 3 + 2], hx.uv[i], lod).r * hw3[i];
        }
        occ[l] = occ[l] / max(hw3.x * float(hw3.x >= 0.01) + hw3.y * float(hw3.y >= 0.01) + hw3.z * float(hw3.z >= 0.01), 1e-4);
        alb[l] *= GROUND_GAIN[src];
        h[l] = dot(alb[l], vec3(0.3, 0.5, 0.2)) * 3.0 + occ[l] * 0.5; // brighter / less occluded = higher
    }
    // Height-based blend: the taller layer wins near the transition (grass blades over litter).
    float hw[GROUND_LAYERS];
    float hsum = 0.0;
    for (int l = 0; l < GROUND_LAYERS; ++l) {
        float v = w[l] * (h[l] + 0.4);
        hw[l] = v * v * v;
        hsum += hw[l];
    }
    albedo = vec3(0.0);
    vec2 nxy = vec2(0.0);
    ao = 0.0;
    for (int l = 0; l < GROUND_LAYERS; ++l) {
        float k = hw[l] / max(hsum, 1e-12);
        albedo += alb[l] * k;
        nxy += nrm[l] * k;
        ao += occ[l] * k;
    }
    if (g.iparams.y == 11u) {
        // debug: dominant ground layers colour-coded (floor orange, moss dark green, grass lime,
        // dry yellow, rock grey / dark grey, snow white, scree red, lichen purple, mud brown,
        // pebbles cyan, wet grass blue)
        const vec3 PAL[12] = vec3[12](vec3(1.0, 0.45, 0.0), vec3(0.0, 0.35, 0.0), vec3(0.5, 1.0, 0.0), vec3(1.0, 1.0, 0.0),
            vec3(0.6), vec3(0.25), vec3(1.0), vec3(1.0, 0.0, 0.0), vec3(0.6, 0.0, 1.0), vec3(0.3, 0.15, 0.05), vec3(0.0, 1.0, 1.0), vec3(0.0, 0.2, 1.0));
        albedo = vec3(0.0);
        for (int l = 0; l < GROUND_LAYERS; ++l) albedo += PAL[l] * hw[l] / max(hsum, 1e-12);
        albedo *= 0.25;
    }
    // Large-scale variation: brightness over ~80 m and a warm/cool shift over ~16 m, so open
    // ground does not read as one flat colour from a distance (lowland layers only).
    float vary = mix(1.0, 0.5, wFloor) * low;
    albedo *= mix(1.0, mix(0.72, 1.2, nC) * mix(0.78, 1.15, smoothstep(0.25, 0.75, nD)), vary)
            * mix(vec3(1.0), mix(vec3(1.12, 1.0, 0.78), vec3(0.9, 1.03, 1.14), smoothstep(0.3, 0.7, nB)), vary)
            * mix(1.0, mix(0.9, 1.1, nA), low);
    ao = mix(1.0, ao, 0.85);
    // Sky occlusion under far forest: AO rays stop at 600 m, so beyond that the canopy's shade on
    // the forest floor comes from the forest density.
    float dc = distance(p, g.camPos.xyz);
    float coverF = smoothstep(0.1, 0.7, mk.r);
    ao *= mix(1.0, 0.3, smoothstep(450.0, 900.0, dc) * coverF * low);
    // Small-scale relief the mesh cannot hold (hummocks, bumps, 2-20 m): normal from the gradient
    // of a height noise, stronger on steep / mountain ground, faded where the mesh is finer.
    {
        float e = 0.6;
        vec2 q = p.xz * 0.11;
        float h0 = fbmN(q, 3), hx = fbmN(q + vec2(e * 0.11, 0.0), 3), hz = fbmN(q + vec2(0.0, e * 0.11), 3);
        float amp = 0.35 + 1.2 * smoothstep(0.1, 0.4, slope);
        nxy += vec2(-(hx - h0), -(hz - h0)) / e * amp * 1.6;
    }
    // Tangent frame: u along +X, v along +Z.
    vec3 T = normalize(vec3(1.0, 0.0, 0.0) - n * n.x);
    vec3 B = cross(T, n);
    n = normalize(T * nxy.x + B * nxy.y + n * sqrt(max(1.0 - dot(nxy, nxy), 0.05)));
}

vec3 terrainAlbedo(vec3 p, vec3 n) {
    float n1 = vnoise(p.xz * 0.015), n2 = vnoise(p.xz * 0.12), n3 = vnoise(p.xz * 0.9);
    vec3 grass = mix(vec3(0.050, 0.080, 0.022), vec3(0.105, 0.115, 0.040), n1);
    vec3 dirt = vec3(0.085, 0.066, 0.046);
    float m = smoothstep(0.55, 0.85, n2 * 0.7 + n3 * 0.3);
    vec3 c = mix(grass, dirt, m * 0.65);
    c = mix(c, vec3(0.12, 0.11, 0.10), smoothstep(0.22, 0.45, 1.0 - n.y));
    return c * (0.85 + 0.3 * n3);
}

void main() {
    uint ci = gl_InstanceCustomIndexEXT;
    uint meshId = ci & 0x7FFu; // 11 bits; the seed sits above
    uint seed = instSeed(ci);
    MeshInfo m = meshes[meshId];

    uvec3 tri = UVec3Ref(m.idx).v[gl_PrimitiveID];
    Vec3Ref P = Vec3Ref(m.pos);
    Vec3Ref N = Vec3Ref(m.nrm);
    vec3 b = vec3(1.0 - bary.x - bary.y, bary.x, bary.y);
    vec3 p0 = P.v[tri.x], p1 = P.v[tri.y], p2 = P.v[tri.z];
    vec3 n = N.v[tri.x] * b.x + N.v[tri.y] * b.y + N.v[tri.z] * b.z;

    mat3 o2w = mat3(gl_ObjectToWorldEXT);
    vec3 e1 = o2w * (p1 - p0), e2 = o2w * (p2 - p0);
    vec3 ng = normalize(cross(e1, e2));
    n = normalize(o2w * n);

    uint word = UintRef(m.mat).v[gl_PrimitiveID >> 2];
    uint mbyte = (word >> ((gl_PrimitiveID & 3u) * 8u)) & 0xFFu;
    uint mat = mbyte & 3u;       // low 2 bits: material
    uint leafId = mbyte >> 2;    // high 6 bits: random per individual leaf (from the converter)
    uint extraFlags = 0u;
    bool textured = g.params.y > 0.5 && mat != MAT_TERRAIN;

    float ao = 1.0;
    vec3 trans = vec3(0.0);
    vec3 albedo;
    uint ts = m.texSet * 4u;
    SpeciesInfo spc = species[m.texSet];
    if (mat == MAT_TERRAIN) {
        vec3 wp = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
        float spread = g.camFwd.w;
        uint layerMask = uint(g.lodParams.w); // scanned ground layers present (bit per layer)
        // Water data from the terrain mesh UVs (world.rs): x = water depth over the bed (m), y = bank
        // proximity.
        Vec2Ref RV = Vec2Ref(m.uv);
        vec2 riv = RV.v[tri.x] * b.x + RV.v[tri.y] * b.y + RV.v[tri.z] * b.z;
        if ((layerMask & 15u) == 15u && spread > 0.0) groundMaterial(wp, n, gl_HitTEXT * spread, layerMask, riv.y, albedo, ao);
        else albedo = terrainAlbedo(wp, n);
        if (riv.x > 0.01) {
            // Under water: this is the bed (pebbles / mud from the bank layers, wet and dark);
            // main.rgen puts the water surface above it (refraction, absorption, flow, foam).
            albedo *= 0.55;
            pl.wdepth = riv.x;
        } else {
            // wet banks just above the water line
            albedo *= mix(1.0, 0.62, smoothstep(0.86, 1.0, riv.y));
        }
    } else if (textured) {
        Vec2Ref UV = Vec2Ref(m.uv);
        vec2 t0 = UV.v[tri.x], t1 = UV.v[tri.y], t2 = UV.v[tri.z];
        vec2 uv = t0 * b.x + t1 * b.y + t2 * b.z;
        vec2 d1 = t1 - t0, d2 = t2 - t0;
        float uvArea = abs(d1.x * d2.y - d1.y * d2.x);
        float worldArea = length(cross(e1, e2));
        // Ray-cone mip selection (RT shaders have no derivatives): footprint of the pixel at the
        // hit vs the texel size on this triangle. Negative spread = fixed footprint (impostor bake).
        float spread = g.camFwd.w;
        float footprint = spread > 0.0 ? gl_HitTEXT * spread : -spread;
        float texRes = float(textureSize(matTex[nonuniformEXT(ts + (mat == MAT_LEAF ? 0u : 2u))], 0).x);
        float lod = log2(max(footprint, 1e-6)) + 0.5 * log2(max(uvArea, 1e-14) / max(worldArea, 1e-14)) + log2(texRes);
        lod = clamp(lod, 0.0, 12.0);

        // Tangent frame from the triangle's UV gradients (image space, v down).
        float det = d1.x * d2.y - d1.y * d2.x;
        float r = abs(det) > 1e-14 ? 1.0 / det : 0.0;
        vec3 T = (e1 * d2.y - e2 * d1.y) * r;
        vec3 B = (e2 * d1.x - e1 * d2.x) * r;
        T = normalize(T - n * dot(n, T) + 1e-9);
        B = normalize(B - n * dot(n, B) + 1e-9);

        bool underside = dot(ng, gl_WorldRayDirectionEXT) > 0.0;
        vec4 nrmTex;
        if (mat == MAT_LEAF) {
            vec4 ca = textureLod(matTex[nonuniformEXT(ts + 0u)], uv, lod);
            nrmTex = textureLod(matTex[nonuniformEXT(ts + 1u)], uv, lod);
            // set below, once the albedo is known
            // Subtle per-leaf variation on top of the texture (texture already has per-leaf colour).
            uint lr = pcg(leafId * 7919u + uint(gl_InstanceID) * 104729u + 17u);
            float h1 = float(lr & 0xFFFFu) / 65535.0, h2 = float(lr >> 16) / 65535.0;
            vec3 hue = mix(vec3(1.06, 1.0, 0.88), vec3(0.94, 1.0, 1.08), h1) * mix(0.88, 1.12, h2);
            albedo = ca.rgb * ueTint(spc.leafTint.rgb) * spc.leafTint.w * hue * treeTint(seed, 1.0);
            if (underside) albedo *= vec3(1.18, 1.15, 1.25); // paler underside
            // Transmission like Unreal's two-sided foliage: albedo x translucency tint x mask.
            trans = albedo * spc.transTint.rgb * (nrmTex.b * spc.transTint.w);
            extraFlags = underside ? 0u : FLAG_GLOSSY;
        } else {
            albedo = textureLod(matTex[nonuniformEXT(ts + 2u)], uv, lod).rgb * ueTint(spc.barkTint.rgb);
            nrmTex = textureLod(matTex[nonuniformEXT(ts + 3u)], uv, lod);
            ao = pow(nrmTex.b, 1.5); // AO Strength 1.5
        }
        // Normal map (DirectX convention, Unreal). The back side of a leaf sees the mirrored bumps.
        vec2 xy = nrmTex.rg * 2.0 - 1.0;
        float z = sqrt(max(1.0 - dot(xy, xy), 0.0));
        vec3 nb = (underside && mat == MAT_LEAF) ? -n : n;
        n = normalize(T * xy.x + B * xy.y + nb * z);
        if (underside && mat == MAT_LEAF) n = -n; // keep "front" orientation; raygen flips towards the viewer
    } else if (mat == MAT_LEAF) {
        // No textures available: procedural leaf colour.
        bool underside = dot(ng, gl_WorldRayDirectionEXT) > 0.0;
        vec3 base = underside ? vec3(0.100, 0.135, 0.052) : vec3(0.048, 0.088, 0.020);
        uint lr = pcg(leafId * 7919u + uint(gl_InstanceID) * 104729u + 17u);
        float h1 = float(lr & 0xFFFFu) / 65535.0, h2 = float(lr >> 16) / 65535.0;
        vec3 hue = mix(vec3(1.15, 1.0, 0.65), vec3(0.86, 1.0, 1.18), h1);
        albedo = base * hue * mix(0.70, 1.28, h2 * h2) * treeTint(seed, 1.0);
        extraFlags = underside ? 0u : FLAG_GLOSSY;
        trans = 0.33 * spc.transTint.rgb * albedo;
    } else {
        albedo = vec3(0.105, 0.082, 0.064);
    }

    pl.albedo = albedo;
    pl.t = gl_HitTEXT;
    pl.n = n;
    pl.ng = ng;
    pl.flags = mat | extraFlags;
    pl.ao = ao;
    pl.sunVis = 1.0;
    pl.shadowMask = 0xFFu;
    pl.trans = trans;
}
