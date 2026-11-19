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
// 6 textures per species (texSet; textures.rs SLOTS): foliage colour + alpha (BC7), normal x / y
// (BC5), translucency (BC4); bark colour (BC7), normal x / y (BC5), occlusion (BC4).
layout(set = 0, binding = 13) uniform sampler2D matTex[6 * MAX_SPECIES];
layout(set = 0, binding = 15, scalar) readonly buffer Species { SpeciesInfo species[]; };
layout(set = 0, binding = 12, scalar) readonly buffer Insts { InstInfo insts[]; };

// Unreal material tints are around 0.5 = neutral; applied at reduced strength.
vec3 ueTint(vec3 t) { return mix(vec3(1.0), clamp(t * 2.0, 0.0, 2.0), 0.6); }

// Scanned ground layers (binding 16, 3 textures each: albedo sRGB, normal, ORM):
//   0 forest floor  1 mossy grass  2 uncut grass  3 dried grass  4 rock cliff  5 dark rock cliff
//   6 snow  7 scree  8 lichen rock  9 river mud  10 river pebbles  11 wet grass (catalog.rs
//   GROUND_LAYERS).  Slot MAX_GROUND_TEX-1 = near
//   biome mask (R canopy cover, G moisture, B dryness, A understory, World::ground_mask), slot
//   MAX_GROUND_TEX-2 = far mask over the whole terrain (R forest site density, World::far_mask).
layout(set = 0, binding = 16) uniform sampler2D groundTex[MAX_GROUND_TEX];
// the impostor atlases (impostor.rint): here for the lawn, whose atlas is one of the slots
layout(set = 0, binding = 7) uniform sampler2D depthAtlas[MAX_VARIANTS];
layout(set = 0, binding = 14) uniform sampler2D normalAtlas[MAX_VARIANTS];
layout(set = 0, binding = 6) uniform sampler2D albedoAtlas[MAX_VARIANTS];
layout(set = 0, binding = 11) uniform sampler2D visAtlas[MAX_VARIANTS];
#include "lawn.glsl"
#include "lawnh.glsl"
const float GROUND_TILE = 2.2; // metres per texture repeat
// The scans' mean albedos differ ~10x (moss 0.02 .. dried grass 0.19); bring them to 0.05..0.1.
// Mountain scans: neutral until calibrated.
// The biome presets (biome.rs) map a slot to any of the 26 scans: their gains as in globe.rchit.
const float GROUND_GAIN[26] = float[26](0.7, 2.4, 0.68, 0.5, 1.0, 1.0, 0.75, 1.0, 1.0, 1.0, 0.8, 1.0,
                                        1.02, 1.99, 4.0, 2.27, 3.3, 0.85, 0.58, 4.09, 3.69, 2.65, 2.62, 3.97, 3.44, 4.02);
// The scan a ground slot shows in this biome (Globals.biome, one byte per slot).
int groundSrc(int l) { return int((g.biome[l >> 2] >> (8u * uint(l & 3))) & 0xFFu); }

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
// lawn: how much of the ground here is grass (the grass, wet grass and mossy grass slots).
void groundMaterial(vec3 p, inout vec3 n, float footprint, uint layerMask, float bank, out vec3 albedo, out float ao, out float lawn) {
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
    // (the park, lawn3.z: a kept lawn, even in colour, with only a hint of the mossy patches)
    wMoss *= 1.0 - 0.95 * g.lawn3.z;
    // Sun-facing slopes dry out (more insolation), shaded ones stay lush.
    vec2 sunH = g.sunDir.xz / max(length(g.sunDir.xz), 1e-3);
    float sunny = dot(n.xz, sunH);
    float dryN = dry + 1.2 * sunny + (nC - 0.5) * 0.2 + (nB - 0.5) * 0.3 + (nA - 0.5) * 0.25 - 2.0 * g.lawn3.z;
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
        // The biome's scan for this slot (its own one where that scan is not downloaded). Missing
        // river scans borrow a lowland scan: mud <- forest floor (darker), wet grass <- mossy grass.
        int src = groundSrc(l);
        if (src >= 26 || (layerMask & (1u << uint(src))) == 0u) src = l;
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
        // the arid soil scan is a saturated brick red: dusty red-brown earth instead (as globe.rchit)
        if (src == 14) alb[l] = mix(alb[l], vec3(dot(alb[l], vec3(0.3, 0.59, 0.11))) * vec3(1.25, 0.95, 0.7), 0.55) * 0.85;
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
    lawn = (hw[2] + 0.8 * hw[11] + 0.6 * hw[1]) / max(hsum, 1e-12);
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

    uvec3 tri = meshTri(m, gl_PrimitiveID);
    Vec3Ref P = Vec3Ref(m.pos);
    Vec3Ref N = Vec3Ref(m.nrm);
    vec3 b = vec3(1.0 - bary.x - bary.y, bary.x, bary.y);
    vec3 p0 = P.v[tri.x], p1 = P.v[tri.y], p2 = P.v[tri.z];
    vec3 n = N.v[tri.x] * b.x + N.v[tri.y] * b.y + N.v[tri.z] * b.z;

    mat3 o2w = mat3(gl_ObjectToWorldEXT);
    vec3 e1 = o2w * (p1 - p0), e2 = o2w * (p2 - p0);
    vec3 ng = normalize(cross(e1, e2));
    n = normalize(o2w * n);

    // (a grid mesh is the terrain, all of it: no material buffer)
    uint mbyte = MAT_TERRAIN;
    if ((m.flags >> 8) == 0u) {
        uint word = UintRef(m.mat).v[gl_PrimitiveID >> 2];
        mbyte = (word >> ((gl_PrimitiveID & 3u) * 8u)) & 0xFFu;
    }
    uint mat = mbyte & 3u;       // low 2 bits: material
    uint leafId = mbyte >> 2;    // high 6 bits: random per individual leaf (from the converter)
    uint extraFlags = 0u;
    bool textured = g.params.y > 0.5 && mat != MAT_TERRAIN;

    float ao = 1.0;
    vec3 trans = vec3(0.0);
    vec3 albedo;
    // (the lawn moves the hit up to a blade, in the blades' own shade)
    float tHit = gl_HitTEXT, sunVis = 1.0;
    uint ts = m.texSet * 6u;
    SpeciesInfo spc = species[m.texSet];
    // What this instance is, for the per-leaf colour variation: the plant and which of its parts
    // (the same whenever the instance list is rebuilt, so the leaves keep their colours as the
    // camera moves), and its biome's dryness. The impostor bake has only the plant's parts, in
    // order, and no tint (impostor.rchit applies the plant's own).
    bool bake = g.lodParams.z < -0.5;
    InstInfo inst = insts[bake ? 0 : gl_InstanceID];
    uint instId = bake ? uint(gl_InstanceID) : (inst.flags >> 5) + (inst.data & 0xFFFFFFu) * 2053u;
    float instDry = bake ? 0.0 : float((inst.flags >> 1) & 15u) / 15.0;
    if (mat == MAT_TERRAIN) {
        vec3 wp = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
        float spread = g.camFwd.w;
        uint layerMask = floatBitsToUint(g.lodParams.w); // scanned ground layers present (bit per layer)
        // Water data from the terrain mesh UVs (world.rs): x = water depth over the bed (m), y = bank
        // proximity.
        Vec2Ref RV = Vec2Ref(m.uv);
        vec2 riv = RV.v[tri.x] * b.x + RV.v[tri.y] * b.y + RV.v[tri.z] * b.z;
        // (without the ground scans: grass wherever it is not steep)
        float lawnAmt = 1.0 - smoothstep(0.1, 0.3, 1.0 - n.y);
        if ((layerMask & 15u) == 15u && spread > 0.0) groundMaterial(wp, n, gl_HitTEXT * spread, layerMask, riv.y, albedo, ao, lawnAmt);
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
        // ---- the lawn (lawn.glsl): the ray is marched through the baked blades in the slab of
        // one blade height above the ground it has just hit. A blade: the hit moves up to it. No
        // blade: the soil between them, in their shade.
        Lawn lw;
        lw.H = g.lawn.x;
        vec3 up = ng.y < 0.0 ? -ng : ng;
        float down = -dot(gl_WorldRayDirectionEXT, up); // how fast the ray comes down on the ground
        lw.cover = smoothstep(0.25, 0.6, lawnAmt) * (1.0 - smoothstep(0.6 * g.lawn4.w, g.lawn4.w, gl_HitTEXT));
        // (around the camera the lawn is real blades, renderer.rs place_grass: the layer comes in
        // under them as they shrink into the ground at the edge of that circle)
        if (g.lawn3.w > 0.0) lw.cover *= smoothstep(0.5, 0.88, length(wp - g.camPos.xyz) / g.lawn3.w);
        if (lw.H > 0.0 && spread > 0.0 && riv.x <= 0.01 && down > 1e-3 && up.y > 0.3 && lw.cover > 0.03) {
            lw.slot = uint(g.lawn2.w);
            lw.T = g.lawn.y;
            lw.n = g.lawn.zw;
            lw.slopes = g.lawn3.xy;
            lw.res = g.lawn2.x;
            lw.gut = g.lawn2.y;
            // A real lawn is not one square repeated: it is taller here and shorter there, and no
            // row of tiles shows. The blades' height varies over a few metres (the baked blades
            // are squashed: the ray is stretched the other way, and marched as baked; the park is
            // mown, nearly even), and the
            // tiling is bent by a slow, smooth offset.
            // (lawnh.glsl: the real blades near the camera stand as tall)
            float tall = lawnStands(wp.xz);
            // (and in tufts a foot across, with their own shade: the texture a lawn keeps from
            // further away than its blades show; gone before a pixel is as wide as a tuft)
            float tufts = 1.0 - smoothstep(30.0, 110.0, gl_HitTEXT);
            vec2 tuft = vec2(vnoise(wp.xz * 3.3 + 5.0), vnoise(wp.xz * 5.9 + 17.0));
            tall *= mix(1.0, mix(0.84, 1.0, tuft.x), tufts);
            // (gently: every copy of the tile is shifted on its own already, lawn.glsl lawnTile,
            // and a strong bend shears tall blades into swirls)
            vec2 bend = (vec2(vnoise(wp.xz * 0.31 + 3.7), vnoise(wp.xz * 0.31 + 57.1)) - 0.5) * 0.22
                      + (vec2(vnoise(wp.xz * 0.07 + 23.9), vnoise(wp.xz * 0.07 + 91.3)) - 0.5) * 0.8;
            // tile space: world x and z, y the height above the ground (measured up, also on a
            // slope) in baked blade heights
            vec3 D = vec3(gl_WorldRayDirectionEXT.x, -down / (up.y * tall), gl_WorldRayDirectionEXT.z);
            float s1 = min(1.04 * lw.H / -D.y, gl_HitTEXT - gl_RayTminEXT);
            vec3 P0 = vec3(wp.x + bend.x, 0.0, wp.z + bend.y) - D * s1;
            vec3 sun = normalize(vec3(g.sunDir.x, g.sunDir.y / tall, g.sunDir.z));
            float texel = lw.T / (lw.res - 2.0 * lw.gut);
            lw.lod = clamp(log2(max(gl_HitTEXT * spread / texel, 1e-6)) - 1.0, 0.0, g.lawn2.z - 1.0);
            lw.marchLod = floor(lw.lod + 0.5);
            uint seed = hash2(gl_LaunchIDEXT.xy) ^ pcg(g.iparams.z * 7919u + 0x1a3fu);
            lw.thr = mix(0.03, 0.97, rnd(seed));
            vec2 cell = lawnPick(lw, normalize(-D), rnd(seed));
            float sH;
            vec2 uvH;
            bool blade = lawnMarch(lw, cell, P0, D, s1, sH, uvH);
            // Down between the blades without meeting one: from above that is the soil. Seen at a
            // low angle it cannot be (the ray has crossed the whole depth of the lawn sideways):
            // then it is the base of the blades, deep in their shade.
            float deep = 1.0;
            if (!blade && rnd(seed) > smoothstep(0.2, 0.6, -normalize(D).y)) {
                blade = true;
                deep = 0.45;
            }
            if (blade) {
                vec2 auv = lawnAtlasUV(lw, cell, uvH);
                vec4 a = textureLod(albedoAtlas[nonuniformEXT(lw.slot)], auv, lw.lod);
                vec2 nxy = textureLod(normalAtlas[nonuniformEXT(lw.slot)], auv, lw.lod).rg * 2.0 - 1.0;
                vec4 vs = textureLod(visAtlas[nonuniformEXT(lw.slot)], auv, max(lw.lod - 1.0, 0.0)); // half-res atlas
                // (the atlas stores sqrt(albedo)) The blades take on the colour of the ground they
                // grow from: lush, mossy or dry as the ground texture is there.
                vec3 tint = clamp(albedo / vec3(0.050, 0.085, 0.022), 0.5, 1.8);
                albedo = a.rgb * a.rgb * mix(vec3(1.0), tint, 0.6) * mix(0.82, 1.12, vnoise(wp.xz * 0.55 + 41.0)) * mix(1.0, mix(0.88, 1.1, tuft.y), tufts);
                n = octDecode(nxy);
                ng = n;
                ao = a.a * deep;
                trans = albedo * 2.0;
                mat = MAT_LEAF;
                extraFlags = FLAG_GLOSSY | FLAG_IMPOSTOR | FLAG_LAWN;
                tHit = gl_HitTEXT - (s1 - sH);
                // blade normals point every way within a texel: the spread scales the sunlight
                // (as for the grass impostors), and the blades towards the sun shade it
                sunVis = clamp(vs.y, 0.2, 1.0) * lawnSun(lw, P0 + D * sH, sun, seed) * deep;
            } else {
                ao *= mix(1.0, 0.45, lw.cover);
                sunVis = lawnSun(lw, P0 + D * s1, sun, seed);
            }
        }
    } else if ((m.flags & 2u) != 0u) {
        // A blade of the lawn tile (lawn.rs; only ever seen by the lawn's bake): coloured from
        // where on the blade this is (uv.y: root 0 .. tip 1) and the blade's random number.
        Vec2Ref UV = Vec2Ref(m.uv);
        vec2 uv = UV.v[tri.x] * b.x + UV.v[tri.y] * b.y + UV.v[tri.z] * b.z;
        // (a seed stalk of unmown grass: uv.y + 2)
        bool stalk = uv.y > 1.5;
        if (stalk) uv.y -= 2.0;
        uint lr = pcg(leafId * 7919u + 91u);
        float h1 = float(lr & 0xFFFFu) / 65535.0, h2 = float(lr >> 16) / 65535.0;
        // darker and bluer at the root, lighter and yellower at the tip; every blade its own
        // shade, a few of them straw
        vec3 green = mix(vec3(0.026, 0.050, 0.012), vec3(0.070, 0.118, 0.028), smoothstep(0.0, 0.85, uv.y));
        green *= mix(vec3(1.10, 1.0, 0.72), vec3(0.90, 1.0, 1.10), h1) * mix(0.78, 1.2, h2);
        vec3 straw = vec3(0.16, 0.13, 0.06) * mix(0.7, 1.1, h1);
        albedo = mix(green, straw, step(0.94, h2) * smoothstep(0.2, 0.7, uv.y));
        albedo *= mix(0.86, 1.0, abs(uv.x * 2.0 - 1.0)); // the midrib
        // (the stalk dries towards its seed head, which is tan)
        if (stalk) albedo = mix(mix(green, straw, smoothstep(0.25, 0.7, uv.y)), vec3(0.125, 0.105, 0.055) * mix(0.75, 1.15, h1), smoothstep(0.68, 0.76, uv.y));
        if (!bake) {
            // A real blade near the camera (renderer.rs place_grass): in the colour the layer
            // gives its blades further out, that of the ground it grows from (the lawn above).
            vec3 wp = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
            uint layerMask = floatBitsToUint(g.lodParams.w);
            float spread = g.camFwd.w;
            if ((layerMask & 15u) == 15u && spread > 0.0) {
                vec3 gn = vec3(0.0, 1.0, 0.0), galb;
                float gao, gl;
                groundMaterial(wp, gn, gl_HitTEXT * spread, layerMask, 0.0, galb, gao, gl);
                vec3 tint = clamp(galb / vec3(0.050, 0.085, 0.022), 0.5, 1.8);
                float tufts = 1.0 - smoothstep(30.0, 110.0, gl_HitTEXT);
                albedo *= mix(vec3(1.0), tint, 0.6) * mix(0.82, 1.12, vnoise(wp.xz * 0.55 + 41.0)) * mix(1.0, mix(0.88, 1.1, vnoise(wp.xz * 5.9 + 17.0)), tufts);
            }
        }
        // (a real blade near the camera has no highlight of its own: every blade glinting makes
        // the grass pale next to the layer further out, whose texels are many blades each)
        extraFlags = bake ? FLAG_GLOSSY : 0u;
        trans = albedo * 2.0;
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
        float texRes = float(textureSize(matTex[nonuniformEXT(ts + (mat == MAT_LEAF ? 0u : 3u))], 0).x);
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
            nrmTex = vec4(textureLod(matTex[nonuniformEXT(ts + 1u)], uv, lod).rg, textureLod(matTex[nonuniformEXT(ts + 2u)], uv, lod).r, 1.0);
            // set below, once the albedo is known
            // Subtle per-leaf variation on top of the texture (texture already has per-leaf colour).
            uint lr = pcg(leafId * 7919u + instId * 104729u + 17u);
            float h1 = float(lr & 0xFFFFu) / 65535.0, h2 = float(lr >> 16) / 65535.0;
            vec3 hue = mix(vec3(1.06, 1.0, 0.88), vec3(0.94, 1.0, 1.08), h1) * mix(0.88, 1.12, h2);
            albedo = ca.rgb * ueTint(spc.leafTint.rgb) * spc.leafTint.w * hue * treeTint(seed, 1.0);
            albedo = dryTint(albedo, instDry);
            if (underside) albedo *= vec3(1.18, 1.15, 1.25); // paler underside
            // Transmission like Unreal's two-sided foliage: albedo x translucency tint x mask.
            trans = albedo * spc.transTint.rgb * (nrmTex.b * spc.transTint.w);
            extraFlags = underside ? 0u : FLAG_GLOSSY;
        } else {
            albedo = textureLod(matTex[nonuniformEXT(ts + 3u)], uv, lod).rgb * ueTint(spc.barkTint.rgb);
            nrmTex = vec4(textureLod(matTex[nonuniformEXT(ts + 4u)], uv, lod).rg, textureLod(matTex[nonuniformEXT(ts + 5u)], uv, lod).r, 1.0);
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
        uint lr = pcg(leafId * 7919u + instId * 104729u + 17u);
        float h1 = float(lr & 0xFFFFu) / 65535.0, h2 = float(lr >> 16) / 65535.0;
        vec3 hue = mix(vec3(1.15, 1.0, 0.65), vec3(0.86, 1.0, 1.18), h1);
        albedo = base * hue * mix(0.70, 1.28, h2 * h2) * treeTint(seed, 1.0);
        extraFlags = underside ? 0u : FLAG_GLOSSY;
        trans = 0.33 * spc.transTint.rgb * albedo;
    } else {
        albedo = vec3(0.105, 0.082, 0.064);
    }

    pl.albedo = albedo;
    pl.t = tHit;
    pl.n = n;
    pl.ng = ng;
    pl.flags = mat | extraFlags;
    pl.ao = ao;
    pl.sunVis = sunVis;
    pl.shadowMask = 0xFFu;
    pl.trans = trans;
}
