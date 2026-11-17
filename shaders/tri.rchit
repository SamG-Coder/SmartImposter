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
//   0 forest floor  1 mossy grass  2 uncut grass  3 dried grass.  Slot MAX_GROUND_TEX-1 = biome
//   mask over the terrain (R canopy cover, G moisture, B dryness, A understory), see World::ground_mask.
layout(set = 0, binding = 16) uniform sampler2D groundTex[MAX_GROUND_TEX];
const float GROUND_TILE = 2.2; // metres per texture repeat
// The scans' mean albedos differ ~10x (moss 0.02 .. dried grass 0.19); bring them to 0.05..0.1.
const vec4 GROUND_GAIN = vec4(0.7, 2.4, 0.68, 0.5);

// Blended ground material at world position p. lodBias: ray-cone mip for 1 m texel spacing.
void groundMaterial(vec3 p, inout vec3 n, float footprint, out vec3 albedo, out float ao) {
    vec4 mk = textureLod(groundTex[MAX_GROUND_TEX - 1], (p.xz + TERRAIN_SIZE * 0.5) / TERRAIN_SIZE, 0.0);
    float cover = mk.r, wet = mk.g, dry = mk.b, under = mk.a;
    float n1 = vnoise(p.xz * 0.13), n2 = vnoise(p.xz * 0.027 + 17.0);
    float slope = 1.0 - n.y;
    // Leaf litter only under closed canopy (and on steep banks); partial cover = mossy grass.
    float wFloor = smoothstep(0.3, 0.65, cover + 0.2 * under + (n1 - 0.5) * 0.35 + slope * 1.0);
    float open = 1.0 - wFloor;
    float wMoss = (smoothstep(0.4, 0.8, wet + (n2 - 0.5) * 0.5) + 0.8 * smoothstep(0.15, 0.6, cover)) * (open + 0.4 * wFloor);
    float wDry = open * smoothstep(0.45, 0.85, dry + (n2 - 0.5) * 0.35) * (1.0 - 0.7 * wet);
    float wGrass = open * (1.0 - 0.6 * smoothstep(0.4, 0.8, wet)) * (1.0 - smoothstep(0.35, 0.75, dry));
    vec4 w = vec4(wFloor, wMoss, wGrass, wDry) + 1e-4;

    // Mip from the ray cone, widened at grazing angles (geometric mean of the footprint axes).
    vec3 V = -gl_WorldRayDirectionEXT;
    float cosv = max(abs(dot(n, V)), 0.03);
    float texel = GROUND_TILE / float(textureSize(groundTex[0], 0).x);
    float lod = max(log2(footprint / texel) - 0.5 * log2(cosv), 0.0);
    vec2 uv = p.xz / GROUND_TILE;
    // Anti-tiling: a second, rotated, larger-scale albedo sample mixed in by low-frequency noise.
    vec2 uv2 = mat2(0.8, -0.6, 0.6, 0.8) * p.xz / (GROUND_TILE * 3.3) + 0.37;
    float m2 = 0.45 * smoothstep(0.25, 0.75, vnoise(p.xz * 0.045));

    vec3 alb[4];
    vec4 nrm[4];
    vec4 orm[4];
    float h[4];
    for (int l = 0; l < 4; ++l) {
        if (w[l] < 0.02) { alb[l] = vec3(0.0); nrm[l] = vec4(0.5, 0.5, 1.0, 0.0); orm[l] = vec4(1.0); h[l] = 0.0; continue; }
        alb[l] = mix(textureLod(groundTex[l * 3], uv, lod).rgb, textureLod(groundTex[l * 3], uv2, lod + 1.7).rgb, m2);
        nrm[l] = textureLod(groundTex[l * 3 + 1], uv, lod);
        orm[l] = textureLod(groundTex[l * 3 + 2], uv, lod);
        h[l] = dot(alb[l], vec3(0.3, 0.5, 0.2)) * 3.0 + orm[l].r * 0.5; // brighter / less occluded = higher
    }
    // Height-based blend: the taller layer wins near the transition (grass blades over litter).
    vec4 hw = w * (vec4(h[0], h[1], h[2], h[3]) + 0.4);
    hw = hw * hw * hw;
    hw /= dot(hw, vec4(1.0));
    albedo = vec3(0.0);
    vec2 nxy = vec2(0.0);
    ao = 0.0;
    for (int l = 0; l < 4; ++l) {
        albedo += alb[l] * hw[l] * GROUND_GAIN[l];
        nxy += (nrm[l].rg * 2.0 - 1.0) * hw[l];
        ao += orm[l].r * hw[l];
    }
    albedo *= mix(0.85, 1.15, n1); // mild macro variation
    ao = mix(1.0, ao, 0.85);
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
        if (g.lodParams.w >= 4.0 && spread > 0.0) groundMaterial(wp, n, gl_HitTEXT * spread, albedo, ao);
        else albedo = terrainAlbedo(wp, n);
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
