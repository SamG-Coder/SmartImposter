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
layout(set = 0, binding = 13) uniform sampler2D matTex[4];

const float TEX_RES = 4096.0;
// Material instance parameters (MI_Black_Alder_*): Unreal's 0.5 = neutral tint.
const vec3 BARK_TINT = vec3(0.4406, 0.4933, 0.5) * 2.0;
const vec3 LEAF_TINT = vec3(0.4922, 0.4988, 0.5) * 2.0;

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
    uint meshId = ci & 0xFFu;
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

    float ao = 1.0, trans = 0.0;
    vec3 albedo;
    if (mat == MAT_TERRAIN) {
        vec3 wp = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
        albedo = terrainAlbedo(wp, n);
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
        float lod = log2(max(footprint, 1e-6)) + 0.5 * log2(max(uvArea, 1e-14) / max(worldArea, 1e-14)) + log2(TEX_RES);
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
            vec4 ca = textureLod(matTex[0], uv, lod);
            nrmTex = textureLod(matTex[1], uv, lod);
            trans = nrmTex.b;
            // Subtle per-leaf variation on top of the texture (texture already has per-leaf colour).
            uint lr = pcg(leafId * 7919u + uint(gl_InstanceID) * 104729u + 17u);
            float h1 = float(lr & 0xFFFFu) / 65535.0, h2 = float(lr >> 16) / 65535.0;
            vec3 hue = mix(vec3(1.06, 1.0, 0.88), vec3(0.94, 1.0, 1.08), h1) * mix(0.88, 1.12, h2);
            albedo = ca.rgb * LEAF_TINT * hue * treeTint(seed, 1.0);
            if (underside) albedo *= vec3(1.18, 1.15, 1.25); // paler underside
            extraFlags = underside ? 0u : FLAG_GLOSSY;
        } else {
            albedo = textureLod(matTex[2], uv, lod).rgb * BARK_TINT;
            nrmTex = textureLod(matTex[3], uv, lod);
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
        trans = 0.33;
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
