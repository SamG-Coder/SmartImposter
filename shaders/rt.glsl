// Ray tracing declarations shared by every RT stage.
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_nonuniform_qualifier : require

#include "shared.glsl"

struct HitPayload {
    vec3 albedo;
    float t;      // < 0 on miss
    vec3 n;       // shading normal (world)
    uint flags;   // material id in low byte, FLAG_IMPOSTOR
    vec3 ng;      // geometric normal (world)
    float ao;     // baked / precomputed occlusion
    float sunVis; // baked self-shadowing towards the sun (impostors), 1 for meshes
    uint shadowMask; // instance mask for this hit's shadow rays (impostors exclude themselves)
    float trans;  // leaf translucency (texture), 0 for opaque materials
};

// Impostor hit attributes passed from the intersection to the closest-hit shader.
struct ImpHit {
    uint albedoAo; // unorm4x8: sqrt(albedo).rgb, ao
    uint normal;   // snorm2x16 octahedral, object space
    uint vis;      // unorm4x8: directional self-visibility a, b/3 + 0.5
};

// Custom index layout: low 8 bits mesh/variant id, bits 8..10 impostor mask colour, bits 11..23 seed.
uint instSeed(uint ci) { return ci >> 11; }
uint instColor(uint ci) { return (ci >> 8) & 7u; }

struct MeshInfo { uint64_t pos; uint64_t nrm; uint64_t idx; uint64_t mat; uint64_t uv; };

struct ImpostorInfo {
    vec4 centerRadius; // bounding sphere in tree space
    vec4 bmin;
    vec4 bmax;
    uint grid;         // frames per side
    uint res;          // texels per frame side
    uint mips;
    uint pad;
};

#include "globals.glsl"

// Per TLAS instance (indexed by gl_InstanceID): the tree it belongs to, for LOD cross-fades.
struct InstInfo {
    vec3 center;  // tree bounding-sphere centre (world)
    uint data;    // tree index (24 bits) | flags << 24 (bit 0: tree also has mesh instances)
};

// Cross-fade weights. Inside a band, every ray picks one representation per tree with this
// probability, using the same random number in the mesh any-hit and the impostor intersection
// shader, so exactly one copy of the tree is ever visible (no holes, no doubled shadows).
float bandFade(float centre, float width, float d) {
    float x = clamp((centre + 0.5 * width - d) / max(width, 1e-3), 0.0, 1.0);
    return x * x * (3.0 - 2.0 * x);
}
float meshFade(Globals g, float d) { return bandFade(g.camPos.w, g.lodParams.x, d); }      // 1 = mesh
float hiFade(Globals g, float d) { return g.lodParams.y > 0.0 ? bandFade(g.lodParams.y, g.lodParams.z, d) : 0.0; } // 1 = hi tier
float lodRand(Globals g, uint tree, uint channel) {
    uint s = hash2(gl_LaunchIDEXT.xy) ^ pcg(g.iparams.z * 2654435761u + tree * 97u + channel);
    return rnd(s);
}

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer Vec3Ref { vec3 v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer UVec3Ref { uvec3 v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer Vec2Ref { vec2 v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer UintRef { uint v[]; };
