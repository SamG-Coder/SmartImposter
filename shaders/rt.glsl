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
    vec3 trans;   // leaf transmission colour (translucency map x species tint), 0 for opaque
    float wdepth; // terrain under water: depth of the water over this point (m), else 0
    vec3 wtan;    // globe water: world direction of the first Earth-fixed coordinate (pl.trans.xy)
    float wlevel; // globe water: height of the water surface above this point (m)
    float wfloor; // globe water: depth of the sea floor (m, ETOPO; coastal vs open-ocean water)
    uint craft;   // an aircraft hit (FLAG_AIRCRAFT): which one (CraftRec)
};

// Each aircraft or rocket in the air (renderer.rs CraftRec, binding 38; the first is the one
// flown): where a point of it now was a frame ago (rows; motion vectors), and its paint scheme.
struct CraftRec { vec4 back0; vec4 back1; vec4 back2; vec4 info; }; // info: x paint scheme, yzw its squadron's colour

// Impostor hit attributes passed from the intersection to the closest-hit shader.
struct ImpHit {
    uint albedoAo; // unorm4x8: sqrt(albedo).rgb, ao
    uint normal;   // snorm2x16 octahedral, object space
    uint vis;      // unorm4x8: directional self-visibility a, b/3 + 0.5
};

// Custom index layout: low 8 bits mesh/variant id, bits 8..10 impostor mask colour, bits 11..23 seed
// (an aircraft's parts: the mesh in the low 11 bits, which aircraft above them; the smoke and
// fire of --battle: all of it the index into their list, fx.glsl).
// Instance masks: 1 meshes and terrain, 2 impostors, 4 the aircraft flown, 8 smoke and fire (seen
// by the ray that gathers them only: every other ray leaves that bit out), 16 the other aircraft.
uint instSeed(uint ci) { return ci >> 11; }

// Every plant's impostor is one AABB primitive of a cluster BLAS; its record (renderer.rs
// PlantRecGpu) is at gl_InstanceCustomIndexEXT + gl_PrimitiveID: object->world 3x4 (rows),
// variant | seed << 8. 52 bytes, packed (the buffers use scalar layout). The matrix is stored as
// it is used: impostor.rint runs for every box a ray stops at, and a more compact record (a
// quaternion, 32 bytes) measurably slows that down for the arithmetic of unpacking it.
struct PlantRec { vec4 r0; vec4 r1; vec4 r2; uint data; };
mat3 plantMatrix(PlantRec p) { return transpose(mat3(p.r0.xyz, p.r1.xyz, p.r2.xyz)); }
vec3 plantOrigin(PlantRec p) { return vec3(p.r0.w, p.r1.w, p.r2.w); }
uint plantVariant(PlantRec p) { return p.data & 0xFFu; }
uint plantSeed(PlantRec p) { return (p.data >> 8) & 0x1FFFu; }
// How dry the plant's biome is, 0..1 (forest.rs dryness): dusty, sun-bleached foliage.
float plantDry(PlantRec p) { return float((p.data >> 21) & 15u) / 15.0; }
// Globe tree tiles: how many levels coarser than the finest the plant's tile is (forest.rs: such a
// tile keeps 1 / 4^k of the trees).
uint plantThin(PlantRec p) { return (p.data >> 25) & 7u; }

// The tree density the globe aims for at distance d, as levels of thinning (continuous): tiles k
// levels coarser cover [THIN_BASE 2^k, THIN_BASE 2^(k+1)) beyond THIN_NEAR (forest.rs
// tiles_near: SPLIT x the finest tile edge; NEAR), full density closer in. Trees fade out within
// their tile towards the next level's density, so neighbouring tiles meet without a step; the
// terrain's canopy colour (globe.rchit) follows the same curve.
const float THIN_BASE = 2443.4;  // forest.rs SPLIT x the level-15 tile edge, m
const float THIN_NEAR = 6000.0;  // forest.rs NEAR
float thinLevels(float d) {
    return max(log2(d / THIN_BASE), 0.0) * smoothstep(0.5 * THIN_NEAR, THIN_NEAR, d);
}
uint instColor(uint ci) { return (ci >> 8) & 7u; }

struct MeshInfo { uint64_t pos; uint64_t nrm; uint64_t idx; uint64_t mat; uint64_t uv; uint texSet; uint flags; }; // flags: 1 = alpha-tested (opacity in the leaf texture alpha); >> 8: a grid mesh's vertices per side (meshTri)

// Per plant variant (= atlas slot).
struct ImpostorInfo {
    vec4 centerRadius; // bounding sphere in tree space
    vec4 bmin;         // w: species index
    vec4 bmax;
    vec4 trans;        // rgb: mean leaf transmission colour of the species (for impostor hits)
    uint grid;         // frames per side
    uint res;          // texels per frame: across | up << 16
    uint mips;
    float lodScale;    // LOD distances scale with plant size (trees 1, shrubs/ferns smaller)
    vec4 frame;        // xy: half the width and height of a frame (m), around the bounding sphere's centre
};

// Per species: Unreal material tints (0.5 = neutral) and translucency colour.
struct SpeciesInfo { vec4 leafTint; vec4 barkTint; vec4 transTint; };

#include "globals.glsl"

// Per TLAS instance (indexed by gl_InstanceID): the tree it belongs to, for LOD cross-fades.
struct InstInfo {
    vec3 center;  // tree bounding-sphere centre (world)
    uint data;    // tree index (24 bits) | variant (bits 24..31)
    uint flags;   // bit 0: the tree also has mesh instances (renderer.rs INST_HAS_MESH); bits 1..4: dryness; from bit 5: which part of the plant
};
uint instVariant(uint data) { return data >> 24; }

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

// The vertices of triangle `prim` of a mesh. A grid mesh (the forest scene's terrain, world.rs
// World::mesh: n x n vertices, two triangles per cell, in rows) has no index buffer.
uvec3 meshTri(MeshInfo m, uint prim) {
    uint n = m.flags >> 8;
    if (n == 0u) return UVec3Ref(m.idx).v[prim];
    uint q = prim >> 1;
    uint a = q + q / (n - 1u); // row j = q / (n - 1), vertex j n + i
    return (prim & 1u) == 0u ? uvec3(a, a + n, a + 1u) : uvec3(a + 1u, a + n, a + n + 1u);
}
