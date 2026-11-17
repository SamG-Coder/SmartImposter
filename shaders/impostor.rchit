#version 460
#extension GL_GOOGLE_include_directive : require
#include "rt.glsl"

layout(location = 0) rayPayloadInEXT HitPayload pl;
hitAttributeEXT ImpHit hit;

layout(set = 0, binding = 3) uniform GlobalsUBO { Globals g; };
layout(set = 0, binding = 5, scalar) readonly buffer Imps { ImpostorInfo imps[]; };

void main() {
    vec4 a = unpackUnorm4x8(hit.albedoAo);
    vec3 alb = a.rgb * a.rgb; // atlas stores sqrt(albedo) for precision
    vec3 n = normalize(mat3(gl_ObjectToWorldEXT) * octDecode(unpackSnorm2x16(hit.normal)));
    // Bark is grey-brown (g/r ~ 0.78); every leaf variant (incl. pale, yellow-tinted undersides) has g/r > 1.
    uint ci = gl_InstanceCustomIndexEXT;
    // Grass and ferns have no bark (bmin.w = 1): pale / dry blades must not be classified as bark.
    float leafiness = imps[ci & 0xFFu].bmin.w > 0.5 ? 1.0 : smoothstep(0.85, 1.0, alb.g / max(alb.r, 1e-4));

    // Baked visibility: x = a, y = length of the averaged normal, z = b.y (see bake.rgen).
    vec4 v = unpackUnorm4x8(hit.vis);
    bool foliageOnly = imps[ci & 0xFFu].bmin.w > 0.5;

    pl.albedo = alb * treeTint(instSeed(ci), leafiness);
    pl.t = gl_HitTEXT;
    // Normal spread (length of the averaged baked normal, bake.rgen). Trees: the normal is
    // shortened to it, so main.rgen's Lambert term gives the mean diffuse over the leaves while
    // transmission keeps its full strength (leaves face every way: back-lit ones always exist).
    // Grass / ferns: blade normals lean up (tools/grass_to_mtree.py), so seen from above hardly any
    // blade is back-lit either: the spread scales all sunlight (pl.sunVis below).
    float spread = clamp(v.y, 0.2, 1.0);
    pl.n = foliageOnly ? n : n * spread;
    pl.ng = n;
    pl.flags = FLAG_IMPOSTOR | (gl_HitKindEXT == 1u ? FLAG_IMPOSTOR_HI : 0u) | (leafiness > 0.5 ? (MAT_LEAF | FLAG_GLOSSY) : MAT_BARK);
    // Large-scale sky occlusion from the same baked visibility, cosine-weighted around up (tree
    // space up = world up, trees only rotate about Y): interior leaves see little sky.
    float skyVis = clamp(v.x + (2.0 / 3.0) * (v.z - 0.5) * 3.0, 0.0, 1.0); // a + 2/3 b.y
    pl.ao = a.a * mix(0.7, 1.0, skyVis); // strength calibrated against mesh AO rays (tools/compare.ps1)
    // Self-shadowing is now traced (dappled, like the real leaves): shadow rays march this tree's
    // own impostor layer by layer. The baked visibility is kept for the sky term only.
    // 1.13: averaged impostor normals correlate less with the sunlit spots than real leaves do
    // (a sunlit leaf usually also faces the sun); calibrated with tools/compare.ps1.
    // No calibration constant: the normal spread (pl.n length) and the traced layered shadows
    // account for crown / sward self-shadowing.
    pl.sunVis = foliageOnly ? spread : 1.0;
    pl.trans = pl.albedo * imps[ci & 0xFFu].trans.rgb * leafiness; // albedo x species mean transmission
    pl.shadowMask = 0xFFu; // includes this tree: traced self-shadowing
}
