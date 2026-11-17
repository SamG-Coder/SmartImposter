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
    float leafiness = smoothstep(0.85, 1.0, alb.g / max(alb.r, 1e-4));
    uint ci = gl_InstanceCustomIndexEXT;

    // Baked self-shadowing: linear directional visibility evaluated at the sun direction in tree space.
    vec4 v = unpackUnorm4x8(hit.vis);
    vec3 Lobj = normalize(mat3(gl_WorldToObjectEXT) * g.sunDir.xyz);
    // The linear fit of a binary visibility function slightly underestimates the sunlit fraction of
    // the outer leaves; 1.1x is calibrated against the real geometry (tools/compare.ps1, --debug 5).
    float selfVis = clamp((v.x + dot((v.yzw - 0.5) * 3.0, Lobj)) * 1.1, 0.0, 1.0);

    pl.albedo = alb * treeTint(instSeed(ci), leafiness);
    pl.t = gl_HitTEXT;
    pl.n = n;
    pl.ng = n;
    pl.flags = FLAG_IMPOSTOR | (gl_HitKindEXT == 1u ? FLAG_IMPOSTOR_HI : 0u) | (leafiness > 0.5 ? (MAT_LEAF | FLAG_GLOSSY) : MAT_BARK);
    // Large-scale sky occlusion from the same baked visibility, cosine-weighted around up (tree
    // space up = world up, trees only rotate about Y): interior leaves see little sky.
    float skyVis = clamp(v.x + (2.0 / 3.0) * (v.z - 0.5) * 3.0, 0.0, 1.0);
    pl.ao = a.a * mix(0.7, 1.0, skyVis); // strength calibrated against mesh AO rays (tools/compare.ps1)
    // Self-shadowing is now traced (dappled, like the real leaves): shadow rays march this tree's
    // own impostor layer by layer. The baked visibility is kept for the sky term only.
    // 1.13: averaged impostor normals correlate less with the sunlit spots than real leaves do
    // (a sunlit leaf usually also faces the sun); calibrated with tools/compare.ps1.
    pl.sunVis = 1.13;
    pl.trans = pl.albedo * imps[ci & 0xFFu].trans.rgb * leafiness; // albedo x species mean transmission
    pl.shadowMask = 0xFFu; // includes this tree: traced self-shadowing
}
