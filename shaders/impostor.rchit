#version 460
#extension GL_GOOGLE_include_directive : require
#include "rt.glsl"

layout(location = 0) rayPayloadInEXT HitPayload pl;
hitAttributeEXT ImpHit hit;

layout(set = 0, binding = 3) uniform GlobalsUBO { Globals g; };
layout(set = 0, binding = 5, scalar) readonly buffer Imps { ImpostorInfo imps[]; };
layout(set = 0, binding = 23, scalar) readonly buffer Plants { PlantRec plants[]; };

void main() {
    PlantRec pr = plants[gl_InstanceCustomIndexEXT + gl_PrimitiveID];
    uint vid = plantVariant(pr);
    vec4 a = unpackUnorm4x8(hit.albedoAo);
    vec3 alb = a.rgb * a.rgb; // atlas stores sqrt(albedo) for precision
    // plant -> instance space -> world (globe tiles rotate into the camera frame; identity in the forest scene)
    vec3 n = normalize(mat3(gl_ObjectToWorldEXT) * (plantMatrix(pr) * octDecode(unpackSnorm2x16(hit.normal))));
    // Grass and ferns have no bark (bmin.w = 1): pale / dry blades must not be classified as bark.
    // Otherwise: bark is grey-brown (g/r ~ 0.78); every leaf variant has g/r > 1.
    bool foliageOnly = imps[vid].bmin.w > 0.5;
    float leafiness = foliageOnly ? 1.0 : smoothstep(0.85, 1.0, alb.g / max(alb.r, 1e-4));

    // Baked visibility: x = a, y = length of the averaged normal, z = b.y (see bake.rgen).
    vec4 v = unpackUnorm4x8(hit.vis);

    pl.albedo = alb * treeTint(plantSeed(pr), leafiness);
    pl.t = gl_HitTEXT;
    // Normal spread (length of the averaged baked normal, bake.rgen). Trees: the normal is
    // shortened to it, so main.rgen's Lambert term gives the mean diffuse over the leaves while
    // transmission keeps its full strength (leaves face every way: back-lit ones always exist).
    // Grass / ferns: blade normals lean up (tools/grass_to_mtree.py), so seen from above hardly any
    // blade is back-lit either: the spread scales all sunlight (pl.sunVis below).
    float spread = clamp(v.y, 0.2, 1.0);
    pl.n = foliageOnly ? n : n * spread;
    pl.ng = n;
    pl.flags = FLAG_IMPOSTOR | (leafiness > 0.5 ? (MAT_LEAF | FLAG_GLOSSY) : MAT_BARK);
    // Large-scale sky occlusion from the baked visibility, cosine-weighted around up (plants only
    // rotate about Y, or tilt slightly with the slope): interior leaves see little sky.
    float skyVis = clamp(v.x + (2.0 / 3.0) * (v.z - 0.5) * 3.0, 0.0, 1.0); // a + 2/3 b.y
    pl.ao = a.a * mix(0.7, 1.0, skyVis);
    // Self-shadowing is traced (shadow rays march this plant's own impostor layer by layer).
    pl.sunVis = foliageOnly ? spread : 1.0;
    pl.trans = pl.albedo * imps[vid].trans.rgb * leafiness; // albedo x species mean transmission
    pl.shadowMask = 0xFFu;
}
