#version 460
#extension GL_GOOGLE_include_directive : require
#include "rt.glsl"
#include "livery.glsl"

// Closest hit for the aircraft of --test-flight (src/aircraft.rs). Its materials are flat colours
// with a metalness and a roughness, except the painted skin: one atlas in which the aircraft is
// drawn from above, from below and from each side (tools/f16_paint.py), as three maps:
//   0 colour (sRGB)   1 occlusion, roughness, metal   2 normal
// main.rgen lights it (FLAG_AIRCRAFT): the payload's trans carries roughness, metal and opacity,
// wtan the emission.
layout(location = 0) rayPayloadInEXT HitPayload pl;
hitAttributeEXT vec2 bary;

layout(set = 0, binding = 3) uniform GlobalsUBO { Globals g; };
layout(set = 0, binding = 4, scalar) readonly buffer Meshes { MeshInfo meshes[]; };
layout(set = 0, binding = 36) uniform sampler2D acTex[3];
struct AcMaterial { vec4 base; vec4 mre; vec4 emit; };
layout(set = 0, binding = 37, scalar) readonly buffer AcMats { AcMaterial acMats[]; };
layout(set = 0, binding = 38, scalar) readonly buffer Crafts { CraftRec crafts[]; };

void main() {
    MeshInfo m = meshes[gl_InstanceCustomIndexEXT & 0x7FFu];
    uvec3 tri = UVec3Ref(m.idx).v[gl_PrimitiveID];
    Vec3Ref P = Vec3Ref(m.pos);
    Vec3Ref N = Vec3Ref(m.nrm);
    vec3 b = vec3(1.0 - bary.x - bary.y, bary.x, bary.y);
    vec3 p0 = P.v[tri.x], p1 = P.v[tri.y], p2 = P.v[tri.z];
    vec3 no = normalize(N.v[tri.x] * b.x + N.v[tri.y] * b.y + N.v[tri.z] * b.z); // the part's own frame
    mat3 o2w = mat3(gl_ObjectToWorldEXT);
    vec3 ng = normalize(o2w * cross(p1 - p0, p2 - p0));

    uint word = UintRef(m.mat).v[gl_PrimitiveID >> 2];
    AcMaterial mt = acMats[(word >> ((gl_PrimitiveID & 3u) * 8u)) & 0xFFu];
    CraftRec craft = crafts[gl_InstanceCustomIndexEXT >> 11];
    // (its squadron's colour on what is not the painted skin: the drones, the trucks)
    vec3 albedo = mt.base.rgb * craft.info.yzw;
    float metal = mt.mre.x, rough = mt.mre.y;
    if (mt.mre.z > 0.5) {
        Vec2Ref T = Vec2Ref(m.uv);
        vec2 uv = T.v[tri.x] * b.x + T.v[tri.y] * b.y + T.v[tri.z] * b.z;
        // the mip for the pixel's footprint: the atlas has 256 texels per metre
        float foot = gl_HitTEXT * g.camFwd.w / max(abs(dot(ng, gl_WorldRayDirectionEXT)), 0.25);
        float lod = max(log2(foot * 256.0), 0.0);
        albedo = textureLod(acTex[0], uv, lod).rgb;
        // (the other two maps are kept at half the size)
        vec3 orm = textureLod(acTex[1], uv, max(lod - 1.0, 0.0)).rgb;
        rough = orm.g;
        metal = orm.b;
        // its squadron's colours (--battle)
        albedo = liveryColour(int(craft.info.x), uv, albedo, metal);
        // The normal map's two axes: along the atlas (aft on the aircraft), and up it: to the
        // right in the drawings from above and below (the atlas's first 5248 rows of 8192,
        // counted from the bottom), upwards in the two from the side.
        vec2 nt = textureLod(acTex[2], uv, max(lod - 1.0, 0.0)).rg * 2.0 - 1.0;
        vec3 tu = vec3(0.0, 0.0, 1.0);
        vec3 tv = (1.0 - uv.y) * 8192.0 < 5248.0 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
        tu -= no * dot(no, tu);
        tv -= no * dot(no, tv);
        no = normalize(no + nt.x * normalize(tu + 1e-5) + nt.y * normalize(tv + 1e-5));
    }

    pl.albedo = albedo;
    pl.t = gl_HitTEXT;
    pl.n = normalize(o2w * no);
    pl.ng = ng;
    pl.flags = MAT_AIRCRAFT | FLAG_AIRCRAFT;
    pl.ao = 1.0;
    pl.sunVis = 1.0;
    pl.shadowMask = 0xFFu;
    pl.trans = vec3(rough, metal, mt.base.a);
    pl.wtan = mt.emit.rgb;
    pl.wdepth = 0.0;
    pl.craft = gl_InstanceCustomIndexEXT >> 11;
}
