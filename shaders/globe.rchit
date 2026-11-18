#version 460
#extension GL_GOOGLE_include_directive : require
#include "globe.glsl"

layout(set = 0, binding = 2) uniform GG { GlobeGlobals gg; };
layout(set = 0, binding = 3, scalar) readonly buffer TileTable { uint64_t vtx[]; } table;
layout(buffer_reference, scalar) readonly buffer Verts { float v[]; };
layout(buffer_reference, scalar) readonly buffer Indices { uint i[]; };

layout(location = 0) rayPayloadInEXT GlobePayload pl;
hitAttributeEXT vec2 bary;

// tiles.rs vertex: position (3), normal (3), elevation, water
void main() {
    Verts V = Verts(table.vtx[gl_InstanceCustomIndexEXT]);
    Indices I = Indices(packUint2x32(gg.addr.xy));
    uint k = gl_PrimitiveID * 3u;
    uvec3 tri = uvec3(I.i[k], I.i[k + 1u], I.i[k + 2u]);
    vec3 w = vec3(1.0 - bary.x - bary.y, bary.x, bary.y);
    vec3 n = vec3(0.0);
    float elev = 0.0, water = 0.0;
    for (int c = 0; c < 3; ++c) {
        uint b = tri[c] * 8u;
        n += vec3(V.v[b + 3u], V.v[b + 4u], V.v[b + 5u]) * w[c];
        elev += V.v[b + 6u] * w[c];
        water += V.v[b + 7u] * w[c];
    }
    pl.n = normalize(n);
    pl.t = gl_HitTEXT;
    pl.elev = elev;
    pl.water = water;
}
