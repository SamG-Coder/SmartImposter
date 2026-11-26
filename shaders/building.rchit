#version 460
#extension GL_GOOGLE_include_directive : require
#include "rt.glsl"

// Closest hit for the procedural buildings (city.rs; their own hit records, renderer.rs
// SBT_BUILDING). Material byte per triangle: low 4 bits the material, high 4 its palette entry.
// Texture coordinates are facade cells (x the window column, y the storey), so every window can
// be told apart: blinds, the glass's own tint, its reflectivity. Nothing here depends on the
// instance, so the impostor bake sees exactly what the mesh shows.
//
// Glass: the albedo is the tint its coating reflects, payload trans.z how much of the surface is
// glass (main.rgen: dark interior, traced Fresnel reflection, GGX sun highlight). The bake keeps
// no channel for it: it is stored in the length of the baked normal (bake.rgen's "normal spread",
// which is ~1 on flat facades anyway): n x (1 - 0.5 glass), read back by impostor.rchit.
layout(location = 0) rayPayloadInEXT HitPayload pl;
hitAttributeEXT vec2 bary;

layout(set = 0, binding = 3) uniform GlobalsUBO { Globals g; };
layout(set = 0, binding = 4, scalar) readonly buffer Meshes { MeshInfo meshes[]; };

const vec3 B_WALL[9] = vec3[9](vec3(0.83, 0.79, 0.69), vec3(0.87, 0.75, 0.55), vec3(0.82, 0.75, 0.60), vec3(0.79, 0.65, 0.48),
                               vec3(0.91, 0.71, 0.58), vec3(0.91, 0.81, 0.56), vec3(0.85, 0.81, 0.70), vec3(0.81, 0.69, 0.54), vec3(0.69, 0.62, 0.51));
const vec3 B_GLASS[8] = vec3[8](vec3(0.32, 0.50, 0.52), vec3(0.24, 0.36, 0.52), vec3(0.36, 0.38, 0.40), vec3(0.48, 0.56, 0.58),
                                vec3(0.60, 0.63, 0.66), vec3(0.34, 0.58, 0.62), vec3(0.20, 0.22, 0.24), vec3(0.32, 0.48, 0.40));
const vec3 B_FRAME[4] = vec3[4](vec3(0.10, 0.105, 0.11), vec3(0.46, 0.47, 0.48), vec3(0.76, 0.76, 0.74), vec3(0.04, 0.04, 0.045));

// A building's own wall colour (its record's seed: the same in its mesh and its impostor, kept
// in step with impostor.rchit): a palette of whites, creams and pastels; seed 0 (the bake) none.
vec3 buildingTint(uint seed) {
    if (seed == 0u) return vec3(1.0);
    const vec3 T[8] = vec3[8](vec3(0.97, 0.96, 0.93), vec3(1.0, 0.82, 0.58), vec3(1.0, 0.72, 0.58), vec3(0.86, 0.73, 0.56),
                              vec3(0.78, 0.92, 0.78), vec3(1.0, 0.91, 0.64), vec3(0.74, 0.67, 0.58), vec3(0.98, 0.70, 0.70));
    return T[pcg(seed * 747796405u) & 7u];
}

// A car's paint by its seed (keep in step with impostor.rchit): mostly white, black, silver and
// grey, some red, blue, dark blue, beige; seed 0 (the bake) white.
vec3 carPaint(uint seed) {
    if (seed == 0u) return vec3(1.0);
    const vec3 C[12] = vec3[12](vec3(1.0), vec3(1.0), vec3(0.06), vec3(0.06), vec3(0.62), vec3(0.62), vec3(0.32), vec3(0.65, 0.06, 0.05),
                                vec3(0.08, 0.16, 0.45), vec3(0.05, 0.07, 0.18), vec3(0.75, 0.66, 0.52), vec3(0.15));
    return C[pcg(seed * 2891336453u) % 12u];
}

float bHash(vec2 c, uint k) { return float(pcg(uint(int(c.x) + 4096) * 7919u ^ uint(int(c.y) + 4096) * 104729u ^ k) & 0xFFFFu) / 65535.0; }

void main() {
    uint meshId = gl_InstanceCustomIndexEXT & 0x7FFu;
    MeshInfo m = meshes[meshId];
    uvec3 tri = UVec3Ref(m.idx).v[gl_PrimitiveID];
    Vec3Ref P = Vec3Ref(m.pos);
    Vec3Ref N = Vec3Ref(m.nrm);
    vec3 b = vec3(1.0 - bary.x - bary.y, bary.x, bary.y);
    vec3 p0 = P.v[tri.x], p1 = P.v[tri.y], p2 = P.v[tri.z];
    vec3 po = p0 * b.x + p1 * b.y + p2 * b.z;
    mat3 o2w = mat3(gl_ObjectToWorldEXT);
    vec3 e1 = o2w * (p1 - p0), e2 = o2w * (p2 - p0);
    vec3 ng = normalize(cross(e1, e2));
    vec3 n = normalize(o2w * N.v[tri.x]);
    uint word = UintRef(m.mat).v[gl_PrimitiveID >> 2];
    uint mbyte = (word >> ((gl_PrimitiveID & 3u) * 8u)) & 0xFFu;
    uint mt = mbyte & 15u, pal = mbyte >> 4;

    Vec2Ref UV = Vec2Ref(m.uv);
    vec2 t0 = UV.v[tri.x], t1 = UV.v[tri.y], t2 = UV.v[tri.z];
    vec2 uv = t0 * b.x + t1 * b.y + t2 * b.z;
    // how big the pixel is in cells (ray cone; the bake has a fixed footprint): details finer
    // than that fade to their average
    vec2 d1 = t1 - t0, d2 = t2 - t0;
    float uvArea = abs(d1.x * d2.y - d1.y * d2.x), worldArea = max(length(cross(e1, e2)), 1e-8);
    float cellsPerM = sqrt(uvArea / worldArea);
    float spread = g.camFwd.w;
    float footprint = spread > 0.0 ? gl_HitTEXT * spread : -spread;
    float fpc = footprint * cellsPerM; // pixel size in cells
    float detail = 1.0 - smoothstep(0.12, 0.45, fpc);
    vec2 cell = floor(uv), f = uv - cell;
    float r1 = bHash(cell, meshId * 31u + 1u), r2 = bHash(cell, meshId * 31u + 2u);
    // large-scale weathering, in the mesh's own metres
    float grime = mix(0.9, 1.04, vnoise(po.xz * 0.11 + po.y * 0.07)) * mix(0.94, 1.0, vnoise(vec2(po.y * 0.35, po.x * 0.21 + po.z * 0.17)));
    vec3 albedo;
    float glass = 0.0, coat = 0.0;
    vec3 emit = vec3(0.0); // light of its own at dusk (main.rgen scales it by the dark)
    if (mt == 1u || mt == 7u) {
        // glass: window by window a little different (coating, blinds behind it)
        vec3 tint = B_GLASS[min(pal, 7u)];
        albedo = tint * mix(1.0, mix(0.86, 1.12, r1), detail);
        glass = mt == 7u ? 0.45 : mix(1.0, mix(0.8, 1.0, r2), detail);
        if (mt == 7u) albedo = mix(albedo, vec3(0.62, 0.66, 0.66), 0.45); // rails: frosted, lighter
        if (mt == 1u && pal != 6u) {
            // (plain glass, pal 6: cars' and boats' windows, has no facade cells and no frames)
            // mullions at the cell edges and a spandrel band at the foot of each storey (opaque
            // glass): drawn where the pixel resolves them, their average beyond
            float mw = clamp(0.05 * cellsPerM, 0.01, 0.12);
            float mull = 1.0 - smoothstep(mw * 0.6, mw, min(f.x, 1.0 - f.x));
            float span = 1.0 - smoothstep(0.17, 0.2, f.y);
            vec3 frameC = B_FRAME[pal & 3u];
            float fm = mix(mw * 1.6, mull, detail), fs = mix(0.18, span, detail);
            albedo = mix(albedo, tint * 0.55, fs * 0.7);
            glass = mix(glass, 0.75, fs);
            albedo = mix(albedo, frameC, fm);
            glass = mix(glass, 0.15, fm);
            // a lit room behind about two windows in five, warm or cool, some dimmer; its
            // average where the pixel holds many windows
            float lit = mix(0.2, step(bHash(cell, meshId * 31u + 3u), 0.2), detail);
            vec3 room = mix(vec3(1.0, 0.56, 0.24), vec3(0.85, 0.9, 1.0), step(0.88, r1)) * mix(0.5, 1.0, r2);
            emit = room * lit * (1.0 - fm) * (1.0 - fs);
        }
    } else if (mt == 0u || mt == 10u) {
        albedo = B_WALL[min(pal, 8u)] * grime * (mt == 10u ? 0.92 : 1.0) * buildingTint(instSeed(gl_InstanceCustomIndexEXT));
        // weathering: dirt splashed up the foot of the walls, rain streaks down them
        albedo *= mix(0.8, 1.0, smoothstep(0.0, 2.5, po.y));
        albedo *= mix(1.0, mix(0.86, 1.0, vnoise(vec2((po.x + po.z) * 2.3, po.y * 0.07))), 0.8 * detail);
    } else if (mt == 2u) {
        albedo = B_FRAME[min(pal, 3u)];
        glass = 0.25;
    } else if (mt == 3u) {
        albedo = vec3(0.74, 0.73, 0.70) * grime;
    } else if (mt == 4u) {
        // flat roofs: membrane and gravel, stained
        float nr = vnoise(po.xz * 0.6) * 0.6 + vnoise(po.xz * 2.7) * 0.4;
        albedo = vec3(0.38, 0.38, 0.37) * mix(0.85, 1.1, nr) * mix(0.9, 1.06, vnoise(po.xz * 0.05 + 3.0));
        // each building's own membrane: grey, beige, tan, dark, pale (impostor.rchit the same)
        uint rs = instSeed(gl_InstanceCustomIndexEXT);
        const vec3 RT[6] = vec3[6](vec3(1.0), vec3(1.1, 1.02, 0.86), vec3(0.98, 0.88, 0.72), vec3(0.55, 0.56, 0.58), vec3(1.22, 1.2, 1.16), vec3(0.8, 0.8, 0.8));
        if (rs != 0u) albedo *= RT[pcg(rs * 1597334677u) % 6u];
    } else if (mt == 5u) {
        // clay roof tiles: courses down the slope, every tile its own shade
        vec3 tc = pal == 0u ? vec3(0.46, 0.20, 0.12) : pal == 1u ? vec3(0.54, 0.31, 0.18) : pal == 2u ? vec3(0.40, 0.24, 0.17) : pal == 3u ? vec3(0.27, 0.28, 0.29) : vec3(0.56, 0.45, 0.36); // (4: faded sun-bleached clay)
        vec2 q = vec2(po.x + po.z, po.y) * vec2(3.2, 4.0);
        float course = smoothstep(0.0, 0.25, fract(q.y));
        albedo = tc * mix(1.0, mix(0.75, 1.15, bHash(floor(q), 7u)) * mix(0.8, 1.0, course), detail) * grime;
    } else if (mt == 6u) {
        albedo = (pal == 0u ? vec3(0.42) : pal == 1u ? vec3(0.58, 0.58, 0.56) : vec3(0.30, 0.31, 0.32)) * grime;
        glass = 0.1;
    } else if (mt == 8u) {
        albedo = B_WALL[min(pal, 8u)] * 0.9 * grime;
    } else if (mt == 11u) {
        // timber: decking planks (teak on boats, weathered grey-brown on docks), piles darker
        vec3 wc = pal == 0u ? vec3(0.34, 0.24, 0.15) : vec3(0.16, 0.13, 0.10);
        float plank = smoothstep(0.0, 0.08, min(fract(po.z / 0.16), 1.0 - fract(po.z / 0.16)));
        albedo = wc * mix(0.82, 1.1, vnoise(vec2(po.x * 0.4, floor(po.z / 0.16)) * 3.1)) * mix(1.0, mix(0.7, 1.0, plank), detail);
    } else if (mt == 12u) {
        // gelcoat and paint: glossy
        const vec3 PAINTC[5] = vec3[5](vec3(0.03, 0.05, 0.13), vec3(0.36, 0.05, 0.04), vec3(0.04, 0.22, 0.24), vec3(0.02, 0.02, 0.025), vec3(0.86, 0.86, 0.84));
        albedo = PAINTC[min(pal, 4u)];
        glass = 0.3;
        if (pal == 5u) {
            // a clipped hedge: dense small leaves, matte, uneven dark green
            float hn = vnoise(po.xz * 2.3 + po.y * 1.7) * 0.6 + vnoise(po.zy * 6.0 + po.x * 3.0) * 0.4;
            albedo = vec3(0.045, 0.075, 0.028) * mix(0.6, 1.3, hn);
            glass = 0.0;
        }
    } else if (mt == 14u) {
        // car paint: each car's own colour (its record's seed; white in the bake, the impostor
        // applies the same colour: impostor.rchit carPaint), glossy
        albedo = vec3(0.85) * carPaint(instSeed(gl_InstanceCustomIndexEXT));
        // panel shading: the sills and the lower doors a shade darker (road grime, the body's
        // turn under), a shut line along the bonnet and the roof's edge
        albedo *= mix(0.72, 1.0, smoothstep(0.3, 0.75, po.y));

        albedo *= mix(1.0, 0.8, (1.0 - smoothstep(0.0, 0.04, abs(abs(po.z) - 0.55))) * step(0.9, po.y) * detail);
        coat = 1.0;
    } else if (mt == 15u) {
        // a bridge's roadway: asphalt with its centre line
        albedo = vec3(0.075, 0.075, 0.078) * mix(0.85, 1.15, vnoise(po.xz * 0.3));
        float cl = (1.0 - smoothstep(0.1, 0.14, abs(po.z))) * step(fract(po.x / 9.0), pal == 0u ? 1.0 : 0.4) * detail;
        if (pal == 2u) {
            // the highway: dashed lanes either side of the median, solid edge lines
            float az = abs(po.z);
            float dash = step(fract(po.x / 12.0), 0.3);
            cl = max(max(1.0 - smoothstep(0.08, 0.12, abs(az - 4.0)), 1.0 - smoothstep(0.08, 0.12, abs(az - 7.6))) * dash,
                     max(1.0 - smoothstep(0.08, 0.12, abs(az - 0.6)), 1.0 - smoothstep(0.08, 0.12, abs(az - 13.8)))) * detail;
        } else if (pal == 3u) {
            cl = (1.0 - smoothstep(0.08, 0.12, abs(abs(po.z) - 4.6))) * detail;
        }
        albedo = mix(albedo, pal == 0u ? vec3(0.55, 0.42, 0.10) : vec3(0.6), cl);
    } else if (mt == 13u && pal >= 2u) {
        // lamps: a street light's head, headlights, taillights
        albedo = pal == 4u ? vec3(0.4, 0.03, 0.02) : vec3(0.8);
        glass = 0.2;
        emit = pal == 2u ? vec3(1.0, 0.8, 0.55) * 40.0 : pal == 3u ? vec3(1.0, 0.95, 0.85) * 60.0 : vec3(1.0, 0.08, 0.04) * 20.0;
    } else if (mt == 13u && pal == 1u) {
        // a wake's foam: bright where it is fresh, a broken lace of it
        albedo = vec3(0.78) * mix(0.55, 1.0, vnoise(po.xz * 1.7));
    } else if (mt == 13u) {
        // a rooftop pool's water
        albedo = vec3(0.10, 0.42, 0.48);
        glass = 0.6;
    } else {
        albedo = vec3(0.025, 0.026, 0.028);
    }
    bool bake = g.lodParams.z < -0.5;
    pl.albedo = albedo;
    pl.t = gl_HitTEXT;
    pl.n = bake ? n * (1.0 - 0.5 * glass) : n;
    pl.ng = ng;
    pl.flags = MAT_BUILDING | FLAG_BUILDING;
    pl.ao = 1.0;
    pl.sunVis = 1.0;
    pl.shadowMask = 0xFFu;
    // (x: a clearcoat over the paint, main.rgen)
    // (y: plain tinted glass, cars' and boats' windows: F0 0.04, not a coated curtain wall)
    pl.trans = vec3(coat, (mt == 1u && pal == 6u) ? 1.0 : 0.0, glass);
    pl.wtan = emit;
}
