// The paint schemes of the squadrons of `--battle` (battle.rs): the same aircraft in other colours.
// The skin's atlas is the aircraft drawn from above, from below and from each side at 256 texels
// a metre (tools/f16_paint.py), so a texture coordinate says where on the aircraft a point is,
// and a scheme can be painted there: its camouflage in place of the greys, its emblem on the
// wings and the fin, its colour in the band across the top of the fin. The panel lines, the wear
// and the bare metal of the painted skin are kept: the scheme takes the atlas's shade, not its
// colour. The squadrons are made up, and so are their emblems.
#ifndef LIVERY_GLSL
#define LIVERY_GLSL

struct Scheme {
    vec3 c0, c1, c2; // the camouflage's colours (linear)
    vec3 under;      // the underside
    vec3 markA, markB;
    float size;      // metres across a patch of the camouflage
    int pattern;     // 0 blotches, 1 stripes, 2 splinters
    int emblem;
};

Scheme scheme(int id) {
    switch (id) {
    case 1: // desert
        return Scheme(vec3(0.46, 0.33, 0.18), vec3(0.20, 0.11, 0.055), vec3(0.34, 0.27, 0.17), vec3(0.36, 0.42, 0.50), vec3(0.02, 0.26, 0.28), vec3(0.80), 2.6, 0, 0);
    case 2: // forest
        return Scheme(vec3(0.045, 0.085, 0.035), vec3(0.13, 0.14, 0.055), vec3(0.24, 0.19, 0.10), vec3(0.34, 0.39, 0.42), vec3(0.75, 0.55, 0.03), vec3(0.02), 2.2, 0, 2);
    case 3: // sea
        return Scheme(vec3(0.035, 0.09, 0.21), vec3(0.20, 0.30, 0.40), vec3(0.42, 0.46, 0.50), vec3(0.42, 0.47, 0.52), vec3(0.80), vec3(0.55, 0.04, 0.03), 2.0, 2, 6);
    case 4: // night
        return Scheme(vec3(0.016, 0.018, 0.021), vec3(0.045, 0.048, 0.055), vec3(0.016, 0.018, 0.021), vec3(0.03, 0.032, 0.036), vec3(0.55, 0.03, 0.02), vec3(0.016), 3.0, 0, 0);
    case 5: // arctic
        return Scheme(vec3(0.62, 0.64, 0.66), vec3(0.30, 0.33, 0.36), vec3(0.12, 0.17, 0.24), vec3(0.60, 0.62, 0.64), vec3(0.02), vec3(0.85, 0.30, 0.02), 2.4, 2, 4);
    case 6: // tiger
        return Scheme(vec3(0.62, 0.30, 0.03), vec3(0.015), vec3(0.62, 0.30, 0.03), vec3(0.40, 0.40, 0.38), vec3(0.015), vec3(0.85), 1.1, 1, 4);
    default: // crimson
        return Scheme(vec3(0.33, 0.045, 0.03), vec3(0.24, 0.25, 0.26), vec3(0.07, 0.075, 0.08), vec3(0.30, 0.32, 0.34), vec3(0.85), vec3(0.02, 0.05, 0.25), 2.0, 2, 1);
    }
}

float lvHash(ivec3 p) {
    uint h = uint(p.x) * 0x8da6b343u ^ uint(p.y) * 0xd8163841u ^ uint(p.z) * 0xcb1ab31fu;
    h = (h ^ (h >> 13)) * 0x5bd1e995u;
    h = (h ^ (h >> 15)) * 0x846ca68bu;
    return float(h ^ (h >> 16)) * (1.0 / 4294967295.0);
}
float lvNoise(vec3 p) {
    ivec3 i = ivec3(floor(p));
    vec3 f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);
    float a = mix(lvHash(i), lvHash(i + ivec3(1, 0, 0)), u.x);
    float b = mix(lvHash(i + ivec3(0, 1, 0)), lvHash(i + ivec3(1, 1, 0)), u.x);
    float c = mix(lvHash(i + ivec3(0, 0, 1)), lvHash(i + ivec3(1, 0, 1)), u.x);
    float d = mix(lvHash(i + ivec3(0, 1, 1)), lvHash(i + ivec3(1, 1, 1)), u.x);
    return mix(mix(a, b, u.y), mix(c, d, u.y), u.z);
}

// How much of the emblem's outer and inner shapes cover e (the emblem fills the unit circle;
// e.x points forward along the aircraft).
vec2 emblemAt(int kind, vec2 e) {
    const float S = 0.045; // the soft edge
    float r = length(e);
    float disc = 1.0 - smoothstep(1.0 - S, 1.0 + S, r);
    if (kind == 0) return vec2(disc, 1.0 - smoothstep(0.5 - S, 0.5 + S, r)); // a roundel
    if (kind == 1) {
        // an arrowhead pointing forward, a smaller one inside it
        float d = max(-e.x - 0.55, max(0.5 * e.x + 0.866 * e.y, 0.5 * e.x - 0.866 * e.y) - 0.55);
        return vec2(1.0 - smoothstep(-S, S, d), 1.0 - smoothstep(-S, S, d + 0.3));
    }
    if (kind == 2) {
        float d = abs(e.x) + abs(e.y);
        return vec2(1.0 - smoothstep(1.0 - S, 1.0 + S, d), 1.0 - smoothstep(0.5 - S, 0.5 + S, d)); // a diamond
    }
    if (kind == 4) {
        // three bars
        float box = (1.0 - smoothstep(0.85 - S, 0.85 + S, abs(e.y))) * (1.0 - smoothstep(0.80 - S, 0.80 + S, abs(e.x)));
        float bars = smoothstep(0.26 - S, 0.26 + S, abs(abs(e.x) - 0.53));
        return vec2(box * (1.0 - bars * smoothstep(0.14 - S, 0.14 + S, abs(e.x))), box * (1.0 - smoothstep(0.14 - S, 0.14 + S, abs(e.x))));
    }
    // a disc in two halves
    return vec2(disc, disc * smoothstep(-S, S, e.x));
}

// The colour of the skin at atlas coordinate uv in scheme `id`, from the atlas's own (`base`,
// linear) and how metallic the skin is there.
vec3 liveryColour(int id, vec2 uv, vec3 base, float metal) {
    if (id <= 0) return base;
    Scheme s = scheme(id);
    float rb = (1.0 - uv.y) * 8192.0;
    float x = -0.45 + uv.x * 16.0; // metres aft of the radome's tip
    int region = rb < 2624.0 ? 0 : rb < 5248.0 ? 1 : rb < 6528.0 ? 2 : 3;
    // metres to the right (the drawings from above and below) or up (the two from the side)
    float v = region == 0 ? rb / 256.0 - 5.125 : region == 1 ? (rb - 2624.0) / 256.0 - 5.125 : region == 2 ? (rb - 5248.0) / 256.0 - 1.2 : (rb - 6528.0) / 256.0 - 1.2;
    // a place in the aircraft, near enough for the drawings to agree where they meet on its shoulder
    vec3 at = region < 2 ? vec3(x, v, 0.6) : vec3(x, region == 2 ? -0.8 : 0.8, v);
    float seed = float(id) * 13.7;
    float n;
    if (s.pattern == 1) {
        float warp = lvNoise(at * 0.9 + seed) + 0.5 * lvNoise(at * 2.1 + seed);
        n = 0.5 + 0.5 * sin(at.x * 6.2831853 / s.size + 5.0 * warp + 1.3 * at.y + 1.1 * at.z);
        n = smoothstep(0.52, 0.60, n) * 0.5;
    } else if (s.pattern == 2) {
        vec2 gq = vec2(at.x * 0.9 + (at.y + at.z) * 0.45, (at.y + at.z) * 0.95 - at.x * 0.25) / s.size + seed;
        vec2 fr = fract(gq);
        n = lvHash(ivec3(ivec2(floor(gq)), fr.x + fr.y > 1.0 ? 1 : 0));
    } else {
        vec3 q = at / s.size + seed;
        n = 0.6 * lvNoise(q) + 0.3 * lvNoise(q * 2.3 + 3.1) + 0.1 * lvNoise(q * 5.1);
        n = smoothstep(0.30, 0.70, n);
    }
    vec3 paint = mix(s.c0, s.c1, smoothstep(0.37, 0.40, n));
    paint = mix(paint, s.c2, smoothstep(0.66, 0.69, n));
    if (region == 1) paint = s.under;
    if (x < 1.89) paint = vec3(0.09, 0.095, 0.10); // the radome
    // the emblem: on the wings, above and below, and on each side of the fin
    vec2 e = vec2(9.0);
    float size = 1.0;
    if (region == 0) { e = vec2(9.15 - x, abs(v) - 3.0); size = 0.42; }
    else if (region == 1) { e = vec2(9.2 - x, abs(v) - 2.42); size = 0.34; }
    else { e = vec2(12.72 - x, v - 2.45); size = 0.33; }
    vec2 em = emblemAt(s.emblem, e / size) * 0.93;
    paint = mix(paint, s.markA, em.x);
    paint = mix(paint, s.markB, em.y);
    // the band across the top of the fin
    if (region >= 2 && v > 3.12 && v < 3.30 && x > 11.5) paint = s.markA;
    // the atlas's shade (panel lines, wear, repainted patches), not its greys
    float lum = dot(base, vec3(0.2126, 0.7152, 0.0722));
    vec3 col = paint * clamp(pow(lum / 0.17, 0.55), 0.5, 1.3);
    // bare metal, the cockpit's floor and the gun's port stay as they are
    float keep = max(smoothstep(0.3, 0.6, metal), 1.0 - smoothstep(0.010, 0.022, lum));
    return mix(col, base, keep);
}

#endif
