// Shared math used by both ray tracing and compute shaders. No extensions required.
#ifndef SHARED_GLSL
#define SHARED_GLSL

#define PI 3.14159265
#define MAT_BARK 0u
#define MAT_LEAF 1u
#define MAT_TERRAIN 2u
#define FLAG_IMPOSTOR 256u
#define FLAG_IMPOSTOR_HI 512u
#define FLAG_GLOSSY 1024u // waxy leaf top: gets a specular sun highlight
#define MAX_VARIANTS 128 // atlas slots (one per plant variant)
#define MAX_SPECIES 16
#define MAX_GROUND_TEX 16 // 3 per ground layer; the last slot is the biome mask
#define TERRAIN_SIZE 3072.0

// ---------------------------------------------------------------- hashing / random
uint pcg(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
uint hash2(uvec2 v) { return pcg(v.x ^ pcg(v.y + 0x9e3779b9u)); }
float rnd(inout uint s) { s = pcg(s); return float(s) * (1.0 / 4294967296.0); }

vec3 cosineSample(vec3 n, inout uint s) {
    float u1 = rnd(s), u2 = rnd(s);
    float r = sqrt(u1), phi = 6.2831853 * u2;
    vec3 t = normalize(abs(n.y) < 0.99 ? cross(n, vec3(0, 1, 0)) : cross(n, vec3(1, 0, 0)));
    vec3 b = cross(n, t);
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u1)));
}

float hashf(vec2 p) {
    uvec2 q = uvec2(ivec2(floor(p)) + 65536);
    return float(hash2(q)) * (1.0 / 4294967295.0);
}
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = hashf(i), b = hashf(i + vec2(1, 0)), c = hashf(i + vec2(0, 1)), d = hashf(i + vec2(1, 1));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// ---------------------------------------------------------------- octahedral mappings
float signNZ(float v) { return v >= 0.0 ? 1.0 : -1.0; }

// Full-sphere octahedral encoding with Y as the pole axis. Result in [-1,1]^2.
vec2 octEncode(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    vec2 p = n.xz;
    if (n.y < 0.0) p = (1.0 - abs(p.yx)) * vec2(signNZ(p.x), signNZ(p.y));
    return p;
}
vec3 octDecode(vec2 e) {
    vec3 v = vec3(e.x, 1.0 - abs(e.x) - abs(e.y), e.y);
    if (v.y < 0.0) {
        vec2 p = (1.0 - abs(v.zx)) * vec2(signNZ(v.x), signNZ(v.z));
        v.x = p.x;
        v.z = p.y;
    }
    return normalize(v);
}

// Hemi-octahedral mapping of the upper hemisphere (Y up) to [0,1]^2.
// Square edges map to the horizon, the centre to straight up.
vec2 hemiOctEncode(vec3 d) {
    d.y = max(d.y, 0.0);
    d /= (abs(d.x) + d.y + abs(d.z));
    return vec2(d.x + d.z, d.x - d.z) * 0.5 + 0.5;
}
vec3 hemiOctDecode(vec2 uv) {
    vec2 o = uv * 2.0 - 1.0;
    vec2 p = vec2(o.x + o.y, o.x - o.y) * 0.5;
    return normalize(vec3(p.x, 1.0 - abs(p.x) - abs(p.y), p.y));
}

// Orthographic camera basis of an impostor frame. `f` points from the tree towards the viewer.
void frameBasis(vec3 f, out vec3 r, out vec3 u) {
    vec3 up = abs(f.y) > 0.999 ? vec3(0, 0, -1) : vec3(0, 1, 0);
    r = normalize(cross(up, f));
    u = cross(f, r);
}

// Direction of impostor frame `cell` in an N x N hemi-octahedral grid (frames at grid vertices).
vec3 frameDir(vec2 cell, float N) { return hemiOctDecode(cell / (N - 1.0)); }

// ---------------------------------------------------------------- materials
vec3 treeTint(uint seed, float leafiness) {
    if (seed == 0u) return vec3(1.0);
    uint s = pcg(seed * 7919u);
    float h1 = float(s & 0xFFFFu) / 65535.0;
    float h2 = float(s >> 16) / 65535.0;
    vec3 leafT = mix(vec3(0.82, 1.0, 0.70), vec3(1.18, 1.02, 0.62), h1) * mix(0.78, 1.18, h2);
    return mix(vec3(1.0), leafT, leafiness);
}

#endif
