// Shared math used by both ray tracing and compute shaders. No extensions required.
#ifndef SHARED_GLSL
#define SHARED_GLSL

#define PI 3.14159265
#define MAT_BARK 0u
#define MAT_LEAF 1u
#define MAT_TERRAIN 2u
#define MAT_AIRCRAFT 3u
#define FLAG_IMPOSTOR 256u
#define FLAG_IMPOSTOR_HI 512u
#define FLAG_WATER 2048u  // river surface: reflective water (main.rgen)
#define FLAG_GLOSSY 1024u // waxy leaf top: gets a specular sun highlight
#define FLAG_SEA 4096u    // globe: the water is sea (deep blue, salt), not a lake or river
#define FLAG_LAWN 8192u   // a blade of the lawn (lawn.glsl): an impostor hit on the ground
#define FLAG_AIRCRAFT 16384u // the aircraft (aircraft.rchit): roughness, metal in the payload's trans, emission in wtan
#define MAX_VARIANTS 256 // atlas slots (one per plant variant)
#define MAX_SPECIES 64
#define MAX_GROUND_TEX 84 // 3 per ground layer (27 layers); then the grass map (--lawn auto) and the far / near masks
// world.rs: near ground mask over +-NEAR_HALF, far mask over +-TERRAIN_HALF, forest tiles to TILE_FAR
#define NEAR_HALF 2048.0
#define TERRAIN_HALF 60000.0
#define TILE_FAR 18000.0

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
// Value noise with its derivatives: (value, d/dx, d/dy), quintic so the slopes are smooth too.
vec3 vnoiseD(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    vec2 du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
    float a = hashf(i), b = hashf(i + vec2(1, 0)), c = hashf(i + vec2(0, 1)), d = hashf(i + vec2(1, 1));
    float k = a - b - c + d;
    return vec3(a + (b - a) * u.x + (c - a) * u.y + k * u.x * u.y,
                du.x * ((b - a) + k * u.y),
                du.y * ((c - a) + k * u.x));
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

// The lawn's views (src/lawn.rs): n.x azimuths round the compass x n.y slopes, the slope
// (cot of the elevation) geometric from slopes.x (steepest) to slopes.y (flattest). `cell`:
// (azimuth index, slope index); the direction points from the lawn towards the viewer.
vec3 lawnFrameDir(vec2 cell, vec2 n, vec2 slopes) {
    float az = cell.x / n.x * 6.2831853;
    float s = slopes.x * pow(slopes.y / slopes.x, cell.y / (n.y - 1.0));
    return normalize(vec3(s * cos(az), 1.0, s * sin(az)));
}
// Where direction v lies among the views: x in [0, n.x) (it wraps), y in [0, n.y - 1] (steeper
// and flatter directions: the first and the last row).
vec2 lawnFrameCoord(vec3 v, vec2 n, vec2 slopes) {
    float az = atan(v.z, v.x) / 6.2831853;
    float s = clamp(length(v.xz) / max(v.y, 1e-4), slopes.x, slopes.y);
    return vec2(fract(az + 1.0) * n.x, log(s / slopes.x) / log(slopes.y / slopes.x) * (n.y - 1.0));
}

// ---------------------------------------------------------------- materials
// Foliage in dry biomes (dry 0..1): towards a dusty grey-olive of the same brightness, a little
// paler (waxy, dust-covered leaves of desert and Mediterranean scrub).
vec3 dryTint(vec3 alb, float dry) {
    float l = dot(alb, vec3(0.30, 0.59, 0.11));
    vec3 dusty = l * vec3(1.06, 1.0, 0.80) * 1.12;
    return mix(alb, dusty, 0.62 * dry);
}

// Foliage looks: 4 bits per plant (forest.rs look(), world.rs foliage(), --foliage), one of a
// small table: the leaves move towards a colour at their own brightness. Free to switch: no
// textures, no re-bake (the atlases hold the plant's natural colour), a few instructions per hit.
// 0 natural, 1 lush (deep tropical green), 2 mangrove (dark glossy olive-green), 3 boreal (dark
// blue-green), 4 autumn (each tree gold, orange, red or still green), 5 spring (fresh yellow-green),
// 6 dusty (sun-bleached grey-olive).
vec3 foliageLook(vec3 alb, uint look, uint seed) {
    if (look == 0u) return alb;
    // the greens: a filter on the leaf's own colour, so each tree keeps its own shade (treeTint)
    // and each leaf its texture
    // (lush and mangrove: deeper and bluer, not more saturated: yellow-green reads as lime)
    if (look == 1u) return alb * vec3(0.76, 0.86, 0.84);
    if (look == 2u) return alb * vec3(0.62, 0.74, 0.70);
    if (look == 3u) return alb * vec3(0.82, 0.96, 0.94);
    if (look == 5u) return alb * vec3(1.06, 1.12, 0.74);
    // the turned leaves: towards a colour at the leaf's own brightness
    float l = dot(alb, vec3(0.30, 0.59, 0.11));
    vec4 t; // rgb: the colour per unit of brightness; a: how far towards it
    if (look == 4u) {
        float h = float(pcg(seed * 3571u + 11u) & 0xFFFFu) / 65535.0;
        t = h < 0.3 ? vec4(1.55, 1.18, 0.30, 0.85) : h < 0.55 ? vec4(1.85, 0.92, 0.22, 0.85)
          : h < 0.75 ? vec4(1.80, 0.55, 0.28, 0.80) : vec4(1.15, 1.05, 0.40, 0.40);
    } else if (look == 6u) t = vec4(1.19, 1.12, 0.90, 0.62);
    else return alb;
    return mix(alb, l * t.rgb, t.a);
}

vec3 treeTint(uint seed, float leafiness) {
    if (seed == 0u) return vec3(1.0);
    // The seed's low 6 bits are the hue (blue-green .. yellow-green), the 7 above the brightness:
    // read as they are, so whoever plants the trees can give neighbours kindred seeds (the
    // globe's stands, forest.rs) or random ones (the forest scenes: every tree its own).
    float h1 = float(seed & 63u) / 63.0;
    float h2 = float((seed >> 6) & 127u) / 127.0;
    vec3 leafT = mix(vec3(0.80, 1.0, 0.74), vec3(1.20, 1.02, 0.60), h1) * mix(0.72, 1.22, h2);
    return mix(vec3(1.0), leafT, leafiness);
}

#endif
