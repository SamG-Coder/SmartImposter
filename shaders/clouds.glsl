// Volumetric clouds (globe mode, main.rgen): a layer of cumulus between CLOUD_BASE and CLOUD_TOP
// above sea level, Earth-fixed (the camera frame turns into Earth axes through g.earthX..Z),
// drifting with a constant wind. Ray marched along primary and reflection rays (sunlight marched
// towards the sun through the cloud, dual-lobe phase, a multiple-scattering floor, sky ambient),
// and marched again from every lit point towards the sun for the cloud shadows on the ground.
// Needs atmosphere.glsl, shared.glsl (pcg) and the Globals `g`.
#ifndef CLOUDS_GLSL
#define CLOUDS_GLSL

const float CLOUD_BASE = 1500.0;
const float CLOUD_TOP = 4200.0;
const float CLOUD_SIGMA = 0.035; // extinction at full density, 1/m (cumulus: ~0.02-0.1)

bool cloudsOn() { return g.planet.w > 0.5 && g.cloud.y > 0.5; }

// Earth-fixed position in km (ECEF) of a point relative to the planet centre (camera frame).
vec3 cloudEarth(vec3 x) {
    return vec3(dot(g.earthX.xyz, x), dot(g.earthY.xyz, x), dot(g.earthZ.xyz, x)) * 1e-3;
}

float chash3(ivec3 c) {
    uint h = uint(c.x) * 1597334677u ^ uint(c.y) * 3812015801u ^ uint(c.z) * 2541459121u;
    h = (h ^ (h >> 16)) * 0x7feb352du;
    h = (h ^ (h >> 15)) * 0x846ca68bu;
    return float(h ^ (h >> 16)) * (1.0 / 4294967295.0);
}

float cnoise3(vec3 p) {
    ivec3 i = ivec3(floor(p));
    vec3 f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);
    float a = mix(chash3(i), chash3(i + ivec3(1, 0, 0)), u.x);
    float b = mix(chash3(i + ivec3(0, 1, 0)), chash3(i + ivec3(1, 1, 0)), u.x);
    float c = mix(chash3(i + ivec3(0, 0, 1)), chash3(i + ivec3(1, 0, 1)), u.x);
    float d = mix(chash3(i + ivec3(0, 1, 1)), chash3(i + ivec3(1, 1, 1)), u.x);
    return mix(mix(a, b, u.y), mix(c, d, u.y), u.z);
}

float cfbm3(vec3 p, int oct) {
    float s = 0.0, a = 0.5, n = 0.0;
    for (int i = 0; i < oct; ++i) {
        s += a * cnoise3(p);
        n += a;
        p = p * 2.03 + vec3(17.1, 3.7, 9.3);
        a *= 0.5;
    }
    return s / n;
}

// Wind: the whole layer drifts (km); the shapes also change slowly.
vec3 cloudWind() { return vec3(0.011, 0.004, 0.0) * g.params.x; }

// Cloud cover (0..1) of the column over Earth-fixed point e (km): weather cells of ~40 km and
// clusters of ~8 km, fewer clouds over the subtropical deserts belts, more at the equator.
float cloudCover(vec3 e) {
    vec3 q = e + cloudWind();
    float lat = degrees(asin(clamp(e.z / length(e), -1.0, 1.0)));
    float belt = 1.0 - 0.55 * exp(-pow((abs(lat) - 24.0) / 9.0, 2.0)) + 0.25 * exp(-pow(lat / 8.0, 2.0));
    float cov = clamp(g.cloud.x * belt, 0.0, 1.0);
    // weather systems (~600 km: fronts, clear highs), cells of ~40 km, clusters of ~8 km
    float sys = cfbm3(q / 600.0 + 11.0, 3);
    cov = clamp(cov + (sys - 0.5) * 0.9, 0.0, 1.0);
    float w = cfbm3(q / 40.0, 2) * 0.55 + cfbm3(q / 8.0 + 3.0, 3) * 0.45;
    return smoothstep(1.0 - cov - 0.12, 1.0 - cov + 0.12, w);
}

// Density (0..1) at x (relative to the planet centre, camera frame); `detail` erodes the edges
// (off for the light and shadow marches).
float cloudDensity(vec3 x, bool detail) {
    float h = length(x) - R_GROUND;
    if (h < CLOUD_BASE || h > CLOUD_TOP) return 0.0;
    vec3 e = cloudEarth(x);
    float c = cloudCover(e);
    if (c <= 0.0) return 0.0;
    float hn = (h - CLOUD_BASE) / (CLOUD_TOP - CLOUD_BASE);
    // flat bases, rounded tops that rise where the cover is denser (towers in the cells)
    float shape = smoothstep(0.0, 0.08, hn) * (1.0 - smoothstep(0.35 + 0.6 * c, 1.0, hn));
    float d = c * shape;
    if (detail && d > 0.0) {
        vec3 q = e + cloudWind() * 1.6;
        float n = cfbm3(q / 0.9, 2);
        d = clamp((d - (1.0 - n) * 0.55 * (1.0 - 0.4 * hn)) / 0.45, 0.0, 1.0);
    }
    return d;
}

// Sun transmittance through the clouds from x (planet-centred) towards L: the cloud shadows.
float cloudShadow(vec3 x, vec3 L) {
    if (!cloudsOn()) return 1.0;
    float h = length(x) - R_GROUND;
    if (h > CLOUD_TOP) return 1.0;
    vec2 top = raySphere(x, L, R_GROUND + CLOUD_TOP);
    vec2 base = raySphere(x, L, R_GROUND + CLOUD_BASE);
    float t0 = h < CLOUD_BASE ? max(base.y, 0.0) : 0.0;
    float t1 = top.y;
    if (t1 <= t0) return 1.0;
    t1 = min(t1, t0 + 12000.0);
    const int N = 6;
    float ds = (t1 - t0) / float(N), od = 0.0;
    for (int i = 0; i < N; ++i) od += cloudDensity(x + L * (t0 + (float(i) + 0.5) * ds), false) * ds;
    return exp(-CLOUD_SIGMA * 0.7 * od);
}

float hgPhase(float mu, float k) { return (1.0 - k * k) / (4.0 * PI * pow(1.0 + k * k - 2.0 * k * mu, 1.5)); }

// March the clouds along ro + rd * [0, tmax] (ro relative to the planet centre): in-scattered
// light (rgb, already dimmed by the air in front of the cloud) and transmittance (a). `steps`:
// 40 for primary rays, fewer for reflections. `jitter` in [0, 1) per pixel and frame.
vec4 cloudMarch(vec3 ro, vec3 rd, float tmax, int steps, float jitter) {
    if (!cloudsOn()) return vec4(0.0, 0.0, 0.0, 1.0);
    float rb = R_GROUND + CLOUD_BASE, rt = R_GROUND + CLOUD_TOP;
    float h0 = length(ro) - R_GROUND;
    vec2 top = raySphere(ro, rd, rt), base = raySphere(ro, rd, rb);
    float t0, t1;
    if (h0 < CLOUD_BASE) {
        t0 = base.y; t1 = top.y;                       // from below: up through the layer
    } else if (h0 > CLOUD_TOP) {
        if (top.y < 0.0) return vec4(0.0, 0.0, 0.0, 1.0);
        t0 = max(top.x, 0.0);                          // from above: down to the base or out again
        t1 = (base.x > 0.0 && base.y > 0.0) ? base.x : top.y;
    } else {
        t0 = 0.0;
        t1 = (base.x > 0.0 && base.y > 0.0) ? base.x : top.y;
    }
    t1 = min(t1, tmax);
    if (t1 <= t0) return vec4(0.0, 0.0, 0.0, 1.0);
    t1 = min(t1, t0 + 90000.0); // the layer seen edge-on: beyond this it is haze
    vec3 L = g.sunDir.xyz;
    float mu = dot(rd, L);
    float phase = mix(hgPhase(mu, 0.72), hgPhase(mu, -0.15), 0.35);
    // sunlight at the layer (through the air above it) and the sky light: once per ray
    vec3 xm = ro + rd * (0.5 * (t0 + t1));
    vec3 Esun = g.sunDir.w * extinction(lightDepth(xm, L));
    vec3 amb = Esun * vec3(0.06, 0.075, 0.11) + vec3(0.02, 0.03, 0.05) * g.sunDir.w * max(dot(normalize(xm), L) + 0.2, 0.0);
    float ds = (t1 - t0) / float(steps);
    float T = 1.0, tFirst = -1.0;
    vec3 S = vec3(0.0);
    for (int i = 0; i < steps; ++i) {
        float t = t0 + (float(i) + jitter) * ds;
        vec3 x = ro + rd * t;
        float d = cloudDensity(x, true);
        if (d <= 0.0) continue;
        if (tFirst < 0.0) tFirst = t;
        // towards the sun: 3 samples over ~1.6 km (coarse density)
        float odL = 0.0;
        for (int j = 0; j < 3; ++j) {
            float tl = (float(j) + 0.5) * 530.0;
            odL += cloudDensity(x + L * tl, false) * 530.0;
        }
        float Tl = exp(-CLOUD_SIGMA * odL);
        float ms = max(Tl, 0.3 * exp(-CLOUD_SIGMA * 0.25 * odL)); // multiple scattering floor
        float hn = (length(x) - R_GROUND - CLOUD_BASE) / (CLOUD_TOP - CLOUD_BASE);
        vec3 Li = Esun * phase * ms + amb * (0.5 + 0.5 * hn);  // brighter tops
        float sig = CLOUD_SIGMA * d;
        float Ts = exp(-sig * ds);
        S += T * Li * (1.0 - Ts);                      // energy-conserving step integral
        T *= Ts;
        if (T < 0.01) break;
    }
    if (tFirst < 0.0) return vec4(0.0, 0.0, 0.0, 1.0);
    // The air in front of the cloud: dims the cloud and adds its own haze. The background (sky or
    // ground) already carries that haze, and the cloud covers (1 - T) of it: added back, so far
    // clouds fade into the sky colour instead of turning brown.
    vec3 Ta;
    vec3 Sa = atmosphere(ro, rd, tFirst, L, g.sunDir.w, Ta);
    return vec4(S * Ta + Sa * (1.0 - T), T);
}

#endif
