// Volumetric clouds (main.rgen: the globe, and the forest scenes over a sphere under them), after Schneider's Nubis (Horizon Zero Dawn / Forbidden
// West), Hillaire's Frostbite sky and Wrenninge's multiple-scattering approximation:
//
// - Three layers, fixed to the Earth (the camera frame turns into Earth axes through g.earthX..Z):
//   low clouds from 600 m (stratus and stratocumulus sheets ~1 km thick, cumulus to ~3 km,
//   cumulonimbus towers to 11 km, by cloud type), altocumulus at 3.5-5 km, cirrus at 8-9.5 km.
// - Where they are (weatherAt): the globe's weather map (globe/weather.rs, one texture lookup):
//   the climate of real cloud cover, cyclones with their fronts along the storm tracks, the ITCZ's
//   clusters, tropical storms; and under a texel of it cells of ~12 km that grow and decay over
//   tens of minutes. Far away, where a pixel is larger than the clouds, a field keeps its cover
//   (lowDensityC) instead of thresholding averaged noise.
// - Shapes: a tiling Perlin-Worley base eroded by Worley fBm (cloudnoise.rs, 3D textures 34, 35),
//   shaped by a height profile per type (flat bases, rounded tops), wispy below, billowy above.
// - Light: sunlight marched towards the sun, summed over Wrenninge's scattering octaves (the first
//   bounces: the silver lining, soft interiors), Jendersie & d'Eon's droplet phase function (HG +
//   Draine: the sharp forward peak, fogbow and glory lobes), the "powder" darkening of thin edges
//   seen away from the sun; the many-times scattered sunlight by the two-stream picture (a thick
//   cloud turns back most of the sunlight on it, white from every lit side, and passes the rest
//   down diffused: grey bases); sky light from the whole dome and the sunlit surface's bounce
//   below. Clouds shadow the ground (cloudShadow: the beam from the sun's disc and the aureole of
//   forward-scattered light round it: soft edges, dark cores) and dim the sky light under them.
// - The march (cloudScatter): long steps on the shape alone through clear air, short ones with
//   the detail in cloud, lengthening again through the shape's empty fringe; the low layer only
//   as high as the region's clouds stand; the light's setup only for rays that meet cloud.
//
// The forest scenes (the generated forest, a site, the park: g.planet.w = 0) have the same clouds:
// main.rs puts a planet under the scene (its centre in g.planet.xyz, the terrain already carries
// the Earth's curvature) at a fixed place in the mid-latitudes, CLOUD_LIFT below sea level so
// the cumulus bases stand ~1 km over the scene; their weather is a fair day's (weatherAt).
// Needs atmosphere.glsl, shared.glsl and the Globals `g`.
#ifndef CLOUDS_GLSL
#define CLOUDS_GLSL

layout(set = 0, binding = 32) uniform sampler2D cloudElev;   // ETOPO (globe.rchit elevTex): land / sea
layout(set = 0, binding = 34) uniform sampler3D cloudShape;  // R Perlin-Worley, GBA Worley fBm
layout(set = 0, binding = 35) uniform sampler3D cloudDetail; // RGB Worley fBm (fine)
layout(set = 0, binding = 38) uniform sampler2D cloudWeather; // globe/weather.rs: low cover, low type, mid cover, high cover

const float LOW_BASE = 600.0, LOW_TOP = 11000.0;
const float MID_BASE = 3500.0, MID_TOP = 5000.0;
const float HIGH_BASE = 8000.0, HIGH_TOP = 9500.0;
// extinction at full density (1/m): cumulus ~0.03-0.15 (liquid water 0.2-1 g/m3, 10 um droplets),
// altocumulus thinner, cirrus (ice) ~0.001-0.003
const float SIGMA_LOW = 0.06, SIGMA_MID = 0.03, SIGMA_HIGH = 0.0008;

// (main.rs CLOUD_LIFT) the forest scenes' sea level is this far above the clouds' ground sphere
const float CLOUD_LIFT = 400.0;
const float CLOUD_LOCAL_GAIN = 2.2;

bool cloudsOn() { return g.cloud.y > 0.5; }

// The size of a pixel (m) where the current sample lies: the noise is read at the mip of that
// footprint (no speckle from noise finer than a pixel), and the fine erosion fades out where its
// billows are smaller than a pixel (from far away, or high above, a cumulus field is its shapes).
float cFoot = 0.0;
float cloudPixAngle() { return 2.0 * g.camUp.w / float(gl_LaunchSizeEXT.y); }
// the forest scenes (not the globe)
bool cloudsLocal() { return g.planet.w < 0.5; }

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

float cremap(float v, float lo, float hi, float nlo, float nhi) { return nlo + (v - lo) / max(hi - lo, 1e-4) * (nhi - nlo); }

// Wind (km of drift) and the time axis of the evolving noise: the layers drift with the westerly
// / trade winds of their latitude (slower aloft for the low layer), and shapes change over minutes.
float cloudTime() { return g.cloud.w; } // seconds of animation (main.rs State.clock)
vec3 cloudDrift(vec3 e, float speed) {
    vec3 u = normalize(e);
    vec3 east = normalize(cross(vec3(0.0, 0.0, 1.0), u) + vec3(1e-5, 0.0, 0.0));
    float lat = abs(u.z);
    float dirSign = lat > 0.5 ? 1.0 : -1.0; // westerlies poleward of 30 degrees, easterly trades within
    return east * dirSign * speed * cloudTime() * 1e-3;
}

// Weather of a column (Earth-fixed unit direction u):
// x low cover, y low type (0 stratus / stratocumulus, 0.5 cumulus, 1 cumulonimbus), z mid cover,
// w high (cirrus) cover.
// land fraction around u (~60 km blur): coastlines change the weather gradually, not at a line
float landAtLod(vec3 u, float lod) {
    float lat = asin(clamp(u.z, -1.0, 1.0)), lon = atan(u.y, u.x);
    float e = textureLod(cloudElev, vec2(lon / (2.0 * PI) + 0.5, 0.5 - lat / PI), lod).r * 32767.0;
    return smoothstep(-150.0, 150.0, e);
}
float landAt(vec3 u) { return landAtLod(u, 6.0); }
vec4 weatherAtLod(vec3 e, float lod) {
    if (cloudsLocal()) {
        // The forest scenes: a fair day. Fields of cumulus with clearer lanes between them
        // (~40 km), a few altocumulus where the cumulus is thickest, cirrus streaks elsewhere.
        vec3 q = e / 40.0 + vec3(0.0, 0.0, cloudTime() / 20000.0);
        float sys = cnoise3(q) * 0.65 + cnoise3(q * 2.3 + 7.7) * 0.35;
        float k = g.cloud.x / 0.45;
        return vec4(clamp((0.245 + (sys - 0.5) * 0.08) * k, 0.0, 1.0), 0.35,
                    clamp(0.12 * smoothstep(0.6, 0.85, sys) * k, 0.0, 1.0),
                    clamp((0.03 + 0.2 * smoothstep(0.4, 0.75, 1.0 - sys)) * min(k, 1.5), 0.0, 1.0));
    }
    // The globe: the weather map (globe/weather.rs: the climate of cloud cover, cyclones with
    // their fronts, the ITCZ's clusters, tropical storms), drifting with the wind of its latitude
    // (the trades blow west, the westerlies east; metres in a session, not a shear one sees).
    vec3 u = normalize(e);
    float lat = asin(clamp(u.z, -1.0, 1.0)), lon = atan(u.y, u.x);
    float wind = mix(-5.0, 10.0, smoothstep(0.35, 0.7, abs(lat))); // m/s eastward
    lon -= wind * cloudTime() / (6.371e6 * max(cos(lat), 0.05));
    vec4 Wm = textureLod(cloudWeather, vec2(lon / (2.0 * PI) + 0.5, clamp(0.5 - lat / PI, 0.0003, 0.9997)), lod);
    float k = g.cloud.x / 0.45;
    return vec4(clamp(Wm.x * k, 0.0, 1.0), Wm.y, clamp(Wm.z * k, 0.0, 1.0), clamp(Wm.w * min(k, 1.5), 0.0, 1.0));
}
vec4 weatherAt(vec3 e) { return weatherAtLod(e, 0.0); }

// Top of the low layer for a cloud type (m above sea level): stratus ~1.5 km, cumulus ~3 km,
// cumulonimbus to 11 km.
float lowTop(float type) {
    return mix(mix(LOW_BASE + 900.0, LOW_BASE + 2600.0, smoothstep(0.0, 0.5, type)), LOW_TOP, smoothstep(0.6, 1.0, type));
}

// Low layer density at e (km, Earth-fixed) and height h (m), weather W; `detail`: the fine
// erosion (off for the light and shadow marches).
// The cover at e: the weather's, and cells of ~12 km that grow and decay (the tile's z axis is
// time). It changes over kilometres: a march of a few steps (the light's) takes it once.
float lowCover(vec3 e, vec4 W) {
    vec3 p = e - cloudDrift(e, 8.0);
    float cell = textureLod(cloudShape, p / 48.0 + vec3(0.0, 0.0, cloudTime() / 3000.0), max(log2(cFoot / 375.0), 0.0)).r;
    // (the tile repeats every 48 km: where a pixel is kilometres wide the repeat would show as a
    // lattice of dots; there the cells give way to their mean, and the weather map's own texture,
    // which does not repeat, is what one sees)
    cell = mix(cell, 0.669, smoothstep(400.0, 2200.0, cFoot));
    return clamp(W.x + (cell - 0.5) * 0.8, 0.0, 1.0);
}

float lowNearC(vec3 e, float h, vec4 W, float cov, bool detail) {
    float top = lowTop(W.y);
    float hn = (h - LOW_BASE) / (top - LOW_BASE);
    if (hn <= 0.0 || hn >= 1.0 || cov <= 0.02) return 0.0;
    vec3 p = e - cloudDrift(e, 8.0);
    float t = cloudTime();
    // height profile: flat base, rounded top; stratus thin and even, towers anvil-wide at the top
    float prof = smoothstep(0.0, 0.08 + 0.1 * W.y, hn) * smoothstep(1.0, mix(0.55, 0.75, W.y), hn);
    if (W.y > 0.85) prof = max(prof, smoothstep(0.82, 0.9, hn) * smoothstep(1.0, 0.94, hn)); // anvil
    vec4 s = textureLod(cloudShape, p / 9.0 + vec3(0.0, 0.0, t / 1200.0), max(log2(cFoot / 70.0), 0.0));
    float wf = s.g * 0.625 + s.b * 0.25 + s.a * 0.125;
    float base = cremap(s.r, wf - 1.0, 1.0, 0.0, 1.0) * prof;
    float d = clamp(cremap(base, 1.0 - cov, 1.0, 0.0, 1.0), 0.0, 1.0) * cov;
    if (d <= 0.0 || !detail) return d;
    vec3 dt = textureLod(cloudDetail, p / 1.2 + vec3(0.0, 0.0, t / 400.0), max(log2(cFoot / 37.0), 0.0)).rgb;
    float df = dt.r * 0.625 + dt.g * 0.25 + dt.b * 0.125;
    float m = mix(df, 1.0 - df, clamp(hn * 8.0, 0.0, 1.0)); // wispy bases, billowy tops
    return clamp(cremap(d, m * 0.35, 1.0, 0.0, 1.0), 0.0, 1.0);
}

// The density a cloud of the field has inside it at the last sample (cloudScatter: how much
// sunlight the column turns back).
float cInner = 0.0;

// Where a pixel is larger than the clouds (from high up, from orbit: cFoot of kilometres) the
// noise it reads is an average, and a threshold of that is either nothing or a solid sheet:
// scattered cumulus vanished and anything denser was a slab with a cut edge. What a pixel of a
// cloud field shows is the field's cover: the share f of it that is cloud hides that share of the
// ground (clumps: T = 1 - f, not exp(-mean tau)). So far away the density goes over to the one
// that gives the column that transmittance. f from the cover, as measured on the near field's
// own noise (the share of columns that hold cloud: 6% at a weather cover of 0.13, a third at 0.2,
// 87% at 0.3, overcast from 0.4; `cov` here has the cells' mean in it, +0.135).
float lowDensityC(vec3 e, float h, vec4 W, float cov, bool detail) {
    float lodT = smoothstep(400.0, 4000.0, cFoot);
    float d = lodT < 1.0 ? lowNearC(e, h, W, cov, detail) : 0.0;
    cInner = d;
    if (lodT <= 0.0) return d;
    float top = lowTop(W.y);
    float hn = (h - LOW_BASE) / (top - LOW_BASE);
    if (hn <= 0.0 || hn >= 1.0 || cov <= 0.02) return 0.0;
    float prof = smoothstep(0.0, 0.08 + 0.1 * W.y, hn) * smoothstep(1.0, mix(0.55, 0.75, W.y), hn);
    float f = pow(smoothstep(0.2, 0.5, cov), 1.4);
    float dField = -log(max(1.0 - f, 0.003)) / (SIGMA_LOW * 0.7 * (top - LOW_BASE)) * smoothstep(0.0, 0.15, prof);
    // (how deep the cloud is there, for the light it turns back: a veil at the edge of a cloud
    // shield has an optical depth of a few and is grey over the sea, the core of a front or a
    // storm one of 50-100 and is white)
    float core = smoothstep(0.3, 1.1, cov);
    cInner = mix(d, max(d, 0.02 + 0.75 * core * core), lodT);
    return mix(d, dField, lodT);
}

float lowDensity(vec3 e, float h, vec4 W, bool detail) {
    if (h <= LOW_BASE || h >= lowTop(W.y) || W.x <= 0.0) return 0.0;
    return lowDensityC(e, h, W, lowCover(e, W), detail);
}

// Altocumulus: a thin layer of small puffs.
float midDensity(vec3 e, float h, vec4 W, bool detail) {
    float hn = (h - MID_BASE) / (MID_TOP - MID_BASE);
    if (hn <= 0.0 || hn >= 1.0 || W.z <= 0.0) return 0.0;
    vec3 p = e - cloudDrift(e, 15.0);
    float t = cloudTime();
    vec4 s = textureLod(cloudShape, p / 5.0 + vec3(0.0, 0.0, t / 1500.0), 0.0);
    float cell = textureLod(cloudShape, p / 30.0 + vec3(0.0, 0.0, t / 3000.0), 0.0).g;
    float cov = clamp(W.z + (cell - 0.5) * 0.9, 0.0, 1.0);
    float prof = smoothstep(0.0, 0.25, hn) * smoothstep(1.0, 0.6, hn);
    float base = cremap(s.r, s.g - 1.0, 1.0, 0.0, 1.0) * prof;
    float d = clamp(cremap(base, 1.0 - cov, 1.0, 0.0, 1.0), 0.0, 1.0) * cov;
    if (d > 0.0 && detail) d = clamp(cremap(d, textureLod(cloudDetail, p / 0.8, 0.0).r * 0.4, 1.0, 0.0, 1.0), 0.0, 1.0);
    return d;
}

// Cirrus: thin ice streaks drawn out along the wind aloft.
float highDensity(vec3 e, float h, vec4 W) {
    float hn = (h - HIGH_BASE) / (HIGH_TOP - HIGH_BASE);
    if (hn <= 0.0 || hn >= 1.0 || W.w <= 0.0) return 0.0;
    vec3 p = e - cloudDrift(e, 30.0);
    vec3 u = normalize(e);
    vec3 east = normalize(cross(vec3(0.0, 0.0, 1.0), u) + vec3(1e-5, 0.0, 0.0)), north = cross(u, east);
    // streak frame: long along east (the jet), short across
    vec3 sp = vec3(dot(p, east) / 40.0, dot(p, north) / 6.0, dot(p, u) / 6.0);
    float s = textureLod(cloudShape, sp + vec3(0.0, 0.0, cloudTime() / 4000.0), 0.0).g;
    float f = textureLod(cloudDetail, vec3(dot(p, east) / 8.0, dot(p, north) / 1.0, 0.5), 0.0).r;
    float prof = smoothstep(0.0, 0.3, hn) * smoothstep(1.0, 0.5, hn);
    return clamp(cremap(s * 0.7 + f * 0.3, 1.0 - W.w, 1.0, 0.0, 1.0), 0.0, 1.0) * prof;
}

// Column cover (0..1) over Earth-fixed e: the sky light is dimmed under it (main.rgen).
float cloudCover(vec3 e) {
    vec4 W = weatherAt(e);
    float cell = textureLod(cloudShape, (e - cloudDrift(e, 8.0)) / 48.0 + vec3(0.0, 0.0, cloudTime() / 3000.0), 0.0).r;
    return clamp(max(W.x + (cell - 0.5) * 0.8, W.z * 0.6), 0.0, 1.0);
}

// The droplet phase function: Jendersie & d'Eon 2023 (HG + Draine, fitted to Mie for droplets
// of diameter D = 16 um); `k` < 1 widens it (Wrenninge's later octaves).
float hgPhase(float mu, float g) { return (1.0 - g * g) / (4.0 * PI * pow(max(1.0 + g * g - 2.0 * g * mu, 1e-5), 1.5)); }
float drainePhase(float mu, float g, float a) {
    return hgPhase(mu, g) * (1.0 + a * mu * mu) / (1.0 + a * (1.0 + 2.0 * g * g) / 3.0);
}
float dropletPhase(float mu, float k) {
    const float D = 16.0;
    float gHG = exp(-0.0990567 / (D - 1.67154));
    float gD = exp(-2.20679 / (D + 3.91029) - 0.428934);
    float aD = exp(3.62489 - 8.29288 / (D + 5.52825));
    float wD = exp(-0.599085 / (D - 0.641583) - 0.665888);
    return mix(hgPhase(mu, gHG * k), drainePhase(mu, gD * k, aD), wD);
}

// Optical depth (per unit extinction) towards the sun from x through one layer (coarse density).
// `fine`: the first ~200 m through the detailed shape, so each billow shades the next one
// towards the shadow side: the lit and shaded lobes that make a cumulus read as a solid.
// (`cov`: lowCover at x; the coarse rays, of reflections and the sky light, take 5 longer steps)
float lightDepthCloud(vec3 x, vec3 L, int layer, vec4 W, float cov, bool fine) {
    float od = 0.0, t = 0.0, dl = layer == 0 ? (fine ? 25.0 : 60.0) : 120.0;
    for (int j = 0; j < (fine ? 7 : 5); ++j) {
        t += dl;
        vec3 y = x + L * (t - 0.5 * dl);
        float h = length(y) - R_GROUND;
        vec3 e = cloudEarth(y);
        float d = layer == 0 ? lowDensityC(e, h, W, cov, fine && j < 3) : midDensity(e, h, W, false);
        od += d * dl;
        dl *= 2.0;
    }
    return od;
}

// Optical depth of the clouds from x (planet-centred) towards L; `jit` in [0, 1) shifts the samples.
float cloudDepthTo(vec3 x, vec3 L, float h0, float jit) {
    vec4 W = weatherAt(cloudEarth(x));
    float od = 0.0;
    // low layer: where the ray crosses it, 8 samples
    vec2 bot = raySphere(x, L, R_GROUND + LOW_BASE), top = raySphere(x, L, R_GROUND + lowTop(min(W.y + 0.1, 1.0)));
    float t0 = h0 < LOW_BASE ? max(bot.y, 0.0) : 0.0, t1 = min(top.y, t0 + 15000.0);
    if (t1 > t0) {
        // (the weather and the cover where the ray is halfway through the layer: they change
        // over kilometres, the ray crosses the layer in a few)
        vec3 em = cloudEarth(x + L * (0.5 * (t0 + t1)));
        vec4 Wl = cloudsLocal() ? W : weatherAt(em);
        float cov = lowCover(em, Wl);
        float ds = (t1 - t0) / 8.0;
        for (int i = 0; i < 8 && cov > 0.02; ++i) {
            vec3 y = x + L * (t0 + (float(i) + jit) * ds);
            // (with the detail: the coarse shape alone has thin fringes far beyond the cloud one
            // sees, which the detail erodes away; without it the shadows spread well past the
            // clouds and the sunlit gaps between them close)
            float d = lowDensityC(cloudEarth(y), length(y) - R_GROUND, Wl, cov, true);
            od += d * ds * SIGMA_LOW;
        }
    }
    vec2 mb = raySphere(x, L, R_GROUND + MID_BASE), mt = raySphere(x, L, R_GROUND + MID_TOP);
    if (h0 < MID_BASE && mt.y > mb.y && W.z > 0.0) {
        float ds = (mt.y - mb.y) / 3.0;
        for (int i = 0; i < 3; ++i) {
            vec3 y = x + L * (mb.y + (float(i) + jit) * ds);
            float d = midDensity(cloudEarth(y), length(y) - R_GROUND, W, false);
            if (cloudsLocal()) d = clamp((d - 0.2) / 0.8, 0.0, 1.0);
            od += d * ds * SIGMA_MID;
        }
    }
    vec2 hb = raySphere(x, L, R_GROUND + HIGH_BASE + 700.0);
    if (h0 < HIGH_BASE && hb.y > 0.0 && W.w >= 0.1) {
        vec3 y = x + L * hb.y;
        od += highDensity(cloudEarth(y), length(y) - R_GROUND, W) * 1500.0 * SIGMA_HIGH;
    }
    return od;
}

// The sun's light through the clouds at x (planet-centred), as a share of the clear sky's: the
// cloud shadows. `xi`: three random numbers per pixel and frame.
//
// Sunlight comes through a cloud of optical depth tau in two parts. The beam itself, exp(-tau),
// from the sun's disc (0.27 degrees in radius): the sharp part of a shadow, gone under any cloud
// one cannot see the sun through. And the light the droplets have turned by a few degrees only
// (half of what a droplet scatters is diffracted into a narrow forward peak): it goes on to the
// ground as an aureole round the sun, exp(-tau (1 - g^2)) - exp(-tau) of the beam with g^2 = 0.72
// (the delta-Eddington scaling), from a cone ~2.5 degrees wide. Under a cloud's thin edge that is
// most of the light, and it comes from 100-200 m around the sun's line: a shadow's edge is soft
// by that much, its core dark; and next to an edge the ground has the beam and the aureole of the
// cloud beside it both: brighter than under a clear sky, as measured (up to ~20%).
float cloudShadow(vec3 x, vec3 L, vec3 xi) {
    if (!cloudsOn()) return 1.0;
    float h0 = length(x) - R_GROUND;
    if (h0 > HIGH_TOP) return 1.0;
    vec3 t1 = normalize(cross(L, abs(L.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0))), t2 = cross(L, t1);
    float a = 6.2831853 * xi.x;
    vec2 k = vec2(cos(a), sin(a)) * sqrt(xi.y);
    float tauDisc = cloudDepthTo(x, normalize(L + (t1 * k.x + t2 * k.y) * 0.00465), h0, xi.z);
    float tauCone = cloudDepthTo(x, normalize(L + (t2 * k.x - t1 * k.y) * 0.044), h0, fract(xi.z + 0.5));
    return exp(-tauDisc) + exp(-0.28 * tauCone) - exp(-tauCone);
}
// (without random numbers: along the sun's line alone)
float cloudShadow(vec3 x, vec3 L) { return cloudShadow(x, L, vec3(0.0, 0.0, 0.5)); }

// ---- the aircraft's wake (--test-flight): where it has flown through cloud, the cloud is
// stirred and thinned. Its engine's heat and the sinking, warming air in its wake evaporate the
// droplets (a dissipation trail: a clear lane through thin cloud), and its tip vortices roll the
// cloud round the path. The wake is a string of points with their age (g.wake); round each
// stretch the density is cut in a tube that widens with age, and the cloud's shapes are turned
// about the path. The tube closes again after a minute and a half.
vec3 wakePoint(int i) { return g.wake[i].xyz - g.planet.xyz; }

// The stretches of the wake that the ray ro + rd t comes near (at most 6): their first points.
int wakeNear(vec3 ro, vec3 rd, out int cand[6]) {
    int nc = 0;
    int n = g.acState.x > 0.5 ? int(g.acState2.y) : 0;
    for (int i = 0; i + 1 < n && nc < 6; ++i) {
        if (max(g.wakeB[i].y, g.wakeB[i + 1].y) <= 0.0) continue;
        vec3 a = wakePoint(i), b = wakePoint(i + 1);
        float rad = 0.5 * length(b - a) + 3.0 * (9.0 + 1.4 * g.wake[i + 1].w);
        vec3 oc = 0.5 * (a + b) - ro;
        float tc = dot(oc, rd);
        if (tc < -rad) continue;
        vec3 perp = oc - rd * tc;
        if (dot(perp, perp) > rad * rad) continue;
        cand[nc++] = i;
    }
    return nc;
}

// x as the wake has stirred it (the cloud is looked up there), and how much of the cloud at x
// the wake has cleared (0..1).
vec3 wakeWarp(vec3 x, int cand[6], int nc, out float carve) {
    carve = 0.0;
    vec3 best = x;
    float bw = 0.0;
    for (int k = 0; k < nc; ++k) {
        int i = cand[k];
        vec3 a = wakePoint(i), ab = wakePoint(i + 1) - a;
        float s = clamp(dot(x - a, ab) / max(dot(ab, ab), 1e-3), 0.0, 1.0);
        vec3 c = a + ab * s, r = x - c;
        float age = mix(g.wake[i].w, g.wake[i + 1].w, s);
        float st = mix(g.wakeB[i].y, g.wakeB[i + 1].y, s) * smoothstep(100.0, 60.0, age);
        float R = 9.0 + 1.4 * age;
        float d2 = dot(r, r) / (R * R);
        if (d2 > 9.0 || st <= 0.0) continue;
        float w = st * exp(-1.2 * d2);
        carve = max(carve, w);
        if (w > bw) {
            bw = w;
            vec3 u = normalize(ab);
            float th = 2.2 * st * exp(-0.5 * d2);
            best = c + r * cos(th) + cross(u, r) * sin(th) + u * dot(u, r) * (1.0 - cos(th));
        }
    }
    carve *= 0.96;
    return best;
}

// The sky's radiance from x towards rd, coarsely (N steps, packed towards x): for the light of
// the whole dome on a cloud, which wants its amount, not its gradients (atmosphere() takes 20
// steps, each with the sun's path through the air: three of those per ray cost as much as the
// clouds themselves).
vec3 airLite(vec3 x, vec3 rd, float tmax, vec3 L, float sunI, int N, out vec3 T) {
    T = vec3(1.0);
    float t1 = min(raySphere(x, rd, R_TOP).y, tmax);
    if (t1 <= 0.0) return vec3(0.0);
    vec2 od = vec2(0.0);
    vec3 sumR = vec3(0.0), sumM = vec3(0.0);
    for (int i = 0; i < N; ++i) {
        float ua = float(i) / float(N), ub = float(i + 1) / float(N);
        float ta = t1 * ua * ua, tb = t1 * ub * ub;
        vec3 y = x + rd * (0.5 * (ta + tb));
        float h = max(length(y) - R_GROUND, 0.0);
        vec2 d = vec2(exp(-h / H_R), exp(-h / H_M)) * (tb - ta);
        vec3 tr = extinction(od + 0.5 * d + lightDepth(y, L));
        od += d;
        sumR += d.x * tr;
        sumM += d.y * tr;
    }
    float mu = dot(rd, L);
    float pR = 3.0 / (16.0 * PI) * (1.0 + mu * mu);
    float g2 = MIE_G * MIE_G;
    float pM = 3.0 / (8.0 * PI) * ((1.0 - g2) * (1.0 + mu * mu)) / ((2.0 + g2) * pow(1.0 + g2 - 2.0 * MIE_G * mu, 1.5));
    T = extinction(od);
    return sunI * (sumR * BETA_R * pR + sumM * BETA_M * pM);
}
vec3 skyLite(vec3 x, vec3 rd, vec3 L, float sunI) {
    vec3 T;
    return airLite(x, rd, 1e30, L, sunI, 4, T);
}

// One layer's slab [rb, rt] (radii) along ro + rd t within tmax: (t0, t1), empty if t1 <= t0.
vec2 cloudSlab(vec3 ro, vec3 rd, float rb, float rt, float tmax) {
    float r0 = length(ro);
    vec2 top = raySphere(ro, rd, rt), base = raySphere(ro, rd, rb);
    float t0, t1;
    if (r0 < rb) {
        t0 = base.y; t1 = top.y;
    } else if (r0 > rt) {
        if (top.y < 0.0) return vec2(0.0, -1.0);
        t0 = max(top.x, 0.0);
        t1 = (base.x > 0.0 && base.y > 0.0) ? base.x : top.y;
    } else {
        t0 = 0.0;
        t1 = (base.x > 0.0 && base.y > 0.0) ? base.x : top.y;
    }
    return vec2(t0, min(t1, tmax));
}

// March the clouds along ro + rd * [0, tmax] (ro relative to the planet centre): in-scattered
// light (rgb, before the air in front of it) and transmittance (a); tFirst: where the ray met the
// first cloud (< 0: none). `steps`: 40 for primary rays, fewer for reflections; `jitter` in
// [0, 1) per pixel and frame.
vec4 cloudScatter(vec3 ro, vec3 rd, float tmax, int steps, float jitter, out float tFirst) {
    tFirst = -1.0;
    if (!cloudsOn()) return vec4(0.0, 0.0, 0.0, 1.0);
    vec3 L = g.sunDir.xyz;
    float mu = dot(rd, L);
    // phase, with Wrenninge's octaves (a = 0.5 attenuation, b = 0.5 contribution, c = 0.5 eccentricity)
    // (four octaves: each a later generation of bounces inside the cloud, which reaches deeper
    // (extinction x 0.25 per octave), adds half as much and scatters more evenly. With fewer, or a
    // gentler falloff, the sunlight dies within tens of metres and the clouds are lit by the sky
    // alone: flat grey. A cumulus is bright to its core because light bounces through it.)
    float ph[4] = float[4](dropletPhase(mu, 1.0), dropletPhase(mu, 0.5), dropletPhase(mu, 0.25), dropletPhase(mu, 0.125));
    float T = 1.0;
    vec3 S = vec3(0.0);
    float q = float(steps) / 40.0;
    // (opaque enough to stop: 1% for the view, 4% in a reflection, 10% for the sky's light)
    float tStop = q > 0.75 ? 0.01 : q > 0.3 ? 0.04 : 0.1;
    vec3 sky = vec3(-1.0); // the sky light over the clouds: once per ray (it varies over 100s of km)
    // --test-flight: the cloud close to the camera is what one flies through. The samples crowd
    // towards the camera (the first few metres apart, not a kilometre), and the wake stirs it.
    bool craft = g.acState.x > 0.5;
    int cand[6];
    int nc = craft ? wakeNear(ro, rd, cand) : 0;
    for (int layer = 0; layer < 3; ++layer) {
        float rb = R_GROUND + (layer == 0 ? LOW_BASE : layer == 1 ? MID_BASE : HIGH_BASE);
        float rt = R_GROUND + (layer == 0 ? LOW_TOP : layer == 1 ? MID_TOP : HIGH_TOP);
        vec2 sl = cloudSlab(ro, rd, rb, rt, tmax);
        if (sl.y <= sl.x) continue;
        sl.y = min(sl.y, sl.x + (layer == 2 ? 150000.0 : 100000.0) * (q < 0.3 ? 0.4 : 1.0)); // edge-on: beyond is haze
        // The weather of the stretch, blurred (~160 km, the map's mip 4), at its ends and middle:
        // is there any cloud of this layer, and how high do the low clouds stand here? Where they
        // are fair-weather cumulus the march stops at their tops (~3.5 km), not at the 11 km a
        // thunderstorm reaches: a third of the way, in steps a third as long.
        vec4 Wa = weatherAtLod(cloudEarth(ro + rd * sl.x), 4.0), Wb = weatherAtLod(cloudEarth(ro + rd * sl.y), 4.0);
        vec4 Wm = weatherAtLod(cloudEarth(ro + rd * (0.5 * (sl.x + sl.y))), 4.0);
        vec4 Wx = max(max(Wa, Wb), Wm);
        if (layer == 0 && Wx.x <= 0.0) continue;
        if (layer == 1 && Wx.z <= 0.0) continue;
        // (cirrus under ~10% cover: its noise never reaches the threshold, there is none to see)
        if (layer == 2 && Wx.w < 0.1) continue;
        if (layer == 0 && Wx.y < 0.55) {
            sl = cloudSlab(ro, rd, rb, R_GROUND + lowTop(0.65), tmax);
            if (sl.y <= sl.x) continue;
            sl.y = min(sl.y, sl.x + 100000.0);
        }
        float len = sl.y - sl.x;
        int n = int(float(layer == 0 ? 40 : layer == 1 ? 14 : 6) * q) + 2;
        float pw = craft ? mix(2.4, 1.0, smoothstep(0.0, 3000.0, sl.x)) : 1.0;
        float sigma = layer == 0 ? SIGMA_LOW : layer == 1 ? SIGMA_MID : SIGMA_HIGH;
        // light at the layer: the sun through the air above it, the sky above (atmosphere) and
        // the sunlit ground below (albedo ~0.15), once per layer
        // (worked out at the layer's first cloud sample: most rays through a broken sky meet none)
        vec3 xm = ro + rd * (0.5 * (sl.x + sl.y));
        vec3 up = normalize(xm);
        vec3 Esun = vec3(0.0), bounce = vec3(0.0);
        bool lightReady = false;
        float odPrev = -1.0;
        int lit = 0;
        // Two speeds (Schneider, Nubis): long steps on the cheap shape alone through the clear air
        // between clouds; on touching one, back up a long step and go on in short steps with the
        // detail until the ray has been clear for a while. The detail goes where the clouds are,
        // nothing is spent on the empty sky between cumulus, and the short step follows a pixel's
        // footprint (fine on near clouds, ~100 m on the ones at the horizon). (The shape alone is
        // never thinner than with the detail, which only erodes it: no cloud is stepped over.)
        bool adaptive = !craft && layer < 2;
        float dsBig = len / float(n);
        float tc = sl.x + jitter * dsBig;
        float tDone = sl.x; // marched in short steps up to here: never backed into again
        int fine = 0;
        int miss = 0;       // short steps in a row that found no cloud
        for (int i = 0; i < (adaptive ? 3 * n : n); ++i) {
            float t, ds;
            if (adaptive) {
                if (tc >= sl.y) break;
                if (fine == 0) {
                    vec3 xc = ro + rd * tc;
                    vec4 Wc = weatherAt(cloudEarth(xc));
                    float hc = length(xc) - R_GROUND;
                    float dc = layer == 0 ? lowDensity(cloudEarth(xc), hc, Wc, false) : midDensity(cloudEarth(xc), hc, Wc, false);
                    if (dc <= 0.0) {
                        tc += dsBig;
                        continue;
                    }
                    tc = max(tc - dsBig, tDone);
                    fine = 8;
                    miss = 0;
                }
                // (fewer steps asked for: the sky light's and reflections' rays, coarser)
                ds = min(max(tc * 0.005, (layer == 0 ? 40.0 : 80.0) / q), dsBig * 0.5);
                // The shape alone reaches far beyond the cloud the detail leaves (a fringe often
                // wider than the cloud): through it the step lengthens with every sample that
                // finds nothing, up to 4x; on meeting cloud in a long step the march goes back to
                // where that step began and on in short ones. The fringe in a few strides, the
                // cloud sampled as finely as before.
                ds *= float(min(1 + miss, 4));
                t = tc + jitter * ds;
                tc += ds;
                tDone = tc;
                --fine;
            } else {
                float ta = len * pow(float(i) / float(n), pw);
                ds = len * pow(float(i + 1) / float(n), pw) - ta;
                t = sl.x + ta + jitter * ds;
            }
            vec3 x = ro + rd * t;
            // (the coarse rays are blurred by what they are for: rippled water, the light of a
            // whole sky; they read the noise at ~100 m)
            cFoot = max(t * cloudPixAngle(), q < 0.75 ? 100.0 : 0.0);
            float carve = 0.0;
            if (nc > 0 && layer < 2) x = wakeWarp(x, cand, nc, carve);
            float h = length(x) - R_GROUND;
            vec3 e = cloudEarth(x);
            vec4 W = weatherAt(e);
            float d = layer == 0 ? lowDensity(e, h, W, true) : layer == 1 ? midDensity(e, h, W, true) : highDensity(e, h, W);
            d *= 1.0 - carve;
            if (d <= 0.0) {
                ++miss;
                continue;
            }
            if (adaptive && miss > 0) {
                tc -= ds;
                tDone = tc;
                miss = 0;
                fine = 8;
                continue;
            }
            if (adaptive) fine = 8;
            if (tFirst < 0.0) tFirst = t;
            if (!lightReady) {
                lightReady = true;
                // light at the layer: the sun through the air above it, the sky above and the
                // sunlit surface below
                Esun = g.sunDir.w * extinction(lightDepth(xm, L));
                if (sky.x < 0.0) {
                    // The sky's irradiance on the cloud: the whole dome, not the zenith alone (the
                    // horizon is two to three times brighter, and round the sun brighter still):
                    // the zenith and 25 degrees up towards the sun, coarsely, the side away from
                    // the sun taken as 1.3 x the zenith; weighted by their shares of the dome.
                    // Once per ray.
                    vec3 hz = normalize(L - up * dot(L, up) + vec3(1e-5, 0.0, 0.0));
                    sky = (0.75 * skyLite(xm, up, L, g.sunDir.w) + 0.33 * skyLite(xm, normalize(hz * 0.906 + up * 0.423), L, g.sunDir.w)) * PI;
                }
                if (cloudsLocal()) {
                    // The forest scenes' light is not the atmosphere's: their own sun (main.rgen
                    // sunColor) and sky fill, in their sky's brightness (CLOUD_LOCAL_GAIN).
                    Esun = vec3(1.0, 0.92, 0.80) * g.sunDir.w * smoothstep(-0.02, 0.1, L.y);
                    sky = vec3(0.13, 0.20, 0.36) * 1.6;
                }
                // Light from below: the sunlit surface under the layer, sea ~0.06, land ~0.15, and
                // where the layer is broken cumulus over more cloud, the cloud deck's ~0.6 (clouds
                // light each other's flanks and bases)
                float below = mix(mix(0.06, 0.15, landAt(normalize(cloudEarth(xm)))), 0.6, clamp(Wm.x * 1.2, 0.0, 1.0) * (layer > 0 ? 1.0 : 0.4));
                bounce = Esun * max(dot(up, L), 0.0) * below;
            }
            float hn = layer == 0 ? (h - LOW_BASE) / (lowTop(W.y) - LOW_BASE) : layer == 1 ? (h - MID_BASE) / (MID_TOP - MID_BASE) : 0.5;
            // sunlight: optical depth to the sun, Wrenninge's octaves
            // (the march towards the sun changes little from one sample to the next: every other
            // sample reuses it)
            float odL;
            if (layer == 2) odL = d * 600.0;
            // (coarse rays: one in four; faint fringe samples, which add little light, reuse it too)
            // (and once most of the view is covered: what is left adds under a third)
            else if (((lit & 1) == 1 || (q < 0.3 && (lit & 3) != 0) || d < 0.05 || T < 0.3) && odPrev >= 0.0) odL = odPrev;
            else odL = lightDepthCloud(x, L, layer, W, layer == 0 ? lowCover(e, W) : 0.0, q > 0.5);
            odPrev = odL;
            ++lit;
            float sunS = 0.0, a = 1.0, b = 1.0;
            for (int o = 0; o < 4; ++o) {
                sunS += b * ph[o] * exp(-sigma * a * odL);
                a *= 0.3;
                b *= 0.45;
            }
            // powder: thin edges catch little light scattered towards the eye, seen away from the sun
            float powder = 1.0 - exp(-sigma * d * 400.0);
            sunS *= mix(1.0, powder, 0.65 * (0.5 - 0.5 * mu));
            // Sunlight scattered many times: most of a cloud's light. A thick cloud turns back
            // 60-90% of the sunlight that falls on it, in all directions (white from every side
            // that sees the sun, the brightest thing on Earth from space), and passes the rest
            // down diffused (grey bases). Two-stream, without absorption: under a sunlit surface
            // the diffuse field is ~E mu0 / pi; it falls with the optical depth from the sun's
            // side as 1 / (1 + 0.75 tau (1 - g)), g ~ 0.85 (the light that has diffused that
            // deep); and a thin column lets most of the sunlight through instead of turning it
            // back: by its albedo R = x / (1 + x), x = 0.75 tau_col (1 - g). (tau from the march
            // towards the sun, plus what lies above its reach; tau_col from the density here: a
            // cumulus is denser and deeper than a sheet of stratus, a front more than both.)
            float top = layer == 0 ? lowTop(W.y) : layer == 1 ? MID_TOP : HIGH_TOP;
            float reach = layer == 0 ? 1860.0 : 3720.0;
            float tauUp = sigma * (odL + max(top - h - reach, 0.0) * (layer == 0 ? W.x : W.z) * 0.4);
            float diff = 1.0 / (1.0 + 0.75 * 0.15 * tauUp);
            float din = layer == 0 ? max(cInner, d) : d;
            float xcol = 0.75 * 0.15 * sigma * din * 0.5 * (layer == 0 ? top - LOW_BASE : layer == 1 ? MID_TOP - MID_BASE : 4000.0);
            vec3 ms = Esun * max(dot(up, L), 0.0) * (0.75 / PI) * diff * xcol / (1.0 + xcol);
            // the sky's light and the sunlit surface's bounce from below, diffused likewise
            vec3 amb = (sky * diff + bounce * (1.0 - hn) * 0.6) / (4.0 * PI) + ms;
            vec3 Li = Esun * sunS + amb;
            float Ts = exp(-sigma * d * ds);
            S += T * Li * (1.0 - Ts);                  // single-scattering albedo ~1
            T *= Ts;
            if (T < tStop) break;
        }
        if (T < tStop) break;
    }
    // (the forest scenes' painted sky is some three times as bright as the atmosphere's single
    // scattering for the same sun: their clouds are lit to match it)
    if (cloudsLocal()) S *= CLOUD_LOCAL_GAIN;
    return vec4(S, T);
}

// The globe: the clouds along ro + rd * [0, tmax] with the air in front of them (rgb to add, a
// the transmittance of what is behind).
vec4 cloudMarch(vec3 ro, vec3 rd, float tmax, int steps, float jitter) {
    float tFirst;
    vec4 c = cloudScatter(ro, rd, tmax, steps, jitter, tFirst);
    if (tFirst < 0.0) return vec4(0.0, 0.0, 0.0, 1.0);
    if (steps < 30) {
        // (a reflection: the air in front of the cloud in 6 steps)
        vec3 Tl;
        vec3 Sl = airLite(ro, rd, tFirst, g.sunDir.xyz, g.sunDir.w, 6, Tl);
        return vec4(c.rgb * Tl + Sl * (1.0 - c.a), c.a);
    }
    // The air in front of the cloud: dims the cloud and adds its own haze. The background (sky or
    // ground) already carries that haze, and the cloud covers (1 - T) of it: added back, so far
    // clouds fade into the sky colour instead of turning brown.
    vec3 Ta;
    vec3 Sa = atmosphere(ro, rd, tFirst, g.sunDir.xyz, g.sunDir.w, Ta);
    return vec4(c.rgb * Ta + Sa * (1.0 - c.a), c.a);
}

#endif
