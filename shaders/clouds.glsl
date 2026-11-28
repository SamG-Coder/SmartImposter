// Volumetric clouds (main.rgen: the globe, and the forest scenes over a sphere under them), after Schneider's Nubis (Horizon Zero Dawn / Forbidden
// West), Hillaire's Frostbite sky and Wrenninge's multiple-scattering approximation:
//
// - Three layers, fixed to the Earth (the camera frame turns into Earth axes through g.earthX..Z):
//   low clouds from 600 m (stratus and stratocumulus sheets ~1 km thick, cumulus to ~3 km,
//   cumulonimbus towers to 11 km, by cloud type), altocumulus at 3.5-5 km, cirrus at 8-9.5 km.
// - Where they are (weatherAt): the climate of real cloud cover (ISCCP / MODIS): the equatorial
//   band of towering convection (ITCZ), clear subtropics and deserts (sinking air), marine
//   stratocumulus decks off subtropical west coasts (ocean with land not far to the east), the
//   cloudy storm tracks of the mid-latitudes, polar stratus; moving weather systems of ~800 km and
//   cells of ~12 km that grow and decay over tens of minutes.
// - Shapes: a tiling Perlin-Worley base eroded by Worley fBm (cloudnoise.rs, 3D textures 34, 35),
//   shaped by a height profile per type (flat bases, rounded tops), wispy below, billowy above.
// - Light: sunlight marched towards the sun, summed over Wrenninge's scattering octaves (multiple
//   scattering: bright, soft interiors), Jendersie & d'Eon's droplet phase function (HG + Draine:
//   the sharp forward peak of the silver lining, fogbow and glory lobes), the "powder" darkening of
//   thin edges seen away from the sun, sky light from the atmosphere above and the sunlit ground's
//   bounce below. Clouds shadow the ground (cloudShadow) and dim the sky light under them.
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
vec4 weatherAt(vec3 e) {
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
    vec3 u = normalize(e);
    float lat = degrees(asin(clamp(u.z, -1.0, 1.0)));
    float a = abs(lat);
    float landF = landAt(u);
    // climate bands
    float itcz = exp(-pow((lat - 6.0) / 7.0, 2.0));                        // north of the equator
    float subtrop = smoothstep(10.0, 18.0, a) * (1.0 - smoothstep(30.0, 40.0, a));
    float storm = smoothstep(35.0, 45.0, a) * (1.0 - smoothstep(65.0, 75.0, a));
    float polar = smoothstep(62.0, 72.0, a);
    // marine stratocumulus: subtropical ocean with land 4-16 degrees to the east (west coasts)
    float deck = 0.0;
    if (landF < 0.9 && a > 8.0 && a < 38.0) {
        vec3 east = normalize(cross(vec3(0.0, 0.0, 1.0), u));
        float land = 0.0;
        // nearer land counts more: the deck thins out westward, away from the coast
        for (int i = 1; i <= 6; ++i) land = max(land, landAt(normalize(u + east * (0.035 * float(i)))) * (1.12 - 0.15 * float(i)));
        land = smoothstep(0.0, 1.0, land) * (0.75 + 0.5 * cnoise3(e / 300.0 + 4.0));
        deck = (1.0 - landF) * land * smoothstep(8.0, 16.0, a) * (1.0 - smoothstep(30.0, 38.0, a));
    }
    // weather systems: fronts and highs ~800 km, moving and changing over hours
    vec3 q = e / 800.0 + vec3(0.0, 0.0, cloudTime() / 20000.0);
    float sys = cnoise3(q) * 0.65 + cnoise3(q * 2.3 + 7.7) * 0.35;
    // The subtropics are clear only where the air sinks dry: the continental interiors (deserts,
    // land ~500 km around) and the cool eastern oceans (where the decks form). The trade-wind
    // oceans and the humid coasts (Florida, the Caribbean, south China) have fair-weather cumulus
    // most days (~20-35% cover).
    float inland = smoothstep(0.5, 0.9, landAtLod(u, 9.0));
    float dry = max(inland, deck);
    float trade = subtrop * (1.0 - dry);
    // (broken cumulus on a typical day; overcast only where the systems and the climate agree)
    // (decks: closed cells, ~75% cover, not a featureless sheet)
    // (the trades: the same scattered cumulus day after day, rarely overcast)
    float c = 0.16 + 0.25 * itcz + 0.22 * storm + 0.2 * polar + 0.5 * deck - 0.4 * subtrop * dry - 0.05 * trade;
    c += (sys - 0.5) * (0.8 + 0.4 * storm) * (1.0 - 0.55 * trade);
    c = clamp(c * (g.cloud.x / 0.45), 0.0, 1.0);
    // type: towering over the ITCZ (and over warm land on unsettled days), sheets in the decks and
    // the polar regions, cumulus in the trades, mixed in the storm tracks
    float type = 0.5 + 0.45 * itcz * smoothstep(0.4, 0.7, sys) - 0.5 * deck - 0.35 * polar - 0.2 * storm * (1.0 - sys);
    type = clamp(type + 0.2 * landF * (1.0 - subtrop) * smoothstep(0.55, 0.8, sys), 0.0, 1.0);
    // altocumulus and cirrus with the weather: ahead of fronts in the storm tracks, the ITCZ's
    // anvils; little in the subtropical highs
    float mid = clamp((0.6 * storm + 0.35 * itcz) * smoothstep(0.45, 0.75, sys), 0.0, 1.0) * (g.cloud.x / 0.45);
    float high = clamp((0.08 + 0.45 * itcz + 0.45 * storm * smoothstep(0.35, 0.75, sys)) * (1.0 - 0.7 * subtrop), 0.0, 1.0) * min(g.cloud.x / 0.45, 1.5);
    return vec4(c, type, mid, high);
}

// Top of the low layer for a cloud type (m above sea level): stratus ~1.5 km, cumulus ~3 km,
// cumulonimbus to 11 km.
float lowTop(float type) {
    return mix(mix(LOW_BASE + 900.0, LOW_BASE + 2600.0, smoothstep(0.0, 0.5, type)), LOW_TOP, smoothstep(0.6, 1.0, type));
}

// Low layer density at e (km, Earth-fixed) and height h (m), weather W; `detail`: the fine
// erosion (off for the light and shadow marches).
float lowDensity(vec3 e, float h, vec4 W, bool detail) {
    float top = lowTop(W.y);
    float hn = (h - LOW_BASE) / (top - LOW_BASE);
    if (hn <= 0.0 || hn >= 1.0 || W.x <= 0.0) return 0.0;
    vec3 p = e - cloudDrift(e, 8.0);
    float t = cloudTime();
    // cells of ~12 km that grow and decay (the tile's z axis is time)
    float cell = textureLod(cloudShape, p / 48.0 + vec3(0.0, 0.0, t / 3000.0), max(log2(cFoot / 375.0), 0.0)).r;
    float cov = clamp(W.x + (cell - 0.5) * 0.8, 0.0, 1.0);
    if (cov <= 0.02) return 0.0;
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
float lightDepthCloud(vec3 x, vec3 L, int layer, vec4 W, bool fine) {
    float od = 0.0, t = 0.0, dl = layer == 0 ? 25.0 : 120.0;
    for (int j = 0; j < 7; ++j) {
        t += dl;
        vec3 y = x + L * (t - 0.5 * dl);
        float h = length(y) - R_GROUND;
        vec3 e = cloudEarth(y);
        float d = layer == 0 ? lowDensity(e, h, W, fine && j < 3) : midDensity(e, h, W, false);
        od += d * dl;
        dl *= 2.0;
    }
    return od;
}

// Sun transmittance through the clouds from x (planet-centred) towards L: the cloud shadows.
float cloudShadow(vec3 x, vec3 L) {
    if (!cloudsOn()) return 1.0;
    float h0 = length(x) - R_GROUND;
    if (h0 > HIGH_TOP) return 1.0;
    vec4 W = weatherAt(cloudEarth(x));
    float od = 0.0;
    // low and mid layers: where the sun ray crosses them, 8 coarse samples
    vec2 bot = raySphere(x, L, R_GROUND + LOW_BASE), top = raySphere(x, L, R_GROUND + lowTop(W.y));
    float t0 = h0 < LOW_BASE ? max(bot.y, 0.0) : 0.0, t1 = min(top.y, t0 + 15000.0);
    if (t1 > t0) {
        float ds = (t1 - t0) / 8.0;
        for (int i = 0; i < 8; ++i) {
            vec3 y = x + L * (t0 + (float(i) + 0.5) * ds);
            // (with the detail: the coarse shape alone has thin fringes far beyond the cloud one
            // sees, which the detail erodes away; without it the shadows spread well past the
            // clouds and the sunlit gaps between them close)
            float d = lowDensity(cloudEarth(y), length(y) - R_GROUND, W, true);
            od += d * ds * SIGMA_LOW;
        }
    }
    vec2 mb = raySphere(x, L, R_GROUND + MID_BASE), mt = raySphere(x, L, R_GROUND + MID_TOP);
    if (h0 < MID_BASE && mt.y > mb.y) {
        float ds = (mt.y - mb.y) / 3.0;
        for (int i = 0; i < 3; ++i) {
            vec3 y = x + L * (mb.y + (float(i) + 0.5) * ds);
            float d = midDensity(cloudEarth(y), length(y) - R_GROUND, W, false);
            if (cloudsLocal()) d = clamp((d - 0.2) / 0.8, 0.0, 1.0);
            od += d * ds * SIGMA_MID;
        }
    }
    vec2 hb = raySphere(x, L, R_GROUND + HIGH_BASE + 700.0);
    if (h0 < HIGH_BASE && hb.y > 0.0) {
        vec3 y = x + L * hb.y;
        od += highDensity(cloudEarth(y), length(y) - R_GROUND, W) * 1500.0 * SIGMA_HIGH;
    }
    // (multiple scattering lets ~30% more light through a cloud than single extinction says)
    return exp(-0.7 * od);
}

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
        sl.y = min(sl.y, sl.x + (layer == 2 ? 150000.0 : 100000.0)); // edge-on: beyond is haze
        float len = sl.y - sl.x;
        // weather at the segment's two ends, interpolated along it (it varies over 100s of km)
        vec4 Wa = weatherAt(cloudEarth(ro + rd * sl.x)), Wb = weatherAt(cloudEarth(ro + rd * sl.y));
        if (layer == 0 && max(Wa.x, Wb.x) <= 0.0) continue;
        if (layer == 1 && max(Wa.z, Wb.z) <= 0.0) continue;
        if (layer == 2 && max(Wa.w, Wb.w) <= 0.0) continue;
        int n = int(float(layer == 0 ? 56 : layer == 1 ? 14 : 6) * q) + 2;
        float pw = craft ? mix(2.4, 1.0, smoothstep(0.0, 3000.0, sl.x)) : 1.0;
        float sigma = layer == 0 ? SIGMA_LOW : layer == 1 ? SIGMA_MID : SIGMA_HIGH;
        // light at the layer: the sun through the air above it, the sky above (atmosphere) and
        // the sunlit ground below (albedo ~0.15), once per layer
        vec3 xm = ro + rd * (0.5 * (sl.x + sl.y));
        vec3 up = normalize(xm);
        vec3 Esun = g.sunDir.w * extinction(lightDepth(xm, L));
        if (sky.x < 0.0) {
            // The sky's irradiance on the cloud: the whole dome, not the zenith alone (the
            // horizon is two to three times brighter, and round the sun brighter still): the
            // zenith, and 25 degrees up towards and away from the sun, weighted by the share of
            // the dome each stands for. Once per ray.
            vec3 Tz;
            vec3 hz = normalize(L - up * dot(L, up) + vec3(1e-5, 0.0, 0.0));
            vec3 dTo = normalize(hz * 0.906 + up * 0.423), dAway = normalize(-hz * 0.906 + up * 0.423);
            sky = (0.40 * atmosphere(xm, up, 1e30, L, g.sunDir.w, Tz) + 0.33 * atmosphere(xm, dTo, 1e30, L, g.sunDir.w, Tz)
                 + 0.27 * atmosphere(xm, dAway, 1e30, L, g.sunDir.w, Tz)) * PI;
        }
        if (cloudsLocal()) {
            // The forest scenes' light is not the atmosphere's: their own sun (main.rgen
            // sunColor) and sky fill, in their sky's brightness (CLOUD_LOCAL_GAIN).
            Esun = vec3(1.0, 0.92, 0.80) * g.sunDir.w * smoothstep(-0.02, 0.1, L.y);
            sky = vec3(0.13, 0.20, 0.36) * 1.6;
        }
        // Light from below: the sunlit surface under the layer, sea ~0.06, land ~0.15, and where
        // the layer is broken cumulus over more cloud, the cloud deck's ~0.6 (clouds light each
        // other's flanks and bases)
        vec4 Wm = mix(Wa, Wb, 0.5);
        float below = mix(mix(0.06, 0.15, landAt(normalize(cloudEarth(xm)))), 0.6, clamp(Wm.x * 1.2, 0.0, 1.0) * (layer > 0 ? 1.0 : 0.4));
        vec3 bounce = Esun * max(dot(up, L), 0.0) * below;
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
        for (int i = 0; i < (adaptive ? 3 * n : n); ++i) {
            float t, ds;
            if (adaptive) {
                if (tc >= sl.y) break;
                if (fine == 0) {
                    vec3 xc = ro + rd * tc;
                    vec4 Wc = mix(Wa, Wb, (tc - sl.x) / max(len, 1.0));
                    float hc = length(xc) - R_GROUND;
                    float dc = layer == 0 ? lowDensity(cloudEarth(xc), hc, Wc, false) : midDensity(cloudEarth(xc), hc, Wc, false);
                    if (dc <= 0.0) {
                        tc += dsBig;
                        continue;
                    }
                    tc = max(tc - dsBig, tDone);
                    fine = 8;
                }
                // (fewer steps asked for: the sky light's and reflections' rays, coarser)
                ds = min(max(tc * 0.005, (layer == 0 ? 40.0 : 80.0) / q), dsBig * 0.5);
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
            cFoot = t * cloudPixAngle();
            float carve = 0.0;
            if (nc > 0 && layer < 2) x = wakeWarp(x, cand, nc, carve);
            float h = length(x) - R_GROUND;
            vec3 e = cloudEarth(x);
            vec4 W = mix(Wa, Wb, (t - sl.x) / max(len, 1.0));
            float d = layer == 0 ? lowDensity(e, h, W, true) : layer == 1 ? midDensity(e, h, W, true) : highDensity(e, h, W);
            d *= 1.0 - carve;
            if (d <= 0.0) continue;
            if (adaptive) fine = 8;
            if (tFirst < 0.0) tFirst = t;
            float hn = layer == 0 ? (h - LOW_BASE) / (lowTop(W.y) - LOW_BASE) : layer == 1 ? (h - MID_BASE) / (MID_TOP - MID_BASE) : 0.5;
            // sunlight: optical depth to the sun, Wrenninge's octaves
            // (the march towards the sun changes little from one sample to the next: every other
            // sample reuses it)
            float odL;
            if (layer == 2) odL = d * 600.0;
            // (coarse rays: one in four; faint fringe samples, which add little light, reuse it too)
            else if (((lit & 1) == 1 || (q < 0.3 && (lit & 3) != 0) || d < 0.05) && odPrev >= 0.0) odL = odPrev;
            else odL = lightDepthCloud(x, L, layer, W, q > 0.5);
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
            // Light diffused down through the cloud above (multiple scattering: a cloud of optical
            // depth tau passes ~1 / (1 + 0.75 tau (1 - g)) of it, g ~ 0.85): the sun's and the
            // sky's, plus the sunlit ground's bounce from below. Dark bases, bright tops.
            // (tau from the march towards the sun: the cloud's own thick and thin parts show as
            // darker and lighter patches on its base, plus what lies above that march's reach)
            float top = layer == 0 ? lowTop(W.y) : layer == 1 ? MID_TOP : HIGH_TOP;
            float reach = layer == 0 ? 1860.0 : 3720.0;
            float tauUp = sigma * (odL + max(top - h - reach, 0.0) * (layer == 0 ? W.x : W.z) * 0.4);
            float diff = 1.0 / (1.0 + 0.75 * 0.15 * tauUp);
            vec3 amb = (sky * diff + Esun * max(dot(up, L), 0.0) * diff * (1.0 - diff) * 2.0 + bounce * (1.0 - hn) * 0.6) / (4.0 * PI);
            vec3 Li = Esun * sunS + amb;
            float Ts = exp(-sigma * d * ds);
            S += T * Li * (1.0 - Ts);                  // single-scattering albedo ~1
            T *= Ts;
            if (T < 0.01) break;
        }
        if (T < 0.01) break;
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
    // The air in front of the cloud: dims the cloud and adds its own haze. The background (sky or
    // ground) already carries that haze, and the cloud covers (1 - T) of it: added back, so far
    // clouds fade into the sky colour instead of turning brown.
    vec3 Ta;
    vec3 Sa = atmosphere(ro, rd, tFirst, g.sunDir.xyz, g.sunDir.w, Ta);
    return vec4(c.rgb * Ta + Sa * (1.0 - c.a), c.a);
}

#endif
