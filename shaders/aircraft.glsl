// What the aircraft of --test-flight does to the air around it (main.rgen): it bends the light
// (craftBend, below), and it fills it with thin volumes integrated along the view ray:
//
// - the afterburner's flame behind the nozzle, with its shock diamonds;
// - condensation: the cloud that forms round the aircraft near the speed of sound in humid air
//   (the pressure falls behind the shocks and the air cools below its dew point), the vapour
//   over the wings and the cores of the tip and strake vortices in hard turns;
// - the condensation trail behind the engine where the air is colder than about -40 C.
//
// Tubes (the flame, the vortex cores, the trail) are Gaussian in section and integrated exactly
// (tubeDepth); the two clouds are marched in the aircraft's own frame (x right, y up, z aft,
// metres from its centre of gravity). Needs atmosphere.glsl, clouds.glsl and the Globals `g`.
#ifndef AIRCRAFT_GLSL
#define AIRCRAFT_GLSL

bool craftOn() { return g.acState.x > 0.5; }
vec3 craftDir(vec3 d) { return vec3(dot(g.acInv0.xyz, d), dot(g.acInv1.xyz, d), dot(g.acInv2.xyz, d)); }
vec3 craftPoint(vec3 x) { return craftDir(x) + vec3(g.acInv0.w, g.acInv1.w, g.acInv2.w); }

float erfA(float x) {
    float a = abs(x);
    if (a < 0.5) return 1.1283792 * x * (1.0 - x * x * (0.33333333 - 0.1 * x * x));
    float t = 1.0 / (1.0 + 0.3275911 * a);
    float y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
    return x < 0.0 ? -y : y;
}

// The integral of exp(-d^2 / R^2) along ro + rd t, t in [0, tmax], d the distance to the axis
// a..b: the depth of a tube of Gaussian section, over the stretch of the ray abreast of it.
float tubeDepth(vec3 ro, vec3 rd, float tmax, vec3 a, vec3 b, float R) {
    vec3 ab = b - a;
    float len = length(ab);
    if (len < 1e-4) return 0.0;
    vec3 u = ab / len;
    float c = dot(rd, u);
    float s2 = max(1.0 - c * c, 1e-8);
    vec3 w = ro - a;
    float wr = dot(w, rd), wu = dot(w, u);
    float t0 = (c * wu - wr) / s2; // where the ray passes closest to the axis
    vec3 off = w + rd * t0 - u * (wu + c * t0);
    float d2 = dot(off, off) / (R * R);
    if (d2 > 16.0) return 0.0;
    float ta = 0.0, tb = tmax;
    if (abs(c) > 1e-5) {
        float t1 = -wu / c, t2 = (len - wu) / c;
        ta = max(ta, min(t1, t2));
        tb = min(tb, max(t1, t2));
    } else if (wu < 0.0 || wu > len) {
        return 0.0;
    }
    if (tb <= ta) return 0.0;
    float k = sqrt(s2) / R;
    return exp(-d2) * 0.8862269 / k * (erfA((tb - t0) * k) - erfA((ta - t0) * k));
}

// The light a white vapour sends back, where the camera is: the sun's (warm) and the sky's (blue).
void vapourLights(out vec3 sun, out vec3 sky) {
    vec3 L = g.sunDir.xyz;
    if (g.planet.w > 0.5) {
        vec3 x = -g.planet.xyz, Tz;
        sky = atmosphere(x, normalize(x), 1e30, L, g.sunDir.w, Tz) * 0.9;
        sun = g.sunDir.w * extinction(lightDepth(x, L)) * 0.29;
    } else {
        sun = vec3(1.0, 0.92, 0.80) * g.sunDir.w * 0.29;
        sky = vec3(0.13, 0.20, 0.36) * 0.9;
    }
}
vec3 vapourLight() {
    vec3 sun, sky;
    vapourLights(sun, sky);
    return sun + sky;
}

// Optical depth of the condensation trail along org + dir * [0, tmax] (the renderer's frame).
// It spreads at about a metre a second and thins as it does.
float contrailDepth(vec3 org, vec3 dir, float tmax) {
    int n = int(g.acState2.y);
    float od = 0.0;
    for (int i = 0; i + 1 < n; ++i) {
        float c = 0.5 * (g.wakeB[i].x + g.wakeB[i + 1].x);
        if (c <= 0.002) continue;
        float age = 0.5 * (g.wake[i].w + g.wake[i + 1].w);
        float R = 1.0 + 0.9 * age;
        od += c * 0.75 / pow(R, 1.3) * smoothstep(95.0, 55.0, age) * tubeDepth(org, dir, tmax, g.wake[i].xyz, g.wake[i + 1].xyz, R);
    }
    return od;
}

// The air round the aircraft bends the light that passes through it (it is seen through, like
// the heat over a road): by the shock waves it trails once it is supersonic, and by its exhaust.
//
// - A shock wave is a sheet across which the air is suddenly denser, and denser air bends light:
//   a ray is refracted as it goes in through the sheet and again on its way out. Seen edge on
//   (along the cone's silhouette) the bending is strongest, so the land behind is drawn with
//   two sharp, shifted lines trailing back from the aircraft: the bow shock from the nose and the
//   one from the tail, each a cone of half angle asin(1 / Mach), a flat disc at Mach 1 that
//   folds back as the speed rises. The bending is many times the real air's (which only a
//   schlieren camera shows), so that it can be seen; it weakens with distance from the aircraft
//   and with height, as the air thins.
// - The exhaust is hot, thin, churning gas: what is seen through it shimmers.
//
// Returns how far along org + dir t the bending begins (< 0: this ray passes clear), and the
// change to its direction: main.rgen looks again along the bent ray for whatever lies behind.
float craftBend(vec3 org, vec3 dir, out vec3 bend) {
    bend = vec3(0.0);
    vec3 ro = craftPoint(org), rd = craftDir(dir);
    float bq = dot(ro, rd), cq = dot(ro, ro) - 4900.0;
    if (cq > 0.0 && (bq > 0.0 || bq * bq < cq)) return -1.0;
    vec3 flow = g.acVel.xyz;
    float mach = g.acState.z, tFirst = 1e30;
    vec3 b = vec3(0.0);
    float strong = smoothstep(0.96, 1.03, mach) * (0.3 + 0.7 * g.acState2.z);
    if (strong > 0.0) {
        float c2 = 1.0 - 1.0 / max(mach * mach, 1.06); // cos^2 of the Mach angle
        float cm = sqrt(c2);
        for (int k = 0; k < 2; ++k) {
            vec3 apex = k == 0 ? vec3(0.0, -0.2, -8.6) : vec3(0.0, 0.1, 1.6);
            vec3 oa = ro - apex;
            float da = dot(rd, flow), oaa = dot(oa, flow);
            float A = da * da - c2, B = 2.0 * (da * oaa - c2 * dot(rd, oa)), C = oaa * oaa - c2 * dot(oa, oa);
            float disc = B * B - 4.0 * A * C;
            if (disc <= 0.0 || abs(A) < 1e-6) continue;
            float sq = sqrt(disc);
            for (int r = 0; r < 2; ++r) {
                float t = (-B + (r == 0 ? -sq : sq)) / (2.0 * A);
                vec3 v = oa + rd * t;
                float along = dot(v, flow), len = length(v);
                if (t <= 0.0 || along <= 0.0 || len > 48.0) continue; // (the cone behind the apex only)
                vec3 n = normalize(flow - cm * v / len);                // towards the cone's inside: the denser air
                float eps = 0.005 * strong * smoothstep(48.0, 10.0, len) * smoothstep(0.0, 1.5, len) * (k == 0 ? 1.0 : 0.6);
                bool entering = dot(rd, n) > 0.0;
                vec3 bent = refract(rd, entering ? -n : n, entering ? 1.0 / (1.0 + eps) : 1.0 + eps);
                vec3 turn = dot(bent, bent) > 0.0 ? bent - rd : vec3(0.0);
                b += turn * min(1.0, 0.03 / max(length(turn), 1e-6)); // (edge on, the bend is capped)
                tFirst = min(tFirst, t);
            }
        }
    }
    // the exhaust: a shimmer that streams aft, stronger the harder the engine runs
    float power = g.acVel.w;
    const vec3 NOZ = vec3(0.0, 0.085, 5.85);
    vec3 axis = normalize(vec3(0.0, 0.0, 1.0) + 0.25 * flow);
    float hot = tubeDepth(ro, rd, 1e9, NOZ, NOZ + axis * (10.0 + 30.0 * power), 0.55);
    if (hot > 1e-3) {
        vec3 q = (ro + rd * max(dot(NOZ + axis * 6.0 - ro, rd), 0.0)) * 2.6 - axis * (g.acState2.w * 55.0);
        vec3 nz = vec3(cnoise3(q), cnoise3(q + 17.3), cnoise3(q.zxy + 31.7)) - 0.5;
        b += (nz - rd * dot(nz, rd)) * (0.012 + 0.02 * power) * min(hot, 1.2);
        tFirst = min(tFirst, max(dot(NOZ - ro, rd), 0.05));
    }
    if (tFirst > 1e29) return -1.0;
    // back from the aircraft's frame (the rows of the frame -> model turn are its columns)
    bend = g.acInv0.xyz * b.x + g.acInv1.xyz * b.y + g.acInv2.xyz * b.z;
    return tFirst;
}

// The slab lo <= p[axis] <= hi along ro + rd t: narrows (t0, t1).
void slabClip(float o, float d, float lo, float hi, inout float t0, inout float t1) {
    if (abs(d) < 1e-6) {
        if (o < lo || o > hi) t1 = -1.0;
        return;
    }
    float a = (lo - o) / d, b = (hi - o) / d;
    t0 = max(t0, min(a, b));
    t1 = min(t1, max(a, b));
}

// The flame and the vapour round the aircraft along org + dir * [0, tmax]: light to add (rgb),
// and how much of what is behind still shows (a). `jitter` in [0, 1) per pixel and frame.
vec4 craftVolumes(vec3 org, vec3 dir, float tmax, float jitter) {
    vec3 ro = craftPoint(org), rd = craftDir(dir);
    // (all of it lies within 80 m of the aircraft)
    float bq = dot(ro, rd), cq = dot(ro, ro) - 6400.0;
    if (cq > 0.0 && (bq > 0.0 || bq * bq < cq)) return vec4(0.0, 0.0, 0.0, 1.0);
    vec3 flow = g.acVel.xyz;
    vec3 add = vec3(0.0);
    float T = 1.0;

    // ---- condensation
    float cone = g.acState.w, wing = g.acState2.x;
    if (cone > 0.01 || wing > 0.01) {
        vec3 LcSun, LcSky;
        vapourLights(LcSun, LcSky);
        vec3 Lc = LcSun + LcSky;
        float tm = g.acState2.w;
        vec3 sunM = craftDir(g.sunDir.xyz); // the sun in the aircraft's frame
        if (cone > 0.01) {
            // Near the speed of sound the air speeds up over the canopy and the wing and cools
            // below its dew point, until the shock behind sets it back: a thin veil of mist in
            // the shape of a cone, open at the front, widening aft and ending sharply at the
            // shock. It is thin, ragged, never the same from one moment to the next, and
            // thicker over the top of the aircraft (where the wing's suction is).
            const float Z0 = -1.6, Z1 = 2.7;
            float t0 = 0.0, t1 = tmax;
            slabClip(ro.z, rd.z, Z0, Z1 + 0.25, t0, t1);
            slabClip(ro.x, rd.x, -4.2, 4.2, t0, t1);
            slabClip(ro.y, rd.y, -4.0, 4.4, t0, t1);
            if (t1 > t0) {
                const int N = 40;
                float ds = (t1 - t0) / float(N);
                float flick = 0.75 + 0.25 * sin(tm * 31.0) * sin(tm * 12.7 + 1.3);
                for (int i = 0; i < N; ++i) {
                    vec3 p = ro + rd * (t0 + (float(i) + jitter) * ds);
                    vec2 q = p.xy - vec2(0.0, 0.2);
                    float z = p.z - Z0, r = length(q);
                    float Rc = 0.55 + 0.62 * z, w = 0.10 + 0.07 * z;
                    float shell = exp(-(r - Rc) * (r - Rc) / (w * w));
                    // wisps drawn out along the cone, streaming aft
                    vec3 np = vec3(atan(q.y, q.x) * 2.2, r * 1.5, p.z * 0.9 - tm * 26.0);
                    float n1 = cnoise3(np), n2 = cnoise3(np * 3.1 + 7.0);
                    float wisp = smoothstep(0.30, 0.72, 0.65 * n1 + 0.35 * n2);
                    // ragged at the shock, soft at the front
                    float back = Z1 - Z0 + 0.55 * (n1 - 0.5) + 0.2 * (n2 - 0.5);
                    float along = smoothstep(0.0, 2.2, z) * (1.0 - smoothstep(back - 0.18, back, z));
                    float top = 0.55 + 0.45 * smoothstep(-0.6, 0.8, q.y / max(r, 0.2));
                    // (it stands off the skin: none on the fuselage or the wing itself)
                    float clear = smoothstep(0.85, 1.5, r) * smoothstep(0.12, 0.5, abs(q.y - 0.04) + 0.25 * step(4.8, abs(q.x)));
                    float d = cone * flick * shell * wisp * along * top * clear;
                    float a = 1.0 - exp(-d * 0.9 * ds);
                    // Not a flat white: the side of the veil towards the sun is warm and bright,
                    // the far side takes the sky's blue; and where the veil is thinnest it shows
                    // the pastel fringes of thin cloud near the sun (droplets all of one size
                    // bend each colour by its own angle).
                    float toSun = smoothstep(-0.5, 0.8, dot(normalize(vec3(q, -0.4 * r)), sunM));
                    vec3 colr = LcSun * (0.15 + 1.05 * toSun) + LcSky * (1.9 - 1.1 * toSun);
                    float edge = abs(r - Rc) / w;
                    vec3 fringe = 0.5 + 0.5 * cos(6.2831853 * (0.55 * edge + 0.25 * n2 + vec3(0.0, 0.33, 0.67)));
                    colr *= mix(vec3(1.0), 0.35 + 1.3 * fringe, 0.85 * smoothstep(0.3, 1.2, edge));
                    add += T * colr * a;
                    T *= 1.0 - a;
                }
            }
        }
        if (wing > 0.01) {
            // In a hard turn the suction over the wings does the same: a thin sheet of mist
            // lying on the upper surface, in streaks along the flow, torn at its edges.
            float t0 = 0.0, t1 = tmax;
            slabClip(ro.y, rd.y, 0.02, 1.0, t0, t1);
            slabClip(ro.x, rd.x, -4.7, 4.7, t0, t1);
            slabClip(ro.z, rd.z, -2.6, 2.9, t0, t1);
            if (t1 > t0) {
                const int N = 22;
                float ds = (t1 - t0) / float(N);
                for (int i = 0; i < N; ++i) {
                    vec3 p = ro + rd * (t0 + (float(i) + jitter) * ds);
                    float ax = abs(p.x);
                    float le = -2.71 + 0.839 * ax;
                    float plan = smoothstep(le + 0.1, le + 0.9, p.z) * smoothstep(2.6, 1.6, p.z) * smoothstep(0.8, 1.4, ax) * smoothstep(4.7, 4.2, ax);
                    float thick = 0.10 + 0.22 * wing * smoothstep(le, 2.3, p.z);
                    vec3 np = vec3(p.x * 4.5, p.y * 6.0, p.z * 0.7 - tm * 30.0);
                    float streak = smoothstep(0.42, 0.80, 0.6 * cnoise3(np) + 0.4 * cnoise3(np * 2.7 + 3.0));
                    float d = wing * plan * exp(-max(p.y - 0.06, 0.0) / thick) * streak;
                    float a = 1.0 - exp(-d * 0.38 * ds);
                    add += T * Lc * a;
                    T *= 1.0 - a;
                }
            }
            // the vortex cores: from the wing tips, and from the strakes over the wings
            float od = 0.0;
            for (int s = -1; s <= 1; s += 2) {
                vec3 tip = vec3(float(s) * 4.75, 0.04, 2.1);
                od += tubeDepth(ro, rd, tmax, tip, tip + flow * 8.0, 0.045) * 1.6;
                od += tubeDepth(ro, rd, tmax, tip + flow * 8.0, tip + flow * 28.0, 0.08) * 0.5;
                od += tubeDepth(ro, rd, tmax, tip + flow * 28.0, tip + flow * 60.0, 0.16) * 0.12;
                od += tubeDepth(ro, rd, tmax, vec3(float(s) * 1.0, 0.35, -2.6), vec3(float(s) * 1.75, 0.95, 2.8), 0.2) * 0.5;
            }
            float a = 1.0 - exp(-od * 1.3 * wing);
            add += T * Lc * a;
            T *= 1.0 - a;
        }
    }

    // ---- the afterburner: ten stretches of flame along the jet, every other one a shock diamond
    float ab = g.acState.y;
    if (ab > 0.01) {
        const vec3 NOZ = vec3(0.0, 0.085, 5.85);
        vec3 axis = normalize(vec3(0.0, 0.0, 1.0) + 0.25 * flow);
        float len = 2.5 + 6.5 * ab;
        vec3 sum = vec3(0.0);
        float od = 0.0;
        for (int k = 0; k < 10; ++k) {
            float s = (float(k) + 0.5) / 10.0;
            float o = tubeDepth(ro, rd, tmax, NOZ + axis * (len * float(k) / 10.0), NOZ + axis * (len * float(k + 1) / 10.0), mix(0.30, 0.10, s));
            o *= pow(1.0 - s, 1.3) * ((k & 1) == 0 ? 1.0 : 0.4);
            sum += o * mix(vec3(1.0, 0.36, 0.08), vec3(0.95, 0.70, 0.62), pow(1.0 - s, 3.0));
            od += o;
        }
        if (od > 1e-4) add += T * sum / od * (1.0 - exp(-0.9 * od)) * (0.5 + 2.0 * ab);
    }
    return vec4(add, T);
}

#endif
