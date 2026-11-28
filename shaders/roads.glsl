// Roads from OpenStreetMap (globe/roads.rs): the line segments of the ways round an anchor point,
// in two grids (every road within ~2 km, the main roads to ~16 km), and the road surface painted
// on the ground from them: no mesh, exact vectors at any distance.
//
// The surface is RealisticDrive's (crates/akari/shaders/road.glsl roadShade, the parts that carry
// from a road one drives on to roads seen from the air too): the asphalt scan greyed, stretches
// resurfaced at different times, wheel paths polished and darkened by tyres, oil between them,
// dust and grit along the edges, cracks, and worn paint: edge lines, lane lines 3 m on / 9 m off,
// the centre line (yellow in the Americas, white elsewhere). Unpaved tracks are two ruts in the
// local soil.
//
// Needs globals.glsl (g.road0..2) and shared.glsl (vnoise).
#ifndef ROADS_GLSL
#define ROADS_GLSL

layout(set = 0, binding = 39, scalar) readonly buffer RoadBuf { uint w[]; } roadBuf;

// (globe/roads.rs has the same numbers)
const int ROAD_GRID = 256;
const float ROAD_FINE = 16.0, ROAD_COARSE = 128.0;
const uint ROAD_IDX_AT = 2u * 256u * 256u;
const uint ROAD_SEG_AT = ROAD_IDX_AT + 1000000u;
const uint ROAD_HALF = ROAD_SEG_AT + 200000u * 6u;

const uint RC_MOTORWAY = 0u, RC_TRUNK = 1u, RC_PRIMARY = 2u, RC_SECONDARY = 3u, RC_TERTIARY = 4u, RC_STREET = 5u, RC_SERVICE = 6u, RC_TRACK = 7u, RC_LINK = 8u;
const uint RF_ONEWAY = 1u, RF_BRIDGE = 2u, RF_UNPAVED = 8u;

struct RoadHit {
    float edge;    // metres inside the road's edge (negative: outside it)
    float across;  // metres from the centre line, + to the left of the way's direction
    float along;   // metres along the way
    float hw;      // half the paved width
    uint cls, lanes, flags, seed;
    bool junction; // another road crosses here
};

bool roadsOn() { return g.road2.y > 0.5; }

// A point of the camera frame, in metres east and north of the roads' anchor.
vec2 roadXY(vec3 p) { return vec2(dot(g.road0.xyz, p) + g.road0.w, dot(g.road1.xyz, p) + g.road1.w); }

float roadLanes(uint cls, uint lanes, uint flags) {
    if (lanes > 0u) return float(lanes);
    bool oneway = (flags & RF_ONEWAY) != 0u;
    if (cls == RC_MOTORWAY) return 3.0;
    if (cls == RC_TRUNK) return 2.0;
    if (cls == RC_PRIMARY) return oneway ? 2.0 : 4.0;
    if (cls == RC_SECONDARY) return 2.0;
    if (cls == RC_TERTIARY || cls == RC_STREET) return oneway ? 1.0 : 2.0;
    return 1.0;
}

// Half the paved width (m): roads.rs half_width.
float roadHalfWidth(uint cls, uint lanes, uint flags) {
    float n = roadLanes(cls, lanes, flags);
    if (cls <= RC_TRUNK) return n * 1.85 + 1.5;
    if (cls <= RC_SECONDARY) return n * 1.7 + 0.4;
    if (cls == RC_TERTIARY) return n * 1.6 + 0.3;
    if (cls == RC_STREET) return max(n * 1.5 + 0.4, 2.8);
    if (cls == RC_SERVICE) return 1.9;
    if (cls == RC_TRACK) return 1.5;
    return n * 1.85 + 0.8;
}

// The road at q (metres from the anchor), if q is within `margin` of one's edge: the one it lies
// deepest inside.
bool roadAt(vec2 q, float margin, out RoadHit h) {
    h.edge = -1e9;
    h.junction = false;
    if (!roadsOn()) return false;
    float fineHalf = 0.5 * float(ROAD_GRID) * ROAD_FINE, coarseHalf = 0.5 * float(ROAD_GRID) * ROAD_COARSE;
    bool fine = max(abs(q.x), abs(q.y)) < fineHalf;
    float cell = fine ? ROAD_FINE : ROAD_COARSE, halfW = fine ? fineHalf : coarseHalf;
    if (max(abs(q.x), abs(q.y)) >= halfW) return false;
    uint base = g.road2.x > 0.5 ? ROAD_HALF : 0u;
    ivec2 c = clamp(ivec2((q + halfW) / cell), ivec2(0), ivec2(ROAD_GRID - 1));
    uint cw = roadBuf.w[base + (fine ? 0u : uint(ROAD_GRID * ROAD_GRID)) + uint(c.y * ROAD_GRID + c.x)];
    uint off = cw >> 8, n = min(cw & 255u, 24u);
    vec2 bestDir = vec2(0.0);
    for (uint k = 0u; k < n; ++k) {
        uint s = base + ROAD_SEG_AT + roadBuf.w[base + ROAD_IDX_AT + off + k] * 6u;
        vec2 a = vec2(uintBitsToFloat(roadBuf.w[s]), uintBitsToFloat(roadBuf.w[s + 1u]));
        vec2 b = vec2(uintBitsToFloat(roadBuf.w[s + 2u]), uintBitsToFloat(roadBuf.w[s + 3u]));
        uint info = roadBuf.w[s + 5u];
        uint cls = info & 15u, lanes = (info >> 4) & 15u, flags = (info >> 8) & 15u;
        vec2 ab = b - a;
        float len = length(ab);
        vec2 d = ab / max(len, 1e-4);
        float t = clamp(dot(q - a, d), 0.0, len);
        vec2 r = q - (a + d * t);
        float hw = roadHalfWidth(cls, lanes, flags);
        float e = hw - length(r);
        if (e > h.edge) {
            // (the road it was on till now crosses this one, if it lay inside it too)
            if (h.edge > 0.3 && abs(d.x * bestDir.y - d.y * bestDir.x) > 0.35) h.junction = true;
            h.edge = e;
            h.across = d.x * r.y - d.y * r.x;
            h.along = uintBitsToFloat(roadBuf.w[s + 4u]) + t;
            h.hw = hw;
            h.cls = cls;
            h.lanes = lanes;
            h.flags = flags;
            h.seed = info >> 16;
            bestDir = d;
        } else if (e > 0.3 && abs(d.x * bestDir.y - d.y * bestDir.x) > 0.35) {
            h.junction = true;
        }
    }
    return h.edge > -margin;
}

float rdHash(float x) { return fract(sin(x * 127.1 + 311.7) * 43758.5453); }
float rdFbm(vec2 p) { return vnoise(p) * 0.6 + vnoise(p * 2.13 + 7.1) * 0.4; }

// The surface of the road `h` at q (metres from the anchor): its colour over `ground` (the land's
// own, which shows on the shoulders and between a track's ruts), how much of the pixel is road
// (0..1: edges, and roads narrower than a pixel), and how much paint (smoother, no texture).
// `asphalt`: the scan's colour there (grey ~0.08 without one); footprint: the pixel's size (m).
vec3 roadSurface(RoadHit h, vec2 q, vec3 ground, vec3 asphalt, float footprint, bool yellowCentre, out float cover, out float paint) {
    float aa = abs(h.across);
    float seed = float(h.seed);
    paint = 0.0;
    // (an edge a pixel wide at least; a road narrower than a pixel covers its share of it)
    cover = clamp(h.edge / max(footprint, 0.02) + 0.5, 0.0, 1.0) * clamp(2.0 * h.hw / max(footprint, 1e-3), 0.0, 1.0);
    if ((h.flags & RF_UNPAVED) != 0u) {
        // an unsealed track: two wheel ruts worn to the bare soil (sand, shell, marl), a grown-over
        // strip between them, ragged edges
        vec3 soil = mix(vec3(0.42, 0.37, 0.28), vec3(0.50, 0.46, 0.38), rdFbm(q * 0.05 + seed));
        soil *= mix(0.85, 1.1, rdFbm(q * 1.7));
        float rut = exp(-pow((aa - 0.75) / 0.42, 2.0));
        float bare = clamp(rut * 1.25 + 0.55 * smoothstep(0.35, 0.7, rdFbm(q * 0.3 + 3.0)), 0.0, 1.0);
        float ragged = smoothstep(-0.15, 0.25, h.edge + 0.5 * (rdFbm(vec2(h.along * 0.4, seed)) - 0.5));
        cover *= ragged;
        return mix(mix(ground, soil, 0.35), soil, bare);
    }
    // ---- asphalt: nearly grey; stretches resurfaced at different times (dark and smooth when
    // new, greyer and bleached warm when old), a long-wave change of tone along each
    // (laid on the ground, not per way: OpenStreetMap cuts a road into a way per block and per
    // change of lanes, and a tone per way was a patchwork of squares. The age drifts over
    // 200-600 m, with now and then the straight joint of a resurfacing across the road.)
    float drift = rdFbm(q * 0.004 + 17.0);
    float seg = floor((h.along + rdHash(seed) * 300.0) / 260.0);
    float age = clamp(drift + (rdHash(seg * 7.13 + seed * 3.1) - 0.5) * 0.22, 0.0, 1.0); // 0 new .. 1 old
    float lum = dot(asphalt, vec3(0.2126, 0.7152, 0.0722));
    vec3 alb = mix(vec3(lum), asphalt, 0.5);
    // (from the road the stretches differ plainly; from the air a road is one dark ribbon, its
    // repairs a faint change of tone)
    float agedTone = mix(0.84, 1.14, age);
    alb *= mix(agedTone, mix(0.94, 1.06, age), smoothstep(0.05, 0.6, footprint)) * mix(0.93, 1.07, rdFbm(q * 0.02 + seed));
    alb = mix(alb, alb * vec3(1.03, 1.0, 0.95), age * 0.6);
    // (service roads and car parks: older, paler, patched)
    if (h.cls == RC_SERVICE) alb *= mix(1.05, 1.3, rdFbm(q * 0.08 + seed));
    // ---- wheel paths: two per lane, polished and darkened by tyres; oil drips between them
    float nl = roadLanes(h.cls, h.lanes, h.flags);
    bool oneway = (h.flags & RF_ONEWAY) != 0u;
    bool marked = h.cls <= RC_TERTIARY || h.cls == RC_LINK;
    // the lanes share the width inside the edge lines (a shoulder beyond them on the big roads)
    float sh = h.cls <= RC_TRUNK ? 1.2 : marked ? 0.25 : 0.1;
    float lw = (2.0 * (h.hw - sh)) / nl;                 // lane width
    float x = h.across + (h.hw - sh);                     // from the right edge line
    float inLane = (fract(clamp(x / lw, 0.0, nl - 1e-3)) - 0.5) * lw;
    float track = exp(-pow((abs(inLane) - 0.85) / 0.33, 2.0)) * step(0.0, x) * step(x, nl * lw);
    alb *= 1.0 - 0.12 * track;
    float drip = exp(-pow(inLane / 0.35, 2.0)) * smoothstep(0.55, 0.85, rdFbm(vec2(h.along * 0.7, h.across * 3.0) + seed));
    alb *= 1.0 - 0.3 * drip * smoothstep(0.5, 0.05, footprint);
    // ---- cracks and patches on the old stretches (near: from far they are the stretch's tone)
    if (footprint < 0.25) {
        float crackN = abs(vnoise(q * 0.9 + seed) - 0.5) + 0.35 * abs(vnoise(q * 3.1 + 5.0) - 0.5);
        float crack = (1.0 - smoothstep(0.004, 0.012 + footprint * 0.15, crackN)) * smoothstep(0.35, 0.8, age) * smoothstep(0.25, 0.1, footprint);
        alb *= 1.0 - 0.55 * crack;
        float patchy = smoothstep(0.72, 0.76, rdFbm(floor(q * 0.35) * 0.71 + seed)) * step(0.6, age);
        alb *= 1.0 - 0.25 * patchy;
    }
    // ---- the edge: dust and grit blown to the side, the seal's ragged edge
    float dust = (1.0 - smoothstep(0.0, 0.6, h.edge)) * mix(0.3, 1.0, rdFbm(q * 1.5)) * smoothstep(0.8, 0.2, footprint);
    alb = mix(alb, alb * vec3(1.15, 1.1, 1.02) + 0.01, dust * 0.45);
    // ---- paint, worn: where tyres cross it, and flaking with age. None across a junction.
    if (marked && !h.junction && footprint < 1.2) {
        // (a line narrower than a pixel is drawn a pixel wide and covers its share of it: from the
        // air the markings are faint, they do not whiten the road)
        float lineW = 0.06, f = footprint * 0.5;
        float thin = clamp(2.0 * lineW / max(footprint, 1e-3), 0.0, 1.0);
        // edge lines
        float p = 1.0 - smoothstep(lineW, lineW + f, abs(aa - (h.hw - sh)));
        vec3 pcol = vec3(0.78, 0.78, 0.74);
        // lane lines: broken, 3 m on / 9 m off; between opposing lanes the centre line instead
        float k = x / lw;
        float nearest = floor(k + 0.5);
        if (nearest > 0.5 && nearest < nl - 0.5) {
            float dl = abs(k - nearest) * lw;
            bool centre = !oneway && abs(nearest - nl * 0.5) < 0.25;
            if (centre) {
                // (two solid lines on the wider roads, one broken one on a two-lane road)
                float c = nl >= 3.5 ? 1.0 - smoothstep(lineW, lineW + f, abs(dl - 0.11)) : (1.0 - smoothstep(lineW, lineW + f, dl)) * step(fract((h.along + seed * 3.0) / 12.0), 0.25);
                if (c > p) { p = c; pcol = yellowCentre ? vec3(0.72, 0.55, 0.10) : vec3(0.78, 0.78, 0.74); }
            } else {
                p = max(p, (1.0 - smoothstep(lineW, lineW + f, dl)) * step(fract((h.along + seed * 3.0) / 12.0), 0.25));
            }
        }
        float flake = smoothstep(0.3, 0.55, rdFbm(q * 6.0 + seed * 3.0) + 0.25 * (1.0 - age));
        paint = p * flake * (1.0 - 0.6 * track) * thin * smoothstep(1.2, 0.6, footprint);
        alb = mix(alb, pcol * mix(0.85, 1.0, rdFbm(q * 20.0)), paint);
    }
    return alb;
}

#endif
