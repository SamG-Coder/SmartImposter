// The lawn (src/lawn.rs): short grass as a tiling parallax layer, marched in the terrain's hit
// shader. Included by tri.rchit after the atlas samplers and the globals.
//
// Tile space: x, z are world metres, y is the height above the ground. A view (frame) from
// direction f stores, for the line along f through every point of the plane of the blade tips,
// the height of the first blade on that line (depth atlas r), its coverage (g), colour, normal
// and sky visibility. The frame coordinates of a point P are therefore
//     uv = (P.xz + f.xz / f.y * (H - P.y)) / T        height = P.y / H
// which is linear in P: a ray is a straight line in every frame, as for the plant impostors.

#define LAWN_FADE 220.0 // lawn.rs FADE

struct Lawn {
    uint slot;      // atlas slot
    float H, T;     // blade height (m), tile edge (m)
    vec2 n;         // views: azimuths, slopes
    vec2 slopes;    // cot(elevation) of the steepest and the flattest views
    float res, gut; // texels per view, of which gutter on each side
    float lod, marchLod;
    float thr;      // this pixel's coverage threshold
    float cover;    // how much of the lawn's coverage is there (its amount here x distance fade)
};

// Where in the tile the frame coordinates uv are. Every copy of the tile is shifted by its own
// random amount (the tile wraps around: a shifted copy is whole), so no clump lines up with its
// copies into rows; the borders between copies are ragged, not straight cuts through the blades.
// (In frame coordinates: a ray along the view stays in one copy.)
vec2 lawnTile(vec2 uv) {
    vec2 c = floor(uv + 0.04 * sin(uv.yx * 39.0) + 0.02 * sin(uv.yx * 101.0 + 1.3));
    uint h = pcg((uint(int(c.x)) * 0x9E3779B9u) ^ pcg(uint(int(c.y)) + 0x51u));
    return fract(uv + vec2(float(h & 0xFFFFu), float(h >> 16)) * (1.0 / 65536.0));
}

vec2 lawnAtlasUV(Lawn m, vec2 cell, vec2 uv) {
    vec2 t = (m.gut + lawnTile(uv) * (m.res - 2.0 * m.gut)) / m.res;
    return (cell + t) / m.n;
}

// Is the point with frame coordinates S at or behind the first blade seen from the frame (and
// within `thick` of it: a blade is a thin sheet, not everything below its tip)?
bool lawnSolid(Lawn m, vec2 cell, vec3 S, float thick) {
    vec2 dz = textureLod(depthAtlas[nonuniformEXT(m.slot)], lawnAtlasUV(m, cell, S.xy), m.marchLod).rg;
    return dz.y * m.cover > 0.02 && S.z <= dz.x && S.z >= dz.x - thick;
}

// One of the four views around direction v (towards the viewer), with probability its sharpened
// bilinear weight: the nearest view nearly always, a mix only half way between two (as
// impostor.rint pickFrame).
vec2 lawnPick(Lawn m, vec3 v, float xi) {
    vec2 gp = lawnFrameCoord(v, m.n, m.slopes);
    vec2 base = floor(gp), fr = gp - base;
    vec4 w = vec4((1.0 - fr.x) * (1.0 - fr.y), fr.x * (1.0 - fr.y), (1.0 - fr.x) * fr.y, fr.x * fr.y);
    // (neighbouring views of a lawn differ far more than a plant's: a mix of two is a blur, so
    // the nearest view is used except right at the middle between two)
    w = w * w;
    w = w * w;
    w = w * w;
    w /= (w.x + w.y + w.z + w.w);
    int k = xi < w.x ? 0 : (xi < w.x + w.y ? 1 : (xi < w.x + w.y + w.z ? 2 : 3));
    vec2 cell = base + vec2(k & 1, k >> 1);
    return vec2(mod(cell.x, m.n.x), min(cell.y, m.n.y - 1.0));
}

// March P0 + D s, s in [0, s1], through the view `cell`: the first place the ray goes from free
// space into a blade, decided by coverage once per crossing (impostor.rint marchFrame).
bool lawnMarch(inout Lawn m, vec2 cell, vec3 P0, vec3 D, float s1, out float sHit, out vec2 uvHit) {
    vec3 f = lawnFrameDir(cell, m.n, m.slopes);
    vec2 sh = f.xz / f.y;
    vec3 A = vec3((P0.xz + sh * (m.H - P0.y)) / m.T, P0.y / m.H);
    vec3 B = vec3((D.xz - sh * D.y) / m.T, D.y / m.H);
    // How far behind the first blade of its line a point still counts as blade. The plants use
    // a thin shell, so rays entering from the side do not hit a wall; a lawn is dense, and a ray
    // off its view's direction that slips behind a blade has all but certainly gone through one:
    // a thin shell leaves holes down to the soil. But the shell is a height, and behind a blade
    // it reaches that height x the view's slope along the ground: in a flat view a thick shell
    // is a slab many blades long, which a ray a few degrees off the view sees from the side as a
    // smear. So it is thick in the steep views of a lawn, and never longer along the ground
    // than a lawn's blade is high (4.5 cm), however flat the view or tall the grass.
    float SHELL = clamp(0.045 / (m.H * max(length(sh), 1e-3)), 0.04, 0.4);
    // Steps: no more than 80% of the shell in height, nor ~1.5 texels sideways, per step. A ray
    // far off its view (a low eye close to the lawn, flatter than the flattest view) would need
    // more than MAX: it is marched through a coarser level instead, blurred rather than aliased.
    const float MAX = 32.0;
    float texels = length(B.xy) * s1 * (m.res - 2.0 * m.gut) * exp2(-m.marchLod);
    if (texels > 1.5 * MAX) {
        float up = ceil(log2(texels / (1.5 * MAX)));
        m.marchLod += up;
        m.lod = max(m.lod, m.marchLod);
        texels *= exp2(-up);
    }
    int steps = clamp(int(ceil(max(abs(B.z) * s1 / (0.8 * SHELL), texels / 1.5))), 6, int(MAX));
    float dt = s1 / float(steps), tPrev = 0.0;
    bool inside = false;
    for (int i = 0; i <= steps; ++i) {
        float t = dt * float(i);
        bool s = lawnSolid(m, cell, A + B * t, SHELL);
        if (inside || i == 0) {
            inside = s;
            tPrev = t;
            continue;
        }
        if (s) {
            float lo = tPrev, hi = t;
            for (int k = 0; k < 5; ++k) {
                float mid = 0.5 * (lo + hi);
                if (lawnSolid(m, cell, A + B * mid, SHELL)) hi = mid; else lo = mid;
            }
            vec3 Sc = A + B * hi;
            float cov = textureLod(depthAtlas[nonuniformEXT(m.slot)], lawnAtlasUV(m, cell, Sc.xy), m.marchLod).g * m.cover;
            if (cov > m.thr) {
                sHit = hi;
                uvHit = Sc.xy;
                return true;
            }
            inside = true;
        }
        tPrev = t;
    }
    // (nothing: where the ray reaches the ground, for the caller)
    sHit = s1;
    uvHit = (A + B * s1).xy;
    return false;
}

// How much sun reaches the point P of the slab (0..1): the view nearest the sun is a shadow
// map, P is lit if nothing on its line towards the sun is above it. A few samples up the ray
// (the view is not exactly along the sun); each takes away its coverage where the first blade
// on its line is clearly above it, less where it is only just above (the stored heights are
// block-compressed and filtered: a hard test speckles every blade with its own shadow).
float lawnSun(Lawn m, vec3 P, vec3 L, inout uint seed) {
    if (L.y <= 0.02) return 1.0;
    vec2 cell = lawnPick(m, L, rnd(seed));
    vec3 f = lawnFrameDir(cell, m.n, m.slopes);
    vec2 sh = f.xz / f.y;
    float sExit = (m.H - P.y) / L.y;
    float jitter = rnd(seed);
    float lit = 1.0;
    const int N = 4;
    for (int k = 0; k < N; ++k) {
        vec3 Q = P + L * (sExit * (float(k) + 0.25 + 0.5 * jitter) / float(N));
        vec3 S = vec3((Q.xz + sh * (m.H - Q.y)) / m.T, Q.y / m.H);
        vec2 dz = textureLod(depthAtlas[nonuniformEXT(m.slot)], lawnAtlasUV(m, cell, S.xy), m.marchLod).rg;
        lit *= 1.0 - dz.y * m.cover * smoothstep(0.04, 0.22, dz.x - S.z);
    }
    return lit;
}
