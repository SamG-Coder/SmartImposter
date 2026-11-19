"""Builds an F-16C in Blender from published dimensions and a digitised three-view, paints it, and
exports it as glTF.

    blender -b --python tools/make_f16.py -- <out_dir> [--renders] [--views a,b] [--shot name=x,y,z,tx,ty,tz,lens]
                                              [--clay] [--no-paint]

Writes f16c.glb (its textures inside), f16c.blend, the three texture maps and f16c.acm, the model
as SmartImposter's own renderer loads it (--test-flight). --renders adds pictures
(Cycles): --views picks which, --clay renders the bare shape, --no-paint reuses the maps already there.
The painting is in f16_paint.py.

What is real data:
  * wing: reference span 30 ft, root chord 16.5 ft, tip chord 3.5 ft (300 ft^2), 40 deg leading
    edge sweep, NACA 64A204 section (derived from the tabulated NACA 64A210: same mean line,
    thickness scaled to 4%), 3 deg washout;
  * overall length 15.03 m with the probe, span over the launchers 9.45 m;
  * the outline: the side profile (top and bottom lines of the body, canopy, fin, ventral fin,
    nozzle), the plan (radome, strakes, shelf, tailplanes, flap hinge lines) and the widths in the
    front view were measured from the three-view drawing "Lockheed F-16A Fighting Falcon 3-view
    line drawing.svg" on Wikimedia Commons, scaled by the published length (87.6 px per metre
    in its 1920 px rendering). Its span over the launchers comes out at 9.48 m and its height on
    the wheels at 5.18 m against the published 9.45 and 5.09, which is the accuracy to expect.
What is not: the cross-sections between those outlines (superellipses blended into the strake and
the shelf), the cockpit, the inside of the intake and of the nozzle, and the paint: the greys, panel
lines and markings follow the type's usual scheme and layout, not one airframe; the tail code and
serial are made up. The wing pylons (two each side, empty) are sized by eye. No landing gear, no pilot.

Aircraft axes while building: x aft from the tip of the radome, y to the right, z up from the
height of that tip (metres). Written out with the origin at the centre of gravity, nose towards -Y
in Blender (+Z in glTF), Z up. The moving surfaces are separate objects with their origin on the
hinge line.
"""
import math
import os
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f16_paint as paint  # noqa: E402

FT = 0.3048

# ------------------------------------------------------------------ published data
SPAN = 30.0 * FT          # reference wing span (without tip launchers)
HALF = SPAN / 2
C_ROOT = 16.5 * FT        # at the centre line
C_TIP = 3.5 * FT
SWEEP_LE = math.radians(40.0)
WASHOUT = 3.0             # degrees, at the tip
LENGTH = 15.03            # probe tip to the fin tip's trailing edge
TAIL_ANHEDRAL = math.radians(10.0)

# ------------------------------------------------------------------ from the three-view
PROBE = 0.30              # the probe ahead of the radome
X_WING = 5.62             # wing leading edge at the centre line
Z_WING = 0.24             # the wing plane, and the edge of the strakes
X_NOZ0, X_NOZ1 = 13.48, 14.25   # the nozzle's petals
Z_ENG = 0.285             # the engine's axis
Y_SHELF = 1.044           # the body's edge behind the wing: flaperon and tailplane roots
Y_LEF0 = 1.35             # where the strake meets the wing's leading edge
Y_FL1 = 3.61              # outboard end of the flaperon
MAC = 11.32 * FT
_y_mac = SPAN / 6.0 * (1 + 2 * C_TIP / C_ROOT) / (1 + C_TIP / C_ROOT)
X_CG = X_WING + _y_mac * math.tan(SWEEP_LE) + 0.35 * MAC   # 35% of the mean chord
Z_CG = 0.20

# NACA 64A210 (UIUC / airfoiltools "naca64a210-il"), upper then lower surface
N64A210 = """1.000000 0.000210 0.950270 0.007850 0.900520 0.015510 0.850740 0.023010 0.800760 0.030370
0.750630 0.037020 0.700540 0.043100 0.650420 0.048520 0.600280 0.053230 0.550120 0.057140
0.499940 0.060140 0.449750 0.062080 0.399550 0.062740 0.349350 0.061920 0.299170 0.059840
0.249000 0.056560 0.198850 0.052000 0.148740 0.045920 0.098680 0.037920 0.073690 0.032880
0.048740 0.026850 0.023870 0.018950 0.011530 0.013420 0.006650 0.010440 0.004240 0.008560
0.000000 0.000000 0.005760 -0.007440 0.008350 -0.008860 0.013470 -0.011000 0.026130 -0.014730
0.051260 -0.019630 0.076310 -0.023160 0.101320 -0.026000 0.151260 -0.030300 0.201150 -0.033400
0.251000 -0.035540 0.300830 -0.036880 0.350650 -0.037440 0.400450 -0.037160 0.450250 -0.035800
0.500060 -0.033540 0.549880 -0.030620 0.599720 -0.027190 0.649580 -0.023420 0.699460 -0.019440
0.749370 -0.015420 0.799240 -0.011670 0.849260 -0.008590 0.899480 -0.005710 0.949740 -0.002950
1.000000 -0.000210"""
_a = np.array(N64A210.split(), dtype=float).reshape(-1, 2)
_le = int(np.argmin(_a[:, 0]))
_up, _lo = _a[: _le + 1][::-1], _a[_le:]


def sec_64a204(xc):
    """Upper and lower ordinates (per unit chord) of the NACA 64A204 at chord fractions xc."""
    yu, yl = np.interp(xc, _up[:, 0], _up[:, 1]), np.interp(xc, _lo[:, 0], _lo[:, 1])
    camber, half = 0.5 * (yu + yl), 0.5 * (yu - yl) * 0.4
    return camber + np.maximum(half, 1e-4), camber - np.maximum(half, 1e-4)


def biconvex(t):
    """A biconvex section of thickness t (tail surfaces), with a slightly rounded nose."""
    def f(xc):
        h = np.maximum(2.0 * t * xc * (1.0 - xc) + 0.12 * t * np.sqrt(np.clip(xc, 0, 1)) * (1.0 - xc), 1e-4)
        return h, -h
    return f


def chord(y):
    return C_ROOT + (C_TIP - C_ROOT) * abs(y) / HALF


def x_le(y):
    return X_WING + abs(y) * math.tan(SWEEP_LE)


def x_te(y):
    return x_le(y) + chord(y)


def le_hinge(y):
    """x of the leading-edge flap's hinge line."""
    return x_le(y) + 0.62 + (0.33 - 0.62) * (abs(y) - Y_LEF0) / (HALF - Y_LEF0)


def fl_hinge(y):
    """x of the flaperon's hinge line (continued to the tip as the wing box's rear edge)."""
    return x_te(y) - (0.89 + (0.47 - 0.89) * (abs(y) - Y_SHELF) / (Y_FL1 - Y_SHELF))


# ------------------------------------------------------------------ the body's lines
class Curve:
    """A table of (x, value): linear between the points, then smoothed (sigma in metres)."""

    def __init__(self, pts, sigma=0.12, keep=None):
        t = np.array(pts, float)
        self.x = np.arange(t[0, 0], t[-1, 0] + 0.005, 0.01)
        v = np.interp(self.x, t[:, 0], t[:, 1])
        if sigma > 0:
            n = int(3 * sigma / 0.01)
            k = np.exp(-0.5 * (np.arange(-n, n + 1) * 0.01 / sigma) ** 2)
            vs = np.convolve(np.pad(v, n, mode='edge'), k / k.sum(), mode='valid')
            if keep is not None:
                w = np.clip((self.x - keep) / 0.5, 0.0, 1.0)
                vs = v * (1 - w) + vs * w
            v = vs
        self.v = v

    def __call__(self, x):
        return np.interp(x, self.x, self.v)


# half-width in plan: radome, forebody, strake, then the shelf beside the engine
Y_EDGE = Curve([(0.00, 0.006), (0.14, 0.023), (0.25, 0.046), (0.365, 0.091), (0.48, 0.137), (0.59, 0.194), (0.71, 0.240),
                (0.82, 0.274), (0.94, 0.320), (1.05, 0.354), (1.16, 0.377), (1.28, 0.411), (1.39, 0.434), (1.51, 0.451),
                (1.62, 0.462), (1.73, 0.479), (1.85, 0.491), (1.96, 0.502), (2.08, 0.514), (2.19, 0.525), (2.31, 0.536),
                (2.42, 0.548), (2.53, 0.565), (2.65, 0.616), (2.76, 0.633), (2.88, 0.650), (2.99, 0.673), (3.10, 0.690),
                (3.33, 0.713), (3.56, 0.736), (3.79, 0.753), (4.02, 0.770), (4.245, 0.800), (4.47, 0.833), (4.70, 0.873),
                (4.93, 0.924), (5.16, 0.981), (5.39, 1.044), (5.61, 1.113), (5.84, 1.187), (6.07, 1.244), (6.30, 1.290),
                (6.53, 1.335), (6.76, 1.352), (7.3, 1.20), (8.0, 1.10), (9.0, 1.05), (10.56, Y_SHELF), (13.48, Y_SHELF)],
               0.05, keep=0.5)
# the top line (under the canopy: the deck the canopy sits on) and the bottom line
Z_TOP = Curve([(0.0, 0.004), (0.05, 0.04), (0.16, 0.055), (0.27, 0.09), (0.39, 0.148), (0.50, 0.194), (0.62, 0.24), (0.73, 0.285),
               (0.84, 0.331), (0.96, 0.377), (1.07, 0.411), (1.19, 0.445), (1.30, 0.485), (1.415, 0.525), (1.53, 0.559),
               (1.64, 0.593), (1.76, 0.628), (1.87, 0.650), (1.985, 0.673), (2.10, 0.70), (2.21, 0.725), (2.33, 0.742),
               (2.44, 0.765), (2.8, 0.80), (3.5, 0.87), (4.2, 0.95), (4.8, 1.03), (5.4, 1.12), (5.9, 1.17), (6.1, 1.155),
               (6.5, 1.125), (7.0, 1.06), (7.5, 1.03), (12.26, 1.03), (12.75, 0.95), (13.48, 0.85)], 0.08, keep=0.4)
Z_BOT = Curve([(0.0, -0.004), (0.16, -0.011), (0.27, -0.017), (0.39, -0.046), (0.50, -0.063), (0.62, -0.08), (0.73, -0.091),
               (0.96, -0.114), (1.19, -0.131), (1.4, -0.137), (1.8, -0.13), (2.15, -0.114), (2.44, -0.10), (2.9, -0.063),
               (3.13, -0.046), (3.47, -0.03), (3.93, 0.0), (4.27, 0.017), (4.4, 0.02), (5.0, 0.07), (5.64, 0.137), (5.87, 0.19),
               (6.0, 0.21), (6.15, 0.21), (6.3, -0.2), (6.45, -0.55), (6.6, -0.64), (7.0, -0.62), (7.7, -0.605),
               (8.7, -0.58), (9.9, -0.555), (10.2, -0.55), (11.57, -0.485), (12.75, -0.38), (13.48, -0.27)], 0.08, keep=0.4)
# the height of the greatest width (the strake and the shelf are at Z_EDGE)
Z_MID = Curve([(0.0, 0.0), (0.5, 0.065), (1.0, 0.135), (1.5, 0.20), (2.0, 0.25), (2.44, 0.255), (2.7, Z_WING), (9.0, Z_WING),
               (11.0, 0.27), (12.75, Z_ENG), (13.48, 0.29)], 0.15, keep=0.5)
Z_EDGE = Curve([(0.0, 0.0), (0.5, 0.065), (1.0, 0.135), (1.5, 0.20), (2.0, 0.25), (2.44, 0.255), (2.7, Z_WING), (10.5, Z_WING),
                (11.5, 0.225), (13.48, 0.22)], 0.15, keep=0.5)
W_CORE = Curve([(0.0, 0.56), (2.4, 0.56), (4.0, 0.58), (5.0, 0.62), (6.0, 0.72), (7.0, 0.76), (9.0, 0.76), (10.5, 0.73),
                (11.5, 0.69), (12.75, 0.65), (13.48, 0.56)], 0.2)
W_BOX = Curve([(0.0, 0.56), (6.15, 0.56), (6.6, 0.70), (9.5, 0.70), (10.5, 0.69), (11.5, 0.67),
               (12.75, 0.65), (13.48, 0.56)], 0.1)
Z_BOX = Curve([(0.0, 0.0), (0.5, 0.065), (1.0, 0.135), (1.5, 0.20), (2.0, 0.25), (2.44, 0.255), (2.7, Z_WING), (6.15, Z_WING),
               (6.6, 0.08), (10.5, 0.08), (11.5, 0.16), (12.75, Z_ENG), (13.48, 0.29)], 0.15, keep=0.5)
N_TOP = Curve([(0.0, 2.0), (1.9, 2.0), (2.5, 3.0), (5.0, 3.2), (6.5, 1.75), (10.5, 1.75), (12.75, 2.0), (13.48, 2.0)], 0.25)
N_BOT = Curve([(0.0, 2.0), (2.5, 2.0), (6.15, 2.2), (6.6, 4.4), (9.5, 4.4), (10.5, 3.5), (11.5, 2.6),
               (12.75, 2.0), (13.48, 2.0)], 0.2)
H_EDGE = Curve([(0.0, 0.0), (2.5, 0.0), (2.7, 0.004), (6.6, 0.004), (7.2, 0.05), (10.4, 0.05), (10.7, 0.07), (13.0, 0.06),
                (13.48, 0.05)], 0.05)
HR_TOP = Curve([(0.0, 0.0), (2.4, 0.0), (2.7, 0.18), (5.0, 0.33), (6.8, 0.50), (10.5, 0.45), (11.5, 0.25), (13.48, 0.18)], 0.15)
HR_BOT = Curve([(0.0, 0.0), (2.4, 0.0), (2.7, 0.12), (5.0, 0.20), (6.8, 0.28), (10.5, 0.28), (11.5, 0.22), (13.48, 0.16)], 0.15)

# the canopy: half-width in plan and its top line
CAN_W = Curve([(2.44, 0.0), (2.50, 0.12), (2.56, 0.24), (2.74, 0.36), (2.94, 0.405), (3.24, 0.434), (3.67, 0.44), (4.13, 0.43),
               (4.59, 0.37), (4.93, 0.30), (5.27, 0.22), (5.61, 0.13), (5.90, 0.04), (5.98, 0.0)], 0.04)
CAN_T = Curve([(2.44, 0.765), (2.556, 0.827), (2.67, 0.919), (2.78, 1.004), (2.90, 1.084), (3.01, 1.153), (3.13, 1.21),
               (3.24, 1.267), (3.355, 1.312), (3.47, 1.352), (3.58, 1.381), (3.70, 1.415), (3.81, 1.438), (3.93, 1.449),
               (4.04, 1.461), (4.27, 1.461), (4.38, 1.455), (4.50, 1.449), (4.61, 1.438), (4.72, 1.426), (4.84, 1.409),
               (4.95, 1.381), (5.07, 1.369), (5.18, 1.347), (5.295, 1.324), (5.41, 1.301), (5.52, 1.278), (5.64, 1.255),
               (5.75, 1.22), (5.865, 1.198), (5.98, 1.17)], 0.04)

# the fin (side view): leading edge, trailing edge, rudder hinge
FIN_Z0, FIN_ZR0, FIN_ZR1, FIN_Z1 = 1.45, 1.50, 3.38, 3.58


def fin_le(z):
    return 11.11 + (z - FIN_Z0) * (13.53 - 11.11) / (FIN_Z1 - FIN_Z0)


def fin_te(z):
    return 13.66 + (z - FIN_Z0) * (14.60 - 13.66) / (3.41 - FIN_Z0)


def fin_hinge(z):
    return 12.83 + (z - FIN_Z0) * (14.17 - 12.83) / (3.41 - FIN_Z0)


# the tailplane (plan view)
TAIL_Y1, TAIL_Z = 2.807, 0.22


def tail_le(y):
    return 11.81 + (abs(y) - Y_SHELF) * (13.40 - 11.81) / (TAIL_Y1 - Y_SHELF)


def tail_te(y):
    return 14.23 - max(abs(y) - 2.53, 0.0) * 0.23 / (TAIL_Y1 - 2.53)


# ------------------------------------------------------------------ mesh helpers
def loft(rings, caps=True):
    n = len(rings[0])
    verts = [tuple(p) for r in rings for p in r]
    faces = []
    for i in range(len(rings) - 1):
        for j in range(n):
            faces.append((i * n + j, i * n + (j + 1) % n, (i + 1) * n + (j + 1) % n, (i + 1) * n + j))
    if caps:
        faces.append(tuple(range(n - 1, -1, -1)))
        faces.append(tuple(range((len(rings) - 1) * n, len(rings) * n)))
    return verts, faces


def strip(rows):
    """An open sheet through rows of points (no wrap round)."""
    n = len(rows[0])
    verts = [tuple(p) for r in rows for p in r]
    faces = [(i * n + j, i * n + j + 1, (i + 1) * n + j + 1, (i + 1) * n + j) for i in range(len(rows) - 1) for j in range(n - 1)]
    return verts, faces


def merge(parts):
    verts, faces = [], []
    for v, f in parts:
        k = len(verts)
        verts += v
        faces += [tuple(i + k for i in face) for face in f]
    return verts, faces


def mirror(part):
    v, f = part
    return [(x, -y, z) for x, y, z in v], [tuple(reversed(face)) for face in f]


def both(part):
    return merge([part, mirror(part)])


def section_ring(p0, c, fa, fb, sec, tdir, twist=0.0, m=16):
    """A closed loop round a lifting surface's section from chord fraction fa to fb: p0 the
    leading edge, the chord along +x, tdir the direction of its thickness."""
    s = 0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, m))
    xc = fa + (fb - fa) * s
    up, lo = sec(xc)
    u = np.concatenate([xc, xc[::-1]]) * c
    v = np.concatenate([up, lo[::-1]]) * c
    a = math.radians(twist)
    q = 0.25 * c
    u2 = q + (u - q) * math.cos(a) + v * math.sin(a)
    v2 = -(u - q) * math.sin(a) + v * math.cos(a)
    p0, tdir = np.array(p0, float), np.array(tdir, float)
    return p0[None, :] + u2[:, None] * np.array([1.0, 0.0, 0.0])[None, :] + v2[:, None] * tdir[None, :]


def lathe(profile, seg=32, x_axis_z=0.0, y0=0.0):
    """Revolve (x, r) pairs about the x axis through (y0, x_axis_z)."""
    rings = []
    for x, r in profile:
        a = np.linspace(0, 2 * math.pi, seg, endpoint=False)
        rings.append(np.stack([np.full(seg, x), y0 + r * np.cos(a), x_axis_z + r * np.sin(a)], axis=1))
    return loft(rings)


def box(x0, x1, hw, z0, z1, y0=0.0):
    return loft([np.array([(x, y0 - hw, z1), (x, y0 + hw, z1), (x, y0 + hw, z0), (x, y0 - hw, z0)]) for x in (x0, x1)])


def sup_ring(x, hw, zt, zc, zb, nt, nb, seg=44):
    """A section of half-width hw, widest at zc, a superellipse of exponent nt above it and nb below."""
    a = np.linspace(0, 2 * math.pi, seg, endpoint=False)
    ca, sa = np.cos(a), np.sin(a)
    n = np.where(sa >= 0, nt, nb)
    y = hw * np.sign(ca) * np.abs(ca) ** (2.0 / n)
    z = zc + np.where(sa >= 0, zt - zc, zc - zb) * np.sign(sa) * np.abs(sa) ** (2.0 / n)
    return np.stack([np.full(seg, x), y, z], axis=1)


# ------------------------------------------------------------------ the body
def smax(a, b, k):
    if k <= 1e-6:
        return np.maximum(a, b)
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.maximum(a, b) + h * h * k * 0.25


_sections = {}
SILL = 0.075            # the canopy's sill: from the glass in to the cockpit's rim
TUB_DEPTH = 0.42


def canopy_w(x):
    return float(CAN_W(x)) if 2.44 <= x <= 5.98 else 0.0


def tub_ramp(x):
    """1 over the cockpit, 0 outside it: from under the glare shield to behind the seat."""
    f = lambda t: min(max(t, 0.0), 1.0) ** 2 * (3.0 - 2.0 * min(max(t, 0.0), 1.0))
    return f((x - 3.30) / 0.12) * f((4.92 - x) / 0.10)


def in_tub(c, n):
    """Material index for a face of the body: 1 inside the cockpit."""
    return 1 if (tub_ramp(c[0]) > 0.02 and abs(c[1]) < canopy_w(c[0]) - SILL + 0.002) else 0


def body_section(x, n=500):
    """The body's section at station x as two height fields over y (0 .. edge): its top and its
    bottom. A core (the fuselage proper: a superellipse above and one below the height of its
    greatest width) joined smoothly with a shelf (the strake ahead of the wing, the shelf that
    carries the tailplanes behind it)."""
    key = round(float(x), 4)
    if key in _sections:
        return _sections[key]
    ye = float(Y_EDGE(x))
    wc, wb = min(float(W_CORE(x)), ye), min(float(W_BOX(x)), ye)
    zt, zb, zm, ze, zmb = float(Z_TOP(x)), float(Z_BOT(x)), float(Z_MID(x)), float(Z_EDGE(x)), float(Z_BOX(x))
    nt, nb, he = float(N_TOP(x)), float(N_BOT(x)), float(H_EDGE(x))
    # samples across the half width, crowded where the core's walls stand upright (a superellipse
    # climbs most of its height in its last millimetres)
    y = ye * (0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, n)))
    near = np.concatenate([w * (1.0 - np.logspace(-0.5, -7.0, 90)) for w in (wc, wb)])
    y = np.unique(np.concatenate([y, near[(near > 0.0) & (near < ye)]]))
    s = y / ye
    core = np.where(y < wc, zm + (zt - zm) * (1.0 - np.clip(y / wc, 0, 1) ** nt) ** (1.0 / nt), zm - 3.0)
    shelf = ze + he + float(HR_TOP(x)) * (1.0 - s) ** 1.3
    zu = smax(core, shelf, 0.14 * min(max((ye - wc) / 0.15, 0.0), 1.0))
    # where the core's wall comes down onto the shelf: found exactly, not on the sample grid, so
    # that the row of vertices put there runs smoothly along the body
    yu = wc
    for _ in range(4):
        sh = ze + he + float(HR_TOP(x)) * (1.0 - yu / ye) ** 1.3
        r = (sh - zm) / max(zt - zm, 1e-6)
        yu = wc * (1.0 - min(max(r, 0.0), 0.999) ** nt) ** (1.0 / nt)
    yu = min(yu, ye * (1.0 - 1e-6))
    # the cockpit: a tub let into the deck under the canopy, and the line its rim follows
    rim = min(max(canopy_w(x) - SILL, 0.3 * wc), 0.8 * yu)
    deck = zu
    depth = TUB_DEPTH * tub_ramp(x)
    if depth > 1e-4:
        t = np.clip((y - (rim - 0.03)) / 0.03, 0.0, 1.0)
        t = t * t * (3.0 - 2.0 * t)
        zu = np.where(y < rim, (zu[0] - depth) * (1.0 - t) + zu * t, zu)
    core = np.where(y < wb, zmb - (zmb - zb) * (1.0 - np.clip(y / wb, 0, 1) ** nb) ** (1.0 / nb), zmb + 3.0)
    shelf = ze - he - float(HR_BOT(x)) * (1.0 - s) ** 1.3
    zl = -smax(-core, -shelf, 0.10 * min(max((ye - wb) / 0.15, 0.0), 1.0))
    yl = wb
    for _ in range(4):
        sh = ze - he - float(HR_BOT(x)) * (1.0 - yl / ye) ** 1.3
        r = (zmb - sh) / max(zmb - zb, 1e-6)
        yl = wb * (1.0 - min(max(r, 0.0), 0.999) ** nb) ** (1.0 / nb)
    yl = min(max(yl, 1e-4), ye * (1.0 - 1e-6))
    _sections[key] = (y, zu, np.minimum(zl, zu), yu, yl, rim, deck)
    return _sections[key]


def body_top(x, y=0.0):
    """The height of the body's upper surface (the deck, where the cockpit is cut into it)."""
    sec = body_section(min(max(x, 0.0), X_NOZ0))
    return float(np.interp(abs(y), sec[0], sec[6]))


def body_bot(x, y=0.0):
    ys, _, zl = body_section(min(max(x, 0.0), X_NOZ0))[:3]
    return float(np.interp(abs(y), ys, zl))


def resample(y, z, m, crowd=0):
    """m points along a curve, evenly by length; crowd +1 gathers them towards its end, -1 towards
    its start (where a fillet is)."""
    d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(y), np.diff(z)))])
    t = np.linspace(0.0, 1.0, m)
    t = 1.0 - (1.0 - t) ** 1.7 if crowd > 0 else t ** 1.7 if crowd < 0 else t
    return np.interp(t * d[-1], d, y), np.interp(t * d[-1], d, z)


def fuselage(nr=10, nc=17, ns=13, ms=11, mc=21):
    """The body as one skin. Each ring has a vertex on the cockpit's rim and on the line where the
    core meets the shelf, above and below, so those lines run cleanly along the body."""
    xs = np.concatenate([np.linspace(0.0, 0.3, 9)[:-1], np.linspace(0.3, 2.0, 30)[:-1], np.arange(2.0, X_NOZ0, 0.06), [X_NOZ0]])
    rings = []
    for x in xs:
        y, zu, zl, yu, yl, rim = body_section(x)[:6]

        def cut(z, a, b, back=False):
            """The piece of a curve between y = a and y = b, its ends exact."""
            m = (y > a + 1e-9) & (y < b - 1e-9)
            py = np.concatenate([[a], y[m], [b]])
            pz = np.concatenate([[np.interp(a, y, z)], z[m], [np.interp(b, y, z)]])
            return (py[::-1], pz[::-1]) if back else (py, pz)
        ye = y[-1]
        parts = [resample(*cut(zu, 0.0, rim), nr), resample(*cut(zu, rim, yu), nc, 1), resample(*cut(zu, yu, ye), ns, -1),          # centre -> edge
                 resample(*cut(zl, yl, ye, True), ms, 1), resample(*cut(zl, 0.0, yl, True), mc, -1)]                                # edge -> centre
        half = []
        for k, (py, pz) in enumerate(parts):
            half += list(zip(py, pz))[(1 if k in (1, 2, 4) else 0):]
        rings.append(np.array([(x, a, b) for a, b in half] + [(x, -a, b) for a, b in half[-2:0:-1]]))
    verts, faces = loft(rings, caps=False)      # open at the back: the nozzle goes in there
    faces.append(tuple(range(len(rings[0]) - 1, -1, -1)))
    # Which drawing each face is painted from. Choosing by each face's own normal makes the seam
    # zig-zag wherever the surface turns slowly, so the seams are put on rows of vertices: above,
    # a row on the shoulder of the forebody and of the engine bay, and the crease itself where the
    # body is tent shaped between them; below, a row at the turn of the bilge. Where a seam is not
    # the crease the painter draws a panel line along it.
    R = np.array(rings)
    half = (R.shape[1] + 2) // 2 - 1                      # index of the bottom centre
    crease, edge, low = nr + nc - 2, nr + nc + ns - 3, nr + nc + ns + ms - 3
    top_row = lambda x: (nr - 1) + (4 if x < 5.73 else nc - 1 if x < 10.63 else 5)
    bot_row = low + 12
    regions = []
    for i in range(len(R) - 1):
        xm = 0.5 * (xs[i] + xs[i + 1])
        for j in range(R.shape[1]):
            right = j < half
            jr = j if right else 2 * half - 1 - j
            side = 3 if right else 2
            if jr < top_row(xm) or crease <= jr < edge:
                regions.append(0)
            elif jr < crease or jr == edge or low <= jr < bot_row:
                regions.append(side)
            else:
                regions.append(1)
    regions.append(0)
    seams = dict(top=[tuple(R[i, top_row(xs[i])]) for i in range(len(R))], bottom=[tuple(R[i, bot_row]) for i in range(len(R))])
    return (verts, faces), regions, seams



def speedbrakes():
    """The end of the shelf beside the nozzle: the split speed brakes, closed."""
    rows = []
    for x, t, yo in ((13.40, 0.13, 1.045), (13.50, 0.10, 1.04), (13.65, 0.06, 1.035), (13.90, 0.012, 1.02)):
        rows.append(np.array([(x, 0.50, 0.22 + t), (x, yo, 0.22 + 0.6 * t), (x, yo, 0.22 - 0.6 * t), (x, 0.50, 0.22 - t)]))
    return both(loft(rows))


def wing_parts():
    """(fixed wing box, leading-edge flap, flaperon), right side."""
    def ring(y, fa, fb):
        c = chord(y)
        a = fa if isinstance(fa, float) else (fa(y) - x_le(y)) / c
        b = fb if isinstance(fb, float) else (fb(y) - x_le(y)) / c
        return section_ring((x_le(y), y, Z_WING), c, a, b, sec_64a204, (0, 0, 1), -WASHOUT * y / HALF, m=18)

    def seg(y0, y1, fa, fb, n=6):
        return loft([ring(y, fa, fb) for y in np.linspace(y0, y1, n)])

    wing_box = merge([seg(0.85, Y_SHELF, 0.0, 1.0, 2), seg(Y_SHELF, Y_LEF0, 0.0, fl_hinge, 3), seg(Y_LEF0, Y_FL1, le_hinge, fl_hinge, 9),
                      seg(Y_FL1, HALF, le_hinge, 1.0, 5)])
    le_flap = seg(Y_LEF0, HALF, 0.0, le_hinge, 12)
    flaperon = seg(Y_SHELF, Y_FL1, fl_hinge, 1.0, 9)
    return wing_box, (le_flap, (le_hinge(Y_LEF0), Y_LEF0, Z_WING)), (flaperon, (fl_hinge(Y_SHELF), Y_SHELF, Z_WING))


def launcher_and_missile():
    """The wing tip launcher and an AIM-120 on it (3.66 m long, 0.178 m across), right side."""
    x0, x1, ya, yb = 8.11, 10.77, HALF - 0.012, 4.74
    yc = 0.5 * (ya + yb)
    rail = loft([np.array([(x, yc - hw, Z_WING + t), (x, yc + hw, Z_WING + t), (x, yc + hw, Z_WING - t), (x, yc - hw, Z_WING - t)])
                 for x, hw, t in ((x0, 0.012, 0.012), (x0 + 0.22, 0.5 * (yb - ya), 0.058), (x1 - 0.5, 0.5 * (yb - ya), 0.058), (x1, 0.03, 0.02))])
    ym, zm, r, xm = yb + 0.089 + 0.012, Z_WING - 0.005, 0.089, 7.60
    body = lathe([(xm, 0.004), (xm + 0.10, 0.036), (xm + 0.28, 0.068), (xm + 0.55, r), (xm + 0.82, r), (xm + 0.90, r), (xm + 1.35, r), (xm + 1.43, r),
                  (xm + 3.60, r), (xm + 3.66, 0.07)], 18, zm, ym)
    fins = []
    for xf, root, span_f, tip in ((xm + 1.55, 0.62, 0.175, 0.05), (xm + 3.24, 0.36, 0.225, 0.20)):
        for k in range(4):
            a = math.radians(45 + 90 * k)
            d = np.array([0.0, math.cos(a), math.sin(a)])
            n = np.array([0.0, -math.sin(a), math.cos(a)]) * 0.006
            p = np.array([0.0, ym, zm])
            q = [p + d * (r - 0.01) + [xf, 0, 0], p + d * (r + span_f) + [xf + root - tip, 0, 0], p + d * (r + span_f) + [xf + root, 0, 0],
                 p + d * (r - 0.01) + [xf + root, 0, 0]]
            fins.append(loft([np.array([v + n for v in q]), np.array([v - n for v in q])]))
    return rail, body, merge(fins)


def tailplane():
    """One all-moving tailplane (right) and its pivot."""
    rings = []
    ys = [1.0, Y_SHELF, 1.4, 1.8, 2.2, 2.53, 2.67, TAIL_Y1]
    tdir = (0.0, math.sin(TAIL_ANHEDRAL), math.cos(TAIL_ANHEDRAL))
    for y in ys:
        s = (y - 1.0) / (TAIL_Y1 - 1.0)
        z = TAIL_Z - (y - Y_SHELF) * math.tan(TAIL_ANHEDRAL)
        rings.append(section_ring((tail_le(y), y, z), tail_te(y) - tail_le(y), 0.0, 1.0, biconvex(0.055 - 0.02 * s), tdir))
    return loft(rings), (13.25, Y_SHELF, TAIL_Z)


def fin_parts():
    """(fixed fin with the fairing at its foot, rudder and its hinge)."""
    def ring(z, fa, fb):
        s = (z - FIN_Z0) / (FIN_Z1 - FIN_Z0)
        c = fin_te(z) - fin_le(z)
        a = fa if isinstance(fa, float) else (fa(z) - fin_le(z)) / c
        b = fb if isinstance(fb, float) else (fb(z) - fin_le(z)) / c
        return section_ring((fin_le(z), 0.0, z), c, a, b, biconvex(0.05 - 0.018 * s), (0, 1, 0), m=14)

    def seg(za, zb, fa, fb, n):
        return loft([ring(z, fa, fb) for z in np.linspace(za, zb, n)])

    fin = merge([seg(1.38, FIN_ZR0, 0.0, 1.0, 2), seg(FIN_ZR0, FIN_ZR1, 0.0, fin_hinge, 9), seg(FIN_ZR1, FIN_Z1, 0.0, 1.0, 3)])
    rudder = seg(FIN_ZR0, FIN_ZR1, fin_hinge, 1.0, 9)
    # the dorsal fillet and the fairing the fin stands on
    top = Curve([(9.10, 1.0), (9.18, 1.03), (11.11, 1.45), (13.58, 1.45), (13.70, 1.36)], 0.03)
    hw = Curve([(9.10, 0.012), (10.6, 0.05), (11.3, 0.12), (13.3, 0.13), (13.70, 0.10)], 0.08)
    rings = []
    for x in np.linspace(9.10, 13.70, 60):
        a = np.linspace(0.0, math.pi, 13)
        zb = body_top(x) - 0.06
        rings.append(np.stack([np.full(13, x), hw(x) * np.sign(np.cos(a)) * np.abs(np.cos(a)) ** (2 / 3.5),
                               zb + (top(x) - zb) * np.sin(a) ** (2 / 3.5)], axis=1))
    return merge([fin, loft(rings)]), (rudder, (fin_hinge(FIN_ZR0), 0.0, FIN_ZR0))


def ventral_fin():
    cant = math.radians(15.0)
    span_dir = np.array([0.0, math.sin(cant), -math.cos(cant)])
    tdir = (0.0, math.cos(cant), math.sin(cant))
    root = np.array([10.18, 0.47, -0.36])
    rings = []
    for s in np.linspace(0.0, 1.0, 4):
        # leading edge swept 30 degrees; the tip is cut on a slant, deeper at the front
        le = 0.36 * s
        c = 1.39 - le
        r = section_ring(root + span_dir * (0.70 * s) + np.array([le, 0, 0]), c, 0.0, 1.0, biconvex(0.035), tdir, m=8)
        k = (r[:, 0] - (root[0] + le)) / c
        rings.append(r - span_dir[None, :] * (0.20 * s * k)[:, None])
    return loft(rings)


def intake():
    """The chin intake: its fairing, which carries on aft as the lower body until the duct ends and
    the body's own skin takes over; the duct, the splitter plate, the diverter and the strut."""
    #         x      hw      zt      zc      zb     nt   nb
    shell = [(4.285, 0.512, -0.088, -0.20, -0.598, 5.0, 2.4), (4.31, 0.535, -0.068, -0.20, -0.617, 5.0, 2.4),
             (4.40, 0.555, -0.060, -0.185, -0.630, 5.0, 2.5), (4.70, 0.60, 0.02, -0.12, -0.655, 4.5, 3.0),
             (5.10, 0.655, 0.14, -0.02, -0.667, 4.0, 3.7), (5.50, 0.697, 0.26, 0.10, -0.667, 4.0, 4.4),
             (6.10, 0.699, 0.26, 0.09, -0.660, 4.0, 4.4), (6.60, 0.697, 0.20, 0.08, -0.637, 4.0, 4.4)]
    duct = [(4.285, 0.512, -0.088, -0.20, -0.598, 5.0, 2.4), (4.31, 0.488, -0.108, -0.21, -0.580, 5.0, 2.4),
            (4.60, 0.47, -0.10, -0.20, -0.59, 4.5, 2.4), (5.40, 0.42, -0.02, -0.15, -0.55, 3.5, 2.4),
            (6.10, 0.38, 0.10, -0.10, -0.42, 2.5, 2.2)]
    outer = loft([sup_ring(*s) for s in shell], caps=False)
    inner = loft([sup_ring(*s) for s in duct], caps=False)
    face = loft([sup_ring(6.08, 0.38, 0.10, -0.10, -0.42, 2.5, 2.2), sup_ring(6.10, 0.38, 0.10, -0.10, -0.42, 2.5, 2.2)])
    # the plate between the intake and the fuselage (boundary layer splitter) and the diverter behind it
    plate = loft([np.array([(x, -hw, -0.058), (x, hw, -0.058), (x, hw, -0.074), (x, -hw, -0.074)])
                  for x, hw in ((4.02, 0.26), (4.07, 0.40), (4.16, 0.47), (4.31, 0.515), (4.5, 0.53))])
    diverter = loft([np.array([(x, -hw, 0.06), (x, hw, 0.06), (x, hw, -0.07), (x, -hw, -0.07)]) for x, hw in ((4.14, 0.004), (4.45, 0.20), (4.75, 0.30), (5.2, 0.34))])
    strut = loft([np.array([(x, -hw, -0.04), (x, hw, -0.04), (x, hw, -0.60), (x, -hw, -0.60)]) for x, hw in ((4.72, 0.002), (4.86, 0.014), (5.5, 0.014))])
    return merge([outer, plate, diverter]), merge([inner, strut]), face


def canopy(m=25):
    """The bubble and the fixed part behind it, as one sheet, and which of its faces are frame."""
    rows = []
    xs = np.linspace(2.47, 5.95, 64)
    for x in xs:
        cw, ct = float(CAN_W(x)), float(CAN_T(x))
        zb = body_top(x, cw) - 0.025
        a = np.linspace(0.0, math.pi, m)
        rows.append(np.stack([np.full(m, x), cw * np.sign(np.cos(a)) * np.abs(np.cos(a)) ** (2 / 2.3),
                              zb + max(ct - zb, 0.01) * np.sin(a) ** (2 / 2.3)], axis=1))
    return strip(rows)


def canopy_frame(c, n):
    """Material index for a canopy face (0 glass, 1 frame) from its centre in aircraft axes: the
    two rows of faces along each sill, the bow behind the seat, the two ends."""
    x, y, z = c
    low = (abs(y) / max(canopy_w(x), 1e-3)) ** 1.15 > 0.96
    return 1 if (low or 4.76 < x < 4.87 or x < 2.51 or x > 5.80) else 0


def body_side(x, z):
    """How far out the body's upper wall is at height z."""
    sec = body_section(x)
    on = np.nonzero(sec[6] >= z)[0]
    return float(sec[0][on[-1]]) if len(on) else 0.0


def gun_fairing():
    """The blister over the gun on the left side, behind its port."""
    rings = []
    for x in np.linspace(4.98, 6.45, 16):
        t = (x - 4.98) / (6.45 - 4.98)
        r = 0.085 * math.sin(math.pi * min(t * 1.7, 0.5 + 0.5 * t)) ** 0.7 + 0.002
        zc = 0.455 + 0.03 * t
        yc = -(body_side(x, zc) - 0.45 * r)
        a = np.linspace(0, 2 * math.pi, 12, endpoint=False)
        rings.append(np.stack([np.full(12, x), yc + 0.75 * r * np.cos(a), zc + r * np.sin(a)], axis=1))
    return loft(rings)


def ellipsoid(c, r, seg=14, rings=9):
    v, f = lathe([(-math.cos(t), math.sin(t) + 1e-4) for t in np.linspace(0.0, math.pi, rings)], seg)
    return [(c[0] + r[0] * x, c[1] + r[1] * y, c[2] + r[2] * z) for x, y, z in v], f


def cockpit():
    """What shows under the canopy: the glare shield and head-up display, the instrument panel, the
    side consoles and the seat. Returns {material: part}."""
    deck = body_top
    floor = lambda x: body_top(x) - TUB_DEPTH * tub_ramp(x)
    rim = lambda x: canopy_w(x) - SILL
    rings = []
    for x, hw, h in ((2.66, 0.20, 0.03), (2.85, 0.27, 0.09), (3.20, 0.31, 0.15), (3.50, 0.31, 0.18), (3.56, 0.29, 0.16)):
        a = np.linspace(0.0, math.pi, 13)
        rings.append(np.stack([np.full(13, x), hw * np.sign(np.cos(a)) * np.abs(np.cos(a)) ** (2 / 3.0),
                               deck(x) - 0.03 + (h + 0.03) * np.sin(a) ** (2 / 3.0)], axis=1))
    shield = loft(rings)
    z = deck(3.5) + 0.16
    hud_frame = merge([box(3.41, 3.425, 0.005, z, z + 0.17, y0=s * 0.085) for s in (-1, 1)] + [box(3.36, 3.50, 0.095, z - 0.02, z + 0.015)])
    hud_glass = ([(3.46, -0.08, z + 0.015), (3.46, 0.08, z + 0.015), (3.39, 0.08, z + 0.19), (3.39, -0.08, z + 0.19)], [(0, 1, 2, 3)])
    # the instrument panel under the shield, its two displays, the pedestal between the knees
    zp = deck(3.47)
    panel = merge([box(3.44, 3.50, rim(3.47) + 0.01, floor(3.52) + 0.10, zp + 0.10), box(3.50, 3.80, 0.085, floor(3.65) - 0.01, floor(3.65) + 0.27)])
    quad = lambda x, y0, y1, z0, z1: ([(x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1)], [(0, 1, 2, 3)])
    screens = merge([quad(3.503, s * 0.225, s * 0.075, zp - 0.20, zp - 0.05) for s in (-1, 1)] + [quad(3.503, -0.05, 0.05, zp - 0.02, zp + 0.06)])
    dials = merge([quad(3.502, -0.055, 0.055, zp - 0.21, zp - 0.05)] + [quad(3.502, s * 0.33, s * 0.245, zp - 0.17, zp - 0.07) for s in (-1, 1)])
    # the consoles along each wall, with the stick on the right and the throttle on the left
    xs = np.linspace(3.50, 4.82, 9)
    consoles = both(loft([np.array([(x, rim(x) - 0.14, deck(x) - 0.17), (x, rim(x) + 0.02, deck(x) - 0.17), (x, rim(x) + 0.02, floor(x) - 0.01),
                                    (x, rim(x) - 0.14, floor(x) - 0.01)]) for x in xs]))
    zc = deck(4.0) - 0.17
    controls = merge([box(3.99, 4.03, 0.014, zc, zc + 0.10, y0=rim(4.0) - 0.07), box(3.98, 4.05, 0.022, zc + 0.10, zc + 0.16, y0=rim(4.0) - 0.07),
                      box(3.86, 3.98, 0.03, zc, zc + 0.07, y0=-(rim(3.9) - 0.07))])
    # the seat: pan, a back leaning 27 degrees, headrest
    zf = floor(4.2)
    z0, z1 = zf + 0.15, deck(4.7) + 0.30
    lean = lambda zz: 4.38 + 0.50 * (zz - z0)

    def slab(za, zb, back, front, hw, y0=0.0):
        return loft([np.array([(lean(zz) - front, y0 - hw, zz), (lean(zz) - front, y0 + hw, zz), (lean(zz) + back, y0 + hw, zz),
                               (lean(zz) + back, y0 - hw, zz)]) for zz in (za, zb)])
    frame = merge([box(3.98, 4.44, 0.205, zf - 0.01, zf + 0.20), slab(z0, z1, 0.10, 0.0, 0.21), box(4.50, 4.92, 0.30, deck(4.7) - 0.25, deck(4.7) - 0.02)])
    cushion = merge([box(4.00, 4.40, 0.18, zf + 0.20, zf + 0.245), slab(z0 + 0.06, z1 - 0.22, 0.0, 0.035, 0.175), slab(z1 - 0.19, z1 - 0.01, 0.0, 0.05, 0.13)])
    straps = merge([slab(z0 + 0.10, z1 - 0.24, -0.035, 0.045, 0.024, y0=s * 0.075) for s in (-1, 1)] +
                   [box(4.02, 4.36, 0.024, zf + 0.245, zf + 0.255, y0=s * 0.10) for s in (-1, 1)])
    handle = merge([box(3.955, 3.985, 0.055, zf + 0.20, zf + 0.29), box(3.955, 3.985, 0.035, zf + 0.29, zf + 0.31)])
    return {'cockpit': merge([shield, hud_frame, panel]), 'hud': hud_glass, 'screen': screens, 'dial': dials, 'console': merge([consoles, controls]),
            'seat': frame, 'cushion': cushion, 'strap': straps, 'handle': handle}


def pylon(y, xa, xb, drop=0.36, thick=0.075):
    """An underwing pylon at span station y (right wing), from xa to xb along the wing."""
    rings = []
    for dz, fa, fb in ((0.0, 0.0, 0.0), (-0.55 * drop, 0.10, 0.02), (-drop, 0.24, 0.07)):
        x0, x1 = xa + fa * (xb - xa), xb - fb * (xb - xa)
        r = section_ring((x0, y, Z_WING + dz), x1 - x0, 0.0, 1.0, biconvex(thick / (x1 - x0)), (0, 1, 0), m=10)
        if dz == 0.0:        # the top follows the wing's chord line, which is twisted nose down
            r[:, 2] = Z_WING + (r[:, 0] - (x_le(y) + 0.25 * chord(y))) * math.sin(math.radians(WASHOUT * y / HALF))
        rings.append(r)
    return loft(rings)


def annulus(x0, x1, r0, r1, seg=30):
    """A flat ring about the engine's axis."""
    rings = []
    for a in np.linspace(0.0, 2 * math.pi, seg + 1):
        c, sn = math.cos(a), math.sin(a)
        rings.append(np.array([(x, r * c, Z_ENG + r * sn) for x, r in ((x0, r0), (x0, r1), (x1, r1), (x1, r0))]))
    return loft(rings, caps=False)


def nozzle(petals=15):
    """The nozzle's petals (outside and inside), the burner duct behind them, and what stands in
    it: the exhaust cone, two flame-holder rings and the struts that carry them."""
    def ring(x, r, step):
        pts = []
        for k in range(petals):
            for f, rr in ((0.02, r), (0.25, r), (0.48, r), (0.70, r), (0.74, r - step), (0.98, r - step)):
                a = 2 * math.pi * (k + f) / petals
                pts.append((x, rr * math.cos(a), Z_ENG + rr * math.sin(a)))
        return np.array(pts)
    outer = loft([ring(X_NOZ0 - 0.03, 0.566, 0.004), ring(X_NOZ0 + 0.20, 0.552, 0.012), ring(X_NOZ1, 0.44, 0.012), ring(X_NOZ1, 0.425, 0.010)], caps=False)
    inner = loft([ring(X_NOZ1, 0.425, 0.010), ring(13.95, 0.375, 0.008), ring(13.50, 0.46, 0.006), ring(12.95, 0.50, 0.0)], caps=False)
    wall = lambda x: np.array([(x, 0.50 * math.cos(a), Z_ENG + 0.50 * math.sin(a)) for a in np.linspace(0.0, 2 * math.pi, 30, endpoint=False)])
    duct = merge([loft([wall(12.40), wall(12.96)], caps=False), lathe([(12.40, 0.50), (12.42, 0.50)], 30, Z_ENG)])
    cone = lathe([(12.42, 0.26), (12.75, 0.20), (13.05, 0.09), (13.22, 0.012)], 20, Z_ENG)
    struts = []
    for k in range(8):
        a = 2 * math.pi * (k + 0.5) / 8
        d, n = np.array([0.0, math.cos(a), math.sin(a)]), np.array([0.0, -math.sin(a), math.cos(a)]) * 0.009
        c = np.array([0.0, 0.0, Z_ENG])
        struts.append(loft([np.array([c + d * r + n + [12.86, 0, 0], c + d * r - n + [12.86, 0, 0], c + d * r - n + [12.95, 0, 0], c + d * r + n + [12.95, 0, 0]])
                            for r in (0.10, 0.50)]))
    holder = merge([annulus(12.90, 12.96, 0.215, 0.255), annulus(12.90, 12.96, 0.365, 0.405), cone] + struts)
    return outer, inner, duct, holder


def blade(x, y, z, length, height, up=1.0, t=0.012):
    """A blade antenna standing on (x, y, z)."""
    rings = []
    for s in (0.0, 0.6, 1.0):
        c = length * (1.0 - 0.55 * s)
        x0 = x + 0.45 * length * s
        zz = z + up * height * s - up * 0.03 * (s == 0.0)
        tt = t * (1.0 - 0.5 * s)
        rings.append(np.array([(x0, y, zz), (x0 + 0.4 * c, y + tt, zz), (x0 + c, y, zz), (x0 + 0.4 * c, y - tt, zz)]))
    return loft(rings)


def details():
    """Small things that stand off the skin."""
    parts = [blade(6.72, 0.0, body_top(6.8), 0.34, 0.24), blade(5.25, 0.0, -0.667, 0.30, 0.20, up=-1.0),
             blade(9.0, 0.0, body_bot(9.1), 0.26, 0.16, up=-1.0)]
    for y in (-0.19, -0.065, 0.065, 0.19):      # the identification antennas ahead of the windscreen
        parts.append(blade(2.20, y, body_top(2.24, y), 0.11, 0.065, t=0.004))
    for s in (-1, 1):                           # angle of attack probes on the radome
        ye = float(Y_EDGE(1.62))
        parts.append(lathe([(1.50, 0.002), (1.62, 0.016), (1.66, 0.016)], 8, float(Z_MID(1.62)), s * (ye + 0.012)))
    # the arrester hook under the engine
    parts.append(loft([np.array([(x, -0.03, body_bot(x) - 0.005), (x, 0.03, body_bot(x) - 0.005), (x, 0.02, body_bot(x) - 0.06), (x, -0.02, body_bot(x) - 0.06)])
                       for x in (11.7, 12.3, 12.95)]))
    return merge(parts)


def fin_top():
    """The light and the antenna fairings at the top of the fin."""
    cap = loft([np.array([(x, -hw, FIN_Z1 + h), (x, hw, FIN_Z1 + h), (x, hw, FIN_Z1 - 0.01), (x, -hw, FIN_Z1 - 0.01)])
                for x, hw, h in ((13.60, 0.012, 0.0), (13.9, 0.03, 0.035), (14.55, 0.03, 0.035), (14.74, 0.012, 0.0))])
    rear = lathe([(14.42, 0.03), (14.50, 0.042), (14.68, 0.042), (14.73, 0.02)], 10, 3.33)
    return merge([cap, rear])


def probe():
    return lathe([(-PROBE, 0.004), (-0.06, 0.008), (0.04, 0.014)], 8, 0.0)


# ------------------------------------------------------------------ Blender scene
COLOURS = {
    # (base colour sRGB 0..1, metallic, roughness, alpha)
    'paint_mid': ((0.50, 0.525, 0.545), 0.0, 0.55, 1.0),
    'frame': ((0.33, 0.35, 0.37), 0.0, 0.55, 1.0),
    'nozzle': ((0.21, 0.195, 0.185), 0.75, 0.5, 1.0),
    'burner': ((0.11, 0.10, 0.095), 0.2, 0.7, 1.0),
    'holder': ((0.34, 0.31, 0.28), 0.3, 0.6, 1.0),
    'ceramic': ((0.50, 0.47, 0.42), 0.0, 0.7, 1.0),
    'duct': ((0.80, 0.81, 0.80), 0.0, 0.35, 1.0),
    'dark': ((0.03, 0.03, 0.032), 0.0, 0.8, 1.0),
    'cockpit': ((0.07, 0.075, 0.08), 0.0, 0.8, 1.0),
    'seat': ((0.10, 0.105, 0.11), 0.0, 0.7, 1.0),
    'cushion': ((0.20, 0.23, 0.17), 0.0, 0.9, 1.0),
    'strap': ((0.52, 0.50, 0.42), 0.0, 0.85, 1.0),
    'handle': ((0.80, 0.62, 0.08), 0.0, 0.5, 1.0),
    'console': ((0.115, 0.12, 0.125), 0.0, 0.65, 1.0),
    'screen': ((0.01, 0.03, 0.025), 0.0, 0.08, 1.0),
    'dial': ((0.025, 0.025, 0.028), 0.0, 0.25, 1.0),
    'paint_light': ((0.57, 0.595, 0.615), 0.0, 0.55, 1.0),
    'hud': ((0.60, 0.78, 0.68), 0.0, 0.03, 0.12),
    'canopy': ((0.40, 0.34, 0.24), 1.0, 0.04, 0.18),
    'missile': ((0.70, 0.71, 0.72), 0.0, 0.45, 1.0),
    'missile_fin': ((0.55, 0.56, 0.57), 0.0, 0.5, 1.0),
    'missile_nose': ((0.86, 0.85, 0.82), 0.0, 0.3, 1.0),
    'band_yellow': ((0.80, 0.62, 0.10), 0.0, 0.5, 1.0),
    'band_brown': ((0.38, 0.22, 0.12), 0.0, 0.5, 1.0),
    'metal': ((0.42, 0.43, 0.44), 0.7, 0.4, 1.0),
    'clay': ((0.62, 0.62, 0.62), 0.0, 0.6, 1.0),
}
SKIN = {}        # texture paths, filled in by main()
VIEWS = []       # --views a,b: render only these
SHOTS = {}       # --shot name=x,y,z,tx,ty,tz,lens: a camera at x,y,z looking at tx,ty,tz (aircraft axes)
CLAY = False


def srgb_to_linear(c):
    return tuple(((v + 0.055) / 1.055) ** 2.4 if v > 0.04045 else v / 12.92 for v in c)


def principled(m):
    try:
        m.use_nodes = True
    except Exception:
        pass
    return m.node_tree.nodes.get('Principled BSDF')


def material(name):
    if CLAY and name not in ('canopy', 'hud', 'dark'):
        name = 'clay'
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    m = bpy.data.materials.new(name)
    b = principled(m)
    if name == 'skin':
        nt = m.node_tree

        def tex(key, colour):
            n = nt.nodes.new('ShaderNodeTexImage')
            n.image = bpy.data.images.load(SKIN[key])
            if not colour:
                n.image.colorspace_settings.name = 'Non-Color'
            return n
        nt.links.new(tex('albedo', True).outputs['Color'], b.inputs['Base Color'])
        sep = nt.nodes.new('ShaderNodeSeparateColor')
        nt.links.new(tex('orm', False).outputs['Color'], sep.inputs[0])
        nt.links.new(sep.outputs[1], b.inputs['Roughness'])
        nt.links.new(sep.outputs[2], b.inputs['Metallic'])
        nm = nt.nodes.new('ShaderNodeNormalMap')
        nt.links.new(tex('normal', False).outputs['Color'], nm.inputs['Color'])
        nt.links.new(nm.outputs['Normal'], b.inputs['Normal'])
        return m
    col, metal, rough, alpha = COLOURS[name]
    b.inputs['Base Color'].default_value = (*srgb_to_linear(col), 1.0)
    b.inputs['Metallic'].default_value = metal
    b.inputs['Roughness'].default_value = rough
    b.inputs['Alpha'].default_value = alpha
    if name == 'screen':          # the displays are lit
        try:
            b.inputs['Emission Color'].default_value = (0.04, 0.20, 0.14, 1.0)
            b.inputs['Emission Strength'].default_value = 0.15
        except Exception:
            pass
    if alpha < 1.0:
        for attr, val in (('surface_render_method', 'BLENDED'), ('blend_method', 'BLEND')):
            try:
                setattr(m, attr, val)
            except Exception:
                pass
    m.diffuse_color = (*col, alpha)
    return m


def to_blender(p):
    """Aircraft axes (x aft, y right, z up, from the nose tip) -> Blender, origin at the CG."""
    return Vector((-p[1], p[0] - X_CG, p[2] - Z_CG))


def add(name, part, mat='skin', origin=None, proj='auto', pick=None, flat=False, sharp=38.0, regions=None):
    """A mesh object from (verts, faces) in aircraft axes. mat 'skin': the painted atlas, each face
    drawn from above, below or the side ('auto': whichever it faces; 'plan'; 'side'; or regions,
    an index into 'TBLR' for every face). A list of
    materials goes with pick(centre, normal) -> index, both in aircraft axes."""
    verts, faces = part
    o = to_blender(origin) if origin is not None else Vector((0, 0, 0))
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(to_blender(v) - o) for v in verts], [], faces)
    bm = bmesh.new()
    bm.from_mesh(me)
    given = None
    if regions is not None:
        given = bm.faces.layers.int.new('region')
        for f, r in zip(bm.faces, regions):
            f[given] = r
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=2e-5)
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=1e-5)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    names = mat if isinstance(mat, (list, tuple)) else [mat]
    air = lambda p: (p.y + o.y + X_CG, -(p.x + o.x), p.z + o.z + Z_CG)
    uvl = bm.loops.layers.uv.new('atlas') if 'skin' in names and SKIN else None
    for f in bm.faces:
        f.smooth = not flat
        n = (f.normal.y, -f.normal.x, f.normal.z)
        if pick is not None:
            f.material_index = pick(air(f.calc_center_median()), n)
        if uvl is not None and names[f.material_index] == 'skin':
            if given is not None:
                region = 'TBLR'[f[given]]
            elif proj == 'plan' or (proj == 'auto' and abs(n[2]) > 0.55):
                region = 'T' if n[2] >= 0 else 'B'
            else:
                region = 'L' if n[1] < 0 else 'R'
            for lp in f.loops:
                x, y, z = air(lp.vert.co)
                lp[uvl].uv = paint.uv(region, x, y if region in 'TB' else z)
    for e in bm.edges:
        if len(e.link_faces) == 2 and e.calc_face_angle(0.0) > math.radians(sharp):
            e.smooth = False
    bm.to_mesh(me)
    bm.free()
    for n in names:
        me.materials.append(material(n))
    ob = bpy.data.objects.new(name, me)
    ob.location = o
    bpy.context.scene.collection.objects.link(ob)
    return ob


def build():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    root = bpy.data.objects.new('F-16C', None)
    bpy.context.scene.collection.objects.link(root)
    obs = []
    side = lambda p: (p[0], -p[1], p[2])
    body, regions = fuselage()[:2]
    obs.append(add('fuselage', body, ['skin', 'cockpit'], pick=in_tub, sharp=100.0, regions=regions))
    obs.append(add('speedbrakes', speedbrakes(), proj='plan'))
    obs.append(add('gun_fairing', gun_fairing()))
    wing_box, (le_flap, hinge_le), (flaperon, hinge_fl) = wing_parts()
    obs.append(add('wing', both(wing_box), proj='plan'))
    obs.append(add('le_flap_R', le_flap, origin=hinge_le, proj='plan'))
    obs.append(add('le_flap_L', mirror(le_flap), origin=side(hinge_le), proj='plan'))
    obs.append(add('flaperon_R', flaperon, origin=hinge_fl, proj='plan'))
    obs.append(add('flaperon_L', mirror(flaperon), origin=side(hinge_fl), proj='plan'))
    rail, missile, fins = launcher_and_missile()
    obs.append(add('launchers', both(rail), 'paint_mid'))
    bands = lambda c, n: 1 if c[0] < 8.14 else 2 if 8.42 < c[0] < 8.50 else 3 if 8.95 < c[0] < 9.03 else 0
    obs.append(add('missiles', both(missile), ['missile', 'missile_nose', 'band_yellow', 'band_brown'], pick=bands))
    obs.append(add('missile_fins', both(fins), 'missile_fin'))
    tail, pivot = tailplane()
    obs.append(add('tailplane_R', tail, origin=pivot, proj='plan'))
    obs.append(add('tailplane_L', mirror(tail), origin=side(pivot), proj='plan'))
    fin, (rudder, hinge_r) = fin_parts()
    obs.append(add('fin', fin, proj='side'))
    obs.append(add('rudder', rudder, origin=hinge_r, proj='side'))
    obs.append(add('ventral_fins', both(ventral_fin()), proj='side'))
    obs.append(add('fin_top', fin_top(), 'paint_mid'))
    fairing, duct, face = intake()
    obs.append(add('intake', fairing))
    obs.append(add('intake_duct', duct, 'duct'))
    obs.append(add('engine_face', face, 'dark'))
    obs.append(add('canopy', canopy(), ['canopy', 'frame'], pick=canopy_frame))
    for mat, part in cockpit().items():
        obs.append(add('cockpit_' + mat if mat != 'cockpit' else 'cockpit', part, mat))
    pylons = merge([pylon(1.80, 7.03, 9.50, thick=0.06), pylon(3.05, 8.24, 9.66, drop=0.30, thick=0.05)])
    obs.append(add('pylons', both(pylons), 'paint_light'))
    outer, inner, duct, holder = nozzle()
    obs.append(add('nozzle', outer, 'nozzle', sharp=30.0))
    obs.append(add('nozzle_inner', inner, 'ceramic', sharp=30.0))
    obs.append(add('burner_duct', duct, 'burner'))
    obs.append(add('flame_holder', holder, 'holder'))
    obs.append(add('antennas', details(), 'paint_mid'))
    obs.append(add('probe', probe(), 'metal'))
    for ob in obs:
        ob.parent = root
    return root, obs


def measure(obs):
    lo, hi = Vector((1e9,) * 3), Vector((-1e9,) * 3)
    tris = 0
    for ob in obs:
        ob.data.calc_loop_triangles()
        tris += len(ob.data.loop_triangles)
        for v in ob.data.vertices:
            p = ob.location + v.co
            lo, hi = Vector(map(min, lo, p)), Vector(map(max, hi, p))
    return lo, hi, tris


def export_renderer(path, obs):
    """The aircraft for SmartImposter's own renderer (src/aircraft.rs): triangles with a normal and
    a texture coordinate at every corner, in the model frame the renderer flies (x right, y up,
    z aft, origin at the centre of gravity, metres). Parts: the body, the glass, and each moving
    surface about its own pivot with its hinge axis. Little endian:
      'SIAC', version; materials: name, base colour (linear) and alpha, metal, roughness, emission,
      flags (1 = the painted skin's textures); parts: name, pivot, hinge axis, flags (1 = glass),
      triangle count, positions, normals, texture coordinates (top-left origin), a material per
      triangle (padded to a multiple of four)."""
    import struct
    model = lambda v: (-v[0], v[2], v[1])                 # Blender (x left, y aft, z up) -> model
    air = lambda v: (v[1], v[2], v[0])                    # a direction in aircraft axes -> model
    unit = lambda v: tuple(np.array(v, float) / max(np.linalg.norm(v), 1e-9))
    c10, s10 = math.cos(TAIL_ANHEDRAL), math.sin(TAIL_ANHEDRAL)
    le = (le_hinge(HALF) - le_hinge(Y_LEF0), HALF - Y_LEF0, 0.0)
    fl = (fl_hinge(Y_FL1) - fl_hinge(Y_SHELF), Y_FL1 - Y_SHELF, 0.0)
    left = lambda v: (v[0], -v[1], v[2])
    axes = {'le_flap_R': le, 'le_flap_L': left(le), 'flaperon_R': fl, 'flaperon_L': left(fl), 'tailplane_R': (0.0, c10, -s10),
            'tailplane_L': (0.0, -c10, -s10), 'rudder': (fin_hinge(FIN_ZR1) - fin_hinge(FIN_ZR0), 0.0, FIN_ZR1 - FIN_ZR0)}
    names = []
    parts = {'body': dict(pivot=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 0.0), flags=0, tris=[]),
             'glass': dict(pivot=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 0.0), flags=1, tris=[])}
    for ob in obs:
        me = ob.data
        me.calc_loop_triangles()
        uvl = me.uv_layers.get('atlas')
        normals = me.corner_normals
        moving = ob.name in axes
        if moving:
            parts[ob.name] = dict(pivot=model(ob.location), axis=unit(air(axes[ob.name])), flags=0, tris=[])
        for t in me.loop_triangles:
            mat = me.materials[t.material_index].name
            if mat not in names:
                names.append(mat)
            part = parts[ob.name] if moving else parts['glass' if mat in ('canopy', 'hud') else 'body']
            corners = []
            for li, vi in zip(t.loops, t.vertices):
                co = me.vertices[vi].co if moving else me.vertices[vi].co + ob.location
                uv = uvl.data[li].uv if uvl is not None and mat == 'skin' else (0.0, 1.0)
                corners.append((model(co), model(normals[li].vector), (uv[0], 1.0 - uv[1])))
            part['tris'].append((corners, names.index(mat)))
    with open(path, 'wb') as f:
        def text(sx):
            b = sx.encode()
            f.write(struct.pack('<B', len(b)) + b)
        f.write(b'SIAC' + struct.pack('<II', 1, len(names)))
        for n in names:
            text(n)
            if n == 'skin':
                f.write(struct.pack('<4f2f3fI', 1, 1, 1, 1, 0, 0.55, 0, 0, 0, 1))
            else:
                col, metal, rough, alpha = COLOURS[n]
                glow = (0.006, 0.03, 0.021) if n == 'screen' else (0.0, 0.0, 0.0)
                f.write(struct.pack('<4f2f3fI', *srgb_to_linear(col), alpha, metal, rough, *glow, 0))
        f.write(struct.pack('<I', len(parts)))
        total = 0
        for name, part in parts.items():
            tris = part['tris']
            total += len(tris)
            text(name)
            f.write(struct.pack('<3f3fII', *part['pivot'], *part['axis'], part['flags'], len(tris)))
            for k in range(3):
                f.write(np.array([c[k] for tri, _ in tris for c in tri], dtype='<f4').tobytes())
            mats = bytes(m for _, m in tris)
            f.write(mats + bytes((-len(mats)) % 4))
    print(f'wrote {path}: {len(parts)} parts, {total} triangles, {len(names)} materials, {os.path.getsize(path)} bytes')


def text_triangles(s):
    """The triangles of a line of text (font units), for the painter."""
    cu = bpy.data.curves.new('text', 'FONT')
    cu.body = s
    for path in ('C:/Windows/Fonts/bahnschrift.ttf', 'C:/Windows/Fonts/arialbd.ttf'):
        if os.path.exists(path):
            try:
                cu.font = bpy.data.fonts.load(path, check_existing=True)
                break
            except Exception:
                pass
    ob = bpy.data.objects.new('text', cu)
    bpy.context.scene.collection.objects.link(ob)
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    me.calc_loop_triangles()
    tris = np.array([[(me.vertices[i].co.x, me.vertices[i].co.y) for i in t.vertices] for t in me.loop_triangles], dtype=np.float32)
    ev.to_mesh_clear()
    bpy.data.objects.remove(ob, do_unlink=True)
    bpy.data.curves.remove(cu)
    return tris


def save_image(path, rgb, colour):
    """Write an (H, W, 3) array (0..1, row 0 at the bottom) as an 8 bit PNG."""
    h, w = rgb.shape[:2]
    img = bpy.data.images.new(os.path.basename(path), w, h, alpha=False)
    if not colour:
        img.colorspace_settings.name = 'Non-Color'
    px = np.ones((h, w, 4), np.float32)
    px[..., :3] = rgb
    img.pixels.foreach_set(px.ravel())
    img.filepath_raw = path
    img.file_format = 'PNG'
    img.save()
    bpy.data.images.remove(img)


def paint_skin(out_dir):
    geo = dict(y_edge=Y_EDGE, z_top=Z_TOP, z_bot=Z_BOT, x_le=x_le, x_te=x_te, chord=chord, le_hinge=le_hinge, fl_hinge=fl_hinge,
               fin_le=fin_le, fin_te=fin_te, fin_hinge=fin_hinge, tail_le=tail_le, tail_te=tail_te, text=text_triangles,
               HALF=HALF, Y_SHELF=Y_SHELF, Y_LEF0=Y_LEF0, Y_FL1=Y_FL1, Z_WING=Z_WING, TAIL_Y1=TAIL_Y1, FIN=(FIN_Z0, FIN_ZR0, FIN_ZR1, FIN_Z1),
               can_w=CAN_W, seams=fuselage()[2])
    maps = paint.paint(geo)
    for key, colour in (('albedo', True), ('orm', False), ('normal', False)):
        SKIN[key] = os.path.join(out_dir, f'f16c_{key}.png')
        save_image(SKIN[key], maps.pop(key), colour)
        print('wrote', SKIN[key])


def use_gpu(sc):
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        for kind in ('OPTIX', 'CUDA'):
            try:
                prefs.compute_device_type = kind
                prefs.get_devices()
                on = [d for d in prefs.devices if d.type == kind]
                if on:
                    for d in prefs.devices:
                        d.use = d.type == kind
                    sc.cycles.device = 'GPU'
                    return kind
            except Exception:
                pass
    except Exception:
        pass
    return 'CPU'


def render(out_dir, tag=''):
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    print('rendering on', use_gpu(sc))
    sc.cycles.samples = 96
    try:
        sc.cycles.use_denoising = True
    except Exception:
        pass
    sc.render.resolution_x, sc.render.resolution_y = 1600, 900
    sc.render.image_settings.file_format = 'PNG'
    world = bpy.data.worlds.new('sky')
    try:
        world.use_nodes = True
    except Exception:
        pass
    nt = world.node_tree
    bg = nt.nodes['Background']
    sun_el, sun_az = math.radians(42.0), math.radians(-55.0)
    sky = nt.nodes.new('ShaderNodeTexSky')
    kind = None
    for k in ('MULTIPLE_SCATTERING', 'SINGLE_SCATTERING', 'NISHITA', 'HOSEK_WILKIE'):
        try:
            sky.sky_type = k
            kind = k
            break
        except Exception:
            pass
    print('sky', kind)
    for attr, val in (('sun_elevation', sun_el), ('sun_rotation', sun_az), ('sun_disc', True), ('altitude', 2500.0),
                      ('air_density', 1.0), ('aerosol_density', 1.0), ('dust_density', 1.0), ('ozone_density', 1.0)):
        try:
            setattr(sky, attr, val)
        except Exception:
            pass
    nt.links.new(sky.outputs[0], bg.inputs[0])
    bg.inputs[1].default_value = 0.09
    sc.world = world
    sc.view_settings.exposure = -0.45
    # the land far below: fields and woods, fading into the haze
    me = bpy.data.meshes.new('land')
    s = 90000.0
    me.from_pydata([(-s, -s, -3000), (s, -s, -3000), (s, s, -3000), (-s, s, -3000)], [], [(0, 1, 2, 3)])
    m = bpy.data.materials.new('land')
    b = principled(m)
    lt = m.node_tree
    co = lt.nodes.new('ShaderNodeTexCoord')
    fields = lt.nodes.new('ShaderNodeTexVoronoi')        # a patchwork of fields
    fields.inputs['Scale'].default_value = 0.0028
    try:
        fields.distance = 'CHEBYCHEV'
    except Exception:
        pass
    lt.links.new(co.outputs['Object'], fields.inputs['Vector'])
    crop = lt.nodes.new('ShaderNodeValToRGB')
    el = crop.color_ramp.elements
    el[0].position, el[0].color = 0.0, (0.030, 0.052, 0.022, 1.0)
    el[1].position, el[1].color = 1.0, (0.115, 0.105, 0.060, 1.0)
    for pos, col in ((0.35, (0.050, 0.075, 0.030, 1.0)), (0.7, (0.085, 0.090, 0.045, 1.0))):
        e = el.new(pos)
        e.color = col
    sep = lt.nodes.new('ShaderNodeSeparateColor')
    lt.links.new(fields.outputs['Color'], sep.inputs[0])
    lt.links.new(sep.outputs[0], crop.inputs[0])
    woods = lt.nodes.new('ShaderNodeTexNoise')           # woods and the lie of the land over them
    woods.inputs['Scale'].default_value = 0.00022
    woods.inputs['Detail'].default_value = 9.0
    lt.links.new(co.outputs['Object'], woods.inputs['Vector'])
    edge = lt.nodes.new('ShaderNodeValToRGB')
    edge.color_ramp.elements[0].position, edge.color_ramp.elements[1].position = 0.47, 0.56
    lt.links.new(woods.outputs[0], edge.inputs[0])
    land = lt.nodes.new('ShaderNodeMixRGB')
    land.inputs[2].default_value = (0.018, 0.032, 0.014, 1.0)
    lt.links.new(edge.outputs[0], land.inputs[0])
    lt.links.new(crop.outputs[0], land.inputs[1])
    lt.links.new(land.outputs[0], b.inputs['Base Color'])
    b.inputs['Roughness'].default_value = 0.9
    dist = lt.nodes.new('ShaderNodeCameraData')
    fog = lt.nodes.new('ShaderNodeMath')
    fog.operation = 'MULTIPLY'
    fog.inputs[1].default_value = 1.0 / 45000.0
    fog.use_clamp = True
    lt.links.new(dist.outputs['View Distance'], fog.inputs[0])
    haze = lt.nodes.new('ShaderNodeEmission')
    haze.inputs[0].default_value = (0.50, 0.62, 0.80, 1.0)
    haze.inputs[1].default_value = 0.6
    mix = lt.nodes.new('ShaderNodeMixShader')
    lt.links.new(fog.outputs[0], mix.inputs[0])
    lt.links.new(b.outputs[0], mix.inputs[1])
    lt.links.new(haze.outputs[0], mix.inputs[2])
    lt.links.new(mix.outputs[0], lt.nodes['Material Output'].inputs[0])
    me.materials.append(m)
    sc.collection.objects.link(bpy.data.objects.new('land', me))
    cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam'))
    cam.data.clip_end = 200000
    sc.collection.objects.link(cam)
    sc.camera = cam
    centre = Vector((0.0, -0.9, 0.35))
    #        position, lens, target
    views = {'front34': ((-13.5, -21, 5.5), 62, centre), 'side': ((-40, -0.9, 0.5), 80, centre), 'top': ((0.01, -0.9, 44), 50, centre),
             'rear34': ((13, 21, 5.0), 62, centre), 'below': ((-12, -19, -7.5), 62, centre), 'front': ((0.0, -40, 1.2), 105, centre),
             'nose': ((-5.2, -9.5, 1.9), 60, Vector((0.0, -4.6, 0.55))), 'tail': ((6.0, 9.5, 3.0), 55, Vector((0.0, 4.6, 1.0))),
             'gun': ((6.5, -6.8, 1.6), 60, Vector((0.6, -2.9, 0.3))),
             'chin': ((-3.2, -7.5, 0.3), 85, Vector((-0.5, -3.6, 0.0))), 'cockpit': ((2.3, -6.4, 2.5), 45, Vector((0.0, -4.25, 0.75))),
             'wing': ((-7.5, 4.5, 4.2), 50, Vector((-1.2, 0.6, 0.2))), 'belly': ((5.0, -7.0, -4.5), 50, Vector((0.0, 0.5, -0.3)))}
    views['hero'] = ((-15.0, -21.5, 3.6), 66, Vector((0.3, -0.4, 0.3)))
    if VIEWS or SHOTS:
        views = {k: v for k, v in views.items() if k in VIEWS}
    for name, v in SHOTS.items():        # close-ups asked for on the command line, in aircraft axes
        views[name] = (tuple(to_blender(v[0:3])), v[6], to_blender(v[3:6]))
    root = bpy.data.objects['F-16C']
    for name, (pos, lens, target) in views.items():
        cam.location = Vector(pos)
        cam.rotation_euler = (target - cam.location).to_track_quat('-Z', 'Y').to_euler()
        cam.data.lens = lens
        # the last picture: banked towards the camera, climbing a little
        root.rotation_euler = (math.radians(7.0), math.radians(-32.0), math.radians(4.0)) if name == 'hero' else (0.0, 0.0, 0.0)
        sc.cycles.samples = 256 if name == 'hero' else 96
        sc.render.filepath = os.path.join(out_dir, f'f16_{tag}{name}.png')
        bpy.ops.render.render(write_still=True)


def main():
    global CLAY, VIEWS
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    out_dir = argv[0] if argv else '.'
    os.makedirs(out_dir, exist_ok=True)
    CLAY = '--clay' in argv
    if '--views' in argv:
        VIEWS = argv[argv.index('--views') + 1].split(',')
    for i, a in enumerate(argv):
        if a == '--shot':
            name, vals = argv[i + 1].split('=')
            SHOTS[name] = [float(v) for v in vals.split(',')]
    if not CLAY and '--no-paint' not in argv:
        paint_skin(out_dir)
    elif not CLAY:
        for key in ('albedo', 'orm', 'normal'):
            SKIN[key] = os.path.join(out_dir, f'f16c_{key}.png')
    root, obs = build()
    lo, hi, tris = measure(obs)
    print(f'F-16C: {len(obs)} objects, {tris} triangles')
    print(f'  length {hi.y - lo.y:.2f} m (published {LENGTH}), span {hi.x - lo.x:.2f} m (9.45 over the launchers, ~10.0 with missiles), '
          f'fin tip {hi.z + Z_CG:.2f} m above the nose, belly {-(lo.z + Z_CG):.2f} m below it')
    print(f'  centre of gravity at x = {X_CG:.2f} m, wing leading edge at the centre line x = {X_WING:.2f} m')
    if not CLAY:
        path = os.path.join(out_dir, 'f16c.glb')
        bpy.ops.export_scene.gltf(filepath=path, export_format='GLB', export_yup=True, export_image_format='JPEG', export_jpeg_quality=90)
        print('wrote', path, os.path.getsize(path), 'bytes')
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, 'f16c.blend'))
        export_renderer(os.path.join(out_dir, 'f16c.acm'), obs)
    if '--renders' in argv:
        render(out_dir, 'clay_' if CLAY else '')


main()
