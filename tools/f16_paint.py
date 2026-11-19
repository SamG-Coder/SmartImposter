"""Paints the F-16's skin for make_f16.py: one atlas in which the aircraft is drawn four times, from
above (T), from below (B) and from each side (L, R), at 256 pixels per metre. Every face of the
model takes its texture from the drawing it faces, so everything here is drawn in aircraft
coordinates (x aft from the radome's tip, y right, z up, metres): the paint scheme, panel lines,
rivet rows, markings, then the wear. Returns base colour, occlusion/roughness/metal and a normal
map (rows bottom to top, as Blender stores images).
"""
import math

import numpy as np

D = 256.0
W, H = 4096, 8192
X0 = -0.45
#        first row, the lowest y or z, rows
REG = {'T': (0, -5.125, 2624), 'B': (2624, -5.125, 2624), 'L': (5248, -1.2, 1280), 'R': (6528, -1.2, 1280)}

# the greys of the scheme (sRGB): FS 36118 above, 36270 on the nose, fin and outer panels, 36375 below
C_DARK = np.array((0.345, 0.365, 0.385), np.float32)
C_MID = np.array((0.46, 0.485, 0.505), np.float32)
C_LIGHT = np.array((0.57, 0.595, 0.615), np.float32)
C_RADOME = np.array((0.43, 0.445, 0.45), np.float32)
C_METAL = np.array((0.37, 0.355, 0.34), np.float32)
C_MARK = np.array((0.25, 0.265, 0.28), np.float32)
C_SLIME = np.array((0.70, 0.73, 0.50), np.float32)
C_DECK = np.array((0.07, 0.075, 0.08), np.float32)
RINGS = (1.89, 2.40, 3.07, 4.29, 5.04, 5.73, 6.41, 7.02, 7.67, 8.60, 9.72, 10.63, 11.55, 12.26, 12.78, 13.12)


def uv(region, a, b):
    r0, b0, _ = REG[region]
    return ((a - X0) * D / W, (r0 + (b - b0) * D) / H)


def sstep(d, w):
    t = np.clip(0.5 + d / w, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def noise(shape, cy, cx, rng):
    """Value noise, 0..1, cells of cy by cx pixels."""
    h, w = shape
    g = rng.random((int(h / cy) + 3, int(w / cx) + 3)).astype(np.float32)
    yy, xx = np.arange(h, dtype=np.float32) / cy, np.arange(w, dtype=np.float32) / cx
    y0, x0 = yy.astype(np.int32), xx.astype(np.int32)
    fy, fx = yy - y0, xx - x0
    fy, fx = (fy * fy * (3 - 2 * fy))[:, None], (fx * fx * (3 - 2 * fx))[None, :]
    r0, r1 = g[y0], g[y0 + 1]
    return (r0[:, x0] * (1 - fx) + r0[:, x0 + 1] * fx) * (1 - fy) + (r1[:, x0] * (1 - fx) + r1[:, x0 + 1] * fx) * fy


def fbm(shape, cell, rng, octaves=4):
    out, amp, total = np.zeros(shape, np.float32), 1.0, 0.0
    for _ in range(octaves):
        out += amp * noise(shape, cell, cell, rng)
        total += amp
        amp *= 0.5
        cell = max(cell / 2.0, 1.5)
    return out / total


def blur(a, r):
    out = a
    for axis in (0, 1):
        acc = np.zeros_like(out)
        for k in range(-r, r + 1):
            acc += np.roll(out, k, axis=axis)
        out = acc / (2 * r + 1)
    return out


def stroke(mask, pts, width=1.6, value=1.0):
    """An anti-aliased polyline, pts in pixels (column, row)."""
    n, w = mask.shape
    hw = 0.5 * width
    for (c0, r0), (c1, r1) in zip(pts[:-1], pts[1:]):
        k = int(math.hypot(c1 - c0, r1 - r0) / 200.0) + 1
        for i in range(k):
            a0, b0 = c0 + (c1 - c0) * i / k, r0 + (r1 - r0) * i / k
            a1, b1 = c0 + (c1 - c0) * (i + 1) / k, r0 + (r1 - r0) * (i + 1) / k
            ca, cb = max(int(math.floor(min(a0, a1) - hw - 1)), 0), min(int(math.ceil(max(a0, a1) + hw + 2)), w)
            ra, rb = max(int(math.floor(min(b0, b1) - hw - 1)), 0), min(int(math.ceil(max(b0, b1) + hw + 2)), n)
            if ca >= cb or ra >= rb:
                continue
            cc, rr = np.arange(ca, cb, dtype=np.float32)[None, :], np.arange(ra, rb, dtype=np.float32)[:, None]
            dx, dy = a1 - a0, b1 - b0
            l2 = dx * dx + dy * dy
            t = np.clip(((cc - a0) * dx + (rr - b0) * dy) / l2, 0.0, 1.0) if l2 > 1e-9 else 0.0
            d = np.hypot(cc - (a0 + t * dx), rr - (b0 + t * dy))
            sub = mask[ra:rb, ca:cb]
            np.maximum(sub, value * np.clip(hw + 0.5 - d, 0.0, 1.0), out=sub)


def dots(mask, pts, spacing):
    """A row of rivets along a polyline."""
    n, w = mask.shape
    cs, rs, carry = [], [], 0.0
    for (c0, r0), (c1, r1) in zip(pts[:-1], pts[1:]):
        length = math.hypot(c1 - c0, r1 - r0)
        if length < 1e-6:
            continue
        t = np.arange(carry, length, spacing)
        carry = (t[-1] + spacing - length) if len(t) else carry - length
        cs.append(c0 + (c1 - c0) * t / length)
        rs.append(r0 + (r1 - r0) * t / length)
    if not cs:
        return
    ci, ri = np.rint(np.concatenate(cs)).astype(np.int64), np.rint(np.concatenate(rs)).astype(np.int64)
    ok = (ci >= 1) & (ci < w - 1) & (ri >= 1) & (ri < n - 1)
    ci, ri = ci[ok], ri[ok]
    mask[ri, ci] = 1.0
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        np.maximum.at(mask, (ri + dr, ci + dc), 0.3)


def fill(mask, poly, value=1.0, ss=3):
    """A filled polygon (even-odd), pts in pixels."""
    poly = np.asarray(poly, np.float32)
    n, w = mask.shape
    ca, cb = max(int(math.floor(poly[:, 0].min())) - 1, 0), min(int(math.ceil(poly[:, 0].max())) + 2, w)
    ra, rb = max(int(math.floor(poly[:, 1].min())) - 1, 0), min(int(math.ceil(poly[:, 1].max())) + 2, n)
    if ca >= cb or ra >= rb:
        return
    cc = (np.arange((cb - ca) * ss, dtype=np.float32) + 0.5) / ss + ca - 0.5
    rr = (np.arange((rb - ra) * ss, dtype=np.float32) + 0.5) / ss + ra - 0.5
    inside = np.zeros((len(rr), len(cc)), bool)
    for (xa, ya), (xb, yb) in zip(poly, np.roll(poly, -1, axis=0)):
        if ya == yb:
            continue
        cond = (ya > rr) != (yb > rr)
        xi = xa + (rr - ya) * (xb - xa) / (yb - ya)
        inside ^= cond[:, None] & (cc[None, :] < xi[:, None])
    cov = inside.reshape(rb - ra, ss, cb - ca, ss).mean(axis=(1, 3)).astype(np.float32)
    sub = mask[ra:rb, ca:cb]
    np.maximum(sub, value * cov, out=sub)


class Layer:
    """One of the four drawings: its part of the atlas, in aircraft coordinates."""

    def __init__(self, region, alb, rough, metal, height):
        self.region = region
        self.r0, self.b0, self.n = REG[region]
        sl = slice(self.r0, self.r0 + self.n)
        self.alb, self.rough, self.metal, self.height = alb[sl], rough[sl], metal[sl], height[sl]
        self.x = (X0 + (np.arange(W, dtype=np.float32) + 0.5) / D)[None, :]
        self.b = (self.b0 + (np.arange(self.n, dtype=np.float32) + 0.5) / D)[:, None]
        self.shape = (self.n, W)
        self.lines = np.zeros(self.shape, np.float32)
        self.rivets = np.zeros(self.shape, np.float32)
        self.plan = region in 'TB'
        self.panels = []

    def px(self, pts):
        return [((a - X0) * D - 0.5, (b - self.b0) * D - 0.5) for a, b in pts]

    def line(self, pts, width=1.5, value=1.0):
        stroke(self.lines, self.px(pts), width, value)

    def rivet(self, pts, spacing=0.05):
        dots(self.rivets, self.px(pts), spacing * D)

    def rrect(self, a0, b0, a1, b1, r=0.05, width=1.4, rivets=True):
        """A panel with rounded corners, and the fasteners inside its edge."""
        a0, a1, b0, b1 = min(a0, a1), max(a0, a1), min(b0, b1), max(b0, b1)
        self.panels.append((a0, b0, a1, b1))
        for inset, draw in ((0.0, True), (0.028, False)):
            pts = []
            rr = max(r - inset, 0.005)
            for ca, cb, t0 in ((a1 - r, b1 - r, 0), (a0 + r, b1 - r, 90), (a0 + r, b0 + r, 180), (a1 - r, b0 + r, 270)):
                for t in np.linspace(math.radians(t0), math.radians(t0 + 90), 5):
                    pts.append((ca + rr * math.cos(t), cb + rr * math.sin(t)))
            pts.append(pts[0])
            if draw:
                self.line(pts, width)
            elif rivets:
                self.rivet(pts, 0.04)

    def mask(self):
        return np.zeros(self.shape, np.float32)

    def poly(self, m, pts, value=1.0):
        fill(m, self.px(pts), value)

    def tint(self, m, colour, alpha=1.0):
        a = (m * alpha)[..., None]
        self.alb *= 1.0 - a
        self.alb += a * np.asarray(colour, np.float32)

    def text(self, m, tris, ca, cb, height, flip=False):
        """A line of text (triangles from the font) centred on (ca, cb), its capitals height metres."""
        lo, hi = tris.reshape(-1, 2).min(axis=0), tris.reshape(-1, 2).max(axis=0)
        s = height / (hi[1] - lo[1])
        mid = 0.5 * (lo + hi)
        for t in tris:
            self.poly(m, [(ca + (-1.0 if flip else 1.0) * (p[0] - mid[0]) * s, cb + (p[1] - mid[1]) * s) for p in t])


# ------------------------------------------------------------------ the paint scheme
def zones(L, G, rng):
    x, b = L.x, L.b
    wob = 0.035 * (fbm(L.shape, 90, rng, 2) - 0.5)         # the edges are sprayed freehand
    col = np.empty(L.shape + (3,), np.float32)
    col[:] = C_MID
    if L.region == 'T':
        ay = np.abs(b)
        yd = np.where(x < 10.7, np.clip(1.9 + 0.25 * (x - 7.0), 1.7, 2.6), 1.12)
        dark = sstep(x - (2.1 + 2.3 * ay) + wob, 0.10) * sstep(yd - ay + wob, 0.12) * sstep(12.9 - x + wob, 0.10)
        col += dark[..., None] * (C_DARK - C_MID)
    elif L.region == 'B':
        ay = np.abs(b)
        light = sstep(x - 4.25, 0.05) * np.ones_like(ay)
        col += light[..., None] * (C_LIGHT - C_MID)
    else:
        # the same edge as in the drawing from above, where the two meet on the shoulder
        zl = np.where(x < 3.13, 0.62 + (3.13 - x) * 2.0, np.where(x < 3.7, 0.62 + (x - 3.13) * (0.18 - 0.62) / 0.57, 0.18))
        fin = sstep(b - 1.06, 0.02) * sstep(x - 9.0, 0.05)
        dark = sstep(b - zl + wob, 0.08) * sstep(12.9 - x + wob, 0.10) * (1.0 - fin)
        light = (1.0 - sstep(b - 0.18 + wob, 0.08)) * sstep(x - 4.25, 0.05)
        col += dark[..., None] * (C_DARK - C_MID) + light[..., None] * (C_LIGHT - C_MID)
    rad = (x < 1.89).astype(np.float32) * np.ones_like(b)
    col += rad[..., None] * (C_RADOME - col)
    L.alb[:] = col
    L.rough[:] = 0.56 - 0.06 * rad
    # the bare metal round the engine's nozzle
    boom = (x < 0.24).astype(np.float32) * np.ones_like(b)       # the metal foot of the nose probe
    L.alb += boom[..., None] * (np.array((0.52, 0.53, 0.54), np.float32) - L.alb)
    if L.plan:
        met = sstep(x - 12.78, 0.006) * sstep(0.70 - np.abs(b), 0.01)
    else:
        met = sstep(x - 12.78, 0.006) * (1.0 - sstep(0.075 - np.abs(b - 0.22), 0.01)) * (b < 1.0) * (b > -0.5)
    heat = noise(L.shape, 30, 110, rng)
    band = 0.5 + 0.5 * np.sin((x - 12.78) * 21.0 + 3.0 * heat)        # straw to blue, ring by ring
    tintc = C_METAL[None, None, :] * (0.85 + 0.25 * heat[..., None]) + np.array((0.02, 0.0, -0.03), np.float32) * (band[..., None] - 0.5) * 2.0
    L.alb += met[..., None] * (tintc - L.alb)
    L.metal[:] = np.maximum(0.92 * met, 0.85 * boom)
    L.rough += np.maximum(met, boom) * (0.40 - L.rough)
    if L.region == 'T':                                   # the cockpit floor under the glass
        cw = np.interp(x, G['can_w'].x, G['can_w'].v, left=0.0, right=0.0)
        deck = sstep(cw - 0.035 - np.abs(b), 0.01) * (x > 2.5) * (x < 5.95)
        L.alb += deck[..., None] * (C_DECK - L.alb)
        L.rough += deck * (0.85 - L.rough)


# ------------------------------------------------------------------ panel lines and rivets
def wing_lines(L, G):
    half, ys, yl, yf = G['HALF'], G['Y_SHELF'], G['Y_LEF0'], G['Y_FL1']
    lh, fh, xle, xte = G['le_hinge'], G['fl_hinge'], G['x_le'], G['x_te']
    for s in (-1.0, 1.0):
        L.line([(lh(y), s * y) for y in (yl, half)], 2.2)
        L.line([(fh(y), s * y) for y in (ys, yf)], 2.2)
        L.line([(fh(y), s * y) for y in (yf, half - 0.03)], 1.3)
        L.line([(fh(yf), s * yf), (xte(yf), s * yf)], 2.2)
        L.line([(xle(yl) - 0.1, s * yl), (lh(yl), s * yl)], 2.0)
        L.line([(7.0, s * ys), (13.9, s * ys)], 1.5)
        for y in (1.75, 2.3, 2.85, 3.4, 3.95):
            L.line([(lh(y), s * y), (fh(y), s * y)], 1.2)
            for o in (-0.03, 0.03):
                L.rivet([(lh(y + o) + 0.03, s * (y + o)), (fh(y + o) - 0.03, s * (y + o))])
        L.line([(xle(4.47), s * 4.47), (xte(4.47), s * 4.47)], 1.3)
        for f in (0.04, 0.2, 0.36, 0.52, 0.68, 0.84, 0.96):
            L.rivet([(lh(y) + f * (fh(y) - lh(y)), s * y) for y in (1.42, 4.45)])
        for f in (0.35, 0.75):
            L.rivet([(xle(y) + f * (lh(y) - xle(y)), s * y) for y in (yl + 0.04, half - 0.04)])
        L.rivet([(fh(y) + 0.5 * (xte(y) - fh(y)), s * y) for y in (ys + 0.04, yf - 0.04)])
        # the tailplane
        tl, tt, y1 = G['tail_le'], G['tail_te'], G['TAIL_Y1']
        L.line([(tl(y) + 0.14, s * y) for y in (ys, y1)], 1.3)
        for y in (1.32, 2.0, 2.70):
            L.line([(tl(y) + (0.14 if y < 2.6 else 0.0), s * y), (tt(y), s * y)], 1.2)
        for f in (0.3, 0.5, 0.7, 0.9):
            L.rivet([(tl(y) + 0.14 + f * (tt(y) - tl(y) - 0.14), s * y) for y in (ys + 0.04, 2.66)])


def structure(L, G):
    ye, zt, zb = G['y_edge'], G['z_top'], G['z_bot']
    for xr in RINGS:
        w = 2.4 if xr in (1.89, 12.78) else 1.4
        if L.plan:
            yb = min(float(ye(xr)), G['Y_SHELF'] if xr > 6.8 else 9.0)
            L.line([(xr, -yb), (xr, yb)], w)
            L.rivet([(xr + 0.03, -yb), (xr + 0.03, yb)])
        else:
            L.line([(xr, float(zb(xr))), (xr, float(zt(xr)))], w)
            L.rivet([(xr + 0.03, float(zb(xr))), (xr + 0.03, float(zt(xr)))])
    for f in (-0.45, 0.0, 0.45):                         # the lightning strips along the radome
        xs = np.linspace(0.30, 1.6, 12)
        if L.plan:
            L.line([(xx, f * float(ye(xx))) for xx in xs], 1.3, 0.55)
        else:
            L.line([(xx, 0.5 * (float(zt(xx)) + float(zb(xx))) + f * 0.5 * (float(zt(xx)) - float(zb(xx)))) for xx in xs], 1.3, 0.55)
    # the seams between the drawings run along rows of the body's mesh: a panel line on each
    top = [p for p in G['seams']['top'] if 0.4 < p[0] < 5.70], [p for p in G['seams']['top'] if 10.66 < p[0] < 12.74]
    bot = [p for p in G['seams']['bottom'] if 0.4 < p[0] < 4.25], [p for p in G['seams']['bottom'] if 6.65 < p[0] < 12.74]
    for run in (top if L.region == 'T' else bot if L.region == 'B' else top + bot):
        for sgn in ((-1.0, 1.0) if L.plan else (1.0,)):
            L.line([(p[0], sgn * p[1]) if L.plan else (p[0], p[2]) for p in run], 1.4)
    if L.plan:
        wing_lines(L, G)
    if L.region == 'T':
        for s in (-1.0, 1.0):
            L.line([(6.0, s * 0.17), (11.0, s * 0.17)], 1.3)
            L.line([(6.0, s * 0.46), (7.67, s * 0.46)], 1.3)
            L.line([(11.55, s * 0.79), (12.78, s * 0.79)], 1.3)
            for i in range(5):
                for ya, yb in ((0.22, 0.48), (0.52, 0.78)):
                    L.rrect(7.75 + 0.385 * i, s * ya, 7.75 + 0.385 * i + 0.34, s * yb)
            for j in range(2):
                for ya, yb in ((0.21, 0.40), (0.44, 0.62), (0.66, 0.80)):
                    L.rrect(9.80 + 0.84 * j, s * ya, 9.80 + 0.84 * j + 0.78, s * yb)
            L.rrect(5.78, s * 0.56, 6.38, s * 0.95, 0.06)
            L.rrect(6.47, s * 0.20, 6.98, s * 0.70, 0.05)
            L.rrect(12.95, s * 0.70, 13.88, s * 1.03, 0.01, 2.0, rivets=False)
            L.line([(12.95, s * 0.86), (13.88, s * 0.86)], 1.2)
        L.rrect(1.95, -0.20, 2.38, 0.20)
        L.line([(7.72, -0.10), (8.40, -0.07), (8.40, 0.07), (7.72, 0.10), (7.72, -0.10)], 1.5)
        L.rrect(7.80, -0.035, 8.20, 0.035, 0.02, rivets=False)
    elif L.region == 'B':
        L.rrect(4.95, -0.14, 6.05, 0.14, 0.04, 2.0)
        L.line([(4.95, 0.0), (6.05, 0.0)], 1.5)
        L.rrect(1.95, -0.25, 2.9, 0.25)
        L.line([(9.72, 0.0), (12.78, 0.0)], 1.3)
        for s in (-1.0, 1.0):
            L.rrect(7.30, s * 0.12, 9.00, s * 0.55, 0.06, 2.0)
            L.line([(8.2, s * 0.12), (8.2, s * 0.55)], 1.3)
            L.line([(7.3, s * 0.10), (9.72, s * 0.10)], 1.2)
            L.line([(9.72, s * 0.36), (12.78, s * 0.36)], 1.3)
            L.line([(4.4, s * 0.30), (7.3, s * 0.30)], 1.2)
            L.rrect(9.85, s * 0.40, 10.6, s * 0.55)
            L.rrect(6.3, s * 0.34, 7.1, s * 0.55)
            for yp, xa, xb in ((1.80, 7.1, 9.5), (3.05, 8.3, 9.66)):
                L.rrect(xa, s * yp - 0.045, xb, s * yp + 0.045, 0.03, rivets=False)
            L.rrect(12.95, s * 0.70, 13.88, s * 1.03, 0.01, 2.0, rivets=False)
    else:
        L.rrect(1.98, -0.04, 2.36, 0.42)
        # the doors on the forebody's wall, clear of the fillet where it meets the strake
        L.rrect(2.50, 0.38, 3.02, 0.63, 0.05)
        L.rrect(3.12, 0.40, 4.22, 0.70, 0.06)
        L.line([(6.41, 0.62), (12.26, 0.62)], 1.3)
        L.line([(7.67, 0.82), (9.72, 0.82)], 1.2)
        L.line([(4.45, -0.33), (7.30, -0.33)], 1.3)
        L.line([(5.5, -0.04), (7.3, -0.04)], 1.2)
        L.rrect(7.30, -0.46, 8.98, -0.10, 0.06, 1.8)
        L.line([(9.72, -0.12), (12.78, -0.12)], 1.3)
        L.rrect(10.75, -0.36, 11.45, -0.16)
        L.rrect(9.85, 0.34, 10.55, 0.56)
        # the fin
        z0, zr0, zr1, z1 = G['FIN']
        fl, ft, fh = G['fin_le'], G['fin_te'], G['fin_hinge']
        L.line([(fl(z) + 0.12, z) for z in (z0, 3.41)], 1.3)
        L.line([(fh(z), z) for z in (zr0, zr1)], 2.2)
        for z in (zr0, zr1):
            L.line([(fh(z), z), (ft(z), z)], 2.2)
        for z in (1.95, 2.45, 2.95):
            L.line([(fl(z) + 0.12, z), (fh(z), z)], 1.2)
            L.rivet([(fl(z) + 0.15, z + 0.03), (fh(z) - 0.03, z + 0.03)])
        L.line([(fl(3.41), 3.41), (ft(3.41), 3.41)], 1.5)
        L.line([(11.11, z0), (13.66, z0)], 1.5)
        L.line([(9.2, 1.04), (13.6, 1.04)], 1.3)
        for xv in (11.6, 12.4, 13.2):
            L.line([(xv, 1.04), (xv, z0)], 1.2)
        for off in (0.45, 0.80, 1.15):
            ztop = min(3.38, z0 + (12.83 - 11.11 - off - 0.08) / (1.134 - 0.684))
            if ztop > zr0 + 0.2:
                L.rivet([(fl(z) + off, z) for z in (zr0, ztop)])
        L.rivet([(fh(z) + 0.5 * (ft(z) - fh(z)), z) for z in (zr0 + 0.04, zr1 - 0.04)])


def apply_structure(L, rng):
    # each access panel has been off and on, and repainted, on its own: a shade of its own
    for a0, b0, a1, b1 in L.panels:
        ca, cb = int((a0 - X0) * D) + 1, int((a1 - X0) * D)
        ra, rb = int((b0 - L.b0) * D) + 1, int((b1 - L.b0) * D)
        if 0 <= ra < rb <= L.n and ca < cb:
            L.alb[ra:rb, ca:cb] *= 1.0 + rng.uniform(-0.05, 0.035)
            L.rough[ra:rb, ca:cb] += rng.uniform(-0.04, 0.04)
    halo = np.clip(blur(L.lines, 6) * 3.0, 0.0, 1.0)
    # paint chipped off along the edges of panels, down to the primer
    chip = np.clip(noise(L.shape, 2.2, 3.5, rng) * noise(L.shape, 30, 60, rng) * 3.0 - 1.25, 0.0, 1.0) * halo * (1.0 - L.metal)
    L.tint(chip, (0.60, 0.62, 0.56), 0.55)
    L.alb *= ((1.0 - 0.55 * L.lines) * (1.0 - 0.22 * L.rivets) * (1.0 - 0.15 * halo))[..., None]
    L.rough += 0.15 * L.lines + 0.05 * halo
    L.height -= L.lines + 0.5 * L.rivets


# ------------------------------------------------------------------ markings
def insignia(L, m, ca, cb, r, up, squash=1.0):
    """The national star and bars, one colour: the disc and bars filled, the star left bare."""
    ua, ub = up
    ra, rb = ub, -ua
    p = lambda u, v: (ca + u * ra + v * squash * ua, cb + u * rb + v * squash * ub)
    tmp = L.mask()
    L.poly(tmp, [p(r * math.cos(t), r * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 48, endpoint=False)])
    for s in (-1.0, 1.0):
        L.poly(tmp, [p(s * 0.97 * r, 0.309 * r), p(s * 2.0 * r, 0.309 * r), p(s * 2.0 * r, -0.191 * r), p(s * 0.97 * r, -0.191 * r)])
    star = L.mask()
    L.poly(star, [p((r if k % 2 == 0 else 0.382 * r) * math.sin(math.radians(36 * k)), (r if k % 2 == 0 else 0.382 * r) * math.cos(math.radians(36 * k)))
                  for k in range(10)])
    np.maximum(m, tmp * (1.0 - star), out=m)


def markings(L, G, rng):
    m, slime = L.mask(), L.mask()
    if L.region == 'T':
        insignia(L, m, 9.15, -3.0, 0.31, (-1.0, 0.0))
        for s in (-1.0, 1.0):                             # on the body's shoulder, which slopes 45 degrees
            insignia(L, m, 11.2, s * 0.40, 0.16, (0.0, -s), 0.77)
    elif L.region == 'B':
        insignia(L, m, 9.2, 2.42, 0.24, (-1.0, 0.0))        # between the two pylons
    if L.plan:
        for s in (-1.0, 1.0):
            L.poly(slime, [(9.75, s * 4.38), (10.35, s * 4.38), (10.35, s * 4.43), (9.75, s * 4.43)])
    else:
        flip = L.region == 'R'
        text = G['text']
        L.text(m, text('SI'), 13.02, 2.50, 0.46, flip)
        L.text(m, text('AF 26'), 12.36, 2.13, 0.10, flip)
        L.text(m, text('016'), 12.42, 1.88, 0.25, flip)
        # warning triangle and rescue arrow under the canopy, stencils by the intake
        L.poly(m, [(3.30, 0.72), (3.44, 0.72), (3.37, 0.60)], 0.9)
        L.poly(m, [(3.62, 0.70), (3.86, 0.70), (3.86, 0.73), (3.94, 0.675), (3.86, 0.62), (3.86, 0.65), (3.62, 0.65)], 0.8)
        for z in (-0.50, -0.14):
            L.poly(m, [(4.50, z), (4.80, z), (4.80, z + 0.02), (4.50, z + 0.02)], 0.8)
        L.poly(slime, [(2.02, 0.385), (2.62, 0.385), (2.62, 0.43), (2.02, 0.43)])
        L.poly(slime, [(11.75, 0.50), (12.35, 0.50), (12.35, 0.545), (11.75, 0.545)])
        fl = G['fin_le']
        L.poly(slime, [(fl(2.50) + 0.16, 2.50), (fl(2.50) + 0.22, 2.50), (fl(2.98) + 0.22, 2.98), (fl(2.98) + 0.16, 2.98)])
        # the band across the top of the fin
        band, edge, cap = L.mask(), L.mask(), L.mask()
        fl, ft = G['fin_le'], G['fin_te']
        L.poly(cap, [(fl(3.41), 3.41), (ft(3.41), 3.41), (ft(3.60), 3.60), (fl(3.60), 3.60)])      # the dielectric cap
        L.tint(cap, (0.52, 0.53, 0.525), 0.85)
        L.poly(band, [(fl(3.12), 3.12), (ft(3.12), 3.12), (ft(3.30), 3.30), (fl(3.30), 3.30)])
        for z in (3.12, 3.30):
            L.poly(edge, [(fl(z), z - 0.012), (ft(z), z - 0.012), (ft(z), z + 0.012), (fl(z), z + 0.012)])
        L.tint(band, (0.13, 0.19, 0.36), 0.92)
        L.tint(edge, (0.72, 0.73, 0.72), 0.9)
        light = L.mask()
        L.poly(light, [(4.95 + 0.04 * math.cos(t), -0.30 + 0.03 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 16, endpoint=False)])
        L.tint(light, (0.10, 0.55, 0.25) if flip else (0.65, 0.08, 0.06), 1.0)
        L.rough -= 0.3 * light
    # small stencilling: rows of dashes where the servicing notes are
    sten = L.mask()
    for _ in range(70):
        a = rng.uniform(2.0, 12.5)
        b = rng.uniform(-0.9, 0.9) if L.plan else rng.uniform(-0.55, 0.95)
        for row in range(int(rng.integers(1, 4))):
            c = a
            for _ in range(int(rng.integers(2, 6))):
                wd = rng.uniform(0.02, 0.07)
                stroke(sten, L.px([(c, b - 0.022 * row), (c + wd, b - 0.022 * row)]), 1.3, 0.75)
                c += wd + 0.015
    lum = L.alb.mean(axis=-1)
    L.tint(sten, C_MARK, 0.55)
    L.tint(m * (lum > 0.46), C_MARK, 0.9)
    L.tint(m * (lum <= 0.46), C_MID, 0.75)
    L.tint(slime, C_SLIME, 0.95)
    L.rough -= 0.2 * slime


# ------------------------------------------------------------------ wear
def weather(L, G, rng):
    x, b = L.x, L.b
    big = 0.6 * G['along'][None, :] + 0.4 * fbm(L.shape, 320, rng, 4)
    fine = noise(L.shape, 2.5, 2.5, rng)
    st = np.clip(noise(L.shape, 6, 420, rng) * noise(L.shape, 24, 700, rng) * 3.2 - 0.95, 0.0, 1.0)
    dirt = np.clip(fbm(L.shape, 150, rng, 3) * 2.2 - 0.75, 0.0, 1.0)
    painted = 1.0 - L.metal
    shade = 1.0 + painted * (0.13 * (big - 0.5) + 0.035 * (fine - 0.5))
    # repainted panels: rectangles a shade off
    for _ in range(28):
        a0, w = rng.uniform(1.0, 13.3), rng.uniform(0.15, 0.7)
        if L.plan:
            b0, h = rng.uniform(-4.4, 4.4), rng.uniform(0.12, 0.6)
        else:
            b0, h = rng.uniform(-0.6, 3.2), rng.uniform(0.1, 0.4)
        ca, cb = int((a0 - X0) * D), int((a0 + w - X0) * D)
        ra, rb = int((b0 - L.b0) * D), int((b0 + h - L.b0) * D)
        if 0 <= ra < rb <= L.n:
            shade[ra:rb, ca:cb] *= 1.0 + rng.uniform(-0.045, 0.04)
    L.alb *= shade[..., None]
    k = 0.16
    grime = np.array((0.55, 0.55, 0.55), np.float32)
    L.alb *= 1.0 - (k * (0.6 * st + 0.5 * dirt) * painted)[..., None] * (1.0 - grime)
    if L.region == 'B':                                   # the oily centre of the belly
        oil = sstep(0.42 - np.abs(b), 0.3) * sstep(x - 5.0, 1.0) * (0.35 + 0.4 * st + 0.4 * dirt) * painted
        L.alb *= 1.0 - 0.30 * oil[..., None] * np.array((0.85, 0.95, 1.1), np.float32)
        L.rough -= 0.12 * oil
    if L.region == 'T':                                   # where the ground crew walk: strakes and wing roots
        walk = sstep(np.abs(b) - 0.55, 0.2) * sstep(1.55 - np.abs(b), 0.3) * sstep(x - 4.4, 0.6) * sstep(9.6 - x, 0.8)
        scuff = np.clip(fbm(L.shape, 40, rng, 3) * 2.4 - 0.9, 0.0, 1.0) * walk * painted
        L.alb *= 1.0 - 0.11 * scuff[..., None]
        L.rough += 0.08 * scuff
    if L.region == 'L':                                   # the gun: its port and the soot behind it
        port = L.mask()
        L.poly(port, [(4.86 + 0.20 * math.cos(t), 0.455 + 0.045 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 28, endpoint=False)])
        soot = np.exp(-(np.clip(x - 4.9, 0.0, None) / 1.1) ** 2 - ((b - 0.47 - 0.02 * (x - 4.9)) / 0.075) ** 2) * (x > 4.75)
        L.alb *= 1.0 - (0.45 * soot * (0.6 + 0.4 * st))[..., None]
        L.tint(port, (0.03, 0.03, 0.03), 1.0)
        L.height -= 2.0 * port
    # fluid that has leaked from a drain or a seam and been blown back along the skin
    if L.region != 'T':
        for _ in range(9):
            a0 = rng.uniform(4.8, 12.3)
            b0 = rng.uniform(-0.45, 0.45) if L.plan else rng.uniform(-0.40, 0.10)
            length, wide, dark = rng.uniform(0.4, 1.4), rng.uniform(0.010, 0.025), rng.uniform(0.08, 0.20)
            ca, cb = int((a0 - X0) * D), int((a0 + 3.0 * length - X0) * D)
            ra, rb = max(int((b0 - 4 * wide - L.b0) * D), 0), min(int((b0 + 4 * wide - L.b0) * D), L.n)
            xx, bb = x[:, ca:cb] - a0, b[ra:rb] - b0
            smear = np.exp(-(bb / (wide * (1.0 + 0.8 * xx / length))) ** 2) * np.exp(-xx / length) * np.clip(xx / 0.03, 0.0, 1.0)
            smear = smear * dark * painted[ra:rb, ca:cb]
            L.alb[ra:rb, ca:cb] *= 1.0 - smear[..., None] * np.array((0.85, 0.95, 1.1), np.float32)
            L.rough[ra:rb, ca:cb] -= 0.25 * smear
    # heat on the paint ahead of the bare metal, wear on the leading edges
    hot = sstep(x - 12.1, 0.7) * painted * (np.abs(b) < 0.8 if L.plan else b < 1.0)
    L.alb *= 1.0 - 0.14 * hot[..., None]
    if L.plan:
        edge = L.mask()
        for s in (-1.0, 1.0):
            stroke(edge, L.px([(G['x_le'](y) + 0.012, s * y) for y in (G['Y_LEF0'], G['HALF'])]), 5.0)
            stroke(edge, L.px([(G['tail_le'](y) + 0.012, s * y) for y in (G['Y_SHELF'], G['TAIL_Y1'])]), 4.0)
        edge *= 0.35 + 0.65 * noise(L.shape, 3, 30, rng)
        L.tint(edge, (0.66, 0.67, 0.67), 0.45)
        L.rough -= 0.1 * edge
    L.rough += painted * (0.10 * (big - 0.5) + 0.10 * st + 0.08 * dirt)


def paint(G):
    rng = np.random.default_rng(16)
    G = dict(G, along=fbm((1, W), 320, rng, 4)[0])
    alb = np.empty((H, W, 3), np.float32)
    alb[:] = C_MID
    rough = np.full((H, W), 0.55, np.float32)
    metal = np.zeros((H, W), np.float32)
    height = np.zeros((H, W), np.float32)
    for region in 'TBLR':
        L = Layer(region, alb, rough, metal, height)
        zones(L, G, rng)
        markings(L, G, rng)
        weather(L, G, rng)
        structure(L, G)
        apply_structure(L, rng)
        print('painted', region)
    gx = 0.5 * (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1))
    gy = 0.5 * (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0))
    nrm = np.stack([-gx * 1.4, -gy * 1.4, np.ones_like(gx)], axis=-1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    orm = np.stack([np.ones_like(rough), np.clip(rough, 0.05, 1.0), np.clip(metal, 0.0, 1.0)], axis=-1)
    return dict(albedo=np.clip(alb, 0.0, 1.0), orm=orm, normal=nrm * 0.5 + 0.5)
