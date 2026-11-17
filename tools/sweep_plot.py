"""Plot mean tree luminance vs camera distance for smooth vs hard LOD switching (see sweep.ps1).
Luminance is normalised by tree pixel count, so the curve shows appearance changes, not size."""
import sys

import numpy as np
from PIL import Image, ImageDraw

rows = [line.strip().split(",") for line in open(sys.argv[1]) if line.strip()]
out = sys.argv[2]
series = {}
for mode, d, lit, nrm in rows:
    n = np.asarray(Image.open(nrm).convert("RGB")).astype(float)
    im = np.asarray(Image.open(lit).convert("RGB")).astype(float)
    sky = np.abs(n - [89, 115, 153]).sum(-1) < 20
    ground = (n[..., 1] > 240) & (n[..., 0] < 140) & (n[..., 2] < 140)
    mask = ~sky & ~ground
    lum = (im[mask] @ [0.2126, 0.7152, 0.0722]).mean() if mask.any() else 0
    series.setdefault(mode, []).append((float(d), lum))

W, H, pad = 1100, 520, 60
img = Image.new("RGB", (W, H), (255, 255, 255))
dr = ImageDraw.Draw(img)
allv = [v for s in series.values() for _, v in s]
lo, hi = min(allv) - 2, max(allv) + 2
x = lambda d: pad + (125 - d) / (125 - 30) * (W - 2 * pad)
y = lambda v: H - pad - (v - lo) / (hi - lo) * (H - 2 * pad)
for b in (50, 100):
    dr.line([(x(b), pad), (x(b), H - pad)], fill=(200, 200, 200), width=2)
    dr.text((x(b) + 4, pad), f"{b} m", fill=(120, 120, 120))
colors = {"smooth": (20, 120, 220), "hard": (220, 60, 40)}
for mode, pts in series.items():
    pts.sort(reverse=True)
    dr.line([(x(d), y(v)) for d, v in pts], fill=colors[mode], width=3)
    jumps = [abs(pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1)]
    print(f"{mode:6s}: max step between 2.5 m samples = {max(jumps):.2f} lum, mean step = {np.mean(jumps):.2f}")
dr.text((pad, 15), "tree luminance vs camera distance (m), camera approaching; blue = smooth bands, red = hard switch", fill=(0, 0, 0))
dr.line([(pad, H - pad), (W - pad, H - pad)], fill=(0, 0, 0))
for d in range(30, 126, 10):
    dr.text((x(d) - 8, H - pad + 6), str(d), fill=(0, 0, 0))
img.save(out)
print("wrote", out)
