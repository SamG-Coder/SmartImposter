"""Mean colour of tree pixels in impostor vs mesh renders from compare.ps1.
Tree pixels are found from the normals debug render (not sky colour, not the up-facing ground).
Usage: python treestats.py shots/_cmp_m2_l0.png shots/_cmp_m0_l0.png shots/_cmp_m0_l1000.png
"""
import sys

import numpy as np
from PIL import Image

normals, imp, mesh = (np.asarray(Image.open(p).convert("RGB")).astype(float) for p in sys.argv[1:4])
sky = np.abs(normals - [89, 115, 153]).sum(-1) < 20
ground = (normals[..., 1] > 240) & (normals[..., 0] < 140) & (normals[..., 2] < 140)
mask = ~sky & ~ground
lum = np.array([0.2126, 0.7152, 0.0722])
print(f"tree pixels: {int(mask.sum())}")
for name, im in (("impostor", imp), ("mesh", mesh)):
    px = im[mask]
    print(f"{name:9s} mean rgb {px.mean(0).round(1)}  lum {(px @ lum).mean():.1f}  std {(px @ lum).std():.1f}")
