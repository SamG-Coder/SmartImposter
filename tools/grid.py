"""Assemble impostor/mesh screenshot pairs into a labelled grid.
Usage: python grid.py "a.png;b.png;..." "0,3,2,4" out.png
"""
import sys

from PIL import Image, ImageDraw

files = sys.argv[1].split(";")
modes = sys.argv[2].split(",")
out = sys.argv[3]
names = {"0": "lit", "1": "LOD view", "2": "normals", "3": "albedo", "4": "AO", "5": "sun visibility"}
ims = [Image.open(f).convert("RGB") for f in files]
w, h = ims[0].size
w2, h2 = w // 2, h // 2
grid = Image.new("RGB", (w2 * 2, h2 * len(modes)))
d = ImageDraw.Draw(grid)
for i, im in enumerate(ims):
    r, c = divmod(i, 2)
    grid.paste(im.resize((w2, h2), Image.LANCZOS), (c * w2, r * h2))
    label = f"{names.get(modes[r], modes[r])} - {'impostor' if c == 0 else 'mesh'}"
    d.rectangle((c * w2 + 6, r * h2 + 6, c * w2 + 10 + 7 * len(label), r * h2 + 22), fill=(0, 0, 0))
    d.text((c * w2 + 8, r * h2 + 8), label, fill=(255, 255, 255))
grid.save(out)
print("wrote", out)
