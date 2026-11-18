"""Rasterise RESOLVE Ecoregions 2017 (846 ecoregions, 14 biomes) into the globe's biome grid.

    python tools/bake_biomes.py            # assets/earth/ecoregions -> eco_grid.bin + eco_table.json

Output (next to the shapefile, read by src/globe/biomes.rs):
  eco_grid.bin    "SIECO1\\0\\0", u32 width, u32 height, then zlib(u16 ECO_ID per cell), row 0 at 90 N,
                  column 0 at 180 W, CELLS per degree (0 = no ecoregion: sea, ice-free nothing)
  eco_table.json  {ECO_ID: [BIOME_NUM, REALM, ECO_NAME, BIOME_NAME]}

Polygons are filled largest first so smaller ecoregions inside another's holes land on top.
No dependencies beyond Pillow (the .shp / .dbf readers are below).
"""

import json
import os
import struct
import sys
import zlib

from PIL import Image, ImageDraw

CELLS = 30  # per degree: ~3.7 km at the equator
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "earth", "ecoregions")


def read_dbf(path):
    with open(path, "rb") as f:
        head = f.read(32)
        n = struct.unpack("<I", head[4:8])[0]
        hlen, rlen = struct.unpack("<HH", head[8:12])
        fields = []
        while True:
            d = f.read(32)
            if d[0] == 0x0D:
                break
            fields.append((d[:11].split(b"\0")[0].decode(), d[16]))
        f.seek(hlen)
        rows = []
        for _ in range(n):
            r = f.read(rlen)
            pos, row = 1, {}
            for name, size in fields:
                row[name] = r[pos:pos + size].decode("utf-8", "replace").strip()
                pos += size
            rows.append(row)
    return rows


def read_shp(path):
    """Polygon records: list of rings ([(lon, lat), ...]) per record."""
    out = []
    with open(path, "rb") as f:
        f.seek(100)
        while True:
            h = f.read(8)
            if len(h) < 8:
                break
            length = struct.unpack(">i", h[4:8])[0] * 2
            rec = f.read(length)
            shape = struct.unpack("<i", rec[:4])[0]
            if shape == 0:
                out.append([])
                continue
            if shape not in (5, 15, 25):
                sys.exit(f"unsupported shape type {shape}")
            nparts, npts = struct.unpack("<ii", rec[36:44])
            parts = list(struct.unpack(f"<{nparts}i", rec[44:44 + 4 * nparts])) + [npts]
            base = 44 + 4 * nparts
            pts = struct.unpack(f"<{2 * npts}d", rec[base:base + 16 * npts])
            rings = []
            for a, b in zip(parts[:-1], parts[1:]):
                rings.append([(pts[2 * i], pts[2 * i + 1]) for i in range(a, b)])
            out.append(rings)
    return out


def ring_area(r):
    return abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(r, r[1:] + r[:1]))) * 0.5


def main():
    rows = read_dbf(os.path.join(DIR, "Ecoregions2017.dbf"))
    shapes = read_shp(os.path.join(DIR, "Ecoregions2017.shp"))
    assert len(rows) == len(shapes), (len(rows), len(shapes))
    w, h = 360 * CELLS, 180 * CELLS
    img = Image.new("I", (w, h), 0)
    draw = ImageDraw.Draw(img)
    table = {}
    rings = []
    for row, shape in zip(rows, shapes):
        eco = int(float(row["ECO_ID"] or 0))
        if eco <= 0:
            continue
        table[eco] = [int(float(row["BIOME_NUM"] or 0)), row["REALM"], row["ECO_NAME"], row["BIOME_NAME"]]
        for r in shape:
            rings.append((ring_area(r), eco, r))
    rings.sort(key=lambda x: -x[0])
    for _, eco, r in rings:
        pts = [((lon + 180.0) * CELLS, (90.0 - lat) * CELLS) for lon, lat in r]
        if len(pts) >= 3:
            draw.polygon(pts, fill=eco)
    data = img.tobytes()  # int32 little-endian
    u16 = bytearray(w * h * 2)
    import array

    a = array.array("i")
    a.frombytes(data)
    b = array.array("H", (min(max(v, 0), 65535) for v in a))
    u16 = b.tobytes()
    with open(os.path.join(DIR, "eco_grid.bin"), "wb") as f:
        f.write(b"SIECO1\0\0" + struct.pack("<II", w, h) + zlib.compress(u16, 6))
    with open(os.path.join(DIR, "eco_table.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in sorted(table.items())}, f, ensure_ascii=False, indent=0)
    filled = sum(1 for v in b if v)
    print(f"{len(table)} ecoregions, {len(rings)} rings -> {w}x{h} grid, {100.0 * filled / (w * h):.1f}% of cells in an ecoregion")


if __name__ == "__main__":
    main()
