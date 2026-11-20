"""Fit a growth recipe to a Megaplant with separate woody axes and instanced leafy shoots.

python tools/fit_tree.py assets/species/Tree_Black_Alder.mtree Tree_Black_Alder_01_A assets/procedural/alder.sgt
Requires NumPy, SciPy and Pillow. The scan and fitted recipe stay in the local assets directory.
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from PIL import Image


def read_tree(path, variant):
    with path.open("rb") as f:

        def u32():
            return struct.unpack("<I", f.read(4))[0]

        def string():
            return f.read(u32()).decode()

        if f.read(4) != b"MTRE" or u32() != 2:
            raise ValueError("expected a version 2 .mtree")
        meshes = []
        for _ in range(u32()):
            string()
            nv, nt = u32(), u32()
            pos = np.fromfile(f, "<f4", nv * 3).reshape(-1, 3)
            f.seek(nv * 20, 1)
            tri = np.fromfile(f, "<u4", nt * 3).reshape(-1, 3)
            mat = np.fromfile(f, "u1", (nt + 3) // 4 * 4)[:nt] & 3
            meshes.append((pos, tri, mat))
        for _ in range(u32()):
            name = string()
            bounds = np.fromfile(f, "<f4", 6)
            parts = [(u32(), np.fromfile(f, "<f4", 12).reshape(3, 4)) for _ in range(u32())]
            if name == variant:
                return meshes, parts, bounds[4] - bounds[1]
    raise ValueError(f"variant {variant!r} missing in {path}")


def woody_axes(pos, tri):
    # UV seams duplicate positions. Weld them before finding individual woody limbs.
    pts, inv = np.unique(np.round(pos, 4), axis=0, return_inverse=True)
    a, b, c = inv[tri].T
    graph = coo_matrix(
        (np.ones(3 * len(a), dtype=np.uint8), (np.r_[a, b, c], np.r_[b, c, a])), shape=(len(pts), len(pts))
    ).tocsr()
    _, labels = connected_components(graph, directed=False)
    order = np.argsort(labels[a], kind="stable")
    faces = np.column_stack([a, b, c])
    for group in np.split(order, np.flatnonzero(np.diff(labels[a][order])) + 1):
        group_faces = faces[group]
        vertices = pts[np.unique(group_faces)]
        if len(vertices) < 12:
            continue
        axis = np.linalg.eigh(np.cov(vertices.T))[1][:, -1]
        if axis[1] < 0:
            axis = -axis
        corners = pts[group_faces]
        along = corners @ axis
        lo, hi = along.min(), along.max()
        if hi - lo < 0.07:
            continue
        rings = []
        for level in np.linspace(lo + 0.003, hi - 0.003, max(3, int(np.ceil((hi - lo) / 0.60)) + 1)):
            crosses = (along.min(1) < level) & (level < along.max(1))
            t, d = corners[crosses], along[crosses] - level
            t2, d2 = np.roll(t, 1, axis=1), np.roll(d, 1, axis=1)
            edges = (d > 0) != (d2 > 0)
            # Each crossed triangle contributes a line segment to the section. Length
            # weighting keeps dense tessellation on one side from pulling the axis off centre.
            line = (t[edges] + (t2[edges] - t[edges]) * (d[edges] / (d[edges] - d2[edges]))[:, None]).reshape(
                -1, 2, 3
            )
            lengths = np.linalg.norm(line[:, 1] - line[:, 0], axis=1)
            if lengths.sum() < 0.001:
                continue
            mid = line.mean(axis=1)
            center = np.average(mid, axis=0, weights=lengths)
            radius = np.average(np.linalg.norm(mid - center, axis=1), weights=lengths)
            rings.append([*center, max(0.002, radius)])
        if len(rings) >= 2:
            rings = np.array(rings)
            # A slanted open end has only a sliver at the extreme cut. Continue the
            # neighbouring complete sections to the end instead of pinching a trunk to a point.
            if len(rings) >= 3:
                for end, inner, next_inner in [(0, 1, 2), (-1, -2, -3)]:
                    a, b, c = rings[end], rings[inner], rings[next_inner]
                    if b[3] > 0.075 and a[3] < 0.65 * b[3] and b[3] >= 0.82 * c[3]:
                        along = (a[:3] - b[:3]) @ axis
                        span = (b[:3] - c[:3]) @ axis
                        if abs(span) > 0.001:
                            a[:3] = b[:3] + (b[:3] - c[:3]) * (along / span)
                            a[3] = np.clip(b[3] + (b[3] - c[3]) * along / span, b[3] * 0.8, b[3] * 1.5)
            yield rings


def palette(path):
    colors = [[74, 92, 37], [83, 72, 53]]
    meta = path.with_suffix(".json")
    if meta.exists():
        material = json.loads(meta.read_text())
        for i, key in enumerate(("foliage_ca", "bark_c")):
            name = material.get("textures", {}).get(key)
            if name and (meta.parent / name).is_file():
                with Image.open(meta.parent / name) as image:
                    rgba = np.array(image.convert("RGBA").resize((192, 192), Image.Resampling.BOX))
                pixels = rgba[:, :, :3][rgba[:, :, 3] > 220] / 255.0
                if len(pixels):
                    mean = np.where(
                        pixels <= 0.04045, pixels / 12.92, ((pixels + 0.055) / 1.055) ** 2.4
                    ).mean(0)
                    colors[i] = (
                        np.clip(
                            np.where(mean <= 0.0031308, 12.92 * mean, 1.055 * mean ** (1 / 2.4) - 0.055)
                            * 255,
                            0,
                            255,
                        )
                        .round()
                        .astype(int)
                    )
    return bytes([int(c) for rgb in colors for c in rgb])


def fit(path, variant, output):
    meshes, parts, height = read_tree(path, variant)
    axes, shoots = [], []
    traits = {}
    for mi, xf in parts:
        pos, tri, mat = meshes[mi]
        if (mat == 1).mean() < 0.10:
            axes.extend(woody_axes(pos @ xf[:, :3].T + xf[:, 3], tri[mat == 0]))
            continue
        if mi not in traits:
            a, b, c = pos[tri[mat == 1]].transpose(1, 0, 2)
            area = np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() * 0.5
            traits[mi] = (pos[:, 1].max(), area)
        scale = np.linalg.norm(xf[:, :3], axis=0)
        length, area = traits[mi]
        shoots.append((xf[:, 3], xf[:, 1] / scale[1], length * scale[1], area * scale.mean() ** 2))
    if not axes or not shoots:
        raise ValueError("this fitter needs separate woody limbs and Y-up leafy shoot instances")
    starts = np.concatenate([p[:-1, :3] for p in axes])
    ends = np.concatenate([p[1:, :3] for p in axes])
    delta = ends - starts
    origins, directions, lengths, areas = zip(*shoots)
    origins, directions = np.array(origins), np.array(directions)
    # Nearest segment by projection, among the nearest section midpoints.
    near = cKDTree((starts + ends) / 2).query(origins, k=list(range(1, min(24, len(starts)) + 1)))[1]
    d = delta[near]
    t = np.clip(((origins[:, None] - starts[near]) * d).sum(2) / np.maximum((d * d).sum(2), 1e-12), 0, 1)
    distance = np.linalg.norm(starts[near] + t[:, :, None] * d - origins[:, None], axis=2)
    best = distance.argmin(1)
    row = np.arange(len(shoots))
    seg, t = near[row, best], t[row, best]
    root_error = distance[row, best]
    out = bytearray(b"SGT2")
    name = variant.encode()
    out.extend(struct.pack("<H", len(name)) + name)
    out.extend(palette(path))
    out.extend(struct.pack("<fffI", height, np.mean(lengths), np.mean(areas), len(axes)))
    k = 0
    for p in axes:
        out.extend(struct.pack("<H", len(p)))
        for x, y, z, r in p:
            out.extend(struct.pack("<hhhH", round(x * 100), round(y * 100), round(z * 100), round(r * 1000)))
        for _ in p[1:]:
            members = seg == k
            count = members.sum()
            if count:
                # A growth interval and a mean direction per occupied section. Its vector
                # length measures agreement between directions and controls the growth cone.
                ts = t[members]
                direction = (directions[members].mean(0) * 127).round().astype(int)
                out.extend(
                    struct.pack("<HBBbbb", count, round(ts.min() * 255), round(ts.max() * 255), *direction)
                )
            else:
                out.extend(struct.pack("<H", 0))
            k += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(out)
    print(
        f"{variant}: {len(axes)} woody axes, {len(starts)} sections, {len(shoots)} shoots, "
        f"shoot length {np.mean(lengths):.3f} m, leaf area {np.mean(areas):.3f} m2, "
        f"p95 attachment residual {np.percentile(root_error, 95):.3f} m, {len(out):,} recipe bytes"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan", type=Path)
    parser.add_argument("variant")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    fit(args.scan, args.variant, args.output)
