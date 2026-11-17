"""Inspect leaf geometry in an .mtree: leaf sizes, shared vertices, normal consistency, UVs in USD."""
import struct
import sys

import numpy as np

path = sys.argv[1]
d = open(path, "rb").read()
p = 4
ver, nm = struct.unpack_from("<II", d, p); p += 8
for mi in range(nm):
    (ln,) = struct.unpack_from("<I", d, p); p += 4
    name = d[p:p + ln].decode(); p += ln
    v, t = struct.unpack_from("<II", d, p); p += 8
    pos = np.frombuffer(d, "<f4", v * 3, p).reshape(-1, 3); p += v * 12
    nrm = np.frombuffer(d, "<f4", v * 3, p).reshape(-1, 3); p += v * 12
    idx = np.frombuffer(d, "<u4", t * 3, p).reshape(-1, 3); p += t * 12
    mat = np.frombuffer(d, "u1", t, p); p += (t + 3) // 4 * 4
    if mi > 2:
        continue
    leaf = idx[mat == 1]
    a, b, c = pos[leaf[:, 0]], pos[leaf[:, 1]], pos[leaf[:, 2]]
    fn = np.cross(b - a, c - a)
    area = np.linalg.norm(fn, axis=1) * 0.5
    fn = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    vn = (nrm[leaf[:, 0]] + nrm[leaf[:, 1]] + nrm[leaf[:, 2]]) / 3
    vn = vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    agree = (fn * vn).sum(1)
    used = np.unique(leaf)
    shared_with_bark = np.intersect1d(used, np.unique(idx[mat == 0])).size
    print(f"{name}: leaf tris {len(leaf)}, leaf verts {used.size}, verts shared with bark {shared_with_bark}")
    print(f"  leaf tri area: median {np.median(area)*1e4:.2f} cm^2, total {area.sum():.3f} m^2")
    print(f"  smooth-vs-face normal dot: mean {agree.mean():.2f}, <0.5: {(agree < 0.5).mean():.1%}, <0: {(agree < 0).mean():.1%}")
    # duplicated (double-sided) sheets: same triangle centroid appearing twice
    cen = np.round((a + b + c) / 3, 5)
    _, counts = np.unique(cen, axis=0, return_counts=True)
    print(f"  coincident duplicate leaf triangles: {(counts > 1).sum()} of {len(cen)}")
