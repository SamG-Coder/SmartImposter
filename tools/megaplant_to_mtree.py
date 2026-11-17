"""Convert a Megaplant species exported by tools/ue_export_megaplants.py into an .mtree pack.

  python megaplant_to_mtree.py assets/megaplants/Tree_European_Beech assets/species/Tree_European_Beech

Writes <out>.mtree (format v2, see usd_to_mtree.py) and <out>.json (kind, textures, tints).

Each variant is a Nanite assembly tree: mesh own geometry + nodes placing part meshes (which may be
assemblies themselves). The tree is flattened top-down: a subtree whose total triangle count is
below MERGE_LIMIT becomes ONE merged mesh (e.g. a twig with all its leaves), larger nodes are kept
as instances of their children. That keeps a near tree at hundreds to a few thousand TLAS
instances while unique BLAS memory stays bounded.

Coordinates: assembly node transforms are in Unreal space (left-handed, Z up, cm); the USD
exporter mirrors Y (USD space = Unreal with Y negated). Engine space is right-handed, Y up, metres.
"""
import json
import os
import struct
import sys
import time

import numpy as np
from pxr import Usd, UsdGeom, UsdShade

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from usd_to_mtree import MAT_BARK, MAT_LEAF, corner_uvs, leaf_variation, smooth_normals, triangulate  # noqa: E402

MERGE_LIMIT = 400_000
S4 = np.diag([1.0, -1.0, 1.0, 1.0])                         # Unreal <-> USD (mirror Y)
C = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float) * 0.01  # USD (Z up, cm) -> engine (Y up, m)
C4 = np.eye(4)
C4[:3, :3] = C
C4_INV = np.linalg.inv(C4)


def quat_mat(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def node_matrix(n):
    """Assembly node (Unreal FTransform: scale, then rotate, then translate) -> USD-space 4x4."""
    m = np.eye(4)
    m[:3, :3] = quat_mat(*n["rot"]) @ np.diag(n["scale"])
    m[:3, 3] = n["trans"]
    return S4 @ m @ S4


def is_bark(material_name):
    return material_name is not None and "bark" in material_name.lower()


class Geo:
    """Triangle soup with per-vertex position/normal/uv and per-triangle material (USD space, cm)."""

    def __init__(self, pos, nrm, uv, tris, mat):
        self.pos, self.nrm, self.uv, self.tris, self.mat = pos, nrm, uv, tris, mat

    @staticmethod
    def empty():
        return Geo(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 2)), np.zeros((0, 3), np.int64), np.zeros(0, np.uint8))

    def transformed(self, m):
        r = m[:3, :3]
        pos = self.pos @ r.T + m[:3, 3]
        nrm = self.nrm @ np.linalg.inv(r)  # inverse-transpose
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        tris = self.tris if np.linalg.det(r) > 0 else self.tris[:, [0, 2, 1]]
        return Geo(pos, nrm, self.uv, tris, self.mat)

    @staticmethod
    def concat(geos):
        geos = [g for g in geos if len(g.tris)]
        if not geos:
            return Geo.empty()
        off = np.cumsum([0] + [len(g.pos) for g in geos[:-1]])
        return Geo(
            np.concatenate([g.pos for g in geos]),
            np.concatenate([g.nrm for g in geos]),
            np.concatenate([g.uv for g in geos]),
            np.concatenate([g.tris + o for g, o in zip(geos, off)]),
            np.concatenate([g.mat for g in geos]),
        )


def load_own_geometry(usd_path, slot_names):
    if not usd_path or not os.path.exists(usd_path):
        return Geo.empty()
    stage = Usd.Stage.Open(usd_path)
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    geos = []
    for prim in meshes:
        m = UsdGeom.Mesh(prim)
        pts = np.array(m.GetPointsAttr().Get() or [], dtype=np.float64)
        counts = m.GetFaceVertexCountsAttr().Get()
        if len(pts) == 0 or not counts:
            continue
        fvi = np.array(m.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        ctris, face = triangulate(counts, np.arange(len(fvi)))
        if m.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
            ctris = ctris[:, [0, 2, 1]]
        uvc = corner_uvs(prim, fvi)
        key = np.stack([fvi, np.round(uvc[:, 0] * 65536.0), np.round(uvc[:, 1] * 65536.0)], axis=1).astype(np.int64)
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        inv = inv.reshape(-1)
        vert_pt = fvi[first]
        uv = uvc[first]
        uv[:, 1] = 1.0 - uv[:, 1]
        tris = inv[ctris]
        tris_pts = fvi[ctris]
        # Material per face: GeomSubsets bound to Unreal materials, else the mesh's first slot.
        face_mat = np.full(len(counts), MAT_BARK if is_bark(slot_names[0] if slot_names else None) else MAT_LEAF, np.uint8)
        for s in UsdGeom.Subset.GetAllGeomSubsets(m):
            mat = UsdShade.MaterialBindingAPI(s.GetPrim()).ComputeBoundMaterial()[0]
            names = [s.GetPrim().GetName()]
            if mat:
                names.append(mat.GetPrim().GetName())
                for a in mat.GetPrim().GetAttributes():
                    v = a.Get()
                    if isinstance(v, str):
                        names.append(v)
            idx = np.array(s.GetIndicesAttr().Get(), dtype=np.int64)
            face_mat[idx] = MAT_BARK if any(is_bark(n) for n in names) else MAT_LEAF
        mat = face_mat[face]
        a, b, c = pts[tris_pts[:, 0]], pts[tris_pts[:, 1]], pts[tris_pts[:, 2]]
        keep = np.linalg.norm(np.cross(b - a, c - a), axis=1) > 1e-10
        tris, tris_pts, mat = tris[keep], tris_pts[keep], mat[keep]
        nrm = smooth_normals(pts, tris_pts)[vert_pt].astype(np.float64)
        xf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64).T
        geos.append(Geo(pts[vert_pt], nrm, uv, tris, mat).transformed(xf))
    return Geo.concat(geos)


def main():
    src, out = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else MERGE_LIMIT
    t0 = time.time()
    asm = json.load(open(os.path.join(src, "assembly.json")))
    mats = json.load(open(os.path.join(src, "materials.json")))
    meshes = asm["meshes"]

    own = {}
    for path, info in meshes.items():
        own[path] = load_own_geometry(info.get("usd"), info.get("materials") or [])

    expanded = {}

    def tri_count(path):
        if path not in expanded:
            info = meshes[path]
            expanded[path] = len(own[path].tris) + sum(tri_count(info["parts"][n["part"]]) for n in info["nodes"])
        return expanded[path]

    merged_geo = {}

    def merged(path):
        if path not in merged_geo:
            info = meshes[path]
            parts = [own[path]] + [merged(info["parts"][n["part"]]).transformed(node_matrix(n)) for n in info["nodes"]]
            merged_geo[path] = Geo.concat(parts)
        return merged_geo[path]

    out_meshes = []   # (name, Geo in USD space)
    mesh_index = {}

    def mesh_id(key, name, geo_fn):
        if key not in mesh_index:
            geo = geo_fn()
            if len(geo.tris) == 0:
                mesh_index[key] = None
            else:
                mesh_index[key] = len(out_meshes)
                out_meshes.append((name, geo))
        return mesh_index[key]

    def emit(path, world, parts):
        info = meshes[path]
        short = path.split("/")[-1]
        if tri_count(path) <= limit:
            mid = mesh_id(("merged", path), short + "_merged", lambda: merged(path))
            if mid is not None:
                parts.append((mid, world))
            return
        mid = mesh_id(("own", path), short, lambda: own[path])
        if mid is not None:
            parts.append((mid, world))
        for n in info["nodes"]:
            emit(info["parts"][n["part"]], world @ node_matrix(n), parts)

    trees = []
    for v in asm["variants"]:
        parts = []
        emit(v, np.eye(4), parts)
        trees.append((v.split("/")[-1], parts))
        print(f"  variant {v.split('/')[-1]}: {len(parts)} instances, {tri_count(v) / 1e6:.1f}M triangles expanded", flush=True)

    # Engine space conversion + per-mesh attributes.
    packed = []
    for name, g in out_meshes:
        pos = g.pos @ C.T
        nrm = g.nrm @ (C / 0.01).T
        leaf = leaf_variation(g.tris, (g.mat & 3) == MAT_LEAF)
        mat = (g.mat & 3) | (leaf << 2)
        packed.append((name, pos.astype(np.float32), nrm.astype(np.float32), g.tris.astype(np.uint32), mat.astype(np.uint8), g.uv.astype(np.float32)))

    def to_engine(m):
        e = C4 @ m @ C4_INV
        return e[:3, :].astype(np.float32)

    tree_out = []
    for name, parts in trees:
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        eparts = []
        for mid, w in parts:
            t = to_engine(w)
            p = packed[mid][1]
            mn, mx = p.min(0), p.max(0)
            corners = np.array([[x, y, z] for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for z in (mn[2], mx[2])])
            wc = corners @ t[:, :3].T + t[:, 3]
            lo, hi = np.minimum(lo, wc.min(0)), np.maximum(hi, wc.max(0))
            eparts.append((mid, t))
        tree_out.append((name, lo, hi, eparts))
        print(f"  {name}: bounds {lo.round(2)} .. {hi.round(2)}")

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out + ".mtree", "wb") as f:
        w = f.write
        w(b"MTRE")
        w(struct.pack("<II", 2, len(packed)))
        for name, pos, nrm, tris, mat, uv in packed:
            nb = name.encode()
            w(struct.pack("<I", len(nb)) + nb)
            w(struct.pack("<II", len(pos), len(tris)))
            w(pos.astype("<f4").tobytes())
            w(nrm.astype("<f4").tobytes())
            w(uv.astype("<f4").tobytes())
            w(tris.astype("<u4").tobytes())
            w(mat.tobytes())
            w(b"\0" * ((4 - len(mat) % 4) % 4))
        w(struct.pack("<I", len(tree_out)))
        for name, lo, hi, parts in tree_out:
            nb = name.encode()
            w(struct.pack("<I", len(nb)) + nb)
            w(np.concatenate([lo, hi]).astype("<f4").tobytes())
            w(struct.pack("<I", len(parts)))
            for mid, t in parts:
                w(struct.pack("<I", mid) + np.asarray(t, dtype="<f4").tobytes())

    # Species description for the engine: kind, textures, Unreal material tints.
    sp = asm["species"]
    kind = "tree" if sp.startswith("Tree_") else ("shrub" if sp.startswith("Shrub_") else "plant")
    tex = {k: v["file"] for k, v in mats["textures"].items() if v.get("file")}

    def pick(suffix, foliage):
        for k, v in tex.items():
            if k.endswith(suffix) and (("Foliage" in k or "Bark" not in k) if foliage else "Bark" in k):
                return os.path.relpath(v, os.path.dirname(os.path.abspath(out)))
        return None

    def vec(mi_kw, key, default):
        for name, m in mats["materials"].items():
            if mi_kw in name and key in m["vectors"]:
                return m["vectors"][key][:3]
        return default

    desc = {
        "species": sp,
        "kind": kind,
        "textures": {"foliage_ca": pick("_CA", True), "foliage_nt": pick("_NT", True), "bark_c": pick("_C", False), "bark_nah": pick("_NAH", False)},
        "leaf_tint": vec("Foliage", "BaseColor Tint Leaves", vec("Foliage", "BaseColor Tint", vec("", "BaseColor Tint", [0.5, 0.5, 0.5]))),
        "bark_tint": vec("Bark", "BaseColor Tint", [0.5, 0.5, 0.5]),
        "translucency_tint": vec("", "Translucency Tint", [0.21, 0.27, 0.12]),
        "variants": [t[0] for t in tree_out],
        "height": [float(t[2][1] - t[1][1]) for t in tree_out],
    }
    json.dump(desc, open(out + ".json", "w"), indent=1)
    tris_total = sum(len(p[3]) for p in packed)
    print(f"wrote {out}.mtree: {len(packed)} unique meshes, {tris_total / 1e6:.2f}M unique triangles, "
          f"{len(tree_out)} variants ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
