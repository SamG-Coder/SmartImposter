"""Convert Quixel Megaplant USD trees into a compact .mtree binary for the Rust engine.

Layout (little-endian):
  b"MTRE", u32 version(2)
  u32 mesh_count
    per mesh: u32 name_len, name, u32 vcount, u32 tcount,
              f32[3*v] positions, f32[3*v] normals, f32[2*v] uvs (image space, top-left origin),
              u32[3*t] indices, u8[t] material (padded to 4 bytes; low 2 bits material, high 6 bits leaf id)
  u32 tree_count
    per tree: u32 name_len, name, f32[6] bounds(min,max), u32 part_count,
              per part: u32 mesh_index, f32[12] transform (3x4 row-major, column-vector convention)

Materials: 0 = bark (Tileable), 1 = leaf (TwoSided).
Coordinates are meters, Y up (as authored). Winding is converted to counter-clockwise (right-handed).

Usage: python usd_to_mtree.py out.mtree tree_A.usd [tree_B.usd ...]
"""
import struct
import sys
import time

import numpy as np
from pxr import Gf, Usd, UsdGeom

MAT_BARK, MAT_LEAF = 0, 1


def triangulate(counts, indices):
    """Fan-triangulate polygons. Returns (tris[N,3], face_of_tri[N])."""
    counts = np.asarray(counts, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    tris, faces = [], []
    for n in np.unique(counts):
        if n < 3:
            continue
        sel = np.nonzero(counts == n)[0]
        s = starts[sel]
        for k in range(1, n - 1):
            tri = np.stack([indices[s], indices[s + k], indices[s + k + 1]], axis=1)
            tris.append(tri)
            faces.append(sel)
    tris = np.concatenate(tris)
    faces = np.concatenate(faces)
    order = np.argsort(faces, kind="stable")
    return tris[order], faces[order]


def smooth_normals(pos, tris):
    a, b, c = pos[tris[:, 0]], pos[tris[:, 1]], pos[tris[:, 2]]
    fn = np.cross(b - a, c - a)  # area weighted
    n = np.zeros_like(pos)
    for k in range(3):
        np.add.at(n, tris[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(ln > 1e-20, n / np.maximum(ln, 1e-20), np.array([0.0, 1.0, 0.0]))
    return n.astype(np.float32)


def corner_uvs(prim, fvi):
    """Texture coordinates per face corner from primvar 'st' (USD: origin bottom-left)."""
    pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
    if not pv:
        return np.zeros((len(fvi), 2))
    uv = np.array(pv.ComputeFlattened(), dtype=np.float64)
    if pv.GetInterpolation() == UsdGeom.Tokens.faceVarying:
        return uv
    if pv.GetInterpolation() in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying):
        return uv[fvi]
    return np.zeros((len(fvi), 2))


def mesh_data(prim):
    m = UsdGeom.Mesh(prim)
    pos_pts = np.array(m.GetPointsAttr().Get(), dtype=np.float64)
    fvi = np.array(m.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    # Triangulate face corners (not points) so per-corner UVs survive, then weld corners that share
    # both point and UV into one vertex.
    corner_tris, face = triangulate(m.GetFaceVertexCountsAttr().Get(), np.arange(len(fvi)))
    if m.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
        corner_tris = corner_tris[:, [0, 2, 1]]
    uvc = corner_uvs(prim, fvi)
    key = np.stack([fvi, np.round(uvc[:, 0] * 65536.0), np.round(uvc[:, 1] * 65536.0)], axis=1).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    vert_pt = fvi[first]
    uv = uvc[first]
    uv[:, 1] = 1.0 - uv[:, 1]  # USD st (bottom-left origin) -> image space (top-left origin)
    tris = inv[corner_tris]
    tris_pts = fvi[corner_tris]
    pos = pos_pts[vert_pt]
    nfaces = len(m.GetFaceVertexCountsAttr().Get())
    face_mat = np.zeros(nfaces, dtype=np.uint8)
    for s in UsdGeom.Subset.GetAllGeomSubsets(m):
        name = s.GetPrim().GetName().lower()
        idx = np.array(s.GetIndicesAttr().Get(), dtype=np.int64)
        if "twosided" in name or "leaf" in name:
            face_mat[idx] = MAT_LEAF
    mat = face_mat[face]
    # Drop degenerate triangles (they waste BLAS space and produce NaN normals).
    a, b, c = pos[tris[:, 0]], pos[tris[:, 1]], pos[tris[:, 2]]
    keep = np.linalg.norm(np.cross(b - a, c - a), axis=1) > 1e-14
    tris, tris_pts, mat = tris[keep], tris_pts[keep], mat[keep]
    # Smooth normals per original point (so UV seams stay smooth), copied to the split vertices.
    nrm = smooth_normals(pos_pts, tris_pts)[vert_pt]
    # Material byte: low 2 bits material, high 6 bits a random value per individual leaf
    # (connected component of leaf triangles) for per-leaf colour variation.
    leaf_rand = leaf_variation(tris_pts, mat == MAT_LEAF)
    mat = (mat & 3) | (leaf_rand << 2)
    return pos.astype(np.float32), nrm, tris.astype(np.uint32), mat.astype(np.uint8), uv.astype(np.float32)


def leaf_variation(tris, is_leaf):
    """Random 0..63 per connected component of leaf triangles (one per leaf), 0 for bark."""
    out = np.zeros(len(tris), dtype=np.uint8)
    lt = tris[is_leaf]
    if len(lt) == 0:
        return out
    parent = {}

    def find(x):
        root = x
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(x, x) != root:
            parent[x], x = root, parent[x]
        return root

    for a, b, c in lt.tolist():
        ra, rb, rc = find(a), find(b), find(c)
        parent[rb] = ra
        parent[find(rc)] = ra
    roots = np.array([find(a) for a in lt[:, 0].tolist()], dtype=np.int64)
    _, comp = np.unique(roots, return_inverse=True)
    rng = np.random.default_rng(1234)
    rand = rng.integers(0, 64, comp.max() + 1, dtype=np.uint8)
    out[np.nonzero(is_leaf)[0]] = rand[comp]
    print(f"    {comp.max() + 1} individual leaves", flush=True)
    return out


def to_3x4(m):
    """USD row-vector Gf.Matrix4d -> 3x4 row-major for column vectors (VkTransformMatrixKHR)."""
    a = np.array(m, dtype=np.float64).reshape(4, 4).T  # now column-vector convention
    return a[:3, :].astype(np.float32)


def main():
    out_path, inputs = sys.argv[1], sys.argv[2:]
    meshes = []  # (name, pos, nrm, tris, mat)
    mesh_key = {}  # (name, vcount) -> index
    trees = []
    t_start = time.time()

    def get_mesh(prim):
        name = prim.GetName()
        m = UsdGeom.Mesh(prim)
        key = (name, len(m.GetPointsAttr().Get()))
        if key not in mesh_key:
            data = mesh_data(prim)
            mesh_key[key] = len(meshes)
            meshes.append((name, *data))
            print(f"  mesh {name}: {len(data[0])} verts, {len(data[2])} tris, "
                  f"leaf tris {int(((data[3] & 3) == MAT_LEAF).sum())}", flush=True)
        return mesh_key[key]

    for path in inputs:
        stage = Usd.Stage.Open(path)
        assert UsdGeom.GetStageUpAxis(stage) == "Y", "expected Y-up"
        mpu = UsdGeom.GetStageMetersPerUnit(stage)
        scale = np.diag([mpu, mpu, mpu]).astype(np.float32)
        root = stage.GetPseudoRoot().GetChildren()[0]
        tc = Usd.TimeCode.Default()
        xcache = UsdGeom.XformCache(tc)
        parts = []
        print(f"{root.GetName()} ({path})", flush=True)
        instancer_protos = set()
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.PointInstancer):
                pi = UsdGeom.PointInstancer(prim)
                targets = pi.GetProtoIndicesAttr().Get()
                protos = [stage.GetPrimAtPath(t) for t in pi.GetPrototypesRel().GetTargets()]
                xforms = pi.ComputeInstanceTransformsAtTime(tc, tc)
                inst_world = xcache.GetLocalToWorldTransform(prim)
                proto_meshes = []
                for proto in protos:
                    instancer_protos.add(proto.GetPath())
                    pw_inv = xcache.GetLocalToWorldTransform(proto).GetInverse()
                    lst = []
                    for p in Usd.PrimRange(proto):
                        if p.IsA(UsdGeom.Mesh):
                            rel = xcache.GetLocalToWorldTransform(p) * pw_inv
                            lst.append((get_mesh(p), rel))
                    proto_meshes.append(lst)
                for pidx, xf in zip(targets, xforms):
                    for mi, rel in proto_meshes[pidx]:
                        world = rel * Gf.Matrix4d(xf) * inst_world
                        parts.append((mi, to_3x4(world)))
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh) and not any(prim.GetPath().HasPrefix(pp) for pp in instancer_protos):
                parts.append((get_mesh(prim), to_3x4(xcache.GetLocalToWorldTransform(prim))))
        if mpu != 1.0:
            parts = [(mi, (scale @ t[:, :3]).tolist() and np.hstack([scale @ t[:, :3], t[:, 3:] * mpu]))
                     for mi, t in parts]
        # Bounds from transformed mesh bounds (exact enough: transform the 8 corners).
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for mi, t in parts:
            p = meshes[mi][1]
            mn, mx = p.min(0), p.max(0)
            corners = np.array([[x, y, z] for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for z in (mn[2], mx[2])])
            w = corners @ t[:, :3].T + t[:, 3]
            lo, hi = np.minimum(lo, w.min(0)), np.maximum(hi, w.max(0))
        total = sum(len(meshes[mi][3]) for mi, _ in parts)
        print(f"  parts {len(parts)}, expanded tris {total/1e6:.1f}M, bounds {lo.round(2)} {hi.round(2)}", flush=True)
        trees.append((root.GetName(), lo, hi, parts))

    with open(out_path, "wb") as f:
        w = f.write
        w(b"MTRE")
        w(struct.pack("<II", 2, len(meshes)))
        for name, pos, nrm, tris, mat, uv in meshes:
            nb = name.encode()
            w(struct.pack("<I", len(nb)) + nb)
            w(struct.pack("<II", len(pos), len(tris)))
            w(pos.astype("<f4").tobytes())
            w(nrm.astype("<f4").tobytes())
            w(uv.astype("<f4").tobytes())
            w(tris.astype("<u4").tobytes())
            w(mat.astype(np.uint8).tobytes())
            w(b"\0" * ((4 - len(mat) % 4) % 4))
        w(struct.pack("<I", len(trees)))
        for name, lo, hi, parts in trees:
            nb = name.encode()
            w(struct.pack("<I", len(nb)) + nb)
            w(np.concatenate([lo, hi]).astype("<f4").tobytes())
            w(struct.pack("<I", len(parts)))
            for mi, t in parts:
                w(struct.pack("<I", mi) + np.asarray(t, dtype="<f4").tobytes())
    print(f"wrote {out_path}: {len(meshes)} meshes, {len(trees)} trees in {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
