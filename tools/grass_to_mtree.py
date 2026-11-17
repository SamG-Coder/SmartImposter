"""Megascans 3D grass (Fab glTF download) -> SmartImposter species pack of grass *patches*.

A Megascans grass asset is a few small clumps (variants A/B/C, each with LOD0/LOD1 alpha cards and
an LOD2 camera-facing billboard). One clump is ~0.3-1 m across, far too small to instance across a
landscape. This tool builds patch variants instead: a few metres of randomly placed, rotated and
scaled clumps (instanced parts of the LOD0 meshes). The engine treats a patch like any other plant:
real alpha-tested cards near the camera, one baked parallax impostor (a single AABB) beyond, and
the ground texture past its draw distance.

    python tools/grass_to_mtree.py <Fab folder of the grass, e.g. .../FabLibrary/Grass_Clumps-d22b2fc3> <out prefix>
        [--patch 2.6] [--patches 3] [--density 1.0]

Writes <out>.mtree, <out>.json (kind "grass", alpha-tested) and composes the engine's foliage NT
texture (normal RG + translucency B) next to the textures in assets/grass/<name>/.
"""

import glob
import json
import os
import struct
import sys
import time
import zlib

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from usd_to_mtree import MAT_LEAF, leaf_variation, smooth_normals  # noqa: E402

COMPONENT_TYPES = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def accessor(g, binary, idx):
    a = g["accessors"][idx]
    bv = g["bufferViews"][a["bufferView"]]
    dtype = COMPONENT_TYPES[a["componentType"]]
    n = COMPONENTS[a["type"]]
    off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = bv.get("byteStride", 0)
    item = np.dtype(dtype).itemsize * n
    if stride and stride != item:
        raw = np.frombuffer(binary, dtype=np.uint8, count=stride * a["count"], offset=off).reshape(a["count"], stride)[:, :item]
        return np.frombuffer(raw.tobytes(), dtype=dtype).reshape(a["count"], n)
    return np.frombuffer(binary, dtype=dtype, count=a["count"] * n, offset=off).reshape(a["count"], n)


def quat_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def load_clumps(gltf_path):
    """LOD0 clump meshes in engine space (Y up, metres, base at y = 0)."""
    g = json.load(open(gltf_path))
    binary = open(os.path.join(os.path.dirname(gltf_path), g["buffers"][0]["uri"]), "rb").read()
    clumps = []
    for node in g["nodes"]:
        name = node.get("name", "")
        if "mesh" not in node or "_LOD" in name:
            continue
        m = np.eye(3)
        if "rotation" in node:
            m = quat_mat(node["rotation"])
        s = np.array(node.get("scale", [1, 1, 1]))
        t = np.array(node.get("translation", [0, 0, 0]))
        pos_all, uv_all, tri_all = [], [], []
        base = 0
        for prim in g["meshes"][node["mesh"]]["primitives"]:
            p = accessor(g, binary, prim["attributes"]["POSITION"]).astype(np.float64)
            uv = accessor(g, binary, prim["attributes"]["TEXCOORD_0"]).astype(np.float32)
            idx = accessor(g, binary, prim["indices"]).reshape(-1, 3).astype(np.int64)
            p = (p * s) @ m.T + t
            pos_all.append(p)
            uv_all.append(uv)
            tri_all.append(idx + base)
            base += len(p)
        pos = np.concatenate(pos_all)
        pos[:, 1] -= pos[:, 1].min()  # base on the ground
        pos[:, [0, 2]] -= (pos[:, [0, 2]].min(0) + pos[:, [0, 2]].max(0)) * 0.5
        tris = np.concatenate(tri_all)
        clumps.append((name, pos.astype(np.float32), np.concatenate(uv_all), tris))
    return g, clumps


def grass_normals(pos, tris):
    """Card normals bent towards up: grass blades light like a canopy, not like flat cards."""
    n = smooth_normals(pos, tris)
    n = np.where(n[:, 1:2] < 0, -n, n)  # both card sides face up-ish
    n = n * 0.45 + np.array([0.0, 1.0, 0.0]) * 0.55
    return (n / np.linalg.norm(n, axis=1, keepdims=True)).astype(np.float32)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(zip([a for a in sys.argv[1:] if a.startswith("--")], [None] * 99))
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a.startswith("--") and i + 1 < len(argv):
            opts[a] = argv[i + 1]
    args = [a for i, a in enumerate(argv) if not a.startswith("--") and not (i > 0 and argv[i - 1].startswith("--"))]
    src, out = args[0], args[1]
    patch = float(opts.get("--patch") or 2.6)
    n_patches = int(opts.get("--patches") or 3)
    density = float(opts.get("--density") or 1.0)
    t0 = time.time()

    gltf = sorted(glob.glob(os.path.join(src, "gltf", "*", "*", "standard", "*_nonUE.gltf")))[0]
    tex_dir = os.path.join(os.path.dirname(os.path.dirname(gltf)), "Textures")
    g, clumps = load_clumps(gltf)
    species = os.path.basename(os.path.normpath(src)).split("-")[0]  # e.g. Grass_Clumps
    print(f"{species}: {len(clumps)} clumps from {os.path.relpath(gltf, src)}")

    meshes = []
    sizes = []
    for name, pos, uv, tris in clumps:
        nrm = grass_normals(pos, tris)
        mat = (leaf_variation(tris, np.ones(len(tris), bool)) << 2) | MAT_LEAF
        meshes.append((name, pos, nrm, tris, mat.astype(np.uint8), uv))
        ext = pos.max(0) - pos.min(0)
        sizes.append((max(ext[0], ext[2]), ext[1]))
        print(f"  {name}: {len(tris)} tris, footprint {max(ext[0], ext[2]):.2f} m, height {ext[1]:.2f} m")

    # Patches: jittered grid of clumps inside a disc with a ragged edge, so overlapping patches
    # blend into a continuous sward.
    foot = float(np.mean([s[0] for s in sizes]))
    spacing = max(0.12, 0.55 * foot / np.sqrt(density))
    variants = []
    rng = np.random.default_rng(zlib.crc32(species.encode()))  # stable layout per species
    for v in range(n_patches):
        parts = []
        k = int(np.ceil(patch / spacing))
        for gz in range(k):
            for gx in range(k):
                x = (gx + 0.5 + rng.uniform(-0.45, 0.45)) * spacing - patch * 0.5
                z = (gz + 0.5 + rng.uniform(-0.45, 0.45)) * spacing - patch * 0.5
                r = np.hypot(x, z) / (patch * 0.5)
                if rng.uniform() > np.clip((1.15 - r) / 0.35, 0, 1):
                    continue
                c = int(rng.integers(0, len(meshes)))
                yaw = rng.uniform(0, 2 * np.pi)
                sc = rng.uniform(0.8, 1.25)
                cy, sy = np.cos(yaw) * sc, np.sin(yaw) * sc
                m = np.array([[cy, 0, sy, x], [0, sc, 0, 0], [-sy, 0, cy, z]], dtype=np.float32)
                parts.append((c, m))
        lo, hi = np.full(3, 1e9), np.full(3, -1e9)
        for c, m in parts:
            p = meshes[c][1] @ m[:, :3].T + m[:, 3]
            lo, hi = np.minimum(lo, p.min(0)), np.maximum(hi, p.max(0))
        variants.append((f"{species}_Patch_{chr(65 + v)}", lo.astype(np.float32), hi.astype(np.float32), parts))
        print(f"  patch {chr(65 + v)}: {len(parts)} clumps, {lo.round(2)} .. {hi.round(2)}")

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out + ".mtree", "wb") as f:
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
            w(mat.tobytes())
            w(b"\0" * ((4 - len(mat) % 4) % 4))
        w(struct.pack("<I", len(variants)))
        for name, lo, hi, parts in variants:
            nb = name.encode()
            w(struct.pack("<I", len(nb)) + nb)
            w(np.concatenate([lo, hi]).astype("<f4").tobytes())
            w(struct.pack("<I", len(parts)))
            for c, m in parts:
                w(struct.pack("<I", c) + m.astype("<f4").tobytes())

    # Textures: colour+opacity as the foliage CA; normal (RG) + translucency (ORT blue) as NT.
    def tex(suffix):
        hits = [p for p in glob.glob(os.path.join(tex_dir, "*.png")) if p.endswith(suffix) and "Billboard" not in p]
        return hits[0] if hits else None

    ca, nrm_png, ort = tex("_B-O.png"), tex("_N.png"), tex("_ORT.png")
    grass_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(out))), "grass", species)
    os.makedirs(grass_dir, exist_ok=True)
    nt_path = os.path.join(grass_dir, f"T_{species}_NT.png")
    n_img = np.asarray(Image.open(nrm_png).convert("RGB"))
    t_img = np.asarray(Image.open(ort).convert("RGB").resize(n_img.shape[1::-1]))
    nt = np.dstack([n_img[..., 0], n_img[..., 1], t_img[..., 2], np.full(n_img.shape[:2], 255, np.uint8)])
    Image.fromarray(nt, "RGBA").save(nt_path)

    params = {p["Name"]: p["Value"] for p in g["materials"][0].get("extras", {}).get("params", [])}
    tt = params.get("Translucency Tint", [0.35, 0.5, 0.3, 1])[:3]
    tt = (np.array(tt) / max(max(tt), 1e-6) * 0.5).round(4).tolist()  # hue only, 0.5 = neutral strength
    rel = lambda p: os.path.relpath(p, os.path.dirname(os.path.abspath(out))) if p else None  # noqa: E731
    desc = {
        "species": species,
        "kind": "grass",
        "alpha": True,
        "textures": {"foliage_ca": rel(ca), "foliage_nt": rel(nt_path), "bark_c": None, "bark_nah": None},
        "leaf_tint": [0.5, 0.5, 0.5],
        "bark_tint": [0.5, 0.5, 0.5],
        "translucency_tint": tt,
        "variants": [v[0] for v in variants],
        "height": [float(v[2][1] - v[1][1]) for v in variants],
        "clump_height": float(np.mean([s[1] for s in sizes])),
    }
    json.dump(desc, open(out + ".json", "w"), indent=1)
    print(f"wrote {out}.mtree/.json and {os.path.relpath(nt_path)} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
