"""Megascans 3D plants and scanned props (Fab glTF downloads) -> SmartImposter species packs.

Two kinds of Fab 3D asset:

* plant: alpha-tested cards in one atlas (palms, ferns, yucca, dead shrubs). Every LOD0 node
  (SM_<id>_VarA, VarB, ...) becomes a variant; the colour + opacity texture is the foliage CA, the
  normal (RG) and translucency (ORT blue) are composed into the foliage NT.

      python tools/fab_to_mtree.py plant <Fab folder> <out prefix> [--kind shrub|plant]

* solid: opaque scans with their own unique UVs (cacti, rocks, logs, snags), one mesh each. A group
  of them becomes ONE species (one texture set: species slots are limited): their textures are
  packed into a grid atlas (colour as the bark C; normal RG + occlusion as the bark NAH) and their
  UVs moved into their cells. Every scan is a variant.

      python tools/fab_to_mtree.py solid <out prefix> <Fab folder> [<Fab folder> ...] [--cell 1024]

Writes <out>.mtree and <out>.json (the catalog reads both) and the textures under
assets/props/<species>/. The normal maps already follow the engine's convention (checked against
the Megaplant bark: same sign of the height field's curl), so they are copied as they are.
"""

import glob
import json
import math
import os
import re
import struct
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grass_to_mtree import accessor, quat_mat  # noqa: E402
from usd_to_mtree import MAT_BARK, MAT_LEAF, leaf_variation  # noqa: E402

LOD = re.compile(r"_LOD\d+$")


def node_matrices(g):
    """World matrix (4x4) of every node, through the scene hierarchy."""
    local = []
    for n in g["nodes"]:
        if "matrix" in n:
            m = np.array(n["matrix"], dtype=np.float64).reshape(4, 4).T
        else:
            m = np.eye(4)
            r = quat_mat(n["rotation"]) if "rotation" in n else np.eye(3)
            s = np.array(n.get("scale", [1, 1, 1]), dtype=np.float64)
            m[:3, :3] = r * s
            m[:3, 3] = n.get("translation", [0, 0, 0])
        local.append(m)
    world = [None] * len(g["nodes"])

    def walk(i, parent):
        world[i] = parent @ local[i]
        for c in g["nodes"][i].get("children", []):
            walk(c, world[i])

    roots = g["scenes"][g.get("scene", 0)]["nodes"] if g.get("scenes") else range(len(g["nodes"]))
    for r in roots:
        walk(r, np.eye(4))
    return [w if w is not None else local[i] for i, w in enumerate(world)]


def load_meshes(gltf_path):
    """LOD0 meshes (name, pos, nrm, uv, tris) in engine space: Y up, metres, base at y = 0,
    centred on the trunk / footprint."""
    g = json.load(open(gltf_path))
    binary = open(os.path.join(os.path.dirname(gltf_path), g["buffers"][0]["uri"]), "rb").read()
    mats = node_matrices(g)
    out = []
    for ni, node in enumerate(g["nodes"]):
        name = node.get("name", "")
        if "mesh" not in node or LOD.search(name) or "billboard" in name.lower():
            continue
        m = mats[ni]
        nm = np.linalg.inv(m[:3, :3]).T
        P, N, U, T = [], [], [], []
        base = 0
        for prim in g["meshes"][node["mesh"]]["primitives"]:
            mat = g["materials"][prim["material"]].get("name", "") if "material" in prim else ""
            if "billboard" in mat.lower():
                continue
            a = prim["attributes"]
            p = accessor(g, binary, a["POSITION"]).astype(np.float64)
            p = p @ m[:3, :3].T + m[:3, 3]
            if "NORMAL" in a:
                n = accessor(g, binary, a["NORMAL"]).astype(np.float64) @ nm.T
                n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            else:
                n = np.tile([0.0, 1.0, 0.0], (len(p), 1))
            uv = accessor(g, binary, a["TEXCOORD_0"]).astype(np.float32)
            idx = accessor(g, binary, prim["indices"]).reshape(-1, 3).astype(np.int64)
            P.append(p)
            N.append(n)
            U.append(uv)
            T.append(idx + base)
            base += len(p)
        if not P:
            continue
        pos = np.concatenate(P)
        pos[:, 1] -= pos[:, 1].min()
        pos[:, [0, 2]] -= (pos[:, [0, 2]].min(0) + pos[:, [0, 2]].max(0)) * 0.5
        out.append((name, pos.astype(np.float32), np.concatenate(N).astype(np.float32), np.concatenate(U), np.concatenate(T)))
    return g, out


def write_mtree(path, meshes, variants):
    """meshes: (name, pos, nrm, tris, mat u8 per triangle, uv); variants: (name, lo, hi, [(mesh, 3x4)])."""
    with open(path, "wb") as f:
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
        w(struct.pack("<I", len(variants)))
        for name, lo, hi, parts in variants:
            nb = name.encode()
            w(struct.pack("<I", len(nb)) + nb)
            w(np.concatenate([lo, hi]).astype("<f4").tobytes())
            w(struct.pack("<I", len(parts)))
            for c, m in parts:
                w(struct.pack("<I", c) + m.astype("<f4").tobytes())


IDENTITY = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)


def textures_of(gltf):
    d = os.path.join(os.path.dirname(gltf), "Textures")
    if not os.path.isdir(d):
        d = os.path.join(os.path.dirname(os.path.dirname(gltf)), "Textures")

    def tex(suffix):
        hits = sorted(p for p in glob.glob(os.path.join(d, "*.png")) if p.endswith(suffix) and "Billboard" not in p)
        return hits[0] if hits else None

    return tex


def species_name(folder):
    return os.path.basename(os.path.normpath(folder)).split("-")[0]


def props_dir(out, species):
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(out))), "props", species)
    os.makedirs(d, exist_ok=True)
    return d


def rel(out, p):
    return os.path.relpath(p, os.path.dirname(os.path.abspath(out))) if p else None


def plant(src, out, kind, max_variants):
    t0 = time.time()
    gltf = sorted(glob.glob(os.path.join(src, "gltf", "*", "*", "standard", "*_nonUE.gltf")))[0]
    g, meshes_in = load_meshes(gltf)
    # the largest variants only: every variant costs an impostor atlas, the tiny ones add little
    vol = lambda m: float(np.prod(m[1].max(0) - m[1].min(0) + 0.05))  # noqa: E731
    meshes_in = sorted(sorted(meshes_in, key=vol, reverse=True)[:max_variants], key=lambda m: m[0])
    species = os.path.basename(out)  # (two Fab listings can share a name: the caller picks)
    tex = textures_of(gltf)
    meshes, variants = [], []
    for i, (name, pos, nrm, uv, tris) in enumerate(meshes_in):
        mat = (leaf_variation(tris, np.ones(len(tris), bool)) << 2) | MAT_LEAF
        meshes.append((name, pos, nrm, tris, mat.astype(np.uint8), uv))
        var = name.rsplit("_", 1)[-1] if "_Var" in name else chr(65 + i)
        variants.append((f"{species}_{var}", pos.min(0), pos.max(0), [(i, IDENTITY)]))
        ext = pos.max(0) - pos.min(0)
        print(f"  {name}: {len(tris)} tris, {ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f} m")
    write_mtree(out + ".mtree", meshes, variants)

    ca, nrm_png, ort = tex("_B-O.png"), tex("_N.png"), tex("_ORT.png")
    d = props_dir(out, species)
    nt_path = os.path.join(d, f"T_{species}_NT.png")
    n_img = np.asarray(Image.open(nrm_png).convert("RGB"))
    t = np.asarray(Image.open(ort).convert("RGB").resize(n_img.shape[1::-1]))[..., 2] if ort else np.full(n_img.shape[:2], 85, np.uint8)
    Image.fromarray(np.dstack([n_img[..., 0], n_img[..., 1], t, np.full(n_img.shape[:2], 255, np.uint8)]), "RGBA").save(nt_path)
    params = {p["Name"]: p["Value"] for p in g["materials"][0].get("extras", {}).get("params", [])}
    tt = params.get("Translucency Tint", [0.35, 0.5, 0.3, 1])[:3]
    tt = (np.array(tt) / max(max(tt), 1e-6) * 0.5).round(4).tolist()
    desc = {
        "species": species,
        "kind": kind,
        "alpha": True,
        "textures": {"foliage_ca": rel(out, ca), "foliage_nt": rel(out, nt_path), "bark_c": None, "bark_nah": None},
        "leaf_tint": [0.5, 0.5, 0.5],
        "bark_tint": [0.5, 0.5, 0.5],
        "translucency_tint": tt,
        "variants": [v[0] for v in variants],
        "height": [float(v[2][1] - v[1][1]) for v in variants],
    }
    json.dump(desc, open(out + ".json", "w"), indent=1)
    print(f"{species}: {len(variants)} variants -> {out}.mtree/.json ({time.time() - t0:.1f}s)")


def solid(out, folders, cell):
    t0 = time.time()
    species = os.path.basename(out)
    k = math.ceil(math.sqrt(len(folders)))
    size = k * cell
    colour = Image.new("RGB", (size, size), (90, 80, 70))
    nah = Image.new("RGB", (size, size), (128, 128, 255))
    meshes, variants = [], []
    for i, src in enumerate(folders):
        gltf = sorted(p for p in glob.glob(os.path.join(src, "gltf", "*", "*", "*.gltf")) if "nonUE" not in p)[0]
        _, parts = load_meshes(gltf)
        tex = textures_of(gltf)
        col, row = i % k, i // k
        b = Image.open(tex("_B.png")).convert("RGB").resize((cell, cell), Image.LANCZOS)
        n = np.asarray(Image.open(tex("_N.png")).convert("RGB").resize((cell, cell), Image.LANCZOS))
        orm = tex("_ORM.png")
        ao = np.asarray(Image.open(orm).convert("RGB").resize((cell, cell), Image.LANCZOS))[..., 0] if orm else np.full((cell, cell), 255, np.uint8)
        colour.paste(b, (col * cell, row * cell))
        nah.paste(Image.fromarray(np.dstack([n[..., 0], n[..., 1], ao]), "RGB"), (col * cell, row * cell))
        # one mesh per scan (all its LOD0 nodes merged), UVs moved into the cell
        pos = np.concatenate([p[1] for p in parts])
        nrm = np.concatenate([p[2] for p in parts])
        uv = np.concatenate([p[3] for p in parts])
        offs = np.cumsum([0] + [len(p[1]) for p in parts[:-1]])
        tris = np.concatenate([p[4] + o for p, o in zip(parts, offs)])
        pos[:, 1] -= pos[:, 1].min()
        pos[:, [0, 2]] -= (pos[:, [0, 2]].min(0) + pos[:, [0, 2]].max(0)) * 0.5
        if uv.min() < -1e-3 or uv.max() > 1.001:
            print(f"  warning: {species_name(src)} UVs {uv.min():.2f}..{uv.max():.2f} outside 0..1 (clamped)")
        uv = (np.clip(uv, 0.0, 1.0) + np.array([col, row], np.float32)) / k
        name = species_name(src)
        meshes.append((name, pos, nrm, tris, np.full(len(tris), MAT_BARK, np.uint8), uv.astype(np.float32)))
        variants.append((f"{species}_{name}", pos.min(0), pos.max(0), [(i, IDENTITY)]))
        ext = pos.max(0) - pos.min(0)
        print(f"  {name}: {len(tris)} tris, {ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f} m, cell {col},{row}")
    write_mtree(out + ".mtree", meshes, variants)
    d = props_dir(out, species)
    c_path, n_path = os.path.join(d, f"T_{species}_C.png"), os.path.join(d, f"T_{species}_NAH.png")
    colour.save(c_path)
    nah.save(n_path)
    desc = {
        "species": species,
        "kind": "rock",
        "alpha": False,
        "textures": {"foliage_ca": None, "foliage_nt": None, "bark_c": rel(out, c_path), "bark_nah": rel(out, n_path)},
        "leaf_tint": [0.5, 0.5, 0.5],
        "bark_tint": [0.5, 0.5, 0.5],
        "translucency_tint": [0.0, 0.0, 0.0],
        "variants": [v[0] for v in variants],
        "height": [float(v[2][1] - v[1][1]) for v in variants],
    }
    json.dump(desc, open(out + ".json", "w"), indent=1)
    print(f"{species}: {len(variants)} variants, {size}^2 atlas -> {out}.mtree/.json ({time.time() - t0:.1f}s)")


def main():
    argv = sys.argv[1:]
    opts = {a: argv[i + 1] for i, a in enumerate(argv) if a.startswith("--") and i + 1 < len(argv)}
    args = [a for i, a in enumerate(argv) if not a.startswith("--") and not (i > 0 and argv[i - 1].startswith("--"))]
    mode = args[0]
    if mode == "plant":
        plant(args[1], args[2], opts.get("--kind", "plant"), int(opts.get("--max-variants", 4)))
    elif mode == "solid":
        solid(args[1], args[2:], int(opts.get("--cell", 1024)))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
