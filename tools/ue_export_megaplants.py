"""Run inside Unreal (UnrealEditor-Cmd -run=pythonscript): export every Megaplant under
/Game/Megaplant_Library for SmartImposter.

Megaplants are Nanite assemblies: each variant (e.g. Tree_European_Beech_01_A) is a skeletal mesh
whose Nanite settings list part meshes (branches, twigs, leaves; possibly assemblies themselves)
and nodes placing them (component-space transform + bone). The stock USD exporter only writes a
mesh's own geometry, so this script walks the assembly tree, exports every unique mesh's own
geometry once, and writes the placement hierarchy to JSON for tools/megaplant_to_mtree.py.

Output (default <repo>/assets/megaplants, override with SI_MEGAPLANT_OUT):
  <Species>/assembly.json      {"variants": [...], "meshes": {path: {"usd", "parts", "nodes"}}}
  <Species>/meshes/*.usd       own geometry of every mesh in the hierarchy (Unreal space, cm)
  <Species>/textures/*.png     foliage CA/NT, bark C/NAH
  <Species>/materials.json     material instance parameters
Env: SI_ONLY=<substring> to export a subset of species, SI_SKIP_TEXTURES=1.
"""
import json
import os
import re

import unreal

LIB = "/Game/Megaplant_Library"
try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _here = os.getcwd()
OUT = os.environ.get("SI_MEGAPLANT_OUT", os.path.join(_here, "..", "assets", "megaplants"))
ONLY = os.environ.get("SI_ONLY", "")
SKIP_TEX = os.environ.get("SI_SKIP_TEXTURES", "") == "1"

reg = unreal.AssetRegistryHelpers.get_asset_registry()

F = r"(-?[\d.eE+-]+)"
RE_PART = re.compile(r'MeshObjectPath="([^"]+)"')
RE_NODE = re.compile(
    r"\(PartIndex=(\d+)(?:,TransformSpace=(\w+))?,Transform=\(Rotation=\(X=" + F + ",Y=" + F + ",Z=" + F + ",W=" + F +
    r"\),Translation=\(X=" + F + ",Y=" + F + ",Z=" + F + r"\),Scale3D=\(X=" + F + ",Y=" + F + ",Z=" + F +
    r"\)\)(?:,BoneInfluences=\(\(BoneIndex=(\d+))?"
)


def export(obj, filename):
    task = unreal.AssetExportTask()
    task.object = obj
    task.filename = filename
    task.automated = True
    task.replace_identical = True
    task.prompt = False
    return unreal.Exporter.run_asset_export_task(task) and os.path.exists(filename)


def assembly_of(mesh):
    try:
        txt = mesh.get_editor_property("nanite_settings").get_editor_property("nanite_assembly_data").export_text()
    except Exception:
        return [], []
    parts = [p.split(".")[0] for p in RE_PART.findall(txt)]
    nodes = []
    for m in RE_NODE.finditer(txt):
        g = m.groups()
        nodes.append({
            "part": int(g[0]),
            "space": g[1] or "default",
            "rot": [float(g[2]), float(g[3]), float(g[4]), float(g[5])],
            "trans": [float(g[6]), float(g[7]), float(g[8])],
            "scale": [float(g[9]), float(g[10]), float(g[11])],
            "bone": int(g[12]) if g[12] is not None else -1,
        })
    return parts, nodes


species_dirs = sorted({str(a.package_path) for a in reg.get_assets_by_path(LIB, recursive=True)})
species = sorted({p[len(LIB) + 1:].split("/")[0] for p in species_dirs if p.startswith(LIB + "/")})
summary = {}
for sp in species:
    if ONLY and ONLY not in sp:
        continue
    root = f"{LIB}/{sp}"
    out_sp = os.path.join(OUT, sp)
    os.makedirs(os.path.join(out_sp, "meshes"), exist_ok=True)
    variants, textures, materials = [], {}, {}
    for ad in reg.get_assets_by_path(root, recursive=True):
        cls = str(ad.asset_class_path.asset_name)
        name = str(ad.asset_name)
        path = str(ad.package_name)
        pkg_dir = str(ad.package_path)
        if pkg_dir.endswith("_01") and cls in ("SkeletalMesh", "StaticMesh") and len(name) > 2 and name[-2] == "_" and name[-1].isalpha():
            variants.append(path)
        elif cls == "Texture2D" and "Displacement" not in name and not SKIP_TEX:
            os.makedirs(os.path.join(out_sp, "textures"), exist_ok=True)
            tex = unreal.load_asset(path)
            fn = os.path.join(out_sp, "textures", f"{name}.png")
            ok = export(tex, fn)
            textures[name] = {"file": fn if ok else None, "size": [tex.blueprint_get_size_x(), tex.blueprint_get_size_y()]}
        elif cls == "MaterialInstanceConstant":
            mi = unreal.load_asset(path)
            d = {"scalars": {}, "vectors": {}, "textures": {}}
            for p in mi.scalar_parameter_values:
                d["scalars"][str(p.parameter_info.name)] = p.parameter_value
            for p in mi.vector_parameter_values:
                c = p.parameter_value
                d["vectors"][str(p.parameter_info.name)] = [c.r, c.g, c.b, c.a]
            for p in mi.texture_parameter_values:
                d["textures"][str(p.parameter_info.name)] = p.parameter_value.get_name() if p.parameter_value else None
            materials[name] = d

    # Walk every variant's assembly tree; export each unique mesh's own geometry once.
    meshes = {}
    stack = list(variants)
    while stack:
        path = stack.pop()
        if path in meshes:
            continue
        mesh = unreal.load_asset(path)
        if mesh is None:
            unreal.log_warning(f"[{sp}] missing {path}")
            meshes[path] = {"usd": None, "parts": [], "nodes": []}
            continue
        parts, nodes = assembly_of(mesh)
        fn = os.path.join(out_sp, "meshes", path.split("/")[-1] + ".usd")
        ok = export(mesh, fn)
        slots = []
        try:
            for m in mesh.get_editor_property("materials"):
                mi = m.get_editor_property("material_interface")
                slots.append(mi.get_name() if mi else None)
        except Exception:
            try:
                for m in mesh.get_editor_property("static_materials"):
                    mi = m.get_editor_property("material_interface")
                    slots.append(mi.get_name() if mi else None)
            except Exception:
                pass
        meshes[path] = {"usd": fn if ok else None, "class": type(mesh).__name__, "materials": slots, "parts": parts, "nodes": nodes}
        stack.extend(p for p in parts if p not in meshes)
    with open(os.path.join(out_sp, "assembly.json"), "w") as f:
        json.dump({"species": sp, "variants": sorted(variants), "meshes": meshes}, f, indent=1)
    with open(os.path.join(out_sp, "materials.json"), "w") as f:
        json.dump({"textures": textures, "materials": materials}, f, indent=1)
    total_nodes = sum(len(m["nodes"]) for m in meshes.values())
    summary[sp] = f"{len(variants)} variants, {len(meshes)} meshes, {total_nodes} nodes, {len(textures)} textures"
    unreal.log(f"[{sp}] {summary[sp]}")
unreal.log(f"SMARTIMPOSTER MEGAPLANT EXPORT DONE {summary}")
