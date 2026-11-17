"""Run inside Unreal (UnrealEditor-Cmd -run=pythonscript): export the Megaplant's textures as image
files and dump its material instance parameters to JSON."""
import json
import os

import unreal

SRC = os.environ.get("SI_MEGAPLANT", "/Game/Megaplant_Library/Tree_Black_Alder")
# Default: <repo>/assets/textures (this script lives in <repo>/tools). Override with SI_TEXTURE_OUT.
try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _here = os.getcwd()
OUT = os.environ.get("SI_TEXTURE_OUT", os.path.join(_here, "..", "assets", "textures"))
os.makedirs(OUT, exist_ok=True)

reg = unreal.AssetRegistryHelpers.get_asset_registry()
assets = reg.get_assets_by_path(SRC, recursive=True)
info = {"_assets": []}


def export(obj, filename):
    task = unreal.AssetExportTask()
    task.object = obj
    task.filename = filename
    task.automated = True
    task.replace_identical = True
    task.prompt = False
    return unreal.Exporter.run_asset_export_task(task)


for ad in assets:
    cls = str(ad.asset_class_path.asset_name)
    name = str(ad.asset_name)
    path = str(ad.package_name)
    info["_assets"].append([cls, path])
    if cls == "Texture2D":
        tex = unreal.load_asset(path)
        entry = {
            "size": [tex.blueprint_get_size_x(), tex.blueprint_get_size_y()],
            "srgb": bool(tex.srgb),
            "compression": str(tex.compression_settings),
        }
        for ext in ("png", "tga", "exr"):
            fn = os.path.join(OUT, f"{name}.{ext}")
            if export(tex, fn) and os.path.exists(fn):
                entry["file"] = fn
                break
        info[name] = entry
        unreal.log(f"exported {name}: {entry}")
    elif cls == "MaterialInstanceConstant":
        mi = unreal.load_asset(path)
        d = {"parent": mi.parent.get_path_name() if mi.parent else None, "scalars": {}, "vectors": {}, "textures": {}}
        for p in mi.scalar_parameter_values:
            d["scalars"][str(p.parameter_info.name)] = p.parameter_value
        for p in mi.vector_parameter_values:
            c = p.parameter_value
            d["vectors"][str(p.parameter_info.name)] = [c.r, c.g, c.b, c.a]
        for p in mi.texture_parameter_values:
            d["textures"][str(p.parameter_info.name)] = p.parameter_value.get_path_name() if p.parameter_value else None
        info[name] = d
        unreal.log(f"material {name}: {d}")

with open(os.path.join(OUT, "export_info.json"), "w") as f:
    json.dump(info, f, indent=1)
unreal.log("SMARTIMPOSTER EXPORT DONE")
