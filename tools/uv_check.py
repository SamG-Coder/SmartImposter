"""Look at the UV primvars of a Megaplant twig: are st/st2 per-leaf constants or texture UVs?"""
import sys

import numpy as np
from pxr import Usd, UsdGeom

stage = Usd.Stage.Open(sys.argv[1])
for prim in stage.Traverse():
    if not prim.IsA(UsdGeom.Mesh) or not prim.GetName().startswith("Twig1"):
        continue
    m = UsdGeom.Mesh(prim)
    pv = UsdGeom.PrimvarsAPI(prim)
    counts = np.array(m.GetFaceVertexCountsAttr().Get())
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    leaf_faces = None
    for s in UsdGeom.Subset.GetAllGeomSubsets(m):
        if "TwoSided" in s.GetPrim().GetName():
            leaf_faces = np.array(s.GetIndicesAttr().Get())
    for name in ["st", "st2", "st3", "st4"]:
        p = pv.GetPrimvar(name)
        if not p or p.Get() is None:
            print(name, "missing / empty")
            continue
        uv = np.array(p.Get())
        print(f"{name}: interp {p.GetInterpolation()}, shape {uv.shape}, min {uv.min(0).round(3)}, max {uv.max(0).round(3)}")
        if leaf_faces is not None and p.GetInterpolation() == "faceVarying":
            fv = uv[starts[leaf_faces]]
            print(f"   on leaf faces: min {fv.min(0).round(3)} max {fv.max(0).round(3)}, unique values {len(np.unique(fv.round(4), axis=0))}")
    break
