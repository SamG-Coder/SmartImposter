"""Detailed per-mesh info for a Megaplant USD: normals, colors, subsets, bind transforms, instance stats, bounds."""
import sys
import numpy as np
from pxr import Usd, UsdGeom, UsdShade, UsdSkel

stage = Usd.Stage.Open(sys.argv[1])
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=False)
root = [p for p in stage.GetPseudoRoot().GetChildren()][0]
print("root", root.GetPath(), "bounds", cache.ComputeWorldBound(root).ComputeAlignedRange())

for prim in stage.Traverse():
    if not prim.IsA(UsdGeom.Mesh):
        continue
    m = UsdGeom.Mesh(prim)
    pv = UsdGeom.PrimvarsAPI(prim)
    print("\nMESH", prim.GetPath().name, "parent", prim.GetParent().GetPath())
    print("  subdiv:", m.GetSubdivisionSchemeAttr().Get(), "orientation:", m.GetOrientationAttr().Get(),
          "doubleSided:", m.GetDoubleSidedAttr().Get())
    n = m.GetNormalsAttr().Get()
    print("  normals:", None if n is None else (len(n), m.GetNormalsInterpolation()))
    fvc = np.array(m.GetFaceVertexCountsAttr().Get())
    print("  face sizes:", dict(zip(*np.unique(fvc, return_counts=True))))
    for name in ["displayColor", "displayOpacity", "st", "st2", "skel:geomBindTransform"]:
        p = pv.GetPrimvar(name)
        if not p:
            continue
        v = p.Get()
        arr = np.array(v) if v is not None else None
        info = (p.GetInterpolation(), None if arr is None else arr.shape)
        if name.startswith("display") and arr is not None:
            info += ("min", arr.min(axis=0).round(3).tolist(), "max", arr.max(axis=0).round(3).tolist())
        if name == "skel:geomBindTransform":
            info += (str(v),)
        print("  ", name, info)
    for s in UsdGeom.Subset.GetAllGeomSubsets(m):
        mat = UsdShade.MaterialBindingAPI(s.GetPrim()).ComputeBoundMaterial()[0]
        print("   subset", s.GetPrim().GetName(), s.GetElementTypeAttr().Get(), len(s.GetIndicesAttr().Get()),
              "->", mat.GetPath().name if mat else None)
    xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    print("  localToWorld identity:", xf == xf.GetIdentity() if hasattr(xf, 'GetIdentity') else str(xf))
    print("  bound:", cache.ComputeWorldBound(prim).ComputeAlignedRange())

for prim in stage.Traverse():
    if prim.IsA(UsdGeom.PointInstancer):
        pi = UsdGeom.PointInstancer(prim)
        pos = np.array(pi.GetPositionsAttr().Get())
        sc = pi.GetScalesAttr().Get()
        ori = pi.GetOrientationsAttr().Get()
        idx = np.array(pi.GetProtoIndicesAttr().Get())
        print("\nINSTANCER", prim.GetPath().name, "count", len(idx), "per-proto", np.bincount(idx).tolist())
        print("  pos min", pos.min(0).round(2).tolist(), "max", pos.max(0).round(2).tolist())
        print("  scales:", None if sc is None else (np.array(sc).min(0).round(3).tolist(), np.array(sc).max(0).round(3).tolist()))
        print("  orientations:", None if ori is None else len(ori))
        print("  instancer xform:", UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        for t in pi.GetPrototypesRel().GetTargets():
            pp = stage.GetPrimAtPath(t)
            print("  proto", t.name, "xform", UsdGeom.Xformable(pp).GetLocalTransformation(),
                  "arcs", [str(a.GetArcType()) for a in Usd.PrimCompositionQuery(pp).GetCompositionArcs()][:4])

for prim in stage.Traverse():
    if prim.IsA(UsdSkel.Skeleton):
        sk = UsdSkel.Skeleton(prim)
        print("\nSKEL", prim.GetPath(), "joints", len(sk.GetJointsAttr().Get() or []))
