"""Print a structural summary of a USD file: prim types, mesh sizes, instancers, materials, textures."""
import sys
from collections import Counter
from pxr import Usd, UsdGeom, UsdShade, Sdf

path = sys.argv[1]
max_list = int(sys.argv[2]) if len(sys.argv) > 2 else 25
stage = Usd.Stage.Open(path)
print("file:", path)
print("upAxis:", UsdGeom.GetStageUpAxis(stage), "metersPerUnit:", UsdGeom.GetStageMetersPerUnit(stage))
print("defaultPrim:", stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else None)
print("layers:", [l.identifier for l in stage.GetLayerStack()])
print("used layers:", [l.identifier for l in stage.GetUsedLayers()])

types = Counter()
meshes = []
instancers = []
textures = set()
for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
    types[prim.GetTypeName()] += 1
    if prim.IsA(UsdGeom.Mesh):
        m = UsdGeom.Mesh(prim)
        pts = m.GetPointsAttr().Get()
        fvc = m.GetFaceVertexCountsAttr().Get()
        primvars = [p.GetPrimvarName() for p in UsdGeom.PrimvarsAPI(prim).GetPrimvars()]
        mat = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        meshes.append((str(prim.GetPath()), len(pts) if pts else 0, len(fvc) if fvc else 0, primvars,
                       str(mat.GetPath()) if mat else None, prim.IsInstanceProxy()))
    if prim.IsA(UsdGeom.PointInstancer):
        pi = UsdGeom.PointInstancer(prim)
        idx = pi.GetProtoIndicesAttr().Get()
        instancers.append((str(prim.GetPath()), len(idx) if idx else 0,
                           [str(t) for t in pi.GetPrototypesRel().GetTargets()]))
    if prim.IsA(UsdShade.Shader):
        sh = UsdShade.Shader(prim)
        for inp in sh.GetInputs():
            v = inp.Get()
            if isinstance(v, Sdf.AssetPath):
                textures.add((str(prim.GetPath()), inp.GetBaseName(), v.path, v.resolvedPath))
        print("shader", prim.GetPath(), sh.GetIdAttr().Get(),
              {i.GetBaseName(): (str(i.Get()) if i.Get() is not None else '<conn>') for i in sh.GetInputs()})

print("\nprim types:", dict(types))
print(f"\nmeshes: {len(meshes)} (showing {max_list})")
for m in meshes[:max_list]:
    print("  ", m)
print(f"\npoint instancers: {len(instancers)}")
for i in instancers[:max_list]:
    print("  ", i[0], "instances:", i[1], "protos:", i[2][:6], "..." if len(i[2]) > 6 else "")
print("\ntextures:")
for t in sorted(textures)[:max_list]:
    print("  ", t)
