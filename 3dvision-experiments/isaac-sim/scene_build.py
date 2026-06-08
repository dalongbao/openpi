"""scene_build.py — the SINGLE definition of the eval scene's manipulable objects (ball +
bowl) and the offline asset patching, shared by scene_preview.py and the eval scripts so the
preview and the eval render the IDENTICAL scene (no drift).

Division of labour:
  - scene_build.py  : ball + bowl geometry/colour/position + offline asset payload patching
  - scene_fidelity.py: visual realism (reddish floor, plywood tabletop, greenish walls, light)
  - ego_view.py     : the egocentric policy camera (+ optional arm hiding)

The ball is a 4-colour tessellated MESH sphere (per-face wedges); an implicit UsdGeom.Sphere
has no UVs, so a 4-colour texture would average to one muddy colour. Both ball and bowl carry
physics (collider; the ball also a rigid body) so the eval can grasp/drop — the preview just
ignores physics (pass physics=False to pin the ball for the static snapshot).
"""
import math
from pxr import UsdGeom, Gf, Sdf, UsdPhysics

# Episode initial state: bowl on the table, ball OUTSIDE it in front (robot then places it).
OBJECT_POS = (0.78, -0.48, 1.862)
BOWL_POS   = (0.88, -0.10, 1.807)

_ASSET_PATCHES = {
    "/World/fr3": "/workspace/assets/fr3_full/fr3.usd",
    "/World/SM_HeavyDutyPackingTable_C02_01": "/workspace/assets/table/SM_HeavyDutyPackingTable_C02_01.usd",
}
_DROP_PRIMS = ("/World/plate_small", "/World/SM_Crate_A07_Yellow_01_physics")

_BALL_COLS = [Gf.Vec3f(0.04, 0.31, 0.68), Gf.Vec3f(0.65, 0.67, 0.14),
              Gf.Vec3f(0.03, 0.40, 0.39), Gf.Vec3f(0.44, 0.08, 0.24)]  # blue, yellow, green, red
_BOWL_COL = Gf.Vec3f(0.227, 0.192, 0.396)   # dark dusty purple RGB(58,49,101)


def patch_offline_assets(stage):
    """Point fr3 + table payloads at local USD (offline node) and drop the old plate/crate."""
    for prim_path, local in _ASSET_PATCHES.items():
        p = stage.GetPrimAtPath(prim_path)
        if p.IsValid():
            p.GetPayloads().ClearPayloads(); p.GetPayloads().AddPayload(local)
    for old in _DROP_PRIMS:
        p = stage.GetPrimAtPath(old)
        if p.IsValid():
            p.SetActive(False)


def add_ball(stage, path, pos, r=0.055, mass=0.03, physics=True, nlon=24, nlat=12):
    """4-colour ball: tessellated sphere mesh, per-face colour in 4 longitude wedges."""
    pts = []
    for i in range(nlat + 1):
        th = math.pi * i / nlat
        for j in range(nlon):
            ph = 2 * math.pi * j / nlon
            pts.append(Gf.Vec3f(r * math.sin(th) * math.cos(ph),
                                r * math.sin(th) * math.sin(ph), r * math.cos(th)))
    counts, idx, facecols = [], [], []
    for i in range(nlat):
        for j in range(nlon):
            a = i * nlon + j; b = i * nlon + (j + 1) % nlon
            c = (i + 1) * nlon + (j + 1) % nlon; d = (i + 1) * nlon + j
            counts.append(4); idx.extend([a, b, c, d])
            ph = 2 * math.pi * (j + 0.5) / nlon
            facecols.append(_BALL_COLS[int(ph / (math.pi / 2)) % 4])
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(pts); m.CreateFaceVertexCountsAttr(counts); m.CreateFaceVertexIndicesAttr(idx)
    m.CreateSubdivisionSchemeAttr("none")
    pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.uniform)
    pv.Set(facecols)
    UsdGeom.Xformable(m).AddTranslateOp().Set(Gf.Vec3d(*pos))
    if physics:
        prim = m.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("boundingSphere")
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    return m


def add_bowl(stage, path, pos, Rb=0.14, Rt=0.19, H=0.10, wall=0.025, n=32, physics=True):
    """Wide, shallow open cup (opens +Z); watertight, exact-mesh collider so the ball drops in."""
    pts = []

    def ring(r, z):
        b = len(pts)
        for j in range(n):
            a = 2 * math.pi * j / n
            pts.append(Gf.Vec3f(r * math.cos(a), r * math.sin(a), z))
        return b

    ob = ring(Rb, 0.0); ot = ring(Rt, H); it = ring(Rt - wall, H); ib = ring(Rb - wall, wall)
    oc = len(pts); pts.append(Gf.Vec3f(0, 0, 0.0)); ic = len(pts); pts.append(Gf.Vec3f(0, 0, wall))
    counts, idx = [], []

    def quad(a, b, c, d): counts.append(4); idx.extend([a, b, c, d])
    def tri(a, b, c): counts.append(3); idx.extend([a, b, c])

    for j in range(n):
        k = (j + 1) % n
        quad(ob + j, ob + k, ot + k, ot + j); quad(ot + j, ot + k, it + k, it + j)
        quad(it + j, it + k, ib + k, ib + j); tri(ic, ib + k, ib + j); tri(oc, ob + j, ob + k)
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(pts); m.CreateFaceVertexCountsAttr(counts); m.CreateFaceVertexIndicesAttr(idx)
    m.CreateSubdivisionSchemeAttr("none"); m.CreateDoubleSidedAttr(True)
    UsdGeom.Xformable(m).AddTranslateOp().Set(Gf.Vec3d(*pos))
    m.CreateDisplayColorAttr([_BOWL_COL])
    if physics:
        prim = m.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
    return m


def build_objects(stage, object_path="/World/object", bowl_path="/World/bowl", physics=True):
    """Patch offline assets, then add the ball + bowl at the canonical positions."""
    patch_offline_assets(stage)
    add_ball(stage, object_path, OBJECT_POS, physics=physics)
    add_bowl(stage, bowl_path, BOWL_POS, physics=physics)
