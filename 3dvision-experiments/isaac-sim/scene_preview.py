"""
scene_preview.py — render ONLY the scene (no policy, no closed loop) so you can iterate
on scene_fidelity.py fast. Boots Isaac, opens the scene, builds the object + bowl,
applies scene_fidelity.apply_fidelity, places the policy camera EXACTLY as the eval does,
renders the 224x224 policy view + an HD view to PNG, and exits.

~3-4 min vs ~15 for a full eval (no policy load, no JAX, no 3000-step loop).

Workflow:
  1. edit scene_fidelity.py
  2. git push; on Euler: git pull; cp scene_fidelity.py scene_preview.py -> pi0_test/
  3. sbatch --partition=gpu.4h --time=00:10:00 --mem-per-cpu=8G --cpus-per-task=8 \
            --gpus=rtx_3090:1 submit.sh scene_preview.py
  4. scp ...:/cluster/scratch/$USER/pi0_test/preview_policy.png .   (and preview_hd.png)
  5. look, repeat

preview_policy.png is the exact 224x224 image pi0.5 receives — judge OOD-ness against
a real Aria frame. preview_hd.png is a 3rd-person view for overall scene sanity.
"""
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True, "livestream": 0, "width": 1280, "height": 720})

import math
import os
import sys
import numpy as np
import cv2

from omni.isaac.core import World
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.sensor import Camera
import omni.usd
from pxr import UsdGeom, Gf, UsdPhysics

sys.path.insert(0, "/workspace")
import scene_fidelity

USD_PATH   = "/workspace/kitchen_scene_1.usd"
POLICY_RES = (224, 224)
HD_RES     = (1280, 720)

# Same object/bowl positions + camera as eval_script_object_in_bowl.py so the preview
# matches what the policy will actually see. Keep these in sync if the eval changes.
# Preview-only placement: bowl near the camera look-target with the ball sitting in it,
# so both read large/centred like the real Aria frame (the eval uses its own positions).
OBJECT_POS = (0.85, -0.13, 1.86)
BOWL_POS   = (0.85, -0.15, 1.807)
CAM_POS    = Gf.Vec3d(-0.2, -0.8, 2.6)
CAM_TARGET = Gf.Vec3d(0.8, -0.15, 1.82)
CAM_HFOV   = 90.0

open_stage(usd_path=USD_PATH)
stage = omni.usd.get_context().get_stage()

# Patch fr3 + table payloads to local assets (offline); drop the old plate/crate.
for prim_path, local in {
    "/World/fr3": "/workspace/assets/fr3_full/fr3.usd",
    "/World/SM_HeavyDutyPackingTable_C02_01": "/workspace/assets/table/SM_HeavyDutyPackingTable_C02_01.usd",
}.items():
    p = stage.GetPrimAtPath(prim_path)
    if p.IsValid():
        p.GetPayloads().ClearPayloads()
        p.GetPayloads().AddPayload(local)
for old in ("/World/plate_small", "/World/SM_Crate_A07_Yellow_01_physics"):
    p = stage.GetPrimAtPath(old)
    if p.IsValid():
        p.SetActive(False)


def add_sphere(path, pos, r=0.04):
    s = UsdGeom.Sphere.Define(stage, path)
    s.CreateRadiusAttr(r)
    UsdGeom.Xformable(s).AddTranslateOp().Set(Gf.Vec3d(*pos))
    s.CreateDisplayColorAttr([Gf.Vec3f(0.80, 0.15, 0.15)])


def add_bowl(path, pos, Rb=0.10, Rt=0.18, H=0.13, wall=0.025, n=32):
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
    m.CreateDisplayColorAttr([Gf.Vec3f(0.227, 0.192, 0.396)])   # dark dusty purple, RGB(58,49,101)


add_sphere("/World/object", OBJECT_POS)
add_bowl("/World/bowl", BOWL_POS)

# The visual fidelity we iterate on:
scene_fidelity.apply_fidelity(stage)

# Camera — identical placement to the eval.
cp = stage.GetPrimAtPath("/World/ExternalCamera")
if cp.IsValid():
    xf = UsdGeom.Xformable(cp); xf.ClearXformOpOrder(); xf.AddTranslateOp().Set(CAM_POS)
    look = (CAM_TARGET - CAM_POS).GetNormalized()
    q = Gf.Rotation(Gf.Vec3d(0, 0, -1), look).GetQuat()
    xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q.GetReal(), *q.GetImaginary()))
    c = UsdGeom.Camera(cp); ha = 36.0
    c.CreateHorizontalApertureAttr(ha); c.CreateVerticalApertureAttr(ha)
    c.CreateFocalLengthAttr(ha / (2 * math.tan(math.radians(CAM_HFOV) / 2)))

# Optional egocentric view (override the camera + hide the arm) to match the Aria training view.
if os.environ.get("EGOCENTRIC", "0").lower() in ("1", "true", "yes", "y"):
    import ego_view
    ego_view.apply_egocentric(stage, hide_arm=os.environ.get("EGO_HIDE_ARM", "0").lower() in ("1", "true", "yes", "y"))

world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 50.0, rendering_dt=1.0 / 50.0)
world.reset()
pol_cam = Camera(prim_path="/World/ExternalCamera", resolution=POLICY_RES); pol_cam.initialize()
hd_cam  = Camera(prim_path="/World/RecordingCamera", resolution=HD_RES); hd_cam.initialize()
for _ in range(40):
    world.step(render=True)


def grab(cam, res):
    rgba = cam.get_rgba()
    if rgba is None or rgba.size == 0:
        return np.zeros((res[1], res[0], 3), np.uint8)
    return rgba[:, :, :3]


cv2.imwrite("/workspace/preview_policy.png", cv2.cvtColor(grab(pol_cam, POLICY_RES), cv2.COLOR_RGB2BGR))
cv2.imwrite("/workspace/preview_hd.png", cv2.cvtColor(grab(hd_cam, HD_RES), cv2.COLOR_RGB2BGR))
print("[preview] wrote /workspace/preview_policy.png (224x224 policy view) + preview_hd.png (HD)")
simulation_app.close()
print("[preview] done.")
