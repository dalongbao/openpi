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
from pxr import UsdGeom, Gf, UsdPhysics, Sdf

sys.path.insert(0, "/workspace")
import scene_fidelity
import scene_build

USD_PATH   = "/workspace/kitchen_scene_1.usd"
POLICY_RES = (224, 224)
HD_RES     = (1280, 720)

# Camera defaults (overridden by ego_view when EGOCENTRIC=1).
CAM_POS    = Gf.Vec3d(-0.2, -0.8, 2.6)
CAM_TARGET = Gf.Vec3d(0.8, -0.15, 1.82)
CAM_HFOV   = 90.0

open_stage(usd_path=USD_PATH)
stage = omni.usd.get_context().get_stage()

# Objects (ball + bowl) + offline asset patching from the SHARED scene module, so the preview
# renders exactly what the eval builds. physics=False pins the ball for the static snapshot.
scene_build.build_objects(stage, physics=False)

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
