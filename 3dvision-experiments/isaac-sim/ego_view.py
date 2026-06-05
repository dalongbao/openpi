"""
ego_view.py — make the policy camera resemble the egocentric Aria training view that the
egoverse models were trained on: a head-height camera looking DOWN-FORWARD across the
workspace, with the bulky robot arm optionally hidden so the frame shows table + objects
+ (at most) a small end-effector — closer to the human-hand egocentric data than the
external "third-person robot arm" view that both models failed to ground on.

Toggle from the eval/preview:
  EGOCENTRIC=1     reposition the policy camera to the egocentric pose
  EGO_HIDE_ARM=1   also hide the bulky arm links (keep wrist+hand+fingers)

Tune the constants below and re-render with scene_preview.py (no policy, ~3-4 min).
"""
import math
from pxr import UsdGeom, Gf

# Head-height, operator (-Y) side, looking down-forward across the table at the objects,
# so the gripper enters the lower frame like a hand in an egocentric view.
EGO_CAM_POS    = (0.30, -1.00, 2.40)
EGO_CAM_TARGET = (0.95, -0.10, 1.81)
EGO_HFOV_DEG   = 95.0          # Aria is wide (~110 hardware); 95 is a reasonable sim value

# Arm-hiding: drop the bulky base/shoulder/elbow links; keep wrist + hand + fingers so a
# small end-effector still appears near the objects.
_HIDE_SUBSTRINGS = ("link0", "link1", "link2", "link3", "link4", "link5")
_KEEP_SUBSTRINGS = ("link6", "link7", "hand", "finger")


def place_egocentric_camera(stage, cam_path="/World/ExternalCamera"):
    cp = stage.GetPrimAtPath(cam_path)
    if not cp.IsValid():
        print(f"[ego] {cam_path} not found"); return
    xf = UsdGeom.Xformable(cp); xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*EGO_CAM_POS))
    look = (Gf.Vec3d(*EGO_CAM_TARGET) - Gf.Vec3d(*EGO_CAM_POS)).GetNormalized()
    q = Gf.Rotation(Gf.Vec3d(0, 0, -1), look).GetQuat()
    xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q.GetReal(), *q.GetImaginary()))
    c = UsdGeom.Camera(cp); ha = 36.0
    c.CreateHorizontalApertureAttr(ha); c.CreateVerticalApertureAttr(ha)
    c.CreateFocalLengthAttr(ha / (2 * math.tan(math.radians(EGO_HFOV_DEG) / 2)))
    print(f"[ego] camera -> egocentric pos={EGO_CAM_POS} target={EGO_CAM_TARGET} hfov={EGO_HFOV_DEG}")


def hide_robot_arm(stage, root="/World/fr3"):
    """Hide bulky arm links (render-only; physics/IK unaffected)."""
    n = 0
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(root):
            continue
        name = prim.GetName().lower()
        if any(k in name for k in _KEEP_SUBSTRINGS):
            continue
        if any(h in name for h in _HIDE_SUBSTRINGS):
            UsdGeom.Imageable(prim).MakeInvisible(); n += 1
    print(f"[ego] hid {n} arm-link prims (render-only)")


def apply_egocentric(stage, hide_arm=False):
    place_egocentric_camera(stage)
    if hide_arm:
        hide_robot_arm(stage)
