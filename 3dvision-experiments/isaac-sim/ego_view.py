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
# Above the table on the operator (-Y) side, tilted ~45 deg forward so the table fills the
# near field and the floor + wall show beyond it. Level horizon comes from the look-at.
EGO_CAM_POS    = (0.80, -0.78, 2.75)   # higher above the table
EGO_CAM_TARGET = (0.85,  0.10, 1.60)   # ~53 deg below horizontal (steeper, more floor visible)
EGO_HFOV_DEG   = 95.0          # Aria is wide (~110 hardware); 95 is a reasonable sim value

# RecordingCamera = the HD video camera (not the policy input). The USD ships a stale pose
# that now stares at empty sky, so we re-aim it at the workspace: a pulled-back 3rd-person
# shot from the operator (-Y) side so the whole robot + table + objects are in the video.
REC_CAM_POS    = (0.80, -2.10, 3.00)
REC_CAM_TARGET = (0.85,  0.05, 1.70)
REC_HFOV_DEG   = 60.0          # narrower than the policy cam -> a clean framed 3rd-person shot

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
    # Proper look-at with world up = +Z so the horizon is LEVEL (no arbitrary roll/tilt).
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*EGO_CAM_POS), Gf.Vec3d(*EGO_CAM_TARGET), Gf.Vec3d(0, 0, 1))
    q = view.GetInverse().ExtractRotationQuat()
    xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q.GetReal(), *q.GetImaginary()))
    c = UsdGeom.Camera(cp); ha = 36.0
    c.CreateHorizontalApertureAttr(ha); c.CreateVerticalApertureAttr(ha)
    c.CreateFocalLengthAttr(ha / (2 * math.tan(math.radians(EGO_HFOV_DEG) / 2)))
    print(f"[ego] camera -> egocentric pos={EGO_CAM_POS} target={EGO_CAM_TARGET} hfov={EGO_HFOV_DEG}")


def place_recording_camera(stage, cam_path="/World/RecordingCamera"):
    """Re-aim the HD video camera at the workspace (16:9), so the recording isn't empty sky."""
    cp = stage.GetPrimAtPath(cam_path)
    if not cp.IsValid():
        print(f"[ego] {cam_path} not found"); return
    xf = UsdGeom.Xformable(cp); xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*REC_CAM_POS))
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*REC_CAM_POS), Gf.Vec3d(*REC_CAM_TARGET), Gf.Vec3d(0, 0, 1))
    q = view.GetInverse().ExtractRotationQuat()
    xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q.GetReal(), *q.GetImaginary()))
    c = UsdGeom.Camera(cp); ha = 36.0
    c.CreateHorizontalApertureAttr(ha); c.CreateVerticalApertureAttr(ha * 9.0 / 16.0)  # 16:9 HD
    c.CreateFocalLengthAttr(ha / (2 * math.tan(math.radians(REC_HFOV_DEG) / 2)))
    print(f"[ego] RecordingCamera -> 3rd-person pos={REC_CAM_POS} target={REC_CAM_TARGET}")


def set_arm_visible(stage, visible, root="/World/fr3"):
    """Show/hide the bulky arm links (link0-5) at runtime; render-only, physics/IK unaffected.
    Lets the eval keep the POLICY view armless while toggling the arm ON for the 3rd-person
    recording capture, so the outside video shows the robot in motion."""
    n = 0
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(root):
            continue
        name = prim.GetName().lower()
        if any(k in name for k in _KEEP_SUBSTRINGS):
            continue
        if any(h in name for h in _HIDE_SUBSTRINGS):
            im = UsdGeom.Imageable(prim)
            im.MakeVisible() if visible else im.MakeInvisible()
            n += 1
    return n


def hide_robot_arm(stage, root="/World/fr3"):
    """Hide bulky arm links (render-only; physics/IK unaffected)."""
    n = set_arm_visible(stage, False, root)
    print(f"[ego] hid {n} arm-link prims (render-only)")


def apply_egocentric(stage, hide_arm=False):
    place_egocentric_camera(stage)
    place_recording_camera(stage)
    if hide_arm:
        hide_robot_arm(stage)
