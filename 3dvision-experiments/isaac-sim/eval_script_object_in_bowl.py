"""
vla_eval.py
Run pi0.5 (checkpoint 29999) on a Franka FR3 in Isaac Sim
Task: object_in_bowl — pick up the object and place it in the bowl.
"""

# === MUST BE THE ABSOLUTE FIRST ISAAC-RELATED LINE ===
from isaacsim import SimulationApp

CONFIG = {
    "headless": True,
    "livestream": 0,      # no streaming needed
    "width": 1280,
    "height": 720,
}
simulation_app = SimulationApp(CONFIG)
# ======================================================

import dataclasses
import os
import sys
import csv
import time
import traceback
import numpy as np
import torch
import cv2

from omni.isaac.core import World
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.sensor import Camera

# --------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------
USD_PATH       = "/workspace/kitchen_scene_1.usd"

# --- Model selection (env-overridable; presets live in models/*.env) ----------
# Pick a model by passing its name as submit.sh's 2nd arg, which sources the matching
# models/<name>.env. All of these are robust to an empty ("") forwarded value.
#   CONFIG_NAME     training config             default pi05_egoverse (5-ep object_in_bowl)
#   CHECKPOINT_DIR  orbax checkpoint dir        default the 5-ep checkpoint
#   NORM_STATS_DIR  dir holding norm_stats.json empty -> the config's default asset dir
#   PROMPT          language command            must match the training task label
#   MODEL_NAME      output subfolder            -> results/<MODEL_NAME>/
#   RUN_TAG         output filename suffix      e.g. _quick, _fidelity
CONFIG_NAME    = os.environ.get("CONFIG_NAME") or "pi05_egoverse"
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR") or "/checkpoints/pi05_egoverse/test/29999"
NORM_STATS_DIR = os.environ.get("NORM_STATS_DIR") or ""
LANGUAGE_COMMAND = os.environ.get("PROMPT") or "put the object in the bowl"
MODEL_NAME     = os.environ.get("MODEL_NAME") or "default"
RUN_TAG        = os.environ.get("RUN_TAG") or ""

RESULTS_DIR    = f"/workspace/results/{MODEL_NAME}"
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_CSV    = f"{RESULTS_DIR}/results{RUN_TAG}.csv"
VIDEO_PATH     = f"{RESULTS_DIR}/evaluation{RUN_TAG}.mp4"
print(f"[init] model={MODEL_NAME} config={CONFIG_NAME} ckpt={CHECKPOINT_DIR}")
print(f"[init] outputs -> {RESULTS_DIR}/  (tag='{RUN_TAG}')")

NUM_STEPS      = int(os.environ.get("NUM_STEPS") or "3000")   # 60 s at 50 Hz
NUM_ARM_JOINTS = 7

POLICY_CAM_RES = (224, 224)   # pi0.5 input — ExternalCamera
HD_VIDEO_RES   = (1280, 720)  # RecordingCamera → evaluation.mp4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[init] device = {device}")


# --------------------------------------------------------------------
# LOAD THE pi0.5 POLICY
# --------------------------------------------------------------------
sys.path.insert(0, "/workspace/openpi/src")
sys.path.insert(0, "/workspace/openpi/packages/openpi-client/src")
sys.path.insert(0, "/isaac_packages")

# Force-load the correct typing_extensions before openpi imports
# (Isaac Sim caches an ancient version in sys.modules at startup)
import importlib.util as _ilu
_te_spec = _ilu.spec_from_file_location("typing_extensions", "/isaac_packages/typing_extensions.py")
_te_mod  = _ilu.module_from_spec(_te_spec)
sys.modules["typing_extensions"] = _te_mod
_te_spec.loader.exec_module(_te_mod)
del _ilu, _te_spec, _te_mod

try:
    from openpi.policies import policy_config
    from openpi.shared import normalize
    from openpi.training import config as _config

    cfg       = _config.get_config(CONFIG_NAME)
    cfg       = dataclasses.replace(cfg, assets_base_dir="/workspace/openpi/assets")
    data_cfg  = cfg.data.create(cfg.assets_dirs, cfg.model)
    # Norm stats: prefer an explicit dir (e.g. the checkpoint's own assets), else the
    # config's default asset dir under /workspace/openpi/assets.
    import pathlib as _pl
    _ns_dir = _pl.Path(NORM_STATS_DIR) if NORM_STATS_DIR else (cfg.assets_dirs / data_cfg.repo_id)
    print(f"[init] norm stats from {_ns_dir}")
    norm_stats = normalize.load(_ns_dir)

    policy = policy_config.create_trained_policy(
        cfg, CHECKPOINT_DIR, norm_stats=norm_stats,
        default_prompt=LANGUAGE_COMMAND,
    )
    print(f"[init] Loaded {CONFIG_NAME} from {CHECKPOINT_DIR}")
except Exception as e:
    print(f"[FATAL] Could not load policy: {e}")
    traceback.print_exc()
    simulation_app.close()
    sys.exit(1)


# --------------------------------------------------------------------
# LOAD THE SCENE AND PATCH ALL S3 PAYLOADS TO LOCAL ASSETS
# --------------------------------------------------------------------
print(f"[init] Opening stage {USD_PATH}")
open_stage(usd_path=USD_PATH)

import omni.usd
import math
from pxr import Sdf, UsdGeom, Gf, UsdPhysics, UsdShade  # noqa: Gf still used for Gf.Vec3f in attribute writes

_stage = omni.usd.get_context().get_stage()

# Map of prim path -> local USD file.
# fr3_full has the complete asset including configuration/fr3_robot_schema.usd
# which defines joint limits and damping — critical for stable behaviour.
_PAYLOAD_PATCHES = {
    "/World/fr3":                           "/workspace/assets/fr3_full/fr3.usd",
    "/World/SM_HeavyDutyPackingTable_C02_01": "/workspace/assets/table/SM_HeavyDutyPackingTable_C02_01.usd",
    # plate + crate intentionally dropped — replaced by the object/bowl below.
}

for prim_path, local_usd in _PAYLOAD_PATCHES.items():
    prim = _stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        prim.GetPayloads().ClearPayloads()
        prim.GetPayloads().AddPayload(local_usd)
        print(f"[init] Patched {prim_path} -> {local_usd}")
    else:
        print(f"[WARN] {prim_path} not found in stage — skipping patch")

# ExternalCamera: match Aria RGB FoV + a more egocentric (looking-down-at-workspace) pose.
# Scene: table top z≈1.807, robot (0.09,0.07,1.807), cube (0.53,-0.41,1.85), bowl (1.46,-0.02,1.807).
# Aria RGB hFOV: README target is 76°; hardware spec is closer to ~110°. Tune ARIA_HFOV_DEG if needed.
# Pose: above + slightly in front (operator/-Y side), looking steeply down at the workspace center —
# closer to a head-mounted egocentric view than the previous near-horizontal standing-back shot.
ARIA_HFOV_DEG = 90.0   # Aria hardware FOV ~110°; 90° is a reasonable sim approximation
_CAM_POS    = Gf.Vec3d(-0.2, -0.8, 2.6)    # front-left of scene, ~35° down — matches Aria egocentric angle
_CAM_TARGET = Gf.Vec3d(0.8, -0.15, 1.82)   # workspace center (sphere+bowl), robot arm appears upper-right
_cam_prim = _stage.GetPrimAtPath("/World/ExternalCamera")
if _cam_prim.IsValid():
    _xf = UsdGeom.Xformable(_cam_prim)
    _xf.ClearXformOpOrder()
    _xf.AddTranslateOp().Set(_CAM_POS)
    _look_dir = (_CAM_TARGET - _CAM_POS).GetNormalized()
    _rot = Gf.Rotation(Gf.Vec3d(0.0, 0.0, -1.0), _look_dir)  # camera looks down its local -Z
    _quat = _rot.GetQuat()
    _xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(_quat.GetReal(), *_quat.GetImaginary()))
    # FoV: only the aperture/focalLength ratio matters; square sensor for the 224x224 policy input.
    _cam = UsdGeom.Camera(_cam_prim)
    _h_aperture = 36.0
    _focal = _h_aperture / (2.0 * math.tan(math.radians(ARIA_HFOV_DEG) / 2.0))
    _cam.CreateHorizontalApertureAttr(_h_aperture)
    _cam.CreateVerticalApertureAttr(_h_aperture)
    _cam.CreateFocalLengthAttr(_focal)
    print(f"[init] ExternalCamera: pos={tuple(_CAM_POS)} -> target={tuple(_CAM_TARGET)}, hFOV={ARIA_HFOV_DEG}deg")
else:
    print("[WARN] ExternalCamera prim not found — using original position")

# --------------------------------------------------------------------
# OBJECT_IN_BOWL SCENE EDIT (authored in-code with Isaac's USD)
# Remove the plate + crate; add a graspable cube ("object") and a bowl.
# Native geometry => no external assets, loads offline on the GPU node.
# Positions reused from the originals (known on-table, in ExternalCamera view).
# Units are scene units (stage metersPerUnit ~0.55); sizes chosen to be
# graspable by the FR3 gripper and to let the cube drop into the bowl.
# --------------------------------------------------------------------
for _old in ("/World/plate_small", "/World/SM_Crate_A07_Yellow_01_physics"):
    _p = _stage.GetPrimAtPath(_old)
    if _p.IsValid():
        _p.SetActive(False)
        print(f"[init] Deactivated {_old}")

# --- Object + bowl from the SHARED scene module: build EXACTLY the scene_preview scene
# (4-colour mesh ball + wide dusty-purple bowl at the canonical positions), with physics so
# the ball is graspable and drops into the bowl. The table look is handled by scene_fidelity. ---
sys.path.insert(0, "/workspace")
import scene_build
# BALL_JITTER_SEED moves the ball off OBJECT_POS by a seeded 10-15 cm planar offset, WITHOUT
# touching the calibration (which stays anchored to nominal OBJECT_POS). Tracking test: if the
# reach follows the moved ball the policy is localizing it from vision; if it stays at nominal
# it is replaying a fixed/calibrated pose (no visual tracking).
_ball_seed = int(os.environ.get("BALL_JITTER_SEED") or "0")
_ball_pos = scene_build.jittered_object_pos(_ball_seed) if _ball_seed else scene_build.OBJECT_POS
if _ball_seed:
    print(f"[ball] jittered seed={_ball_seed} pos={tuple(round(v, 3) for v in _ball_pos)} "
          f"(calib still anchored to nominal {scene_build.OBJECT_POS})")
# BALL_RADIUS enlarges the ball (default 0.055 m) for a bigger, more salient target in the
# 224x224 policy view — steadier visual tracking. Bowl scales with it for proportion.
_ball_r = float(os.environ.get("BALL_RADIUS") or "0.055")
scene_build.add_ball(_stage, "/World/object", _ball_pos, r=_ball_r)
scene_build.add_bowl(_stage, "/World/bowl", scene_build.BOWL_POS)

# Base-frame position of the ACTUAL ball (possibly jittered) for the reach-vs-ball diagnostic.
# The policy's commanded EE target IS its implicit "where I think the ball is" (the task is to
# reach the ball), so the loop logs that target vs this and the xy error.
try:
    from pxr import Gf as _Gfb
    from pxr import UsdGeom as _UGb
    _w2b_ball = _UGb.XformCache().GetLocalToWorldTransform(_stage.GetPrimAtPath("/World/fr3")).GetInverse()
    _v = _w2b_ball.Transform(_Gfb.Vec3d(*[float(x) for x in _ball_pos]))
    _BALL_ACTUAL_BASE = np.array([_v[0], _v[1], _v[2]], np.float64)
    print(f"[ball] actual ball in BASE frame = {np.round(_BALL_ACTUAL_BASE, 3).tolist()}")
except Exception as _e:
    _BALL_ACTUAL_BASE = None
    print(f"[ball] could not compute base-frame ball pos: {_e}")

# Opt-in scene realism (lights, floor, backdrop) to reduce SigLIP OOD. Off by default
# so the validated path is unchanged; enable with SCENE_FIDELITY=1. Preview the look
# fast (no policy) with scene_preview.py before running a full eval.
if os.environ.get("SCENE_FIDELITY", "0").lower() in ("1", "true", "yes", "y"):
    sys.path.insert(0, "/workspace")
    import scene_fidelity
    scene_fidelity.apply_fidelity(_stage)

# Optional egocentric camera (+ arm-hiding), matching the training view; overrides the
# ExternalCamera placed above. Enable with EGOCENTRIC=1 (this is robot data, so leave the
# arm visible — only set EGO_HIDE_ARM=1 for the human-hand models).
if os.environ.get("EGOCENTRIC", "0").lower() in ("1", "true", "yes", "y"):
    sys.path.insert(0, "/workspace")
    import ego_view
    ego_view.apply_egocentric(_stage, hide_arm=os.environ.get("EGO_HIDE_ARM", "0").lower() in ("1", "true", "yes", "y"))
else:
    # ALWAYS re-aim the HD RecordingCamera at the workspace: the USD ships a stale pose that
    # stares at empty sky / the fidelity wall -> blank-blue video. This touches ONLY the output
    # video, never the policy camera, so it's safe to run on the validated (non-egocentric)
    # path. (apply_egocentric already re-aims it, hence the else.) Opt out with REC_CAM_REAIM=0.
    if os.environ.get("REC_CAM_REAIM", "1").lower() in ("1", "true", "yes", "y"):
        sys.path.insert(0, "/workspace")
        import ego_view
        ego_view.place_recording_camera(_stage)

# Data is 50 Hz; step physics at 1/50 s so the policy runs at its training control rate
# (Isaac default is 1/60, which would desync observation/action cadence from training).
world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 50.0, rendering_dt=1.0 / 50.0)
world.reset()

# --- Robot ---
franka = Articulation(prim_path="/World/fr3", name="franka")
franka.initialize()
print(f"[init] Franka has {franka.num_dof} DOF")

# --- Policy camera: ExternalCamera in USD (224×224, repositioned by user) ---
external_cam = Camera(prim_path="/World/ExternalCamera", resolution=POLICY_CAM_RES)
external_cam.initialize()

# --- Recording camera: RecordingCamera in USD (1280×720, 3rd person view) ---
recording_cam = Camera(prim_path="/World/RecordingCamera", resolution=HD_VIDEO_RES)
recording_cam.initialize()

# Warm up so camera buffers are filled
print("[init] Warming up cameras...")
for _ in range(20):
    world.step(render=True)

print("[init] Cameras ready: ExternalCamera (policy, 224×224), RecordingCamera (HD, 1280×720)")


# --------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------
def get_frame(cam, expected_res):
    """Return (H,W,3) uint8 from Camera, or a black frame on failure."""
    rgba = cam.get_rgba()
    if rgba is None or rgba.size == 0:
        return np.zeros((expected_res[1], expected_res[0], 3), dtype=np.uint8)
    return rgba[:, :, :3]


# Save the first policy-camera frame so we can verify the camera view
_diag_frame = get_frame(external_cam, POLICY_CAM_RES)
cv2.imwrite(f"{RESULTS_DIR}/policy_cam_init.png", cv2.cvtColor(_diag_frame, cv2.COLOR_RGB2BGR))
print(f"[init] Saved policy camera preview → {RESULTS_DIR}/policy_cam_init.png")



# --------------------------------------------------------------------
# FK/IK — the arm half of state/action is an EE POSE, not joints (convention
# resolved 2026-06-03: base frame, xyzw, panda_hand, absolute). So:
#   observation arm state = FK(sim joints)   [sim joints -> EE pose]
#   action execution      = IK(policy pose)  [EE pose -> joint targets]
# This replaces the old (wrong) "_SIM_TO_TRAIN permutation feeding pose into joints".
# --------------------------------------------------------------------
sys.path.insert(0, "/workspace")
import ik_fk_helpers

EE_FRAME  = os.environ.get("EE_FRAME", "panda_hand")
QUAT_WXYZ = os.environ.get("QUAT_WXYZ", "0").lower() in ("1", "true", "yes", "y")
kin = ik_fk_helpers.FrankaKinematics(ee_frame=EE_FRAME, quat_wxyz=QUAT_WXYZ)

# Starting EE pose (base frame, stored xyzw) ~ a typical demo frame-0: gripper above the
# workspace pointing down. IK'd once for the home posture so the first observation is in
# distribution. (xyzw frame-0 [0.997,...] = ~173° about +x = pointing down.)
START_EE_POSE = np.array([0.47, 0.02, 0.28, 0.997, 0.004, -0.042, -0.057], dtype=np.float64)


# --------------------------------------------------------------------
# OPTIONAL FRAME CALIBRATION (EXPERIMENTAL, default OFF — RID_CALIBRATE=1)
# --------------------------------------------------------------------
# object_in_bowl arm actions are CLAIMED to be in the Franka base frame (convention resolved
# 2026-06-03), but the 5-ep model froze at the mean, so this was never tested with a model
# that actually moves. If a stronger model (rid30/rid64) un-freezes yet reaches the WRONG
# place — exactly what the oic model did — its EE frame is egocentric, not base. This solves
# the SAME rigid transform (R,t,s) the oic path used (oic_frame_calib.umeyama), anchoring the
# demo's start/grasp/release positions to the sim home/ball/bowl, then maps model<->base in
# the observation (FK) and action (IK) hooks below.
#
# Diagnostic property: if the model is genuinely base-frame already, the solve returns R≈I,
# s≈1, t≈0 and the hooks are near-no-ops — so the printed scale/residual TELLS you which frame
# the model lives in. Needs rid_demo.npz (see make_rid_demo.py). Position is mapped always;
# orientation only if RID_CALIB_ROT=1 (needs scipy in /isaac_packages).
_CALIB_T = None   # (R, t, s): base_pos = s*R@model_pos + t ; None => identity (no-op)
if os.environ.get("RID_CALIBRATE", "0").lower() in ("1", "true", "yes", "y"):
    import oic_frame_calib
    import scene_build
    from pxr import Gf as _Gf
    from pxr import UsdGeom as _UG
    _demo_npz = os.environ.get("RID_DEMO_NPZ") or "/workspace/rid_demo.npz"
    _demo_pos = np.asarray(np.load(_demo_npz)["actions24"], np.float64)[:, :3]
    _w2b = _UG.XformCache().GetLocalToWorldTransform(_stage.GetPrimAtPath("/World/fr3")).GetInverse()

    def _w2b_pt(p):
        v = _w2b.Transform(_Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
        return np.array([v[0], v[1], v[2]], np.float64)

    _home_base = np.asarray(START_EE_POSE[:3], np.float64)   # sim home is already base-frame
    _ball_base = _w2b_pt(scene_build.OBJECT_POS)
    _bowl_base = _w2b_pt(scene_build.BOWL_POS)
    # scale=1 (RID_CALIB_SCALE=0) maps without amplifying motion (fixes overreach when the live
    # trajectory is larger than the demo); RID_CALIB_SCALE_MUL scales the solved s (e.g. 0.7 to
    # damp a too-aggressive reach without dropping all the way to 1.0).
    _with_scale = os.environ.get("RID_CALIB_SCALE", "1").lower() in ("1", "true", "yes", "y")
    _R, _t, _s, _fr, _res = oic_frame_calib.build_transform(
        _demo_pos, (_home_base, _ball_base, _bowl_base),
        anchor_frames=os.environ.get("RID_ANCHOR_FRAMES") or None, with_scale=_with_scale)
    _s *= float(os.environ.get("RID_CALIB_SCALE_MUL") or "1.0")
    # Re-solve translation so the anchors stay centered after the scale tweak (t = mean_base - s*R@mean_model).
    _mm = _demo_pos[list(_fr)].mean(0); _mb = np.array([_home_base, _ball_base, _bowl_base]).mean(0)
    _t = _mb - _s * (_R @ _mm)
    _CALIB_T = (_R, _t, _s)
    print(f"[calib] frames(start,grasp,release)={_fr} scale={_s:.3f} resid_m={np.round(_res, 3).tolist()}")
    print(f"[calib] home_base={np.round(_home_base, 3).tolist()} ball_base={np.round(_ball_base, 3).tolist()} "
          f"bowl_base={np.round(_bowl_base, 3).tolist()}")
    _tn = float(np.linalg.norm(_t))
    _near_identity = abs(_s - 1.0) < 0.1 and _tn < 0.05
    print(f"[calib] t={np.round(_t, 3).tolist()} ||t||={_tn:.3f} scale={_s:.3f} => "
          + ("model ≈ base-frame (calib is a near no-op)" if _near_identity
             else "model is NOT base-frame; calib repositions + rescales the reach"))

_CALIB_ROT = os.environ.get("RID_CALIB_ROT", "0").lower() in ("1", "true", "yes", "y")


def _to_base_pose(pose7):
    """[x,y,z, qx,qy,qz,qw] MODEL frame -> BASE frame (for IK). No-op if uncalibrated."""
    if _CALIB_T is None:
        return pose7
    R, t, s = _CALIB_T
    out = np.array(pose7, np.float64)
    out[:3] = s * (R @ out[:3]) + t
    if _CALIB_ROT:
        from scipy.spatial.transform import Rotation as _Rot
        out[3:7] = _Rot.from_matrix(R @ _Rot.from_quat(out[3:7]).as_matrix()).as_quat()
    return out


def _to_model_pose(pose7):
    """Inverse: BASE-frame pose -> MODEL frame (for the observation state). No-op if uncalibrated."""
    if _CALIB_T is None:
        return pose7
    R, t, s = _CALIB_T
    out = np.array(pose7, np.float64)
    out[:3] = (R.T @ (out[:3] - t)) / s
    if _CALIB_ROT:
        from scipy.spatial.transform import Rotation as _Rot
        out[3:7] = _Rot.from_matrix(R.T @ _Rot.from_quat(out[3:7]).as_matrix()).as_quat()
    return out


# Training-mean hand state (dims 7-23) across all 5 training episodes.
# Feeding zeros was OOD; using the mean keeps state in the training distribution.
# Updated each step with the policy's own hand action prediction (autoregressive).
HAND_MEAN = np.array([
     0.236, -0.282,  0.555,  0.717,  0.583,  0.049,  1.000,
     0.157,  0.092,  0.854,  1.020,  0.101,  0.964,  0.919,
     0.110,  0.984,  0.943,
], dtype=np.float32)

_hand_state = HAND_MEAN.copy()   # updated each step


def build_observation(ext_img_uint8, joint_pos):
    # Arm state = FK(current 7 arm joints) -> EE pose [x,y,z, xyzw] in base frame,
    # matching the training state (qpos_arm is an EE pose, same convention as actions).
    ee_pose = kin.fk(joint_pos[:7])           # (7,) stored convention, BASE frame
    ee_pose = _to_model_pose(ee_pose)         # -> MODEL frame (no-op unless RID_CALIBRATE=1)
    state = np.zeros(24, dtype=np.float32)
    state[:7]   = ee_pose.astype(np.float32)
    state[7:24] = _hand_state                 # prev predicted hand action (in-distribution)
    return {
        "observation/image": ext_img_uint8,
        "observation/state": state,
        "prompt": LANGUAGE_COMMAND,
    }


def to_gripper_positions(gripper_cmd):
    gripper_cmd = float(np.clip(gripper_cmd, 0.0, 1.0))
    finger_pos = 0.04 * (1.0 - gripper_cmd)
    return finger_pos, finger_pos


# --------------------------------------------------------------------
# MAIN LOOP
# --------------------------------------------------------------------
csv_file = open(RESULTS_CSV, "w", newline="")
writer   = csv.writer(csv_file)
writer.writerow(["step", "infer_ms", "ik_ok", "tx", "ty", "tz"] + [f"j{i}" for i in range(9)])

# HD video from RecordingCamera at 50 fps
video_writer = cv2.VideoWriter(
    VIDEO_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    50,
    HD_VIDEO_RES,
)

last_action_chunk = None
chunk_idx         = 0
step              = 0
smoothed_cmd      = None      # exponential moving average over joint position targets
# Blend new joint target with previous (EMA). 1.0 = no smoothing (raw, jittery); lower =
# steadier but laggier. 0.8 default; ACTION_SMOOTH_ALPHA=0.5 damps the ball<->bowl oscillation
# seen with rid64 (the old "0.4 froze" note was the WEAK 5-ep model; rid64 has real signal).
ACTION_SMOOTH_ALPHA = float(os.environ.get("ACTION_SMOOTH_ALPHA") or "0.8")

try:
    world.reset()
    franka.initialize()          # must re-init after world.reset()
    for _ in range(20):
        world.step(render=True)

    # Home: IK the starting EE pose so the first observation is in-distribution.
    home_joints, home_ok = kin.ik(START_EE_POSE, None)
    if not home_ok or home_joints is None:
        print("[WARN] home IK failed — falling back to Franka ready pose")
        home_joints = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    _warm = np.asarray(home_joints, dtype=np.float64)   # IK warmstart carried through the loop
    home_cmd = np.zeros(9, dtype=np.float32)
    home_cmd[:7] = home_joints
    home_cmd[7] = home_cmd[8] = 0.02
    print(f"[init] Home via IK(START_EE_POSE): ok={home_ok} joints={np.round(home_joints, 3)}")
    for _ in range(100):
        franka.apply_action(ArticulationAction(joint_positions=home_cmd))
        world.step(render=True)
    pos = franka.get_joint_positions()
    print(f"[init] At home after 100 steps: {pos[:7].round(3)}")

    print("[run] Starting evaluation...")
    for step in range(NUM_STEPS):

        # ---- LOOK ----
        policy_img = get_frame(external_cam, POLICY_CAM_RES)
        hd_img     = get_frame(recording_cam, HD_VIDEO_RES)
        joint_pos  = franka.get_joint_positions()
        if joint_pos is None:
            joint_pos = np.zeros(9, dtype=np.float32)

        # ---- THINK ----
        # Re-query every step for the first 200 steps so the arm responds quickly
        # to the initial scene, then switch to chunk-based inference (every 10 steps).
        t0 = time.time()
        _force_requery = step < 200
        if last_action_chunk is None or chunk_idx >= len(last_action_chunk) or _force_requery:
            obs = build_observation(policy_img, joint_pos)
            with torch.no_grad():
                result = policy.infer(obs)
            last_action_chunk = np.asarray(result["actions"])
            chunk_idx = 0

        action    = last_action_chunk[chunk_idx]
        chunk_idx += 1
        infer_ms  = (time.time() - t0) * 1000

        # Log raw policy arm actions at key steps to detect OOD/frozen policy
        if step == 0:
            print(f"[diag] chunk shape={last_action_chunk.shape}  arm EE-pose actions (first 3 chunks):")
            for _ci in range(min(3, len(last_action_chunk))):
                _a = last_action_chunk[_ci]
                print(f"  chunk[{_ci}] EE-pose={np.round(_a[:7], 3)}")
            cv2.imwrite(f"{RESULTS_DIR}/policy_cam_step0.png",
                        cv2.cvtColor(policy_img, cv2.COLOR_RGB2BGR))
        if step % 500 == 0 and step > 0:
            print(f"[diag] step {step} action EE-pose={np.round(action[:7], 3)}")

        # ---- ACT ----
        arm_pose      = action[:NUM_ARM_JOINTS]    # policy output is an EE pose (base, xyzw)
        hand_action   = action[NUM_ARM_JOINTS:]
        gripper_cmd   = float(np.mean(hand_action[:3]))
        _hand_state[:] = hand_action               # autoregressive hand state (in-distribution)

        # EE pose -> joint targets via IK; warmstart from the previous solution for continuity.
        # _to_base_pose maps MODEL->BASE frame when RID_CALIBRATE=1 (no-op otherwise).
        base_pose = _to_base_pose(arm_pose)
        arm_joints, ik_ok = kin.ik(base_pose, _warm)

        # "Where the robot thinks the ball is" = its commanded base-frame reach target, vs the
        # actual ball. xy_err shrinking toward the ball => it's reaching the right place.
        if step % 50 == 0 and _BALL_ACTUAL_BASE is not None:
            _xy = float(np.linalg.norm(np.asarray(base_pose[:2], float) - _BALL_ACTUAL_BASE[:2]))
            # grip≈1 = closing/closed, ≈0 = open. If it never rises near the ball (low xy_err),
            # the grasp isn't being commanded -> that's why the reach won't dwell/complete.
            print(f"[reach] step {step:4d} target_base={np.round(base_pose[:3], 3).tolist()} "
                  f"ball_base={np.round(_BALL_ACTUAL_BASE, 3).tolist()} xy_err={_xy:.3f}m grip={gripper_cmd:.2f}")
            # Full 17-dim ORCA hand action: is a grasp encoded in ANY dim when xy_err is small?
            # max/argmax tells us which dim to drive the gripper from if mean(hand[:3]) misses it.
            print(f"[hand] step {step:4d} max={hand_action.max():.2f}@{int(hand_action.argmax())} "
                  f"min={hand_action.min():.2f} hand={np.round(hand_action, 2).tolist()}")
        if ik_ok and arm_joints is not None:
            _warm = np.asarray(arm_joints, dtype=np.float64)
        # else: hold the previous joints (don't jump the arm on an IK failure)

        finger_l, finger_r = to_gripper_positions(gripper_cmd)
        full_cmd = np.zeros(9, dtype=np.float32)
        full_cmd[:7] = _warm
        full_cmd[7] = finger_l
        full_cmd[8] = finger_r
        # Smooth joint targets to reduce chunk-boundary jitter
        if smoothed_cmd is None:
            smoothed_cmd = full_cmd.copy()
        else:
            smoothed_cmd = ACTION_SMOOTH_ALPHA * full_cmd + (1.0 - ACTION_SMOOTH_ALPHA) * smoothed_cmd
        franka.apply_action(ArticulationAction(joint_positions=smoothed_cmd))

        # ---- STEP ----
        world.step(render=True)

        # ---- RECORD (HD from RecordingCamera) ----
        video_writer.write(cv2.cvtColor(hd_img, cv2.COLOR_RGB2BGR))

        # ---- LOG ---- (tx,ty,tz = commanded EE-pose target this step)
        writer.writerow([step, f"{infer_ms:.1f}", int(ik_ok),
                         f"{arm_pose[0]:.3f}", f"{arm_pose[1]:.3f}", f"{arm_pose[2]:.3f}"]
                        + joint_pos.tolist())
        if step % 50 == 0:
            jp = joint_pos[:7]
            print(f"[run] step {step:4d} | infer {infer_ms:5.1f}ms | "
                  f"arm {jp.round(3)}")

except Exception as e:
    print(f"[FATAL] Crashed at step {step}: {e}")
    traceback.print_exc()

finally:
    print("[exit] Closing...")
    csv_file.close()
    video_writer.release()
    print(f"[exit] Video saved to {VIDEO_PATH}")
    simulation_app.close()
    print("[exit] Done.")
