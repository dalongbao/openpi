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
CHECKPOINT_DIR = "/checkpoints/pi05_egoverse/test/29999"
RESULTS_CSV    = "/workspace/results.csv"
VIDEO_PATH     = "/workspace/evaluation.mp4"

LANGUAGE_COMMAND = "put the object in the bowl"  # must match training task label exactly

NUM_STEPS      = 3000   # 60 s at 50 Hz
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

    cfg       = _config.get_config("pi05_egoverse")
    cfg       = dataclasses.replace(cfg, assets_base_dir="/workspace/openpi/assets")
    data_cfg  = cfg.data.create(cfg.assets_dirs, cfg.model)
    norm_stats = normalize.load(cfg.assets_dirs / data_cfg.repo_id)

    policy = policy_config.create_trained_policy(
        cfg, CHECKPOINT_DIR, norm_stats=norm_stats,
        default_prompt=LANGUAGE_COMMAND,
    )
    print(f"[init] Loaded pi0.5 from {CHECKPOINT_DIR}")
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
ARIA_HFOV_DEG = 76.0
_CAM_POS    = Gf.Vec3d(0.90, -0.70, 2.90)   # above workspace, on the operator (-Y) side
_CAM_TARGET = Gf.Vec3d(0.98, -0.20, 1.81)   # workspace center (between cube & bowl), at table height
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

_OBJECT_POS = (0.527, -0.405, 1.85)   # was plate_small (Z lifted so cube rests on table)
_BOWL_POS   = (1.463, -0.020, 1.807)  # was crate (bowl bottom at table surface)


def _add_sphere_object(stage, path, pos, radius=0.04):
    # Small ball to match the real scene (red/green ball; rendered solid red here).
    sph = UsdGeom.Sphere.Define(stage, path)
    sph.CreateRadiusAttr(radius)
    UsdGeom.Xformable(sph).AddTranslateOp().Set(Gf.Vec3d(*pos))
    prim = sph.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)            # analytic sphere collider
    UsdPhysics.RigidBodyAPI.Apply(prim)            # dynamic
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(0.03)
    sph.CreateDisplayColorAttr([Gf.Vec3f(0.80, 0.15, 0.15)])
    print(f"[init] Added sphere object at {path} {pos} r={radius}")


def _build_bowl_mesh(Rb=0.10, Rt=0.18, H=0.13, wall=0.025, n=24):
    """Open conical cup (watertight, no degenerate apex). Opens +Z (stage upAxis=Z)."""
    pts = []

    def ring(r, z):
        base = len(pts)
        for j in range(n):
            a = 2.0 * math.pi * j / n
            pts.append(Gf.Vec3f(r * math.cos(a), r * math.sin(a), z))
        return base

    ob = ring(Rb, 0.0)            # outer bottom
    ot = ring(Rt, H)             # outer top (rim, outer)
    it = ring(Rt - wall, H)      # inner top (rim, inner)
    ib = ring(Rb - wall, wall)   # inner bottom
    oc = len(pts); pts.append(Gf.Vec3f(0, 0, 0.0))    # outer bottom center
    ic = len(pts); pts.append(Gf.Vec3f(0, 0, wall))   # inner floor center

    counts, idx = [], []

    def quad(a, b, c, d):
        counts.append(4); idx.extend([a, b, c, d])

    def tri(a, b, c):
        counts.append(3); idx.extend([a, b, c])

    for j in range(n):
        k = (j + 1) % n
        quad(ob + j, ob + k, ot + k, ot + j)   # outer wall
        quad(ot + j, ot + k, it + k, it + j)   # rim annulus
        quad(it + j, it + k, ib + k, ib + j)   # inner wall
        tri(ic, ib + k, ib + j)                # inner floor
        tri(oc, ob + j, ob + k)                # outer underside
    return pts, counts, idx


def _add_bowl(stage, path, pos):
    mesh = UsdGeom.Mesh.Define(stage, path)
    pts, counts, idx = _build_bowl_mesh()
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(idx)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    UsdGeom.Xformable(mesh).AddTranslateOp().Set(Gf.Vec3d(*pos))
    prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")  # exact triangle mesh => concave
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.45, 0.22, 0.55)])  # purple, matches real bowl
    print(f"[init] Added bowl at {path} {pos}")


def _make_table_wooden(stage, table_path="/World/SM_HeavyDutyPackingTable_C02_01"):
    # Override the table's (broken/offline) MDL materials with a flat wooden-brown
    # UsdPreviewSurface, bound strongerThanDescendants so it wins over the mesh bindings.
    tprim = stage.GetPrimAtPath(table_path)
    if not tprim.IsValid():
        print(f"[WARN] table {table_path} not found — skipping wooden override")
        return
    mat = UsdShade.Material.Define(stage, table_path + "/WoodenMat")
    shader = UsdShade.Shader.Define(stage, table_path + "/WoodenMat/PBR")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.40, 0.26, 0.13))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.75)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(tprim).Bind(
        mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    print(f"[init] Bound wooden-brown material to {table_path}")


_add_sphere_object(_stage, "/World/object", _OBJECT_POS)
_add_bowl(_stage, "/World/bowl", _BOWL_POS)
_make_table_wooden(_stage)

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
cv2.imwrite("/workspace/policy_cam_init.png", cv2.cvtColor(_diag_frame, cv2.COLOR_RGB2BGR))
print(f"[init] Saved policy camera preview → /workspace/policy_cam_init.png")



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
    ee_pose = kin.fk(joint_pos[:7])           # (7,) stored convention
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
ACTION_SMOOTH_ALPHA = 0.8    # blend new target with previous (0.4 was too slow — arm froze at mean)

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
            cv2.imwrite("/workspace/policy_cam_step0.png",
                        cv2.cvtColor(policy_img, cv2.COLOR_RGB2BGR))
        if step % 500 == 0 and step > 0:
            print(f"[diag] step {step} action EE-pose={np.round(action[:7], 3)}")

        # ---- ACT ----
        arm_pose      = action[:NUM_ARM_JOINTS]    # policy output is an EE pose (base, xyzw)
        hand_action   = action[NUM_ARM_JOINTS:]
        gripper_cmd   = float(np.mean(hand_action[:3]))
        _hand_state[:] = hand_action               # autoregressive hand state (in-distribution)

        # EE pose -> joint targets via IK; warmstart from the previous solution for continuity.
        arm_joints, ik_ok = kin.ik(arm_pose, _warm)
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
