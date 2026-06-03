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
from pxr import Sdf, UsdGeom, Gf, UsdPhysics  # noqa: Gf still used for Gf.Vec3f in attribute writes

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


def _add_cube_object(stage, path, pos, size=0.06):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(size)
    UsdGeom.Xformable(cube).AddTranslateOp().Set(Gf.Vec3d(*pos))
    prim = cube.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)            # analytic box collider
    UsdPhysics.RigidBodyAPI.Apply(prim)            # dynamic
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(0.05)
    cube.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.2, 0.15)])
    print(f"[init] Added cube object at {path} {pos}")


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
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.80, 0.70)])
    print(f"[init] Added bowl at {path} {pos}")


_add_cube_object(_stage, "/World/object", _OBJECT_POS)
_add_bowl(_stage, "/World/bowl", _BOWL_POS)

world = World(stage_units_in_meters=1.0)
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



# Joint permutation: Isaac Sim FR3 vs Egoverse training data differ in dims 3 and 5.
# Isaac Sim idx3 = fr3_joint4 (elbow, always negative, limits [-3.04, -0.152] rad)
# Isaac Sim idx5 = fr3_joint6 (wrist, always positive, limits [0.544, 4.52] rad)
# Training dim3 is always strongly positive (~0.69-1.00) → corresponds to FR3 j6 (sim idx5)
# Training dim5 is mostly negative (~-0.43 to 0.19) → corresponds to FR3 j4 (sim idx3)
# Permutation: sim[0,1,2,3,4,5,6] ↔ training[0,1,2,5,4,3,6]
_SIM_TO_TRAIN = [0, 1, 2, 5, 4, 3, 6]   # sim_idx -> training_dim
_TRAIN_TO_SIM = [0, 1, 2, 5, 4, 3, 6]   # training_dim -> sim_idx (same permutation: self-inverse)


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
    state = np.zeros(24, dtype=np.float32)
    for sim_idx, train_dim in enumerate(_SIM_TO_TRAIN):
        state[train_dim] = joint_pos[sim_idx]
    state[7:24] = _hand_state    # use prev predicted hand action, not zeros
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
writer.writerow(["step", "infer_ms"] + [f"j{i}" for i in range(9)])

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

    # Move the robot to the training home position before running the policy.
    # dim-to-sim permutation: [0,1,2,5,4,3,6] so training dims [0,1,2,3,4,5,6]
    # map to sim indices [0,1,2,5,4,3,6].
    # FR3 j4 (sim idx3) has upper limit -0.152; dim5 home ≈ -0.025 so clip to -0.152.
    # FR3 j6 (sim idx5) home = dim3 ≈ 0.994.
    HOME_TRAIN = np.array([0.476, 0.120, 0.292, 0.994, 0.105, -0.025, -0.009], np.float32)
    home_cmd = np.zeros(9, dtype=np.float32)
    for train_dim, sim_idx in enumerate(_TRAIN_TO_SIM):
        home_cmd[sim_idx] = HOME_TRAIN[train_dim]
    home_cmd[7] = 0.02
    home_cmd[8] = 0.02
    print(f"[init] Moving to home: sim cmds = {home_cmd[:7].round(3)}")
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
            print(f"[diag] chunk shape={last_action_chunk.shape}  arm actions (first 3 chunks):")
            for _ci in range(min(3, len(last_action_chunk))):
                _a = last_action_chunk[_ci]
                print(f"  chunk[{_ci}] arm={np.round(_a[:7], 3)}")
            cv2.imwrite("/workspace/policy_cam_step0.png",
                        cv2.cvtColor(policy_img, cv2.COLOR_RGB2BGR))
        if step % 500 == 0 and step > 0:
            print(f"[diag] step {step} action train_arm={np.round(action[:7], 3)}")

        # ---- ACT ----
        hand_action   = action[NUM_ARM_JOINTS:]
        gripper_cmd   = float(np.mean(hand_action[:3]))

        # Update hand state for next observation (autoregressive — keeps state in-distribution)
        _hand_state[:] = hand_action

        finger_l, finger_r = to_gripper_positions(gripper_cmd)
        full_cmd = np.zeros(9, dtype=np.float32)
        for train_dim, sim_idx in enumerate(_TRAIN_TO_SIM):
            full_cmd[sim_idx] = action[train_dim]
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

        # ---- LOG ----
        writer.writerow([step, f"{infer_ms:.1f}"] + joint_pos.tolist())
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
