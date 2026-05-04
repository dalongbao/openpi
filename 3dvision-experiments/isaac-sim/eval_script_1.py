"""
vla_eval.py
Run pi0.5 (checkpoint 29999) on a Franka FR3 in Isaac Sim
Task: Place the plate into the yellow crate.
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

LANGUAGE_COMMAND = "place the plate into the yellow crate"

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
from pxr import Sdf, UsdGeom, Gf  # noqa: Gf still used for Gf.Vec3f in attribute writes

_stage = omni.usd.get_context().get_stage()

# Map of prim path -> local USD file.
# fr3_full has the complete asset including configuration/fr3_robot_schema.usd
# which defines joint limits and damping — critical for stable behaviour.
_PAYLOAD_PATCHES = {
    "/World/fr3":                           "/workspace/assets/fr3_full/fr3.usd",
    "/World/SM_HeavyDutyPackingTable_C02_01": "/workspace/assets/table/SM_HeavyDutyPackingTable_C02_01.usd",
    "/World/plate_small":                   "/workspace/assets/plate/plate_small.usd",
    "/World/SM_Crate_A07_Yellow_01_physics": "/workspace/assets/crate/SM_Crate_A07_Yellow_01_physics.usd",
}

for prim_path, local_usd in _PAYLOAD_PATCHES.items():
    prim = _stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        prim.GetPayloads().ClearPayloads()
        prim.GetPayloads().AddPayload(local_usd)
        print(f"[init] Patched {prim_path} -> {local_usd}")
    else:
        print(f"[WARN] {prim_path} not found in stage — skipping patch")

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


def build_observation(ext_img_uint8, joint_pos):
    state = np.zeros(24, dtype=np.float32)
    state[:7] = joint_pos[:7]
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

try:
    world.reset()
    franka.initialize()          # must re-init after world.reset()
    for _ in range(20):
        world.step(render=True)

    print("[run] Starting evaluation...")
    for step in range(NUM_STEPS):

        # ---- LOOK ----
        policy_img = get_frame(external_cam, POLICY_CAM_RES)
        hd_img     = get_frame(recording_cam, HD_VIDEO_RES)
        joint_pos  = franka.get_joint_positions()
        if joint_pos is None:
            joint_pos = np.zeros(9, dtype=np.float32)

        # ---- THINK ----
        t0 = time.time()
        if last_action_chunk is None or chunk_idx >= len(last_action_chunk):
            obs = build_observation(policy_img, joint_pos)
            with torch.no_grad():
                result = policy.infer(obs)
            last_action_chunk = np.asarray(result["actions"])
            chunk_idx = 0

        action    = last_action_chunk[chunk_idx]
        chunk_idx += 1
        infer_ms  = (time.time() - t0) * 1000

        # ---- ACT ----
        target_joints = action[:NUM_ARM_JOINTS]
        hand_action   = action[NUM_ARM_JOINTS:]
        gripper_cmd   = float(np.mean(hand_action[:3]))

        finger_l, finger_r = to_gripper_positions(gripper_cmd)
        full_cmd = np.zeros(9, dtype=np.float32)
        full_cmd[:NUM_ARM_JOINTS] = target_joints
        full_cmd[7] = finger_l
        full_cmd[8] = finger_r
        franka.apply_action(ArticulationAction(joint_positions=full_cmd))

        # ---- STEP ----
        world.step(render=True)

        # ---- RECORD (HD from RecordingCamera) ----
        video_writer.write(cv2.cvtColor(hd_img, cv2.COLOR_RGB2BGR))

        # ---- LOG ----
        writer.writerow([step, f"{infer_ms:.1f}"] + joint_pos.tolist())
        if step % 50 == 0:
            print(f"[run] step {step:4d} | infer {infer_ms:5.1f}ms | "
                  f"j0-j2 {joint_pos[:3].round(2)}")

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
