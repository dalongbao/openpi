"""
vla_eval.py
Run pi0.5 (checkpoint 29999) on a Franka FR3 in Isaac Sim
Task: Place the plate into the yellow crate.
"""

# === MUST BE THE ABSOLUTE FIRST ISAAC-RELATED LINE ===
from isaacsim import SimulationApp

CONFIG = {
    "headless": True,     # No GUI on the cluster
    "livestream": 2,      # WebRTC streaming so we can tunnel in and watch
    "width": 1280,
    "height": 720,
}
simulation_app = SimulationApp(CONFIG)
# ======================================================

# NOW the rest of the imports are safe
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
CHECKPOINT_DIR = "/checkpoints/pi05_egoverse/test/29999"  # mounted via apptainer --bind
RESULTS_CSV    = "/workspace/results.csv"
VIDEO_PATH     = "/workspace/evaluation.mp4"

# The actual language command the model receives
LANGUAGE_COMMAND = "place the plate into the yellow crate"

NUM_STEPS         = 3000   # ~60s at 50Hz
INFERENCE_EVERY   = 1      # how often to re-run the model (1 = every step)
CAMERA_RES        = (224, 224)  # pi0.5 typically uses 224x224

# Joint order assumed: [j1..j7, finger1, finger2]. Verified after first run.
NUM_ARM_JOINTS = 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[init] device = {device}")


# --------------------------------------------------------------------
# LOAD THE pi0.5 POLICY
# --------------------------------------------------------------------
# NOTE: The exact import path depends on which pi0.5 codebase you pulled.
# This uses the openpi convention. Adjust if your repo is different.
sys.path.insert(0, "/workspace/openpi/src")
sys.path.insert(0, "/workspace/openpi/packages/openpi-client/src")
sys.path.insert(0, "/isaac_packages")

try:
    from openpi.policies import policy_config
    from openpi.shared import normalize
    from openpi.training import config as _config

    cfg = _config.get_config("pi05_egoverse")
    # Override assets_base_dir so container finds norm stats at /workspace/openpi/assets/
    cfg = dataclasses.replace(cfg, assets_base_dir="/workspace/openpi/assets")
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    norm_stats = normalize.load(cfg.assets_dirs / data_cfg.repo_id)

    policy = policy_config.create_trained_policy(
        cfg,
        CHECKPOINT_DIR,
        norm_stats=norm_stats,
        default_prompt=LANGUAGE_COMMAND,
    )
    print(f"[init] Loaded pi0.5 from {CHECKPOINT_DIR}")
except Exception as e:
    print(f"[FATAL] Could not load policy: {e}")
    traceback.print_exc()
    simulation_app.close()
    sys.exit(1)


# --------------------------------------------------------------------
# LOAD THE SCENE
# --------------------------------------------------------------------
print(f"[init] Opening stage {USD_PATH}")
open_stage(usd_path=USD_PATH)

world = World(stage_units_in_meters=1.0)
world.reset()

# --- Robot ---
franka = Articulation(prim_path="/World/Franka", name="franka")
franka.initialize()
print(f"[init] Franka has {franka.num_dof} DOF")  # Expect 9

# --- Cameras ---
external_cam = Camera(
    prim_path="/World/ExternalCamera",
    resolution=CAMERA_RES,
)
wrist_cam = Camera(
    prim_path="/World/fr3/fr3_hand/WristCamera",
    resolution=CAMERA_RES,
)
external_cam.initialize()
wrist_cam.initialize()

# Warm up the renderer so camera buffers get filled
print("[init] Warming up cameras...")
for _ in range(15):
    world.step(render=True)


# --------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------
def prepare_image(raw_rgba):
    """Isaac's (H,W,4) uint8 -> (H,W,3) uint8 for pi0.5 (policy normalizes internally)."""
    if raw_rgba is None or raw_rgba.size == 0:
        raise RuntimeError("Camera returned empty frame")
    return raw_rgba[:, :, :3]


def build_observation(ext_img_uint8, joint_pos):
    """Pack observations in the flat-key format egoverse_policy expects."""
    state = np.zeros(24, dtype=np.float32)
    state[:7] = joint_pos[:7]  # arm joints; hand dims padded with zeros
    return {
        "observation/image": ext_img_uint8,
        "observation/state": state,
        "prompt": LANGUAGE_COMMAND,
    }


def to_gripper_positions(gripper_cmd):
    """Map pi0.5's gripper output (0=open, 1=closed) to Franka finger positions."""
    gripper_cmd = float(np.clip(gripper_cmd, 0.0, 1.0))
    finger_pos = 0.04 * (1.0 - gripper_cmd)  # 0.04m=open, 0m=closed
    return finger_pos, finger_pos


# --------------------------------------------------------------------
# MAIN LOOP
# --------------------------------------------------------------------
csv_file = open(RESULTS_CSV, "w", newline="")
writer   = csv.writer(csv_file)
writer.writerow(["step", "infer_ms"] + [f"j{i}" for i in range(9)])

video_writer = cv2.VideoWriter(
    VIDEO_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    50,  # fps matches 50Hz sim
    CAMERA_RES,
)

last_action_chunk = None
chunk_idx = 0

try:
    world.reset()
    for _ in range(5):
        world.step(render=True)  # post-reset warmup

    print("[run] Starting evaluation...")
    for step in range(NUM_STEPS):

        # ---- LOOK ----
        ext_img   = prepare_image(external_cam.get_rgba())
        joint_pos = franka.get_joint_positions()

        # ---- THINK ----
        t0 = time.time()
        if last_action_chunk is None or chunk_idx >= len(last_action_chunk):
            obs = build_observation(ext_img, joint_pos)
            with torch.no_grad():
                result = policy.infer(obs)
            # pi0.5 returns an action chunk; shape (horizon, 24) for egoverse
            last_action_chunk = np.asarray(result["actions"])
            chunk_idx = 0

        action = last_action_chunk[chunk_idx]
        chunk_idx += 1
        infer_ms = (time.time() - t0) * 1000

        # ---- ACT ----
        # egoverse policy outputs 24 dims: 7 arm + 17 hand.
        # Map to Franka's 9 DOF: use dims 0-6 for arm, collapse hand dims to gripper.
        target_joints = action[:NUM_ARM_JOINTS]            # dims 0–6
        hand_action   = action[NUM_ARM_JOINTS:]            # dims 7–23 (17 hand joints)
        gripper_cmd   = float(np.mean(hand_action[:3]))   # average first 3 hand dims as open/close proxy

        # Assume absolute targets by default. If robot doesn't move, try delta:
        # target_joints = joint_pos[:NUM_ARM_JOINTS] + target_joints

        finger_l, finger_r = to_gripper_positions(gripper_cmd)
        full_cmd = np.zeros(9, dtype=np.float32)
        full_cmd[:NUM_ARM_JOINTS] = target_joints
        full_cmd[7] = finger_l
        full_cmd[8] = finger_r

        franka.apply_action(ArticulationAction(joint_positions=full_cmd))

        # ---- STEP ----
        world.step(render=True)

        # ---- RECORD ----
        video_writer.write(cv2.cvtColor(ext_img, cv2.COLOR_RGB2BGR))

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