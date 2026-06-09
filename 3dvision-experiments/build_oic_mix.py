"""build_oic_mix.py — assemble the unified R-ID + H-ID co-training dataset `egoverse/oic_mix`.

Puts robot teleop (R-ID) and human video (H-ID) in ONE 24D action space so a single pi0.5 head
can co-train. Unified frame = ROBOT BASE (so robot data + ablation_eval.py stay untouched; only
the human is moved). See `openpi.policies.egoverse_unify` for the math (validated self-test).

  ROBOT (object_in_bowl h5): native 24D base frame [pos, quat xyzw, hand17] -> IDENTITY,
                             action_mask = [1]*24 (everything supervised).
  HUMAN (oic_human LeRobot 6D head frame [pos, yaw,pitch,roll]) -> base 24D via egoverse_unify
                             (euler ZYX->quat, head->base remap, 17 hand slots = 0),
                             action_mask = [1]*7 + [0]*17 (human never supervises the hand).

Features written: image(480,640,3) uint8, state(24) f32, actions(24) f32, action_mask(24) f32, task.
EgoverseUnifiedInputs forwards `action_mask` as the model's per-dim loss mask.

Run in the openpi uv env (has lerobot):
  export HF_LEROBOT_HOME=/cluster/scratch/lichin/lerobot UV_FROZEN=1
  uv run python 3dvision-experiments/build_oic_mix.py \
    --robot_dir /cluster/work/cvg/data/Egoverse/raw_timesynced_h5/object_in_bowl_processed_50hz \
    --human_root /cluster/work/cvg/data/Egoverse/lerobot_egoverse/egoverse/oic_human \
    --held_out_file held_out_rid.txt
  # SMOKE-TEST FIRST (verifies image decode + shapes on 1+1 episodes):
  #   ... --max_robot 1 --max_human 1

Caveats: robot is 50 Hz, human 30 Hz — the action_horizon chunk spans different real time per
source (minor; flag in the writeup). Output goes to $HF_LEROBOT_HOME/<repo_id>.
"""
import glob
import io
import pathlib
import shutil

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pandas as pd
from PIL import Image
import tyro

from openpi.policies import egoverse_unify as U

ROBOT_KEYS = ["observations/images/aria_rgb_cam/color", "observations/qpos_arm",
              "observations/qpos_hand", "actions_arm", "actions_hand"]


def _decode_image(cell, root: pathlib.Path) -> np.ndarray:
    """Decode a LeRobot 'image' parquet cell: embedded bytes, {'bytes'/'path'} struct, path str, or array."""
    if isinstance(cell, dict):
        if cell.get("bytes") is not None:
            return np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))
        cell = cell.get("path")
    if isinstance(cell, (bytes, bytearray)):
        return np.asarray(Image.open(io.BytesIO(cell)).convert("RGB"))
    if isinstance(cell, str):
        p = pathlib.Path(cell)
        if not p.is_absolute():
            p = root / cell
        return np.asarray(Image.open(p).convert("RGB"))
    arr = np.asarray(cell)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def main(
    robot_dir: str = "/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/object_in_bowl_processed_50hz",
    human_root: str = "/cluster/work/cvg/data/Egoverse/lerobot_egoverse/egoverse/oic_human",
    repo_id: str = "egoverse/oic_mix",
    held_out_file: str | None = None,           # text file of robot h5 basenames to EXCLUDE (the eval set)
    robot_max_frames: int = 10_000,
    max_robot: int | None = None,
    max_human: int | None = None,
    low_mem: bool = False,             # synchronous image writing -> bounded memory, cannot OOM (slower)
    robot_task: str = "put the object in the bowl",
    human_task: str = "put the object in the bowl",
):
    out = HF_LEROBOT_HOME / repo_id
    if out.exists():
        shutil.rmtree(out)
    wt, wp = (0, 0) if low_mem else (4, 4)
    ds = LeRobotDataset.create(
        repo_id=repo_id, robot_type="franka", fps=50,
        features={
            "image": {"dtype": "image", "shape": (480, 640, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (U.ACTION_DIM,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (U.ACTION_DIM,), "names": ["actions"]},
            "action_mask": {"dtype": "float32", "shape": (U.ACTION_DIM,), "names": ["action_mask"]},
        },
        image_writer_threads=wt, image_writer_processes=wp,
    )

    # ---------- ROBOT (R-ID): identity, full mask ----------
    held = set(l.strip() for l in open(held_out_file)) if held_out_file else set()
    h5s = sorted(pathlib.Path(robot_dir).glob("*.h5"))
    if max_robot is not None:
        h5s = h5s[:max_robot]
    n_rob = 0
    for h5_path in h5s:
        if h5_path.name in held:
            print(f"  [robot] HELD-OUT, skip {h5_path.name}"); continue
        with h5py.File(h5_path, "r") as f:
            if any(k not in f for k in ROBOT_KEYS):
                print(f"  [robot] missing keys, skip {h5_path.name}"); continue
            n = f["observations/qpos_arm"].shape[0]
            if n > robot_max_frames:
                print(f"  [robot] {n}>{robot_max_frames} frames, skip {h5_path.name}"); continue
            imgs = f["observations/images/aria_rgb_cam/color"]
            state = np.concatenate([f["observations/qpos_arm"][:], f["observations/qpos_hand"][:]], axis=1).astype(np.float32)
            act = np.concatenate([f["actions_arm"][:], f["actions_hand"][:]], axis=1).astype(np.float32)
            mask = np.ones((U.ACTION_DIM,), dtype=np.float32)
            for i in range(n):
                ds.add_frame({"image": np.asarray(imgs[i]), "state": state[i], "actions": act[i],
                              "action_mask": mask, "task": robot_task})
            ds.save_episode(); n_rob += 1
            print(f"  [robot] +{h5_path.name} ({n} frames)")

    # ---------- HUMAN (H-ID): 6D head -> 24D base, arm-only mask ----------
    pqs = sorted(glob.glob(str(pathlib.Path(human_root) / "data" / "**" / "*.parquet"), recursive=True))
    if max_human is not None:
        pqs = pqs[:max_human]
    root = pathlib.Path(human_root)
    human_mask = np.concatenate([np.ones(U.ARM_DIM, np.float32), np.zeros(U.HAND_DIM, np.float32)])
    n_hum = 0
    for pq in pqs:
        df = pd.read_parquet(pq)
        if "frame_index" in df.columns:
            df = df.sort_values("frame_index")
        state6 = np.stack([np.asarray(v, np.float64).reshape(-1)[:6] for v in df["state"].to_list()])
        act6 = np.stack([np.asarray(v, np.float64).reshape(-1)[:6] for v in df["actions"].to_list()])
        state24 = U.to_unified(U.human6d_to_base_arm7(state6), None)[0]
        act24 = U.to_unified(U.human6d_to_base_arm7(act6), None)[0]
        for i, cell in enumerate(df["image"].to_list()):
            ds.add_frame({"image": _decode_image(cell, root), "state": state24[i], "actions": act24[i],
                          "action_mask": human_mask, "task": human_task})
        ds.save_episode(); n_hum += 1
        if n_hum % 100 == 0 or max_human:
            print(f"  [human] +{pathlib.Path(pq).name} ({len(df)} frames)  [{n_hum} eps]")

    print(f"\nDone. {n_rob} robot + {n_hum} human episodes -> {out}")
    print("Next: compute norm stats for the mix config, then train pi05_ego_mix_oic.")


if __name__ == "__main__":
    tyro.cli(main)
