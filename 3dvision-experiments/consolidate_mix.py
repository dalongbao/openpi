"""Consolidate human (12-dim Cartesian) + teleop (48-dim joint) into one mixed LeRobot dataset.

Per-sample layout in the unified action/state vector (length 60):
  [0:12]  human Cartesian ee_pose (left 6 + right 6)
  [12:26] teleop arm qpos (7 + 7)
  [26:60] teleop hand qpos (17 + 17)

Per-sample action_mask (bool[60]):
  human:  [True]*12 + [False]*48
  teleop: [False]*12 + [True]*48

Both branches keep the SAME image key ("image"), single Aria-style camera.
"""

import io
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm
import tyro


UNIFIED_DIM = 60
CART_DIM = 12  # 6 left + 6 right
ARM_DIM = 14
HAND_DIM = 34
SRC_FPS_TELEOP = 50
DST_FPS = 30


def _resample_indices(n_src: int) -> np.ndarray:
    n_dst = int(round(n_src * DST_FPS / SRC_FPS_TELEOP))
    if n_dst <= 0:
        return np.zeros(0, dtype=np.int64)
    return np.round(np.linspace(0, n_src - 1, n_dst)).astype(np.int64)


def _decode_human_image(raw) -> np.ndarray:
    if isinstance(raw, dict) and "bytes" in raw:
        return np.array(Image.open(io.BytesIO(raw["bytes"])).convert("RGB"))
    if isinstance(raw, bytes):
        return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    arr = np.asarray(raw)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating):
        arr = (arr * 255).astype(np.uint8)
    return arr


def _decode_teleop_image(raw) -> np.ndarray:
    if isinstance(raw, bytes):
        return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    arr = np.asarray(raw)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating):
        arr = (arr * 255).astype(np.uint8)
    return arr


def _add_human_episodes(dataset: LeRobotDataset, src_dir: str, task: str, max_episodes: int | None):
    src_path = Path(src_dir)
    recording_dirs = sorted(
        d for d in src_path.iterdir() if d.is_dir() and (d / "meta" / "info.json").exists()
    )
    print(f"\n=== HUMAN source: {src_dir} ({len(recording_dirs)} recordings) ===")

    human_mask = np.array([1.0] * CART_DIM + [0.0] * (UNIFIED_DIM - CART_DIM), dtype=np.float32)
    n_episodes = 0

    for rec_dir in tqdm(recording_dirs, desc="human"):
        info = json.loads((rec_dir / "meta" / "info.json").read_text())
        if info["total_episodes"] == 0:
            continue
        chunk_dir = rec_dir / "data" / "chunk-000"
        for pf in sorted(chunk_dir.glob("*.parquet")):
            try:
                table = pq.read_table(pf)
            except Exception as e:
                print(f"  corrupt {pf}: {e}")
                continue

            for i in range(len(table)):
                img = _decode_human_image(table.column("observations.images.front_img_1")[i].as_py())
                cart_state = np.asarray(
                    table.column("observations.state.ee_pose")[i].as_py(), dtype=np.float32
                )
                actions_chunk = np.asarray(table.column("actions_cartesian")[i].as_py())
                cart_action = actions_chunk[0].astype(np.float32)

                state = np.zeros(UNIFIED_DIM, dtype=np.float32)
                state[:CART_DIM] = cart_state[:CART_DIM]
                action = np.zeros(UNIFIED_DIM, dtype=np.float32)
                action[:CART_DIM] = cart_action[:CART_DIM]

                dataset.add_frame(
                    {"image": img, "state": state, "actions": action, "action_mask": human_mask, "task": task}
                )

            dataset.save_episode()
            n_episodes += 1
            if max_episodes is not None and n_episodes >= max_episodes:
                return n_episodes
    return n_episodes


def _add_teleop_episodes(dataset: LeRobotDataset, src_dir: str, task: str, max_episodes: int | None):
    h5_files = sorted(Path(src_dir).glob("*.h5"))
    print(f"\n=== TELEOP source: {src_dir} ({len(h5_files)} episodes) ===")

    teleop_mask = np.array([0.0] * CART_DIM + [1.0] * (UNIFIED_DIM - CART_DIM), dtype=np.float32)
    n_episodes = 0

    for h5_path in tqdm(h5_files, desc="teleop"):
        try:
            with h5py.File(h5_path, "r") as f:
                arm_l = np.asarray(f["actions_arm_left"])
                arm_r = np.asarray(f["actions_arm_right"])
                hand_l = np.asarray(f["actions_hand_left"])
                hand_r = np.asarray(f["actions_hand_right"])
                qarm_l = np.asarray(f["observations/qpos_arm_left"])
                qarm_r = np.asarray(f["observations/qpos_arm_right"])
                qhand_l = np.asarray(f["observations/qpos_hand_left"])
                qhand_r = np.asarray(f["observations/qpos_hand_right"])
                images = f["observations/images/aria_rgb_cam/color"]

                T = arm_l.shape[0]
                if T == 0:
                    continue

                for i in _resample_indices(T):
                    img = _decode_teleop_image(images[int(i)])
                    joint_state = np.concatenate(
                        [qarm_l[int(i)], qarm_r[int(i)], qhand_l[int(i)], qhand_r[int(i)]]
                    ).astype(np.float32)
                    joint_action = np.concatenate(
                        [arm_l[int(i)], arm_r[int(i)], hand_l[int(i)], hand_r[int(i)]]
                    ).astype(np.float32)

                    state = np.zeros(UNIFIED_DIM, dtype=np.float32)
                    state[CART_DIM:] = joint_state
                    action = np.zeros(UNIFIED_DIM, dtype=np.float32)
                    action[CART_DIM:] = joint_action

                    dataset.add_frame(
                        {"image": img, "state": state, "actions": action, "action_mask": teleop_mask, "task": task}
                    )

                dataset.save_episode()
                n_episodes += 1
                if max_episodes is not None and n_episodes >= max_episodes:
                    return n_episodes
        except Exception as e:
            print(f"  skip {h5_path.name}: {e}")
    return n_episodes


def main(
    human_dir: str = "/cluster/work/cvg/jiaqchen/EGOVERSE_DATA_3DV/bag_grocery",
    teleop_dir: str = "/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries",
    repo_name: str = "egoverse/bag_grocery_mix",
    task: str = "bag the groceries",
    dst_dir: str = "/cluster/work/cvg/data/Egoverse/lerobot_egoverse",
    max_human_episodes: int | None = None,
    max_teleop_episodes: int | None = None,
):
    import os
    os.environ["HF_LEROBOT_HOME"] = dst_dir

    output_path = Path(dst_dir) / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="aria_bimanual_mix",
        fps=DST_FPS,
        features={
            "image": {"dtype": "image", "shape": (480, 640, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (UNIFIED_DIM,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (UNIFIED_DIM,), "names": ["actions"]},
            "action_mask": {"dtype": "float32", "shape": (UNIFIED_DIM,), "names": ["action_mask"]},
        },
        image_writer_threads=2,
        image_writer_processes=2,
    )

    n_h = _add_human_episodes(dataset, human_dir, task, max_human_episodes)
    n_t = _add_teleop_episodes(dataset, teleop_dir, task, max_teleop_episodes)

    print(f"\nDone. human={n_h}, teleop={n_t}, total={n_h + n_t} -> {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
