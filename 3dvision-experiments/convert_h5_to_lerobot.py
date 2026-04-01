"""Convert Egoverse h5 data to LeRobot dataset format.

Usage:
  uv run python 3dvision-experiments/convert_h5_to_lerobot.py --data_dir /path/to/h5_dir
  uv run python 3dvision-experiments/convert_h5_to_lerobot.py --data_dir /path/to/h5_dir --max_episodes 5
"""

import shutil

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from tqdm import tqdm
import tyro

REPO_NAME = "egoverse/all"
MAX_FRAMES = 5000  # skip episodes longer than this to avoid OOM

EXPECTED_KEYS = [
    "observations/images/aria_rgb_cam/color",
    "observations/qpos_arm",
    "observations/qpos_hand",
    "actions_arm",
    "actions_hand",
]


def main(data_dir: str, *, max_episodes: int | None = None, task: str = "put the object in the bowl"):
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="franka",
        fps=50,
        features={
            "image": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (24,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (24,),
                "names": ["actions"],
            },
        },
        image_writer_threads=2,
        image_writer_processes=2,
    )

    from pathlib import Path
    h5_files = sorted(Path(data_dir).glob("*.h5"))
    if max_episodes is not None:
        h5_files = h5_files[:max_episodes]

    print(f"Converting {len(h5_files)} episodes...")
    converted = 0

    for h5_path in tqdm(h5_files):
        with h5py.File(h5_path, "r") as f:
            # Skip files with wrong structure.
            if any(key not in f for key in EXPECTED_KEYS):
                print(f"\n  Skipping {h5_path.name}: missing keys")
                continue

            n = f["observations/qpos_arm"].shape[0]
            if n > MAX_FRAMES:
                print(f"\n  Skipping {h5_path.name}: {n} frames > {MAX_FRAMES}")
                continue

            images = f["observations/images/aria_rgb_cam/color"]
            state = np.concatenate(
                [f["observations/qpos_arm"][:], f["observations/qpos_hand"][:]], axis=1
            ).astype(np.float32)
            actions = np.concatenate(
                [f["actions_arm"][:], f["actions_hand"][:]], axis=1
            ).astype(np.float32)

            for i in range(n):
                dataset.add_frame(
                    {
                        "image": images[i],
                        "state": state[i],
                        "actions": actions[i],
                        "task": task,
                    }
                )
            dataset.save_episode()
            converted += 1

    print(f"\nDone. Converted {converted}/{len(h5_files)} episodes to {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
