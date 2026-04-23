"""Consolidate bag_grocery LeRobot v2 recordings into a single dataset.

Each recording dir is already LeRobot v2 with prestacked actions [100, 12].
This script:
  1. Reads all recording dirs
  2. Unpacks prestacked actions to per-frame (takes actions_cartesian[0] per row)
  3. Renames columns to standard names (image, state, actions)
  4. Transposes CHW images to HWC
  5. Writes one combined LeRobot v2 dataset

Usage:
  uv run python 3dvision-experiments/consolidate_bag_grocery.py \
      --src-dir /cluster/work/cvg/jiaqchen/EGOVERSE_DATA_3DV/bag_grocery \
      --dst-dir /cluster/work/cvg/data/rytsui/lerobot
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from tqdm import tqdm
import tyro

REPO_NAME = "egoverse/bag_grocery"


def main(
    src_dir: str = "/cluster/work/cvg/jiaqchen/EGOVERSE_DATA_3DV/bag_grocery",
    dst_dir: str | None = None,
    max_episodes: int | None = None,
    task: str = "bag the groceries",
):
    if dst_dir is not None:
        import os
        os.environ["HF_LEROBOT_HOME"] = dst_dir

    output_path = Path(dst_dir or HF_LEROBOT_HOME) / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="aria_bimanual",
        fps=30,
        features={
            "image": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (12,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (12,),
                "names": ["actions"],
            },
        },
        image_writer_threads=2,
        image_writer_processes=2,
    )

    src_path = Path(src_dir)
    recording_dirs = sorted([d for d in src_path.iterdir() if d.is_dir() and (d / "meta" / "info.json").exists()])
    print(f"Found {len(recording_dirs)} recording dirs in {src_dir}")

    total_episodes = 0
    skipped_recordings = []

    for rec_dir in tqdm(recording_dirs, desc="Recordings"):
        info = json.loads((rec_dir / "meta" / "info.json").read_text())
        n_episodes = info["total_episodes"]
        if n_episodes == 0:
            skipped_recordings.append(rec_dir.name)
            continue

        # Read all parquet files for this recording.
        chunk_dir = rec_dir / "data" / "chunk-000"
        parquet_files = sorted(chunk_dir.glob("*.parquet"))

        for pf in parquet_files:
            df = pd.read_parquet(pf)
            n_frames = len(df)

            for i in range(n_frames):
                row = df.iloc[i]

                # Image: stored as CHW, convert to HWC uint8.
                img = np.array(row["observations.images.front_img_1"])
                if img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                if np.issubdtype(img.dtype, np.floating):
                    img = (img * 255).astype(np.uint8)

                # State: ee_pose [12].
                state = np.array(row["observations.state.ee_pose"], dtype=np.float32)

                # Actions: prestacked [100, 12] -> take first action [12].
                actions_chunk = np.array(row["actions_cartesian"])
                action = actions_chunk[0].astype(np.float32)

                dataset.add_frame({
                    "image": img,
                    "state": state,
                    "actions": action,
                    "task": task,
                })

            dataset.save_episode()
            total_episodes += 1

            if max_episodes is not None and total_episodes >= max_episodes:
                break

        if max_episodes is not None and total_episodes >= max_episodes:
            break

    print(f"\nDone. Converted {total_episodes} episodes to {output_path}")
    if skipped_recordings:
        print(f"Skipped {len(skipped_recordings)} empty recordings: {skipped_recordings}")


if __name__ == "__main__":
    tyro.cli(main)
