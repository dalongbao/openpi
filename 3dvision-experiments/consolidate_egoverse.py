"""Consolidate Egoverse LeRobot v2 recordings into a single dataset.

Handles multiple source dirs (for mixing human + teleop data).
Each recording dir has prestacked actions [100, 12] -> unpacked to per-frame.

Usage:
  # Single source (human only):
  uv run python 3dvision-experiments/consolidate_egoverse.py \
      --src-dirs /path/to/human_data \
      --repo-name egoverse/bag_grocery_human \
      --task "bag the groceries"

  # Mixed (human + teleop):
  uv run python 3dvision-experiments/consolidate_egoverse.py \
      --src-dirs /path/to/human_data /path/to/teleop_data \
      --repo-name egoverse/bag_grocery_mix \
      --task "bag the groceries"
"""

import io
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
from tqdm import tqdm
import tyro


def main(
    src_dirs: list[str],
    repo_name: str,
    task: str,
    dst_dir: str = "/cluster/work/cvg/data/Egoverse/lerobot_egoverse",
    action_dim: int = 12,
    max_episodes: int | None = None,
):
    import os
    os.environ["HF_LEROBOT_HOME"] = dst_dir

    # Auto-detect action_dim from first source dir's info.json if not specified.
    if action_dim == 12:
        first_src = Path(src_dirs[0])
        for d in sorted(first_src.iterdir()):
            info_path = d / "meta" / "info.json"
            if info_path.exists():
                info = json.loads(info_path.read_text())
                detected = info["features"]["observations.state.ee_pose"]["shape"][0]
                if detected != action_dim:
                    print(f"Auto-detected action_dim={detected} from {d.name}")
                    action_dim = detected
                break

    robot_type = "aria_right_arm" if action_dim == 6 else "aria_bimanual"

    output_path = Path(dst_dir) / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type=robot_type,
        fps=30,
        features={
            "image": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["actions"],
            },
        },
        image_writer_threads=2,
        image_writer_processes=2,
    )

    total_episodes = 0
    skipped = []

    for src_dir in src_dirs:
        src_path = Path(src_dir)
        recording_dirs = sorted([
            d for d in src_path.iterdir()
            if d.is_dir() and (d / "meta" / "info.json").exists()
        ])
        print(f"\n=== Source: {src_dir} ({len(recording_dirs)} recordings) ===")

        for rec_dir in tqdm(recording_dirs, desc=src_path.name):
            info = json.loads((rec_dir / "meta" / "info.json").read_text())
            if info["total_episodes"] == 0:
                skipped.append(rec_dir.name)
                continue

            chunk_dir = rec_dir / "data" / "chunk-000"
            parquet_files = sorted(chunk_dir.glob("*.parquet"))

            for pf in parquet_files:
                try:
                    table = pq.read_table(pf)
                except Exception as e:
                    print(f"\n  Skipping corrupt file {pf}: {e}")
                    continue

                for i in range(len(table)):
                    # Image
                    img_raw = table.column("observations.images.front_img_1")[i].as_py()
                    if isinstance(img_raw, dict) and "bytes" in img_raw:
                        img = np.array(Image.open(io.BytesIO(img_raw["bytes"])).convert("RGB"))
                    elif isinstance(img_raw, bytes):
                        img = np.array(Image.open(io.BytesIO(img_raw)).convert("RGB"))
                    else:
                        img = np.array(img_raw)
                        if img.ndim == 3 and img.shape[0] == 3:
                            img = np.transpose(img, (1, 2, 0))
                        if np.issubdtype(img.dtype, np.floating):
                            img = (img * 255).astype(np.uint8)

                    # State
                    state = np.array(table.column("observations.state.ee_pose")[i].as_py(), dtype=np.float32)

                    # Actions: prestacked [100, 12] -> per-frame [12]
                    actions_chunk = np.array(table.column("actions_cartesian")[i].as_py())
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
        if max_episodes is not None and total_episodes >= max_episodes:
            break

    print(f"\nDone. {total_episodes} episodes -> {output_path}")
    if skipped:
        print(f"Skipped {len(skipped)} empty: {skipped}")


if __name__ == "__main__":
    tyro.cli(main)
