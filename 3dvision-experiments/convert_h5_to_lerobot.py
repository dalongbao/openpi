"""Convert Egoverse h5 data to LeRobot dataset format.

Data format per h5 file (1 episode):
  observations/images/aria_rgb_cam/color  (T, 480, 640, 3) uint8
  observations/qpos_arm                   (T, 7)  float64
  observations/qpos_hand                  (T, 17) float64
  actions_arm                             (T, 7)  float64
  actions_hand                            (T, 17) float64

Usage:
  uv run python 3dvision-experiments/convert_h5_to_lerobot.py \
      --data-dir /path/to/object_in_bowl_processed_50hz \
      --repo-id egoverse/all \
      --task "put the object in the bowl"
"""

import dataclasses
import os
from pathlib import Path
import shutil

import h5py
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from tqdm import tqdm
import tyro

STATE_DIM = 24  # qpos_arm (7) + qpos_hand (17)
ACTION_DIM = 24  # actions_arm (7) + actions_hand (17)
FPS = 50

# Skip episodes longer than this to avoid OOM during parquet write.
MAX_FRAMES = 5000

TASK_PROMPTS = {
    "object_in_bowl_processed_50hz": "put the object in the bowl",
    "bag_groceries": "bag the groceries",
}

EXPECTED_KEYS = [
    "observations/images/aria_rgb_cam/color",
    "observations/qpos_arm",
    "observations/qpos_hand",
    "actions_arm",
    "actions_hand",
]

OUTPUT_DIR = Path(os.environ.get("HF_LEROBOT_HOME", os.path.expanduser("~/.cache/huggingface/lerobot")))


@dataclasses.dataclass(frozen=True)
class ConvertConfig:
    data_dir: str
    repo_id: str = "egoverse/all"
    task: str | None = None
    multi_task: bool = False
    max_episodes: int | None = None


def create_dataset(repo_id: str) -> LeRobotDataset:
    output_path = OUTPUT_DIR / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        robot_type="franka",
        features={
            "image": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (STATE_DIM,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (ACTION_DIM,),
                "names": ["actions"],
            },
        },
        image_writer_threads=2,
        image_writer_processes=2,
    )


def convert_episode(dataset: LeRobotDataset, h5_path: Path, task_prompt: str) -> bool:
    try:
        with h5py.File(h5_path, "r") as f:
            for key in EXPECTED_KEYS:
                if key not in f:
                    print(f"\n  Skipping {h5_path.name}: missing key '{key}'")
                    return False

            num_frames = f["observations/qpos_arm"].shape[0]
            if num_frames > MAX_FRAMES:
                print(f"\n  Skipping {h5_path.name}: {num_frames} frames exceeds {MAX_FRAMES} limit")
                return False

            images = f["observations/images/aria_rgb_cam/color"]
            qpos_arm = f["observations/qpos_arm"][:]
            qpos_hand = f["observations/qpos_hand"][:]
            act_arm = f["actions_arm"][:]
            act_hand = f["actions_hand"][:]

            state = np.concatenate([qpos_arm, qpos_hand], axis=1).astype(np.float32)
            actions = np.concatenate([act_arm, act_hand], axis=1).astype(np.float32)

            for i in range(num_frames):
                dataset.add_frame(
                    {
                        "image": images[i],
                        "state": state[i],
                        "actions": actions[i],
                        "task": task_prompt,
                    }
                )

        dataset.save_episode()
        return True

    except Exception as e:
        print(f"\n  Error processing {h5_path.name}: {e}")
        return False


def convert_task_dir(dataset: LeRobotDataset, task_dir: Path, task_prompt: str, max_episodes: int | None) -> int:
    h5_files = sorted(task_dir.glob("*.h5"))
    if not h5_files:
        print(f"  No .h5 files found in {task_dir}")
        return 0

    if max_episodes is not None:
        h5_files = h5_files[:max_episodes]

    print(f"  Converting {len(h5_files)} episodes from {task_dir.name} (task: '{task_prompt}')")
    converted = 0
    skipped = 0
    for h5_path in tqdm(h5_files, desc=f"  {task_dir.name}"):
        if convert_episode(dataset, h5_path, task_prompt):
            converted += 1
        else:
            skipped += 1

    if skipped:
        print(f"  Skipped {skipped}/{len(h5_files)} episodes with incompatible format")
    return converted


def main(config: ConvertConfig) -> None:
    data_dir = Path(config.data_dir)
    dataset = create_dataset(config.repo_id)
    total = 0

    if config.multi_task:
        task_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
        for task_dir in task_dirs:
            prompt = TASK_PROMPTS.get(task_dir.name, task_dir.name.replace("_", " "))
            total += convert_task_dir(dataset, task_dir, prompt, config.max_episodes)
    else:
        prompt = config.task or TASK_PROMPTS.get(data_dir.name, data_dir.name.replace("_", " "))
        total += convert_task_dir(dataset, data_dir, prompt, config.max_episodes)

    print(f"\nDone. Converted {total} episodes to {OUTPUT_DIR / config.repo_id}")


if __name__ == "__main__":
    main(tyro.cli(ConvertConfig))
