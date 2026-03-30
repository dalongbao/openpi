"""Convert Egoverse h5 data to LeRobot dataset v2.0 format.

Data format per h5 file (1 episode):
  observations/images/aria_rgb_cam/color  (T, 480, 640, 3) uint8
  observations/qpos_arm                   (T, 7)  float64
  observations/qpos_hand                  (T, 17) float64
  actions_arm                             (T, 7)  float64
  actions_hand                            (T, 17) float64

Usage:
  # Convert a single task directory:
  uv run 3dvision-experiments/convert_h5_to_lerobot.py \
      --data-dir /cluster/work/cvg/data/Egoverse/raw_timesynced_h5/object_in_bowl_processed_50hz \
      --repo-id egoverse/object_in_bowl \
      --task "put object in bowl"

  # Convert both tasks into one dataset:
  uv run 3dvision-experiments/convert_h5_to_lerobot.py \
      --data-dir /cluster/work/cvg/data/Egoverse/raw_timesynced_h5 \
      --repo-id egoverse/all \
      --multi-task
"""

import dataclasses
from pathlib import Path
import shutil

import h5py
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

try:
    from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME as LEROBOT_HOME
import numpy as np
import tqdm
import tyro

# State = qpos_arm (7) + qpos_hand (17) = 24
# Actions = actions_arm (7) + actions_hand (17) = 24
STATE_DIM = 24
ACTION_DIM = 24
FPS = 50

# Task name mapping from directory names to language prompts.
# Adjust these when you know the exact task instructions.
TASK_PROMPTS = {
    "object_in_bowl_processed_50hz": "put the object in the bowl",
    "bag_groceries": "bag the groceries",
}


@dataclasses.dataclass(frozen=True)
class ConvertConfig:
    # Path to either a single task directory (containing .h5 files)
    # or the parent directory containing multiple task subdirs.
    data_dir: str

    # LeRobot repo_id for the output dataset (stored under $LEROBOT_HOME/<repo_id>).
    repo_id: str = "egoverse/object_in_bowl"

    # If True, treat data_dir as a parent with multiple task subdirs.
    multi_task: bool = False

    # Override the task prompt for single-task conversion.
    task: str | None = None

    # Max episodes to convert per task (None = all). Useful for testing.
    max_episodes: int | None = None

    push_to_hub: bool = False


def create_dataset(repo_id: str) -> LeRobotDataset:
    output_path = LEROBOT_HOME / repo_id
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
        image_writer_threads=10,
        image_writer_processes=5,
    )


EXPECTED_KEYS = [
    "observations/images/aria_rgb_cam/color",
    "observations/qpos_arm",
    "observations/qpos_hand",
    "actions_arm",
    "actions_hand",
]


def convert_episode(dataset: LeRobotDataset, h5_path: Path, task_prompt: str) -> bool:
    """Returns True if the episode was converted, False if skipped."""
    with h5py.File(h5_path, "r") as f:
        # Skip files with missing keys (some h5 files have different structure).
        for key in EXPECTED_KEYS:
            if key not in f:
                print(f"\n  Skipping {h5_path.name}: missing key '{key}'")
                return False

        images = f["observations/images/aria_rgb_cam/color"]
        qpos_arm = f["observations/qpos_arm"][:]
        qpos_hand = f["observations/qpos_hand"][:]
        act_arm = f["actions_arm"][:]
        act_hand = f["actions_hand"][:]

        state = np.concatenate([qpos_arm, qpos_hand], axis=1).astype(np.float32)
        actions = np.concatenate([act_arm, act_hand], axis=1).astype(np.float32)
        num_frames = state.shape[0]

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
    for h5_path in tqdm.tqdm(h5_files, desc=f"  {task_dir.name}"):
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
        # data_dir is the parent; each subdirectory is a task.
        task_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
        for task_dir in task_dirs:
            prompt = TASK_PROMPTS.get(task_dir.name, task_dir.name.replace("_", " "))
            total += convert_task_dir(dataset, task_dir, prompt, config.max_episodes)
    else:
        # data_dir is a single task directory with .h5 files.
        prompt = config.task or TASK_PROMPTS.get(data_dir.name, data_dir.name.replace("_", " "))
        total += convert_task_dir(dataset, data_dir, prompt, config.max_episodes)

    dataset.consolidate()
    print(f"\nDone. Converted {total} episodes to {LEROBOT_HOME / config.repo_id}")

    if config.push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    main(tyro.cli(ConvertConfig))
