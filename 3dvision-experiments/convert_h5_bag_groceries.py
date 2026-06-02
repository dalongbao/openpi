"""Convert bimanual bag_groceries h5 -> LeRobot v2 with 48-dim joint actions.

h5 keys per episode:
  actions_arm_left   (T, 7)
  actions_arm_right  (T, 7)
  actions_hand_left  (T, 17)
  actions_hand_right (T, 17)
  observations/qpos_arm_left/right, qpos_hand_left/right (same shapes)
  observations/images/aria_rgb_cam/color (T, H, W, 3) uint8

Output features:
  image     : aria_rgb_cam frame -> (H, W, 3) uint8
  state     : qpos_arm_left ++ qpos_arm_right ++ qpos_hand_left ++ qpos_hand_right = 48 dims
  actions   : same layout for actions_*

We resample 50Hz -> 30Hz so the dataset's fps matches bag_grocery_human.
"""

import io
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm
import tyro


SRC_FPS = 50
DST_FPS = 30
STATE_DIM = 7 + 7 + 17 + 17  # 48
ACTION_DIM = STATE_DIM


def _resample_indices(n_src: int) -> np.ndarray:
    n_dst = int(round(n_src * DST_FPS / SRC_FPS))
    if n_dst <= 0:
        return np.zeros(0, dtype=np.int64)
    return np.round(np.linspace(0, n_src - 1, n_dst)).astype(np.int64)


def _decode_image(raw) -> np.ndarray:
    if isinstance(raw, bytes):
        return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    arr = np.asarray(raw)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating):
        arr = (arr * 255).astype(np.uint8)
    return arr


def main(
    src_dir: str = "/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries",
    repo_name: str = "egoverse/bag_grocery_teleop",
    task: str = "bag the groceries",
    dst_dir: str = "/cluster/work/cvg/data/Egoverse/lerobot_egoverse",
    max_episodes: int | None = None,
):
    import os
    os.environ["HF_LEROBOT_HOME"] = dst_dir

    output_path = Path(dst_dir) / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="aria_bimanual_joint",
        fps=DST_FPS,
        features={
            "image": {"dtype": "image", "shape": (480, 640, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (STATE_DIM,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (ACTION_DIM,), "names": ["actions"]},
        },
        image_writer_threads=2,
        image_writer_processes=2,
    )

    h5_files = sorted(Path(src_dir).glob("*.h5"))
    print(f"Found {len(h5_files)} h5 files in {src_dir}")

    n_episodes = 0
    skipped = []

    for h5_path in tqdm(h5_files, desc="episodes"):
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
                    skipped.append(h5_path.name)
                    continue

                idxs = _resample_indices(T)
                for i in idxs:
                    img = _decode_image(images[int(i)])
                    state = np.concatenate(
                        [qarm_l[int(i)], qarm_r[int(i)], qhand_l[int(i)], qhand_r[int(i)]]
                    ).astype(np.float32)
                    action = np.concatenate(
                        [arm_l[int(i)], arm_r[int(i)], hand_l[int(i)], hand_r[int(i)]]
                    ).astype(np.float32)
                    dataset.add_frame({"image": img, "state": state, "actions": action, "task": task})

                dataset.save_episode()
                n_episodes += 1

                if max_episodes is not None and n_episodes >= max_episodes:
                    break
        except Exception as e:
            print(f"  skip {h5_path.name}: {e}")
            skipped.append(h5_path.name)

    print(f"Done. {n_episodes} episodes -> {output_path}")
    if skipped:
        print(f"Skipped {len(skipped)}: {skipped}")


if __name__ == "__main__":
    tyro.cli(main)
