"""Baseline inference for pi0.5 on Egoverse h5 episodes.

Loads the pre-trained pi0.5 base weights (the same ones `pi05_egoverse` finetunes
from), applies the Egoverse transforms and norm stats, and runs the policy on
real frames sampled from an h5 file. Prints predicted vs. ground-truth actions
so you get a quantitative baseline number.

Usage:
  uv run python 3dvision-experiments/run_inference.py \
      --h5-path /cluster/work/cvg/data/rytsui/egoverse_h5/<some>.h5 \
      --num-frames 16
"""

import pathlib

import h5py
import numpy as np
import tyro

from openpi.policies import policy_config
from openpi.shared import normalize
from openpi.training import config as _config

CONFIG_NAME = "pi05_egoverse"
# On the cluster the base weights are mirrored locally; off-cluster we pull from
# the public GCS bucket (cached under ~/.cache/openpi after the first download).
CHECKPOINT_DIR_CLUSTER = "/cluster/work/cvg/data/rytsui/pi05_base_jax"
CHECKPOINT_DIR_PUBLIC = "gs://openpi-assets/checkpoints/pi05_base"
DEFAULT_PROMPT = "put the object in the bowl"


def load_frame(h5_path: pathlib.Path, frame_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (image[HWC uint8], state[24], gt_actions[24]) for a single frame."""
    with h5py.File(h5_path, "r") as f:
        image = np.asarray(f["observations/images/aria_rgb_cam/color"][frame_idx])
        state = np.concatenate(
            [f["observations/qpos_arm"][frame_idx], f["observations/qpos_hand"][frame_idx]]
        ).astype(np.float32)
        actions = np.concatenate(
            [f["actions_arm"][frame_idx], f["actions_hand"][frame_idx]]
        ).astype(np.float32)
    return image, state, actions


def main(
    h5_path: pathlib.Path,
    *,
    num_frames: int = 8,
    frame_stride: int = 50,
    start_frame: int = 0,
    prompt: str = DEFAULT_PROMPT,
    checkpoint_dir: str | None = None,
) -> None:
    if checkpoint_dir is None:
        checkpoint_dir = (
            CHECKPOINT_DIR_CLUSTER
            if pathlib.Path(CHECKPOINT_DIR_CLUSTER).exists()
            else CHECKPOINT_DIR_PUBLIC
        )
    cfg = _config.get_config(CONFIG_NAME)

    # Norm stats live under <assets_dirs>/<repo_id>, written by
    # `scripts/compute_norm_stats.py --config-name pi05_egoverse`.
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    norm_stats_dir = cfg.assets_dirs / data_cfg.repo_id
    print(f"Loading norm stats from: {norm_stats_dir}")
    norm_stats = normalize.load(norm_stats_dir)

    print(f"Loading pi0.5 base weights from: {checkpoint_dir}")
    policy = policy_config.create_trained_policy(
        cfg,
        checkpoint_dir,
        norm_stats=norm_stats,
        default_prompt=prompt,
    )

    with h5py.File(h5_path, "r") as f:
        total = f["observations/qpos_arm"].shape[0]
    frame_ids = list(range(start_frame, min(total, start_frame + num_frames * frame_stride), frame_stride))
    print(f"Running inference on {len(frame_ids)} frames from {h5_path.name} (total={total})")

    arm_mses, hand_mses = [], []
    for idx in frame_ids:
        image, state, gt_actions = load_frame(h5_path, idx)
        example = {
            "observation/image": image,
            "observation/state": state,
            "prompt": prompt,
        }
        result = policy.infer(example)
        pred = np.asarray(result["actions"])  # shape: (action_horizon, 24)

        # Ground truth is a single step; compare against the first predicted step.
        pred_t0 = pred[0]
        arm_mse = float(np.mean((pred_t0[:7] - gt_actions[:7]) ** 2))
        hand_mse = float(np.mean((pred_t0[7:] - gt_actions[7:]) ** 2))
        arm_mses.append(arm_mse)
        hand_mses.append(hand_mse)
        print(
            f"  frame={idx:5d}  pred_shape={tuple(pred.shape)}  "
            f"arm_mse={arm_mse:.4f}  hand_mse={hand_mse:.4f}"
        )

    print("\n=== Baseline summary (pi0.5 base, no finetuning) ===")
    print(f"  frames evaluated : {len(frame_ids)}")
    print(f"  mean arm MSE     : {np.mean(arm_mses):.4f}")
    print(f"  mean hand MSE    : {np.mean(hand_mses):.4f}")


if __name__ == "__main__":
    tyro.cli(main)
