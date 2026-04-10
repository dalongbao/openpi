"""Visualize pi0.5 predicted action chunks vs ground truth.

For a handful of frames spread across an episode, plots the full 10-step
predicted action chunk against the next 10 ground-truth actions.  Produces
one figure per sampled frame with arm and hand subplots.

Usage:
  uv run python 3dvision-experiments/visualize_predictions.py \
      --h5-path /path/to/episode.h5 --num-vis-frames 5 --out-dir plots/
"""

import dataclasses
import pathlib

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tyro

from openpi.policies import policy_config
from openpi.shared import normalize
from openpi.training import config as _config

CONFIG_NAME = "pi05_egoverse"
CHECKPOINT_DIR_CLUSTER = "/cluster/work/cvg/data/Egoverse/pi05_base_jax"
CHECKPOINT_DIR_PUBLIC = "gs://openpi-assets/checkpoints/pi05_base"
DEFAULT_PROMPT = "put the object in the bowl"

ARM_DIM = 7
HAND_DIM = 17
ACTION_DIM = ARM_DIM + HAND_DIM
CHUNK_LEN = 10


def load_policy(checkpoint_dir: str | None):
    if checkpoint_dir is None:
        checkpoint_dir = (
            CHECKPOINT_DIR_CLUSTER
            if pathlib.Path(CHECKPOINT_DIR_CLUSTER).exists()
            else CHECKPOINT_DIR_PUBLIC
        )

    cfg = _config.get_config(CONFIG_NAME)
    cfg = dataclasses.replace(
        cfg,
        model=dataclasses.replace(
            cfg.model,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
        ),
    )

    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    norm_stats_dir = cfg.assets_dirs / data_cfg.repo_id
    print(f"Loading norm stats from: {norm_stats_dir}")
    norm_stats = normalize.load(norm_stats_dir)

    print(f"Loading pi0.5 base weights from: {checkpoint_dir}")
    return policy_config.create_trained_policy(
        cfg, checkpoint_dir, norm_stats=norm_stats, default_prompt=DEFAULT_PROMPT,
    )


def get_gt_chunk(h5: h5py.File, frame_idx: int) -> np.ndarray | None:
    """Return ground-truth action chunk of length CHUNK_LEN starting at frame_idx.

    Returns None if not enough frames remain.
    """
    total = h5["actions_arm"].shape[0]
    if frame_idx + CHUNK_LEN > total:
        return None
    arm = np.asarray(h5["actions_arm"][frame_idx : frame_idx + CHUNK_LEN])
    hand = np.asarray(h5["actions_hand"][frame_idx : frame_idx + CHUNK_LEN])
    return np.concatenate([arm, hand], axis=1).astype(np.float32)


def plot_chunk(
    pred: np.ndarray,
    gt: np.ndarray,
    frame_idx: int,
    out_path: pathlib.Path,
) -> None:
    """Plot predicted vs GT action chunk. pred and gt are (CHUNK_LEN, ACTION_DIM)."""
    steps = np.arange(CHUNK_LEN)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Frame {frame_idx}: predicted (blue) vs GT (orange) action chunk", fontsize=13)

    # Arm dims
    ax = axes[0]
    for d in range(ARM_DIM):
        ax.plot(steps, pred[:, d], color="tab:blue", alpha=0.7, label="pred" if d == 0 else None)
        ax.plot(steps, gt[:, d], color="tab:orange", alpha=0.7, linestyle="--", label="GT" if d == 0 else None)
    ax.set_ylabel("Joint value")
    ax.set_title(f"Arm (dims 0-{ARM_DIM - 1})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Hand dims
    ax = axes[1]
    for d in range(ARM_DIM, ACTION_DIM):
        ax.plot(steps, pred[:, d], color="tab:blue", alpha=0.7, label="pred" if d == ARM_DIM else None)
        ax.plot(steps, gt[:, d], color="tab:orange", alpha=0.7, linestyle="--", label="GT" if d == ARM_DIM else None)
    ax.set_ylabel("Joint value")
    ax.set_xlabel("Chunk step (0 = current frame)")
    ax.set_title(f"Hand (dims {ARM_DIM}-{ACTION_DIM - 1})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main(
    *,
    h5_path: pathlib.Path,
    num_vis_frames: int = 5,
    out_dir: pathlib.Path = pathlib.Path("plots"),
    checkpoint_dir: str | None = None,
    prompt: str = DEFAULT_PROMPT,
) -> None:
    policy = load_policy(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        total = f["observations/qpos_arm"].shape[0]
        # Pick frames evenly spread, leaving room for a full chunk at the end.
        max_start = total - CHUNK_LEN
        frame_ids = np.linspace(0, max_start, num_vis_frames, dtype=int).tolist()
        print(f"Episode: {h5_path.name}  total_frames={total}  visualizing frames: {frame_ids}")

        for idx in frame_ids:
            image = np.asarray(f["observations/images/aria_rgb_cam/color"][idx])
            state = np.concatenate(
                [f["observations/qpos_arm"][idx], f["observations/qpos_hand"][idx]]
            ).astype(np.float32)

            result = policy.infer(
                {"observation/image": image, "observation/state": state, "prompt": prompt}
            )
            pred = np.asarray(result["actions"])[:CHUNK_LEN]  # (10, 24)

            gt = get_gt_chunk(f, idx)
            if gt is None:
                print(f"  skipping frame {idx}: not enough frames for GT chunk")
                continue

            ep_name = h5_path.stem
            plot_chunk(pred, gt, idx, out_dir / f"{ep_name}_frame{idx:05d}.png")

    print(f"\nDone. {num_vis_frames} plots saved to {out_dir}/")


if __name__ == "__main__":
    tyro.cli(main)
