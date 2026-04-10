"""Visual version of run_inference.py: plot predicted vs GT action chunks.

Matches the eval logic: stride through the episode every CHUNK_LEN (10) frames,
feed GT image + state each time, plot the full 10-step predicted chunk against
the 10-step GT chunk. Produces one PNG per sampled chunk.

Usage (3 evenly-spaced chunks from each of 5 episodes):
  uv run python 3dvision-experiments/visualise_eval.py \
      --episodes-dir /path/to/dir --num-episodes 5 --num-vis-chunks 3 \
      --out-dir plots_eval/

Usage (5 chunks from a single episode):
  uv run python 3dvision-experiments/visualise_eval.py \
      --h5-path /path/to/episode.h5 --num-vis-chunks 5 --out-dir plots_eval/
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
FPS = 50

ARM_LABELS = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
HAND_LABELS = [f"hand_{i}" for i in range(HAND_DIM)]


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


def load_gt_chunk(h5: h5py.File, frame_idx: int) -> np.ndarray | None:
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
    ep_name: str = "",
) -> None:
    """Plot predicted vs GT action chunk. Two panels (arm/hand), one color per dim."""
    t = np.arange(CHUNK_LEN) / FPS  # seconds
    arm_colors = plt.cm.tab10(np.linspace(0, 1, ARM_DIM))
    hand_colors = plt.cm.tab20(np.linspace(0, 1, HAND_DIM))

    arm_mse = ((pred[:, :ARM_DIM] - gt[:, :ARM_DIM]) ** 2).mean()
    hand_mse = ((pred[:, ARM_DIM:] - gt[:, ARM_DIM:]) ** 2).mean()

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"{ep_name}  frame {frame_idx}:  solid=pred  dashed=GT\n"
        f"arm MSE={arm_mse:.4f}  hand MSE={hand_mse:.4f}",
        fontsize=12,
    )

    # Arm
    ax = axes[0]
    for d in range(ARM_DIM):
        ax.plot(t, pred[:, d], color=arm_colors[d], linewidth=1.5, label=f"{ARM_LABELS[d]} pred")
        ax.plot(t, gt[:, d], color=arm_colors[d], linewidth=1.5, linestyle="--", label=f"{ARM_LABELS[d]} GT")
    ax.set_ylabel("Joint value")
    ax.set_title(f"Arm (dims 0-{ARM_DIM - 1})")
    ax.legend(fontsize=7, ncol=ARM_DIM, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Hand
    ax = axes[1]
    for d in range(HAND_DIM):
        ax.plot(t, pred[:, ARM_DIM + d], color=hand_colors[d], linewidth=1.5, label=f"{HAND_LABELS[d]} pred")
        ax.plot(t, gt[:, ARM_DIM + d], color=hand_colors[d], linewidth=1.5, linestyle="--", label=f"{HAND_LABELS[d]} GT")
    ax.set_ylabel("Joint value")
    ax.set_xlabel("Time (seconds)")
    ax.set_title(f"Hand (dims {ARM_DIM}-{ACTION_DIM - 1})")
    ax.legend(fontsize=6, ncol=6, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def visualize_episode(
    policy,
    h5_path: pathlib.Path,
    num_vis_chunks: int,
    out_dir: pathlib.Path,
    prompt: str,
) -> int:
    saved = 0
    with h5py.File(h5_path, "r") as f:
        total = f["observations/qpos_arm"].shape[0]
        # All chunk-aligned frame indices (stride = CHUNK_LEN)
        all_chunk_starts = list(range(0, total - CHUNK_LEN + 1, CHUNK_LEN))
        if not all_chunk_starts:
            print(f"  skipping {h5_path.name}: too short ({total} frames)")
            return 0

        # Evenly sample num_vis_chunks from all available chunks
        indices = np.linspace(0, len(all_chunk_starts) - 1, min(num_vis_chunks, len(all_chunk_starts)), dtype=int)
        frame_ids = [all_chunk_starts[i] for i in indices]
        print(f"  episode={h5_path.name}  total={total}  {len(all_chunk_starts)} chunks available  visualizing: {frame_ids}")

        for idx in frame_ids:
            image = np.asarray(f["observations/images/aria_rgb_cam/color"][idx])
            state = np.concatenate(
                [f["observations/qpos_arm"][idx], f["observations/qpos_hand"][idx]]
            ).astype(np.float32)

            result = policy.infer(
                {"observation/image": image, "observation/state": state, "prompt": prompt}
            )
            pred = np.asarray(result["actions"])[:CHUNK_LEN]  # (10, 24)

            gt = load_gt_chunk(f, idx)
            if gt is None:
                continue

            ep_name = h5_path.stem
            plot_chunk(pred, gt, idx, out_dir / f"{ep_name}_frame{idx:05d}.png", ep_name=ep_name)
            saved += 1
    return saved


def main(
    *,
    h5_path: pathlib.Path | None = None,
    episodes_dir: pathlib.Path | None = None,
    num_episodes: int = 5,
    num_vis_chunks: int = 3,
    out_dir: pathlib.Path = pathlib.Path("plots_eval"),
    checkpoint_dir: str | None = None,
    prompt: str = DEFAULT_PROMPT,
) -> None:
    if (h5_path is None) == (episodes_dir is None):
        raise ValueError("Pass exactly one of --h5-path or --episodes-dir.")

    policy = load_policy(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if h5_path is not None:
        episode_paths = [h5_path]
    else:
        all_eps = sorted(pathlib.Path(episodes_dir).glob("*.h5"))
        if not all_eps:
            raise ValueError(f"No .h5 files found in {episodes_dir}")
        indices = np.linspace(0, len(all_eps) - 1, min(num_episodes, len(all_eps)), dtype=int)
        episode_paths = [all_eps[i] for i in indices]
        print(f"Found {len(all_eps)} episodes, sampling {len(episode_paths)}")

    total_plots = 0
    for ep_path in episode_paths:
        total_plots += visualize_episode(policy, ep_path, num_vis_chunks, out_dir, prompt)

    print(f"\nDone. {total_plots} plots saved to {out_dir}/")


if __name__ == "__main__":
    tyro.cli(main)
