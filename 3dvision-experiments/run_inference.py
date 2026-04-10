"""Baseline inference for pi0.5 on Egoverse h5 episodes.

Loads pi0.5 base weights (what `pi05_egoverse` finetunes from), runs on real
frames from one or more h5 episodes, and compares pi0.5's first-step prediction
to ground truth. Also reports two naive baselines for calibration:

  - zero_action: predict all-zeros (unnormalized). This is ~variance of the
    action distribution around zero.
  - const_state: predict current state (qpos) as next action. For slow
    manipulation this is a strong "do nothing" baseline.

Per-dim MSE is printed so you can see which joints dominate the error.

Usage (single episode, every 10th frame):
  uv run python 3dvision-experiments/run_inference.py \
      --h5-path /path/to/episode.h5 --frame-stride 10

Usage (all episodes in a directory, every 10th frame each):
  uv run python 3dvision-experiments/run_inference.py \
      --episodes-dir /path/to/dir --frame-stride 10

`--num-frames` limits how many frames per episode (default: all). Use
`--num-frames 8` for a quick smoke test.
"""

import dataclasses
import pathlib

import h5py
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
ACTION_DIM = ARM_DIM + HAND_DIM  # 24


CHUNK_LEN = 10


def load_frame(h5: h5py.File, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (image[HWC uint8], state[24]) for a single frame."""
    image = np.asarray(h5["observations/images/aria_rgb_cam/color"][frame_idx])
    state = np.concatenate(
        [h5["observations/qpos_arm"][frame_idx], h5["observations/qpos_hand"][frame_idx]]
    ).astype(np.float32)
    return image, state


def load_gt_chunk(h5: h5py.File, frame_idx: int) -> np.ndarray | None:
    """Return GT actions for frames [frame_idx, frame_idx+CHUNK_LEN), shape (CHUNK_LEN, 24).

    Returns None if not enough frames remain.
    """
    total = h5["actions_arm"].shape[0]
    if frame_idx + CHUNK_LEN > total:
        return None
    arm = np.asarray(h5["actions_arm"][frame_idx : frame_idx + CHUNK_LEN])
    hand = np.asarray(h5["actions_hand"][frame_idx : frame_idx + CHUNK_LEN])
    return np.concatenate([arm, hand], axis=1).astype(np.float32)


def evaluate_episode(
    policy,
    h5_path: pathlib.Path,
    num_frames: int | None,
    frame_stride: int,
    start_frame: int,
    prompt: str,
) -> dict:
    """Run inference over one episode and return squared-error tensors.

    At each sampled frame, the model predicts a full CHUNK_LEN-step action chunk.
    All steps in the chunk are evaluated against the corresponding GT actions.

    Returns a dict with arrays of shape (N * CHUNK_LEN, 24):
      - pi0_sq_err:   (pi0 pred - gt) ** 2
      - zero_sq_err:  (0 - gt) ** 2
      - const_sq_err: (state - gt) ** 2  (state is broadcast across the chunk)
    """
    with h5py.File(h5_path, "r") as f:
        total = f["observations/qpos_arm"].shape[0]
        end = total if num_frames is None else min(total, start_frame + num_frames * frame_stride)
        frame_ids = list(range(start_frame, end, frame_stride))
        print(f"  episode={h5_path.name}  total={total}  evaluating={len(frame_ids)} chunks of {CHUNK_LEN} steps")

        all_pi0_sq, all_zero_sq, all_const_sq = [], [], []

        for i, idx in enumerate(frame_ids):
            gt_chunk = load_gt_chunk(f, idx)
            if gt_chunk is None:
                break  # not enough frames for a full chunk

            image, state = load_frame(f, idx)
            result = policy.infer(
                {"observation/image": image, "observation/state": state, "prompt": prompt}
            )
            pred = np.asarray(result["actions"])[:CHUNK_LEN]  # (CHUNK_LEN, 24)

            all_pi0_sq.append((pred - gt_chunk) ** 2)
            all_zero_sq.append(gt_chunk ** 2)
            # state broadcast: "do nothing" baseline predicts current state for all 10 steps
            all_const_sq.append((state[None, :] - gt_chunk) ** 2)

            if i % 50 == 0 or i == len(frame_ids) - 1:
                chunk_arm = all_pi0_sq[-1][:, :ARM_DIM].mean()
                chunk_hand = all_pi0_sq[-1][:, ARM_DIM:].mean()
                print(
                    f"    chunk {i + 1}/{len(frame_ids)}  frame={idx}  "
                    f"arm_mse={chunk_arm:.4f}  hand_mse={chunk_hand:.4f}"
                )

    return {
        "pi0": np.concatenate(all_pi0_sq, axis=0),
        "zero": np.concatenate(all_zero_sq, axis=0),
        "const": np.concatenate(all_const_sq, axis=0),
    }


def summarize(name: str, sq_err: np.ndarray) -> None:
    """Print overall + arm/hand split + per-dim breakdown for one method."""
    arm = sq_err[:, :ARM_DIM].mean()
    hand = sq_err[:, ARM_DIM:].mean()
    total = sq_err.mean()
    print(f"  {name:>12s}:  arm={arm:.4f}  hand={hand:.4f}  total={total:.4f}")


def print_per_dim(sq_err: np.ndarray, label: str) -> None:
    per_dim = sq_err.mean(axis=0)
    print(f"  {label} per-dim MSE:")
    print(f"    arm  (dims 0-6):   " + " ".join(f"{v:.3f}" for v in per_dim[:ARM_DIM]))
    print(f"    hand (dims 7-23):  " + " ".join(f"{v:.3f}" for v in per_dim[ARM_DIM:]))


def main(
    *,
    h5_path: pathlib.Path | None = None,
    episodes_dir: pathlib.Path | None = None,
    num_frames: int | None = None,
    frame_stride: int = 10,
    start_frame: int = 0,
    prompt: str = DEFAULT_PROMPT,
    checkpoint_dir: str | None = None,
) -> None:
    if (h5_path is None) == (episodes_dir is None):
        raise ValueError("Pass exactly one of --h5-path or --episodes-dir.")

    if checkpoint_dir is None:
        checkpoint_dir = (
            CHECKPOINT_DIR_CLUSTER
            if pathlib.Path(CHECKPOINT_DIR_CLUSTER).exists()
            else CHECKPOINT_DIR_PUBLIC
        )

    cfg = _config.get_config(CONFIG_NAME)
    # Swap LoRA variants -> plain gemma so the base checkpoint loads cleanly.
    # LoRA adapters init to zero, so the two models are numerically equivalent
    # at step 0; this just avoids the structural mismatch in `restore_params`.
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
    policy = policy_config.create_trained_policy(
        cfg,
        checkpoint_dir,
        norm_stats=norm_stats,
        default_prompt=prompt,
    )

    if h5_path is not None:
        episode_paths = [h5_path]
    else:
        episode_paths = sorted(pathlib.Path(episodes_dir).glob("*.h5"))
        if not episode_paths:
            raise ValueError(f"No .h5 files found in {episodes_dir}")
        print(f"Found {len(episode_paths)} episodes in {episodes_dir}")

    # Accumulate squared errors across all episodes.
    all_pi0, all_zero, all_const = [], [], []
    skipped = []
    for ep_path in episode_paths:
        try:
            stats = evaluate_episode(policy, ep_path, num_frames, frame_stride, start_frame, prompt)
            all_pi0.append(stats["pi0"])
            all_zero.append(stats["zero"])
            all_const.append(stats["const"])
        except Exception as e:
            print(f"  ERROR in {ep_path.name}: {e} — skipping")
            skipped.append(ep_path.name)

    pi0_sq = np.concatenate(all_pi0, axis=0)
    zero_sq = np.concatenate(all_zero, axis=0)
    const_sq = np.concatenate(all_const, axis=0)

    print("\n=== Summary ===")
    print(f"  episodes evaluated: {len(episode_paths) - len(skipped)}/{len(episode_paths)}")
    if skipped:
        print(f"  skipped: {skipped}")
    print(f"  total steps evaluated: {pi0_sq.shape[0]}  ({pi0_sq.shape[0] // CHUNK_LEN} chunks of {CHUNK_LEN})")
    print()
    summarize("pi0.5 base", pi0_sq)
    summarize("zero action", zero_sq)
    summarize("const state", const_sq)
    print()
    print_per_dim(pi0_sq, "pi0.5 base")


if __name__ == "__main__":
    tyro.cli(main)
