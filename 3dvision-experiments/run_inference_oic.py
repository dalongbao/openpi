"""run_inference_oic.py — real-frame inference for the oic_human (6-dim) model.

Runs pi05_ego_human_oic on REAL Aria frames from the oic_human LeRobot dataset and
compares the predicted 6-dim action [x,y,z, e1,e2,e3] to ground truth. This is the
oic analog of run_inference.py — it removes the Isaac sim entirely, so it measures
MODEL QUALITY without the sim visual-gap confound.

Convention-agnostic: pred and GT are both in the data's native Euler convention, so
comparing them needs no knowledge of the (unknown) Euler order. Position direction is
order-independent regardless.

Reports, per 10-step chunk:
  - MSE vs GT (total / pos dims0-2 / rot dims3-5), vs zero-action, vs const-state baseline
  - motion/direction on POSITION: |pred|/|GT| magnitude ratio + cosine(pred,GT) direction

Run on Euler via uv (no Isaac):
  sbatch --export=ALL,UV_FROZEN=1 --time=01:00:00 --mem-per-cpu=16G --cpus-per-task=8 \
         --gpus=rtx_4090:1 3dvision-experiments/run_inference_oic.slurm
"""
import dataclasses
import io
import pathlib

import numpy as np
import tyro

from openpi.policies import policy_config
from openpi.shared import normalize
from openpi.training import config as _config

DATA_ROOT = "/cluster/work/cvg/data/Egoverse/lerobot_egoverse/egoverse/oic_human"
CKPT_DEFAULT = "/cluster/work/cvg/data/rytsui/checkpoints/pi05_ego_human_oic/human_oic/29999"
BASE_WEIGHTS = "/cluster/work/cvg/data/Egoverse/pi05_base_jax"  # untrained-baseline floor
CONFIG_NAME = "pi05_ego_human_oic"
DEFAULT_PROMPT = "put the object in the container"
POS_DIM = 3        # [x,y,z]; dims 3-5 are the Euler angles
CHUNKS_SIZE = 1000


def load_episode(ep: int):
    """Return (images list, state (N,6), actions (N,6)) for one oic episode."""
    import pyarrow.parquet as pq
    from PIL import Image
    p = f"{DATA_ROOT}/data/chunk-{ep // CHUNKS_SIZE:03d}/episode_{ep:06d}.parquet"
    rows = pq.read_table(p).to_pylist()
    state = np.asarray([r["state"] for r in rows], dtype=np.float32)
    actions = np.asarray([r["actions"] for r in rows], dtype=np.float32)
    imgs = []
    for r in rows:
        im = r["image"]
        data = im["bytes"] if isinstance(im, dict) else im
        imgs.append(np.asarray(Image.open(io.BytesIO(data)).convert("RGB")))
    return imgs, state, actions


def motion_report(pred_disp, gt_disp):
    print("\n=== Motion / direction (POSITION, net displacement per chunk; order-independent) ===")
    gmag = np.linalg.norm(gt_disp, axis=1)
    pmag = np.linalg.norm(pred_disp, axis=1)
    moving = gmag > 1e-3
    ratio = pmag[moving].mean() / (gmag[moving].mean() + 1e-9)
    cos = np.sum(pred_disp * gt_disp, axis=1) / (pmag * gmag + 1e-9)
    print(f"  mean |GT disp|   {gmag.mean():.4f} m   mean |pred disp| {pmag.mean():.4f} m")
    print(f"  magnitude ratio  |pred|/|GT|: {ratio:.3f}   (1=matches, <1=timid, ~0=frozen)")
    print(f"  mean cosine(pred,GT) dir:    {np.nanmean(cos[moving]):.3f}   "
          f"(1=right direction, 0=orthogonal, <0=opposite)")
    print(f"  chunks where GT moves: {moving.sum()}/{len(moving)}")


def main(
    *,
    num_episodes: int = 25,
    start_episode: int = 0,
    frame_stride: int = 5,
    checkpoint_dir: str | None = None,
    norm_stats_dir: str | None = None,
    prompt: str = DEFAULT_PROMPT,
    finetuned: bool = True,
):
    # finetuned=False loads the untrained pi0.5 base weights as a floor baseline, mirroring
    # run_inference.py: swap the LoRA gemma variants -> plain gemma so the base checkpoint
    # loads cleanly (LoRA adapters init to zero, so the models match numerically at step 0).
    if checkpoint_dir is None:
        checkpoint_dir = CKPT_DEFAULT if finetuned else BASE_WEIGHTS
    cfg = _config.get_config(CONFIG_NAME)
    if not finetuned:
        cfg = dataclasses.replace(
            cfg,
            model=dataclasses.replace(
                cfg.model, paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
            ),
        )
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    # Base weights carry no oic norm stats; always fall back to the config-default oic stats there.
    ns_dir = pathlib.Path(norm_stats_dir) if norm_stats_dir else pathlib.Path(f"{checkpoint_dir}/assets/egoverse/oic_human")
    if not (ns_dir / "norm_stats.json").exists():
        ns_dir = cfg.assets_dirs / data_cfg.repo_id  # fall back to config default
    print(f"Norm stats: {ns_dir}")
    norm_stats = normalize.load(ns_dir)

    chunk_len = cfg.model.action_horizon
    tag = "oic finetuned" if finetuned else "oic base"
    print(f"Loading {CONFIG_NAME} [{tag}] from {checkpoint_dir} (chunk_len={chunk_len})")
    policy = policy_config.create_trained_policy(cfg, checkpoint_dir, norm_stats=norm_stats, default_prompt=prompt)

    all_pi0, all_zero, all_const, all_gd, all_pd = [], [], [], [], []
    for ep in range(start_episode, start_episode + num_episodes):
        try:
            imgs, state, actions = load_episode(ep)
        except Exception as e:
            print(f"  ep {ep}: skip ({e})"); continue
        n = len(actions)
        ids = list(range(0, n - chunk_len, frame_stride))
        for idx in ids:
            gt = actions[idx:idx + chunk_len]                      # (chunk,6)
            res = policy.infer({"observation/image": imgs[idx],
                                "observation/state": state[idx],
                                "prompt": prompt})
            pred = np.asarray(res["actions"])[:chunk_len]          # (chunk,6)
            all_pi0.append((pred - gt) ** 2)
            all_zero.append(gt ** 2)
            all_const.append((state[idx][None, :] - gt) ** 2)
            all_gd.append(gt[-1, :POS_DIM] - state[idx][:POS_DIM])
            all_pd.append(pred[-1, :POS_DIM] - state[idx][:POS_DIM])
        print(f"  ep {ep}: {len(ids)} chunks  (running total {len(all_pi0)})")

    pi0 = np.concatenate(all_pi0); zero = np.concatenate(all_zero); const = np.concatenate(all_const)
    print(f"\n=== Summary ({len(all_pi0)} chunks of {chunk_len}) ===")
    for name, sq in [(tag, pi0), ("zero action", zero), ("const state", const)]:
        print(f"  {name:>14s}: total={sq.mean():.4f}  pos={sq[:, :POS_DIM].mean():.4f}  rot={sq[:, POS_DIM:].mean():.4f}")
    print("  per-dim MSE:", " ".join(f"{v:.3f}" for v in pi0.mean(axis=0)))
    motion_report(np.stack(all_pd), np.stack(all_gd))


if __name__ == "__main__":
    tyro.cli(main)
