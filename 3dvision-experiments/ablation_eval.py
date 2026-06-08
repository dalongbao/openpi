"""ablation_eval.py — offline evaluation of one checkpoint on held-out R-ID frames.

Common yardstick for the data-mix ablation: every condition's checkpoint is scored on
the SAME held-out object_in_bowl (R-ID) robot teleop episodes, predicting actions and
comparing to ground truth. No sim, no robot — a behavior-cloning-style proxy for task
success (good for RELATIVE comparison of data mixes; see PROJECT notes).

Metrics (per episode, then aggregated):
  - cos_dir / mag_ratio : predicted vs GT EE-POSITION net displacement per chunk
                          (position dims 0-2; embodiment/convention-agnostic, the metric
                          that separated the strong vs weak models).
  - arm_mse / hand_mse / pos_mse vs GT, and vs `zero` and `const-state` baselines.
  - rollout_endpoint_err / rollout_rmse : receding-horizon OPEN-LOOP rollout — free-run the
                          policy's own predicted arm EE-pose across the episode (teacher-forced
                          on images, since there is no sim), compare the resulting EE-position
                          trajectory to the demo's. Captures compounding error.

Writes a JSON of per-episode + summary metrics; aggregate across conditions with
aggregate_ablation.py.

Run on Euler via uv (mirror run_inference.slurm):
  sbatch --export=ALL,UV_FROZEN=1 --time=01:00:00 --mem-per-cpu=16G --cpus-per-task=8 \
    --gpus=rtx_4090:1 3dvision-experiments/run_ablation_eval.slurm \
    --config-name pi05_egoverse --checkpoint-dir <ckpt> --condition R-ID_only \
    --episodes-dir <held_out_dir>
"""
import dataclasses
import json
import pathlib

import h5py
import numpy as np
import tyro

from openpi.policies import policy_config
from openpi.shared import normalize
from openpi.training import config as _config

ARM_DIM = 7        # EE pose [x,y,z, qx,qy,qz,qw]
POS_DIM = 3        # [x,y,z] — direction metric uses this (rep/frame-agnostic)
ACTION_DIM = 24    # 7 arm + 17 hand


def load_frame(f, idx):
    img = f["observations/images/aria_rgb_cam/color"][idx]
    state = np.concatenate([f["observations/qpos_arm"][idx], f["observations/qpos_hand"][idx]]).astype(np.float32)
    return np.asarray(img), state


def eval_episode(policy, h5_path, frame_stride, chunk_len, prompt):
    out = {}
    with h5py.File(h5_path, "r") as f:
        total = f["observations/qpos_arm"].shape[0]
        ids = list(range(0, total - chunk_len, frame_stride))
        pi0, zero, const, gd, pd = [], [], [], [], []
        for idx in ids:
            gt = np.concatenate([f["actions_arm"][idx:idx + chunk_len],
                                 f["actions_hand"][idx:idx + chunk_len]], axis=1)  # (chunk,24)
            img, state = load_frame(f, idx)
            pred = np.asarray(policy.infer(
                {"observation/image": img, "observation/state": state, "prompt": prompt})["actions"])[:chunk_len]
            pi0.append((pred - gt) ** 2); zero.append(gt ** 2); const.append((state[None] - gt) ** 2)
            gd.append(gt[-1, :POS_DIM] - state[:POS_DIM]); pd.append(pred[-1, :POS_DIM] - state[:POS_DIM])

        pi0 = np.concatenate(pi0); zero = np.concatenate(zero); const = np.concatenate(const)
        gd = np.stack(gd); pd = np.stack(pd)
        gmag = np.linalg.norm(gd, axis=1); pmag = np.linalg.norm(pd, axis=1); mv = gmag > 1e-3
        out.update(
            n_chunks=len(ids),
            arm_mse=float(pi0[:, :ARM_DIM].mean()), hand_mse=float(pi0[:, ARM_DIM:].mean()),
            pos_mse=float(pi0[:, :POS_DIM].mean()),
            arm_mse_zero=float(zero[:, :ARM_DIM].mean()), arm_mse_const=float(const[:, :ARM_DIM].mean()),
            mag_ratio=float(pmag[mv].mean() / (gmag[mv].mean() + 1e-9)),
            cos_dir=float(np.nanmean(np.sum(pd * gd, 1)[mv] / (pmag[mv] * gmag[mv] + 1e-9))),
        )

        # --- receding-horizon open-loop rollout (free-run arm state, teacher-force images) ---
        cur = (np.concatenate([f["observations/qpos_arm"][0], f["observations/qpos_hand"][0]])).astype(np.float32)
        gt_pos, roll_pos = [], []
        t = 0
        while t < total - 1:
            img, _ = load_frame(f, t)
            pred = np.asarray(policy.infer(
                {"observation/image": img, "observation/state": cur, "prompt": prompt})["actions"])
            k = min(chunk_len, total - 1 - t)
            cur = pred[k - 1].astype(np.float32)           # advance to end of executed chunk (absolute EE pose)
            roll_pos.append(cur[:POS_DIM]); gt_pos.append(f["actions_arm"][min(t + k, total - 1)][:POS_DIM])
            t += k
        roll_pos = np.stack(roll_pos); gt_pos = np.stack(gt_pos)
        out["rollout_endpoint_err"] = float(np.linalg.norm(roll_pos[-1] - gt_pos[-1]))
        out["rollout_rmse"] = float(np.sqrt(((roll_pos - gt_pos) ** 2).sum(1).mean()))
    return out


def main(
    *,
    condition: str,
    checkpoint_dir: str,
    config_name: str = "pi05_egoverse",
    episodes_dir: str | None = None,
    episodes: list[str] = [],            # explicit held-out h5 paths (overrides episodes_dir glob)
    held_out_file: str | None = None,    # text file of h5 basenames to include from episodes_dir
    frame_stride: int = 10,
    prompt: str = "put the object in the bowl",
    finetuned: bool = True,
    output_dir: str = "/cluster/scratch/lichin/pi0_test/ablation",
):
    cfg = _config.get_config(config_name)
    if not finetuned:  # base weights: swap LoRA->plain gemma (see run_inference.py)
        cfg = dataclasses.replace(cfg, model=dataclasses.replace(
            cfg.model, paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"))
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    norm_stats = normalize.load(cfg.assets_dirs / data_cfg.repo_id)
    chunk_len = cfg.model.action_horizon
    print(f"[{condition}] {config_name} <- {checkpoint_dir}  (chunk_len={chunk_len})")
    policy = policy_config.create_trained_policy(cfg, checkpoint_dir, norm_stats=norm_stats, default_prompt=prompt)

    eps = [pathlib.Path(p) for p in episodes]
    if not eps and episodes_dir:
        allh5 = sorted(pathlib.Path(episodes_dir).glob("*.h5"))
        keep = set(l.strip() for l in open(held_out_file)) if held_out_file else None
        eps = [p for p in allh5 if (keep is None or p.name in keep)]
    if not eps:
        raise ValueError("No episodes — pass --episodes, or --episodes-dir [+ --held-out-file].")
    print(f"[{condition}] {len(eps)} held-out episodes")

    per_ep = []
    for ep in eps:
        try:
            m = eval_episode(policy, ep, frame_stride, chunk_len, prompt)
            m["episode"] = ep.name
            per_ep.append(m)
            print(f"  {ep.name}: cos={m['cos_dir']:.3f} mag={m['mag_ratio']:.2f} "
                  f"arm_mse={m['arm_mse']:.4f}(const {m['arm_mse_const']:.4f}) "
                  f"rollout_end={m['rollout_endpoint_err']:.3f}m")
        except Exception as e:
            print(f"  {ep.name}: SKIP ({e})")

    keys = ["cos_dir", "mag_ratio", "arm_mse", "hand_mse", "pos_mse",
            "arm_mse_zero", "arm_mse_const", "rollout_endpoint_err", "rollout_rmse"]
    summary = {k: float(np.mean([m[k] for m in per_ep])) for k in keys}
    summary["n_episodes"] = len(per_ep)
    print(f"\n[{condition}] SUMMARY: " + "  ".join(f"{k}={summary[k]:.4f}" for k in keys))

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    out = pathlib.Path(output_dir) / f"{condition}.json"
    json.dump({"condition": condition, "config": config_name, "checkpoint": checkpoint_dir,
               "summary": summary, "per_episode": per_ep}, open(out, "w"), indent=2)
    print(f"[{condition}] wrote {out}")


if __name__ == "__main__":
    tyro.cli(main)
