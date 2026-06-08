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

R_ID_DIR = "/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/object_in_bowl_processed_50hz"  # R-ID source
ARM_DIM = 7        # EE pose [x,y,z, qx,qy,qz,qw]
POS_DIM = 3        # [x,y,z] — direction metric uses this (rep/frame-agnostic)
ACTION_DIM = 24    # 7 arm + 17 hand
SUCCESS_THRESH = 0.08   # m; "reached" the object/bowl region (gripper-scale tolerance)


def detect_grasp_release(f, total):
    """From GT, find the object & bowl positions (task-space subgoals), path-invariant.
    object = EE position when the hand first closes; bowl = where it first re-opens.
    Returns (object_pos(3), bowl_pos(3), grasp_frame, release_frame)."""
    arm_pos = f["actions_arm"][:, :POS_DIM]
    hsig = f["actions_hand"][:].mean(axis=1)
    lo, hi = np.percentile(hsig, 5), np.percentile(hsig, 95)
    closed = np.clip((hsig - lo) / (hi - lo + 1e-9), 0, 1) > 0.5
    if closed.any() and (~closed).any():
        g = int(np.argmax(closed))
        after = closed[g:]
        r = g + (int(np.argmin(after)) if (~after).any() else len(after) - 1)
    else:                                   # flat hand signal -> geometry fallback
        g = int(np.argmin(arm_pos[:, 2]))   # lowest EE = at the object on the table
        r = total - 1
    return arm_pos[g], arm_pos[r], g, r


def load_frame(f, idx):
    img = f["observations/images/aria_rgb_cam/color"][idx]
    state = np.concatenate([f["observations/qpos_arm"][idx], f["observations/qpos_hand"][idx]]).astype(np.float32)
    return np.asarray(img), state


def eval_episode(policy, h5_path, frame_stride, chunk_len, prompt, gt_check=False):
    out = {}
    with h5py.File(h5_path, "r") as f:
        total = f["observations/qpos_arm"].shape[0]
        ids = list(range(0, total - chunk_len, frame_stride))
        pi0, zero, const, gd, pd = [], [], [], [], []
        for idx in ids:
            gt = np.concatenate([f["actions_arm"][idx:idx + chunk_len],
                                 f["actions_hand"][idx:idx + chunk_len]], axis=1)  # (chunk,24)
            state = np.concatenate([f["observations/qpos_arm"][idx], f["observations/qpos_hand"][idx]]).astype(np.float32)
            if gt_check:
                pred = gt                                  # self-test: GT is the "prediction"
            else:
                img, _ = load_frame(f, idx)
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

        # --- receding-horizon open-loop rollout (free-run state, teacher-force images) ---
        cur = (np.concatenate([f["observations/qpos_arm"][0], f["observations/qpos_hand"][0]])).astype(np.float32)
        gt_pos, roll_pos, roll_grip = [], [], []
        t = 0
        while t < total - 1:
            k = min(chunk_len, total - 1 - t)
            if gt_check:
                cur = np.concatenate([f["actions_arm"][t + k - 1], f["actions_hand"][t + k - 1]]).astype(np.float32)
            else:
                img, _ = load_frame(f, t)
                pred = np.asarray(policy.infer(
                    {"observation/image": img, "observation/state": cur, "prompt": prompt})["actions"])
                cur = pred[k - 1].astype(np.float32)       # advance to end of chunk (absolute EE pose)
            roll_pos.append(cur[:POS_DIM]); roll_grip.append(float(cur[ARM_DIM:].mean()))
            gt_pos.append(f["actions_arm"][min(t + k, total - 1)][:POS_DIM])
            t += k
        roll_pos = np.stack(roll_pos); gt_pos = np.stack(gt_pos); roll_grip = np.asarray(roll_grip)
        out["rollout_endpoint_err"] = float(np.linalg.norm(roll_pos[-1] - gt_pos[-1]))
        out["rollout_rmse"] = float(np.sqrt(((roll_pos - gt_pos) ** 2).sum(1).mean()))

        # --- TASK-SPACE SUBGOAL metrics (path-invariant: route doesn't matter, reaching does) ---
        obj_pos, bowl_pos, _, _ = detect_grasp_release(f, total)
        d_obj = np.linalg.norm(roll_pos - obj_pos, axis=1)
        d_bowl = np.linalg.norm(roll_pos - bowl_pos, axis=1)
        i_obj, i_bowl = int(d_obj.argmin()), int(d_bowl.argmin())
        reached_obj = d_obj.min() < SUCCESS_THRESH
        reached_bowl = d_bowl.min() < SUCCESS_THRESH
        out["reach_object_err"] = float(d_obj.min())          # closest the hand got to the object (m)
        out["reach_bowl_err"] = float(d_bowl.min())           # ... to the bowl (m)
        out["reached_object"] = float(reached_obj)
        out["reached_bowl"] = float(reached_bowl)
        # ordered success: reach object, THEN bowl (object-before-bowl in the rollout)
        out["ordered_success"] = float(reached_obj and reached_bowl and i_bowl >= i_obj)
        # gripper pattern: more closed at the object than at the bowl (grasp-then-release)
        out["gripper_ok"] = float(roll_grip[i_obj] > roll_grip[i_bowl])
    return out


def main(
    *,
    condition: str = "",
    checkpoint_dir: str = "",
    config_name: str = "pi05_egoverse",
    episodes_dir: str | None = R_ID_DIR,   # default R-ID source; restrict with --held-out-file
    episodes: list[str] = [],            # explicit held-out h5 paths (overrides episodes_dir glob)
    held_out_file: str | None = None,    # text file of h5 basenames to include from episodes_dir
    frame_stride: int = 10,
    prompt: str = "put the object in the bowl",
    finetuned: bool = True,
    gt_check: bool = False,              # self-test: score GT actions (no policy) — expect ordered_success≈1
    output_dir: str = "/cluster/scratch/lichin/pi0_test/ablation",
):
    cfg = _config.get_config(config_name)
    chunk_len = cfg.model.action_horizon
    if gt_check:
        condition = condition or "GT_check"
        policy = None
        print(f"[{condition}] GROUND-TRUTH self-test (no policy, chunk_len={chunk_len}) — expect "
              f"ordered_success≈1, reach errors≈0; use it to calibrate SUCCESS_THRESH={SUCCESS_THRESH}")
    else:
        if not checkpoint_dir:
            raise ValueError("pass --checkpoint-dir (or use --gt-check for the self-test)")
        condition = condition or "unnamed"
        if not finetuned:  # base weights: swap LoRA->plain gemma (see run_inference.py)
            cfg = dataclasses.replace(cfg, model=dataclasses.replace(
                cfg.model, paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"))
        data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
        norm_stats = normalize.load(cfg.assets_dirs / data_cfg.repo_id)
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
            m = eval_episode(policy, ep, frame_stride, chunk_len, prompt, gt_check=gt_check)
            m["episode"] = ep.name
            per_ep.append(m)
            print(f"  {ep.name}: cos={m['cos_dir']:.3f} "
                  f"reach_obj={m['reach_object_err']:.3f}m reach_bowl={m['reach_bowl_err']:.3f}m "
                  f"success={int(m['ordered_success'])} grip_ok={int(m['gripper_ok'])}")
        except Exception as e:
            print(f"  {ep.name}: SKIP ({e})")

    # Subgoal (task-space, path-invariant) metrics lead; pose-match metrics are secondary.
    keys = ["ordered_success", "reached_object", "reached_bowl", "reach_object_err", "reach_bowl_err",
            "gripper_ok", "cos_dir", "mag_ratio", "arm_mse", "arm_mse_const",
            "rollout_endpoint_err", "rollout_rmse", "pos_mse", "hand_mse", "arm_mse_zero"]
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
