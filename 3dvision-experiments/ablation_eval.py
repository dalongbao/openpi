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

POSITION (action dims 0-2) is the cross-model yardstick: it is shared by the 24-dim robot
space and the 6-dim human/oic EE-only space, so models of EITHER dim are scored on the SAME
held-out R-ID frames with NO retraining/convention-unification. Pass `--state-dim 6` for a
6-dim model: its input state is rebuilt from the robot frame as [xyz + Euler], and only
position metrics (cos_dir, mag_ratio, pos_mse, rollout, reach/ordered-success) are reported;
rotation/hand metrics (arm_mse/hand_mse/gripper_ok) are N/A (NaN). CAVEAT: assumes robot and
human POSITION share a frame/scale — if they differ by a rotation, the 6-dim model's score is
a lower bound (frame mismatch penalizes it), not a clean capability read.

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
from scipy.spatial.transform import Rotation as _Rotation

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


def make_state(vec, state_dim):
    """Build the policy input state at the model's expected dim from a robot state vector.
    24-dim robot models get the native [arm EE pose(7: xyz+quat xyzw) + hand(17)]. A 6-dim
    (human/oic) model gets [xyz + Euler(xyz)] — rotation order is a best-effort guess (the
    oic Euler convention is unknown), so only POSITION is trusted downstream."""
    vec = np.asarray(vec, dtype=np.float32)
    if len(vec) == state_dim:
        return vec                                  # already in the model's space (e.g. rollout feedback)
    if state_dim >= ACTION_DIM:
        return vec                                  # 24-dim robot space, pass through
    eul = _Rotation.from_quat(vec[3:ARM_DIM]).as_euler("xyz")  # quat xyzw -> 3 Euler
    return np.concatenate([vec[:POS_DIM], eul]).astype(np.float32)


def eval_episode(policy, h5_path, frame_stride, chunk_len, prompt, gt_check=False, state_dim=ACTION_DIM):
    """Score one R-ID episode. POSITION (dims 0-2) is the cross-model yardstick and works for
    any action dim; the full pose/hand metrics are computed only for 24-dim (robot-space) models."""
    out = {}
    with h5py.File(h5_path, "r") as f:
        total = f["observations/qpos_arm"].shape[0]
        ids = list(range(0, total - chunk_len, frame_stride))
        pos_sq, full_sq, zero_sq, const_sq, gd, pd = [], [], [], [], [], []
        for idx in ids:
            gt = np.concatenate([f["actions_arm"][idx:idx + chunk_len],
                                 f["actions_hand"][idx:idx + chunk_len]], axis=1)  # (chunk,24)
            state24 = np.concatenate([f["observations/qpos_arm"][idx], f["observations/qpos_hand"][idx]]).astype(np.float32)
            if gt_check:
                pred = gt                                  # self-test: GT is the "prediction"
            else:
                img, _ = load_frame(f, idx)
                pred = np.asarray(policy.infer(
                    {"observation/image": img, "observation/state": make_state(state24, state_dim),
                     "prompt": prompt})["actions"])[:chunk_len]
            # position (dims 0-2) is shared across the 24-dim and 6-dim action spaces -> the comparable metric
            pos_sq.append((pred[:, :POS_DIM] - gt[:, :POS_DIM]) ** 2)
            gd.append(gt[-1, :POS_DIM] - state24[:POS_DIM]); pd.append(pred[-1, :POS_DIM] - state24[:POS_DIM])
            if pred.shape[1] >= ACTION_DIM:                # full pose+hand metrics: robot-space (24-dim) only
                full_sq.append((pred - gt) ** 2); zero_sq.append(gt ** 2); const_sq.append((state24[None] - gt) ** 2)

        pos_sq = np.concatenate(pos_sq)
        gd = np.stack(gd); pd = np.stack(pd)
        gmag = np.linalg.norm(gd, axis=1); pmag = np.linalg.norm(pd, axis=1); mv = gmag > 1e-3
        out.update(
            n_chunks=len(ids),
            pos_mse=float(pos_sq.mean()),
            mag_ratio=float(pmag[mv].mean() / (gmag[mv].mean() + 1e-9)),
            cos_dir=float(np.nanmean(np.sum(pd * gd, 1)[mv] / (pmag[mv] * gmag[mv] + 1e-9))),
        )
        if full_sq:
            full_sq = np.concatenate(full_sq); zero_sq = np.concatenate(zero_sq); const_sq = np.concatenate(const_sq)
            out.update(arm_mse=float(full_sq[:, :ARM_DIM].mean()), hand_mse=float(full_sq[:, ARM_DIM:].mean()),
                       arm_mse_zero=float(zero_sq[:, :ARM_DIM].mean()), arm_mse_const=float(const_sq[:, :ARM_DIM].mean()))
        else:                                              # 6-dim model: rotation rep/hand not comparable
            out.update(arm_mse=float("nan"), hand_mse=float("nan"), arm_mse_zero=float("nan"), arm_mse_const=float("nan"))

        # --- receding-horizon open-loop rollout (free-run state, teacher-force images); position-only ---
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
                    {"observation/image": img, "observation/state": make_state(cur, state_dim),
                     "prompt": prompt})["actions"])
                cur = pred[k - 1].astype(np.float32)       # advance to end of chunk (absolute EE pose, native dim)
            roll_pos.append(cur[:POS_DIM])
            if len(cur) >= ACTION_DIM:                     # hand dims present -> track gripper actuation
                roll_grip.append(float(cur[ARM_DIM:].mean()))
            gt_pos.append(f["actions_arm"][min(t + k, total - 1)][:POS_DIM])
            t += k
        roll_pos = np.stack(roll_pos); gt_pos = np.stack(gt_pos)
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
        # gripper pattern: hand more actuated during transport (object->bowl) than before pickup,
        # normalized per-episode. Window means are robust vs single-point comparison.
        # Only meaningful for models with hand dims (24-dim); N/A for the 6-dim EE-only space.
        if roll_grip:
            roll_grip = np.asarray(roll_grip)
            gn = np.clip((roll_grip - np.percentile(roll_grip, 5)) /
                         (np.percentile(roll_grip, 95) - np.percentile(roll_grip, 5) + 1e-9), 0.0, 1.0)
            i0, i1 = min(i_obj, i_bowl), max(i_obj, i_bowl)
            transport = float(gn[i0:i1 + 1].mean()) if i1 > i0 else float(gn[i_obj])
            before = float(gn[:max(i_obj, 1)].mean())
            out["gripper_ok"] = float(transport > before)   # GT expect ~1; if ~0 the hand sign is flipped
        else:
            out["gripper_ok"] = float("nan")
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
    state_dim: int = ACTION_DIM,        # 24 = robot space; 6 = human/oic EE-only model (POSITION-only scoring)
    norm_stats_dir: str | None = None,  # override norm-stats path (e.g. <ckpt>/assets/egoverse/oic_human)
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
        ns_path = pathlib.Path(norm_stats_dir) if norm_stats_dir else cfg.assets_dirs / data_cfg.repo_id
        norm_stats = normalize.load(ns_path)
        print(f"[{condition}] {config_name} <- {checkpoint_dir}  (chunk_len={chunk_len}, state_dim={state_dim})")
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
            m = eval_episode(policy, ep, frame_stride, chunk_len, prompt, gt_check=gt_check, state_dim=state_dim)
            m["episode"] = ep.name
            per_ep.append(m)
            grip = "-" if m["gripper_ok"] != m["gripper_ok"] else int(m["gripper_ok"])  # nan -> "-"
            print(f"  {ep.name}: cos={m['cos_dir']:.3f} "
                  f"reach_obj={m['reach_object_err']:.3f}m reach_bowl={m['reach_bowl_err']:.3f}m "
                  f"success={int(m['ordered_success'])} grip_ok={grip}")
        except Exception as e:
            print(f"  {ep.name}: SKIP ({e})")

    # Subgoal (task-space, path-invariant) metrics lead; pose-match metrics are secondary.
    keys = ["ordered_success", "reached_object", "reached_bowl", "reach_object_err", "reach_bowl_err",
            "gripper_ok", "cos_dir", "mag_ratio", "arm_mse", "arm_mse_const",
            "rollout_endpoint_err", "rollout_rmse", "pos_mse", "hand_mse", "arm_mse_zero"]
    summary = {k: float(np.nanmean([m[k] for m in per_ep])) for k in keys}  # nanmean: skip N/A (6-dim) metrics
    summary["n_episodes"] = len(per_ep)
    print(f"\n[{condition}] SUMMARY: " + "  ".join(f"{k}={summary[k]:.4f}" for k in keys))

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    out = pathlib.Path(output_dir) / f"{condition}.json"
    json.dump({"condition": condition, "config": config_name, "checkpoint": checkpoint_dir,
               "state_dim": state_dim, "summary": summary, "per_episode": per_ep}, open(out, "w"), indent=2)
    print(f"[{condition}] wrote {out}")


if __name__ == "__main__":
    tyro.cli(main)
