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
held-out R-ID frames with NO retraining. Pass `--state-dim 6` for a 6-dim model: its input
state is rebuilt as [xyz + yaw,pitch,roll] in the RESOLVED oic convention (intrinsic ZYX,
radians) and remapped base->head via egoverse_unify; its predicted positions are mapped back
head->base so all metrics live in the robot base frame. Only position metrics (cos_dir,
mag_ratio, pos_mse, rollout, reach/ordered-success) are reported; rotation/hand metrics
(arm_mse/hand_mse/gripper_ok) are N/A (NaN). CAVEAT: the head<->base ORIENTATION map is a
hypothesis — the always-on Kabsch alignment (cos_dir vs cos_dir_aligned) diagnoses a residual
fixed-frame mismatch; `--no-human-frame-remap` reproduces the raw (unremapped) behavior. A
6-dim model's score remains a LOWER BOUND, not a clean capability read.

Writes a JSON of per-episode + summary metrics; aggregate across conditions with
aggregate_ablation.py.

Run on Euler via uv (mirror run_inference.slurm):
  sbatch --export=ALL,UV_FROZEN=1 --time=01:00:00 --mem-per-cpu=16G --cpus-per-task=8 \
    --gpus=rtx_4090:1 3dvision-experiments/run_ablation_eval.slurm \
    --config-name pi05_egoverse --checkpoint-dir <ckpt> --condition R-ID_only \
    --episodes-dir <held_out_dir>
"""
import dataclasses
import hashlib
import json
import pathlib
import subprocess

import h5py
import numpy as np
import tyro
from scipy.spatial.transform import Rotation as _Rotation

from openpi.policies import egoverse_unify as _unify
from openpi.policies import policy_config
from openpi.shared import normalize
from openpi.training import config as _config

R_ID_DIR = "/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/object_in_bowl_processed_50hz"  # R-ID source
ARM_DIM = 7        # EE pose [x,y,z, qx,qy,qz,qw]
POS_DIM = 3        # [x,y,z] — direction metric uses this (rep/frame-agnostic)
ACTION_DIM = 24    # 7 arm + 17 hand
SUCCESS_THRESH = 0.08   # m; "reached" the object/bowl region (gripper-scale tolerance)


def kabsch(P, Q):
    """Best proper rotation R and LS scale s mapping pred vectors P (N,3) onto GT Q (N,3):
    argmin ||s R p_i - q_i||. Vectors (displacements), so no centering. Returns (R, s)."""
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T          # R @ p_i ≈ q_i
    RP = P @ R.T
    s = float((RP * Q).sum() / ((P * P).sum() + 1e-12))   # cosine is scale-free; s is just reported
    return R, s


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


def make_state(vec, state_dim, frame_remap=True):
    """Build the policy input state at the model's expected dim from a robot state vector.
    24-dim robot models get the native [arm EE pose(7: xyz+quat xyzw) + hand(17)]. A 6-dim
    (human/oic) model gets [xyz + yaw,pitch,roll] in the RESOLVED oic convention (intrinsic
    ZYX, radians — EgoMimic pose_utils), with the robot base frame remapped into the egocentric
    head frame (egoverse_unify.HEAD_TO_BASE transposed) so the model sees in-distribution
    states. Rotation remains best-effort (the head<->base ORIENTATION map is a hypothesis);
    POSITION is the trusted axis downstream. `--no-human-frame-remap` reverts to the raw
    base-frame state (the pre-fix behavior, for A/B-ing the remap itself)."""
    vec = np.asarray(vec, dtype=np.float32)
    if len(vec) == state_dim:
        return vec                                  # already in the model's space (e.g. rollout feedback)
    if state_dim >= ACTION_DIM:
        return vec                                  # 24-dim robot space, pass through
    # reuse the converter's OWN base->canonical(=head) map so eval and mix-build share one
    # convention hypothesis (canonical = the EgoMimic head frame the human data lives in)
    arm7 = _unify.robot_arm7_base_to_canonical(vec[:ARM_DIM])[0] if frame_remap else vec[:ARM_DIM]
    ypr = _Rotation.from_quat(arm7[3:7]).as_euler("ZYX")          # intrinsic ZYX -> [yaw, pitch, roll]
    return np.concatenate([arm7[:POS_DIM], ypr]).astype(np.float32)


def to_base_pos(pos, state_dim, frame_remap=True):
    """Map predicted POSITIONS back into the ROBOT BASE frame so every metric is computed in
    one frame. Identity for 24-dim (robot-space) models or when the remap is disabled.
    `pos` is (..., 3) row-stacked head-frame vectors; v_base = HEAD_TO_BASE @ v_head."""
    if state_dim >= ACTION_DIM or not frame_remap:
        return pos
    return pos @ _unify.HEAD_TO_BASE.T


def eval_episode(policy, h5_path, frame_stride, chunk_len, prompt, gt_check=False, state_dim=ACTION_DIM,
                 frame_remap=True):
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
                    {"observation/image": img, "observation/state": make_state(state24, state_dim, frame_remap),
                     "prompt": prompt})["actions"])[:chunk_len]
            # position (dims 0-2) is shared across the 24-dim and 6-dim action spaces -> the comparable
            # metric. 6-dim (head-frame) predictions are mapped back to the BASE frame first so all
            # downstream metrics live in one frame.
            ppos = to_base_pos(pred[:, :POS_DIM], state_dim, frame_remap)
            pos_sq.append((ppos - gt[:, :POS_DIM]) ** 2)
            gd.append(gt[-1, :POS_DIM] - state24[:POS_DIM]); pd.append(ppos[-1] - state24[:POS_DIM])
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
        out["_gd"] = gd[mv]          # raw moving GT/pred displacements (3-vecs) for the global Kabsch fit
        out["_pd"] = pd[mv]          # stripped before JSON in main()
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
                    {"observation/image": img, "observation/state": make_state(cur, state_dim, frame_remap),
                     "prompt": prompt})["actions"])
                cur = pred[k - 1].astype(np.float32)       # advance to end of chunk (absolute EE pose, native dim)
            roll_pos.append(to_base_pos(cur[None, :POS_DIM], state_dim, frame_remap)[0])
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
        # bare ordering bit (threshold-free: argmin times). With reach_*_err this lets any
        # success threshold be recomputed offline: success(t) = (obj_err<t)&(bowl_err<t)&order_ok
        out["subgoal_order_ok"] = float(i_bowl >= i_obj)
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
    align: bool = True,                 # fit ONE global rotation+scale (Kabsch) pred->GT, report post-align cos
    human_frame_remap: bool = True,     # 6-dim models: remap state base->head + predictions head->base
    output_dir: str = "/cluster/scratch/lichin/pi0_test/ablation",
):
    cfg = _config.get_config(config_name)
    chunk_len = cfg.model.action_horizon
    ns_path = ns_md5 = None
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
        # Norm-stats resolution (train/eval mismatch silently corrupts every metric — prefer the
        # snapshot train.py wrote INSIDE the checkpoint, which always matches how it was trained):
        #   1. explicit --norm-stats-dir   2. <ckpt>/assets/<repo_id> (finetuned only)   3. local checkout
        ckpt_ns = pathlib.Path(checkpoint_dir) / "assets" / data_cfg.repo_id
        if norm_stats_dir:
            ns_path = pathlib.Path(norm_stats_dir)
        elif finetuned and (ckpt_ns / "norm_stats.json").exists():
            ns_path = ckpt_ns
        else:
            ns_path = cfg.assets_dirs / data_cfg.repo_id
            if finetuned:
                print(f"WARNING: no norm-stats snapshot inside the checkpoint ({ckpt_ns});\n"
                      f"  falling back to the local checkout's {ns_path} — VERIFY these are the stats this\n"
                      f"  checkpoint was TRAINED with (a train/eval normalizer mismatch corrupts every metric).")
        norm_stats = normalize.load(ns_path)
        ns_md5 = hashlib.md5((pathlib.Path(ns_path) / "norm_stats.json").read_bytes()).hexdigest()
        print(f"[{condition}] {config_name} <- {checkpoint_dir}  (chunk_len={chunk_len}, state_dim={state_dim})")
        print(f"[{condition}] norm stats: {ns_path}  (md5 {ns_md5})")
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
    gd_all, pd_all = [], []          # moving displacement pairs across episodes, for the global Kabsch fit
    for ep in eps:
        try:
            m = eval_episode(policy, ep, frame_stride, chunk_len, prompt, gt_check=gt_check, state_dim=state_dim,
                             frame_remap=human_frame_remap)
            gd_all.append(m.pop("_gd")); pd_all.append(m.pop("_pd"))   # strip arrays (not JSON-serializable)
            m["episode"] = ep.name
            per_ep.append(m)
            grip = "-" if m["gripper_ok"] != m["gripper_ok"] else int(m["gripper_ok"])  # nan -> "-"
            print(f"  {ep.name}: cos={m['cos_dir']:.3f} "
                  f"reach_obj={m['reach_object_err']:.3f}m reach_bowl={m['reach_bowl_err']:.3f}m "
                  f"success={int(m['ordered_success'])} grip_ok={grip}")
        except Exception as e:
            print(f"  {ep.name}: SKIP ({e})")

    if not per_ep:
        raise RuntimeError(f"[{condition}] ALL {len(eps)} episodes failed — no JSON written, eval is void")

    # Subgoal (task-space, path-invariant) metrics lead; pose-match metrics are secondary.
    keys = ["ordered_success", "reached_object", "reached_bowl", "reach_object_err", "reach_bowl_err",
            "gripper_ok", "cos_dir", "mag_ratio", "arm_mse", "arm_mse_const",
            "rollout_endpoint_err", "rollout_rmse", "pos_mse", "hand_mse", "arm_mse_zero"]
    summary = {k: float(np.nanmean([m[k] for m in per_ep])) for k in keys}  # nanmean: skip N/A (6-dim) metrics
    summary["n_episodes"] = len(per_ep)
    print(f"\n[{condition}] SUMMARY: " + "  ".join(f"{k}={summary[k]:.4f}" for k in keys))

    if align:
        # ONE global rotation+scale best-fitting pred displacements -> GT (disentangles a frame/scale
        # mismatch from genuine no-signal). If raw cos~0 but aligned cos jumps high, it's purely a
        # convention mismatch (a frame conversion would recover it); if aligned cos stays ~0, no transfer.
        from scipy.spatial.transform import Rotation as _R
        G = np.concatenate(gd_all); P = np.concatenate(pd_all)
        R, s = kabsch(P, G)
        RP = P @ R.T
        gmag = np.linalg.norm(G, axis=1); pmag = np.linalg.norm(RP, axis=1); mv = (gmag > 1e-3) & (pmag > 1e-9)
        aligned = float(np.nanmean((RP[mv] * G[mv]).sum(1) / (pmag[mv] * gmag[mv])))
        euler = _R.from_matrix(R).as_euler("xyz", degrees=True)
        summary["cos_dir_aligned"] = aligned
        summary["align_scale"] = s
        summary["align_rot_euler_deg"] = [float(x) for x in euler]
        print(f"[{condition}] ALIGNED (global Kabsch, n={mv.sum()} chunks): "
              f"cos_dir {summary['cos_dir']:.3f} -> {aligned:.3f}  | scale={s:.2f}  "
              f"| rot(xyz,deg)=[{euler[0]:.0f},{euler[1]:.0f},{euler[2]:.0f}]")
        raw = summary["cos_dir"]
        verdict = ("frames already agree (no mismatch)" if raw > 0.4 and aligned - raw < 0.2
                   else "frame/scale mismatch (convertible)" if aligned > 0.4
                   else "no transfer even after alignment")
        print(f"[{condition}] -> {verdict}")

    try:  # best-effort provenance: which code produced this JSON
        git_rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                          cwd=pathlib.Path(__file__).parent, text=True).strip()
    except Exception:
        git_rev = None
    skipped = len(eps) - len(per_ep)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    out = pathlib.Path(output_dir) / f"{condition}.json"
    json.dump({"condition": condition, "config": config_name, "checkpoint": checkpoint_dir,
               "state_dim": state_dim, "summary": summary, "per_episode": per_ep,
               "provenance": {"norm_stats_path": str(ns_path) if ns_path else None, "norm_stats_md5": ns_md5,
                              "prompt": prompt, "frame_stride": frame_stride, "finetuned": finetuned,
                              "human_frame_remap": human_frame_remap, "n_skipped": skipped,
                              "git": git_rev}}, open(out, "w"), indent=2)
    print(f"[{condition}] wrote {out}")
    if skipped:
        print(f"[{condition}] WARNING: {skipped}/{len(eps)} episodes were SKIPPED — metrics cover a subset.")
        if skipped > len(eps) // 2:
            raise SystemExit(f"[{condition}] more than half the episodes failed; treat {out} as INVALID")


if __name__ == "__main__":
    tyro.cli(main)
