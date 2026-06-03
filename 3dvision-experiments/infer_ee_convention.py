#!/usr/bin/env python3
"""
infer_ee_convention.py — infer the EE-pose convention of object_in_bowl demos
from the data alone (no Isaac Sim, no kinematics solver).

WHY: actions_arm[:7] / qpos_arm[:7] are Cartesian EE poses [x,y,z, q0,q1,q2,q3],
not joint angles. To replay/eval them via IK we must know:
  (1) FRAME      — is the pose in the robot BASE frame (what Lula IK wants) or a
                   world/Aria frame?
  (2) QUAT ORDER — is the quaternion [w,x,y,z] (scalar-first / wxyz) or
                   [x,y,z,w] (scalar-last / xyzw)?
  (3) ABS/DELTA  — are actions absolute target poses, or deltas from qpos?
  (4) CONTROL PT — flange (link8) vs hand TCP. *NOT* inferable from data alone;
                   needs FK in sim. This script narrows the later sim sweep to
                   just these 2 candidates by nailing (1)-(3).

Run on Euler (has h5py + the data):
  source ~/venvs/3dv/bin/activate
  python infer_ee_convention.py [/path/to/episode.h5 | /path/to/demo_actions.npz]

Default h5: first episode of object_in_bowl_processed_50hz.
"""
import sys
import glob
import numpy as np

DEFAULT_H5_DIR = "/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/object_in_bowl_processed_50hz"
FRANKA_REACH_M = 0.855  # FR3 max reach from base, meters


def load(path):
    """Return (actions_arm (N,7), qpos_arm (N,7) or None). Accepts .h5 or .npz."""
    if path.endswith(".npz"):
        d = np.load(path)
        a = np.asarray(d["actions_arm"], np.float64)
        q = np.asarray(d["qpos_arm"], np.float64) if "qpos_arm" in d else None
        return a, q
    import h5py
    with h5py.File(path, "r") as f:
        a = np.asarray(f["actions_arm"][:], np.float64)
        q = np.asarray(f["observations/qpos_arm"][:], np.float64) \
            if "observations/qpos_arm" in f else None
    return a, q


def report_frame(pos):
    """pos: (N,3). Infer base vs world frame from reachability."""
    print("\n" + "=" * 64)
    print("(1) FRAME  — base (IK-ready) vs world/Aria")
    print("=" * 64)
    mn, mx, mean = pos.min(0), pos.max(0), pos.mean(0)
    for i, ax in enumerate("xyz"):
        print(f"  {ax}: min {mn[i]:+.3f}  max {mx[i]:+.3f}  mean {mean[i]:+.3f}  span {mx[i]-mn[i]:.3f}")
    radius = np.linalg.norm(pos - pos.mean(0), axis=1).max()
    dist0 = np.linalg.norm(pos, axis=1)
    print(f"  max radius about cloud-center: {radius:.3f} m")
    print(f"  distance from ORIGIN: min {dist0.min():.3f}  max {dist0.max():.3f} m")
    # Heuristics
    z_mean = mean[2]
    if dist0.max() <= FRANKA_REACH_M * 1.15 and abs(z_mean) < 1.0:
        verdict = "BASE frame  -> POSE_IN_BASE = True   (positions fit FR3 workspace)"
    elif z_mean > 1.0 or dist0.min() > FRANKA_REACH_M:
        verdict = ("WORLD/Aria frame  -> POSE_IN_BASE = False  (positions outside "
                   "FR3 reach / high z). You must transform into base frame before IK.")
    else:
        verdict = "AMBIGUOUS — eyeball the numbers vs your scene geometry."
    print(f"  >>> {verdict}")


def report_quat(q4):
    """q4: (N,4) the quaternion part, in stored order. Decide scalar-first vs -last."""
    print("\n" + "=" * 64)
    print("(2) QUATERNION ORDER  — wxyz (scalar-first) vs xyzw (scalar-last)")
    print("=" * 64)
    norms = np.linalg.norm(q4, axis=1)
    print(f"  4-vec norm: mean {norms.mean():.4f}  std {norms.std():.4f}  (≈1 confirms quaternion)")
    print(f"  {'dim':>4} {'mean':>8} {'std':>7} {'min':>8} {'max':>8} {'|mean|':>7} {'frac>0':>7} sign-stable")
    stats = []
    for i in range(4):
        c = q4[:, i]
        frac_pos = float((c > 0).mean())
        sign_stable = (frac_pos > 0.97) or (frac_pos < 0.03)  # almost never crosses zero
        stats.append((abs(c.mean()), c.std(), sign_stable, frac_pos))
        flag = "YES" if sign_stable else "no"
        print(f"  {i:>4} {c.mean():>8.3f} {c.std():>7.3f} {c.min():>8.3f} {c.max():>8.3f} "
              f"{abs(c.mean()):>7.3f} {frac_pos:>7.2f}   {flag}")
    # The scalar w: sign-stable AND largest |mean|. Candidates are dim0 (wxyz) or dim3 (xyzw).
    cand = {0: stats[0], 3: stats[3]}
    def score(s):  # prefer sign-stable, then large |mean|
        absmean, std, stable, _ = s
        return (1 if stable else 0, absmean)
    w_idx = max(cand, key=lambda k: score(cand[k]))
    if w_idx == 0:
        verdict = "scalar at dim0  -> QUAT_WXYZ = True  (stored [w,x,y,z])"
    else:
        verdict = "scalar at dim3  -> QUAT_WXYZ = False (stored [x,y,z,w])"
    # confidence note
    s0, s3 = cand[0], cand[3]
    if s0[2] == s3[2] and abs(s0[0] - s3[0]) < 0.1:
        verdict += "\n  >>> WARNING: weak signal (dim0 and dim3 look similar). Confirm via the sim FK sweep."
    print(f"  >>> {verdict}")
    return w_idx


def report_abs_delta(actions, qpos):
    """Compare action pose vs qpos pose to decide absolute targets vs deltas."""
    print("\n" + "=" * 64)
    print("(3) ABSOLUTE vs DELTA actions  (+ do actions & qpos share a convention?)")
    print("=" * 64)
    if qpos is None:
        print("  qpos_arm not present in this file -> cannot test. Re-run on an h5 that has it.")
        return
    ap, qp = actions[:, :3], qpos[:, :3]
    print(f"  |action pos| typical: {np.linalg.norm(ap, axis=1).mean():.3f} m")
    print(f"  |qpos   pos| typical: {np.linalg.norm(qp, axis=1).mean():.3f} m")
    # delta hypothesis: actions are small increments centered near 0
    if np.linalg.norm(ap, axis=1).mean() < 0.05:
        print("  >>> actions look like small DELTAS (near-zero magnitude). "
              "IK target = qpos + action, not action alone.")
        return
    # absolute hypothesis: action[t] should predict qpos[t+1] (closed-loop tracking)
    # report per-axis correlation between action[t] and qpos[t] (lag 0) as a sanity check
    for lag in (0, 1):
        a = ap[:len(ap) - lag]
        b = qp[lag:]
        cors = [np.corrcoef(a[:, k], b[:, k])[0, 1] for k in range(3)]
        rms = np.sqrt(((a - b) ** 2).sum(1)).mean()
        print(f"  lag {lag}: per-axis corr(action[t], qpos[t+{lag}]) = "
              f"[{cors[0]:.3f} {cors[1]:.3f} {cors[2]:.3f}]   mean RMS dist {rms:.3f} m")
    print("  >>> If corr≈1 and RMS small at lag 0/1: actions are ABSOLUTE EE targets "
          "in the SAME frame as qpos (solve one convention, both are solved).")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        hits = sorted(glob.glob(f"{DEFAULT_H5_DIR}/*.h5"))
        if not hits:
            print(f"No h5 found in {DEFAULT_H5_DIR}; pass a path explicitly.")
            sys.exit(1)
        path = hits[0]
    print(f"[load] {path}")
    actions, qpos = load(path)
    print(f"[load] actions_arm {actions.shape}" + (f" | qpos_arm {qpos.shape}" if qpos is not None else " | qpos_arm MISSING"))
    if actions.shape[1] != 7:
        print(f"[WARN] expected 7-D arm pose, got {actions.shape[1]}-D — is this the right key?")

    report_frame(actions[:, :3])
    w_idx = report_quat(actions[:, 3:7])
    report_abs_delta(actions, qpos)

    print("\n" + "#" * 64)
    print("# RECOMMENDED eval_replay_ik.py TUNABLES (verify control-point in sim)")
    print("#" * 64)
    print(f"#   QUAT_WXYZ    = {w_idx == 0}")
    print("#   POSE_IN_BASE = <see verdict (1) above>")
    print("#   EE_FRAME     = sweep {flange 'panda_link8'/'fr3_link8'} vs {TCP 'right_gripper'/'panda_hand'}")
    print("#                  in sim — this is the ONE unknown data can't resolve.")
    print("#" * 64)


if __name__ == "__main__":
    main()
