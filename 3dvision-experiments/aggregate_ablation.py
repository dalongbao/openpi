#!/usr/bin/env python3
"""aggregate_ablation.py — combine per-condition ablation_eval JSONs into one comparison
table with bootstrap CIs and paired deltas vs the R-ID baseline.

Runs locally (just reads JSONs):
  python 3dvision-experiments/aggregate_ablation.py 3dvision-experiments/results/ablation/*.json
"""
import glob
import json
import sys

import numpy as np

PRIMARY = "cos_dir"   # headline metric (higher=better); rollout_endpoint_err is lower=better
BASELINE = "R-ID_only"


def boot_ci(x, n=10000, alpha=0.05):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float(x.mean()) if len(x) else float("nan"),) * 2
    idx = np.random.randint(0, len(x), (n, len(x)))
    means = x[idx].mean(1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def main(paths):
    conds = {}
    for p in paths:
        d = json.load(open(p))
        conds[d["condition"]] = d
    if not conds:
        print("no JSONs found"); return

    keys = ["cos_dir", "mag_ratio", "arm_mse", "arm_mse_const", "rollout_endpoint_err", "rollout_rmse"]
    print(f"{'condition':<16} {'N':>3} " + " ".join(f"{k:>16}" for k in keys))
    for c, d in sorted(conds.items()):
        pe = d["per_episode"]; n = len(pe)
        cells = []
        for k in keys:
            vals = [m[k] for m in pe]
            lo, hi = boot_ci(vals)
            cells.append(f"{np.mean(vals):.3f}[{lo:.2f},{hi:.2f}]")
        print(f"{c:<16} {n:>3} " + " ".join(f"{x:>16}" for x in cells))

    # paired deltas vs baseline on the primary metric (same held-out episodes => paired)
    if BASELINE in conds:
        base = {m["episode"]: m[PRIMARY] for m in conds[BASELINE]["per_episode"]}
        print(f"\nPaired delta on {PRIMARY} vs {BASELINE} (positive = better than teleop baseline):")
        for c, d in sorted(conds.items()):
            if c == BASELINE:
                continue
            diffs = [m[PRIMARY] - base[m["episode"]] for m in d["per_episode"] if m["episode"] in base]
            if not diffs:
                continue
            lo, hi = boot_ci(diffs)
            sig = "*" if (lo > 0 or hi < 0) else " "
            print(f"  {c:<16} d={np.mean(diffs):+.3f}  95%CI[{lo:+.3f},{hi:+.3f}] {sig}")
        print("  (* = CI excludes 0 -> significant difference from the teleop baseline)")


if __name__ == "__main__":
    args = sys.argv[1:]
    files = [f for a in args for f in glob.glob(a)] or glob.glob("3dvision-experiments/results/ablation/*.json")
    main(files)
