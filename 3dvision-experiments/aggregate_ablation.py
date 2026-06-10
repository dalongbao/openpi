#!/usr/bin/env python3
"""aggregate_ablation.py — combine per-condition ablation_eval JSONs into one comparison
table with bootstrap CIs and paired deltas vs the R-ID baseline.

Runs locally (just reads JSONs):
  python 3dvision-experiments/aggregate_ablation.py 3dvision-experiments/results/ablation/*.json
"""
import glob
import json
import os
import sys

import numpy as np

PRIMARY = os.environ.get("ABL_PRIMARY", "reach_object_err")  # continuous reach metric (lower=better) — better
                                                             # powered than binary ordered_success on 12 episodes
BASELINE = os.environ.get("ABL_BASELINE", "rid64")           # paired deltas are vs this condition


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

    keys = ["ordered_success", "reached_object", "reached_bowl", "reach_object_err",
            "reach_bowl_err", "gripper_ok", "cos_dir", "rollout_endpoint_err"]
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
        lower_better = PRIMARY.endswith(("_err", "_mse", "_rmse"))
        direction = "negative = better (error metric)" if lower_better else "positive = better"
        print(f"\nPaired delta on {PRIMARY} vs {BASELINE} ({direction}):")
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
