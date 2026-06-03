#!/usr/bin/env python3
"""
extract_demo_npz.py — pull one episode's arm/hand actions + qpos into demo_actions.npz
for the Isaac replay. Type only the EPISODE ID; the long paths live here, so nothing
gets mangled by terminal line-wrapping.

Usage (on Euler, 3dv venv):
  python ~/scripts/extract_demo_npz.py 20250804_142656
  python ~/scripts/extract_demo_npz.py 20250804_142656 bag_groceries   # other task
  python ~/scripts/extract_demo_npz.py /full/path/to/episode.h5        # explicit file
"""
import os
import sys
import glob
import h5py
import numpy as np

RAW_ROOT = "/cluster/work/cvg/data/Egoverse/raw_timesynced_h5"
DEFAULT_TASK = "object_in_bowl_processed_50hz"
OUT = f"/cluster/scratch/{os.environ.get('USER', 'lichin')}/pi0_test/demo_actions.npz"


def resolve(arg, task):
    if arg.endswith(".h5") and os.path.isfile(arg):
        return arg
    ep = arg[:-3] if arg.endswith(".h5") else arg
    cand = f"{RAW_ROOT}/{task}/{ep}.h5"
    if os.path.isfile(cand):
        return cand
    # not where expected — search every task dir for it
    hits = glob.glob(f"{RAW_ROOT}/*/{ep}.h5")
    if hits:
        print(f"[note] not in '{task}', found elsewhere: {hits[0]}")
        return hits[0]
    print(f"[ERROR] episode '{ep}' not found under {RAW_ROOT}/*/")
    print("[ERROR] available tasks:", [os.path.basename(d) for d in glob.glob(f"{RAW_ROOT}/*")])
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print("usage: extract_demo_npz.py <episode_id|episode.h5|/full/path.h5> [task]")
        sys.exit(1)
    task = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TASK
    src = resolve(sys.argv[1], task)
    print(f"[load] {src}")
    with h5py.File(src, "r") as f:
        keys = {"actions_arm": "actions_arm",
                "actions_hand": "actions_hand",
                "qpos_arm": "observations/qpos_arm"}
        data = {}
        for out_key, h5_key in keys.items():
            if h5_key not in f:
                print(f"[ERROR] missing key '{h5_key}' — schema mismatch? present top-level: {list(f.keys())}")
                sys.exit(1)
            data[out_key] = f[h5_key][:]
    np.savez(OUT, **data)
    n = len(data["actions_arm"])
    print(f"[done] wrote {n} frames -> {OUT}")
    print(f"[done] arm {data['actions_arm'].shape} | hand {data['actions_hand'].shape} | qpos {data['qpos_arm'].shape}")


if __name__ == "__main__":
    main()
