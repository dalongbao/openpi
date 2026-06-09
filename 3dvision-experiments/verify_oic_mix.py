"""verify_oic_mix.py — sanity-check the assembled egoverse/oic_mix dataset (correctness, not just "it ran").

Classifies each episode robot (action_mask sum=24) vs human (=7); checks 24D shapes and no NaN;
confirms the human HAND slots are zero in actions (the masking precondition); and compares robot
vs human ARM-position ranges (both should now be base-frame meters of comparable scale -> the
head->base remap is sane). Reads parquet directly, so it runs in the 3dv venv (pandas only):

  ~/venvs/3dv/bin/python ~/openpi/3dvision-experiments/verify_oic_mix.py \
     --root /cluster/scratch/lichin/lerobot/egoverse/oic_mix
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

ARM_POS, ARM_DIM, ACTION_DIM = 3, 7, 24


def _stk(col):
    return np.stack([np.asarray(v, dtype=np.float64).reshape(-1) for v in col])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()

    info = os.path.join(a.root, "meta", "info.json")
    if os.path.isfile(info):
        d = json.load(open(info))
        print(f"[info] episodes={d.get('total_episodes')} frames={d.get('total_frames')} fps={d.get('fps')}")
        for k, v in d.get("features", {}).items():
            print(f"   feat {k:12s} dtype={v.get('dtype')} shape={v.get('shape')}")

    fs = sorted(glob.glob(os.path.join(a.root, "data", "**", "*.parquet"), recursive=True))
    print(f"\n[data] {len(fs)} episode parquet files")
    n_rob = n_hum = n_other = nan = 0
    rob_pos, hum_pos, hum_hand_max = [], [], 0.0
    for f in fs:
        df = pd.read_parquet(f, columns=["action_mask", "actions", "state"])
        m, act, st = _stk(df["action_mask"]), _stk(df["actions"]), _stk(df["state"])
        if act.shape[1] != ACTION_DIM or st.shape[1] != ACTION_DIM or m.shape[1] != ACTION_DIM:
            print(f"  BAD SHAPE {os.path.basename(f)} act{act.shape} st{st.shape} mask{m.shape}"); n_other += 1; continue
        if not (np.isfinite(act).all() and np.isfinite(st).all()):
            nan += 1
        msum = m.mean(0).sum()
        if np.isclose(msum, ACTION_DIM):
            n_rob += 1; rob_pos.append(act[:, :ARM_POS])
        elif np.isclose(msum, ARM_DIM):
            n_hum += 1; hum_pos.append(act[:, :ARM_POS])
            hum_hand_max = max(hum_hand_max, float(np.abs(act[:, ARM_DIM:]).max()))
        else:
            n_other += 1; print(f"  ODD mask sum={msum:.1f} in {os.path.basename(f)}")

    print(f"\n[mask] robot(=24): {n_rob}   human(=7): {n_hum}   other: {n_other}   episodes_with_nan: {nan}")
    if rob_pos:
        r = np.concatenate(rob_pos)
        print(f"[robot arm-pos m] min={np.round(r.min(0), 3)} max={np.round(r.max(0), 3)}")
    if hum_pos:
        h = np.concatenate(hum_pos)
        print(f"[human arm-pos m] min={np.round(h.min(0), 3)} max={np.round(h.max(0), 3)}")
        print(f"[human hand slots] max|val| in actions = {hum_hand_max:.6f}   (MUST be ~0)")
    ok = n_rob > 0 and n_hum > 0 and n_other == 0 and nan == 0 and (not hum_pos or hum_hand_max < 1e-5)
    print("\n" + ("PASS — mix looks correct" if ok else "CHECK FAILED — see flags above"))


if __name__ == "__main__":
    main()
