"""rid_anchors.py — dataset-wide frame-calibration anchors from the FULL R_ID dataset.

make_rid_demo.py fit the egocentric->base transform from ONE demo's start/grasp/release.
This averages those three keypoints over ALL object_in_bowl episodes, giving a robust,
dataset-level estimate of the demo geometry (the EgoMimic point: calibrate the frame from the
data distribution, not a single trajectory). Run on the LOGIN node (reads raw h5; uv env has
h5py):

    cd ~/openpi && uv run python 3dvision-experiments/isaac-sim/rid_anchors.py \
        /cluster/scratch/$USER/pi0_test/rid_anchors.npz

Per episode it picks start / grasp / release frames the same way the calibrator does
(oic_frame_calib.pick_anchor_frames: speed-dwell minima), takes the EE xyz there, and averages
across episodes. The eval's RID_CALIBRATE block loads this (RID_ANCHORS_NPZ, default
/workspace/rid_anchors.npz) and anchors the rigid transform to these averaged points. The
printed per-axis std tells you how consistent the demos are (small std = a trustworthy anchor).
"""
import glob
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oic_frame_calib  # noqa: E402  (pick_anchor_frames: numpy-only, safe on the login node)

R_ID_DIR = "/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/object_in_bowl_processed_50hz"
out = sys.argv[1] if len(sys.argv) > 1 else "rid_anchors.npz"
files = sorted(glob.glob(f"{R_ID_DIR}/*.h5"))
print(f"[rid_anchors] {len(files)} episodes in {R_ID_DIR}")

S, G, R = [], [], []
for fn in files:
    try:
        with h5py.File(fn, "r") as f:
            arm = np.asarray(f["actions_arm"][:, :3], np.float64)   # EE xyz over the demo
        if len(arm) < 10:
            continue
        i0, ig, ir = oic_frame_calib.pick_anchor_frames(arm)
        S.append(arm[i0]); G.append(arm[ig]); R.append(arm[ir])
    except Exception as e:
        print(f"  skip {os.path.basename(fn)}: {e}")

S, G, R = np.array(S), np.array(G), np.array(R)
start3, grasp3, release3 = S.mean(0), G.mean(0), R.mean(0)
np.savez(out, start3=start3, grasp3=grasp3, release3=release3,
         start_std=S.std(0), grasp_std=G.std(0), release_std=R.std(0), n=len(S))
print(f"[rid_anchors] n={len(S)} episodes averaged")
print(f"  start   = {np.round(start3, 3).tolist()}   std {np.round(S.std(0), 3).tolist()}")
print(f"  grasp   = {np.round(grasp3, 3).tolist()}   std {np.round(G.std(0), 3).tolist()}")
print(f"  release = {np.round(release3, 3).tolist()}   std {np.round(R.std(0), 3).tolist()}")
print(f"  -> {out}  (load in the eval via RID_CALIBRATE=1; it auto-detects rid_anchors.npz)")
