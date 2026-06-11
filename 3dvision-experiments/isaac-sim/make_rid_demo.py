"""make_rid_demo.py — extract one object_in_bowl (R-ID) demo's 24-dim actions into an npz
that eval_script_object_in_bowl.py's RID_CALIBRATE block anchors against.

Run on the LOGIN node (reads raw h5 on /cluster/work; the openpi venv has h5py):
    cd ~/openpi && uv run python 3dvision-experiments/isaac-sim/make_rid_demo.py \
        /cluster/scratch/$USER/pi0_test/rid_demo.npz

Writes key 'actions24' (N,24) = arm EE pose (7) + hand (17), same layout as the model action.
The printed arm-xyz range is a FREE frame diagnostic: if it's a small cube far from the sim
ball's base location (like oic's x[0.16,0.28] y[0.21,0.50] z[0.40,0.55]) the model lives in
an egocentric frame and RID_CALIBRATE is needed; if it spans the robot's base-frame reach
around the table, the actions are already base-frame and calibration will solve ~identity.
"""
import glob
import sys

import h5py
import numpy as np

R_ID_DIR = "/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/object_in_bowl_processed_50hz"
out = sys.argv[1] if len(sys.argv) > 1 else "rid_demo.npz"
h5 = sys.argv[2] if len(sys.argv) > 2 else sorted(glob.glob(f"{R_ID_DIR}/*.h5"))[0]

with h5py.File(h5, "r") as f:
    arm = np.asarray(f["actions_arm"][:], np.float32)    # (N,7) EE pose [xyz, quat]
    hand = np.asarray(f["actions_hand"][:], np.float32)  # (N,17) ORCA hand

actions24 = np.concatenate([arm, hand], axis=1)
np.savez(out, actions24=actions24)
print(f"[make_rid_demo] {h5}\n  -> {out}  actions24={actions24.shape}")
print(f"[make_rid_demo] arm-xyz range  x[{arm[:,0].min():.3f},{arm[:,0].max():.3f}]  "
      f"y[{arm[:,1].min():.3f},{arm[:,1].max():.3f}]  z[{arm[:,2].min():.3f},{arm[:,2].max():.3f}]")
