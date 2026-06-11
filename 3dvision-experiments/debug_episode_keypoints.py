#!/usr/bin/env python3
"""Inspect detect_grasp_release keypoints + episode geometry for one or more R-ID h5 episodes.

Explains ordered_success quirks (e.g. 20250804_105355 fails ordering under every model while
GT passes): prints where the object/bowl keypoints are, how far apart they sit, and how close
the demo START pose is to each — if |obj-bowl| is small, the bowl-first ordering bit is noise
between two adjacent keypoints, not a behavioral finding.

Usage (login node, no GPU):
    cd ~/openpi && uv run python 3dvision-experiments/debug_episode_keypoints.py <ep.h5> [...]
With no args: runs on ALL episodes in held_out_rid.txt (table -> spot the outlier).
"""

import pathlib
import sys

import h5py
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ablation_eval import R_ID_DIR, SUCCESS_THRESH, detect_grasp_release  # noqa: E402


def inspect(path: str) -> None:
    with h5py.File(path, "r") as f:
        total = f["actions_arm"].shape[0]
        obj, bowl, g, r = detect_grasp_release(f, total)
        pos = f["actions_arm"][:, :3]
        start = pos[0]
        hsig = f["actions_hand"][:].mean(axis=1)
        lo, hi = np.percentile(hsig, 5), np.percentile(hsig, 95)
        closed = np.clip((hsig - lo) / (hi - lo + 1e-9), 0, 1) > 0.5
        n_toggle = int(np.abs(np.diff(closed.astype(int))).sum())  # >2 = re-grasp / flicker
        d_obj = np.linalg.norm(pos - obj, axis=1)
        d_bowl = np.linalg.norm(pos - bowl, axis=1)
        near_bowl_pre = float((d_bowl[:g] < SUCCESS_THRESH).mean()) if g > 0 else 0.0
        print(f"\n{pathlib.Path(path).name}  ({total} frames)")
        print(f"  grasp@{g} release@{r}  hand-signal toggles={n_toggle}{'  <-- flicker/regrasp!' if n_toggle > 2 else ''}")
        print(f"  |obj-bowl|   = {np.linalg.norm(obj - bowl) * 100:6.1f} cm{'  <-- ADJACENT (ordering bit = noise)' if np.linalg.norm(obj - bowl) < 2 * SUCCESS_THRESH else ''}")
        print(f"  |start-obj|  = {np.linalg.norm(start - obj) * 100:6.1f} cm")
        print(f"  |start-bowl| = {np.linalg.norm(start - bowl) * 100:6.1f} cm{'  <-- starts AT bowl' if np.linalg.norm(start - bowl) < SUCCESS_THRESH else ''}")
        print(f"  GT frames within {SUCCESS_THRESH*100:.0f}cm of bowl BEFORE grasp: {near_bowl_pre:.0%}")


def main() -> None:
    eps = sys.argv[1:]
    if not eps:
        held = pathlib.Path(__file__).parents[1] / "held_out_rid.txt"
        eps = [str(pathlib.Path(R_ID_DIR) / line.strip()) for line in held.read_text().splitlines() if line.strip()]
    for p in eps:
        try:
            inspect(p)
        except Exception as e:  # noqa: BLE001
            print(f"\n{p}: FAILED ({e})")


if __name__ == "__main__":
    main()
