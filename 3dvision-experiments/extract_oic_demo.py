#!/usr/bin/env python3
"""
extract_oic_demo.py — pull one oic_human (object-in-container) LeRobot episode's 6-dim
actions [x,y,z, e1,e2,e3] + a few reference frames, for the Euler-order GT replay.

The reference PNGs show the HUMAN hand (Aria egocentric) — compare the sim replay's
gripper orientation against these to pick the correct EULER_ORDER.

Run on the Euler login node (3dv venv: pyarrow, pillow, numpy):
  python ~/scripts/extract_oic_demo.py            # episode 0
  python ~/scripts/extract_oic_demo.py 1234       # a specific episode index
"""
import io
import os
import sys
import numpy as np

ROOT = "/cluster/work/cvg/data/Egoverse/lerobot_egoverse/egoverse/oic_human"
OUT_NPZ = f"/cluster/scratch/{os.environ.get('USER','lichin')}/pi0_test/oic_demo.npz"
OUT_DIR = os.path.dirname(OUT_NPZ)
CHUNK = 1000   # chunks_size from meta/info.json


def main():
    ep = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    pq_path = f"{ROOT}/data/chunk-{ep // CHUNK:03d}/episode_{ep:06d}.parquet"
    if not os.path.isfile(pq_path):
        print(f"[err] not found: {pq_path}"); sys.exit(1)
    print(f"[load] {pq_path}")

    import pyarrow.parquet as pq
    from PIL import Image
    t = pq.read_table(pq_path)
    rows = t.to_pylist()
    n = len(rows)
    actions = np.asarray([r["actions"] for r in rows], dtype=np.float64)   # (N,6)
    state   = np.asarray([r["state"]   for r in rows], dtype=np.float64)   # (N,6)
    print(f"[load] {n} frames | actions {actions.shape} | state {state.shape}")
    print(f"[load] pos ranges: x[{actions[:,0].min():.3f},{actions[:,0].max():.3f}] "
          f"y[{actions[:,1].min():.3f},{actions[:,1].max():.3f}] z[{actions[:,2].min():.3f},{actions[:,2].max():.3f}]")
    print(f"[load] euler ranges: e1[{actions[:,3].min():.2f},{actions[:,3].max():.2f}] "
          f"e2[{actions[:,4].min():.2f},{actions[:,4].max():.2f}] e3[{actions[:,5].min():.2f},{actions[:,5].max():.2f}]")

    np.savez(OUT_NPZ, actions6=actions, state6=state, episode=ep, fps=30)
    print(f"[done] wrote {OUT_NPZ}")

    # decode a few reference frames (first / quarters / last) for orientation comparison
    idxs = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
    for i in idxs:
        img = rows[i]["image"]
        data = img["bytes"] if isinstance(img, dict) else img
        try:
            im = Image.open(io.BytesIO(data)).convert("RGB")
            out = f"{OUT_DIR}/oic_demo_ref_f{i:03d}.png"
            im.save(out)
            print(f"[ref ] frame {i:3d} -> {out}  pose6={np.round(actions[i],3).tolist()}")
        except Exception as e:
            print(f"[ref ] frame {i}: could not decode image ({e})")

    print("\nNext: render the Euler-order montage with eval_replay_oic.py and compare the")
    print("sim gripper orientation in each panel against these oic_demo_ref_*.png frames.")


if __name__ == "__main__":
    main()
