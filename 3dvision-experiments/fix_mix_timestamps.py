"""fix_mix_timestamps.py — relabel human episode timestamps to the dataset's declared fps.

The image-linking merge copied oic_human's 30 Hz timestamps into oic_mix, but the dataset is
fps=50 and training gathers 10-step action chunks at 0.02 s (50 Hz) intervals. With 30 Hz human
frames (0.0333 s apart) the chunk lookups miss LeRobot's tolerance and the dataset fails to load.

Fix: rewrite the `timestamp` column of every HUMAN episode (episode_index >= --first-human) to
`frame_index / fps`. Rewrites ONLY that column (image bytes preserved via pyarrow). Robot episodes
(below --first-human) are already 50 Hz and are left untouched. Idempotent — safe to re-run.

  ~/venvs/3dv/bin/python fix_mix_timestamps.py --root /cluster/scratch/<user>/lerobot/egoverse/oic_mix
then re-run the load test (compute_norm_stats) to confirm it loads.
"""
import argparse
import glob
import os
import re

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _ep_num(path: str) -> int:
    m = re.search(r"episode_(\d+)\.parquet", os.path.basename(path))
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--first-human", type=int, default=64,
                    help="episode_index of the first human episode (robot are below this, already correct)")
    a = ap.parse_args()

    fs = sorted(glob.glob(os.path.join(a.root, "data", "**", "*.parquet"), recursive=True))
    todo = [f for f in fs if _ep_num(f) >= a.first_human]
    print(f"{len(fs)} parquet files; fixing {len(todo)} human episodes (idx >= {a.first_human}) "
          f"-> timestamp = frame_index/{a.fps}")
    for k, f in enumerate(todo):
        t = pq.read_table(f)
        new_ts = pa.array((np.arange(t.num_rows) / a.fps).astype(np.float32))
        i = t.schema.get_field_index("timestamp")
        if i < 0:
            raise SystemExit(f"no 'timestamp' column in {f}")
        pq.write_table(t.set_column(i, "timestamp", new_ts), f)
        if (k + 1) % 200 == 0:
            print(f"  {k + 1}/{len(todo)}")
    print("done. Re-run `compute_norm_stats --config-name pi05_ego_mix_oic` to confirm it now loads.")


if __name__ == "__main__":
    main()
