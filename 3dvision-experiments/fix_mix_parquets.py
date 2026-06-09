"""fix_mix_parquets.py — make merged oic_mix parquets loadable by HF datasets / LeRobot.

Fixes BOTH bugs an image-linking merge can introduce, in one pass (no image re-encode):

  1. 'Nested data conversions not implemented for chunked array outputs' — the merge wrote the
     nested `image` column spanning MULTIPLE row groups/chunks; HF `load_dataset` can't convert it.
     Fix: rewrite each parquet as a SINGLE row group with combined chunks (matches what
     LeRobotDataset.create produces, which loads fine).
  2. Human (30 Hz) timestamps in a 50 Hz dataset -> the 50 Hz action-chunk lookups miss tolerance.
     Fix: relabel human (episode_index >= --first-human) `timestamp` to frame_index/fps.

Idempotent. Rewrites every parquet (~30-60 min IO for the full mix). Run by whoever owns the dataset:
  uv run python 3dvision-experiments/fix_mix_parquets.py --root /cluster/scratch/<user>/lerobot/egoverse/oic_mix
then re-run: compute_norm_stats --config-name pi05_ego_mix_oic --max_frames 20000
"""
import argparse
import glob
import os
import re

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _ep(path: str) -> int:
    m = re.search(r"episode_(\d+)\.parquet", os.path.basename(path))
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--first-human", type=int, default=64,
                    help="episode_index of the first human episode (robot below this keep their 50 Hz timestamps)")
    a = ap.parse_args()

    fs = sorted(glob.glob(os.path.join(a.root, "data", "**", "*.parquet"), recursive=True))
    print(f"{len(fs)} parquet files -> single row group + combine_chunks; "
          f"human (idx >= {a.first_human}) timestamp = frame_index/{a.fps}")
    for k, f in enumerate(fs):
        t = pq.read_table(f).combine_chunks()                 # fix #1: one chunk per column
        if _ep(f) >= a.first_human:                            # fix #2: 50 Hz timestamps for human
            i = t.schema.get_field_index("timestamp")
            if i < 0:
                raise SystemExit(f"no 'timestamp' column in {f}")
            t = t.set_column(i, "timestamp", pa.array((np.arange(t.num_rows) / a.fps).astype(np.float32)))
        pq.write_table(t, f, row_group_size=max(t.num_rows, 1))  # single row group
        if (k + 1) % 200 == 0:
            print(f"  {k + 1}/{len(fs)}")
    print("done. Re-run `compute_norm_stats --config-name pi05_ego_mix_oic --max_frames 20000` to confirm it loads.")


if __name__ == "__main__":
    main()
