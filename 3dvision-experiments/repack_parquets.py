"""repack_parquets.py — rewrite every parquet in a LeRobot dataset with ONE row group per file.

Fixes the HF datasets error
    pyarrow.lib.ArrowNotImplementedError: Nested data conversions not implemented for chunked array outputs
which fires inside load_dataset when it casts the nested `image` column from a parquet that has
MULTIPLE row groups (-> a multi-chunk array on read). One row group => one chunk => the cast works.

Only the on-disk row-group LAYOUT changes — values, columns and meta/ are untouched (verify still
PASSes). Reads + rewrites IN PLACE from wherever --root points, so it runs entirely on YOUR scratch
(no cross-user source reads, no re-transform). Much faster than re-merging.

Run as a COMPUTE-NODE job (it's ~tens of GB of I/O — don't do it on the login node):
  sbatch --time=03:00:00 --mem-per-cpu=8G --cpus-per-task=4 -o ~/repack.out --wrap \
    "cd ~/openpi && UV_FROZEN=1 UV_OFFLINE=1 uv run python 3dvision-experiments/repack_parquets.py \
       --root /cluster/scratch/$USER/lerobot/egoverse/oic_mix"
"""
import argparse
import glob
import os

import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="LeRobot dataset dir (rewrites data/**/*.parquet in place)")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.root, "data", "**", "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"no parquet files under {a.root}/data")
    print(f"[repack] {len(files)} parquet files under {a.root}")
    for i, f in enumerate(files):
        t = pq.read_table(f).combine_chunks()           # single-chunk in memory
        tmp = f + ".tmp"
        pq.write_table(t, tmp, row_group_size=max(1, t.num_rows))   # one row group per file
        os.replace(tmp, f)                              # atomic swap
        if (i + 1) % 200 == 0 or i + 1 == len(files):
            print(f"  {i + 1}/{len(files)}")
    print("[repack] done — reload with load_dataset / compute_norm_stats")


if __name__ == "__main__":
    main()
