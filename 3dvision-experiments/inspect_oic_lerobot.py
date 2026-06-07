#!/usr/bin/env python3
"""
inspect_oic_lerobot.py — print the on-disk layout of the converted oic_human LeRobot
dataset so we can write a demo extractor for the Euler-order GT replay. Read-only.

Run on the Euler login node (3dv venv has pyarrow/pandas):
  source ~/venvs/3dv/bin/activate && python ~/scripts/inspect_oic_lerobot.py
"""
import glob
import json
import os

ROOT = "/cluster/work/cvg/data/Egoverse/lerobot_egoverse/egoverse/oic_human"


def main():
    print(f"[root] {ROOT}")
    if not os.path.isdir(ROOT):
        print("  MISSING — check the path"); return

    # top-level layout
    for sub in ("meta", "data", "videos"):
        p = os.path.join(ROOT, sub)
        print(f"  {sub:7s}: {'exists' if os.path.isdir(p) else 'absent'}")

    # meta/info.json — fps, features, shapes
    info = os.path.join(ROOT, "meta", "info.json")
    if os.path.isfile(info):
        d = json.load(open(info))
        print("\n[meta/info.json]")
        for k in ("codebase_version", "robot_type", "fps", "total_episodes",
                  "total_frames", "total_videos", "chunks_size"):
            if k in d:
                print(f"  {k}: {d[k]}")
        feats = d.get("features", {})
        print("  features:")
        for name, spec in feats.items():
            print(f"    {name}: dtype={spec.get('dtype')} shape={spec.get('shape')}")

    # one parquet: columns, dtypes, a sample row's shapes
    pqs = sorted(glob.glob(os.path.join(ROOT, "data", "**", "*.parquet"), recursive=True))
    print(f"\n[data] {len(pqs)} parquet files; inspecting first: {pqs[0] if pqs else '(none)'}")
    if pqs:
        import pyarrow.parquet as pq
        t = pq.read_table(pqs[0])
        print(f"  columns: {t.column_names}")
        print(f"  num_rows: {t.num_rows}")
        row = t.slice(0, 1).to_pylist()[0]
        for k, v in row.items():
            if isinstance(v, list):
                import numpy as np
                a = np.asarray(v)
                print(f"    {k}: list len={len(v)} -> shape {a.shape} dtype {a.dtype}  head={np.round(a.ravel()[:6],3).tolist()}")
            elif isinstance(v, (bytes, dict)):
                print(f"    {k}: {type(v).__name__} (likely image bytes)")
            else:
                print(f"    {k}: {type(v).__name__} = {v}")


if __name__ == "__main__":
    main()
