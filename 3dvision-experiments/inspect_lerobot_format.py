"""inspect_lerobot_format.py — dump the exact on-disk layout of the LeRobot datasets so we can
write a correct image-linking merge (egoverse/all robot + oic_human -> oic_mix) WITHOUT re-encoding.

The two things that decide the merge strategy:
  1. Are images stored as separate PNG files (images/ dir) or embedded as bytes in the parquet?
  2. Exact meta/ files + episode/chunk numbering.

Run in the 3dv venv:
  ~/venvs/3dv/bin/python ~/openpi/3dvision-experiments/inspect_lerobot_format.py
"""
import glob
import os

import pyarrow.parquet as pq

ROOTS = [
    ("oic_human (human source)", "/cluster/work/cvg/data/Egoverse/lerobot_egoverse/egoverse/oic_human"),
    ("all (robot source)", "/cluster/scratch/lichin/lerobot/egoverse/all"),
    ("oic_mix (smoke target)", "/cluster/scratch/lichin/lerobot/egoverse/oic_mix"),
]


def walk_first(d, depth=4):
    cur = d
    for _ in range(depth):
        try:
            ls = sorted(os.listdir(cur))
        except OSError:
            return
        print(f"      {os.path.relpath(cur, d) or '.'}/ -> {ls[:4]}{' ...' if len(ls) > 4 else ''}")
        nxt = [x for x in ls if os.path.isdir(os.path.join(cur, x))]
        if not nxt:
            return
        cur = os.path.join(cur, nxt[0])


for name, root in ROOTS:
    print("=" * 70, "\n", name, "\n ", root)
    if not os.path.isdir(root):
        print("  MISSING"); continue
    print("  top-level:", sorted(os.listdir(root)))
    print("  meta/:", sorted(os.listdir(os.path.join(root, "meta"))))

    imgd = os.path.join(root, "images")
    if os.path.isdir(imgd):
        print("  HAS images/ dir (separate files). tree:")
        walk_first(imgd)
        pngs = glob.glob(os.path.join(imgd, "**", "*.png"), recursive=True)
        print(f"  total PNGs: {len(pngs)}  e.g. {os.path.relpath(pngs[0], root) if pngs else '(none)'}")
    else:
        print("  NO images/ dir -> images likely embedded in parquet bytes")

    pqs = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    print(f"  data/: {len(pqs)} parquet files")
    if pqs:
        t = pq.read_table(pqs[0])
        print("   first parquet:", os.path.relpath(pqs[0], root), "| cols:", t.column_names)
        row = t.slice(0, 1).to_pylist()[0]
        img = row.get("image")
        if isinstance(img, dict):
            print("   image cell = dict, keys:", list(img.keys()),
                  "| bytes?", img.get("bytes") is not None, "| path:", img.get("path"))
        else:
            print("   image cell =", type(img).__name__, "| sample:", str(img)[:90])

    for mf in ("info.json", "episodes.jsonl", "tasks.jsonl", "episodes_stats.jsonl", "stats.json"):
        p = os.path.join(root, "meta", mf)
        if os.path.isfile(p):
            line1 = open(p).readline().strip()
            print(f"   meta/{mf} line1: {line1[:240]}")
