#!/usr/bin/env python3
"""Print the training step stored in an orbax/ocdbt train_state WITHOUT a full restore.

Needed for checkpoints whose step dir name was lost (e.g. the hid_prelim HF download,
which flattened <step>/ into the repo root). Reads only the tiny 'step' scalar array.

Usage (Euler login node, no GPU):
    cd ~/openpi && uv run python 3dvision-experiments/read_ckpt_step.py \
        /cluster/scratch/lichin/hf_ckpts/hid_prelim/train_state
"""

import sys

import tensorstore as ts


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    base = "file://" + sys.argv[1].rstrip("/")

    # Discover leaf paths inside the ocdbt store (keys look like '<leaf>/zarr.json' etc.).
    store = ts.KvStore.open({"driver": "ocdbt", "base": base}).result()
    keys = [k.decode() if isinstance(k, bytes) else k for k in store.list().result()]
    prefixes = sorted({k.split("/")[0] for k in keys})
    step_like = [p for p in prefixes if "step" in p.lower()]
    print(f"[info] {len(keys)} keys, {len(prefixes)} leaf prefixes; step-like: {step_like or 'NONE'}")

    for path in step_like or ["step"]:
        for drv in ("zarr3", "zarr"):
            try:
                t = ts.open({"driver": drv, "kvstore": {"driver": "ocdbt", "base": base, "path": path}},
                            read=True).result()
                print(f"step = {t.read().result()}  (leaf={path!r}, driver={drv})")
                return
            except Exception as e:  # noqa: BLE001 — probing drivers, report and move on
                print(f"[probe] leaf={path!r} driver={drv}: {type(e).__name__}")

    print("[fail] could not read a step leaf; all prefixes were:")
    for p in prefixes:
        print("  ", p)
    sys.exit(1)


if __name__ == "__main__":
    main()
