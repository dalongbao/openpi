"""merge_to_oic_mix.py — FAST builder for `egoverse/oic_mix` by REUSING the two already-converted
LeRobot datasets (robot `egoverse/all` + human `oic_human`), linking/copying images instead of
re-decoding+re-encoding the ~563k frames. Minutes, not the 5-8h of build_oic_mix.py.

It is the fast path for REBUILDS (e.g. a new held-out split) and a backup if the slow build
failed. See 3dvision-experiments/MERGE_DESIGN.md for the full spec. Output must be numerically
identical (state/actions/action_mask) to build_oic_mix.py — only the image bytes ride along
untouched instead of being re-encoded.

Target (LeRobot v2.1, fps=50, robot_type=franka), 24D unified action space:
  robot eps 0..Nr-1 : state/actions native 24D base (egoverse/all = identity), action_mask=ones(24)
  human eps Nr..     : state/actions 6D->24D base (egoverse_unify),            action_mask=[1]*7+[0]*17

RUN ON EULER in the openpi uv env (has pyarrow/pandas/numpy/PIL):
  # 0) is it even needed?  verify the existing build first:
  uv run python 3dvision-experiments/verify_oic_mix.py --root /cluster/scratch/lichin/lerobot/egoverse/oic_mix
  # 1) MANDATORY: confirm image storage (Case A embedded-bytes vs Case B png files):
  uv run python 3dvision-experiments/inspect_lerobot_format.py
  # 2) SMOKE first (2+2 eps), then verify, then cross-check vs build_oic_mix on the same eps:
  uv run python 3dvision-experiments/merge_to_oic_mix.py --max-robot 2 --max-human 2 --target /cluster/scratch/$USER/lerobot/egoverse/oic_mix_smoke
  uv run python 3dvision-experiments/verify_oic_mix.py --root /cluster/scratch/$USER/lerobot/egoverse/oic_mix_smoke
  # 3) FULL only after smoke matches:
  uv run python 3dvision-experiments/merge_to_oic_mix.py --target /cluster/scratch/$USER/lerobot/egoverse/oic_mix
"""
import argparse
import json
import os
import pathlib
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openpi.policies import egoverse_unify as U

ROBOT_DEFAULT = "/cluster/scratch/lichin/lerobot/egoverse/all"
HUMAN_DEFAULT = "/cluster/work/cvg/data/Egoverse/lerobot_egoverse/egoverse/oic_human"
CHUNKS_SIZE = 1000
TASK = "put the object in the bowl"
# Columns we always rebuild (so robot+human parquets share one schema regardless of source version).
_NUM = pa.list_(pa.float32())
_I64 = pa.int64()


# ----------------------------- helpers -----------------------------
def _read_json(p):
    with open(p) as f:
        return json.load(f)


def _read_jsonl(p):
    with open(p) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _list_parquets(root):
    import glob
    return sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))


def _detect_case(root, pqs):
    """Case 'B' if a populated images/ dir exists (separate PNGs), else 'A' (bytes in parquet)."""
    imgd = os.path.join(root, "images")
    if os.path.isdir(imgd):
        import glob
        if glob.glob(os.path.join(imgd, "**", "*.png"), recursive=True):
            return "B"
    # confirm A: the parquet image cell carries bytes (or a struct with bytes)
    t = pq.read_table(pqs[0], columns=["image"]) if pqs else None
    if t is not None and t.num_rows:
        cell = t.slice(0, 1).to_pylist()[0]["image"]
        if isinstance(cell, dict) and cell.get("path") and cell.get("bytes") is None:
            return "B"  # struct points at external file
    return "A"


def _num_col(arr2d):
    """(n,d) float -> pyarrow list<float32> column."""
    return pa.array([row for row in np.asarray(arr2d, np.float32)], type=_NUM)


def _stk(col_pylist):
    return np.stack([np.asarray(v, np.float64).reshape(-1) for v in col_pylist])


def _set(t, name, array):
    """Replace column `name` if present, else append it."""
    i = t.schema.get_field_index(name)
    return t.set_column(i, name, array) if i >= 0 else t.append_column(name, array)


def _episode_stats(idx, length, state24, act24, mask24, src_img_stats):
    """v2.1 episodes_stats entry. Numeric stats recomputed from the 24D arrays (cheap); image
    stats COPIED from the source (bytes are unchanged, so the source's image stats stay valid)."""
    def s(a):
        a = np.asarray(a, np.float64)
        return {"min": a.min(0).tolist(), "max": a.max(0).tolist(),
                "mean": a.mean(0).tolist(), "std": a.std(0).tolist(), "count": [int(length)]}
    stats = {"state": s(state24), "actions": s(act24), "action_mask": s(mask24)}
    if src_img_stats is not None:
        stats["image"] = src_img_stats
    return {"episode_index": idx, "stats": stats}


# ----------------------------- core -----------------------------
def merge(robot_root, human_root, target, max_robot, max_human, force):
    target = pathlib.Path(target)
    if target.exists():
        if not force:
            raise SystemExit(f"{target} exists — pass --force to overwrite")
        shutil.rmtree(target)
    rp, hp = _list_parquets(robot_root), _list_parquets(human_root)
    if not rp:
        raise SystemExit(f"no robot parquets under {robot_root} — is egoverse/all built?")
    if not hp:
        raise SystemExit(f"no human parquets under {human_root}")
    if max_robot is not None:
        rp = rp[:max_robot]
    if max_human is not None:
        hp = hp[:max_human]

    case = _detect_case(robot_root, rp)
    print(f"[merge] image storage = Case {case}  ({'embedded bytes' if case == 'A' else 'separate PNG files'})")
    if case == "B":
        raise SystemExit(
            "Case B (separate PNG files) detected. This build targets Case A (embedded bytes, what "
            "LeRobot's dtype='image' produces). The sources here use dtype='image' so Case A is "
            "expected — if you really have a png images/ tree, tell me and I'll add the hard-link branch.")

    # per-source image stats keyed by source episode_index (for copying, no decode)
    def img_stats_map(root):
        p = os.path.join(root, "meta", "episodes_stats.jsonl")
        if not os.path.isfile(p):
            return {}
        return {e["episode_index"]: e.get("stats", {}).get("image") for e in _read_jsonl(p)}
    rob_imgstats, hum_imgstats = img_stats_map(robot_root), img_stats_map(human_root)

    (target / "meta").mkdir(parents=True, exist_ok=True)
    out_ei = 0          # running target episode index
    out_index = 0       # running global frame index
    episodes_meta, episodes_stats = [], []

    def write_episode(t, state24, act24, mask24, length, src_imgstats):
        nonlocal out_ei, out_index
        t = _set(t, "state", _num_col(state24))
        t = _set(t, "actions", _num_col(act24))
        t = _set(t, "action_mask", _num_col(mask24))
        t = _set(t, "episode_index", pa.array(np.full(length, out_ei, np.int64), _I64))
        t = _set(t, "frame_index", pa.array(np.arange(length, dtype=np.int64), _I64))
        t = _set(t, "index", pa.array(np.arange(out_index, out_index + length, dtype=np.int64), _I64))
        t = _set(t, "task_index", pa.array(np.zeros(length, np.int64), _I64))
        chunk = out_ei // CHUNKS_SIZE
        cdir = target / "data" / f"chunk-{chunk:03d}"
        cdir.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, cdir / f"episode_{out_ei:06d}.parquet")
        episodes_meta.append({"episode_index": out_ei, "tasks": [TASK], "length": int(length)})
        episodes_stats.append(_episode_stats(out_ei, length, state24, act24, mask24, src_imgstats))
        out_ei += 1
        out_index += int(length)

    # ---- ROBOT: identity 24D, full mask ----
    for k, f in enumerate(rp):
        t = pq.read_table(f)
        st = _stk(t.column("state").to_pylist())
        ac = _stk(t.column("actions").to_pylist())
        if st.shape[1] != U.ACTION_DIM or ac.shape[1] != U.ACTION_DIM:
            raise SystemExit(f"robot {os.path.basename(f)} state/actions not 24D (got {st.shape},{ac.shape}) — "
                             "egoverse/all must be the 24D-base build")
        mask = np.ones((len(st), U.ACTION_DIM), np.float32)
        src_ei = int(t.column("episode_index")[0].as_py()) if "episode_index" in t.column_names else k
        write_episode(t, st, ac, mask, len(st), rob_imgstats.get(src_ei))
    n_rob = out_ei
    print(f"[merge] robot: {n_rob} eps -> target 0..{n_rob - 1}")

    # ---- HUMAN: 6D head -> 24D base, arm-only mask ----
    hmask_row = np.concatenate([np.ones(U.ARM_DIM, np.float32), np.zeros(U.HAND_DIM, np.float32)])
    for k, f in enumerate(hp):
        t = pq.read_table(f)
        if "frame_index" in t.column_names:                       # sort to original frame order
            order = np.argsort(np.asarray(t.column("frame_index").to_pylist()))
            t = t.take(pa.array(order))
        s6 = _stk(t.column("state").to_pylist())[:, :6]
        a6 = _stk(t.column("actions").to_pylist())[:, :6]
        s24 = U.to_unified(U.human6d_to_base_arm7(s6), None)[0]
        a24 = U.to_unified(U.human6d_to_base_arm7(a6), None)[0]
        mask = np.tile(hmask_row, (len(s24), 1))
        src_ei = int(t.column("episode_index")[0].as_py()) if "episode_index" in t.column_names else k
        write_episode(t, s24, a24, mask, len(s24), hum_imgstats.get(src_ei))
    n_hum = out_ei - n_rob
    print(f"[merge] human: {n_hum} eps -> target {n_rob}..{out_ei - 1}")

    # ---- meta ----
    rob_info = _read_json(os.path.join(robot_root, "meta", "info.json"))
    feats = dict(rob_info["features"])                            # robot already 24D -> base schema
    feats["action_mask"] = {"dtype": "float32", "shape": [U.ACTION_DIM], "names": ["action_mask"]}
    info = dict(rob_info)
    info.update({
        "total_episodes": out_ei,
        "total_frames": out_index,
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": (out_ei + CHUNKS_SIZE - 1) // CHUNKS_SIZE,
        "chunks_size": CHUNKS_SIZE,
        "fps": 50,
        "robot_type": "franka",
        "splits": {"train": f"0:{out_ei}"},
        "features": feats,
    })
    with open(target / "meta" / "info.json", "w") as fo:
        json.dump(info, fo, indent=4)
    with open(target / "meta" / "episodes.jsonl", "w") as fo:
        for e in episodes_meta:
            fo.write(json.dumps(e) + "\n")
    with open(target / "meta" / "tasks.jsonl", "w") as fo:
        fo.write(json.dumps({"task_index": 0, "task": TASK}) + "\n")
    if any("image" in e["stats"] for e in episodes_stats) or True:
        with open(target / "meta" / "episodes_stats.jsonl", "w") as fo:
            for e in episodes_stats:
                fo.write(json.dumps(e) + "\n")

    print(f"\n[merge] DONE  total_episodes={out_ei}  total_frames={out_index}  -> {target}")
    print(f"[merge] expected by config.py _MIX_HUMAN(range 64..2601): robot=64, human=2537, total=2601")
    if max_robot is None and max_human is None and out_ei != 2601:
        print(f"[merge] NOTE total_episodes={out_ei} != 2601 — update config.py _MIX_HUMAN to range(64,{out_ei}).")
    print("[merge] next: verify_oic_mix.py --root <target>, then cross-check vs build_oic_mix on the same eps.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot-root", default=ROBOT_DEFAULT, help="converted robot dataset (egoverse/all, 24D base)")
    ap.add_argument("--human-root", default=HUMAN_DEFAULT, help="converted human dataset (oic_human, 6D)")
    ap.add_argument("--target", default="/cluster/scratch/lichin/lerobot/egoverse/oic_mix", help="output dataset dir")
    ap.add_argument("--max-robot", type=int, default=None, help="smoke: limit robot episodes")
    ap.add_argument("--max-human", type=int, default=None, help="smoke: limit human episodes")
    ap.add_argument("--force", action="store_true", help="overwrite an existing target")
    a = ap.parse_args()
    merge(a.robot_root, a.human_root, a.target, a.max_robot, a.max_human, a.force)


if __name__ == "__main__":
    main()
