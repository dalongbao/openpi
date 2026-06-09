"""infer_oic_convention.py — infer the euler order / frame of the human `oic` 6D action.

The human object_in_container action is 6D [x,y,z, a,b,c] where (a,b,c) are euler angles
in an UNKNOWN order/frame. The 7D-quat -> 6D-euler conversion happens upstream (EgoVerse
`aria_process/aria_utils.py` itself keeps 7D quat *xyzw* in a gravity-aligned WORLD frame;
the euler reduction is a later, undocumented step). This script narrows that convention.

METHOD B (no external reference; always runs) — diagnostics on one episode's 6D trajectory:
  * units: radians vs degrees (from angle magnitude),
  * position scale + per-axis profile -> a PNG to eyeball the gravity/up axis
    (a pick&place shows a reach-DOWN-then-LIFT signature on the vertical axis),
  * per-euler-order rotation-path length + jerk -> a shortlist of plausible orders
    (re-interpreting the angles under the WRONG order scrambles the rotation path).

METHOD A (decisive; needs a paired 7D quaternion reference via --quat-ref) — brute-force
which (order, units) reproduces the stored euler from the quaternion. Near-zero residual
=> that is the convention. If NO order matches, the 6D lives in a different frame than the
quat (a rotation was applied before euler extraction) -> reported, and is itself the answer
to "is it still in world frame".

Run on Euler in the 3dv venv (needs scipy + matplotlib — install once on the login node):
  ~/venvs/3dv/bin/pip install scipy matplotlib
  ~/venvs/3dv/bin/python infer_oic_convention.py \
      --parquet /cluster/work/cvg/data/Egoverse/lerobot_egoverse/egoverse/oic_human/data/chunk-000/episode_000000.parquet \
      --out-png ~/oic_convention.png
  # later, once you have the 7D quat for the SAME frames:
  #   ... --quat-ref ~/oic_quat_ref.npy
"""
import argparse

import numpy as np
import pandas as pd

# scipy euler convention: lowercase = extrinsic, uppercase = intrinsic.
ORDERS = ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx",
          "XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"]


def load_pose(parquet: str, col: str):
    df = pd.read_parquet(parquet)
    if "frame_index" in df.columns:
        df = df.sort_values("frame_index")
    if col == "auto":
        for c in ("state", "observations.state.ee_pose", "actions", "actions_cartesian"):
            if c in df.columns:
                col = c
                break
        else:
            raise SystemExit(f"no pose column found; columns = {list(df.columns)}")
    arr = np.array([np.asarray(v, dtype=np.float64).reshape(-1) for v in df[col].to_list()])
    # prestacked chunk cells (e.g. actions_cartesian flattened (T,6)) -> use first sub-pose per row
    if arr.shape[1] > 6 and arr.shape[1] % 6 == 0:
        arr = arr.reshape(arr.shape[0], -1, 6)[:, 0, :]
    if arr.shape[1] != 6:
        raise SystemExit(f"column '{col}' is not 6D (got shape {arr.shape})")
    print(f"[load] {parquet}\n[load] column='{col}'  frames={arr.shape[0]}")
    return arr, col


def method_b(pose: np.ndarray, out_png: str):
    from scipy.spatial.transform import Rotation as R

    pos, ang = pose[:, :3], pose[:, 3:6]

    amax = float(np.abs(ang).max())
    units = "degrees" if amax > 3.2 else "radians"
    ang_rad = np.deg2rad(ang) if units == "degrees" else ang
    print(f"\n[units] max|angle| = {amax:.3f}  ->  {units}")

    print("[pos] per-axis  min / max / range / std  (assume meters):")
    for i, axn in enumerate("XYZ"):
        c = pos[:, i]
        print(f"   axis{i}({axn}): {c.min():+.3f} {c.max():+.3f}  range={np.ptp(c):.3f}  std={c.std():.3f}")
    guesses = []
    for i in range(3):
        c = pos[:, i]
        k = int(np.argmin(c))
        interior = 0.1 < k / max(len(c), 1) < 0.9   # lowest point is a turning point, not an endpoint
        guesses.append((np.ptp(c) * (1.0 if interior else 0.3), i, k, interior))
    guesses.sort(reverse=True)
    g = guesses[0]
    print(f"[pos] likely gravity/up axis = axis{g[1]}  (lowest at frame {g[2]}, interior={g[3]}) "
          f"— CONFIRM in the PNG (reach-down-then-lift)")

    print("\n[rot] re-interpret the angles under each euler order; the TRUE order tends to")
    print("      give the shortest, smoothest rotation path. Ranked (shorter path = better):")
    rows = []
    for o in ORDERS:
        Rs = R.from_euler(o, ang_rad)
        step = (Rs[:-1].inv() * Rs[1:]).magnitude()   # geodesic angular step per frame (rad)
        rows.append((float(step.sum()), float(np.diff(step).std()), o))
    for path, jerk, o in sorted(rows):
        print(f"   {o:4s}  path={path:9.3f} rad   jerk={jerk:.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = np.arange(len(pose))
        fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        for i, axn in enumerate("XYZ"):
            ax[0].plot(t, pos[:, i], label=f"pos axis{i} ({axn})")
        ax[0].set_ylabel("position")
        ax[0].set_title("position — the reach-DOWN-then-LIFT axis is gravity/up")
        ax[0].legend()
        for i in range(3):
            ax[1].plot(t, ang[:, i], label=f"euler angle{i}")
        ax[1].set_ylabel(f"euler ({units})")
        ax[1].set_xlabel("frame")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(out_png, dpi=110)
        print(f"\n[png] wrote {out_png}  — open in VSCode remote: eyeball the gravity axis + angle continuity")
    except Exception as e:  # noqa: BLE001
        print(f"[png] skipped ({e})")


def method_a(pose: np.ndarray, quat_ref: str):
    from scipy.spatial.transform import Rotation as R

    q = np.load(quat_ref, allow_pickle=True)
    if hasattr(q, "files"):          # .npz
        q = q[q.files[0]]
    q = np.asarray(q, dtype=np.float64)
    if q.shape[1] != 4:
        raise SystemExit(f"--quat-ref must be (N,4) xyzw, got {q.shape}")
    n = min(q.shape[0], pose.shape[0])
    if q.shape[0] != pose.shape[0]:
        print(f"[A] truncating to {n} paired frames")
    q, ang = q[:n], pose[:n, 3:6]

    print("\n[A] round-trip: which (order, units) reproduces the stored euler from the quat?")
    best = None
    for deg in (False, True):
        for o in ORDERS:
            pred = R.from_quat(q).as_euler(o, degrees=deg)
            d = pred - (ang if deg or True else ang)
            wrap = 360.0 if deg else 2 * np.pi
            res = float(np.abs((d + wrap / 2) % wrap - wrap / 2).mean())
            tag = f"{o} {'deg' if deg else 'rad'}"
            if best is None or res < best[0]:
                best = (res, tag)
            if res < (5.0 if deg else 0.09):
                print(f"   *** MATCH  {tag:8s}  residual={res:.4f}")
    ok = best[0] < (5.0 if "deg" in best[1] else 0.1)
    print(f"[A] best = {best[1]}  residual={best[0]:.4f}  -> "
          + ("THAT is the convention." if ok else
             "no clean match: the 6D is in a DIFFERENT frame than the quat (a rotation precedes the euler step)."))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True, help="one LeRobot episode parquet")
    p.add_argument("--col", default="auto", help="pose column (default auto: state/ee_pose/actions)")
    p.add_argument("--out-png", default="oic_convention.png")
    p.add_argument("--quat-ref", default=None, help="(N,4) xyzw .npy/.npz paired with the SAME frames -> Method A")
    a = p.parse_args()

    pose, _ = load_pose(a.parquet, a.col)
    method_b(pose, a.out_png)
    if a.quat_ref:
        method_a(pose, a.quat_ref)
    else:
        print("\n[A] skipped (no --quat-ref). For the DECISIVE test, get the 7D quat for one "
              "recording from jiaqchen / the raw EgoVerse zarr, save as (N,4) xyzw .npy, and rerun with --quat-ref.")


if __name__ == "__main__":
    main()
