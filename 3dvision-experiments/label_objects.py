"""label_objects.py — quickly hand-label WHICH object the robot puts in the bowl, per
object_in_bowl (R-ID) episode.

For each episode it shows one frame and asks you to type the object name in the terminal.
Saves incrementally to a CSV (filename,object) after every entry, so you can quit and resume
anytime. Does NOT modify the raw h5 files — labels live in a separate sidecar CSV.

How you SEE the image (either works):
  - It writes the current frame to <out_dir>/_view.png. Keep that file open in VSCode
    Remote-SSH preview — it refreshes as you advance.
  - If `chafa`, `timg`, or `catimg` is installed, it ALSO renders the frame inline in the
    terminal (no extra setup; skipped if none are present).

Usage (cluster, 3dv venv — has h5py/numpy/pillow):
  source ~/venvs/3dv/bin/activate
  python label_objects.py
  python label_objects.py --frame first --out ~/object_labels.csv

At the prompt: <text>=set label, [enter]=skip forward, b=back, q=quit & save.
"""
import argparse
import csv
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

DATA_DIR = "/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/object_in_bowl_processed_50hz"
IMG_KEY = "observations/images/aria_rgb_cam/color"


def load_labels(path: Path) -> dict:
    d = {}
    if path.exists():
        with open(path, newline="") as f:
            for row in csv.reader(f):
                if len(row) >= 2 and row[0] != "filename":
                    d[row[0]] = row[1]
    return d


def save_labels(path: Path, labels: dict):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "object"])
        for k in sorted(labels):
            w.writerow([k, labels[k]])


def get_frame(h5_path: Path, choice: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        ds = f[IMG_KEY]
        t = ds.shape[0]
        i = 0 if choice == "first" else (t - 1 if choice == "last" else t // 2)
        return np.asarray(ds[i])


def show_in_terminal(png: Path) -> bool:
    """Render inline if a terminal image viewer exists; otherwise no-op (rely on the PNG)."""
    for tool in ("chafa", "timg", "catimg"):
        if shutil.which(tool):
            try:
                subprocess.run([tool, str(png)], check=False)
                return True
            except Exception:
                pass
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out", default=str(Path.home() / "object_labels.csv"), help="sidecar CSV of labels")
    ap.add_argument("--frame", choices=["first", "middle", "last"], default="middle")
    ap.add_argument("--relabel", action="store_true", help="start from the top and revisit labeled episodes")
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    view = out.parent / "_view.png"
    eps = sorted(Path(args.data_dir).glob("*.h5"))
    labels = load_labels(out)
    if not eps:
        print(f"No .h5 in {args.data_dir}")
        return
    print(f"{len(eps)} episodes | {len(labels)} already labeled | saving -> {out}")
    print(f"Keep this open in VSCode preview: {view}")
    print("Prompt: <text>=label, [enter]=skip, b=back, q=quit & save\n")

    i = 0
    if not args.relabel:                        # resume at first unlabeled episode
        while i < len(eps) and eps[i].name in labels:
            i += 1
    while 0 <= i < len(eps):
        ep = eps[i]
        name = ep.name
        try:
            Image.fromarray(get_frame(ep, args.frame)).save(view)
        except Exception as e:
            print(f"[{i + 1}/{len(eps)}] {name}: ERROR {e} — skipping")
            i += 1
            continue
        show_in_terminal(view)
        cur = labels.get(name, "")
        ans = input(f"[{i + 1}/{len(eps)}] {name}{f'  [{cur}]' if cur else ''} > ").strip()
        if ans == "q":
            break
        if ans == "b":
            i = max(0, i - 1)
            continue
        if ans == "":
            i += 1
            continue
        labels[name] = ans
        save_labels(out, labels)                # crash-safe: persist after every entry
        i += 1

    save_labels(out, labels)
    print(f"\nSaved {len(labels)} labels -> {out}")


if __name__ == "__main__":
    main()
