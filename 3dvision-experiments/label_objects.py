"""label_objects.py — quickly hand-label WHICH object the robot puts in the bowl, per
object_in_bowl (R-ID) episode.

For each episode it shows one frame and asks you to type the object name in the terminal.
Saves incrementally to a CSV (filename,object) after every entry, so you can quit and resume
anytime. Does NOT modify the raw h5 files — labels live in a separate sidecar CSV.

How you SEE the image:
  - DEFAULT: it draws the frame straight into the terminal with truecolor half-block chars
    (no install, works in the VSCode integrated terminal). Tune with --width (chars), 0=off.
  - It also writes <out_dir>/_view.png as a backup you can open in VSCode preview. With
    --width 0 it instead tries an external viewer (chafa/timg/catimg) if present.

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


def render_ansi(img: np.ndarray, width: int = 72):
    """Print the image straight into the terminal using truecolor half-block chars (no deps,
    no external viewer). Each cell = 2 vertically-stacked pixels: fg=top, bg=bottom via '▀'.
    Needs a 24-bit-color terminal (the VSCode integrated terminal qualifies)."""
    im = Image.fromarray(img).convert("RGB")
    w0, h0 = im.size
    rows = max(2, int(round(width * h0 / w0 * 0.5)) * 1)
    im = im.resize((width, rows * 2))
    a = np.asarray(im)
    lines = []
    for r in range(rows):
        top, bot = a[2 * r], a[2 * r + 1]
        cells = [f"\x1b[38;2;{top[c][0]};{top[c][1]};{top[c][2]}m"
                 f"\x1b[48;2;{bot[c][0]};{bot[c][1]};{bot[c][2]}m▀" for c in range(width)]
        lines.append("".join(cells) + "\x1b[0m")
    print("\n".join(lines))


def show_in_terminal(png: Path) -> bool:
    """Prefer an external viewer if present; otherwise no-op (render_ansi already drew it)."""
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
    ap.add_argument("--width", type=int, default=72, help="terminal render width in chars (0 = off, use PNG only)")
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
            frame = get_frame(ep, args.frame)
            Image.fromarray(frame).save(view)
        except Exception as e:
            print(f"[{i + 1}/{len(eps)}] {name}: ERROR {e} — skipping")
            i += 1
            continue
        if args.width > 0:
            render_ansi(frame, args.width)      # draw straight into the terminal
        else:
            show_in_terminal(view)              # fall back to external viewer / PNG preview
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
