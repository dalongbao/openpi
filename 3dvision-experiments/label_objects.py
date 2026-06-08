"""label_objects.py — quickly hand-label WHICH object the robot puts in the bowl, per
object_in_bowl (R-ID) episode.

For each episode it shows one frame and asks you to type the object name in the terminal.
Saves incrementally to a CSV (filename,object) after every entry, so you can quit and resume
anytime. Does NOT modify the raw h5 files — labels live in a separate sidecar CSV.

RECOMMENDED (crisp, full-res): label on your Mac.
  1. ON EULER, dump one PNG per episode:   python label_objects.py --dump ~/oib_frames
  2. rsync to your Mac:   rsync -av euler:~/oib_frames/ ~/oib_frames/
  3. ON MAC, label from the folder:   python label_objects.py --images-dir ~/oib_frames
     Each frame opens in Preview at full resolution; type the label in the terminal.

On Euler directly (lower-res): the frame is drawn into the terminal with truecolor
half-block chars (--width chars, 0=off) plus a backup <out_dir>/_view.png for VSCode preview.

Usage (cluster, 3dv venv — has h5py/numpy/pillow):
  source ~/venvs/3dv/bin/activate
  python label_objects.py
  python label_objects.py --frame first --out ~/object_labels.csv

At the prompt: <text>=set label, [enter]=skip forward, b=back, q=quit & save.
"""
import argparse
import csv
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
# h5py is imported lazily inside get_frame/dump (Euler only) so --images-dir works on a Mac without it.

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
    import h5py  # Euler-only dependency
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
    rows = max(2, int(round(width * h0 / w0 * 0.5)))
    im = im.resize((width, rows * 2), Image.LANCZOS)    # sharp downscale
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


def dump_frames(data_dir: str, out_dir: str, choice: str):
    """ON EULER: write one full-res PNG per episode (named <episode>.png) to out_dir, then exit.
    rsync that folder to your Mac and label there with --images-dir."""
    od = Path(out_dir).expanduser()
    od.mkdir(parents=True, exist_ok=True)
    eps = sorted(Path(data_dir).glob("*.h5"))
    for i, ep in enumerate(eps):
        try:
            Image.fromarray(get_frame(ep, choice)).save(od / f"{ep.stem}.png")
        except Exception as e:
            print(f"  ERROR {ep.name}: {e}")
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(eps)}")
    print(f"Wrote {len(eps)} PNGs -> {od}\nrsync it to your Mac, then: "
          f"python label_objects.py --images-dir <local_dir>")


def open_native(path: Path) -> bool:
    """Open the image in the OS default viewer (macOS Preview / Linux xdg-open)."""
    try:
        if platform.system() == "Darwin":
            subprocess.run(["open", "-g", str(path)], check=False)   # -g: keep terminal focused
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR, help="Euler h5 dir (h5 source)")
    ap.add_argument("--dump", default=None, help="ON EULER: dump one PNG per episode to this dir, then exit")
    ap.add_argument("--images-dir", default=None, help="ON MAC: label from a folder of <episode>.png (no h5 needed)")
    ap.add_argument("--out", default="3dvision-experiments/object_labels.csv", help="sidecar CSV of labels")
    ap.add_argument("--frame", choices=["first", "middle", "last"], default="middle")
    ap.add_argument("--width", type=int, default=100, help="terminal render width in chars (0 = off)")
    ap.add_argument("--relabel", action="store_true", help="start from the top and revisit labeled episodes")
    args = ap.parse_args()

    if args.dump:                               # Euler: just export frames and stop
        dump_frames(args.data_dir, args.dump, args.frame)
        return

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    view = out.parent / "_view.png"

    # Source: a local PNG folder (Mac) OR the h5 episodes (Euler). Each item -> (key, png_or_h5).
    mac_mode = bool(args.images_dir)
    if mac_mode:
        pngs = sorted(Path(args.images_dir).expanduser().glob("*.png"))
        items = [(p.stem + ".h5", p) for p in pngs]   # key = episode basename, for held-out compatibility
    else:
        items = [(ep.name, ep) for ep in sorted(Path(args.data_dir).glob("*.h5"))]
    if not items:
        print("No episodes found (check --images-dir / --data-dir).")
        return

    labels = load_labels(out)
    print(f"{len(items)} episodes | {len(labels)} already labeled | saving -> {out}")
    print(f"FULL-RES image: {'opens in Preview' if mac_mode else view}")
    print("Prompt: <text>=label, [enter]=skip, b=back, q=quit & save\n")

    i = 0
    if not args.relabel:                        # resume at first unlabeled episode
        while i < len(items) and items[i][0] in labels:
            i += 1
    while 0 <= i < len(items):
        key, ref = items[i]
        try:
            if mac_mode:
                open_native(ref)                # Preview at full resolution
                frame = np.asarray(Image.open(ref).convert("RGB")) if args.width > 0 else None
            else:
                frame = get_frame(ref, args.frame)
                Image.fromarray(frame).save(view)
        except Exception as e:
            print(f"[{i + 1}/{len(items)}] {key}: ERROR {e} — skipping")
            i += 1
            continue
        if args.width > 0 and frame is not None:
            render_ansi(frame, args.width)      # small inline reference (Preview is the real view on Mac)
        elif not mac_mode:
            show_in_terminal(view)
        cur = labels.get(key, "")
        ans = input(f"[{i + 1}/{len(items)}] {key}{f'  [{cur}]' if cur else ''} > ").strip()
        if ans == "q":
            break
        if ans == "b":
            i = max(0, i - 1)
            continue
        if ans == "":
            i += 1
            continue
        labels[key] = ans
        save_labels(out, labels)                # crash-safe: persist after every entry
        i += 1

    save_labels(out, labels)
    print(f"\nSaved {len(labels)} labels -> {out}")


if __name__ == "__main__":
    main()
