"""heldout_poster_frames.py — export one representative RGB frame per HELD-OUT R-ID episode
(the 12 in held_out_rid.txt that the data-mix sweep is evaluated on), for a poster.

Pulls the SAME image the policy sees (observations/images/aria_rgb_cam/color, no rotation —
the conversion feeds it raw), so the poster frames are faithful to the eval input. Saves one
PNG per episode (named with its object label) plus an optional captioned contact sheet.

Run on EULER (the 3dv venv has h5py/numpy/pillow):
  source ~/venvs/3dv/bin/activate
  python 3dvision-experiments/heldout_poster_frames.py                      # -> ~/heldout_frames/
  python 3dvision-experiments/heldout_poster_frames.py --frame first --grid # first frame + contact sheet
  python 3dvision-experiments/heldout_poster_frames.py --out ~/poster --cols 4 --sort-by-object

Then rsync to your laptop:
  rsync -avP euler:~/heldout_frames/ ./heldout_frames/
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DATA_DIR = "/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/object_in_bowl_processed_50hz"
IMG_KEY = "observations/images/aria_rgb_cam/color"
# These two ship in the repo; defaults assume the script runs from the repo root on Euler.
HELD_OUT = "held_out_rid.txt"
LABELS = "3dvision-experiments/object_labels.csv"


def load_labels(path: Path) -> dict:
    d = {}
    if path.exists():
        with open(path, newline="") as f:
            for row in csv.reader(f):
                if len(row) >= 2 and row[0] != "filename":
                    d[row[0]] = row[1]
    return d


def load_heldout(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def get_frame(h5_path: Path, choice: str) -> np.ndarray:
    import h5py  # Euler-only dependency
    with h5py.File(h5_path, "r") as f:
        ds = f[IMG_KEY]
        t = ds.shape[0]
        i = 0 if choice == "first" else (t - 1 if choice == "last" else t // 2)
        return np.asarray(ds[i])


def _font(size: int):
    """A legible TrueType font for the captions; fall back to PIL's bitmap default."""
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size)   # Pillow >= 10.1
    except Exception:
        return ImageFont.load_default()


def _tile(img: np.ndarray, label: str, sub: str, tile_w: int) -> Image.Image:
    """One captioned cell: the frame scaled to tile_w, with a label bar beneath it."""
    im = Image.fromarray(img).convert("RGB")
    w0, h0 = im.size
    th = int(round(tile_w * h0 / w0))
    im = im.resize((tile_w, th), Image.LANCZOS)
    bar = max(40, tile_w // 9)
    canvas = Image.new("RGB", (tile_w, th + bar), (20, 20, 24))
    canvas.paste(im, (0, 0))
    d = ImageDraw.Draw(canvas)
    f_main = _font(max(18, bar // 2)); f_sub = _font(max(12, bar // 4))
    d.text((10, th + bar // 2 - bar // 3), label, fill=(255, 255, 255), font=f_main)
    tw = d.textlength(sub, font=f_sub)
    d.text((tile_w - tw - 10, th + bar // 2 - bar // 6), sub, fill=(150, 150, 160), font=f_sub)
    return canvas


def build_grid(tiles: list[Image.Image], cols: int, pad: int = 8) -> Image.Image:
    rows = (len(tiles) + cols - 1) // cols
    cw = max(t.width for t in tiles); ch = max(t.height for t in tiles)
    grid = Image.new("RGB", (cols * cw + (cols + 1) * pad, rows * ch + (rows + 1) * pad), (8, 8, 10))
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        grid.paste(t, (pad + c * (cw + pad), pad + r * (ch + pad)))
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR, help="dir holding the held-out h5 files")
    ap.add_argument("--held-out", default=HELD_OUT, help="held_out_rid.txt (one filename per line)")
    ap.add_argument("--labels", default=LABELS, help="object_labels.csv sidecar (filename,object)")
    ap.add_argument("--out", default="~/heldout_frames", help="output dir for PNGs + contact sheet")
    ap.add_argument("--frame", choices=["first", "middle", "last"], default="middle")
    ap.add_argument("--grid", action="store_true", help="also build a captioned contact sheet")
    ap.add_argument("--cols", type=int, default=4, help="contact-sheet columns")
    ap.add_argument("--tile-w", type=int, default=480, help="contact-sheet tile width (px)")
    ap.add_argument("--sort-by-object", action="store_true", help="group tiles by object label")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out = Path(args.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    labels = load_labels(Path(args.labels).expanduser())
    files = load_heldout(Path(args.held_out).expanduser())
    print(f"{len(files)} held-out episodes | frame={args.frame} | out={out}")

    rows = []  # (object, stem, frame_array)
    for fn in files:
        h5 = data_dir / fn
        if not h5.exists():
            print(f"  MISSING {fn} (not in {data_dir})"); continue
        try:
            rows.append((labels.get(fn, "unlabeled"), Path(fn).stem, get_frame(h5, args.frame)))
        except Exception as e:
            print(f"  ERROR {fn}: {e}")

    if args.sort_by_object:
        rows.sort(key=lambda r: (r[0], r[1]))

    tiles = []
    for i, (obj, stem, frame) in enumerate(rows):
        png = out / f"{i:02d}_{obj}_{stem}.png"
        Image.fromarray(frame).save(png)
        print(f"  [{i:02d}] {obj:<13} {stem} -> {png.name}")
        if args.grid:
            tiles.append(_tile(frame, obj.replace("_", " "), stem, args.tile_w))

    if args.grid and tiles:
        grid = build_grid(tiles, args.cols)
        gp = out / f"heldout_contact_{args.frame}.png"
        grid.save(gp)
        print(f"\nContact sheet ({len(tiles)} tiles, {args.cols} cols) -> {gp}")
    print(f"\nDone. {len(rows)} frames in {out}")


if __name__ == "__main__":
    main()
