"""summarize_inference.py — collect the real-frame inference logs from run_inference.py
(24-dim object_in_bowl) and run_inference_oic.py (6-dim oic) into ONE comparison table.

Both scripts print the same motion block (magnitude ratio + direction cosine) and a
Summary block with the model's MSE plus zero-action / const-state baselines. This parses
any number of their slurm/stdout logs and emits a markdown table so base-vs-finetuned and
the two tasks sit side by side.

IMPORTANT caveat printed with the table: the direction cosine for object_in_bowl is in
7-dim JOINT space; for oic it is in 3-dim CARTESIAN position. Compare base-vs-finetuned
WITHIN a task directly; compare ACROSS tasks only as "is the model reaching correctly at
all", not as identical axes.

Usage (on Euler, after the jobs finish — pass the logs in any order):
  uv run python 3dvision-experiments/summarize_inference.py \
      oic_finetuned=/cluster/scratch/$USER/openpi/slurm-2551944.out \
      oic_base=/cluster/scratch/$USER/openpi/slurm-XXXX.out \
      oib_finetuned=/cluster/scratch/$USER/openpi/slurm-YYYY.out \
      oib_base=/cluster/scratch/$USER/openpi/slurm-ZZZZ.out

Each arg is LABEL=path. LABEL is just what shows in the table's first column.
"""
import pathlib
import re
import sys


def _f(pattern: str, text: str):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def parse_log(text: str) -> dict:
    """Pull the comparable numbers out of one inference log (either script's format)."""
    out = {}
    out["cosine"] = _f(r"mean cosine\(pred,GT\) dir:\s*([-\d.]+)", text)
    out["ratio"] = _f(r"magnitude ratio\s*\|pred\|/\|GT\|:\s*([-\d.]+)", text)
    # task / dim: oic prints "pos=... rot=..."; object_in_bowl prints "arm=... hand=..."
    if "pos=" in text:
        out["task"] = "oic (6-dim Cartesian)"
        out["space"] = "Cartesian pos (3)"
        out["model_mse"] = _f(r"(?:oic finetuned|oic base):\s*total=([\d.]+)", text)
        out["pos_mse"] = _f(r"(?:oic finetuned|oic base):.*?pos=([\d.]+)", text)
        out["zero_mse"] = _f(r"zero action:\s*total=([\d.]+)", text)
        out["const_mse"] = _f(r"const state:\s*total=([\d.]+)", text)
    else:
        out["task"] = "object_in_bowl (24-dim joint)"
        out["space"] = "joint (7)"
        out["model_mse"] = _f(r"pi0\.5 (?:finetuned|base):.*?total=([\d.]+)", text)
        out["pos_mse"] = _f(r"pi0\.5 (?:finetuned|base):\s*arm=([\d.]+)", text)  # arm MSE ~ "position" proxy
        out["zero_mse"] = _f(r"zero action:.*?total=([\d.]+)", text)
        out["const_mse"] = _f(r"const state:.*?total=([\d.]+)", text)
    # chunks evaluated (sanity)
    out["chunks"] = _f(r"chunks where GT moves(?:[^:]*)?:\s*(\d+)", text)
    return out


def main(argv):
    if not argv:
        print(__doc__)
        return
    rows = []
    for arg in argv:
        if "=" not in arg:
            print(f"skip (need LABEL=path): {arg}")
            continue
        label, path = arg.split("=", 1)
        p = pathlib.Path(path)
        if not p.exists():
            print(f"skip (missing): {path}")
            continue
        d = parse_log(p.read_text(errors="ignore"))
        d["label"] = label
        rows.append(d)

    def cell(v, fmt="{:.3f}"):
        return fmt.format(v) if isinstance(v, float) else "—"

    print("\n## Real-frame inference comparison\n")
    hdr = ["model", "task", "dir cosine", "mag ratio", "model MSE", "vs const", "vs zero", "dir space"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        print("| " + " | ".join([
            r["label"], r.get("task", "—"),
            cell(r.get("cosine")), cell(r.get("ratio")),
            cell(r.get("model_mse"), "{:.4f}"),
            cell(r.get("const_mse"), "{:.4f}"),
            cell(r.get("zero_mse"), "{:.4f}"),
            r.get("space", "—"),
        ]) + " |")

    print(
        "\n*cosine 1=correct direction, 0=random, <0=opposite; ratio 1=matches demo reach, "
        "~0=frozen. 'model MSE' beats 'vs const'/'vs zero' when the policy genuinely predicts "
        "motion. NOTE: object_in_bowl cosine is in 7-dim JOINT space, oic in 3-dim CARTESIAN "
        "position — compare base-vs-finetuned within a task directly; across tasks read only "
        "as 'does it reach correctly at all'.*"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
