"""Calibration tooling: compare VLM labels against hand labels.

Computes Cohen's kappa and a per-class confusion matrix on the two most
informative fields (``outcome`` and ``primary_failure``).

Both input files are JSON Lines with the same schema as
``label_directory`` produces, i.e. each line is:

    {"rollout_id": "...", "ok": true, "result": {"label": {<FailureLabel>}, ...}}

The hand-label file uses the same shape — easiest way to produce it is to copy
a VLM-generated JSONL and hand-edit the ``label`` fields, leaving everything
else as-is.

Usage::

    python -m labeling.calibrate \\
        --hand hand_labels.jsonl \\
        --vlm vlm_labels.jsonl \\
        --out calibration_report.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _load_labels(jsonl_path: str | Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not rec.get("ok", True):
                continue
            rid = rec["rollout_id"]
            label = rec.get("result", {}).get("label") or rec.get("label")
            if label is None:
                continue
            out[rid] = label
    return out


def _cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    """Compute Cohen's kappa for a list of ``(rater_a, rater_b)`` pairs."""
    if not pairs:
        return float("nan")
    n = len(pairs)
    categories = sorted({c for pair in pairs for c in pair})
    cat_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    matrix = [[0] * k for _ in range(k)]
    for a, b in pairs:
        matrix[cat_idx[a]][cat_idx[b]] += 1
    agree = sum(matrix[i][i] for i in range(k))
    po = agree / n
    a_totals = [sum(matrix[i]) for i in range(k)]
    b_totals = [sum(matrix[j][i] for j in range(k)) for i in range(k)]
    pe = sum(a_totals[i] * b_totals[i] for i in range(k)) / (n * n)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def _confusion_matrix(pairs: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for a, b in pairs:
        matrix[a][b] += 1
    return {k: dict(v) for k, v in matrix.items()}


def calibrate(hand_jsonl: str | Path, vlm_jsonl: str | Path) -> dict:
    hand = _load_labels(hand_jsonl)
    vlm = _load_labels(vlm_jsonl)
    shared = sorted(set(hand) & set(vlm))
    if not shared:
        return {
            "n_shared": 0,
            "warning": "No rollout_ids appear in both files.",
            "hand_only": sorted(set(hand) - set(vlm)),
            "vlm_only": sorted(set(vlm) - set(hand)),
        }

    outcome_pairs = [(hand[r]["outcome"], vlm[r]["outcome"]) for r in shared]
    failure_pairs = [(hand[r]["primary_failure"], vlm[r]["primary_failure"]) for r in shared]

    return {
        "n_shared": len(shared),
        "outcome": {
            "kappa": _cohen_kappa(outcome_pairs),
            "accuracy": sum(1 for a, b in outcome_pairs if a == b) / len(outcome_pairs),
            "confusion": _confusion_matrix(outcome_pairs),
            "hand_dist": dict(Counter(a for a, _ in outcome_pairs)),
            "vlm_dist": dict(Counter(b for _, b in outcome_pairs)),
        },
        "primary_failure": {
            "kappa": _cohen_kappa(failure_pairs),
            "accuracy": sum(1 for a, b in failure_pairs if a == b) / len(failure_pairs),
            "confusion": _confusion_matrix(failure_pairs),
            "hand_dist": dict(Counter(a for a, _ in failure_pairs)),
            "vlm_dist": dict(Counter(b for _, b in failure_pairs)),
        },
        "ambiguous_in_vlm_but_not_hand": [
            r for r in shared if vlm[r]["outcome"] == "ambiguous" and hand[r]["outcome"] != "ambiguous"
        ],
        "vlm_overconfident_failures": [
            r
            for r in shared
            if hand[r]["outcome"] == "success"
            and vlm[r]["outcome"] == "failure"
            and vlm[r].get("confidence", 0) > 0.8
        ],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="VLM-judge calibration report.")
    p.add_argument("--hand", required=True, help="Hand-labeled JSONL.")
    p.add_argument("--vlm", required=True, help="VLM-labeled JSONL.")
    p.add_argument("--out", default=None, help="Write report JSON here (else stdout).")
    args = p.parse_args(argv)

    report = calibrate(args.hand, args.vlm)
    out_str = json.dumps(report, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(out_str)
        print(f"Wrote calibration report to {args.out}")
    else:
        print(out_str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
