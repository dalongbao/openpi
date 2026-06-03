"""Tests for the calibration tool (Cohen's kappa + confusion matrix)."""

from __future__ import annotations

import json
import math
from pathlib import Path

from labeling.calibrate import _cohen_kappa, calibrate


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _label_record(rid: str, outcome: str, primary: str) -> dict:
    return {
        "rollout_id": rid,
        "ok": True,
        "result": {
            "label": {
                "outcome": outcome,
                "primary_failure": primary,
                "secondary_failures": [],
                "confidence": 0.9,
                "rationale": "fake",
                "grounded_to_instruction": True,
                "object_grasped": True,
                "target_reached": outcome == "success",
            }
        },
    }


def test_perfect_agreement(tmp_path):
    hand = tmp_path / "hand.jsonl"
    vlm = tmp_path / "vlm.jsonl"
    records = [_label_record(f"r{i}", "failure", "wrong_object") for i in range(5)] + [
        _label_record("rsucc", "success", "none")
    ]
    _write_jsonl(hand, records)
    _write_jsonl(vlm, records)

    rep = calibrate(hand, vlm)
    assert rep["n_shared"] == 6
    assert rep["outcome"]["kappa"] == 1.0
    assert rep["outcome"]["accuracy"] == 1.0
    assert rep["primary_failure"]["kappa"] == 1.0


def test_no_overlap(tmp_path):
    hand = tmp_path / "hand.jsonl"
    vlm = tmp_path / "vlm.jsonl"
    _write_jsonl(hand, [_label_record("r1", "success", "none")])
    _write_jsonl(vlm, [_label_record("r2", "success", "none")])

    rep = calibrate(hand, vlm)
    assert rep["n_shared"] == 0
    assert "warning" in rep


def test_partial_agreement(tmp_path):
    hand = tmp_path / "hand.jsonl"
    vlm = tmp_path / "vlm.jsonl"
    hand_recs = [
        _label_record("r1", "success", "none"),
        _label_record("r2", "failure", "wrong_object"),
        _label_record("r3", "failure", "wrong_object"),
        _label_record("r4", "failure", "wrong_position"),
    ]
    vlm_recs = [
        _label_record("r1", "success", "none"),
        _label_record("r2", "failure", "wrong_object"),
        _label_record("r3", "failure", "wrong_position"),  # disagree
        _label_record("r4", "ambiguous", "other"),  # disagree
    ]
    _write_jsonl(hand, hand_recs)
    _write_jsonl(vlm, vlm_recs)

    rep = calibrate(hand, vlm)
    assert rep["n_shared"] == 4
    assert 0.0 < rep["outcome"]["accuracy"] < 1.0
    assert not math.isnan(rep["outcome"]["kappa"])


def test_cohen_kappa_basic():
    # All agree
    assert _cohen_kappa([("a", "a"), ("b", "b")]) == 1.0
    # All disagree, two raters always choose opposite -> kappa <= 0
    k = _cohen_kappa([("a", "b"), ("b", "a"), ("a", "b"), ("b", "a")])
    assert k <= 0.0
