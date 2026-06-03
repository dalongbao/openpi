"""Pydantic round-trip tests for the labeling schemas."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from labeling.schema import FailureLabel, LabelingResult


def _good_label_dict() -> dict:
    return {
        "outcome": "failure",
        "primary_failure": "wrong_position",
        "secondary_failures": ["dropped object mid-air"],
        "confidence": 0.82,
        "rationale": "Robot grasped plate but placed it on the table instead of the crate.",
        "grounded_to_instruction": True,
        "object_grasped": True,
        "target_reached": False,
    }


def test_failure_label_roundtrip():
    raw = _good_label_dict()
    label = FailureLabel.model_validate(raw)
    assert label.outcome == "failure"
    assert label.primary_failure == "wrong_position"
    dumped = label.model_dump()
    assert dumped["outcome"] == raw["outcome"]
    assert dumped["secondary_failures"] == raw["secondary_failures"]
    # JSON round-trip
    rehydrated = FailureLabel.model_validate(json.loads(label.model_dump_json()))
    assert rehydrated == label


def test_failure_label_defaults_secondary():
    raw = _good_label_dict()
    raw.pop("secondary_failures")
    label = FailureLabel.model_validate(raw)
    assert label.secondary_failures == []


def test_failure_label_rejects_bad_confidence():
    raw = _good_label_dict()
    raw["confidence"] = 1.5
    with pytest.raises(ValidationError):
        FailureLabel.model_validate(raw)


def test_failure_label_rejects_unknown_outcome():
    raw = _good_label_dict()
    raw["outcome"] = "kinda_worked"
    with pytest.raises(ValidationError):
        FailureLabel.model_validate(raw)


def test_failure_label_rejects_unknown_failure_mode():
    raw = _good_label_dict()
    raw["primary_failure"] = "the_vibes_were_off"
    with pytest.raises(ValidationError):
        FailureLabel.model_validate(raw)


def test_failure_label_rejects_empty_rationale():
    raw = _good_label_dict()
    raw["rationale"] = "   "
    with pytest.raises(ValidationError):
        FailureLabel.model_validate(raw)


def test_labeling_result_roundtrip():
    result = LabelingResult(
        label=FailureLabel.model_validate(_good_label_dict()),
        prompt_used="some prompt",
        model_id="claude-opus-4-6",
        latency_seconds=2.34,
        raw_response="raw json",
        cost_usd_estimate=0.018,
    )
    rehydrated = LabelingResult.model_validate(json.loads(result.model_dump_json()))
    assert rehydrated == result
    assert rehydrated.cost_usd_estimate == pytest.approx(0.018)


def test_labeling_result_optional_cost():
    result = LabelingResult(
        label=FailureLabel.model_validate(_good_label_dict()),
        prompt_used="prompt",
        model_id="claude-opus-4-6",
        latency_seconds=1.0,
        raw_response="raw",
    )
    assert result.cost_usd_estimate is None
