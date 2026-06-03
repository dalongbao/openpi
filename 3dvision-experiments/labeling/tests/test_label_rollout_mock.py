"""Unit test of ``label_rollout`` with the Anthropic SDK mocked.

We avoid burning API tokens by passing a fake client into ``label_rollout``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from labeling.schema import FailureLabel
from labeling.vlm_judge import VLMJudgeError, label_rollout


FIXTURE_MP4 = Path(__file__).resolve().parents[2] / "isaac-sim" / "evaluation.mp4"


def _fake_response(tool_input: dict) -> SimpleNamespace:
    """Construct a fake anthropic response object with one tool_use block."""
    tool_block = SimpleNamespace(
        type="tool_use",
        name="submit_failure_label",
        input=tool_input,
    )
    text_block = SimpleNamespace(type="text", text="ok")
    return SimpleNamespace(content=[text_block, tool_block])


def _good_input() -> dict:
    return {
        "outcome": "failure",
        "primary_failure": "wrong_position",
        "secondary_failures": [],
        "confidence": 0.7,
        "rationale": "Plate was picked up but placed on the table, not the crate.",
        "grounded_to_instruction": True,
        "object_grasped": True,
        "target_reached": False,
    }


@pytest.mark.skipif(not FIXTURE_MP4.exists(), reason=f"missing fixture {FIXTURE_MP4}")
def test_label_rollout_with_mocked_client():
    client = MagicMock()
    client.messages.create.return_value = _fake_response(_good_input())

    result = label_rollout(
        video_path=str(FIXTURE_MP4),
        instruction="put the plate in the crate",
        client=client,
        n_keyframes=3,
    )

    assert isinstance(result.label, FailureLabel)
    assert result.label.outcome == "failure"
    assert result.label.primary_failure == "wrong_position"
    assert result.latency_seconds >= 0
    assert result.cost_usd_estimate is not None and result.cost_usd_estimate > 0
    assert result.model_id == "claude-opus-4-6"

    # Verify the API was called once with the expected tool config.
    assert client.messages.create.call_count == 1
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-opus-4-6"
    assert kwargs["tool_choice"]["name"] == "submit_failure_label"
    assert len(kwargs["tools"]) == 1
    assert kwargs["tools"][0]["name"] == "submit_failure_label"
    # Check that we sent images.
    msgs = kwargs["messages"]
    image_blocks = [b for b in msgs[0]["content"] if b.get("type") == "image"]
    assert len(image_blocks) == 3


@pytest.mark.skipif(not FIXTURE_MP4.exists(), reason=f"missing fixture {FIXTURE_MP4}")
def test_label_rollout_raises_when_tool_not_called():
    client = MagicMock()
    # Response has only text content, no tool_use block.
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="I refuse to use a tool")]
    )

    with pytest.raises(VLMJudgeError):
        label_rollout(
            video_path=str(FIXTURE_MP4),
            instruction="put the plate in the crate",
            client=client,
            n_keyframes=2,
        )


@pytest.mark.skipif(not FIXTURE_MP4.exists(), reason=f"missing fixture {FIXTURE_MP4}")
def test_label_rollout_raises_on_invalid_tool_input():
    client = MagicMock()
    # Tool was called but with garbage that fails pydantic validation.
    bad = _good_input()
    bad["confidence"] = 1.5  # out of range
    client.messages.create.return_value = _fake_response(bad)

    with pytest.raises(VLMJudgeError):
        label_rollout(
            video_path=str(FIXTURE_MP4),
            instruction="put the plate in the crate",
            client=client,
            n_keyframes=2,
        )


@pytest.mark.skipif(not FIXTURE_MP4.exists(), reason=f"missing fixture {FIXTURE_MP4}")
def test_label_rollout_accepts_dict_blocks():
    """The parser should handle either SDK objects or dict-shaped blocks."""
    client = MagicMock()
    client.messages.create.return_value = {
        "content": [
            {"type": "text", "text": "preamble"},
            {"type": "tool_use", "name": "submit_failure_label", "input": _good_input()},
        ]
    }

    result = label_rollout(
        video_path=str(FIXTURE_MP4),
        instruction="put the plate in the crate",
        client=client,
        n_keyframes=2,
    )
    assert result.label.outcome == "failure"
