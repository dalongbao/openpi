"""Pydantic schemas for the VLM-judge failure-mode labeling pipeline.

The taxonomy extends RoboFAC (arXiv:2505.12224) with three additional outcomes
that show up in simulation rollouts (instruction-ignored, physically-impossible,
and "success").
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


PrimaryFailure = Literal[
    "none",  # used when outcome == "success"
    "wrong_object",
    "wrong_position",
    "wrong_orientation",
    "wrong_timing",
    "instruction_ignored",
    "physically_impossible",
    "other",
]

Outcome = Literal["success", "failure", "ambiguous"]


class FailureLabel(BaseModel):
    """Structured failure label produced by the VLM judge."""

    outcome: Outcome
    primary_failure: PrimaryFailure
    secondary_failures: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    grounded_to_instruction: bool
    object_grasped: bool
    target_reached: bool

    @field_validator("rationale")
    @classmethod
    def _rationale_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("rationale must be a non-empty string")
        return v.strip()

    @field_validator("primary_failure")
    @classmethod
    def _check_consistency(cls, v: PrimaryFailure, info) -> PrimaryFailure:
        # If outcome is success, primary_failure should be "none".
        # We can't access other fields directly in v1-style validators, so do
        # this as a model_validator below.
        return v


class LabelingResult(BaseModel):
    """End-to-end result of a single VLM labeling call."""

    label: FailureLabel
    prompt_used: str
    model_id: str
    latency_seconds: float
    raw_response: str
    cost_usd_estimate: float | None = None
