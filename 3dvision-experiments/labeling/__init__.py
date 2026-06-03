"""VLM-judge failure-mode labeling pipeline.

Public API:
    - :class:`FailureLabel`, :class:`LabelingResult` — pydantic schemas.
    - :func:`extract_keyframes` — sample frames from an MP4.
    - :func:`label_rollout` — label a single rollout.
    - :func:`label_directory` — batch + resumable async labeling.
"""

from .schema import FailureLabel, LabelingResult
from .vlm_judge import (
    VLMJudgeError,
    extract_keyframes,
    label_directory,
    label_rollout,
)

__all__ = [
    "FailureLabel",
    "LabelingResult",
    "VLMJudgeError",
    "extract_keyframes",
    "label_directory",
    "label_rollout",
]
