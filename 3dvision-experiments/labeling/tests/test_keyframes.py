"""Keyframe extraction test using the existing evaluation.mp4 fixture."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from labeling.vlm_judge import extract_keyframes


FIXTURE_MP4 = Path(__file__).resolve().parents[2] / "isaac-sim" / "evaluation.mp4"


@pytest.mark.skipif(not FIXTURE_MP4.exists(), reason=f"missing fixture {FIXTURE_MP4}")
def test_extract_keyframes_default_count():
    frames = extract_keyframes(FIXTURE_MP4, n=8)
    assert len(frames) == 8
    # JPEG bytes start with FF D8 FF
    for f in frames:
        assert isinstance(f, bytes)
        assert f[:3] == b"\xff\xd8\xff", "Frame is not JPEG-encoded"
        # Sanity: encoded payload should be plausibly-sized (>1KB, <2MB)
        assert 1024 < len(f) < 2 * 1024 * 1024


@pytest.mark.skipif(not FIXTURE_MP4.exists(), reason=f"missing fixture {FIXTURE_MP4}")
def test_extract_keyframes_one_frame():
    frames = extract_keyframes(FIXTURE_MP4, n=1)
    assert len(frames) == 1


def test_extract_keyframes_missing_file():
    with pytest.raises(FileNotFoundError):
        extract_keyframes("/tmp/definitely_not_a_real_file.mp4", n=4)


def test_extract_keyframes_bad_n():
    import pytest as _pytest

    with _pytest.raises(ValueError):
        extract_keyframes(FIXTURE_MP4 if FIXTURE_MP4.exists() else __file__, n=0)
