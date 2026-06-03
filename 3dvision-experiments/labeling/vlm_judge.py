"""Main VLM-judge implementation.

Public surface:
  - ``extract_keyframes(video_path, n) -> list[bytes]``
  - ``label_rollout(...) -> LabelingResult``
  - ``label_directory(...)`` — batch + resume + parallel async runner.

The labeler relies on Anthropic tool-use to force a structured response. We
DO NOT regex-parse free-text JSON. If the model fails to call the tool we raise
``VLMJudgeError`` so the caller can retry.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import imageio.v3 as iio
import numpy as np
from PIL import Image
from pydantic import ValidationError

from .prompts import LABEL_TOOL, SYSTEM_PROMPT, build_messages
from .schema import FailureLabel, LabelingResult


logger = logging.getLogger(__name__)


# Approximate per-image cost for Claude Opus vision input (resized to ~1MP).
# This is a rough estimate; the user is expected to verify against current
# Anthropic pricing.
_OPUS_INPUT_PER_MTOK_USD = 15.0  # $/M input tokens
_OPUS_OUTPUT_PER_MTOK_USD = 75.0  # $/M output tokens
_TOKENS_PER_IMAGE_APPROX = 1500  # ~1MP image at typical resolution


class VLMJudgeError(RuntimeError):
    """Raised when the model fails to produce a valid structured label."""


# ---------------------------------------------------------------------------
# Keyframe extraction
# ---------------------------------------------------------------------------


def _resize_jpeg(frame: np.ndarray, max_side: int = 1024, quality: int = 85) -> bytes:
    """Resize an RGB frame and return JPEG-encoded bytes."""
    img = Image.fromarray(frame)
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def extract_keyframes(video_path: str | os.PathLike, n: int = 8) -> list[bytes]:
    """Extract ``n`` evenly-spaced keyframes from an MP4 as JPEG bytes.

    Uses imageio (ffmpeg backend). Raises FileNotFoundError if the video is
    missing.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if n < 1:
        raise ValueError("n must be >= 1")

    # imageio.v3 can iterate frames lazily but to get evenly spaced samples we
    # first need the total frame count. iio.improps gives us metadata.
    try:
        props = iio.improps(str(video_path), plugin="pyav")
        total = int(props.shape[0]) if props.shape and len(props.shape) > 0 else 0
    except Exception:  # noqa: BLE001 — pyav may not be present
        total = 0

    if total <= 0:
        # Fallback: decode everything into memory. Fine for short eval videos
        # (the existing evaluation.mp4 is ~3000 frames at 720p, ~2.5MB on disk).
        frames = list(iio.imiter(str(video_path)))
        total = len(frames)
        if total == 0:
            raise VLMJudgeError(f"No frames decoded from {video_path}")
        indices = np.linspace(0, total - 1, num=min(n, total), dtype=int)
        sampled = [frames[i] for i in indices]
    else:
        indices = np.linspace(0, total - 1, num=min(n, total), dtype=int)
        # iio.imread with index= reads a single frame
        sampled = [iio.imread(str(video_path), index=int(i), plugin="pyav") for i in indices]

    return [_resize_jpeg(np.asarray(f)) for f in sampled]


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


def _estimate_cost_usd(
    n_images: int,
    output_tokens: int = 400,
    input_tokens_text: int = 1200,
) -> float:
    input_tokens = input_tokens_text + n_images * _TOKENS_PER_IMAGE_APPROX
    return (
        input_tokens / 1e6 * _OPUS_INPUT_PER_MTOK_USD
        + output_tokens / 1e6 * _OPUS_OUTPUT_PER_MTOK_USD
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_tool_response(response: Any) -> tuple[FailureLabel, str]:
    """Extract the ``submit_failure_label`` tool call from the API response.

    Returns ``(label, raw_response_text)``.
    """
    raw_repr = ""
    tool_use_block: dict[str, Any] | None = None

    # ``response.content`` is a list of content blocks (TextBlock, ToolUseBlock, ...)
    content = getattr(response, "content", None)
    if content is None and isinstance(response, dict):
        content = response.get("content")

    if content is None:
        raise VLMJudgeError("API response had no content")

    for block in content:
        btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
        if btype == "tool_use":
            name = getattr(block, "name", None) or block.get("name")
            if name == "submit_failure_label":
                tool_use_block = getattr(block, "input", None) or block.get("input")
        elif btype == "text":
            text = getattr(block, "text", None) or block.get("text", "")
            raw_repr += text + "\n"

    if tool_use_block is None:
        raise VLMJudgeError(
            "Model did not call submit_failure_label tool. Raw text: " + raw_repr[:400]
        )

    try:
        label = FailureLabel.model_validate(tool_use_block)
    except ValidationError as e:
        raise VLMJudgeError(f"Tool input failed schema validation: {e}") from e

    # Best-effort raw response capture: dump the tool input as JSON for the
    # audit trail.
    raw_repr += json.dumps(tool_use_block, default=str)
    return label, raw_repr.strip()


# ---------------------------------------------------------------------------
# Single-video labeling
# ---------------------------------------------------------------------------


def label_rollout(
    video_path: str | os.PathLike,
    instruction: str,
    joint_csv_path: str | os.PathLike | None = None,
    model_id: str = "claude-opus-4-6",
    n_keyframes: int = 8,
    api_key: str | None = None,
    client: Any | None = None,
    max_tokens: int = 1024,
) -> LabelingResult:
    """Run the VLM judge on a single rollout.

    Args:
        video_path: Path to the recording MP4.
        instruction: The natural-language instruction the policy was given.
        joint_csv_path: Optional path to the per-step joint CSV. Currently not
            inlined into the prompt (we found Claude is better at scoring from
            visual evidence alone) but the path is logged for traceability.
        model_id: Anthropic model identifier.
        n_keyframes: Number of evenly-spaced frames to sample from the video.
        api_key: Optional override for the API key (else uses
            ``ANTHROPIC_API_KEY``).
        client: Optional pre-built ``anthropic.Anthropic`` client (used by
            tests to mock the SDK).
        max_tokens: Cap on output tokens.

    Returns:
        ``LabelingResult`` with the validated label, raw response, and timing.
    """
    if client is None:
        import anthropic  # local import so tests can run without the SDK installed

        client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    frames = extract_keyframes(video_path, n=n_keyframes)
    messages = build_messages(instruction, frames, n_keyframes=n_keyframes)

    # Capture the prompt that was actually used (without inlining the huge
    # base64 payloads).
    prompt_used = json.dumps(
        {
            "system": SYSTEM_PROMPT,
            "user_text": messages[0]["content"][-1]["text"],
            "n_frames": len(frames),
            "instruction": instruction,
            "joint_csv_path": str(joint_csv_path) if joint_csv_path else None,
        },
        indent=2,
    )

    t0 = time.perf_counter()
    response = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        tools=[LABEL_TOOL],
        tool_choice={"type": "tool", "name": "submit_failure_label"},
        messages=messages,
    )
    latency = time.perf_counter() - t0

    label, raw = _parse_tool_response(response)

    return LabelingResult(
        label=label,
        prompt_used=prompt_used,
        model_id=model_id,
        latency_seconds=latency,
        raw_response=raw,
        cost_usd_estimate=_estimate_cost_usd(n_images=len(frames)),
    )


# ---------------------------------------------------------------------------
# Async helpers for batch labeling
# ---------------------------------------------------------------------------


@dataclass
class _RolloutEntry:
    rollout_id: str
    video_path: Path
    instruction: str
    joint_csv_path: Path | None


def _load_instructions_csv(csv_path: str | os.PathLike) -> dict[str, str]:
    """Load instructions CSV: must have columns ``rollout_id,instruction``."""
    out: dict[str, str] = {}
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "rollout_id" not in reader.fieldnames or "instruction" not in reader.fieldnames:
            raise ValueError(
                f"instructions CSV must have columns 'rollout_id,instruction'; got {reader.fieldnames}"
            )
        for row in reader:
            out[row["rollout_id"].strip()] = row["instruction"].strip()
    return out


def _discover_rollouts(rollouts_dir: Path, instructions: dict[str, str]) -> list[_RolloutEntry]:
    """Find rollout folders under ``rollouts_dir``.

    Each rollout is a folder containing an MP4. The rollout id is the folder
    name. If a ``metadata.json`` with an ``instruction`` field is present it
    overrides the instructions CSV.
    """
    entries: list[_RolloutEntry] = []
    rollouts_dir = Path(rollouts_dir)
    if not rollouts_dir.exists():
        raise FileNotFoundError(rollouts_dir)

    # Two layouts supported: (a) one mp4 per subdir; (b) flat dir of mp4 files.
    candidates: list[tuple[str, Path]] = []
    for child in sorted(rollouts_dir.iterdir()):
        if child.is_dir():
            mp4s = sorted(child.glob("*.mp4"))
            if mp4s:
                candidates.append((child.name, mp4s[0]))
        elif child.is_file() and child.suffix.lower() == ".mp4":
            candidates.append((child.stem, child))

    for rid, mp4 in candidates:
        instruction = instructions.get(rid)
        meta = mp4.parent / "metadata.json"
        if meta.exists():
            try:
                m = json.loads(meta.read_text())
                instruction = m.get("instruction", instruction)
            except Exception:  # noqa: BLE001
                logger.warning("Failed to parse %s", meta)
        if not instruction:
            logger.warning("No instruction for rollout %s; skipping", rid)
            continue

        joint_csv = mp4.parent / "results.csv"
        entries.append(
            _RolloutEntry(
                rollout_id=rid,
                video_path=mp4,
                instruction=instruction,
                joint_csv_path=joint_csv if joint_csv.exists() else None,
            )
        )
    return entries


def _existing_rollout_ids(jsonl_path: Path) -> set[str]:
    """Return rollout ids already labeled in the JSONL file (for resume)."""
    if not jsonl_path.exists():
        return set()
    ids: set[str] = set()
    with jsonl_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rid = rec.get("rollout_id")
                if rid:
                    ids.add(rid)
            except json.JSONDecodeError:
                continue
    return ids


async def _label_one_async(
    entry: _RolloutEntry,
    model_id: str,
    n_keyframes: int,
    sem: asyncio.Semaphore,
    api_key: str | None,
) -> dict[str, Any]:
    """Label a single rollout via the async API."""
    import anthropic

    async_client = anthropic.AsyncAnthropic(
        api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
    )

    async with sem:
        # Keyframe extraction is sync/CPU-bound; run in default loop executor.
        loop = asyncio.get_running_loop()
        frames = await loop.run_in_executor(
            None, lambda: extract_keyframes(entry.video_path, n=n_keyframes)
        )
        messages = build_messages(entry.instruction, frames, n_keyframes=n_keyframes)

        t0 = time.perf_counter()
        try:
            response = await async_client.messages.create(
                model=model_id,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=[LABEL_TOOL],
                tool_choice={"type": "tool", "name": "submit_failure_label"},
                messages=messages,
            )
            latency = time.perf_counter() - t0
            label, raw = _parse_tool_response(response)
            result = LabelingResult(
                label=label,
                prompt_used=json.dumps({"instruction": entry.instruction, "n_frames": len(frames)}),
                model_id=model_id,
                latency_seconds=latency,
                raw_response=raw,
                cost_usd_estimate=_estimate_cost_usd(n_images=len(frames)),
            )
            return {
                "rollout_id": entry.rollout_id,
                "video_path": str(entry.video_path),
                "instruction": entry.instruction,
                "ok": True,
                "result": result.model_dump(),
            }
        except Exception as e:  # noqa: BLE001
            return {
                "rollout_id": entry.rollout_id,
                "video_path": str(entry.video_path),
                "instruction": entry.instruction,
                "ok": False,
                "error": str(e),
            }


def label_directory(
    rollouts_dir: str | os.PathLike,
    instructions_csv: str | os.PathLike,
    output_jsonl: str | os.PathLike,
    parallelism: int = 4,
    resume: bool = True,
    model_id: str = "claude-opus-4-6",
    n_keyframes: int = 8,
    api_key: str | None = None,
) -> None:
    """Batch label all rollouts under ``rollouts_dir``.

    Writes one JSON object per line to ``output_jsonl``. Each line has shape::

        {"rollout_id": "...", "video_path": "...", "instruction": "...",
         "ok": True, "result": {<LabelingResult>}}

    If a rollout fails (model error, validation error, ...) the line has
    ``"ok": False`` and an ``"error"`` field instead of ``"result"``. This
    keeps the JSONL forward-progress invariant: every rollout produces exactly
    one line, success or failure.
    """
    rollouts_dir = Path(rollouts_dir)
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    instructions = _load_instructions_csv(instructions_csv)
    entries = _discover_rollouts(rollouts_dir, instructions)
    logger.info("Discovered %d rollouts in %s", len(entries), rollouts_dir)

    done = _existing_rollout_ids(output_jsonl) if resume else set()
    todo = [e for e in entries if e.rollout_id not in done]
    logger.info("%d already labeled, %d to label", len(done), len(todo))

    if not todo:
        return

    async def _run() -> None:
        sem = asyncio.Semaphore(parallelism)
        tasks = [
            _label_one_async(e, model_id=model_id, n_keyframes=n_keyframes, sem=sem, api_key=api_key)
            for e in todo
        ]
        # Append-as-we-go so a crash mid-batch doesn't lose work.
        with output_jsonl.open("a") as fh:
            for coro in asyncio.as_completed(tasks):
                rec = await coro
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                rid = rec.get("rollout_id")
                ok = rec.get("ok")
                logger.info("Labeled %s ok=%s", rid, ok)

    asyncio.run(_run())
