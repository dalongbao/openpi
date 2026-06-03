"""Prompt construction for the VLM judge.

We use Claude tool-use to force a structured JSON response that matches
``FailureLabel``. The tool schema is exposed as ``LABEL_TOOL`` and is what
``vlm_judge.label_rollout`` passes to the SDK.
"""

from __future__ import annotations

import base64
from typing import Any


SYSTEM_PROMPT = """You are an expert in robotic manipulation analysis. You will be shown a
short video (sampled as evenly-spaced keyframes) of a Franka FR3 robot arm attempting to
perform a tabletop manipulation task in a simulated kitchen scene. Your job is to classify
the attempt using the RoboFAC failure-mode taxonomy (arXiv:2505.12224), extended for
simulator-specific failures.

Outcome categories:
  - "success": the robot completed the requested task.
  - "failure": the robot did not complete the task.
  - "ambiguous": the video is too short, occluded, or otherwise insufficient to tell.

Primary failure modes (use "none" only when outcome=="success"):
  - "wrong_object": the robot interacted with an object other than the one named in the
    instruction.
  - "wrong_position": the robot brought the object somewhere other than the target.
  - "wrong_orientation": the robot grasped/placed with an orientation that prevents the
    task from succeeding (e.g. plate placed upside-down).
  - "wrong_timing": the robot released or grasped at the wrong moment (e.g. dropped the
    object mid-air, released before reaching the target).
  - "instruction_ignored": the robot did nothing relevant — minimal motion, motion unrelated
    to the instruction, or only flailing.
  - "physically_impossible": a simulator artifact made the task impossible (gripper
    clipping through geometry, object spawned wrong, physics explosion).
  - "other": none of the above.

Be conservative with "success" — only label success if the goal state is visibly reached.
Be conservative with "ambiguous" — if you can see motion clearly, prefer a real label over
"ambiguous".

You MUST respond by calling the ``submit_failure_label`` tool. Do not respond in free
text. Provide a 2-4 sentence rationale grounded in specific frames you observed.
"""


USER_PROMPT_TEMPLATE = """The robot was given the instruction:

  "{instruction}"

You are watching a video showing the attempt from a third-person view. {n_keyframes}
keyframes are provided below, in chronological order (earliest first). Joint-trajectory
data is also available out-of-band; you do not need to inspect it directly, but you may
assume the robot has 7 arm joints + a 2-finger gripper.

Classify the attempt. Call the ``submit_failure_label`` tool with your structured
judgment. Ground your rationale in specific frames (e.g. "in frame 3 the gripper is
empty but moving towards the plate").
"""


# Tool definition used to force structured output from Claude. This matches the
# ``FailureLabel`` pydantic model in ``schema.py``.
LABEL_TOOL: dict[str, Any] = {
    "name": "submit_failure_label",
    "description": (
        "Submit a structured RoboFAC-style failure label for the manipulation attempt "
        "shown in the video. You MUST call this tool exactly once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "outcome": {
                "type": "string",
                "enum": ["success", "failure", "ambiguous"],
                "description": "Top-level outcome of the attempt.",
            },
            "primary_failure": {
                "type": "string",
                "enum": [
                    "none",
                    "wrong_object",
                    "wrong_position",
                    "wrong_orientation",
                    "wrong_timing",
                    "instruction_ignored",
                    "physically_impossible",
                    "other",
                ],
                "description": (
                    "Primary failure mode. Use 'none' only when outcome is 'success'."
                ),
            },
            "secondary_failures": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Free-form list of additional issues observed (may be empty)."
                ),
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Your confidence in the primary label, 0-1.",
            },
            "rationale": {
                "type": "string",
                "description": "2-4 sentence explanation grounded in specific frames.",
            },
            "grounded_to_instruction": {
                "type": "boolean",
                "description": "Did the robot attempt the right object per instruction?",
            },
            "object_grasped": {
                "type": "boolean",
                "description": "Did the robot grasp ANY object during the rollout?",
            },
            "target_reached": {
                "type": "boolean",
                "description": "Did the robot bring the object near the target?",
            },
        },
        "required": [
            "outcome",
            "primary_failure",
            "secondary_failures",
            "confidence",
            "rationale",
            "grounded_to_instruction",
            "object_grasped",
            "target_reached",
        ],
    },
}


def build_messages(
    instruction: str,
    video_frames: list[bytes],
    n_keyframes: int = 8,
    image_media_type: str = "image/jpeg",
) -> list[dict[str, Any]]:
    """Build the messages list for the Anthropic API.

    Args:
        instruction: The language instruction the robot was given.
        video_frames: A list of frame bytes (already encoded as JPEG/PNG).
        n_keyframes: Used only for the user-prompt template; the actual frame
            count is ``len(video_frames)``.
        image_media_type: MIME type for the inline images.

    Returns:
        A ``messages`` list ready to pass to ``client.messages.create(...)``.
    """
    if not video_frames:
        raise ValueError("video_frames must be non-empty")

    user_text = USER_PROMPT_TEMPLATE.format(
        instruction=instruction.strip(),
        n_keyframes=len(video_frames),
    )

    content: list[dict[str, Any]] = []
    for i, frame in enumerate(video_frames):
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image_media_type,
                    "data": base64.b64encode(frame).decode("ascii"),
                },
            }
        )
        # Tiny label after each frame so the model can refer to "frame N".
        content.append({"type": "text", "text": f"^ frame {i + 1}/{len(video_frames)}"})

    content.append({"type": "text", "text": user_text})

    return [{"role": "user", "content": content}]
