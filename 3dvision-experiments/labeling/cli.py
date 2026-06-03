"""Command-line interface for the VLM-judge pipeline.

Usage:

    python -m labeling.cli label-one \\
        --video evaluation.mp4 \\
        --instruction "put the plate in the crate"

    python -m labeling.cli label-dir \\
        --rollouts-dir runs/ \\
        --instructions instructions.csv \\
        --out labels.jsonl

Both subcommands honor ``ANTHROPIC_API_KEY`` from the environment.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .vlm_judge import label_directory, label_rollout


def _cmd_label_one(args: argparse.Namespace) -> int:
    result = label_rollout(
        video_path=args.video,
        instruction=args.instruction,
        joint_csv_path=args.joint_csv,
        model_id=args.model_id,
        n_keyframes=args.n_keyframes,
    )
    payload = result.model_dump()
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"Wrote label to {args.out}")
    else:
        json.dump(payload, sys.stdout, indent=2)
        print()
    return 0


def _cmd_label_dir(args: argparse.Namespace) -> int:
    label_directory(
        rollouts_dir=args.rollouts_dir,
        instructions_csv=args.instructions,
        output_jsonl=args.out,
        parallelism=args.parallelism,
        resume=not args.no_resume,
        model_id=args.model_id,
        n_keyframes=args.n_keyframes,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="labeling", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    one = sub.add_parser("label-one", help="Label a single rollout video.")
    one.add_argument("--video", required=True, help="Path to rollout MP4.")
    one.add_argument("--instruction", required=True, help="Language instruction.")
    one.add_argument("--joint-csv", default=None, help="Optional joint CSV path.")
    one.add_argument("--model-id", default="claude-opus-4-6")
    one.add_argument("--n-keyframes", type=int, default=8)
    one.add_argument("--out", default=None, help="Optional file to write JSON to.")
    one.set_defaults(func=_cmd_label_one)

    batch = sub.add_parser("label-dir", help="Batch-label a directory of rollouts.")
    batch.add_argument("--rollouts-dir", required=True)
    batch.add_argument(
        "--instructions",
        required=True,
        help="CSV with columns 'rollout_id,instruction'.",
    )
    batch.add_argument("--out", required=True, help="Output JSON-lines file.")
    batch.add_argument("--parallelism", type=int, default=4)
    batch.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-label rollouts that are already in the output file.",
    )
    batch.add_argument("--model-id", default="claude-opus-4-6")
    batch.add_argument("--n-keyframes", type=int, default=8)
    batch.set_defaults(func=_cmd_label_dir)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
