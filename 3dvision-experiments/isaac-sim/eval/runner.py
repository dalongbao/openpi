"""CLI runner for an Isaac Sim pi0.5 evaluation rollout.

This is the entry point invoked from ``submit.sh`` (via the shim
``eval_script_1.py``). Defaults match the legacy behavior exactly so a
no-args invocation is a parity run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import core
from . import metrics


def _load_json_or_none(path):
    if path is None:
        return None
    with open(path, "r") as f:
        return json.load(f)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a single pi0.5 rollout in Isaac Sim and dump metrics."
    )
    p.add_argument(
        "--checkpoint-dir",
        default=core.DEFAULT_CHECKPOINT_DIR,
        help=f"Path to the orbax checkpoint dir (default: {core.DEFAULT_CHECKPOINT_DIR}).",
    )
    p.add_argument(
        "--scene-usd",
        default=core.DEFAULT_USD_PATH,
        help=f"Path to the kitchen scene USD (default: {core.DEFAULT_USD_PATH}).",
    )
    p.add_argument(
        "--output-dir",
        default=core.DEFAULT_OUTPUT_DIR,
        help=f"Directory for results.csv, evaluation.mp4, metrics.json (default: {core.DEFAULT_OUTPUT_DIR}).",
    )
    p.add_argument(
        "--num-steps",
        type=int,
        default=core.DEFAULT_NUM_STEPS,
        help=f"Number of simulation steps to run (default: {core.DEFAULT_NUM_STEPS}).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for Python/NumPy/JAX (default: 42).",
    )
    p.add_argument(
        "--language-prompt",
        default=core.DEFAULT_LANG_PROMPT,
        help=f'Task prompt (default: "{core.DEFAULT_LANG_PROMPT}").',
    )
    p.add_argument(
        "--perturbation-config",
        default=None,
        help="Optional path to a JSON file describing scene perturbations.",
    )
    p.add_argument(
        "--probe-config",
        default=None,
        help="Optional path to a JSON file describing model probes.",
    )
    p.add_argument(
        "--record-external",
        action="store_true",
        help="Also record the 224x224 ExternalCamera (policy view) to evaluation_external.mp4.",
    )
    p.add_argument(
        "--no-record-recording",
        dest="record_recording",
        action="store_false",
        help="Disable HD RecordingCamera output (evaluation.mp4 will not be written).",
    )
    p.set_defaults(record_recording=True)
    return p


def main(argv=None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    perturbation_cfg = _load_json_or_none(args.perturbation_config)
    probe_cfg = _load_json_or_none(args.probe_config)

    config = core.EvalConfig(
        checkpoint_dir=args.checkpoint_dir,
        scene_usd=args.scene_usd,
        output_dir=args.output_dir,
        num_steps=args.num_steps,
        seed=args.seed,
        language_prompt=args.language_prompt,
        perturbation_config=perturbation_cfg,
        probe_config=probe_cfg,
        record_external_camera=bool(args.record_external),
        record_recording_camera=bool(args.record_recording),
    )

    sim = core.EvalSim(config)
    exit_code = 0
    try:
        sim.setup()
        result = sim.run()
        metrics_path = Path(config.output_dir) / "metrics.json"
        metrics.write_metrics_json(result, metrics_path)
        print(f"[exit] Wrote metrics to {metrics_path}")
        print(
            f"[exit] success={result.success} "
            f"progress={result.progress_fraction:.3f} "
            f"smoothness={result.trajectory_smoothness:.6f} "
            f"runtime={result.runtime_seconds:.1f}s "
            f"steps={result.num_steps_completed}"
        )
        if result.num_steps_completed < config.num_steps:
            exit_code = 2  # crashed mid-rollout
    except Exception as e:
        print(f"[FATAL] runner crashed: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        sim.close()
        print("[exit] Done.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
