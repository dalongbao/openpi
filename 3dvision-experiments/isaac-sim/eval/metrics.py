"""Eval metrics: success heuristics, progress, smoothness, JSON dumper.

The legacy script logged only raw joint positions in a CSV. The refactor
adds first-class metric computations so downstream sweeps can compare
checkpoints/seeds/perturbations programmatically.

All functions are pure NumPy + Python — no Isaac Sim deps, no openpi
deps. Safe to import on a laptop for offline analysis of dumped CSVs.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np


def compute_trajectory_smoothness(per_step_joint_positions: np.ndarray) -> float:
    """Mean per-joint velocity variance across the trajectory.

    Lower is smoother. For an (N, J) array of joint positions sampled at a
    fixed dt, computes finite-difference velocities then averages the
    per-joint variance across joints. Returns 0.0 for trajectories with
    fewer than 3 samples (no meaningful velocity signal).
    """
    arr = np.asarray(per_step_joint_positions, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 3:
        return 0.0
    velocities = np.diff(arr, axis=0)
    per_joint_var = np.var(velocities, axis=0)
    return float(np.mean(per_joint_var))


def compute_progress_fraction(
    per_step_joint_positions: np.ndarray,
    plate_initial_xy: Optional[Sequence[float]] = None,
    plate_final_xy: Optional[Sequence[float]] = None,
    crate_xy: Optional[Sequence[float]] = None,
) -> float:
    """Fraction of the plate->crate distance the plate actually covered.

    If plate positions aren't available (the legacy run only logs joint
    positions, not object poses), we fall back to a proxy: the magnitude
    of cumulative end-effector travel normalized into [0,1] by an
    arbitrary cap. This is a placeholder — real implementations should
    plug in plate poses sampled from the stage.

    Returns a value in [0, 1].
    """
    if plate_initial_xy is not None and plate_final_xy is not None and crate_xy is not None:
        p0 = np.asarray(plate_initial_xy, dtype=np.float64)
        pf = np.asarray(plate_final_xy, dtype=np.float64)
        c  = np.asarray(crate_xy, dtype=np.float64)
        total = np.linalg.norm(c - p0)
        if total < 1e-6:
            return 0.0
        traveled = np.linalg.norm(pf - p0)
        # Project travel onto the start->crate axis to avoid rewarding
        # motion away from the goal.
        axis = (c - p0) / total
        signed = float(np.dot(pf - p0, axis))
        return float(np.clip(signed / total, 0.0, 1.0))

    # Fallback proxy: normalized end-effector path length.
    arr = np.asarray(per_step_joint_positions, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return 0.0
    # Treat first 3 joints as a coarse proxy for end-effector base motion.
    j = arr[:, :3]
    path_len = float(np.sum(np.linalg.norm(np.diff(j, axis=0), axis=1)))
    # Cap at 5.0 (radians of cumulative shoulder rotation) as an arbitrary
    # normalizer. The real value should be plugged in once object poses
    # are tracked.
    return float(np.clip(path_len / 5.0, 0.0, 1.0))


def compute_success_heuristic(
    per_step_joint_positions: np.ndarray,
    plate_final_xy: Optional[Sequence[float]] = None,
    crate_xy: Optional[Sequence[float]] = None,
    threshold_cm: float = 15.0,
) -> Dict[str, Any]:
    """Heuristic task-success check.

    "Success" is defined as the plate's final XY position being within
    ``threshold_cm`` of the crate's XY position. When either pose is
    unavailable, ``success`` is False and ``details["reason"]`` records
    why.
    """
    details: Dict[str, Any] = {
        "threshold_cm": float(threshold_cm),
        "plate_final_xy": list(plate_final_xy) if plate_final_xy is not None else None,
        "crate_xy": list(crate_xy) if crate_xy is not None else None,
    }
    if plate_final_xy is None or crate_xy is None:
        details["reason"] = "object poses unavailable; success undetermined"
        details["distance_cm"] = None
        return {"success": False, "details": details}

    p = np.asarray(plate_final_xy, dtype=np.float64)
    c = np.asarray(crate_xy, dtype=np.float64)
    dist_m = float(np.linalg.norm(p - c))
    dist_cm = dist_m * 100.0
    details["distance_cm"] = dist_cm
    return {"success": bool(dist_cm <= threshold_cm), "details": details}


def write_metrics_json(result, path: Path) -> None:
    """Dump an EvalResult to a JSON file.

    Large arrays (per-step joint positions, observations, actions, probe
    outputs) are NOT embedded — they'd bloat the JSON and are already
    persisted elsewhere (CSV, .npz, MP4). Only scalar/dict summary fields
    are emitted.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if dataclasses.is_dataclass(result):
        d = dataclasses.asdict(result)
    else:
        d = dict(result)

    # Strip heavy fields.
    for k in (
        "per_step_joint_positions",
        "per_step_observations",
        "per_step_actions",
        "per_step_probe_outputs",
    ):
        d.pop(k, None)

    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, Path):
            return str(o)
        return str(o)

    with open(path, "w") as f:
        json.dump(d, f, indent=2, default=_default)
