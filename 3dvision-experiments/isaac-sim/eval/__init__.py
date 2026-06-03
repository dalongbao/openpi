"""Modular pi0.5 + Isaac Sim evaluation package.

Public API:
    EvalConfig, EvalResult, EvalSim   (from .core)
    compute_success_heuristic, compute_progress_fraction,
    compute_trajectory_smoothness, write_metrics_json   (from .metrics)
    main   (from .runner)

The ``probes`` and ``perturbations`` submodules are intentionally NOT
re-exported here — they're imported lazily inside ``EvalSim`` so a
default-flags parity run pulls zero extra code.
"""

from .core import EvalConfig, EvalResult, EvalSim
from .metrics import (
    compute_progress_fraction,
    compute_success_heuristic,
    compute_trajectory_smoothness,
    write_metrics_json,
)
from .runner import main

__all__ = [
    "EvalConfig",
    "EvalResult",
    "EvalSim",
    "compute_progress_fraction",
    "compute_success_heuristic",
    "compute_trajectory_smoothness",
    "write_metrics_json",
    "main",
]
