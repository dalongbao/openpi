"""Mechanistic-interpretability utilities for pi0.5.

Two halves:

* :mod:`probing.delta_norm` — load orbax checkpoints and compute per-layer
  Frobenius norms of (finetuned - base) for every LoRA adapter. Tells
  you where each fine-tune deviates from the pre-trained pi05_base.
* :mod:`probing.activation_hooks` — JAX/Flax forward hooks that record
  attention weights and FFN activations during a rollout. Use during
  Isaac Sim eval (or any policy.infer-driven loop).

See ``probing/README.md`` for end-to-end usage.
"""

from .activation_hooks import ActivationRecorder, attach_hooks
from .delta_norm import (
    aggregate_by_layer,
    compute_lora_delta_norms,
    delta_norm_table,
    load_checkpoint_params,
)

__all__ = [
    # delta_norm
    "load_checkpoint_params",
    "compute_lora_delta_norms",
    "aggregate_by_layer",
    "delta_norm_table",
    # activation_hooks
    "ActivationRecorder",
    "attach_hooks",
]
