"""Probes module — wires `probing.activation_hooks` into the eval rollout.

The legacy version of this file was a no-op stub. We now delegate to
``3dvision-experiments/probing`` so the eval loop and the offline
analysis share one source of truth.

Interface preserved for the runner (see ``core.py:212-257``):

    attach(model) -> handle (ActivationRecorder or None)
        Called once at policy load time. If probing is disabled or the
        recorder fails to construct, returns None and the rollout proceeds
        normally.

    record(recorder, model, params, observation, step_idx) -> dict | None
        Called every time we issue a new action chunk. Pulls activations
        for the current observation. Returning None means "do not log".

We import probing lazily — the entire ``activation_hooks`` module pulls
in JAX, which is fine inside the Isaac container but slow to import.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)


# Make the probing package importable from inside Isaac Sim:
#   /workspace/openpi/3dvision-experiments/probing/...
def _ensure_probing_on_path() -> None:
    for candidate in (
        "/workspace/openpi/3dvision-experiments",
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ),  # ".../3dvision-experiments" when running locally
    ):
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)


# --------------------------------------------------------------------
# attach
# --------------------------------------------------------------------
def attach(model, *, layer_indices=None, every_other: bool = True) -> Any:
    """Build an ``ActivationRecorder`` for the loaded pi0.5 model.

    ``model`` is the openpi ``Policy`` object — the underlying nnx Pi0
    module lives at ``policy._model``. We accept either form.
    """
    _ensure_probing_on_path()
    try:
        from probing.activation_hooks import attach_hooks  # type: ignore
    except Exception as e:
        logger.warning("probing.activation_hooks unavailable: %s — probes disabled", e)
        return None

    inner = getattr(model, "_model", model)
    try:
        recorder = attach_hooks(
            inner,
            layer_indices=layer_indices,
            every_other=every_other,
        )
        logger.info(
            "Attached probe on %d transformer layers: %s",
            len(recorder.layer_indices),
            recorder.layer_indices,
        )
        return recorder
    except Exception as e:
        logger.exception("Failed to attach probe: %s — continuing without probes", e)
        return None


# --------------------------------------------------------------------
# record
# --------------------------------------------------------------------
def record(recorder, model=None, params=None, observation=None, step_idx=0):
    """Record one forward-pass worth of activations.

    Backward-compatible signature: ``core.py`` currently calls
    ``probes.record(policy, observation, action_chunk, step)`` — i.e.
    the first positional is the policy/model and there is no separate
    recorder argument. To keep that working we accept BOTH shapes:

    * ``record(recorder, model=..., params=..., observation=..., step_idx=...)``
    * ``record(model, observation, action, step_idx)`` — legacy

    The legacy shape will succeed only if the caller previously stashed
    a recorder on the model (e.g. ``model._probe_recorder = recorder``).
    """
    # Legacy 4-positional form: record(model, obs, action, step).
    legacy_form = (
        recorder is not None
        and not _looks_like_recorder(recorder)
        and model is not None
        and params is not None
    )
    if legacy_form:
        true_model = recorder
        true_obs = model
        # action, step_idx are ignored by the recorder
        attached = getattr(true_model, "_probe_recorder", None)
        if attached is None:
            return None
        return attached.record(true_model, params=None, observation=true_obs, jit=True)

    if recorder is None:
        return None
    try:
        return recorder.record(
            model, params=params, observation=observation, jit=True
        )
    except Exception as e:
        logger.warning("probe record() failed at step %d: %s", step_idx, e)
        return None


def _looks_like_recorder(obj) -> bool:
    """Heuristic: is the first positional arg an ActivationRecorder?"""
    return hasattr(obj, "record") and hasattr(obj, "summary") and hasattr(obj, "layer_indices")


__all__ = ["attach", "record"]
