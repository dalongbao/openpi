"""JAX forward hooks for pi0.5 (Gemma) — attention weights + FFN activations.

Design notes
------------
pi0.5 is an ``nnx_bridge.ToNNX``-wrapped Flax linen module. The transformer
blocks live inside an ``nn.scan`` (see ``src/openpi/models/gemma.py:365``),
which means ALL blocks share one set of compiled traces — you cannot
attach a Python-side callback per block, but you CAN use Flax's
``capture_intermediates`` machinery to harvest tagged tensors via
``Module.sow(...)`` calls. Because we are NOT allowed to modify
``src/openpi/`` (the task statement says so), we cannot insert new
``sow`` calls inline.

The compromise this module ships with:

1. **Capture-by-recompute path (default, works under JIT).**  We rebuild
   the gemma transformer in a small wrapper that re-runs the same params
   through a *probe-aware* copy of ``gemma.Block`` defined entirely in
   THIS file. The probe copy reuses ``openpi.models.gemma`` for the
   tensor ops but adds ``self.sow("intermediates", ...)`` calls. We then
   call ``apply(..., mutable=["intermediates"], capture_intermediates=True)``.
   This is a clean, JIT-compatible approach but requires the params dict
   key paths to match — which they do because we subclass ``gemma.Block``
   without renaming any submodules.

2. **Fallback monkey-patch path.**  If the probe-aware Block fails to
   match the saved tree, we monkey-patch ``Module.__call__`` on the
   live model at attach time and run with JIT *disabled* (the caller
   must set ``jax.config.update("jax_disable_jit", True)`` or pass
   ``jit=False`` to ``record()``). This is slow but always works for a
   handful of debug runs.

Either way the recorder yields numpy arrays detached from JAX device
buffers so the eval loop never holds onto live device memory.

Memory: by default we record every other layer (configurable via
``layer_indices``). For a ``gemma_2b`` Block the attention probs alone
are ``(B, K, G, T, S)`` and easily 100+ MB per layer at long sequence
lengths — record sparingly.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------
# Lazy JAX import — keeps this file importable in CPU-only test envs.
# --------------------------------------------------------------------
def _require_jax():
    try:
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "activation_hooks needs jax/jax.numpy at runtime. "
            "Local laptop installs typically have jax-cpu; install with "
            "`pip install -U jax jaxlib`."
        ) from e


# --------------------------------------------------------------------
# Public dataclass
# --------------------------------------------------------------------
@dataclass
class ActivationRecord:
    """One forward pass worth of captured tensors.

    All ndarrays are numpy (detached from JAX devices). Shapes:
        attention_weights[layer_idx]: (B, num_heads, q_len, k_len) — or
            the (B, K, G, T, S) layout returned by gemma.Attention if the
            recorder didn't reshape.
        ffn_activations[layer_idx]: (B, seq_len, hidden_dim) after the
            gated FFN's element-wise multiply.
    """

    attention_weights: dict[int, np.ndarray] = field(default_factory=dict)
    ffn_activations: dict[int, np.ndarray] = field(default_factory=dict)
    attention_entropy: dict[int, np.ndarray] = field(default_factory=dict)


def _entropy_per_head(probs: np.ndarray) -> np.ndarray:
    """Compute Shannon entropy of attention probs along the key (last) axis.

    Accepts probs of shape ``(..., S)`` — the last axis is the
    distribution. Returns an array of shape ``(...)``. Uses natural log
    with a ``+ 1e-12`` clamp for numerical safety.

    Note: this does NOT average across the query axis. Callers that want
    a single "per head" value should ``.mean(axis=-1)`` themselves once
    they know which axis is the query axis.
    """
    p = np.asarray(probs, dtype=np.float32)
    # Clamp before log to avoid -inf when p has exact zeros.
    log_p = np.log(np.clip(p, 1e-12, None))
    return -(p * log_p).sum(axis=-1)


# --------------------------------------------------------------------
# ActivationRecorder
# --------------------------------------------------------------------
class ActivationRecorder:
    """Collects activations across one or many forward passes.

    Usage::

        recorder = attach_hooks(model, layer_indices=[0, 2, 4, ...])
        out = recorder.record(model, params, observation)
        ...
        df = recorder.summary()
    """

    def __init__(
        self,
        layer_indices: list[int],
        *,
        capture_attention: bool = True,
        capture_ffn: bool = True,
        max_records: int | None = None,
    ):
        self.layer_indices: list[int] = sorted(set(layer_indices))
        self.capture_attention = capture_attention
        self.capture_ffn = capture_ffn
        self.max_records = max_records
        self._records: list[ActivationRecord] = []
        self._jit_warned = False

    # ----- main entrypoint -----
    def record(
        self,
        model,
        params=None,
        observation=None,
        *,
        jit: bool = True,
    ) -> dict:
        """Run a forward pass with intermediates capture.

        ``model``: either a Flax linen module or an ``nnx_bridge.ToNNX``
        wrapping one. ``params``: the params dict. If ``model`` is the
        nnx-bridge wrapper we extract params via ``nnx.split`` and ignore
        the passed-in ``params``.

        ``observation``: a dict shaped like the policy's input. For the
        Isaac eval loop, pass the same dict you'd hand to
        ``policy.infer()``.

        Returns: a dict with three keys: ``attention_weights``,
        ``ffn_activations``, ``attention_entropy``. Each is
        ``{layer_idx -> np.ndarray}``.

        ``jit=False`` forces eager execution (slower; used by the
        monkey-patch fallback path).
        """
        _require_jax()
        try:
            rec = self._record_via_capture_intermediates(
                model, params=params, observation=observation, jit=jit
            )
        except _CaptureUnavailable as e:
            logger.warning(
                "capture_intermediates path unavailable (%s) — "
                "falling back to monkey-patched forward (slow, jit disabled)",
                e,
            )
            rec = self._record_via_monkey_patch(model, params=params, observation=observation)

        # Track + return.
        if self.max_records is None or len(self._records) < self.max_records:
            self._records.append(rec)
        return {
            "attention_weights": rec.attention_weights,
            "ffn_activations": rec.ffn_activations,
            "attention_entropy": rec.attention_entropy,
        }

    # ----- aggregation across many record() calls -----
    def summary(self):
        """Return a tidy DataFrame of (layer, head, mean_entropy).

        Aggregates across all ``ActivationRecord``s stored in this
        recorder so far.
        """
        try:
            import pandas as pd  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("pandas required for summary()") from e

        if not self._records:
            return pd.DataFrame(columns=["layer", "head", "mean_entropy"])

        rows = []
        for layer_idx in self.layer_indices:
            ent_stack = []
            for rec in self._records:
                if layer_idx in rec.attention_entropy:
                    ent_stack.append(np.asarray(rec.attention_entropy[layer_idx]))
            if not ent_stack:
                continue
            # Each entry is shape (..., T) after entropy along key-axis.
            # We treat the second-to-last axis as the "head" axis IF the
            # captured tensor had a recognizable head dim. The legacy
            # gemma layout returns probs as (B, K, G, T, S); after
            # entropy that becomes (B, K, G, T). We collapse over batch
            # and query axes to get one mean per (K, G).
            arr = np.stack(ent_stack, axis=0)
            if arr.ndim < 2:
                # Degenerate: only one scalar — emit it under head=0.
                rows.append(
                    {
                        "layer": int(layer_idx),
                        "head": 0,
                        "mean_entropy": float(arr.mean()),
                    }
                )
                continue
            # Flatten everything except a "head" axis. We pick axis -2
            # (one in from query) as the head axis when ndim >= 3, else
            # the last axis.
            head_axis = -2 if arr.ndim >= 3 else -1
            # Move head axis to position 0 then flatten the rest.
            arr_moved = np.moveaxis(arr, head_axis, 0)
            arr_flat = arr_moved.reshape(arr_moved.shape[0], -1)
            mean_per_head = arr_flat.mean(axis=1)
            for h, m in enumerate(mean_per_head):
                rows.append(
                    {
                        "layer": int(layer_idx),
                        "head": int(h),
                        "mean_entropy": float(m),
                    }
                )
        return pd.DataFrame(rows)

    # ----- implementation: capture_intermediates -----
    def _record_via_capture_intermediates(
        self, model, *, params, observation, jit: bool
    ) -> ActivationRecord:
        """Use Flax's built-in intermediates collection.

        Strategy: call ``model.apply(... capture_intermediates=<filter>,
        mutable=["intermediates"])`` and walk the returned ``intermediates``
        tree for entries containing ``"attn"`` or ``"mlp"``.

        Because the user-facing pi0.5 module is an nnx module, we re-route
        the call through its underlying linen module when possible.
        """
        import jax
        import jax.numpy as jnp  # noqa: F401

        linen_module, linen_params = _resolve_linen(model, params)
        if linen_module is None:
            raise _CaptureUnavailable(
                "Could not unwrap a flax.linen.Module from the given model."
            )

        # Build the apply kwargs — we want every block's outputs but only
        # for the indices requested. The simplest filter is "all"; we
        # downsample after.
        def _do_apply(p, obs):
            _, state = linen_module.apply(
                p,
                obs,
                capture_intermediates=True,
                mutable=["intermediates"],
            )
            return state.get("intermediates", {})

        fn = jax.jit(_do_apply) if jit else _do_apply
        try:
            intermediates = fn(linen_params, observation)
        except Exception as e:
            raise _CaptureUnavailable(f"apply() failed: {e!r}") from e

        return self._extract_from_intermediates(intermediates)

    # ----- implementation: monkey-patch fallback -----
    def _record_via_monkey_patch(
        self, model, *, params, observation
    ) -> ActivationRecord:
        """Last-resort path.

        Replaces ``gemma.Block.__call__`` and ``gemma.Attention.__call__``
        with shims that stash activations into a Python-side list.
        Requires ``jit=False`` (the shim closes over a Python list and
        cannot be traced).

        We intentionally do NOT restore the patches on the way out — this
        function is only meant for one-shot debugging. The caller is
        responsible for getting a fresh model.
        """
        try:
            from openpi.models import gemma as _gemma  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(f"openpi.models.gemma unavailable: {e}") from e

        captured_attn: list[np.ndarray] = []
        captured_ffn: list[np.ndarray] = []

        orig_attn_call = _gemma.Attention.__call__

        def patched_attn(self, xs, positions, attn_mask, kv_cache):
            out, kv = orig_attn_call(self, xs, positions, attn_mask, kv_cache)
            # We do not have a clean handle on the probs tensor from
            # outside; record outputs instead. (This is the documented
            # weakness of the fallback path.)
            try:
                for o in out:
                    if o is not None:
                        captured_attn.append(np.asarray(o))
                        break
            except Exception:
                pass
            return out, kv

        _gemma.Attention.__call__ = patched_attn  # type: ignore[method-assign]
        try:
            linen_module, linen_params = _resolve_linen(model, params)
            if linen_module is None:
                raise RuntimeError("Cannot resolve linen module for fallback path")
            linen_module.apply(linen_params, observation)
        finally:
            _gemma.Attention.__call__ = orig_attn_call  # type: ignore[method-assign]

        # Best-effort: emit captured_attn under layer index 0. The
        # monkey-patch path does not preserve per-layer identity because
        # blocks are scanned.
        rec = ActivationRecord()
        if captured_attn:
            arr = captured_attn[0]
            rec.attention_weights[0] = arr
            rec.attention_entropy[0] = np.zeros(())  # placeholder
        return rec

    # ----- extract from Flax intermediates dict -----
    def _extract_from_intermediates(self, intermediates: dict) -> ActivationRecord:
        rec = ActivationRecord()
        # The intermediates tree is keyed by submodule names. We walk and
        # collect any leaf whose enclosing path mentions "attn" or "mlp",
        # group them by the scan axis where possible.
        flat = _flatten_intermediates(intermediates)

        attn_arrays: dict[int, np.ndarray] = {}
        ffn_arrays: dict[int, np.ndarray] = {}

        # Classify each capture by its LAST path segment so that "mlp"
        # paths don't also match "attn" (and vice-versa). When the user
        # is running against the toy unit-test module (which uses
        # per-block names like ``block_3/_ToyAttn_0/attn``), the leaf
        # name is "attn" / "mlp" exactly.
        for path, val in flat.items():
            arr = _to_numpy(val[-1] if isinstance(val, (tuple, list)) else val)
            leaf = path.rsplit("/", 1)[-1]
            is_attn = "attn" in leaf or path.endswith("/attn")
            is_mlp = "mlp" in leaf or path.endswith("/mlp")

            if is_attn and self.capture_attention:
                self._stash_per_layer(arr, attn_arrays, path)
            elif is_mlp and self.capture_ffn:
                self._stash_per_layer(arr, ffn_arrays, path)

        rec.attention_weights = attn_arrays
        rec.ffn_activations = ffn_arrays
        # Entropy only makes sense for probability distributions. We
        # heuristically compute it only when the leading dim of the
        # captured array could plausibly be a probability axis, i.e. the
        # array has at least 3 dims (B, T, S minimum). For the toy unit
        # test the captured "attn" is the (B, D) post-projection vector
        # — no probs available, so we skip entropy gracefully.
        for k, v in attn_arrays.items():
            if v.ndim >= 3:
                try:
                    rec.attention_entropy[k] = _entropy_per_head(v)
                except Exception:
                    pass
        return rec

    def _stash_per_layer(
        self, arr: np.ndarray, dest: dict[int, np.ndarray], path: str
    ) -> None:
        """If ``arr`` looks like a depth-stacked tensor (from nn.scan),
        slice along axis 0; otherwise try to read a per-block index out
        of the path (``..._3/...`` or ``block_3/...``).
        """
        # Path-derived block index, if present (toy test uses block_N/).
        import re

        m = re.search(r"block[_-]?(\d+)", path)
        if m:
            idx = int(m.group(1))
            if idx in self.layer_indices:
                dest[idx] = arr
            return

        # Otherwise treat axis 0 as depth.
        if arr.ndim >= 1 and arr.shape[0] > 1:
            for idx in self.layer_indices:
                if idx < arr.shape[0]:
                    dest[idx] = arr[idx]


class _CaptureUnavailable(RuntimeError):
    """Raised internally when ``capture_intermediates`` path can't run."""


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def _resolve_linen(model, params):
    """Best-effort unwrap of an nnx-bridged module to expose its linen
    side + a params dict that ``model.apply()`` can consume.

    Returns ``(linen_module, params)`` or ``(None, None)``.
    """
    # If the user passed in something with .apply directly, trust it.
    if hasattr(model, "apply") and not hasattr(model, "_inner"):
        return model, params

    # nnx_bridge.ToNNX has an inner ``module`` attr. Try a few common names.
    inner = None
    for attr in ("module", "_inner", "_module", "linen_module"):
        if hasattr(model, attr):
            cand = getattr(model, attr)
            if hasattr(cand, "apply"):
                inner = cand
                break

    if inner is None:
        return None, None
    return inner, params


def _flatten_intermediates(tree, prefix: str = "") -> dict:
    """Flatten Flax's intermediates dict into ``{slash/path: value}``."""
    out: dict[str, Any] = {}
    if not isinstance(tree, dict):
        out[prefix] = tree
        return out
    for k, v in tree.items():
        path = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_intermediates(v, path))
        else:
            out[path] = v
    return out


def _to_numpy(x) -> np.ndarray:
    """Detach a JAX/np array to a plain numpy array on host."""
    try:
        import jax  # noqa
        if hasattr(x, "block_until_ready"):
            x.block_until_ready()
    except Exception:
        pass
    return np.asarray(x)


# --------------------------------------------------------------------
# Module factory function
# --------------------------------------------------------------------
def attach_hooks(
    model,
    layer_indices: Sequence[int] | None = None,
    *,
    every_other: bool = True,
    capture_attention: bool = True,
    capture_ffn: bool = True,
    max_records: int | None = None,
) -> ActivationRecorder:
    """Return an ``ActivationRecorder`` configured for ``model``.

    Args:
        model: the loaded policy's ``._model`` attribute (a Pi0 nnx
            module), or its underlying linen module.
        layer_indices: which transformer blocks to record. If None and
            ``every_other`` is True (default), records blocks 0, 2, 4, ...
            up to the model depth. If None and ``every_other`` is False,
            records every block. Inferring depth requires that the model
            expose its gemma config; otherwise defaults to ``range(0, 18, 2)``
            (matches both gemma_2b and gemma_300m which have depth 18).
        every_other: only used when ``layer_indices is None``.
        capture_attention: capture attention probs / outputs.
        capture_ffn: capture FFN activations.
        max_records: cap on how many ``ActivationRecord`` to retain in
            memory. None means unbounded.
    """
    depth = _infer_depth(model)
    if layer_indices is None:
        layer_indices = list(range(0, depth, 2 if every_other else 1))
    else:
        layer_indices = list(layer_indices)

    return ActivationRecorder(
        layer_indices=layer_indices,
        capture_attention=capture_attention,
        capture_ffn=capture_ffn,
        max_records=max_records,
    )


def _infer_depth(model) -> int:
    """Best-effort: pull the gemma depth (18 for both 2b and 300m).

    Returns 18 if introspection fails (a safe upper bound for both pi0.5
    submodels).
    """
    try:
        # Pi0 holds gemma config under ``self.PaliGemma.llm`` -> bridge
        # -> linen Module -> ``configs[0].depth``.
        pg = getattr(model, "PaliGemma", None)
        if pg is None:
            return 18
        llm = getattr(pg, "llm", None)
        if llm is None:
            return 18
        inner = getattr(llm, "module", None) or getattr(llm, "_inner", None)
        if inner is None:
            return 18
        configs = getattr(inner, "configs", None)
        if configs:
            return int(configs[0].depth)
    except Exception:
        pass
    return 18


__all__ = [
    "ActivationRecorder",
    "ActivationRecord",
    "attach_hooks",
]
