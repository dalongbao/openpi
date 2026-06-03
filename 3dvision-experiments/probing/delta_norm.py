"""Per-layer LoRA delta-norm computation across pi0.5 checkpoints.

Pure-function module — no Isaac Sim dependency. Imports openpi only to reuse
the same orbax checkpoint loader the training pipeline uses (so we read the
same on-disk format).

The pi0.5 ``Pi0`` module is a Flax linen+nnx hybrid (``nnx_bridge.ToNNX``)
wrapping a Gemma transformer that holds both the PaliGemma stem and the
action expert in a single ``flax.linen.scan``-ed ``Block`` (see
``src/openpi/models/gemma.py:359-381``). Both experts share parameter
arrays along axis 0 — index 0 is PaliGemma, index 1 is the action expert
(named ``*_1`` in the param tree, see ``_name`` at
``src/openpi/models/gemma.py:443``).

LoRA naming convention (from ``src/openpi/models/lora.py``):
  * Attention einsums use ``lora_a`` / ``lora_b`` (lines 51-52).
  * FeedForward uses ``gating_einsum_lora_a``, ``gating_einsum_lora_b``,
    ``linear_lora_a``, ``linear_lora_b`` (lines 113-121).

Because ``nn.scan`` stacks the per-block weights along axis 0, a LoRA
parameter at path ``...layers/attn/qkv_einsum/lora_a`` has shape
``(depth, ...)`` — one slice per transformer block. We decompose along
that axis when reporting per-block norms.

Memory note: we load the param trees with ``restore_type=np.ndarray`` so
nothing lands on a GPU and JAX device buffers are not created. Trees are
flattened once and compared leaf-by-leaf; we never hold two full trees of
*activations* — only the params themselves, which is unavoidable for a
diff.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Any

import numpy as np

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover - pandas is optional at import time
    pd = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

PyTree = Any  # the orbax restored params are nested dicts of ndarrays.

# Regex matching any LoRA parameter leaf. Both the "raw" lora_a/lora_b
# (attention einsums) and the prefixed forms (gating_einsum_lora_a,
# linear_lora_b, ...) match this pattern.
_LORA_LEAF_RE = re.compile(r"lora_[ab]$")

# Per submodule we expose a friendlier name. Walked against the path string.
_SUBMODULE_PATTERNS: list[tuple[str, str]] = [
    ("qkv_einsum", "attn.qkv"),
    ("q_einsum", "attn.q"),
    ("kv_einsum", "attn.kv"),
    ("attn_vec_einsum", "attn.out"),
    ("mlp/gating_einsum", "mlp.gate"),
    ("mlp/linear", "mlp.down"),
    # Fallback patterns when we don't see the einsum sub-name.
    ("attn", "attn"),
    ("mlp", "mlp"),
]


# --------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------
def load_checkpoint_params(checkpoint_dir: str) -> PyTree:
    """Load orbax checkpoint params for pi05_egoverse and return the params PyTree.

    The directory layout produced by openpi training (see
    ``src/openpi/training/checkpoints.py:84``) is::

        <step>/params/...

    ``checkpoint_dir`` must point at the dir that *contains* the orbax
    ``params`` subtree (typically ``.../pi05_egoverse/<exp>/<step>/params``,
    or pass the step dir and we'll append ``/params`` for you).

    Returns the params dict — i.e. ``{"PaliGemma": {...}, ...}`` — as plain
    numpy arrays so this can run on a CPU box without JAX devices.
    """
    import os

    # Import lazily so this module is importable without JAX/orbax on the
    # local laptop. We only need them when actually loading a checkpoint.
    from openpi.models import model as _model  # type: ignore[import-not-found]

    # Helpful nudge: openpi's convention is that ``params`` is a sub-dir.
    candidate = checkpoint_dir
    if not os.path.basename(candidate.rstrip("/")) == "params":
        joined = os.path.join(candidate, "params")
        if os.path.isdir(joined):
            candidate = joined

    logger.info("Loading checkpoint params from %s", candidate)
    params = _model.restore_params(candidate, restore_type=np.ndarray)
    return params


# --------------------------------------------------------------------
# Path flattening + LoRA detection
# --------------------------------------------------------------------
def _flatten(tree: Mapping, prefix: str = "") -> dict[str, np.ndarray]:
    """Flatten a nested dict of arrays into a flat ``{slash/path: array}``.

    Uses ``/`` as separator to match openpi's own ``flax.traverse_util``
    convention (see ``src/openpi/training/weight_loaders.py:87``).
    """
    out: dict[str, np.ndarray] = {}
    for k, v in tree.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, Mapping):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _is_lora_path(path: str) -> bool:
    """True if the leaf name (last segment) is a LoRA parameter."""
    leaf = path.rsplit("/", 1)[-1]
    return bool(_LORA_LEAF_RE.search(leaf)) or "adapter" in leaf.lower()


# --------------------------------------------------------------------
# Per-leaf delta norms
# --------------------------------------------------------------------
def compute_lora_delta_norms(
    base_params: PyTree,
    finetuned_params: PyTree,
    *,
    expand_scan_axis: bool = True,
) -> dict[str, float]:
    """Walk the params tree and return Frobenius norms of (finetuned - base)
    for every LoRA leaf.

    Args:
        base_params: pre-finetune params (orbax-loaded).
        finetuned_params: post-finetune params, same tree structure.
        expand_scan_axis: pi0.5's gemma layers are stacked along axis 0
            via ``nn.scan`` (gemma.py:365). When True (default), we slice
            each LoRA tensor along that axis and emit one
            ``<path>[block=i]`` entry per block. When False we return the
            full-tensor norm.

    Returns: flat ``{path: norm}`` dict. Norms are plain Python floats so
    the result is trivially serializable. We compute in float32 even if the
    weights were stored as bfloat16 — Frobenius norms in low precision lose
    a lot of dynamic range.
    """
    base_flat = _flatten(base_params)
    fine_flat = _flatten(finetuned_params)

    results: dict[str, float] = {}
    missing_in_finetune: list[str] = []

    for path, base_arr in base_flat.items():
        if not _is_lora_path(path):
            continue
        if path not in fine_flat:
            missing_in_finetune.append(path)
            continue

        fine_arr = fine_flat[path]
        if base_arr.shape != fine_arr.shape:
            logger.warning(
                "Shape mismatch on %s: base=%s finetuned=%s — skipping",
                path,
                base_arr.shape,
                fine_arr.shape,
            )
            continue

        # Promote to float32 to be safe.
        diff = np.asarray(fine_arr, dtype=np.float32) - np.asarray(
            base_arr, dtype=np.float32
        )

        if expand_scan_axis and diff.ndim >= 1 and diff.shape[0] > 1:
            # Each slice along axis 0 corresponds to one transformer block.
            for i in range(diff.shape[0]):
                norm = float(np.linalg.norm(diff[i].reshape(-1)))
                results[f"{path}[block={i}]"] = norm
        else:
            norm = float(np.linalg.norm(diff.reshape(-1)))
            results[path] = norm

    if missing_in_finetune:
        logger.warning(
            "%d LoRA paths present in base but missing in finetuned (first 3: %s)",
            len(missing_in_finetune),
            missing_in_finetune[:3],
        )

    return results


# --------------------------------------------------------------------
# Path -> structured fields
# --------------------------------------------------------------------
_BLOCK_RE = re.compile(r"\[block=(\d+)\]")

# Patterns for individual LoRA parameter roles (q/k/v/o, gate/down/up).
# Ordered by specificity.
_PARAM_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"qkv_einsum"), "qkv"),
    (re.compile(r"q_einsum"), "q"),
    (re.compile(r"kv_einsum"), "kv"),
    (re.compile(r"attn_vec_einsum"), "o"),
    (re.compile(r"gating_einsum"), "gate"),
    (re.compile(r"mlp/linear"), "down"),
]


def _classify_path(path: str) -> dict[str, Any]:
    """Extract (block_index, submodule, parameter, expert) from a LoRA path."""
    # Block index from our [block=i] suffix.
    m = _BLOCK_RE.search(path)
    block_idx: int | None = int(m.group(1)) if m else None

    # Submodule (attn vs mlp).
    if "mlp" in path:
        submodule = "mlp"
    elif "attn" in path:
        submodule = "attn"
    else:
        submodule = "unknown"

    # Parameter role.
    parameter = "unknown"
    for pat, name in _PARAM_PATTERNS:
        if pat.search(path):
            parameter = name
            break

    # Expert: PaliGemma vs action expert.
    # Gemma's ``_name`` (gemma.py:443) appends ``_1`` for the second expert.
    expert = "action_expert" if re.search(r"_1(?:/|\[|$)", path) else "paligemma"

    # Adapter A or B.
    adapter = "A" if path.rsplit("/", 1)[-1].split("[")[0].endswith("lora_a") else "B"

    return {
        "block_index": block_idx,
        "submodule": submodule,
        "parameter": parameter,
        "expert": expert,
        "adapter": adapter,
    }


# --------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------
def aggregate_by_layer(per_param_norms: dict[str, float]):
    """Group per-parameter norms by transformer block index.

    Returns a pandas DataFrame with columns:
        block_index, submodule, parameter, expert, adapter, path, delta_norm.
    """
    if pd is None:
        raise ImportError("pandas is required for aggregate_by_layer(); pip install pandas")

    rows = []
    for path, norm in per_param_norms.items():
        meta = _classify_path(path)
        rows.append({**meta, "path": path, "delta_norm": float(norm)})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Stable ordering: by expert, block, submodule, parameter.
    df = df.sort_values(
        ["expert", "block_index", "submodule", "parameter", "adapter", "path"],
        na_position="last",
    ).reset_index(drop=True)
    return df


# --------------------------------------------------------------------
# Top-level orchestration
# --------------------------------------------------------------------
def delta_norm_table(
    checkpoint_dirs: dict[str, str],
    base_dir: str,
    *,
    combine_AB: bool = True,
):
    """Load base + each named checkpoint, compute LoRA Δ-norms, return a
    tidy wide-format DataFrame.

    Memory: we load base ONCE, then iterate over the finetunes one at a
    time, discarding each before moving on. So peak RAM is ``2 * sizeof(
    params)`` rather than ``(N+1) * sizeof(params)``.

    Args:
        checkpoint_dirs: ``{name: path}`` for each finetuned checkpoint.
        base_dir: path to the base (pre-finetune) checkpoint.
        combine_AB: if True (default), sum the norms of ``lora_a`` and
            ``lora_b`` for the same layer/submodule/parameter into one
            value (gives a single number per "logical" LoRA adapter). If
            False, keep A and B as separate rows.
    """
    if pd is None:
        raise ImportError("pandas is required for delta_norm_table(); pip install pandas")

    logger.info("Loading base checkpoint from %s", base_dir)
    base = load_checkpoint_params(base_dir)

    merged: dict[tuple[Any, ...], dict[str, float]] = {}
    columns_seen: list[str] = []

    for name, path in checkpoint_dirs.items():
        logger.info("Diffing finetuned checkpoint '%s' from %s", name, path)
        ft = load_checkpoint_params(path)
        norms = compute_lora_delta_norms(base, ft)
        df = aggregate_by_layer(norms)
        del ft  # free immediately

        if combine_AB and not df.empty:
            df = (
                df.assign(_dn_sq=df["delta_norm"] ** 2)
                .groupby(
                    ["expert", "block_index", "submodule", "parameter"],
                    dropna=False,
                    as_index=False,
                )["_dn_sq"]
                .sum()
                .assign(delta_norm=lambda d: np.sqrt(d["_dn_sq"]))
                .drop(columns=["_dn_sq"])
            )
            df["adapter"] = "AB"

        columns_seen.append(name)
        for _, row in df.iterrows():
            key = (
                row["expert"],
                row["block_index"],
                row["submodule"],
                row["parameter"],
                row.get("adapter", "AB"),
            )
            merged.setdefault(key, {})[name] = float(row["delta_norm"])

    # Build wide DF.
    out_rows = []
    for (expert, block_idx, submod, param, adapter), col_vals in merged.items():
        row = {
            "expert": expert,
            "block_index": block_idx,
            "submodule": submod,
            "parameter": param,
            "adapter": adapter,
        }
        for cname in columns_seen:
            row[cname] = col_vals.get(cname, np.nan)
        out_rows.append(row)

    wide = pd.DataFrame(out_rows)
    if not wide.empty:
        wide = wide.sort_values(
            ["expert", "block_index", "submodule", "parameter"]
        ).reset_index(drop=True)
    return wide


__all__ = [
    "load_checkpoint_params",
    "compute_lora_delta_norms",
    "aggregate_by_layer",
    "delta_norm_table",
]
