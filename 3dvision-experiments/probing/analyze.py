"""Plotting + roll-up helpers for the probing outputs.

These are intentionally light wrappers around matplotlib / pandas so the
team can build the W2 plot grid quickly. None of this is on the hot
path of the eval loop.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------
# Heatmap: block_index x checkpoint, value = delta_norm
# --------------------------------------------------------------------
def plot_delta_norm_heatmap(
    df,
    out_path: str,
    *,
    expert: str = "paligemma",
    submodule: str = "attn",
    cmap: str = "viridis",
) -> None:
    """Plot a heatmap of Δ-norms.

    The input ``df`` is the wide-format DataFrame from
    ``delta_norm.delta_norm_table()``: one row per
    (expert, block_index, submodule, parameter, adapter), with one
    column per finetune name.

    Heatmap layout: y-axis = block index, x-axis = checkpoint name,
    cells = Δ-norm (summed over the chosen submodule's parameters).

    Args:
        df: wide-format Δ-norm DataFrame.
        out_path: where to save the PNG.
        expert: ``"paligemma"`` or ``"action_expert"`` — which submodel.
        submodule: ``"attn"`` or ``"mlp"``.
        cmap: matplotlib colormap.
    """
    import matplotlib.pyplot as plt  # type: ignore

    if pd is None:
        raise ImportError("pandas required")

    meta_cols = {"expert", "block_index", "submodule", "parameter", "adapter"}
    value_cols = [c for c in df.columns if c not in meta_cols]
    if not value_cols:
        raise ValueError(
            "No checkpoint columns found in df — got columns: "
            f"{list(df.columns)}"
        )

    sub = df[(df["expert"] == expert) & (df["submodule"] == submodule)]
    if sub.empty:
        raise ValueError(
            f"No rows for expert={expert!r} submodule={submodule!r}"
        )

    # Aggregate across parameter+adapter for a single value per (block, ckpt).
    agg = (
        sub.groupby("block_index", dropna=False)[value_cols]
        .sum(min_count=1)
        .sort_index()
    )

    fig, ax = plt.subplots(
        figsize=(max(4.0, 0.6 * len(value_cols) + 2.0), 0.4 * len(agg) + 2.0)
    )
    data = agg.values
    im = ax.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(np.arange(len(value_cols)))
    ax.set_xticklabels(value_cols, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(agg)))
    ax.set_yticklabels([str(i) for i in agg.index])
    ax.set_xlabel("checkpoint")
    ax.set_ylabel("transformer block")
    ax.set_title(f"LoRA Δ-norm — {expert} / {submodule}")
    fig.colorbar(im, ax=ax, label="‖Δ‖_F")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    logger.info("Saved Δ-norm heatmap to %s", out_path)


# --------------------------------------------------------------------
# Attention entropy curve per layer
# --------------------------------------------------------------------
def plot_attention_entropy_per_layer(df, out_path: str) -> None:
    """Plot mean attention entropy vs. transformer block, one line per head.

    Args:
        df: output of ``ActivationRecorder.summary()`` — columns
            ``layer``, ``head``, ``mean_entropy``.
        out_path: where to save the PNG.
    """
    import matplotlib.pyplot as plt  # type: ignore

    if pd is None:
        raise ImportError("pandas required")

    if df.empty:
        logger.warning("Empty entropy DataFrame — nothing to plot")
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for head, g in df.groupby("head"):
        g = g.sort_values("layer")
        ax.plot(g["layer"], g["mean_entropy"], marker="o", label=f"head {head}", alpha=0.6)
    ax.set_xlabel("transformer block")
    ax.set_ylabel("mean attention entropy (nats)")
    ax.set_title("Attention entropy per layer")
    ax.legend(ncol=2, fontsize="x-small")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    logger.info("Saved attention-entropy plot to %s", out_path)


# --------------------------------------------------------------------
# Object-token vs hand-token mass ratio
# --------------------------------------------------------------------
def compare_attention_object_vs_hand_tokens(
    attention_weights: dict[int, np.ndarray],
    object_token_indices: Sequence[int],
    hand_token_indices: Sequence[int],
):
    """Compute per-layer attention mass on object vs hand image patches.

    Args:
        attention_weights: ``{layer_idx: probs}`` from
            ``ActivationRecorder.record()``. Probs are expected to have a
            key-axis as the last dimension; we sum over the
            ``object_token_indices`` slice and divide by the sum over the
            ``hand_token_indices`` slice.
        object_token_indices: positions in the key axis that correspond
            to object image patches.
        hand_token_indices: positions in the key axis that correspond to
            hand image patches.

    Returns: a DataFrame with columns ``layer``, ``object_mass``,
    ``hand_mass``, ``ratio`` (= object/hand).
    """
    if pd is None:
        raise ImportError("pandas required")

    rows = []
    obj_idx = np.asarray(list(object_token_indices), dtype=np.int64)
    hand_idx = np.asarray(list(hand_token_indices), dtype=np.int64)
    for layer, probs in attention_weights.items():
        p = np.asarray(probs, dtype=np.float32)
        if p.ndim < 2:
            logger.warning("Layer %s probs have ndim=%d, skipping", layer, p.ndim)
            continue
        # Sum across all leading axes EXCEPT the key axis (last).
        leading = tuple(range(p.ndim - 1))
        # Mass on object/hand tokens. Mean over leading dims so the
        # scale is per-query-token rather than total.
        obj_mass = float(p[..., obj_idx].sum(axis=-1).mean(axis=leading[:-1] if leading else ()))
        hand_mass = float(p[..., hand_idx].sum(axis=-1).mean(axis=leading[:-1] if leading else ()))
        ratio = obj_mass / max(hand_mass, 1e-12)
        rows.append(
            {
                "layer": int(layer),
                "object_mass": obj_mass,
                "hand_mass": hand_mass,
                "ratio": ratio,
            }
        )

    df = pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)
    return df


__all__ = [
    "plot_delta_norm_heatmap",
    "plot_attention_entropy_per_layer",
    "compare_attention_object_vs_hand_tokens",
]
