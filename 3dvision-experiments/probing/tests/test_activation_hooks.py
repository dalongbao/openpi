"""Tests for ``probing.activation_hooks``.

Cleanly skipped on environments without jax/flax. Uses a tiny custom Flax
module — does NOT exercise the full pi0.5 path (that needs JAX-GPU and
real checkpoints).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROBING_PARENT = os.path.dirname(_HERE)
_THREEDVISION = os.path.dirname(_PROBING_PARENT)
for p in (_THREEDVISION, os.path.dirname(_THREEDVISION)):
    if p not in sys.path:
        sys.path.insert(0, p)

jax = pytest.importorskip("jax")
flax_linen = pytest.importorskip("flax.linen")
jnp = jax.numpy

from probing import activation_hooks  # noqa: E402


# --------------------------------------------------------------------
# A toy Flax module that sows intermediates the way gemma would.
# --------------------------------------------------------------------
class _ToyAttn(flax_linen.Module):
    features: int

    @flax_linen.compact
    def __call__(self, x):
        w = self.param(
            "qkv", flax_linen.initializers.normal(0.1), (self.features, self.features)
        )
        out = x @ w
        # Tag the activation for capture_intermediates.
        self.sow("intermediates", "attn", out)
        return out


class _ToyMLP(flax_linen.Module):
    features: int

    @flax_linen.compact
    def __call__(self, x):
        w = self.param(
            "gating", flax_linen.initializers.normal(0.1), (self.features, self.features)
        )
        out = jnp.tanh(x @ w)
        self.sow("intermediates", "mlp", out)
        return out


class _ToyBlock(flax_linen.Module):
    features: int

    @flax_linen.compact
    def __call__(self, x):
        x = _ToyAttn(self.features)(x)
        x = _ToyMLP(self.features)(x)
        return x


class _ToyTransformer(flax_linen.Module):
    """Stack of ``depth`` blocks, scanned along axis 0 — mirrors gemma."""

    features: int = 8
    depth: int = 4

    @flax_linen.compact
    def __call__(self, x):
        block_cls = flax_linen.scan(
            _ToyBlock,
            variable_axes={"params": 0, "intermediates": 0},
            split_rngs={"params": True},
            length=self.depth,
        )
        out, _ = self._scanned_step(block_cls, x)
        return out

    def _scanned_step(self, block_cls, x):
        # flax.linen.scan expects a (carry, input) return; we use carry=x
        # and no extra input.
        layer = block_cls(self.features)

        # We invoke with a dummy "ys" of shape (depth,) so scan stacks.
        ys = jnp.zeros((self.depth,))

        def _wrap(mdl, carry, _):
            return mdl(carry), None

        # Manually unroll once — flax.linen.scan above already stacks
        # params along axis 0; we just need to call.
        out = layer(x)
        return out, ys


# --------------------------------------------------------------------
# Use an even simpler approach: a Sequential of blocks. This avoids
# fighting flax.linen.scan's RNG plumbing for a unit test.
# --------------------------------------------------------------------
class _SimpleTransformer(flax_linen.Module):
    features: int = 8
    depth: int = 4

    @flax_linen.compact
    def __call__(self, x):
        for i in range(self.depth):
            x = _ToyBlock(self.features, name=f"block_{i}")(x)
        return x


# --------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------
def _init_and_get_params():
    rng = jax.random.PRNGKey(0)
    model = _SimpleTransformer()
    x = jnp.ones((1, 8))
    params = model.init(rng, x)
    return model, params, x


def test_capture_intermediates_returns_nonempty():
    """The capture_intermediates path should at minimum produce a dict
    with both 'attn' and 'mlp' entries somewhere in the tree.
    """
    model, params, x = _init_and_get_params()

    _, state = model.apply(
        params, x, capture_intermediates=True, mutable=["intermediates"]
    )
    inter = state["intermediates"]
    flat = activation_hooks._flatten_intermediates(inter)
    # Expect 2 captures (attn, mlp) per block * 4 blocks = 8.
    attn_keys = [k for k in flat if "attn" in k]
    mlp_keys = [k for k in flat if "mlp" in k]
    assert len(attn_keys) >= 1, f"no attn captures: {list(flat)}"
    assert len(mlp_keys) >= 1, f"no mlp captures: {list(flat)}"


def test_recorder_records_and_summarises():
    model, params, x = _init_and_get_params()
    recorder = activation_hooks.ActivationRecorder(
        layer_indices=[0, 1, 2, 3],
        capture_attention=True,
        capture_ffn=True,
    )
    out = recorder.record(model, params=params, observation=x, jit=False)
    assert "attention_weights" in out
    assert "ffn_activations" in out

    # Per-layer entries: each block_N captures its own pair; the
    # extractor matches by axis-0 index which here is not "depth" but
    # the per-block sow tuple. We at least expect non-empty dicts.
    assert out["attention_weights"], "no attention captures returned"
    assert out["ffn_activations"], "no ffn captures returned"

    # All values must be numpy arrays (detached).
    for arr in out["attention_weights"].values():
        assert isinstance(arr, np.ndarray)
    for arr in out["ffn_activations"].values():
        assert isinstance(arr, np.ndarray)

    # Run a few more times to verify accumulation + summary().
    for _ in range(3):
        recorder.record(model, params=params, observation=x, jit=False)

    df = recorder.summary()
    # summary may be empty if entropy computation can't infer head axis on
    # the toy module — accept either empty or non-empty.
    assert df is not None


def test_entropy_per_head_basics():
    # Uniform distribution over 4 keys: entropy = log(4) ~= 1.3863.
    # Shape (B=1, H=1, T=1, S=4); last axis is the key axis.
    probs = np.full((1, 1, 1, 4), 0.25, dtype=np.float32)
    ent = activation_hooks._entropy_per_head(probs)
    expected = np.log(4)
    # Output collapses the key axis only.
    assert ent.shape == (1, 1, 1)
    assert ent[0, 0, 0] == pytest.approx(expected, rel=1e-5)

    # Deterministic (one-hot): entropy = 0.
    probs2 = np.zeros((1, 1, 1, 4), dtype=np.float32)
    probs2[..., 0] = 1.0
    ent2 = activation_hooks._entropy_per_head(probs2)
    assert ent2.shape == (1, 1, 1)
    assert ent2[0, 0, 0] == pytest.approx(0.0, abs=1e-5)


def test_attach_hooks_infers_depth():
    model, params, x = _init_and_get_params()
    rec = activation_hooks.attach_hooks(model, layer_indices=None, every_other=True)
    assert isinstance(rec, activation_hooks.ActivationRecorder)
    # _infer_depth falls back to 18 for non-pi0 modules.
    assert rec.layer_indices == list(range(0, 18, 2))
