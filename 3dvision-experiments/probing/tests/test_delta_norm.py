"""Tests for ``probing.delta_norm`` — pure-numpy, no JAX, no GPU."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Make probing/ importable when running from repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROBING_PARENT = os.path.dirname(_HERE)  # ".../probing"
_THREEDVISION = os.path.dirname(_PROBING_PARENT)  # ".../3dvision-experiments"
for p in (_THREEDVISION, os.path.dirname(_THREEDVISION)):
    if p not in sys.path:
        sys.path.insert(0, p)

from probing import delta_norm  # noqa: E402


# --------------------------------------------------------------------
# Test 1: zero delta when finetuned == base.
# --------------------------------------------------------------------
def test_zero_when_identical():
    rng = np.random.default_rng(0)
    # Mini params tree mimicking gemma scan output: (depth=4, ...) leading.
    params = {
        "PaliGemma": {
            "llm": {
                "layers": {
                    "attn": {
                        "qkv_einsum": {
                            "w": rng.normal(size=(4, 3, 8, 64, 32)).astype(np.float32),
                            "lora_a": rng.normal(size=(4, 3, 8, 64, 16)).astype(np.float32),
                            "lora_b": rng.normal(size=(4, 3, 8, 16, 32)).astype(np.float32),
                        },
                    },
                    "mlp": {
                        "gating_einsum_lora_a": rng.normal(size=(4, 2, 64, 16)).astype(np.float32),
                        "gating_einsum_lora_b": rng.normal(size=(4, 2, 16, 128)).astype(np.float32),
                    },
                },
            },
        },
    }
    norms = delta_norm.compute_lora_delta_norms(params, params)
    assert len(norms) > 0, "expected at least one LoRA leaf"
    for k, v in norms.items():
        assert v == 0.0, f"{k}: expected 0, got {v}"


# --------------------------------------------------------------------
# Test 2: hand-built finite difference returns the correct Frobenius norm.
# --------------------------------------------------------------------
def test_known_value():
    base = {
        "PaliGemma": {
            "llm": {
                "layers": {
                    "attn": {
                        "qkv_einsum": {
                            "lora_a": np.zeros((2, 4), dtype=np.float32),
                            "lora_b": np.zeros((2, 4), dtype=np.float32),
                        },
                    },
                },
            },
        },
    }
    fine = {
        "PaliGemma": {
            "llm": {
                "layers": {
                    "attn": {
                        "qkv_einsum": {
                            # Block 0: ones (norm = sqrt(4) = 2).
                            # Block 1: 2 * ones (norm = sqrt(16) = 4).
                            "lora_a": np.array(
                                [[1, 1, 1, 1], [2, 2, 2, 2]], dtype=np.float32
                            ),
                            "lora_b": np.zeros((2, 4), dtype=np.float32),
                        },
                    },
                },
            },
        },
    }
    norms = delta_norm.compute_lora_delta_norms(base, fine)
    # Expect per-block expansion: two keys for lora_a, two for lora_b (=0).
    qkv_key_a_0 = next(
        k for k in norms if "lora_a" in k and "block=0" in k
    )
    qkv_key_a_1 = next(
        k for k in norms if "lora_a" in k and "block=1" in k
    )
    assert norms[qkv_key_a_0] == pytest.approx(2.0, rel=1e-5)
    assert norms[qkv_key_a_1] == pytest.approx(4.0, rel=1e-5)
    # All lora_b entries are zero.
    assert all(v == 0.0 for k, v in norms.items() if "lora_b" in k)


# --------------------------------------------------------------------
# Test 3: aggregate_by_layer parses block indices from typical Flax paths.
# --------------------------------------------------------------------
def test_aggregate_by_layer_parses_paths():
    pd = pytest.importorskip("pandas")
    norms = {
        "PaliGemma/llm/layers/attn/qkv_einsum/lora_a[block=0]": 1.0,
        "PaliGemma/llm/layers/attn/qkv_einsum/lora_b[block=0]": 2.0,
        "PaliGemma/llm/layers/attn/qkv_einsum/lora_a[block=5]": 3.0,
        "PaliGemma/llm/layers/mlp/gating_einsum_lora_a[block=10]": 4.0,
        # Action expert: same path with the _1 suffix.
        "PaliGemma/llm/layers/attn_1/qkv_einsum_1/lora_a[block=2]": 5.0,
    }
    df = delta_norm.aggregate_by_layer(norms)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(norms)

    # Block index extraction.
    block_0_rows = df[df["block_index"] == 0]
    assert len(block_0_rows) == 2

    # Submodule classification.
    assert set(df[df["submodule"] == "attn"]["block_index"].tolist()) == {0, 0, 2, 5}
    assert set(df[df["submodule"] == "mlp"]["block_index"].tolist()) == {10}

    # Parameter classification.
    assert "qkv" in df["parameter"].tolist()
    assert "gate" in df["parameter"].tolist()

    # Expert detection: the _1 suffix means action_expert.
    expert_counts = df["expert"].value_counts().to_dict()
    assert expert_counts.get("action_expert", 0) == 1
    assert expert_counts.get("paligemma", 0) == 4

    # Adapter A vs B.
    assert "A" in df["adapter"].tolist() and "B" in df["adapter"].tolist()


# --------------------------------------------------------------------
# Test 4: missing leaves on the finetuned side are tolerated (warn, don't crash).
# --------------------------------------------------------------------
def test_missing_paths_in_finetune():
    base = {
        "PaliGemma": {
            "x": {"lora_a": np.ones((2, 4), dtype=np.float32)},
            "y": {"lora_a": np.ones((2, 4), dtype=np.float32)},
        },
    }
    fine = {
        "PaliGemma": {
            "x": {"lora_a": np.ones((2, 4), dtype=np.float32)},
            # 'y' missing.
        },
    }
    norms = delta_norm.compute_lora_delta_norms(base, fine)
    # We get entries for 'x' (zeroed) and skip 'y'.
    assert all("/x/" in k for k in norms)
    assert all(v == 0.0 for v in norms.values())


# --------------------------------------------------------------------
# Test 5: non-LoRA leaves are ignored.
# --------------------------------------------------------------------
def test_non_lora_skipped():
    base = {
        "PaliGemma": {
            "attn": {
                "w": np.zeros((2, 4), dtype=np.float32),
                "lora_a": np.zeros((2, 4), dtype=np.float32),
            },
        },
    }
    fine = {
        "PaliGemma": {
            "attn": {
                # Even though the base weight has a delta, it is NOT LoRA
                # so it must be skipped by the LoRA detector.
                "w": np.ones((2, 4), dtype=np.float32),
                "lora_a": np.zeros((2, 4), dtype=np.float32),
            },
        },
    }
    norms = delta_norm.compute_lora_delta_norms(base, fine)
    assert all("lora_a" in k for k in norms)
    assert all(v == 0.0 for v in norms.values())
