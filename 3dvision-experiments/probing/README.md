# pi0.5 Probing

Mechanistic-interpretability utilities for the pi0.5 + Egoverse fine-tunes.

Two pieces:

1. **`delta_norm.py`** — load orbax checkpoints, compute per-layer Frobenius
   norms of `(finetuned - base)` for every LoRA adapter. Pure numpy — no
   JAX required. Localizes *where in the model* each fine-tune deviates
   from `pi05_base`.
2. **`activation_hooks.py`** — JAX/Flax forward hooks that record
   attention weights and FFN activations during a rollout. Built on
   Flax's `capture_intermediates`. Falls back to a monkey-patched
   forward (eager-only, `jit=False`) when the capture path can't run.

## Install

The probing package only needs `numpy`, `pandas`, and `matplotlib`.
`delta_norm.py` additionally needs openpi installed (it reuses
`openpi.models.model.restore_params` to read orbax checkpoints).
`activation_hooks.py` needs `jax` + `flax` at runtime.

## Δ-norm CLI

```bash
python 3dvision-experiments/probing/run_delta_norm.py \
    --base-dir /cluster/work/cvg/data/Egoverse/pi05_base_jax \
    --finetuned-dirs \
        object_in_bowl:/cluster/work/cvg/data/rytsui/checkpoints/pi05_egoverse/test/29999 \
        bag_grocery:/cluster/work/cvg/data/rytsui/checkpoints/pi05_egoverse/bag_grocery/29999 \
        human_oic:/cluster/work/cvg/data/rytsui/checkpoints/pi05_ego_human_oic/human_oic/29999 \
        mix:/cluster/work/cvg/data/rytsui/checkpoints/pi05_ego_mix_bag_grocery/mix_bag/29999 \
    --out /tmp/delta_norms.csv
```

**TODO:** the four finetuned paths above are the team's *assumed*
convention pulled from `run_all_egoverse.slurm`. Verify the actual
on-disk paths before W2.

## Activation hooks

```python
from probing import attach_hooks
recorder = attach_hooks(policy._model, layer_indices=[0, 4, 8, 12, 16])
out = recorder.record(policy._model, params=None, observation=obs)
# out["attention_weights"][layer_idx] -> np.ndarray
# out["attention_entropy"][layer_idx]  -> per-head Shannon entropy

df = recorder.summary()  # (layer, head, mean_entropy)
```

In Isaac Sim eval, set `EvalConfig.probe_config = {"every_other": True}`
and the runner will pull `probes.attach()` + `probes.record()` for you.

## Plots

```python
from probing.analyze import plot_delta_norm_heatmap, plot_attention_entropy_per_layer
plot_delta_norm_heatmap(df, "delta_norms_attn.png", expert="paligemma", submodule="attn")
plot_attention_entropy_per_layer(recorder.summary(), "attn_entropy.png")
```

## What's produced

* `delta_norms.csv` — wide table, one row per
  `(expert, block_index, submodule, parameter, adapter)`, one column per
  finetune. Use with pandas.
* `delta_norms_<expert>_<submodule>.png` — heatmap (block × checkpoint).
* `attn_entropy.png` — mean attention entropy per layer per head.

## If hooks fail

The `capture_intermediates` path requires that the linen module beneath
the nnx-bridge wrapper exposes its `apply()` method. The probe-aware
path will raise a clean `_CaptureUnavailable` and fall back to the
monkey-patch path. The fallback REQUIRES `jit=False` and only captures
attention outputs (not probs) — useful for sanity-checking but not for
the full W2 analysis. If you see the fallback path firing, check that:

1. The model passed in is the `Policy._model` nnx module (not the
   `Policy` wrapper). The bridged linen module is reached via
   `model.PaliGemma.llm.module` or similar.
2. JAX is recent enough — Flax `capture_intermediates` predates JAX 0.5
   so the openpi-pinned `jax==0.5.3` is fine.

## openpi internals we depend on

* `openpi.models.model.restore_params` — for orbax loading.
* `openpi.models.lora` — for LoRA naming convention (`lora_a`, `lora_b`,
  `gating_einsum_lora_a`, etc.). See `src/openpi/models/lora.py:51-121`.
* `openpi.models.gemma` — for the per-expert suffix convention
  (`_name` at `gemma.py:443` maps index 1 to action expert).

We do **not** import any private symbol. If a future openpi refactor
renames any of these, `delta_norm.py` will warn (LoRA leaves not found)
rather than crash.

## Risks before W2

* Action-expert intermediates: the nnx scan stacks BOTH experts'
  weights along axis 0. The per-block indexing is the same, but per-key
  axis splitting between experts has not been validated against a real
  pi0.5 checkpoint — verify on the first run.
* Memory: a full-resolution attention probs capture on `gemma_2b` at
  pi0.5's sequence length is ~150 MB / layer / step. Default `every_other`
  + small `max_records` keeps this manageable but DO monitor RSS on Euler.
