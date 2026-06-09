# Egoverse Data-Mix Training — Teammate Guide

We're measuring whether cheap human video (**H-ID**, `object_in_container`) can substitute for
expensive robot teleop (**R-ID**, `object_in_bowl`) when fine-tuning π0.5. Everyone trains a few
configs from the list below; we eval them all on the **same 12 held-out R-ID episodes** and plot
the data-efficiency curve.

**Pick 2–3 configs from §3 and tell the team (shared sheet) so we don't double-train.**
Each person is capped at **2 GPUs**, so coordinate. **Priority = the `*_n64` configs + `hid_only`
first** (that's the go/no-go); the smaller `_n5/_n15/_n30` points fill the curve after.

---

## 1. One-time setup (do once on Euler)

```bash
ssh euler.ethz.ch
git clone https://github.com/dalongbao/openpi ~/openpi && cd ~/openpi
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
UV_FROZEN=1 uv sync                     # builds the venv ON THE LOGIN NODE (compute has no internet)

# Pre-cache the PaliGemma tokenizer ON THE LOGIN NODE. Training downloads it from gs://big_vision,
# but compute nodes have NO internet -> without this the training job dies at startup. Run once:
UV_FROZEN=1 uv run python -c "from openpi.shared import download; download.maybe_download('gs://big_vision/paligemma_tokenizer.model', gs={'token':'anon'})"
```
> ⚠️ Don't skip the tokenizer line — it's the #1 cause of a training job failing instantly on a
> compute node. It's idempotent (instant if already cached), so just run it.

**Get the datasets** (they live in one place — pick the one that works):

- **Option A — read the shared copy (fastest, no download):** point `DATA_HOME` at lichin's scratch.
  Test: `ls /cluster/scratch/lichin/lerobot/egoverse/all`. If that lists files, you're done — use
  `DATA_HOME=/cluster/scratch/lichin/lerobot` (the default in the slurm).
- **Option B — download from HF (if A is permission-blocked):** on the **login node**,
  ```bash
  huggingface-cli login            # one-time, write token
  mkdir -p /cluster/scratch/$USER/lerobot/egoverse
  huggingface-cli download <HF_DATASET_REPO_ALL>     --repo-type dataset --local-dir /cluster/scratch/$USER/lerobot/egoverse/all
  huggingface-cli download <HF_DATASET_REPO_MIX>     --repo-type dataset --local-dir /cluster/scratch/$USER/lerobot/egoverse/oic_mix
  # only if you train hid_only:
  huggingface-cli download <HF_DATASET_REPO_HUMAN>   --repo-type dataset --local-dir /cluster/scratch/$USER/lerobot/egoverse/oic_human
  ```
  then `export DATA_HOME=/cluster/scratch/$USER/lerobot` before submitting.

> Base weights are already shared at `/cluster/work/cvg/data/Egoverse/pi05_base_jax` — nothing to do.

---

## 2. Per-config recipe (3 steps)

For each config you train:

**(a) Norm stats** — login node, once per config (~20–30 min):
```bash
cd ~/openpi && DATA_HOME=${DATA_HOME:-/cluster/scratch/lichin/lerobot} \
  HF_LEROBOT_HOME=$DATA_HOME HF_HOME=/cluster/scratch/$USER/hf_cache HF_DATASETS_CACHE=/cluster/scratch/$USER/hf_cache/datasets \
  UV_FROZEN=1 uv run python scripts/compute_norm_stats.py --config-name <CONFIG> --max_frames 20000
```
> **Two must-dos:**
> - Set `HF_HOME`/`HF_DATASETS_CACHE` to **scratch** — else the dataset cache regen overflows home (50 GB) → `Disk quota exceeded`.
> - Pass **`--max_frames 20000`** — without it the script decodes *every* image of the full dataset
>   (~8 h for `egoverse/all`!); a 20k-frame subsample gives identical 24-dim stats in ~20–30 min.

**(b) Train** — A100, ~half a day (~10–14 h for 30k steps; resumes if requeued):
```bash
cd ~/openpi && DATA_HOME=${DATA_HOME:-/cluster/scratch/lichin/lerobot} \
  sbatch --partition=gpu.24h --time=24:00:00 --mem-per-cpu=16G --cpus-per-task=8 --gpus=a100:1 \
  3dvision-experiments/run_train_shared.slurm <CONFIG> <EXP_NAME> 42
```
Checkpoints → your own `/cluster/scratch/$USER/checkpoints/<CONFIG>/<EXP_NAME>/`.

> **Use `gpu.24h`, not `gpu.120h`.** Training is ~10–14 h, so 24 h is plenty — and the shorter
> partition + backfill **queues much faster**. It's safe even if it doesn't finish: the slurm passes
> `--resume`, checkpoints land every 5k steps, so a resubmit (same command) continues from the last
> one. (If `gpu.24h` doesn't exist on the cluster, just keep `--time=24:00:00` — backfill still helps.)

> **Shorter queue — cast a wider net by GPU memory.** The 40 GB A100s are scarce (3 nodes). Instead
> of pinning `--gpus=a100:1`, request *any* card with enough VRAM to hold batch-32 LoRA and take
> whichever frees first:
> ```bash
> ... --gpus=1 --gres=gpumem:40g ...   # matches a100 (40), a100_80gb (80), pro_6000 (96)
> ```
> Keeps batch-32 unchanged (no ablation confound), unlike a 24 GB RTX 4090 which OOMs at batch-32.
> ⚠️ If it lands on a **pro_6000** (new Blackwell card — *faster*, not slower, but unverified with
> openpi's pinned JAX/CUDA): watch the first minute of the log. If JAX errors at CUDA init,
> `scancel` and resubmit with `--gpus=a100_80gb:1` (5 nodes, also short queue). If Euler rejects the
> `--gres=gpumem` syntax, `--gpus=a100_80gb:1` is the safe fallback.

**(c) Monitor + report:**
```bash
squeue -u $USER && tail -f ~/openpi/slurm-<jobid>.out     # loss should print and drop after JIT
```
When it hits step 30000, push the checkpoint to HF (durable + central eval) and update the sheet:
```bash
huggingface-cli upload --repo-type model --private <HF_MODEL_ORG>/<CONFIG> \
  /cluster/scratch/$USER/checkpoints/<CONFIG>/<EXP_NAME>/30000 .
```

---

## 3. The configs — pick from this list

| Pick | `<CONFIG>` | `<EXP_NAME>` | What it is | Priority |
|------|-----------|--------------|------------|----------|
| R-ID n64 | `pi05_egoverse`           | `rid64`   | robot teleop, all 64 eps (baseline top) | **Wave 1** |
| MIX n64  | `pi05_ego_mix_oic`        | `mix64`   | 64 robot + full human (augmentation top) | **Wave 1** |
| H-ID only| `pi05_ego_human_oic`      | `hid`     | human only (reach, no grasp — control)  | **Wave 1** |
| R-ID n5  | `pi05_egoverse_n5`        | `rid5`    | robot teleop, 5 eps     | Wave 2 |
| R-ID n15 | `pi05_egoverse_n15`       | `rid15`   | robot teleop, 15 eps    | Wave 2 |
| R-ID n30 | `pi05_egoverse_n30`       | `rid30`   | robot teleop, 30 eps    | Wave 2 |
| MIX n5   | `pi05_ego_mix_oic_n5`     | `mix5`    | 5 robot + full human    | Wave 2 |
| MIX n15  | `pi05_ego_mix_oic_n15`    | `mix15`   | 15 robot + full human   | Wave 2 |
| MIX n30  | `pi05_ego_mix_oic_n30`    | `mix30`   | 30 robot + full human   | Wave 2 |

`base` (untrained floor) needs **no training** — it's eval-only (`ablation_eval.py --finetuned false`).

**Example** — to train MIX n15:
```bash
cd ~/openpi && DATA_HOME=${DATA_HOME:-/cluster/scratch/lichin/lerobot} HF_LEROBOT_HOME=$DATA_HOME HF_HOME=/cluster/scratch/$USER/hf_cache HF_DATASETS_CACHE=/cluster/scratch/$USER/hf_cache/datasets UV_FROZEN=1 uv run python scripts/compute_norm_stats.py --config-name pi05_ego_mix_oic_n15 --max_frames 20000
cd ~/openpi && DATA_HOME=${DATA_HOME:-/cluster/scratch/lichin/lerobot} sbatch --partition=gpu.24h --time=24:00:00 --mem-per-cpu=16G --cpus-per-task=8 --gpus=a100:1 3dvision-experiments/run_train_shared.slurm pi05_ego_mix_oic_n15 mix15 42
```

---

## 3.5 What objects each split contains

Objects are **duck / ball / plant**, each in plain / `_trans` (transparent) / `_bucket` variants
(9 classes). Splits are **stratified by object** so the curve measures data *quantity*, not coverage.

**Held-out eval (12 eps — all 9 classes):** ball×2, ball_bucket×1, ball_trans×1, duck×2,
duck_bucket×1, duck_trans×2, plant×1, plant_bucket×1, plant_trans×1.

**Training subsets (nested, 5 ⊂ 15 ⊂ 30 ⊂ 64):**

| N | classes | composition |
|---|---------|-------------|
| 5  | 5 of 9 (**no plant**, no duck_trans) | ball, ball_bucket, ball_trans, duck, duck_bucket — ×1 each |
| 15 | all 9 | ball / ball_bucket / ball_trans / duck / duck_bucket / duck_trans ×2; plant / plant_bucket / plant_trans ×1 |
| 30 | all 9 | ball-family ×4; duck / duck_bucket / duck_trans ×3; plant-family ×3 |
| 64 | all 9 (+3 unlabelled `n`/`dk`) | the full training pool |

N=5 has **no plant** (5 slots can't cover 9 classes) → low-N models are coverage-limited on held-out
plant; this hits `rid_n5` and `mix_n5` equally, so their *comparison* stays fair.

## 4. Rules

- **Claim your configs in the shared sheet first** (`config · owner · status · ckpt path · eval score`) — no double-training.
- **≤2 GPUs per person.** Do a **Wave-1** config before a Wave-2 one if Wave 1 isn't covered yet.
- **A100 required** (LoRA batch-32 OOMs on smaller GPUs).
- All `uv run` need `UV_FROZEN=1`; compute-node jobs also need `UV_OFFLINE=1` (the slurm sets it).
- Don't touch `held_out_rid.txt` / the subset indices — they define the eval; changing them invalidates comparisons.
- Eval is run centrally on the 12 held-out (raw h5, no conversion) once checkpoints are on HF.
