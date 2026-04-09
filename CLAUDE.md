# 3DV Project: Fine-Tuning pi0.5 on Egoverse Data

## Output style

- **All shell commands on a single line** (`&&` to chain). No multi-line code blocks for copy-paste commands.
- Be concise. No preamble, no restating.


## Quick Start (ETH Euler Cluster)

### 1. Clone and install

```bash
ssh euler.ethz.ch
cd ~
git clone <repo-url> openpi && cd openpi
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or ~/.zshrc
uv sync  # creates .venv, installs all deps — takes a few minutes first time
```

### 2. One-time setup: download base weights

Compute nodes have no internet. Download on login node first:

```bash
pip install --user gcsfs && python3 -c "import gcsfs; fs=gcsfs.GCSFileSystem(token='anon'); fs.get('openpi-assets/checkpoints/pi05_base/params','/cluster/work/cvg/data/Egoverse/pi05_base_jax/params',recursive=True); print('Done')"
```

Also cache the tokenizer (needed once):

```bash
cd ~/openpi && HF_LEROBOT_HOME=/cluster/work/cvg/data/Egoverse/lerobot_egoverse uv run python scripts/compute_norm_stats.py --config-name pi05_egoverse
```

### 3. Submit training

```bash
sbatch --partition=gpu.120h --time=120:00:00 --mem-per-cpu=16G --cpus-per-task=8 --gpus=a100:1 3dvision-experiments/run.slurm [config] [exp_name] [seed]
```

Defaults: `pi05_egoverse test 42`. Resume after kill: same command (script has `--resume`).

### 4. Monitor

```bash
tail -f slurm-<jobid>.out
sacct -j <jobid> --format=JobID,State,Elapsed,AllocTRES%60
```

## Shared Paths on Euler

| What | Path |
|------|------|
| Raw h5 data (object_in_bowl, 78 eps) | `/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/object_in_bowl_processed_50hz/` |
| Raw h5 data (bag_groceries, 300 eps) | `/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries/` |
| Converted LeRobot dataset (5 eps) | `/cluster/work/cvg/data/Egoverse/lerobot_egoverse/egoverse/all/` |
| pi0.5 base weights (JAX/orbax) | `/cluster/work/cvg/data/Egoverse/pi05_base_jax/params` |
| pi0.5 base weights (safetensors, unused) | `/cluster/work/cvg/data/rytsui/pi05_base/` |

Per-user paths (checkpoints, experiments):

| What | Path |
|------|------|
| Training checkpoints | `/cluster/work/cvg/data/<username>/checkpoints/pi05_egoverse/<exp_name>/` |
| Norm stats | `~/openpi/assets/pi05_egoverse/egoverse/all` |

## Files We Added

- `3dvision-experiments/convert_h5_to_lerobot.py` — converts Egoverse h5 to LeRobot v2 format
- `3dvision-experiments/convert_data.slurm` — SLURM job for data conversion
- `3dvision-experiments/run.slurm` — SLURM wrapper for training
- `3dvision-experiments/run_inference.py` — baseline inference on real h5 frames (pretrained pi0.5, no finetuning); loads base weights + egoverse norm stats, reports arm/hand MSE vs GT
- `3dvision-experiments/run_inference.slurm` — SLURM wrapper for baseline inference
- `3dvision-experiments/NOTES.md` — detailed setup log
- `src/openpi/policies/egoverse_policy.py` — EgoverseInputs/EgoverseOutputs data transforms
- `src/openpi/training/config.py` — added `LeRobotEgoverseDataConfig` and `pi05_egoverse` config

## Baseline Inference

Submit: `sbatch --time=01:00:00 --mem-per-cpu=16G --cpus-per-task=8 --gpus=1 3dvision-experiments/run_inference.slurm`

Custom args: `sbatch ... 3dvision-experiments/run_inference.slurm <h5_path> <num_frames> <frame_stride>`

Loads `pi05_base_jax` into the `pi05_egoverse` config (LoRA adapters init to zero → functionally base model), runs on real frames from an h5 file, reports arm/hand MSE vs ground truth. First call = 2–5 min JIT compile; later frames fast.

## Converting More Data

```bash
sbatch --cpus-per-task=8 --mem-per-cpu=16G --time=12:00:00 3dvision-experiments/convert_data.slurm
```

Then recompute norm stats on login node:

```bash
cd ~/openpi && HF_LEROBOT_HOME=/cluster/work/cvg/data/Egoverse/lerobot_egoverse uv run python scripts/compute_norm_stats.py --config-name pi05_egoverse
```

## Gotchas

- **No internet on compute nodes.** All downloads (weights, tokenizer, pip packages) must happen on login node.
- **Home dir has limited quota.** Never save checkpoints to `~/`. Use `--checkpoint-base-dir=/cluster/work/cvg/data/<username>/checkpoints`.
- **LoRA requires JAX trainer** (`scripts/train.py`). The PyTorch trainer (`scripts/train_pytorch.py`) does not support LoRA.
- **wandb is disabled** in `run.slurm` (compute has no internet). Loss is logged to stdout / slurm output file.
- **Only `object_in_bowl` is converted** so far. `bag_groceries` is bimanual with different h5 keys and needs separate handling.
- **A100 required for LoRA training.** Model is ~11.5GB; RTX 4090 (24GB) and below OOM during the training step with batch_size=32.
- **SLURM account quirk**: don't put `#SBATCH` directives in the script — pass everything on the `sbatch` command line. The account/partition combo is fragile.

## Training Config

- **Config name**: `pi05_egoverse`
- **Model**: pi0.5 with LoRA (gemma_2b_lora + gemma_300m_lora)
- **Batch size**: 32
- **Steps**: 30k, checkpoint every 5k
- **LR**: 5e-5 cosine with 1k warmup
- **Data**: 5 episodes of `object_in_bowl` at 50Hz, single Aria camera, 24-dim state/actions (7 arm + 17 hand)
