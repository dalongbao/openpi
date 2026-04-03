#!/bin/bash
#SBATCH --job-name=pi05_egoverse
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
#SBATCH --gpus=a100_80gb:1

set -euo pipefail

# ── Arguments ────────────────────────────────────────────────────────
CONFIG="${1:-pi05_egoverse}"
EXP_NAME="${2:-test}"
SEED="${3:-42}"

echo "=== pi0.5 Egoverse Finetuning ==="
echo "Config:   $CONFIG"
echo "Exp name: $EXP_NAME"
echo "Seed:     $SEED"
echo "Node:     $(hostname)"
echo "GPUs:     ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Date:     $(date)"
echo "==========================================="

# ── Environment ──────────────────────────────────────────────────────
cd "$SLURM_SUBMIT_DIR"

# If uv is not on PATH, adjust this.
export PATH="$HOME/.local/bin:$PATH"

# Dataset location (same as convert_data.slurm).
export HF_LEROBOT_HOME="/cluster/work/cvg/data/rytsui/lerobot"

# Let JAX use 90% of GPU memory.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# Wandb project — change as needed.
export WANDB_PROJECT="pi05_egoverse"

# ── Train ────────────────────────────────────────────────────────────
echo "Starting training..."
uv run python scripts/train.py "$CONFIG" \
    --exp-name="$EXP_NAME" \
    --seed="$SEED"
