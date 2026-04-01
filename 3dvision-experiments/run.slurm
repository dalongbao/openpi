#!/bin/bash
#SBATCH --job-name=pi05_egoverse
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
# Adjust partition and GPU type for your cluster:
# #SBATCH --partition=gpu
# #SBATCH --gres=gpu:a100:1

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
echo "GPUs:     $CUDA_VISIBLE_DEVICES"
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

# ── Compute norm stats if missing ────────────────────────────────────
ASSETS_DIR="./assets/${CONFIG}"
if [ ! -d "$ASSETS_DIR" ] || [ -z "$(ls -A "$ASSETS_DIR" 2>/dev/null)" ]; then
    echo "Computing normalization statistics..."
    uv run scripts/compute_norm_stats.py --config-name "$CONFIG"
fi

# ── Train ────────────────────────────────────────────────────────────
echo "Starting training..."
uv run scripts/train.py "$CONFIG" \
    --exp-name="$EXP_NAME" \
    --seed="$SEED"
