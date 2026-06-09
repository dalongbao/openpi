#!/bin/bash
# SHARED training wrapper — any teammate runs this. Datasets are read from a shared,
# group-readable location (default lichin's scratch); checkpoints + HF caches go to
# YOUR OWN scratch ($USER). Norm stats come from the repo's assets/ (commit them once).
#
#   sbatch --partition=gpu.120h --time=120:00:00 --mem-per-cpu=16G --cpus-per-task=8 --gpus=a100:1 \
#          3dvision-experiments/run_train_shared.slurm <config> <exp_name> <seed>
#   e.g.  ... run_train_shared.slurm pi05_ego_mix_oic_n15 mix_n15 42
#
# Prereqs (each teammate, once): clone openpi, `UV_FROZEN=1 uv sync` on the LOGIN node.
# Override DATA_HOME=/path if the shared datasets live elsewhere.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export HF_LEROBOT_HOME="${DATA_HOME:-/cluster/scratch/lichin/lerobot}"   # shared datasets (read-only)
export HF_HOME="/cluster/scratch/$USER/hf_cache"
export HF_DATASETS_CACHE="/cluster/scratch/$USER/hf_cache/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export UV_FROZEN=1
export UV_OFFLINE=1
echo "=== train ${1} exp=${2:-run} seed=${3:-42} user=$USER node=$(hostname) $(date) ==="
uv run python scripts/train.py "${1}" \
    --exp-name="${2:-run}" \
    --seed="${3:-42}" \
    --checkpoint-base-dir="/cluster/scratch/$USER/checkpoints" \
    --resume
