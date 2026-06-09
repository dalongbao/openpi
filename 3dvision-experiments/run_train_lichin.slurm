#!/bin/bash
# Training wrapper for lichin (scratch paths + offline/frozen env). Works for ANY config:
#   sbatch --partition=gpu.120h --time=120:00:00 --mem-per-cpu=16G --cpus-per-task=8 --gpus=a100:1 \
#          3dvision-experiments/run_train_lichin.slurm <config> <exp_name> <seed>
# R-ID baseline:  ... run_train_lichin.slurm pi05_egoverse      rid64    42
# the mix:        ... run_train_lichin.slurm pi05_ego_mix_oic   mix_oic  42
# Prereqs: (1) `UV_FROZEN=1 uv sync` once on the LOGIN node (builds the venv; compute has no internet),
#          (2) norm stats computed for <config> on the login node. Resumes if the checkpoint exists.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export HF_LEROBOT_HOME="/cluster/scratch/lichin/lerobot"
export HF_HOME="/cluster/scratch/lichin/hf_cache"
export HF_DATASETS_CACHE="/cluster/scratch/lichin/hf_cache/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export UV_FROZEN=1
export UV_OFFLINE=1
echo "=== train ${1:-pi05_egoverse} exp=${2:-rid64} seed=${3:-42} on $(hostname) $(date) ==="
uv run python scripts/train.py "${1:-pi05_egoverse}" \
    --exp-name="${2:-rid64}" \
    --seed="${3:-42}" \
    --checkpoint-base-dir=/cluster/scratch/lichin/checkpoints \
    --resume
