#!/bin/bash
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export HF_LEROBOT_HOME="$HOME/hf_lerobot"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE="disabled"

echo "=== pi0.5 trajectory visualization ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Args: $*"

uv run python 3dvision-experiments/visualise_openloop_rollout.py "$@"
