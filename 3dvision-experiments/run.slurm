#!/bin/bash
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export HF_LEROBOT_HOME="/cluster/work/cvg/data/Egoverse/lerobot_egoverse"
export HF_HUB_OFFLINE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE="disabled"
uv run python scripts/train.py "${1:-pi05_egoverse}" \
    --exp-name="${2:-test}" \
    --seed="${3:-42}" \
    --checkpoint-base-dir=/cluster/work/cvg/data/rytsui/checkpoints \
    --resume
