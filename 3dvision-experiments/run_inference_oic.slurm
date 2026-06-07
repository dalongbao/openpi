#!/bin/bash
# Real-frame inference for the oic_human (6-dim) model. Submit via run_oic_real_frame.sh,
# or directly (UV_FROZEN avoids the offline-node lerobot@main git fetch):
#   sbatch --export=ALL,UV_FROZEN=1 --time=01:00:00 --mem-per-cpu=16G --cpus-per-task=8 \
#          --gpus=rtx_4090:1 3dvision-experiments/run_inference_oic.slurm
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export HF_LEROBOT_HOME="$HOME/hf_lerobot"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE="disabled"

echo "=== oic_human real-frame inference ==="
echo "Node: $(hostname)  Date: $(date)"
echo "Args: $*"

uv run python 3dvision-experiments/run_inference_oic.py "$@"
