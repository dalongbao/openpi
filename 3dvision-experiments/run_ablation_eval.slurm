#!/bin/bash
# Offline ablation eval of one checkpoint on held-out R-ID frames. Submit with UV_FROZEN
# (offline-node lerobot@main fix). Pass ablation_eval.py args after the .slurm:
#   sbatch --export=ALL,UV_FROZEN=1 --time=01:00:00 --mem-per-cpu=16G --cpus-per-task=8 \
#     --gpus=rtx_4090:1 3dvision-experiments/run_ablation_eval.slurm \
#     --condition R-ID_only --checkpoint-dir <ckpt> --episodes-dir <held_out_dir>
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export HF_LEROBOT_HOME="$HOME/hf_lerobot"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE="disabled"
export PYTHONUNBUFFERED=1   # stream per-episode lines to the .out live (eval prints <8KB total, so block buffering hides ALL progress until exit)

echo "=== ablation eval ===  Node: $(hostname)  Date: $(date)"
echo "Args: $*"
uv run python 3dvision-experiments/ablation_eval.py "$@"
