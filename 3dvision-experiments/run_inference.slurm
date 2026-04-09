#!/bin/bash
# Submit examples (no #SBATCH in-script — pass all options on the command line):
#   # Single episode, every 10th frame, ALL frames:
#   sbatch --time=01:00:00 --mem-per-cpu=16G --cpus-per-task=8 --gpus=1 \
#          3dvision-experiments/run_inference.slurm \
#          --h5-path /cluster/work/cvg/data/Egoverse/raw_timesynced_h5/object_in_bowl_processed_50hz/20250804_142656.h5 \
#          --frame-stride 10
#
#   # All episodes in a directory, every 10th frame:
#   sbatch --time=02:00:00 --mem-per-cpu=16G --cpus-per-task=8 --gpus=1 \
#          3dvision-experiments/run_inference.slurm \
#          --episodes-dir /cluster/work/cvg/data/Egoverse/raw_timesynced_h5/object_in_bowl_processed_50hz \
#          --frame-stride 10
#
#   # Quick smoke test (8 frames at stride 50):
#   sbatch --time=00:30:00 --mem-per-cpu=16G --cpus-per-task=8 --gpus=1 \
#          3dvision-experiments/run_inference.slurm \
#          --h5-path <path> --num-frames 8 --frame-stride 50
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export HF_LEROBOT_HOME="$HOME/hf_lerobot"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE="disabled"

echo "=== pi0.5 baseline inference ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Args: $*"

uv run python 3dvision-experiments/run_inference.py "$@"
