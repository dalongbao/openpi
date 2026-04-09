#!/bin/bash
# Submit with (no #SBATCH in-script — pass all options on the command line):
#   sbatch --time=01:00:00 --mem-per-cpu=16G --cpus-per-task=8 --gpus=1 \
#          3dvision-experiments/run_inference.slurm [h5_path] [num_frames] [frame_stride]
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export HF_LEROBOT_HOME="/cluster/work/cvg/data/Egoverse/lerobot_egoverse"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE="disabled"

H5_PATH="${1:-/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/object_in_bowl_processed_50hz/20250804_142656.h5}"
NUM_FRAMES="${2:-8}"
FRAME_STRIDE="${3:-50}"

echo "=== pi0.5 baseline inference ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "H5:   $H5_PATH"
echo "N=$NUM_FRAMES stride=$FRAME_STRIDE"

uv run python 3dvision-experiments/run_inference.py --h5-path "$H5_PATH" --num-frames "$NUM_FRAMES" --frame-stride "$FRAME_STRIDE"
