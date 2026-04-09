#!/bin/bash
#SBATCH --job-name=pi05_baseline_infer
#SBATCH --output=logs/infer_%j.out
#SBATCH --error=logs/infer_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --time=01:00:00

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export HF_LEROBOT_HOME="/cluster/work/cvg/data/rytsui/lerobot"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE="disabled"

# Pass an h5 file as $1, or default to one of the processed episodes.
H5_PATH="${1:-/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/object_in_bowl_processed_50hz/20250804_142656.h5}"
NUM_FRAMES="${2:-8}"
FRAME_STRIDE="${3:-50}"

echo "=== pi0.5 baseline inference ==="
echo "Node:       $(hostname)"
echo "Date:       $(date)"
echo "H5 file:    $H5_PATH"
echo "Frames:     $NUM_FRAMES (stride=$FRAME_STRIDE)"

uv run python 3dvision-experiments/run_inference.py \
    --h5-path "$H5_PATH" \
    --num-frames "$NUM_FRAMES" \
    --frame-stride "$FRAME_STRIDE"
