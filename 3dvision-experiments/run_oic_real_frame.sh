#!/bin/bash
# run_oic_real_frame.sh — Track-A-for-oic: run the 2537-ep oic_human model on REAL Aria
# frames (no Isaac) to measure model quality, free of the sim visual-gap confound.
# Bakes the sbatch flags so nothing gets mangled. Extra args pass through to the .py.
#
# Usage (login node):
#   bash ~/scripts/run_oic_real_frame.sh
#   bash ~/scripts/run_oic_real_frame.sh --num-episodes 50 --frame-stride 5
set -euo pipefail

cd "$HOME/openpi"   # uv repo (venv + deps live here); SLURM_SUBMIT_DIR for the .slurm

# UV_FROZEN avoids re-resolving the moving lerobot@main git ref on the offline node.
sbatch --export=ALL,UV_FROZEN=1 \
  --time=01:00:00 --mem-per-cpu=16G --cpus-per-task=8 --gpus=rtx_4090:1 \
  3dvision-experiments/run_inference_oic.slurm "$@"
