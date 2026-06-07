#!/bin/bash
# run_real_frame_eval.sh — Track A: run a pi0.5 checkpoint on REAL Aria frames
# (no Isaac) via run_inference.py, to tell the sim visual-gap apart from a weak model.
#
# Long cluster paths are baked in here so nothing gets mangled by terminal line-wrapping.
# You only type the episode id.
#
# Usage (run on the Euler login node):
#   bash ~/scripts/run_real_frame_eval.sh                       # default episode + 5-ep finetune
#   bash ~/scripts/run_real_frame_eval.sh 20250804_104715       # a different episode
#   bash ~/scripts/run_real_frame_eval.sh 20250804_142656 /checkpoints/.../STEP   # custom checkpoint
set -euo pipefail

EP="${1:-20250804_142656}"
CKPT="${2:-/cluster/work/cvg/data/rytsui/checkpoints/pi05_egoverse/test/29999}"
TASK_DIR=/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/object_in_bowl_processed_50hz
H5="$TASK_DIR/${EP%.h5}.h5"

[ -f "$H5" ]   || { echo "[err] episode not found: $H5"; exit 1; }
[ -d "$CKPT" ] || { echo "[err] checkpoint dir not found: $CKPT"; exit 1; }

cd "$HOME/openpi"   # so SLURM_SUBMIT_DIR is the uv repo (norm stats + venv live here)

echo "[run] episode=$EP"
echo "[run] h5=$H5"
echo "[run] checkpoint=$CKPT"

# UV_FROZEN avoids re-resolving the moving `lerobot @ main` git ref on the offline node.
sbatch --export=ALL,UV_FROZEN=1 \
  --time=01:00:00 --mem-per-cpu=16G --cpus-per-task=8 --gpus=rtx_3090:1 \
  3dvision-experiments/run_inference.slurm \
  --h5-path "$H5" \
  --frame-stride 10 \
  --finetuned \
  --checkpoint-dir "$CKPT" \
  --prompt "put the object in the bowl"
