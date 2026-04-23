#!/bin/bash
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export HF_LEROBOT_HOME="/cluster/work/cvg/data/rytsui/lerobot"
mkdir -p "$HF_LEROBOT_HOME"

echo "=== Consolidating bag_grocery to LeRobot ==="
echo "Node: $(hostname)"
echo "Date: $(date)"

uv run python 3dvision-experiments/consolidate_bag_grocery.py \
    --src-dir /cluster/work/cvg/jiaqchen/EGOVERSE_DATA_3DV/bag_grocery \
    --dst-dir /cluster/work/cvg/data/rytsui/lerobot

echo "=== Done. Run norm stats on login node: ==="
echo "cd ~/openpi && HF_LEROBOT_HOME=/cluster/work/cvg/data/rytsui/lerobot uv run python scripts/compute_norm_stats.py --config-name pi05_egoverse_bimanual"
