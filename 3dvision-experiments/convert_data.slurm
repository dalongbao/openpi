#!/bin/bash
#SBATCH --job-name=convert_egoverse
#SBATCH --output=logs/convert_%j.out
#SBATCH --error=logs/convert_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=12:00:00
# No GPU needed for conversion.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"

# Write datasets to work dir (home has limited quota).
export HF_LEROBOT_HOME="/cluster/work/cvg/data/rytsui/lerobot"
mkdir -p "$HF_LEROBOT_HOME"

echo "=== Converting Egoverse h5 to LeRobot ==="
echo "Node: $(hostname)"
echo "Date: $(date)"

uv run python 3dvision-experiments/convert_h5_to_lerobot.py \
    --data_dir /cluster/work/cvg/data/Egoverse/raw_timesynced_h5/object_in_bowl_processed_50hz \
    --max_episodes 5

echo "=== Conversion done. Run norm stats on login node (needs internet): ==="
echo "cd ~/openpi && HF_LEROBOT_HOME=/cluster/work/cvg/data/rytsui/lerobot uv run python scripts/compute_norm_stats.py --config-name pi05_egoverse"
