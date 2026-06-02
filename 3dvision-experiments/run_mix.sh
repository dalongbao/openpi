#!/bin/bash
# Pipeline: consolidate mix dataset -> norm stats -> train (chained resume).
set -euo pipefail

DST_DIR="/cluster/work/cvg/data/Egoverse/lerobot_egoverse"
HUMAN_DIR="/cluster/work/cvg/jiaqchen/EGOVERSE_DATA_3DV/bag_grocery"
TELEOP_DIR="/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries"
CPU="--cpus-per-task=8 --mem-per-cpu=16G"
GPU="--partition=gpu.24h --time=24:00:00 --mem-per-cpu=16G --cpus-per-task=8 --gpus=a100:1"
ENV="export PATH=\$HOME/.local/bin:\$PATH && cd ~/openpi && export HF_HOME=/cluster/work/cvg/data/Egoverse/hf_cache && export HF_DATASETS_CACHE=/cluster/work/cvg/data/Egoverse/hf_cache/datasets && export HF_LEROBOT_HOME=$DST_DIR"

echo "=== Mix pipeline (human Cartesian + teleop joint, action_mask, action_dim=60) ==="

C=$(sbatch --parsable $CPU --time=12:00:00 --wrap="$ENV && uv run python 3dvision-experiments/consolidate_mix.py --human-dir $HUMAN_DIR --teleop-dir $TELEOP_DIR --repo-name egoverse/bag_grocery_mix --dst-dir $DST_DIR")
echo "  Consolidate: $C"

N=$(sbatch --parsable --dependency=afterok:$C $CPU --time=02:00:00 --wrap="$ENV && export JAX_PLATFORMS=cpu && export HF_HUB_OFFLINE=1 && export HF_DATASETS_OFFLINE=1 && uv run python scripts/compute_norm_stats.py --config-name pi05_ego_mix_bag_grocery --max-frames 5000")
echo "  Norm stats: $N"

T1=$(sbatch --parsable --dependency=afterok:$N $GPU 3dvision-experiments/run.slurm pi05_ego_mix_bag_grocery mix_bag 42)
echo "  Train1: $T1"

T2=$(sbatch --parsable --dependency=afterany:$T1 $GPU 3dvision-experiments/run.slurm pi05_ego_mix_bag_grocery mix_bag 42)
echo "  Train2 (resume): $T2"

echo "=== All mix jobs submitted ==="
