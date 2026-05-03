#!/bin/bash
# Master script: consolidate human data, compute norm stats, launch all training jobs.
#
# Usage:
#   bash 3dvision-experiments/run_all_egoverse.slurm
#
# This submits everything as SLURM jobs with dependencies so they run in order.
set -euo pipefail
cd "$(dirname "$0")/.."

DST_DIR="/cluster/work/cvg/data/Egoverse/lerobot_egoverse"
HF_CACHE="/cluster/work/cvg/data/rytsui/hf_cache"
HUMAN_BAG="/cluster/work/cvg/jiaqchen/EGOVERSE_DATA_3DV/bag_grocery"
HUMAN_OIC="/cluster/work/cvg/jiaqchen/EGOVERSE_DATA_3DV/object_in_container"
TELEOP_BAG="/cluster/work/cvg/jiaqchen/EGOVERSE_DATA_3DV/bag_grocery"

COMMON="--mem-per-cpu=16G --cpus-per-task=8"
GPU="--partition=gpu.24h --time=24:00:00 --mem-per-cpu=16G --cpus-per-task=8 --gpus=a100:1"

ENV="export PATH=\$HOME/.local/bin:\$PATH && cd ~/openpi && export HF_HOME=/cluster/work/cvg/data/rytsui/hf_cache && export HF_DATASETS_CACHE=/cluster/work/cvg/data/rytsui/hf_cache/datasets && export HF_LEROBOT_HOME=$DST_DIR"

echo "=== Step 1: Consolidate datasets ==="

# Human bag_grocery
C1=$(sbatch --parsable $COMMON --time=12:00:00 --wrap="$ENV && uv run python 3dvision-experiments/consolidate_egoverse.py --src-dirs $HUMAN_BAG --repo-name egoverse/bag_grocery_human --task 'bag the groceries' --dst-dir $DST_DIR")
echo "  bag_grocery_human: $C1"

# Human object_in_container (6-dim single arm)
C2=$(sbatch --parsable $COMMON --time=12:00:00 --wrap="$ENV && uv run python 3dvision-experiments/consolidate_egoverse.py --src-dirs $HUMAN_OIC --repo-name egoverse/oic_human --task 'put the object in the container' --action-dim 6 --dst-dir $DST_DIR")
echo "  oic_human: $C2"

# Mixed bag_grocery (human + teleop)
C3=$(sbatch --parsable $COMMON --time=12:00:00 --wrap="$ENV && uv run python 3dvision-experiments/consolidate_egoverse.py --src-dirs $HUMAN_BAG $TELEOP_BAG --repo-name egoverse/bag_grocery_mix --task 'bag the groceries' --dst-dir $DST_DIR")
echo "  bag_grocery_mix: $C3"

echo ""
echo "=== Step 2: Compute norm stats (after consolidation) ==="

N1=$(sbatch --parsable --dependency=afterok:$C1 $COMMON --time=02:00:00 --wrap="$ENV && uv run python scripts/compute_norm_stats.py --config-name pi05_ego_human_bag_grocery")
echo "  norm human_bag: $N1"

N2=$(sbatch --parsable --dependency=afterok:$C2 $COMMON --time=02:00:00 --wrap="$ENV && uv run python scripts/compute_norm_stats.py --config-name pi05_ego_human_oic")
echo "  norm human_oic: $N2"

N3=$(sbatch --parsable --dependency=afterok:$C3 $COMMON --time=02:00:00 --wrap="$ENV && uv run python scripts/compute_norm_stats.py --config-name pi05_ego_mix_bag_grocery")
echo "  norm mix_bag: $N3"

echo ""
echo "=== Step 3: Training (after norm stats) ==="

# Human bag_grocery only
T1=$(sbatch --parsable --dependency=afterok:$N1 $GPU 3dvision-experiments/run.slurm pi05_ego_human_bag_grocery human_bag 42)
echo "  train human_bag: $T1"
sbatch --dependency=afterany:$T1 $GPU 3dvision-experiments/run.slurm pi05_ego_human_bag_grocery human_bag 42

# Human object_in_container only
T2=$(sbatch --parsable --dependency=afterok:$N2 $GPU 3dvision-experiments/run.slurm pi05_ego_human_oic human_oic 42)
echo "  train human_oic: $T2"
sbatch --dependency=afterany:$T2 $GPU 3dvision-experiments/run.slurm pi05_ego_human_oic human_oic 42

# Mixed bag_grocery (human + teleop)
T3=$(sbatch --parsable --dependency=afterok:$N3 $GPU 3dvision-experiments/run.slurm pi05_ego_mix_bag_grocery mix_bag 42)
echo "  train mix_bag: $T3"
sbatch --dependency=afterany:$T3 $GPU 3dvision-experiments/run.slurm pi05_ego_mix_bag_grocery mix_bag 42

echo ""
echo "=== All jobs submitted. ==="
echo "Monitor: squeue -u \$USER"
