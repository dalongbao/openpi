#!/bin/bash
set -euo pipefail

DST_DIR="/cluster/work/cvg/data/Egoverse/lerobot_egoverse"
COMMON="--cpus-per-task=8 --mem-per-cpu=16G"
GPU="--partition=gpu.24h --time=24:00:00 --mem-per-cpu=16G --cpus-per-task=8 --gpus=a100:1"
ENV="export PATH=\$HOME/.local/bin:\$PATH && cd ~/openpi && export HF_HOME=/cluster/work/cvg/data/rytsui/hf_cache && export HF_DATASETS_CACHE=/cluster/work/cvg/data/rytsui/hf_cache/datasets && export HF_LEROBOT_HOME=$DST_DIR"

echo "=== OIC human pipeline ==="

C=$(sbatch --parsable $COMMON --time=12:00:00 --wrap="$ENV && uv run python 3dvision-experiments/consolidate_egoverse.py --src-dirs /cluster/work/cvg/jiaqchen/EGOVERSE_DATA_3DV/object_in_container --repo-name egoverse/oic_human --task 'put the object in the container' --action-dim 6 --dst-dir $DST_DIR")
echo "  Convert: $C"

N=$(sbatch --parsable --dependency=afterok:$C $COMMON --time=02:00:00 --wrap="$ENV && uv run python scripts/compute_norm_stats.py --config-name pi05_ego_human_oic")
echo "  Norm: $N"

T1=$(sbatch --parsable --dependency=afterok:$N $GPU 3dvision-experiments/run.slurm pi05_ego_human_oic human_oic 42)
echo "  Train1: $T1"

T2=$(sbatch --parsable --dependency=afterany:$T1 $GPU 3dvision-experiments/run.slurm pi05_ego_human_oic human_oic 42)
echo "  Train2 (resume): $T2"

echo "=== All OIC jobs submitted ==="
