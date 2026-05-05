#!/bin/bash
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export JAX_PLATFORMS=cpu
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="/cluster/scratch/rytsui/hf_cache"
export HF_DATASETS_CACHE="/cluster/scratch/rytsui/hf_cache/datasets"
export HF_LEROBOT_HOME="/cluster/work/cvg/data/Egoverse/lerobot_egoverse"
uv run python scripts/compute_norm_stats.py --config-name "${1:-pi05_ego_human_bag_grocery}" --max-frames "${2:-5000}"
