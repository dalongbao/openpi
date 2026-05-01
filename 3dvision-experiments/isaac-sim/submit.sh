#!/bin/bash
# SLURM flags are passed on the command line, not as #SBATCH directives (cluster quirk).
#
# Smoke test (5 min):
#   sbatch --partition=gpu.4h --time=00:05:00 --mem-per-cpu=8G --cpus-per-task=8 --mem=64G --gpus=a100:1 3dvision-experiments/isaac-sim/submit.sh
#
# Full run (2 hr):
#   sbatch --partition=gpu.24h --time=02:00:00 --mem-per-cpu=8G --cpus-per-task=8 --mem=64G --gpus=a100:1 3dvision-experiments/isaac-sim/submit.sh

export HTTP_PROXY=http://proxy.ethz.ch:3128
export HTTPS_PROXY=http://proxy.ethz.ch:3128
export PYTHONUNBUFFERED=1

# Isaac Sim shader cache must go to scratch (can be up to 10 GB)
export ISAAC_SIM_CACHE_DIR=/cluster/scratch/$USER/isaac_cache
mkdir -p "$ISAAC_SIM_CACHE_DIR"

WORKSPACE=/cluster/scratch/$USER/pi0_test
CHECKPOINTS=/cluster/work/cvg/data/rytsui/checkpoints

mkdir -p "$ISAAC_SIM_CACHE_DIR/kit"

apptainer exec --nv \
    --bind "$WORKSPACE":/workspace \
    --bind "/cluster/scratch/$USER/openpi":/workspace/openpi \
    --bind "$CHECKPOINTS":/checkpoints \
    --bind "$ISAAC_SIM_CACHE_DIR/kit":/isaac-sim/kit/cache \
    --bind "/cluster/scratch/$USER/isaac_packages":/isaac_packages \
    "$WORKSPACE/isaac-sim_4.5.0.sif" \
    /isaac-sim/python.sh /workspace/eval_script_1.py
