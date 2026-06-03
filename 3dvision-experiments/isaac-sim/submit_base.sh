#!/bin/bash
# Submit the BASELINE (un-fine-tuned) pi0.5 eval on the object_in_bowl scene.
# Usage (from /cluster/scratch/$USER after git pull + cp):
#   sbatch --partition=gpu.4h --time=00:30:00 --mem-per-cpu=8G --cpus-per-task=8 --gpus=rtx_3090:1 /cluster/scratch/$USER/submit_base.sh

export HTTP_PROXY=http://proxy.ethz.ch:3128
export HTTPS_PROXY=http://proxy.ethz.ch:3128
export PYTHONUNBUFFERED=1

export ISAAC_SIM_CACHE_DIR=/cluster/scratch/$USER/isaac_cache
mkdir -p "$ISAAC_SIM_CACHE_DIR"

WORKSPACE=/cluster/scratch/$USER/pi0_test
BASE_WEIGHTS=/cluster/work/cvg/data/Egoverse/pi05_base_jax/params

mkdir -p "$ISAAC_SIM_CACHE_DIR/kit"
mkdir -p "$ISAAC_SIM_CACHE_DIR/ov_home"

apptainer exec --nv \
    --bind "$WORKSPACE":/workspace \
    --bind "/cluster/scratch/$USER/openpi":/workspace/openpi \
    --bind "$BASE_WEIGHTS":/base_weights \
    --bind "$ISAAC_SIM_CACHE_DIR/kit":/isaac-sim/kit/cache \
    --bind "$ISAAC_SIM_CACHE_DIR/ov_home":/cluster/home/$USER \
    --bind "/cluster/scratch/$USER/isaac_packages":/isaac_packages \
    "/cluster/work/cvg/data/isaac-sim_4.5.0.sif" \
    /isaac-sim/python.sh /workspace/eval_script_base_object_in_bowl.py
