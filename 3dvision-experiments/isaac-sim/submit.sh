#!/bin/bash
# SLURM flags are passed on the command line, not as #SBATCH directives (cluster quirk).
#
# First positional arg = which eval script to run (default eval_script_1.py).
# It must already be copied into $WORKSPACE (/cluster/scratch/$USER/pi0_test/).
#
# IK frame-sweep replay (this is the one to run next):
#   sbatch --partition=gpu.4h --time=00:30:00 --mem-per-cpu=8G --cpus-per-task=8 --gpus=rtx_3090:1 3dvision-experiments/isaac-sim/submit.sh eval_replay_ik.py
#
# Smoke test (5 min):
#   sbatch --partition=gpu.4h --time=00:05:00 --mem-per-cpu=8G --cpus-per-task=8 --gpus=a100:1 3dvision-experiments/isaac-sim/submit.sh
#
# Full run (2 hr):
#   sbatch --partition=gpu.24h --time=02:00:00 --mem-per-cpu=8G --cpus-per-task=8 --gpus=a100:1 3dvision-experiments/isaac-sim/submit.sh

export HTTP_PROXY=http://proxy.ethz.ch:3128
export HTTPS_PROXY=http://proxy.ethz.ch:3128
export PYTHONUNBUFFERED=1

# Isaac Sim shader cache must go to scratch (can be up to 10 GB)
export ISAAC_SIM_CACHE_DIR=/cluster/scratch/$USER/isaac_cache
mkdir -p "$ISAAC_SIM_CACHE_DIR"

WORKSPACE=/cluster/scratch/$USER/pi0_test
CHECKPOINTS=/cluster/work/cvg/data/rytsui/checkpoints
BASE_WEIGHTS=/cluster/work/cvg/data/Egoverse/pi05_base_jax   # for base-model variants (/base_weights)
EVAL_SCRIPT="${1:-eval_script_1.py}"   # which script in $WORKSPACE to run

# openpi clone (for src imports) lives in scratch for some users, $HOME for others. Auto-detect.
OPENPI_DIR="${OPENPI_DIR:-}"
if [ -z "$OPENPI_DIR" ]; then
    for cand in "/cluster/scratch/$USER/openpi" "$HOME/openpi"; do
        [ -d "$cand/src/openpi" ] && OPENPI_DIR="$cand" && break
    done
fi
echo "[submit] EVAL_SCRIPT=$EVAL_SCRIPT  OPENPI_DIR=$OPENPI_DIR  EE_FRAME=${EE_FRAME:-<unset>}"

mkdir -p "$ISAAC_SIM_CACHE_DIR/kit"
mkdir -p "$ISAAC_SIM_CACHE_DIR/ov_home"

apptainer exec --nv \
    --env "EE_FRAME=${EE_FRAME:-}" \
    --env "QUAT_WXYZ=${QUAT_WXYZ:-}" \
    --env "POSE_IN_BASE=${POSE_IN_BASE:-}" \
    --env "SCENE_FIDELITY=${SCENE_FIDELITY:-}" \
    --env "NUM_STEPS=${NUM_STEPS:-}" \
    --bind "$WORKSPACE":/workspace \
    --bind "$OPENPI_DIR":/workspace/openpi \
    --bind "$CHECKPOINTS":/checkpoints \
    --bind "$BASE_WEIGHTS":/base_weights \
    --bind "$ISAAC_SIM_CACHE_DIR/kit":/isaac-sim/kit/cache \
    --bind "$ISAAC_SIM_CACHE_DIR/ov_home":/cluster/home/$USER \
    --bind "/cluster/scratch/$USER/isaac_packages":/isaac_packages \
    "/cluster/work/cvg/data/isaac-sim_4.5.0.sif" \
    /isaac-sim/python.sh "/workspace/$EVAL_SCRIPT"
