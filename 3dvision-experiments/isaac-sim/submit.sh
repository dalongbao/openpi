#!/bin/bash
# SLURM flags are passed on the command line, not as #SBATCH directives (cluster quirk).
#
# Args:  submit.sh <eval_script> [model_name]
#   <eval_script>  which .py in $WORKSPACE to run (default eval_script_1.py)
#   [model_name]   optional: sources $WORKSPACE/models/<model_name>.env, which sets
#                  CONFIG_NAME / CHECKPOINT_DIR / NORM_STATS_DIR / PROMPT / MODEL_NAME.
#                  This is how you pick WHICH trained model to evaluate.
#
# Run the 5-episode model:
#   sbatch --partition=gpu.4h --time=00:30:00 --mem-per-cpu=8G --cpus-per-task=8 --gpus=rtx_3090:1 submit.sh eval_script_object_in_bowl.py egoverse_5ep
#
# Run the 2537-episode oic_human model:
#   sbatch --partition=gpu.4h --time=00:30:00 --mem-per-cpu=8G --cpus-per-task=8 --gpus=rtx_3090:1 submit.sh eval_script_object_in_bowl.py oic_human_2537ep
#
# Quick smoke test of a model:
#   sbatch --partition=gpu.4h --time=00:15:00 --mem-per-cpu=8G --cpus-per-task=8 --gpus=rtx_3090:1 submit.sh eval_script_object_in_bowl_quick.py oic_human_2537ep
#
# Scene preview (no policy): submit.sh scene_preview.py
# GT replay:                 submit.sh eval_replay_ik.py

export HTTP_PROXY=http://proxy.ethz.ch:3128
export HTTPS_PROXY=http://proxy.ethz.ch:3128
export PYTHONUNBUFFERED=1

# Isaac Sim shader cache must go to scratch (can be up to 10 GB)
export ISAAC_SIM_CACHE_DIR=/cluster/scratch/$USER/isaac_cache
mkdir -p "$ISAAC_SIM_CACHE_DIR"

WORKSPACE=/cluster/scratch/$USER/pi0_test
CHECKPOINTS=/cluster/work/cvg/data/rytsui/checkpoints
USER_CHECKPOINTS=/cluster/scratch/$USER/checkpoints          # this user's OWN training checkpoints (rid30, mix5, ...) -> /user_checkpoints
SHARED_CHECKPOINTS=${SHARED_CHECKPOINTS:-/cluster/scratch/lichin/checkpoints}  # teammate-owned baselines (rid64, mix64) -> /shared_checkpoints
BASE_WEIGHTS=/cluster/work/cvg/data/Egoverse/pi05_base_jax   # for base-model variants (/base_weights)
EVAL_SCRIPT="${1:-eval_script_1.py}"   # which script in $WORKSPACE to run
MODEL="${2:-}"                          # optional model preset -> models/<MODEL>.env

# Source the model preset so CONFIG_NAME/CHECKPOINT_DIR/NORM_STATS_DIR/PROMPT/MODEL_NAME
# are set, then forwarded into the container below. Values are container paths (valid
# only inside apptainer) — that's fine, they're just strings until used there.
if [ -n "$MODEL" ]; then
    MODEL_ENV="$WORKSPACE/models/$MODEL.env"
    if [ -f "$MODEL_ENV" ]; then
        echo "[submit] sourcing model preset: $MODEL_ENV"
        set -a; . "$MODEL_ENV"; set +a
    else
        echo "[submit] ERROR: model preset not found: $MODEL_ENV"; exit 1
    fi
fi

# openpi clone (for src imports) lives in scratch for some users, $HOME for others. Auto-detect.
OPENPI_DIR="${OPENPI_DIR:-}"
if [ -z "$OPENPI_DIR" ]; then
    for cand in "/cluster/scratch/$USER/openpi" "$HOME/openpi"; do
        [ -d "$cand/src/openpi" ] && OPENPI_DIR="$cand" && break
    done
fi
echo "[submit] EVAL_SCRIPT=$EVAL_SCRIPT  MODEL=${MODEL:-<none>}  MODEL_NAME=${MODEL_NAME:-<unset>}  CONFIG_NAME=${CONFIG_NAME:-<unset>}"

mkdir -p "$ISAAC_SIM_CACHE_DIR/kit"
mkdir -p "$ISAAC_SIM_CACHE_DIR/ov_home"

# Bind this user's own scratch checkpoints at /user_checkpoints (only if present), so model
# presets trained by THIS user (e.g. rid30) resolve without touching rytsui's /checkpoints.
EXTRA_BINDS=()
[ -d "$USER_CHECKPOINTS" ] && EXTRA_BINDS+=(--bind "$USER_CHECKPOINTS":/user_checkpoints)
[ -d "$SHARED_CHECKPOINTS" ] && EXTRA_BINDS+=(--bind "$SHARED_CHECKPOINTS":/shared_checkpoints)

apptainer exec --nv \
    --env "EE_FRAME=${EE_FRAME:-}" \
    --env "QUAT_WXYZ=${QUAT_WXYZ:-}" \
    --env "POSE_IN_BASE=${POSE_IN_BASE:-}" \
    --env "SCENE_FIDELITY=${SCENE_FIDELITY:-}" \
    --env "NUM_STEPS=${NUM_STEPS:-}" \
    --env "EULER_ORDER=${EULER_ORDER:-}" \
    --env "OIC_POS_MAP=${OIC_POS_MAP:-}" \
    --env "OIC_CALIBRATE=${OIC_CALIBRATE:-}" \
    --env "OIC_ANCHOR_FRAMES=${OIC_ANCHOR_FRAMES:-}" \
    --env "BALL_JITTER_SEED=${BALL_JITTER_SEED:-}" \
    --env "BALL_RADIUS=${BALL_RADIUS:-}" \
    --env "ACTION_SMOOTH_ALPHA=${ACTION_SMOOTH_ALPHA:-}" \
    --env "RID_CALIBRATE=${RID_CALIBRATE:-}" \
    --env "RID_CALIB_ROT=${RID_CALIB_ROT:-}" \
    --env "RID_CALIB_SCALE=${RID_CALIB_SCALE:-}" \
    --env "RID_CALIB_SCALE_MUL=${RID_CALIB_SCALE_MUL:-}" \
    --env "RID_DEMO_NPZ=${RID_DEMO_NPZ:-}" \
    --env "RID_ANCHOR_FRAMES=${RID_ANCHOR_FRAMES:-}" \
    --env "REC_CAM_REAIM=${REC_CAM_REAIM:-}" \
    --env "BASE_MODEL=${BASE_MODEL:-}" \
    --env "EGOCENTRIC=${EGOCENTRIC:-}" \
    --env "EGO_HIDE_ARM=${EGO_HIDE_ARM:-}" \
    --env "RUN_TAG=${RUN_TAG:-}" \
    --env "CONFIG_NAME=${CONFIG_NAME:-}" \
    --env "CHECKPOINT_DIR=${CHECKPOINT_DIR:-}" \
    --env "NORM_STATS_DIR=${NORM_STATS_DIR:-}" \
    --env "PROMPT=${PROMPT:-}" \
    --env "MODEL_NAME=${MODEL_NAME:-}" \
    --bind "$WORKSPACE":/workspace \
    --bind "$OPENPI_DIR":/workspace/openpi \
    --bind "$CHECKPOINTS":/checkpoints \
    "${EXTRA_BINDS[@]}" \
    --bind "$BASE_WEIGHTS":/base_weights \
    --bind "$ISAAC_SIM_CACHE_DIR/kit":/isaac-sim/kit/cache \
    --bind "$ISAAC_SIM_CACHE_DIR/ov_home":/cluster/home/$USER \
    --bind "/cluster/scratch/$USER/isaac_packages":/isaac_packages \
    "/cluster/work/cvg/data/isaac-sim_4.5.0.sif" \
    /isaac-sim/python.sh "/workspace/$EVAL_SCRIPT"
