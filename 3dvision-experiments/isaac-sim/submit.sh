#!/bin/bash
# SLURM flags are passed on the command line, not as #SBATCH directives (cluster quirk).
#
# Smoke test (5 min):
#   sbatch --partition=gpu.4h --time=00:05:00 --mem-per-cpu=8G --cpus-per-task=8 --mem=64G --gpus=a100:1 3dvision-experiments/isaac-sim/submit.sh
#
# Full run (2 hr):
#   sbatch --partition=gpu.24h --time=02:00:00 --mem-per-cpu=8G --cpus-per-task=8 --mem=64G --gpus=a100:1 3dvision-experiments/isaac-sim/submit.sh
#
# Optional CLI args (any/all may be supplied AFTER the script path on the sbatch line):
#   --checkpoint <path>           Override checkpoint dir (default: $CHECKPOINTS/pi05_egoverse/test/29999, mapped to /checkpoints/pi05_egoverse/test/29999 inside container).
#   --seed <int>                  Override seed (default: 42).
#   --output-name <name>          If set, outputs go to /workspace/<name>/ instead of /workspace/ — useful when sweeping.
#   --perturbation-config <path>  Path (inside container, typically under /workspace/...) to a perturbation JSON.
#
# Example sweep invocation:
#   sbatch ... submit.sh --checkpoint /checkpoints/pi05_egoverse/test/14999 --seed 7 --output-name run_ckpt14999_seed7

export HTTP_PROXY=http://proxy.ethz.ch:3128
export HTTPS_PROXY=http://proxy.ethz.ch:3128
export PYTHONUNBUFFERED=1

# Isaac Sim shader cache must go to scratch (can be up to 10 GB)
export ISAAC_SIM_CACHE_DIR=/cluster/scratch/$USER/isaac_cache
mkdir -p "$ISAAC_SIM_CACHE_DIR"

WORKSPACE=/cluster/scratch/$USER/pi0_test
CHECKPOINTS=/cluster/work/cvg/data/rytsui/checkpoints

mkdir -p "$ISAAC_SIM_CACHE_DIR/kit"
mkdir -p "$ISAAC_SIM_CACHE_DIR/ov_home"

# --- Parse optional CLI overrides ---------------------------------------------
CHECKPOINT_ARG=""
SEED_ARG=""
OUTPUT_NAME=""
PERTURB_ARG=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)
            CHECKPOINT_ARG="$2"; shift 2 ;;
        --seed)
            SEED_ARG="$2"; shift 2 ;;
        --output-name)
            OUTPUT_NAME="$2"; shift 2 ;;
        --perturbation-config)
            PERTURB_ARG="$2"; shift 2 ;;
        *)
            # Pass anything else through to eval_script_1.py verbatim.
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Build the python-side arg list.
PY_ARGS=()
if [[ -n "$CHECKPOINT_ARG" ]]; then
    PY_ARGS+=(--checkpoint-dir "$CHECKPOINT_ARG")
fi
if [[ -n "$SEED_ARG" ]]; then
    PY_ARGS+=(--seed "$SEED_ARG")
fi
if [[ -n "$PERTURB_ARG" ]]; then
    PY_ARGS+=(--perturbation-config "$PERTURB_ARG")
fi
if [[ -n "$OUTPUT_NAME" ]]; then
    # Make a per-run subdir on the host so concurrent jobs don't stomp on each other.
    mkdir -p "$WORKSPACE/$OUTPUT_NAME"
    PY_ARGS+=(--output-dir "/workspace/$OUTPUT_NAME")
fi
# Append any caller-supplied flags (e.g. --record-external).
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    PY_ARGS+=("${EXTRA_ARGS[@]}")
fi

apptainer exec --nv \
    --bind "$WORKSPACE":/workspace \
    --bind "/cluster/scratch/$USER/openpi":/workspace/openpi \
    --bind "$CHECKPOINTS":/checkpoints \
    --bind "$ISAAC_SIM_CACHE_DIR/kit":/isaac-sim/kit/cache \
    --bind "$ISAAC_SIM_CACHE_DIR/ov_home":/cluster/home/$USER \
    --bind "/cluster/scratch/$USER/isaac_packages":/isaac_packages \
    "$WORKSPACE/isaac-sim_4.5.0.sif" \
    /isaac-sim/python.sh /workspace/eval_script_1.py "${PY_ARGS[@]}"
