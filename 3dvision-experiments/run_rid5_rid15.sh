#!/usr/bin/env bash
# End-to-end: download ChinLR/object_in_bowl_64 -> norm stats -> train rid5 + rid15 -> upload to HF.
# Run on Euler login node:  nohup ./run_rid5_rid15.sh > run.log 2>&1 &
# Then `tail -f run.log` whenever your SSH comes back.

set -euo pipefail

# ---- config ----
HF_DATASET_REPO="ChinLR/object_in_bowl_64"
HF_USER="zhonglongbao"
DATA_HOME="/cluster/scratch/$USER/lerobot"
DATA_DIR="$DATA_HOME/egoverse/all"
HF_CACHE="/cluster/scratch/$USER/hf_cache"
CKPT_BASE="/cluster/scratch/$USER/checkpoints"
SEED=42

export DATA_HOME HF_LEROBOT_HOME="$DATA_HOME" HF_HOME="$HF_CACHE" HF_DATASETS_CACHE="$HF_CACHE/datasets"

cd "$HOME/openpi"

ts() { date +"[%Y-%m-%d %H:%M:%S]"; }
log() { echo "$(ts) $*"; }

# ---- 1. download dataset (~75 GB) ----
log "STEP 1: download $HF_DATASET_REPO -> $DATA_DIR"
mkdir -p "$DATA_DIR"
if [[ -f "$DATA_DIR/meta/info.json" ]]; then
  EPS=$(python3 -c "import json; print(json.load(open('$DATA_DIR/meta/info.json')).get('total_episodes', 0))")
  log "  existing dataset found with $EPS episodes; re-syncing to be safe"
fi
huggingface-cli download "$HF_DATASET_REPO" --repo-type dataset --local-dir "$DATA_DIR"

# ---- 2. verify episode count ----
log "STEP 2: verify dataset"
EPS=$(python3 -c "import json; print(json.load(open('$DATA_DIR/meta/info.json'))['total_episodes'])")
log "  total_episodes = $EPS"
if [[ "$EPS" -lt 64 ]]; then
  log "ABORT: expected 64 episodes, got $EPS"
  exit 1
fi

# ---- 3. norm stats (both configs) ----
for CFG in pi05_egoverse_n5 pi05_egoverse_n15; do
  log "STEP 3: norm stats for $CFG"
  UV_FROZEN=1 uv run python scripts/compute_norm_stats.py --config-name "$CFG"
done

# ---- 4. submit both training jobs ----
log "STEP 4: submit training jobs"
SBATCH_ARGS="--partition=gpu.120h --time=120:00:00 --mem-per-cpu=16G --cpus-per-task=8 --gpus=a100:1"

JID_N5=$(sbatch --parsable $SBATCH_ARGS 3dvision-experiments/run_train_shared.slurm pi05_egoverse_n5  rid5  $SEED)
log "  pi05_egoverse_n5  -> job $JID_N5"

JID_N15=$(sbatch --parsable $SBATCH_ARGS 3dvision-experiments/run_train_shared.slurm pi05_egoverse_n15 rid15 $SEED)
log "  pi05_egoverse_n15 -> job $JID_N15"

# ---- 5. wait for both jobs ----
wait_job() {
  local jid=$1 name=$2
  log "STEP 5: waiting on $name (job $jid)"
  while squeue -j "$jid" -h -o "%T" 2>/dev/null | grep -q .; do
    sleep 120
  done
  local state
  state=$(sacct -j "${jid}" --format=State -n -X | head -1 | awk '{print $1}')
  log "  $name (job $jid) final state: $state"
  if [[ "$state" != "COMPLETED" ]]; then
    log "  WARNING: $name did not COMPLETE — skipping upload"
    return 1
  fi
}

UPLOAD_N5=1;  wait_job "$JID_N5"  pi05_egoverse_n5  || UPLOAD_N5=0
UPLOAD_N15=1; wait_job "$JID_N15" pi05_egoverse_n15 || UPLOAD_N15=0

# ---- 6. upload checkpoints to HF (public) ----
upload_ckpt() {
  local cfg=$1 exp=$2
  local ckpt="$CKPT_BASE/$cfg/$exp/30000"
  log "STEP 6: upload $cfg from $ckpt -> $HF_USER/$cfg"
  if [[ ! -d "$ckpt" ]]; then
    log "  ABORT upload: $ckpt missing"
    return 1
  fi
  huggingface-cli upload --repo-type model "$HF_USER/$cfg" "$ckpt" .
  log "  uploaded $HF_USER/$cfg"
}

[[ "$UPLOAD_N5"  -eq 1 ]] && upload_ckpt pi05_egoverse_n5  rid5  || true
[[ "$UPLOAD_N15" -eq 1 ]] && upload_ckpt pi05_egoverse_n15 rid15 || true

log "DONE"
