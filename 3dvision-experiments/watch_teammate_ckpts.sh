#!/bin/bash
# watch_teammate_ckpts.sh — poll teammates' scratch for accessible+complete 20k checkpoints
# and auto-submit the eval for each condition exactly once.
#
# Why: teammates (rytsui, kdoman) haven't chmod'd yet and their runs may still be short of
# 20k. This watches for the moment BOTH become true (readable AND step-20000 complete) and
# fires run_ablation_eval.slurm, so the eval fleet fills in overnight without babysitting.
#
# Run on a LOGIN node inside tmux (survives laptop sleep):
#   tmux new -s ckptwatch
#   bash ~/openpi/3dvision-experiments/watch_teammate_ckpts.sh
#   # Ctrl+B then D to detach;  tmux attach -t ckptwatch  to check on it
#
# Idempotent: skips a condition if its result JSON exists or it was already submitted by
# this script (marker files in $STATE_DIR). Exits when every condition is handled.
# Env overrides: USERS, STEP, POLL_SECS, EVAL_TIME, EVAL_GPU, RESULTS_DIR.

set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM="$REPO_DIR/3dvision-experiments/run_ablation_eval.slurm"
HELD_OUT="$REPO_DIR/held_out_rid.txt"
RESULTS_DIR="${RESULTS_DIR:-/cluster/scratch/$USER/ablation_results}"
STATE_DIR="${STATE_DIR:-$HOME/.ckptwatch_submitted}"
USERS="${USERS:-rytsui kdoman}"
STEP="${STEP:-20000}"
POLL_SECS="${POLL_SECS:-1800}"
EVAL_TIME="${EVAL_TIME:-01:30:00}"
EVAL_GPU="${EVAL_GPU:-rtx_4090}"

# config-dir name -> "condition state_dim" (mirrors eval_manifest.tsv; hid is the 6-dim model)
declare -A WANT=(
  [pi05_egoverse_n5]="rid5 24"
  [pi05_egoverse_n15]="rid15 24"
  [pi05_ego_mix_oic_n5]="mix5 24"
  [pi05_ego_mix_oic_n15]="mix15 24"
  [pi05_ego_human_oic]="hid 6"
)

mkdir -p "$STATE_DIR" "$RESULTS_DIR"
log() { echo "[$(date '+%F %T')] $*"; }

log "watching users [$USERS] for step-$STEP checkpoints of: ${!WANT[*]}"
log "poll every ${POLL_SECS}s; submissions marked in $STATE_DIR; results in $RESULTS_DIR"

while :; do
  pending=0
  for cfg in "${!WANT[@]}"; do
    read -r cond sd <<< "${WANT[$cfg]}"
    # already handled?
    [[ -f "$RESULTS_DIR/$cond.json" ]] && continue
    [[ -f "$STATE_DIR/$cond" ]] && { pending=$((pending+1)); continue; }  # submitted, job not done yet

    found=""
    for u in $USERS; do
      for d in /cluster/scratch/"$u"/checkpoints/"$cfg"/*/"$STEP"; do
        [[ -d "$d" ]] || continue                              # no access yet, or step not reached
        if [[ -f "$d/_CHECKPOINT_METADATA" && -d "$d/params" && -d "$d/assets" ]]; then
          found="$d"; break 2
        else
          log "[$cond] $d visible but INCOMPLETE (mid-save?) — will retry"
        fi
      done
    done

    if [[ -z "$found" ]]; then pending=$((pending+1)); continue; fi

    log "[$cond] COMPLETE checkpoint found: $found — submitting eval"
    if sbatch --export=ALL,UV_FROZEN=1 --time="$EVAL_TIME" --mem-per-cpu=16G --cpus-per-task=8 \
              --gpus="$EVAL_GPU:1" --job-name=abeval --dependency=singleton "$SLURM" \
              --condition "$cond" --config-name "$cfg" --checkpoint-dir "$found" \
              --state-dim "$sd" --held-out-file "$HELD_OUT" --output-dir "$RESULTS_DIR"; then
      echo "$found" > "$STATE_DIR/$cond"
    else
      log "[$cond] sbatch FAILED — will retry next poll"; pending=$((pending+1))
    fi
  done

  if [[ $pending -eq 0 ]]; then log "all conditions submitted or done — exiting"; break; fi
  log "$pending condition(s) still pending; sleeping ${POLL_SECS}s"
  sleep "$POLL_SECS"
done
