#!/bin/bash
# run_all_evals.sh — one-command offline eval of every ablation condition (manifest-driven).
#
# Usage (from ~/openpi on Euler):
#   bash 3dvision-experiments/run_all_evals.sh gt-check            # 0) calibrate the harness (no GPU)
#   bash 3dvision-experiments/run_all_evals.sh verify              # 1) preflight: ckpts + norm-stats provenance
#   bash 3dvision-experiments/run_all_evals.sh submit [--dry-run]  # 2) submit one eval job per condition
#   bash 3dvision-experiments/run_all_evals.sh sweep <condition>   # 3) optional: eval EVERY retained step
#   bash 3dvision-experiments/run_all_evals.sh aggregate           # 4) cross-condition table + paired deltas
#
# Flags for submit/sweep: --dry-run (print sbatch lines only), --force (re-eval even if the
# JSON exists), --parallel (drop the singleton serialization; default = one eval at a time).
#
# Env overrides: MANIFEST, RESULTS_DIR, EVAL_GPU (default rtx_4090), EVAL_TIME (default 04:00:00).
#
# Design notes:
# - Idempotent: a condition whose JSON already exists in RESULTS_DIR is skipped (use --force).
# - Serialized by default via --dependency=singleton (plays nice with the 2-job courtesy cap;
#   10 conditions x ~1h still finishes overnight).
# - Norm stats: ablation_eval.py now defaults to the snapshot INSIDE each checkpoint
#   (<ckpt>/assets/<repo_id>) so train/eval normalizers always match, whatever a teammate
#   trained with. `verify` prints the md5 provenance table for the report.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${MANIFEST:-$REPO_DIR/3dvision-experiments/eval_manifest.tsv}"
RESULTS_DIR="${RESULTS_DIR:-/cluster/scratch/$USER/ablation_results}"
HELD_OUT="${HELD_OUT:-$REPO_DIR/held_out_rid.txt}"   # 12-ep eval set (repo root, committed)
EVAL_GPU="${EVAL_GPU:-rtx_4090}"
EVAL_TIME="${EVAL_TIME:-04:00:00}"
SLURM="$REPO_DIR/3dvision-experiments/run_ablation_eval.slurm"

CMD="${1:-submit}"; shift || true
DRY=0; FORCE=0; PARALLEL=0; SWEEP_COND=""
for a in "$@"; do case "$a" in
  --dry-run) DRY=1 ;; --force) FORCE=1 ;; --parallel) PARALLEL=1 ;;
  *) SWEEP_COND="$a" ;;
esac; done

[ -f "$MANIFEST" ] || { echo "ERROR: manifest not found: $MANIFEST"; exit 1; }
[ -f "$HELD_OUT" ] || { echo "ERROR: held-out file not found: $HELD_OUT"; exit 1; }

# read manifest -> arrays (expand $USER; skip comments/blanks)
CONDS=(); CONFIGS=(); CKPTS=(); SDIMS=(); EXTRAS=()
while read -r cond config ckpt sd extra; do
  [[ -z "$cond" || "$cond" == \#* ]] && continue
  CONDS+=("$cond"); CONFIGS+=("$config"); CKPTS+=("${ckpt//\$USER/$USER}"); SDIMS+=("$sd"); EXTRAS+=("${extra:-}")
done < <(sed -e "s/\${USER}/$USER/g" "$MANIFEST")

is_base() { [[ "${EXTRAS[$1]}" == *no-finetuned* ]]; }

preflight() {  # $1 = row index; returns 0 = OK to submit, 1 = skip with reason printed
  local i=$1 cond="${CONDS[$i]}" ckpt="${CKPTS[$i]}"
  if [[ "$ckpt" == *EDITME* ]]; then echo "  [$cond] SKIP: checkpoint owner not filled in the manifest"; return 1; fi
  if [[ ! -d "$ckpt" ]]; then echo "  [$cond] SKIP: checkpoint dir missing: $ckpt"; return 1; fi
  if [[ ! -d "$ckpt/params" ]]; then echo "  [$cond] SKIP: $ckpt has no params/ (incomplete checkpoint?)"; return 1; fi
  if ! is_base "$i" && ! compgen -G "$ckpt/assets/*/*/norm_stats.json" >/dev/null; then
    echo "  [$cond] WARN: no norm-stats snapshot in checkpoint — eval will fall back to the local checkout's assets (verify provenance!)"
  fi
  if [[ $FORCE -eq 0 && -f "$RESULTS_DIR/$cond.json" ]]; then echo "  [$cond] SKIP: $RESULTS_DIR/$cond.json exists (--force to redo)"; return 1; fi
  return 0
}

submit_one() {  # $1 = row index, $2 = condition name override (sweep), $3 = ckpt override (sweep)
  local i=$1 cond="${2:-${CONDS[$i]}}" ckpt="${3:-${CKPTS[$i]}}"
  local dep=(); [[ $PARALLEL -eq 0 ]] && dep=(--dependency=singleton)
  local args=(--condition "$cond" --config-name "${CONFIGS[$i]}" --checkpoint-dir "$ckpt"
              --state-dim "${SDIMS[$i]}" --held-out-file "$HELD_OUT" --output-dir "$RESULTS_DIR" ${EXTRAS[$i]})
  local cmd=(sbatch --export=ALL,UV_FROZEN=1 --time="$EVAL_TIME" --mem-per-cpu=16G --cpus-per-task=8
             --gpus="$EVAL_GPU:1" --job-name=abeval "${dep[@]}" "$SLURM" "${args[@]}")
  if [[ $DRY -eq 1 ]]; then echo "  [DRY] ${cmd[*]}"; else echo "  [$cond] ${cmd[*]}"; "${cmd[@]}"; fi
}

case "$CMD" in
  gt-check)
    # Harness self-test on the GT actions themselves — no GPU, login node OK (~2 min).
    # Expect ordered_success ~= 1.0 and reach errors of a few mm; this calibrates SUCCESS_THRESH.
    cd "$REPO_DIR" && UV_FROZEN=1 uv run python 3dvision-experiments/ablation_eval.py \
      --gt-check --held-out-file "$HELD_OUT" --output-dir "$RESULTS_DIR"
    ;;

  verify)
    echo "=== preflight: checkpoints + norm-stats provenance ==="
    can_rid="$REPO_DIR/assets/pi05_egoverse/egoverse/all/norm_stats.json"
    can_mix="$REPO_DIR/assets/pi05_ego_mix_oic/egoverse/oic_mix/norm_stats.json"
    md_rid=$([ -f "$can_rid" ] && md5sum "$can_rid" | cut -d' ' -f1 || echo "-")
    md_mix=$([ -f "$can_mix" ] && md5sum "$can_mix" | cut -d' ' -f1 || echo "-")
    echo "canonical rid (egoverse/all)    md5: $md_rid"
    echo "canonical mix (egoverse/oic_mix) md5: $md_mix"
    echo
    for i in "${!CONDS[@]}"; do
      cond="${CONDS[$i]}"; ckpt="${CKPTS[$i]}"
      if [[ "$ckpt" == *EDITME* ]]; then echo "[$cond] EDITME — owner not filled"; continue; fi
      if [[ ! -d "$ckpt" ]]; then echo "[$cond] MISSING: $ckpt"; continue; fi
      steps=$(ls -d "$(dirname "$ckpt")"/[0-9]* 2>/dev/null | xargs -n1 basename 2>/dev/null | sort -n | tr '\n' ' ')
      ns=$(compgen -G "$ckpt/assets/*/*/norm_stats.json" | head -1 || true)
      if [[ -n "${ns:-}" ]]; then
        md=$(md5sum "$ns" | cut -d' ' -f1)
        tag="RECOMPUTED/OTHER"
        [[ "$md" == "$md_rid" ]] && tag="== canonical rid"
        [[ "$md" == "$md_mix" ]] && tag="== canonical mix"
        echo "[$cond] steps: ${steps:-none}| norm_stats md5: $md ($tag)"
      else
        echo "[$cond] steps: ${steps:-none}| norm_stats: NONE IN CKPT $(is_base "$i" && echo '(base — expected)' || echo '(WARN)')"
      fi
    done
    ;;

  submit)
    [[ $DRY -eq 1 ]] || mkdir -p "$RESULTS_DIR"
    echo "=== submitting evals (results -> $RESULTS_DIR; $([ $PARALLEL -eq 1 ] && echo parallel || echo serialized)) ==="
    for i in "${!CONDS[@]}"; do preflight "$i" && submit_one "$i" || true; done
    echo "Monitor: squeue -u \$USER -n abeval ; then: bash $0 aggregate"
    ;;

  sweep)
    [[ -n "$SWEEP_COND" ]] || { echo "usage: $0 sweep <condition>"; exit 1; }
    [[ $DRY -eq 1 ]] || mkdir -p "$RESULTS_DIR"
    for i in "${!CONDS[@]}"; do
      [[ "${CONDS[$i]}" == "$SWEEP_COND" ]] || continue
      parent="$(dirname "${CKPTS[$i]}")"
      for d in "$parent"/[0-9]*; do
        [[ -d "$d/params" ]] || continue
        step="$(basename "$d")"; cond="${SWEEP_COND}_s${step}"
        [[ $FORCE -eq 0 && -f "$RESULTS_DIR/$cond.json" ]] && { echo "  [$cond] exists, skip"; continue; }
        submit_one "$i" "$cond" "$d"
      done
      exit 0
    done
    echo "ERROR: condition '$SWEEP_COND' not in manifest"; exit 1
    ;;

  aggregate)
    cd "$REPO_DIR" && UV_FROZEN=1 uv run python 3dvision-experiments/aggregate_ablation.py "$RESULTS_DIR"/*.json
    ;;

  *) echo "usage: $0 {gt-check|verify|submit|sweep <cond>|aggregate} [--dry-run|--force|--parallel]"; exit 1 ;;
esac
