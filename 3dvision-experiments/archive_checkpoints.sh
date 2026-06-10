#!/bin/bash
# Preserve every-5k checkpoints for a run started under the default keep_period (where orbax keeps
# only the latest intermediate + the final 30k). Polls the live orbax checkpoint dir and COPIES each
# newly-committed step checkpoint to a sibling archive dir BEFORE orbax deletes it.
#
# Why copy (not move): moving the latest checkpoint out of the managed dir breaks `--resume` and
# orbax's metadata. We leave the original in place for orbax to manage; we just keep our own copy.
# Why it's safe to copy a numeric dir: orbax writes to a temp dir and atomically renames it to the
# pure-numeric `<step>` name on commit, so a numeric-named dir is fully written.
#
# Usage (run in tmux on a LOGIN node — it's just file I/O on scratch, no GPU needed):
#   tmux new -s archive
#   bash 3dvision-experiments/archive_checkpoints.sh <config> <exp_name> [poll_seconds]
#   e.g.  bash 3dvision-experiments/archive_checkpoints.sh pi05_egoverse rid64
#   (Ctrl-B then D to detach; reattach with `tmux attach -t archive`.)
#
# Stop it (Ctrl-C) once the 30000 checkpoint shows up in the archive; or just leave it, it's idle-light.
# Archive lands at /cluster/scratch/$USER/checkpoints_archive/<config>/<exp>/<step>/ .
# NOTE: scratch still auto-purges after ~15 days of no access -> push finals to HF for anything durable.
set -euo pipefail
shopt -s nullglob

CONFIG="${1:?usage: archive_checkpoints.sh <config> <exp_name> [poll_seconds]}"
EXP="${2:?usage: archive_checkpoints.sh <config> <exp_name> [poll_seconds]}"
POLL="${3:-1800}"   # 30 min: checkpoints survive ~5 h (5k steps @ ~3.5 s/it) before orbax deletes them

LIVE="/cluster/scratch/$USER/checkpoints/$CONFIG/$EXP"
ARCHIVE="/cluster/scratch/$USER/checkpoints_archive/$CONFIG/$EXP"
mkdir -p "$ARCHIVE"
echo "[archive] live   = $LIVE"
echo "[archive] archive= $ARCHIVE"
echo "[archive] poll   = ${POLL}s   (copies new numeric step dirs; Ctrl-C to stop)"

while true; do
  if [ -d "$LIVE" ]; then
    for d in "$LIVE"/*/; do
      step="$(basename "$d")"
      case "$step" in ''|*[!0-9]*) continue ;; esac   # only committed (pure-numeric) checkpoints
      [ -d "$ARCHIVE/$step" ] && continue             # already archived
      echo "[archive] $(date '+%F %T')  copying step $step ..."
      tmp="$ARCHIVE/.$step.partial"
      rm -rf "$tmp"
      if rsync -a "$d" "$tmp/"; then
        mv "$tmp" "$ARCHIVE/$step"                    # atomic publish into the archive
        echo "[archive] $(date '+%F %T')  saved step $step -> $ARCHIVE/$step"
      else
        echo "[archive] $(date '+%F %T')  WARN copy failed for $step (orbax may have deleted it mid-copy); skipping"
        rm -rf "$tmp"
      fi
    done
  fi
  sleep "$POLL"
done
