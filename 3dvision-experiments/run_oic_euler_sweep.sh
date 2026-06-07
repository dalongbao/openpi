#!/bin/bash
# run_oic_euler_sweep.sh — submit short oic policy evals across candidate Euler orders.
# The correct order makes the observation (fk6) in-distribution -> smoothest / most
# directed EE trajectory. Compare results_oic_<ORDER>.csv jitter to pick the winner.
#
# Usage (login node):
#   bash ~/scripts/run_oic_euler_sweep.sh                  # default candidate set
#   bash ~/scripts/run_oic_euler_sweep.sh XYZ ZYX xyz zyx  # custom orders
set -euo pipefail

ORDERS=("$@"); [ ${#ORDERS[@]} -eq 0 ] && ORDERS=(ZYX xyz zyx XZY ZXY yzx)
SUBMIT=/cluster/scratch/$USER/submit.sh
STEPS="${NUM_STEPS:-600}"   # short: enough to see reach-vs-jitter (~20s sim)

cd /cluster/scratch/$USER/pi0_test
for O in "${ORDERS[@]}"; do
    echo "[sweep] submitting EULER_ORDER=$O (tag=_$O, steps=$STEPS)"
    EULER_ORDER="$O" RUN_TAG="_$O" NUM_STEPS="$STEPS" \
    sbatch --partition=gpu.4h --time=00:20:00 --mem-per-cpu=8G --cpus-per-task=8 \
           --gpus=rtx_4090:1 "$SUBMIT" eval_script_oic.py oic_human_2537ep
done
echo "[sweep] submitted ${#ORDERS[@]} jobs. Outputs -> results/oic_human_2537ep/results_oic_<ORDER>.csv"
