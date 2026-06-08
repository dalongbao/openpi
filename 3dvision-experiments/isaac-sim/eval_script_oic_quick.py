"""
eval_script_oic_quick.py — fast 5-second smoke test for the oic policy eval.

Boot (~73s) + JAX JIT (~49s) are fixed costs; this just shortens the closed loop to
NUM_STEPS=250 (= 5 s at 50 Hz) so the whole job is ~3 min instead of ~7. Enough to see
whether the arm heads in the RIGHT DIRECTION toward the ball after a scene/camera change,
and to surface any setup crash quickly.

Runs the real eval_script_oic.py (single source of truth), so every fix there is picked up
automatically; outputs land in results/<MODEL_NAME>/evaluation_oic_quick.mp4 and never
clobber a full run.

Submit it like the full eval:
  SCENE_FIDELITY=1 EGOCENTRIC=1 EGO_HIDE_ARM=1 sbatch ... submit.sh eval_script_oic_quick.py oic_human_2537ep
"""
import os
import pathlib

# Defaults only — don't override values the submitter explicitly set (submit.sh forwards
# "" for absent vars, so treat empty as unset).
if not os.environ.get("NUM_STEPS"):
    os.environ["NUM_STEPS"] = "250"          # 5 s at 50 Hz
if not os.environ.get("RUN_TAG"):
    os.environ["RUN_TAG"] = "_quick"

print(f"[quick] NUM_STEPS={os.environ['NUM_STEPS']} RUN_TAG={os.environ['RUN_TAG']}")

_real = pathlib.Path("/workspace/eval_script_oic.py")
exec(compile(_real.read_text(), str(_real), "exec"), {"__name__": "__main__"})
