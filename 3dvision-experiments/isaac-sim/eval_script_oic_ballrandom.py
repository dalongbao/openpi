"""
eval_script_oic_ballrandom.py — move-the-ball VISUAL-TRACKING test for the oic policy.

Places the ball at a small SEEDED offset from nominal (BALL_JITTER_SEED) while the frame
calibration stays anchored to the NOMINAL ball position. So with a FIXED transform and a
MOVED ball, the arm reaches the new ball ONLY if the policy genuinely tracks it from vision
— not if it's just replaying the calibrated demo trajectory.

Read the result from the side video + the `[ball] pos=...` line: does the arm head to the
moved ball, or to where the ball nominally was? Re-run with different BALL_JITTER_SEED values
to sample a few offsets.

Runs the real eval_script_oic.py (single source of truth). Submit (calibration ON):
  BALL_JITTER_SEED=3 OIC_CALIBRATE=1 SCENE_FIDELITY=1 EGOCENTRIC=1 EGO_HIDE_ARM=1 \
    sbatch ... submit.sh eval_script_oic_ballrandom.py oic_human_2537ep
"""
import os
import pathlib

_seed = os.environ.get("BALL_JITTER_SEED") or "0"
os.environ["BALL_JITTER_SEED"] = _seed
if not os.environ.get("RUN_TAG"):
    os.environ["RUN_TAG"] = f"_ball{_seed}"
if not os.environ.get("NUM_STEPS"):
    os.environ["NUM_STEPS"] = "1000"

print(f"[ballrand] BALL_JITTER_SEED={_seed}  RUN_TAG={os.environ['RUN_TAG']}  NUM_STEPS={os.environ['NUM_STEPS']}")

_real = pathlib.Path("/workspace/eval_script_oic.py")
exec(compile(_real.read_text(), str(_real), "exec"), {"__name__": "__main__"})
