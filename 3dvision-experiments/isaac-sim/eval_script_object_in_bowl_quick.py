"""
eval_script_object_in_bowl_quick.py — fast smoke test for the object_in_bowl policy eval.

A full 3000-step eval is ~12-20 min. When you just want to see whether the arm heads
in the RIGHT DIRECTION after a scene/camera/model change, a short episode is enough.

This sets short-run defaults (NUM_STEPS=400, RUN_TAG=_quick) and then runs the real
eval_script_object_in_bowl.py — single source of truth, so every fix to the real eval
is picked up automatically and outputs never clobber the full run (they land in
results/<MODEL_NAME>/evaluation_quick.mp4 / results_quick.csv).

Submit it like the full eval (model name as the 2nd arg picks the checkpoint):
  sbatch ... submit.sh eval_script_object_in_bowl_quick.py oic_human_2537ep
  NUM_STEPS=200 sbatch ... submit.sh eval_script_object_in_bowl_quick.py egoverse_5ep
"""
import os
import pathlib

# Defaults only — don't override values the submitter explicitly set (and treat an
# empty forwarded value as "unset", since submit.sh forwards "" when a var is absent).
if not os.environ.get("NUM_STEPS"):
    os.environ["NUM_STEPS"] = "400"
if not os.environ.get("RUN_TAG"):
    os.environ["RUN_TAG"] = "_quick"

print(f"[quick] NUM_STEPS={os.environ['NUM_STEPS']} RUN_TAG={os.environ['RUN_TAG']}")

_real = pathlib.Path("/workspace/eval_script_object_in_bowl.py")
exec(compile(_real.read_text(), str(_real), "exec"), {"__name__": "__main__"})
