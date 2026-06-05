"""
eval_script_object_in_bowl_quick.py — fast smoke-test wrapper around
eval_script_object_in_bowl.py.

WHY: a full 3000-step eval is ~12-20 min, most of it the closed-loop run. When you
just want to see whether the arm heads in the RIGHT DIRECTION after a scene/camera
tweak, you don't need 60 s of sim. This runs the exact same eval with a much shorter
episode and separate output files, so it never clobbers the real run's artifacts.

It does NOT duplicate the eval logic — it reads the real script's source, patches a
couple of constants in memory, and execs it. So any fix to eval_script_object_in_bowl.py
(camera, scene, observation) is automatically picked up here too.

Tunables via env:
  NUM_STEPS   number of closed-loop steps (default 400 ≈ 8 s of sim)

Outputs (separate from the full run):
  /workspace/evaluation_quick.mp4
  /workspace/results_quick.csv

Submit exactly like the full one, just point submit.sh at this file:
  sbatch ... submit.sh eval_script_object_in_bowl_quick.py
  NUM_STEPS=200 sbatch ... submit.sh eval_script_object_in_bowl_quick.py
"""
import os
import pathlib

# The real script sits next to this one in the workspace.
_REAL = pathlib.Path("/workspace/eval_script_object_in_bowl.py")
_src = _REAL.read_text()

_quick_steps = os.environ.get("NUM_STEPS", "400")

# Patch step count + redirect outputs. Each replacement must hit exactly once;
# assert so we fail loudly if the real script's lines change (rather than silently
# running the full 3000-step version or overwriting the real artifacts).
_patches = [
    ("NUM_STEPS      = int(os.environ.get(\"NUM_STEPS\", \"3000\"))",
     f"NUM_STEPS      = {int(_quick_steps)}"),
    ("NUM_STEPS      = 3000",                       # fallback if env-var form not present yet
     f"NUM_STEPS      = {int(_quick_steps)}"),
    ('RESULTS_CSV    = "/workspace/results.csv"',
     'RESULTS_CSV    = "/workspace/results_quick.csv"'),
    ('VIDEO_PATH     = "/workspace/evaluation.mp4"',
     'VIDEO_PATH     = "/workspace/evaluation_quick.mp4"'),
]

_applied = 0
for _old, _new in _patches:
    if _old in _src:
        _src = _src.replace(_old, _new, 1)
        _applied += 1

# We must have patched the outputs (2) and at least one NUM_STEPS form (1) -> >=3.
if _applied < 3:
    raise RuntimeError(
        f"quick wrapper patched only {_applied} constants — the real script's lines "
        f"changed. Update _patches in eval_script_object_in_bowl_quick.py."
    )

print(f"[quick] Running eval with NUM_STEPS={_quick_steps}, "
      f"outputs -> evaluation_quick.mp4 / results_quick.csv")
exec(compile(_src, str(_REAL), "exec"), {"__name__": "__main__"})
