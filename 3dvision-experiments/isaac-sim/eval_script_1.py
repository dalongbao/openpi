"""eval_script_1.py — thin shim around ``eval.runner.main``.

The legacy monolithic script has been refactored into the ``eval`` package
sitting next to this file. This shim preserves the exact entry-point path
that ``submit.sh`` invokes (``/workspace/eval_script_1.py``) so existing
SLURM one-liners keep working without changes.

To run with custom flags, either edit ``submit.sh`` or invoke this file
directly with the same CLI as ``eval.runner``. See ``eval/README.md``.
"""

import os
import sys

# Make the sibling ``eval`` package importable when this file is invoked
# as a standalone script (e.g. ``/isaac-sim/python.sh eval_script_1.py``).
# We look in two places, in order:
#   1. The directory containing this file (e.g. /workspace/ on Euler).
#   2. The repo's isaac-sim dir under the bind-mounted openpi clone, so
#      callers can copy just this shim and pull the package from git.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ISAAC_DIR = "/workspace/openpi/3dvision-experiments/isaac-sim"
for _p in (_HERE, _REPO_ISAAC_DIR):
    if os.path.isdir(os.path.join(_p, "eval")) and _p not in sys.path:
        sys.path.insert(0, _p)

from eval.runner import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
