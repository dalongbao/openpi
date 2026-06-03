# Isaac Sim pi0.5 Evaluation Package

Modular refactor of `eval_script_1.py`. Same behavior on default flags — but
CLI-driven, seedable, and ready for sweeps.

## Layout

```
3dvision-experiments/isaac-sim/
├── eval_script_1.py        # thin shim — entry point invoked by submit.sh
├── submit.sh               # SLURM wrapper; passes optional CLI args through
└── eval/
    ├── __init__.py         # public exports
    ├── core.py             # EvalConfig, EvalResult, EvalSim (lifecycle + loop)
    ├── runner.py           # argparse CLI -> EvalSim.run() -> dump artifacts
    ├── metrics.py          # success / progress / smoothness / JSON writer
    ├── perturbations.py    # mutates the USD stage before world.reset()
    └── probes.py           # records intermediate-layer activations
```

## Parity run (matches the legacy script exactly)

From inside the container:

```bash
/isaac-sim/python.sh /workspace/eval_script_1.py
```

Or via SLURM (unchanged from before the refactor):

```bash
sbatch --partition=gpu.4h --time=00:30:00 --mem-per-cpu=8G --cpus-per-task=8 --gpus=rtx_3090:1 /cluster/scratch/$USER/submit.sh
```

## CLI examples

Pick a different checkpoint + seed, write to a sweep subdir:

```bash
/isaac-sim/python.sh /workspace/eval_script_1.py \
  --checkpoint-dir /checkpoints/pi05_egoverse/test/14999 \
  --seed 7 \
  --output-dir /workspace/run_ckpt14999_seed7
```

Same thing via SLURM:

```bash
sbatch ... submit.sh --checkpoint /checkpoints/pi05_egoverse/test/14999 --seed 7 --output-name run_ckpt14999_seed7
```

With perturbations:

```bash
/isaac-sim/python.sh /workspace/eval_script_1.py \
  --perturbation-config /workspace/perturbs/plate_jitter_5cm.json
```

## What the runner expects

Inside the Isaac Sim container, mapped at runtime by `submit.sh`:
- `/workspace/kitchen_scene_1.usd` — scene file (copied from repo).
- `/workspace/eval_script_1.py` — this shim (copied from repo).
- `/workspace/eval/` — package source. Either copy from repo
  `3dvision-experiments/isaac-sim/eval/` or rely on the shim's fallback
  which adds `/workspace/openpi/3dvision-experiments/isaac-sim/` to
  `sys.path` if the package isn't in `/workspace/`.
- `/workspace/assets/` — local USD asset cache (table, plate, crate, fr3).
- `/workspace/openpi/` — bind-mounted repo with the `openpi` python pkg.
- `/checkpoints/pi05_egoverse/test/29999/` — orbax checkpoint.
- `/isaac_packages/` — pinned Python deps installed with the container's Python 3.10.

## Outputs

Written to `--output-dir` (default `/workspace`):
- `results.csv` — `[step, infer_ms, j0..j8]` per step (legacy-compatible schema).
- `evaluation.mp4` — 1280x720 HD video from RecordingCamera (50 fps).
- `evaluation_external.mp4` — 224x224 policy-view video (only if `--record-external`).
- `metrics.json` — summary scalars from `EvalResult` (success, progress, smoothness, runtime).

## Importable for offline analysis

`eval.metrics` is pure NumPy and safe to import on a laptop without Isaac Sim.
`eval.core` and `eval.runner` are also importable on a laptop (Isaac Sim is
imported lazily inside `EvalSim.setup()`); only `EvalSim.setup()`/`run()`
require the container.
