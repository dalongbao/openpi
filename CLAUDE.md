# 3DV Project: Fine-Tuning pi0.5 on Egoverse Data + Isaac Sim Evaluation

## Project Overview

We are fine-tuning **pi0.5** (a Vision-Language-Action model) on **Egoverse** robot demonstration data, then evaluating the trained model in simulation to measure whether it learned to perform the task.

### The full pipeline

1. **Fine-tune pi0.5** on Egoverse `object_in_bowl` demonstrations (done via JAX LoRA training on Euler).
2. **Build a simulated test environment** in Isaac Sim on a local laptop — a 3D scene (`kitchen_scene_1.usd`) that replicates the real-world pi0.5 test setup: a Franka FR3 arm, a table, task objects (plate + yellow crate), and two cameras (external shoulder cam + wrist cam).
3. **Write an evaluation script** (`eval_script_1.py`) that loads the trained checkpoint, runs the policy in a closed loop inside the simulation, and records what happens.
4. **Run the sim eval on the Euler cluster** — the simulation is too heavy for a laptop, so we submit a SLURM job that spins up Isaac Sim inside an Apptainer container on a GPU node.
5. **Save a video** (`evaluation.mp4`) of the robot attempting the task, plus a CSV of joint positions over time.
6. **Download and review** the video manually to judge qualitatively whether the model is performing the task, then use the joint position data for quantitative benchmarking.

### What "success" looks like

- The robot moves purposefully toward the plate and attempts to grasp it.
- It transports the plate toward the yellow crate.
- Joint trajectories in `results.csv` show smooth, task-directed motion rather than random flailing.
- Comparing against the baseline (untrained pi0.5) shows improvement after fine-tuning.

### Current status (as of 2026-05-04)

- Fine-tuning: checkpoint at step 29999 exists at `/cluster/work/cvg/data/rytsui/checkpoints/pi05_egoverse/test/29999`.
- Scene: `kitchen_scene_1.usd` built and synced to Euler. Now includes `RecordingCamera` (3rd-person HD) and a repositioned `ExternalCamera` (policy view).
- Eval script: written, all bugs fixed, running end-to-end cleanly with dual-camera recording.
- Python packages: fully installed into `/cluster/scratch/kdoman/isaac_packages/`. Verified working.
- submit.sh: final working version at `/cluster/scratch/kdoman/submit.sh`.
- **First successful run: job 65303261, 2026-05-03, 9 min 24 sec, 3000 steps, exit 0.**
- **Second successful run: job 65336902, 2026-05-04, 11 min 57 sec, 3000 steps, exit 0.** Uses RecordingCamera from USD for HD output.
- **All scene assets now patched locally:** table, plate, crate, and fr3 all load from `/workspace/assets/` — no S3 fetches needed.

---

## Output style

- **All shell commands on a single line** (`&&` to chain). No multi-line code blocks for copy-paste commands.
- Be concise. No preamble, no restating.


## Quick Start (ETH Euler Cluster)

### 1. Clone and install

```bash
ssh euler.ethz.ch
cd ~
git clone <repo-url> openpi && cd openpi
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or ~/.zshrc
uv sync  # creates .venv, installs all deps — takes a few minutes first time
```

### 2. One-time setup: download base weights

Compute nodes have no internet. Download on login node first:

```bash
pip install --user gcsfs && python3 -c "import gcsfs; fs=gcsfs.GCSFileSystem(token='anon'); fs.get('openpi-assets/checkpoints/pi05_base/params','/cluster/work/cvg/data/Egoverse/pi05_base_jax/params',recursive=True); print('Done')"
```

Also cache the tokenizer (needed once):

```bash
cd ~/openpi && HF_LEROBOT_HOME=/cluster/work/cvg/data/Egoverse/lerobot_egoverse uv run python scripts/compute_norm_stats.py --config-name pi05_egoverse
```

### 3. Submit training

```bash
sbatch --partition=gpu.120h --time=120:00:00 --mem-per-cpu=16G --cpus-per-task=8 --gpus=a100:1 3dvision-experiments/run.slurm [config] [exp_name] [seed]
```

Defaults: `pi05_egoverse test 42`. Resume after kill: same command (script has `--resume`).

### 4. Monitor

```bash
tail -f slurm-<jobid>.out
sacct -j <jobid> --format=JobID,State,Elapsed,AllocTRES%60
```

## Euler Filesystem Layout

| Mount | Path | Quota | Characteristics |
|-------|------|-------|-----------------|
| Home | `/cluster/home/rytsui` (`~`) | 50 GB | Persistent, backed up. Too small for data/checkpoints. |
| Work | `/cluster/work/cvg/` | 46 TB shared (group) | Persistent, no auto-delete. **Currently 100% full (as of 2026-05-05).** |
| Scratch | `/cluster/scratch/rytsui/` | 2.5 TB per user, ~211 TB free | **Auto-deleted after 15 days of no access.** Fast, use for checkpoints and caches. |

**Rules:**
- Never write checkpoints or large caches to `~/` or `/cluster/work/cvg/data/rytsui/`.
- Checkpoints go to: `/cluster/scratch/rytsui/checkpoints/`
- HF cache goes to: `/cluster/scratch/rytsui/hf_cache/` (set `HF_HOME`, `HF_DATASETS_CACHE`)
- Source datasets (LeRobot format) remain on work: `/cluster/work/cvg/data/Egoverse/lerobot_egoverse/`
- **Known issue:** HF `datasets` library checks free disk space on the filesystem where source data resides. Since `/cluster/work/cvg` is 100% full, `load_dataset` fails with "Not enough disk space" even when cache is on scratch. Workaround: move datasets to scratch, or patch the space check.

## Shared Paths on Euler

| What | Path |
|------|------|
| Raw h5 data (object_in_bowl, 78 eps) | `/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/object_in_bowl_processed_50hz/` |
| Raw h5 data (bag_groceries, 300 eps) | `/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries/` |
| Raw LeRobot v2 data (jiaqchen's) | `/cluster/work/cvg/jiaqchen/EGOVERSE_DATA_3DV/bag_grocery/` and `.../object_in_container/` |
| Converted LeRobot dataset (object_in_bowl, 5 eps) | `/cluster/work/cvg/data/Egoverse/lerobot_egoverse/egoverse/all/` |
| Converted LeRobot dataset (bag_grocery_human, 1683 eps) | `/cluster/work/cvg/data/Egoverse/lerobot_egoverse/egoverse/bag_grocery_human/` |
| Converted LeRobot dataset (oic_human, 2537 eps) | `/cluster/work/cvg/data/Egoverse/lerobot_egoverse/egoverse/oic_human/` |
| pi0.5 base weights (JAX/orbax) | `/cluster/work/cvg/data/Egoverse/pi05_base_jax/params` |

Per-user paths:

| What | Path |
|------|------|
| Training checkpoints | `/cluster/scratch/rytsui/checkpoints/<config>/<exp_name>/` |
| HF cache | `/cluster/scratch/rytsui/hf_cache/` |
| HF datasets cache | `/cluster/scratch/rytsui/hf_cache/datasets/` |
| Norm stats | `~/openpi/assets/<config>/<repo_id>` |

## Files We Added

- `3dvision-experiments/convert_h5_to_lerobot.py` — converts Egoverse h5 to LeRobot v2 format
- `3dvision-experiments/convert_data.slurm` — SLURM job for data conversion
- `3dvision-experiments/run.slurm` — SLURM wrapper for training
- `3dvision-experiments/run_inference.py` — baseline inference on real h5 frames (pretrained pi0.5, no finetuning); loads base weights + egoverse norm stats, reports arm/hand MSE vs GT
- `3dvision-experiments/run_inference.slurm` — SLURM wrapper for baseline inference
- `3dvision-experiments/NOTES.md` — detailed setup log
- `src/openpi/policies/egoverse_policy.py` — EgoverseInputs/EgoverseOutputs data transforms
- `src/openpi/training/config.py` — added `LeRobotEgoverseDataConfig` and `pi05_egoverse` config

## Baseline Inference

Submit: `sbatch --time=01:00:00 --mem-per-cpu=16G --cpus-per-task=8 --gpus=1 3dvision-experiments/run_inference.slurm`

Custom args: `sbatch ... 3dvision-experiments/run_inference.slurm <h5_path> <num_frames> <frame_stride>`

Loads `pi05_base_jax` into the `pi05_egoverse` config (LoRA adapters init to zero → functionally base model), runs on real frames from an h5 file, reports arm/hand MSE vs ground truth. First call = 2–5 min JIT compile; later frames fast.

## Converting More Data

```bash
sbatch --cpus-per-task=8 --mem-per-cpu=16G --time=12:00:00 3dvision-experiments/convert_data.slurm
```

Then recompute norm stats on login node:

```bash
cd ~/openpi && HF_LEROBOT_HOME=/cluster/work/cvg/data/Egoverse/lerobot_egoverse uv run python scripts/compute_norm_stats.py --config-name pi05_egoverse
```

## Gotchas

- **No internet on compute nodes.** All downloads (weights, tokenizer, pip packages) must happen on login node.
- **Home dir has limited quota.** Never save checkpoints to `~/`. Use `--checkpoint-base-dir=/cluster/work/cvg/data/<username>/checkpoints`.
- **LoRA requires JAX trainer** (`scripts/train.py`). The PyTorch trainer (`scripts/train_pytorch.py`) does not support LoRA.
- **wandb is disabled** in `run.slurm` (compute has no internet). Loss is logged to stdout / slurm output file.
- **Only `object_in_bowl` is converted** so far. `bag_groceries` is bimanual with different h5 keys and needs separate handling.
- **A100 required for LoRA training.** Model is ~11.5GB; RTX 4090 (24GB) and below OOM during the training step with batch_size=32.
- **SLURM account quirk**: don't put `#SBATCH` directives in the script — pass everything on the `sbatch` command line. The account/partition combo is fragile.

## Training Config

- **Config name**: `pi05_egoverse`
- **Model**: pi0.5 with LoRA (gemma_2b_lora + gemma_300m_lora)
- **Batch size**: 32
- **Steps**: 30k, checkpoint every 5k
- **LR**: 5e-5 cosine with 1k warmup
- **Data**: 5 episodes of `object_in_bowl` at 50Hz, single Aria camera, 24-dim state/actions (7 arm + 17 hand)

---

## Isaac Sim Evaluation (Phase 2 goal)

**Goal:** Run the trained pi0.5 checkpoint (step 29999) on a Franka FR3 inside Isaac Sim 4.5.0 on Euler, record `evaluation.mp4` and `results.csv`.

### Key files

| What | Where |
|------|-------|
| SLURM script | `3dvision-experiments/isaac-sim/submit.sh` (repo) → copy to `/cluster/scratch/kdoman/submit.sh` |
| Eval script | `3dvision-experiments/isaac-sim/eval_script_1.py` (repo) → copy to `/cluster/scratch/kdoman/pi0_test/eval_script_1.py` |
| USD scene | `3dvision-experiments/isaac-sim/kitchen_scene_1.usd` (repo) → copy to `/cluster/scratch/kdoman/pi0_test/kitchen_scene_1.usd` |
| Isaac Sim container | `/cluster/scratch/kdoman/pi0_test/isaac-sim_4.5.0.sif` |
| openpi repo (on Euler) | `/cluster/scratch/kdoman/openpi/` (NOT `~/openpi` — home is too small) |
| Python packages | `/cluster/scratch/kdoman/isaac_packages/` |
| Checkpoints | `/cluster/work/cvg/data/rytsui/checkpoints/pi05_egoverse/test/29999` |
| Shader cache | `/cluster/scratch/kdoman/isaac_cache/kit/` (bound to `/isaac-sim/kit/cache` inside container) |
| Outputs | `/cluster/scratch/kdoman/pi0_test/evaluation.mp4` and `results.csv` |

### Python environment problem (critical)

Isaac Sim 4.5.0 uses **Python 3.10.15**. The openpi uv venv uses Python 3.11. These are ABI-incompatible — you cannot share venvs. Solution: install all openpi deps using Isaac Sim's own Python into a bindable directory.

### One-time setup: install Python packages into `/cluster/scratch/kdoman/isaac_packages/`

Do this on the **login node** (compute nodes have no internet). If packages are already there but versions are wrong, wipe first: `rm -rf /cluster/scratch/kdoman/isaac_packages && mkdir -p /cluster/scratch/kdoman/isaac_packages`

**CRITICAL:** After wiping, you MUST immediately run the full reinstall below before submitting any jobs. The directory being empty is not obvious from submit.sh output — the job starts, runs for ~1 min (Isaac Sim boots), then crashes with `ModuleNotFoundError` on the first openpi import.

```bash
APPTAINERENV_ACCEPT_EULA=Y APPTAINERENV_HTTP_PROXY=http://proxy.ethz.ch:3128 APPTAINERENV_HTTPS_PROXY=http://proxy.ethz.ch:3128 apptainer exec --bind /cluster/scratch/kdoman/isaac_packages:/target /cluster/scratch/kdoman/pi0_test/isaac-sim_4.5.0.sif /isaac-sim/python.sh -m pip install --no-cache-dir --target /target "jax[cuda12]==0.5.3" "flax==0.10.2" "jaxtyping==0.2.36" "orbax-checkpoint==0.11.13" "numpy==1.26.4" "beartype==0.19.0" "ml_collections==1.0.0" "chex>=0.1.86" "augmax>=0.3.4" "dm-tree>=0.1.8" "einops>=0.8.0" "equinox>=0.11.8" "flatbuffers>=24.3.25" "gcsfs>=2024.6.0" "imageio>=2.36.1" "numpydantic>=1.6.6" "pillow>=11.0.0" "sentencepiece>=0.2.0" "tqdm-loggable>=0.2" "tyro>=0.9.5" "wandb>=0.19.1" "filelock>=3.16.1" "treescope>=0.1.7" "polars>=1.30.0" "transformers==4.53.2" "draccus" "pytest>=8.3.4" "rich>=14.0.0"
```

**Critical version pins** (these break openpi if wrong):

| Package | Required | Why |
|---------|----------|-----|
| `jax` | `0.5.3` | uv.lock pin; 0.4.x silently wrong |
| `jaxtyping` | `0.2.36` | 0.3.x removed `_check_dataclass_annotations` used in `array_typing.py:29` |
| `orbax-checkpoint` | `0.11.13` | checkpoint format compatibility |
| `numpy` | `1.26.4` | numpy 2.x breaks Isaac Sim's numba |
| `beartype` | `0.19.0` | uv.lock pin |

**Verify versions after install:**
```bash
APPTAINERENV_ACCEPT_EULA=Y apptainer exec --bind /cluster/scratch/kdoman/isaac_packages:/isaac_packages /cluster/scratch/kdoman/pi0_test/isaac-sim_4.5.0.sif /isaac-sim/python.sh -c "import sys; sys.path.insert(0,'/isaac_packages'); import jax, jaxtyping, orbax.checkpoint, tqdm_loggable; print('jax', jax.__version__); print('jaxtyping', jaxtyping.__version__)"
```

### Update files from repo and submit job

```bash
cd /cluster/scratch/kdoman/openpi && git pull && cp 3dvision-experiments/isaac-sim/eval_script_1.py /cluster/scratch/kdoman/pi0_test/eval_script_1.py && sed -i 's/\r//' /cluster/scratch/kdoman/submit.sh && sbatch --partition=gpu.4h --time=00:30:00 --mem-per-cpu=8G --cpus-per-task=8 --gpus=rtx_3090:1 /cluster/scratch/kdoman/submit.sh
```

Use `gpu.4h` + `rtx_3090` (24 GB VRAM, available quickly). Use `gpuhe.4h` + `a100` only if OOM. For a full 60s eval run (3000 steps at 50Hz), `--time=00:30:00` is enough (actual runtime ~12 min with RecordingCamera: 72s Isaac Sim boot + 49s JAX JIT compile on step 0 + ~160ms/step thereafter).

**WARNING on `git pull`:** The repo has had commits with broken camera code (undefined `CAMERA_RES`, `ext_img`, `prepare_image` — see gotchas). Always check `eval_script_1.py` after a pull before submitting.

### Watch the job live

```bash
squeue -u $USER  # note NODELIST, e.g. eu-g4-013
tail -f <dir-where-sbatch-was-run>/slurm-<JOBID>.out
```

SSH tunnel from laptop (replace node name):
```bash
ssh -L 8211:eu-g4-013:8211 -L 49100:eu-g4-013:49100 kdoman@euler.ethz.ch
```
Then open `http://localhost:8211/streaming/webrtc-demo/` in Chrome. If black screen, try `?tcp=true`.

### Download results

```bash
rsync -avP kdoman@euler.ethz.ch:/cluster/scratch/kdoman/pi0_test/evaluation.mp4 ./ && rsync -avP kdoman@euler.ethz.ch:/cluster/scratch/kdoman/pi0_test/results.csv ./
```

### Hard-won gotchas for Isaac Sim on Euler

| Problem | Fix |
|---------|-----|
| Container hangs at startup (>10 min, no output) | Kit shader cache is read-only. Bind `$ISAAC_SIM_CACHE_DIR/kit` → `/isaac-sim/kit/cache` in apptainer call. |
| `python: command not found` | Use `/isaac-sim/python.sh`, not `python` or `python3`. |
| `pip install` fails with DNS errors | Container has no internet. Add `APPTAINERENV_HTTP_PROXY=http://proxy.ethz.ch:3128 APPTAINERENV_HTTPS_PROXY=http://proxy.ethz.ch:3128` before `apptainer exec`. |
| Hash mismatch error from pip | Corrupt pip cache. Add `--no-cache-dir` to pip command. |
| `pip install --target` doesn't overwrite | `--force-reinstall` is not enough. `rm -rf /isaac_packages/<pkg> /isaac_packages/<pkg>-*.dist-info` first, then reinstall. |
| jaxtyping 0.3.x ends up in `/isaac_packages` | chex (dep of optax → flax) pulls in latest jaxtyping as transitive dep. Fix: wipe and reinstall with pinned version, or delete jaxtyping dir before reinstalling. |
| Wrong jaxtyping loaded despite `sys.path.insert(0, ...)` | The wrong version is physically in `/isaac_packages`. Check with `import jaxtyping; print(jaxtyping.__file__)`. Delete and reinstall. |
| SLURM `--mem` flag rejected | Euler only accepts `--mem-per-cpu`. Never use `--mem=`. |
| SBATCH directives ignored | Euler quirk: pass ALL flags on the `sbatch` command line, not as `#SBATCH` lines in the script. |
| CRLF line endings crash bash scripts | Files edited on Windows get `\r\n`. Always run `sed -i 's/\r//' submit.sh` before `sbatch`. |
| SLURM log not found at `~/slurm-*.out` | Log goes to the CWD where `sbatch` was run. Check that directory. |
| `gpuhe.4h` stuck in queue forever | Switch to `gpu.4h --gpus=rtx_3090:1` — more available, 24 GB is enough for inference. |
| openpi imports fail (wrong Python) | Never add the uv `.venv` site-packages (Python 3.11) to sys.path inside the container. Only use `/isaac_packages` (installed with Isaac Sim's Python 3.10). |
| `lerobot` not needed for inference, but was a top-level import | `data_loader.py` imported lerobot at module level, which gets pulled in transitively during inference. Fixed by moving the import inside the function that uses it (`data_loader.py:148`). Do not install lerobot — its deps conflict with Isaac Sim. |
| SLURM log location | `sbatch` output goes to CWD. Run `sbatch` from scratch dir or note CWD. |
| Warp kernel cache / texture cache errors: `Read-only file system: '/cluster/home'` | Isaac Sim plugins write caches to the home dir using the C library `getpwuid()` (not `$HOME`), so `APPTAINERENV_HOME` has no effect and is also blocked by Euler's security policy. Fix: bind-mount a writable scratch dir over `/cluster/home/$USER` — `--bind "$ISAAC_SIM_CACHE_DIR/ov_home":/cluster/home/$USER`. Already in current submit.sh. |
| `isaac_packages` wiped but not reinstalled | Easiest mistake to make: you wipe to fix a version and forget to reinstall. Isaac Sim will boot fine (~73s) then crash with the first `ModuleNotFoundError`. Always reinstall immediately after wiping. |
| `chex` not installed despite being a dep of `optax` | `optax 0.2.8` dropped `chex` as a hard dep, so pip doesn't pull it. Isaac Sim also doesn't ship it. Must be listed explicitly in the pip install command. |
| `cannot import name 'NoDefault' from 'typing_extensions'` | Isaac Sim loads an old `typing_extensions` from its pip_prebundle at startup and caches it in `sys.modules`. Even though `/isaac_packages` has 4.15.0, the stale cache wins. Fix: after `sys.path` setup in `eval_script_1.py`, flush `typing_extensions` from `sys.modules` so the next import picks up the newer version. |
| `AttributeError: module 'datetime' has no attribute 'UTC'` | `datetime.UTC` was added in Python 3.11. Isaac Sim uses Python 3.10. Fixed in `src/openpi/shared/download.py:191` — use `datetime.timezone.utc` instead. |
| `franka.get_joint_positions()` returns `None` at step 0 | After `world.reset()` in the eval loop, the articulation view is cleared. Must call `franka.initialize()` again immediately after `world.reset()`, then step at least 20 times before reading joint positions. Also add a null guard: `if joint_pos is None: joint_pos = np.zeros(9, dtype=np.float32)`. Fixed in `eval_script_1.py`. |
| USD scene assets load but objects are invisible / no physics | Scene references S3 URLs (table, plate, crate). Fixed: all payloads patched in `eval_script_1.py` to load from `/workspace/assets/`. |
| Remote commit has broken camera code | Commits d087bfb / bc49b04 introduced `CAMERA_RES`, `ext_img`, and `prepare_image` references that don't exist anywhere in the file — the code would crash immediately. Always verify camera variable names match the constants defined in the CONFIG block when merging camera changes. |

### How `eval_script_1.py` works (summary)

1. `SimulationApp` starts first (Isaac Sim requirement — must be line 1).
2. After init, `sys.path` is extended with `/workspace/openpi/src`, `/workspace/openpi/packages/openpi-client/src`, `/isaac_packages`.
3. `pi05_egoverse` config loaded; norm stats read from `/workspace/openpi/assets/`.
4. Checkpoint loaded from `/checkpoints/pi05_egoverse/test/29999` (orbax format).
5. `kitchen_scene_1.usd` opened; fr3 + 3 scene assets patched to local USD files; Franka articulation initialized.
6. Two cameras initialized from USD prims: `ExternalCamera` (224×224, policy input) and `RecordingCamera` (1280×720, HD output).
7. 3000-step loop: grab ExternalCamera frame → build observation → `policy.infer()` → apply joint action → step sim → write RecordingCamera frame to MP4.
8. Outputs: `/workspace/evaluation.mp4` (HD 3rd-person video), `/workspace/results.csv` (= `/cluster/scratch/kdoman/pi0_test/` on host).

### apptainer bind mounts in submit.sh

```
$WORKSPACE (/cluster/scratch/kdoman/pi0_test)    →  /workspace
/cluster/scratch/kdoman/openpi                   →  /workspace/openpi
/cluster/work/cvg/data/rytsui/checkpoints        →  /checkpoints
$ISAAC_SIM_CACHE_DIR/kit                         →  /isaac-sim/kit/cache   ← must be writable
$ISAAC_SIM_CACHE_DIR/ov_home                     →  /cluster/home/$USER    ← overrides read-only home for Warp/texture cache
/cluster/scratch/kdoman/isaac_packages           →  /isaac_packages
```

### Timing profile (job 65303261, RTX 3090, 3000 steps, no RecordingCamera)

| Phase | Time |
|-------|------|
| Isaac Sim boot (`Simulation App Starting` → `Startup Complete`) | ~70s |
| Policy load + JAX backend init | ~30s |
| Scene open + fr3 patch + world init + camera warmup (20 steps) | ~25s |
| Step 0 (JAX JIT compile of policy) | ~45s |
| Steps 1–2999 | ~165ms/step avg → ~8 min total |
| **Total wall time** | **~9 min 24 sec** |

### Timing profile (job 65336902, RTX 3090, 3000 steps, with RecordingCamera 1280×720)

| Phase | Time |
|-------|------|
| Isaac Sim boot | ~72s |
| Policy load + JAX backend init | ~30s |
| Scene open + patches + world init + camera warmup | ~30s |
| Step 0 (JAX JIT compile) | ~49s |
| Steps 1–2999 | ~160ms/step avg → ~8 min total |
| **Total wall time** | **~11 min 57 sec** |

---

## Scene Asset Caching (DONE as of 2026-05-03)

All mesh assets (table, plate, yellow crate, fr3 robot) are patched in `eval_script_1.py` to load from local files under `/workspace/assets/` instead of S3 HTTPS URLs. The scene is fully populated on compute nodes.

**The fix (same pattern as the fr3 robot):** download each asset to a local path on the login node, then in `eval_script_1.py` clear each prim's payload and add a local path instead.

### Step 1 — identify which S3 URLs the scene uses (run on login node)

```bash
python3 -c "
from pxr import Usd
stage = Usd.Stage.Open('/cluster/scratch/kdoman/pi0_test/kitchen_scene_1.usd')
for prim in stage.Traverse():
    for p in prim.GetPayloads().GetAddedOrExplicitItems():
        if str(p.assetPath).startswith('http'):
            print(prim.GetPath(), p.assetPath)
"
```

### Step 2 — download assets on the login node

Assets are on Omniverse's public S3. Download with wget or curl (login node has internet):

```bash
ASSETS=/cluster/scratch/kdoman/pi0_test/assets
# Example — replace URLs with actual ones from step 1:
mkdir -p $ASSETS/table $ASSETS/plate $ASSETS/crate
wget -q -O $ASSETS/table/SM_HeavyDutyPackingTable_C02_01.usd "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Mimic/g1_squatting_task/PackingTable/props/SM_HeavyDutyPackingTable_C02_01/SM_HeavyDutyPackingTable_C02_01.usd"
# Download all referenced sub-layers recursively (assets often have sub-USDs)
```

Alternatively, use the Omniverse Nucleus cache approach: open the scene in Isaac Sim on a laptop with internet once — it auto-downloads and caches everything. Then rsync the local cache to Euler.

### Step 3 — patch USD payloads in eval_script_1.py

Add one block per object (same pattern as the existing fr3 patch):

```python
_TABLE_PRIMS = ["/World/SM_HeavyDutyPackingTable_C02_01"]
_PLATE_PRIMS = ["/World/plate_small"]
_CRATE_PRIMS = ["/World/SM_Crate_A07_Yellow_01_physics"]

for prim_path, local_usd in [
    ("/World/SM_HeavyDutyPackingTable_C02_01", "/workspace/assets/table/SM_HeavyDutyPackingTable_C02_01.usd"),
    ("/World/plate_small",                     "/workspace/assets/plate/plate_small.usd"),
    ("/World/SM_Crate_A07_Yellow_01_physics",  "/workspace/assets/crate/SM_Crate_A07_Yellow_01_physics.usd"),
]:
    prim = _stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        prim.GetPayloads().ClearPayloads()
        prim.GetPayloads().AddPayload(local_usd)
        print(f"[init] Patched {prim_path} -> {local_usd}")
    else:
        print(f"[WARN] {prim_path} not found in stage")
```

---

## Setting Up Isaac Sim Eval for a New User on Euler

Everything in `/cluster/scratch/<user>/` is personal. A new team member needs their own copies of the large files. Shared/read-only resources (checkpoints, datasets) live in `/cluster/work/cvg/data/`.

### What they need in their scratch dir

| File | Size | How to get it |
|------|------|--------------|
| `pi0_test/isaac-sim_4.5.0.sif` | ~6 GB | `cp /cluster/scratch/kdoman/pi0_test/isaac-sim_4.5.0.sif /cluster/scratch/<user>/pi0_test/` (fast, same filesystem) |
| `pi0_test/kitchen_scene_1.usd` | small | copy from repo: `cp 3dvision-experiments/isaac-sim/kitchen_scene_1.usd /cluster/scratch/<user>/pi0_test/` |
| `pi0_test/assets/fr3/fr3.usd` | small | `cp -r /cluster/scratch/kdoman/pi0_test/assets /cluster/scratch/<user>/pi0_test/` |
| `isaac_packages/` | ~2 GB | reinstall (see pip command above) — do NOT copy, ABI may differ if Python env differs |
| `isaac_cache/` | auto-created | created by submit.sh on first run |
| `openpi/` (the repo) | ~1 GB | `git clone <repo-url> /cluster/scratch/<user>/openpi` |

### New user one-time setup (run all on login node)

```bash
USER=newuser
SCRATCH=/cluster/scratch/$USER
mkdir -p $SCRATCH/pi0_test/assets $SCRATCH/isaac_packages $SCRATCH/isaac_cache/kit $SCRATCH/isaac_cache/ov_home && cp /cluster/scratch/kdoman/pi0_test/isaac-sim_4.5.0.sif $SCRATCH/pi0_test/ && cp -r /cluster/scratch/kdoman/pi0_test/assets $SCRATCH/pi0_test/ && git clone <repo-url> $SCRATCH/openpi && cp $SCRATCH/openpi/3dvision-experiments/isaac-sim/kitchen_scene_1.usd $SCRATCH/pi0_test/ && cp $SCRATCH/openpi/3dvision-experiments/isaac-sim/submit.sh $SCRATCH/submit.sh
```

Then install python packages (must use the container's Python — do NOT copy from kdoman's packages):

```bash
APPTAINERENV_ACCEPT_EULA=Y APPTAINERENV_HTTP_PROXY=http://proxy.ethz.ch:3128 APPTAINERENV_HTTPS_PROXY=http://proxy.ethz.ch:3128 apptainer exec --bind /cluster/scratch/$USER/isaac_packages:/target /cluster/scratch/$USER/pi0_test/isaac-sim_4.5.0.sif /isaac-sim/python.sh -m pip install --no-cache-dir --target /target "jax[cuda12]==0.5.3" "flax==0.10.2" "jaxtyping==0.2.36" "orbax-checkpoint==0.11.13" "numpy==1.26.4" "beartype==0.19.0" "ml_collections==1.0.0" "chex>=0.1.86" "augmax>=0.3.4" "dm-tree>=0.1.8" "einops>=0.8.0" "equinox>=0.11.8" "flatbuffers>=24.3.25" "gcsfs>=2024.6.0" "imageio>=2.36.1" "numpydantic>=1.6.6" "pillow>=11.0.0" "sentencepiece>=0.2.0" "tqdm-loggable>=0.2" "tyro>=0.9.5" "wandb>=0.19.1" "filelock>=3.16.1" "treescope>=0.1.7" "polars>=1.30.0" "transformers==4.53.2" "draccus" "pytest>=8.3.4" "rich>=14.0.0"
```

Then compute norm stats (uses the openpi uv venv, NOT the Isaac packages):

```bash
cd /cluster/scratch/$USER/openpi && uv sync && HF_LEROBOT_HOME=/cluster/work/cvg/data/Egoverse/lerobot_egoverse uv run python scripts/compute_norm_stats.py --config-name pi05_egoverse
```

This writes norm stats to `/cluster/scratch/$USER/openpi/assets/pi05_egoverse/egoverse/all/norm_stats.json` — required by the eval script.

Edit `submit.sh` if the new user's checkpoint path differs from `rytsui`'s (the `CHECKPOINTS=` line). Then submit:

```bash
cd /cluster/scratch/$USER && sed -i 's/\r//' submit.sh && sbatch --partition=gpu.4h --time=00:30:00 --mem-per-cpu=8G --cpus-per-task=8 --gpus=rtx_3090:1 /cluster/scratch/$USER/submit.sh
```

### Why you cannot just copy `isaac_packages/` between users

The packages are installed with `--target` into a flat directory using the container's Python 3.10 interpreter. They are tied to the exact Python ABI of that container. If the container is the same SIF file, the packages are technically copyable — but pip's `--target` layout doesn't include scripts and can have edge cases. Reinstalling takes ~10 min on the login node and guarantees correctness.
