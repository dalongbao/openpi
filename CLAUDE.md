# 3DV Project: Fine-Tuning pi0.5 on Egoverse Data + Isaac Sim Evaluation

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

## Shared Paths on Euler

| What | Path |
|------|------|
| Raw h5 data (object_in_bowl, 78 eps) | `/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/object_in_bowl_processed_50hz/` |
| Raw h5 data (bag_groceries, 300 eps) | `/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries/` |
| Converted LeRobot dataset (5 eps) | `/cluster/work/cvg/data/Egoverse/lerobot_egoverse/egoverse/all/` |
| pi0.5 base weights (JAX/orbax) | `/cluster/work/cvg/data/Egoverse/pi05_base_jax/params` |
| pi0.5 base weights (safetensors, unused) | `/cluster/work/cvg/data/rytsui/pi05_base/` |

Per-user paths (checkpoints, experiments):

| What | Path |
|------|------|
| Training checkpoints | `/cluster/work/cvg/data/<username>/checkpoints/pi05_egoverse/<exp_name>/` |
| Norm stats | `~/openpi/assets/pi05_egoverse/egoverse/all` |

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

```bash
APPTAINERENV_ACCEPT_EULA=Y APPTAINERENV_HTTP_PROXY=http://proxy.ethz.ch:3128 APPTAINERENV_HTTPS_PROXY=http://proxy.ethz.ch:3128 apptainer exec --bind /cluster/scratch/kdoman/isaac_packages:/target /cluster/scratch/kdoman/pi0_test/isaac-sim_4.5.0.sif /isaac-sim/python.sh -m pip install --no-cache-dir --target /target "jax[cuda12]==0.5.3" "flax==0.10.2" "jaxtyping==0.2.36" "orbax-checkpoint==0.11.13" "numpy==1.26.4" "beartype==0.19.0" "ml_collections==1.0.0" "augmax>=0.3.4" "dm-tree>=0.1.8" "einops>=0.8.0" "equinox>=0.11.8" "flatbuffers>=24.3.25" "gcsfs>=2024.6.0" "imageio>=2.36.1" "numpydantic>=1.6.6" "pillow>=11.0.0" "sentencepiece>=0.2.0" "tqdm-loggable>=0.2" "tyro>=0.9.5" "wandb>=0.19.1" "filelock>=3.16.1" "treescope>=0.1.7" "polars>=1.30.0" "transformers==4.53.2" "draccus" "pytest>=8.3.4" "rich>=14.0.0"
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

Use `gpu.4h` + `rtx_3090` (24 GB VRAM, available quickly). Use `gpuhe.4h` + `a100` only if OOM. For a full 60s eval run (3000 steps), extend `--time=02:00:00`.

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
| `lerobot` not needed | Only imported in `data_loader.py` (training). Not needed for inference. Do not install. |
| SLURM log location | `sbatch` output goes to CWD. Run `sbatch` from scratch dir or note CWD. |

### How `eval_script_1.py` works (summary)

1. `SimulationApp` starts first (Isaac Sim requirement — must be line 1).
2. After init, `sys.path` is extended with `/workspace/openpi/src`, `/workspace/openpi/packages/openpi-client/src`, `/isaac_packages`.
3. `pi05_egoverse` config loaded; norm stats read from `/workspace/openpi/assets/`.
4. Checkpoint loaded from `/checkpoints/pi05_egoverse/test/29999` (orbax format).
5. `kitchen_scene_1.usd` opened; Franka articulation and two cameras initialized.
6. 3000-step loop: grab camera frame → build observation → `policy.infer()` → apply joint action → step sim → write frame to MP4.
7. Outputs: `/workspace/evaluation.mp4`, `/workspace/results.csv` (= `/cluster/scratch/kdoman/pi0_test/` on host).

### apptainer bind mounts in submit.sh

```
$WORKSPACE (/cluster/scratch/kdoman/pi0_test)  →  /workspace
/cluster/scratch/kdoman/openpi                 →  /workspace/openpi
/cluster/work/cvg/data/rytsui/checkpoints      →  /checkpoints
$ISAAC_SIM_CACHE_DIR/kit                       →  /isaac-sim/kit/cache   ← must be writable
/cluster/scratch/kdoman/isaac_packages         →  /isaac_packages
```
