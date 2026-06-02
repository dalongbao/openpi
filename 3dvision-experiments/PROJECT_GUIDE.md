# Project Guide: Fine-tuning pi0.5 on Egoverse + Isaac Sim Evaluation

This is the single source of truth for this project. It covers the full pipeline — from raw data to a fine-tuned robot policy running live in simulation — plus all the cluster setup, code internals, and hard-won debugging knowledge accumulated along the way.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Architecture Overview](#2-architecture-overview)
3. [Cluster Environment Reference](#3-cluster-environment-reference)
4. [Phase 1: Data Preparation](#4-phase-1-data-preparation)
5. [Phase 2: Training (Fine-tuning pi0.5)](#5-phase-2-training-fine-tuning-pi05)
6. [Phase 3: Baseline Inference on Real Data](#6-phase-3-baseline-inference-on-real-data)
7. [Phase 4: Isaac Sim Evaluation](#7-phase-4-isaac-sim-evaluation)
8. [Shared File Paths Reference](#8-shared-file-paths-reference)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. What This Project Does

We fine-tune **pi0.5** — a Vision-Language-Action (VLA) model from Physical Intelligence — on the **Egoverse** dataset, which contains egocentric robot demonstrations of household manipulation tasks recorded with an Aria headset. After training, we evaluate the fine-tuned policy in **Isaac Sim**, a physics simulator, to measure whether the robot learned to perform the task.

The concrete task is `object_in_bowl`: a Franka FR3 arm must pick up an object and place it in a bowl. Training data comes from 78 episodes of a human demonstrating this task.

**What success looks like:**
- The robot moves purposefully toward the target object.
- It grasps and transports it toward the container.
- Joint trajectories in `results.csv` show smooth, task-directed motion rather than random flailing.

---

## 2. Architecture Overview

### pi0.5 — the model

pi0.5 is a diffusion-based VLA model built on PaliGemma (a vision-language backbone combining SigLIP image encoder + Gemma language model) with a flow-matching action head. It takes:
- **Image(s)**: one or more RGB frames from the robot's cameras
- **State**: current joint positions (proprioception)
- **Language command**: a text description of the task

And outputs a **chunk of future actions** (a sequence of joint position targets). The action head runs denoising diffusion to generate smooth, multi-step predictions rather than single-step regression.

**pi0.5 vs pi0**: pi0.5 adds a KV-attention language model on top of the diffusion head to better handle long-horizon instructions and dexterous hands.

### LoRA fine-tuning

We use LoRA (Low-Rank Adaptation) to fine-tune pi0.5 without touching the full 11.5B parameter model. LoRA adds small trainable rank-decomposition matrices to the attention layers of both the Gemma 2B backbone and the Gemma 300M action head. At inference time, the LoRA weights are merged — no extra compute vs. base model.

Config: `pi05_egoverse` in `src/openpi/training/config.py`. Key hyperparameters:
- Batch size: 32
- Steps: 30k, checkpoint every 5k
- LR: 5e-5 cosine decay with 1k warmup
- Data: 5 episodes of `object_in_bowl` at 50 Hz

### Egoverse data format

Each episode is an HDF5 file containing:
- `observations/images/aria_rgb_cam/color` — RGB video frames (480×640 uint8)
- `observations/qpos_arm` — 7-DOF arm joint positions
- `observations/qpos_hand` — 17-DOF hand joint positions
- `actions_arm`, `actions_hand` — corresponding ground-truth actions

Total action dimension: 24 (7 arm + 17 hand). The model always predicts all 24 dims even though Isaac Sim only uses the 7 arm joints.

### Isaac Sim evaluation setup

The simulation uses:
- **Franka FR3** robot arm (7 DOF)
- **kitchen_scene_1.usd** — a 3D scene with a table, plate, and yellow crate
- **ExternalCamera** (224×224) — feeds the policy
- **RecordingCamera** (1280×720) — writes `evaluation.mp4`

The eval runs at 50 Hz for 3000 steps (60 seconds of simulated time). The policy predicts a chunk of 10 actions at each step, but only the first action is applied (or the full chunk is applied with EMA smoothing — see §7.4).

---

## 3. Cluster Environment Reference

### Filesystem layout

| Mount | Path | Quota | Notes |
|-------|------|-------|-------|
| Home | `/cluster/home/<user>/` | 50 GB | Persistent, backed up. Too small for data. |
| Work | `/cluster/work/cvg/` | 46 TB shared | Persistent, no auto-delete. Used for datasets and checkpoints. |
| Scratch | `/cluster/scratch/<user>/` | 2.5 TB | **Auto-deleted after 15 days of no access.** Fast. Use for caches and active runs. |

**Rules:**
- Never write large files to home. Use scratch for everything transient.
- Checkpoints go to `/cluster/scratch/<user>/checkpoints/` or `/cluster/work/cvg/data/rytsui/checkpoints/`.
- Scratch purges happen silently — always have recovery commands ready (§7.5).

### SLURM quirks (Euler-specific)

- **Never put `#SBATCH` directives inside the script** — pass all flags on the `sbatch` command line. Euler's scheduler ignores in-script directives.
- **Never use `--mem=`** — only `--mem-per-cpu=` is accepted.
- SLURM log goes to the **CWD of the `sbatch` invocation**, not to `~/`.
- `gpu.4h + rtx_3090:1` schedules faster than `gpuhe.4h + a100`.
- **No internet on compute nodes.** All downloads must happen on the login node.

### Python environments — two separate ones

This project uses two incompatible Python environments:

| Environment | Python | Where | Used for |
|-------------|--------|-------|---------|
| uv venv (`.venv/`) | 3.11 | `/cluster/scratch/<user>/openpi/.venv/` | Training, inference, data conversion |
| Isaac packages | 3.10 | `/cluster/scratch/<user>/isaac_packages/` | Isaac Sim eval only |

They cannot be mixed. Isaac Sim bundles its own Python 3.10 interpreter — openpi's 3.11 venv is ABI-incompatible with it. The Isaac packages are installed using the container's Python into a flat `--target` directory that gets bind-mounted into the container at runtime.

### One-time repo setup

```bash
ssh <username>@euler.ethz.ch
cd /cluster/scratch/$USER
git clone https://github.com/dalongbao/openpi.git
cd openpi
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv sync   # creates .venv, installs all Python 3.11 deps
```

---

## 4. Phase 1: Data Preparation

### Input data

Raw Egoverse episodes are HDF5 files at:
```
/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/object_in_bowl_processed_50hz/
```
78 episodes total. Each episode ~1000–9000 frames at 50 Hz.

### Convert to LeRobot format

The training pipeline expects data in LeRobot v2 format. Convert with:

```bash
sbatch --cpus-per-task=8 --mem-per-cpu=16G --time=12:00:00 \
  3dvision-experiments/convert_data.slurm
```

This runs `convert_h5_to_lerobot.py`, which:
1. Reads each `.h5` file
2. Extracts `aria_rgb_cam/color` images, `qpos_arm`, `qpos_hand`, `actions_arm`, `actions_hand`
3. Writes a LeRobot dataset to `$HF_LEROBOT_HOME/egoverse/all/`
4. Skips episodes with missing keys or more than 5000 frames (OOM guard)

Output path: `/cluster/work/cvg/data/Egoverse/lerobot_egoverse/egoverse/all/`

To convert only N episodes (for testing):
```bash
uv run python 3dvision-experiments/convert_h5_to_lerobot.py \
  --data_dir /cluster/work/cvg/data/Egoverse/raw_timesynced_h5/object_in_bowl_processed_50hz \
  --max_episodes 5
```

### Compute norm stats

After converting data, compute normalization statistics (required before training and inference):

```bash
cd /cluster/scratch/$USER/openpi
HF_LEROBOT_HOME=/cluster/work/cvg/data/Egoverse/lerobot_egoverse \
  uv run python scripts/compute_norm_stats.py --config-name pi05_egoverse
```

Writes to `assets/pi05_egoverse/egoverse/all/norm_stats.json`.

The norm stats are also backed up in the checkpoints directory:
```
/cluster/work/cvg/data/rytsui/checkpoints/pi05_egoverse/test/29999/assets/egoverse/all/norm_stats.json
```

### Download base weights

The fine-tuning starts from pi0.5 base weights. Download once on the login node (no compute needed):

```bash
pip install --user gcsfs
python3 -c "
import gcsfs
fs = gcsfs.GCSFileSystem(token='anon')
fs.get('openpi-assets/checkpoints/pi05_base/params',
       '/cluster/work/cvg/data/Egoverse/pi05_base_jax/params',
       recursive=True)
print('Done')
"
```

---

## 5. Phase 2: Training (Fine-tuning pi0.5)

### Training config

Config name: `pi05_egoverse` (defined in `src/openpi/training/config.py`).

Key settings:
- Model: pi0.5 with LoRA on both Gemma 2B backbone (`gemma_2b_lora`) and Gemma 300M action head (`gemma_300m_lora`)
- Data: `LeRobotEgoverseDataConfig` pointing to `egoverse/all`
- Loss: flow-matching diffusion on normalized action chunks

### Submit training job

```bash
sbatch --partition=gpu.120h --time=120:00:00 --mem-per-cpu=16G \
  --cpus-per-task=8 --gpus=a100:1 \
  3dvision-experiments/run.slurm [config] [exp_name] [seed]
```

Defaults: `pi05_egoverse test 42`.

The training script (`scripts/train.py`) automatically resumes from the latest checkpoint if one exists (`--resume` flag is always set in `run.slurm`).

**Hardware requirement:** A100 (40 GB VRAM) required for LoRA training with batch_size=32. RTX 3090 and below OOM during the backward pass.

### Monitor training

```bash
tail -f slurm-<jobid>.out
sacct -j <jobid> --format=JobID,State,Elapsed,AllocTRES%60
```

WandB is disabled on compute nodes (no internet). All metrics go to the SLURM output file.

### Checkpoints

Checkpoints are saved every 5k steps to:
```
/cluster/scratch/rytsui/checkpoints/pi05_egoverse/test/<step>/
```

The used checkpoint for all evaluations is **step 29999**, stored at:
```
/cluster/work/cvg/data/rytsui/checkpoints/pi05_egoverse/test/29999/
```
(This copy was moved to work storage for persistence — scratch would purge it.)

---

## 6. Phase 3: Baseline Inference on Real Data

Before evaluating in sim, test the model on real held-out frames to establish a quantitative baseline.

### What it does

`run_inference.py` loads either the base model or a fine-tuned checkpoint, runs it on real Aria frames from h5 episodes, and reports per-step MSE against ground-truth actions. Also computes two naive baselines:
- `zero_action`: predict all-zeros. Measures action variance.
- `const_state`: predict current qpos as action. The "do nothing" baseline.

### Submit

```bash
# Single episode, every 10th frame:
sbatch --time=01:00:00 --mem-per-cpu=16G --cpus-per-task=8 --gpus=1 \
  3dvision-experiments/run_inference.slurm \
  --h5-path /cluster/work/cvg/data/Egoverse/raw_timesynced_h5/object_in_bowl_processed_50hz/20250804_142656.h5 \
  --frame-stride 10

# All 78 episodes:
sbatch --time=02:00:00 --mem-per-cpu=16G --cpus-per-task=8 --gpus=1 \
  3dvision-experiments/run_inference.slurm \
  --episodes-dir /cluster/work/cvg/data/Egoverse/raw_timesynced_h5/object_in_bowl_processed_50hz \
  --frame-stride 10

# Quick smoke test (8 frames):
sbatch --time=00:30:00 --mem-per-cpu=16G --cpus-per-task=8 --gpus=1 \
  3dvision-experiments/run_inference.slurm \
  --h5-path <path> --num-frames 8 --frame-stride 50
```

### Baseline results (2026-04-12, base pi0.5, no fine-tuning)

Typical per-chunk MSE across 78 episodes:
- **Arm MSE**: ~0.03–0.15 (varies per episode/chunk)
- **Hand MSE**: ~0.10–0.50

These numbers establish what to expect from the pre-trained model. Fine-tuning should lower arm MSE on this specific task.

---

## 7. Phase 4: Isaac Sim Evaluation

### 7.1 Overview

The eval runs `eval_script_1.py` inside an Apptainer container (Isaac Sim 4.5.0) on a GPU node, via SLURM. The robot observes `kitchen_scene_1.usd` through a simulated camera, the policy predicts arm actions, and the result is saved as `evaluation.mp4` + `results.csv`.

Timing (RTX 3090, 3000 steps):

| Phase | Duration |
|-------|---------|
| Isaac Sim boot | ~72 s |
| Policy load + JAX init | ~30 s |
| Scene open + asset patches + warmup | ~30 s |
| Step 0 — JAX JIT compile | ~49 s |
| Steps 1–2999 (~160 ms/step) | ~8 min |
| **Total** | **~12 min** |

### 7.2 First-time Setup (new user)

**Prerequisites:**
- Euler cluster account
- CVG group access (to read `/cluster/work/cvg/`)
- `isaac-sim_4.5.0.sif` (6 GB) obtained via link from the project owner — place it at `/cluster/scratch/$USER/pi0_test/isaac-sim_4.5.0.sif`

Run all of the following on the **login node** (compute nodes have no internet).

#### Step 1 — Clone repo and create directories

```bash
cd /cluster/scratch/$USER
git clone https://github.com/dalongbao/openpi.git
mkdir -p pi0_test/assets isaac_packages isaac_cache/kit isaac_cache/ov_home
```

#### Step 2 — Copy scene file and submit script

```bash
cp openpi/3dvision-experiments/isaac-sim/kitchen_scene_1.usd pi0_test/
cp openpi/3dvision-experiments/isaac-sim/submit.sh submit.sh
```

#### Step 3 — Restore norm stats

```bash
mkdir -p openpi/assets/pi05_egoverse/egoverse/all
cp /cluster/work/cvg/data/rytsui/checkpoints/pi05_egoverse/test/29999/assets/egoverse/all/norm_stats.json \
   openpi/assets/pi05_egoverse/egoverse/all/
```

#### Step 4 — Download USD scene assets from Omniverse CDN

```bash
export HTTP_PROXY=http://proxy.ethz.ch:3128
export HTTPS_PROXY=http://proxy.ethz.ch:3128

ASSETS=/cluster/scratch/$USER/pi0_test/assets
BASE5=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac
ARCHVIS=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/ArchVis

mkdir -p $ASSETS/fr3_full/configuration $ASSETS/crate $ASSETS/plate $ASSETS/table

wget -q -O $ASSETS/fr3_full/fr3.usd "$BASE5/Robots/FrankaRobotics/FrankaFR3/fr3.usd"
wget -q -O $ASSETS/fr3_full/configuration/fr3_robot_schema.usd "$BASE5/Robots/FrankaRobotics/FrankaFR3/configuration/fr3_robot_schema.usd"
wget -q -O $ASSETS/crate/SM_Crate_A07_Yellow_01_physics.usd "$BASE5/IsaacLab/Mimic/pick_place_task/pick_place_assets/PackingTable/props/SM_Crate_A07_Yellow_01/SM_Crate_A07_Yellow_01_physics.usd"
wget -q -O $ASSETS/crate/SM_Crate_A07_Yellow_01.usd "$BASE5/IsaacLab/Mimic/pick_place_task/pick_place_assets/PackingTable/props/SM_Crate_A07_Yellow_01/SM_Crate_A07_Yellow_01.usd"
wget -q -O $ASSETS/plate/plate_small.usd "$ARCHVIS/Residential/Kitchen/Kitchenware/Dinnerware/plate_small.usd"
wget -q -O $ASSETS/table/SM_HeavyDutyPackingTable_C02_01.usd "$BASE5/IsaacLab/Mimic/g1_squatting_task/PackingTable/props/SM_HeavyDutyPackingTable_C02_01/SM_HeavyDutyPackingTable_C02_01.usd"
```

#### Step 4b — Download MDL shaders and textures

Without MDL files, all objects render as plain white geometry. The policy fails immediately because the visual distribution is completely wrong.

```bash
TABLE_BASE="$BASE5/IsaacLab/Mimic/g1_squatting_task/PackingTable/props/SM_HeavyDutyPackingTable_C02_01"
CRATE_BASE="$BASE5/IsaacLab/Mimic/pick_place_task/pick_place_assets/PackingTable/props/SM_Crate_A07_Yellow_01"

mkdir -p $ASSETS/table/materials/textures/{Metal_Glossy_A,MetalPainted_White_Glossy_A,Wood_Laminate_Gray_A,Wood_Pressed_A}
for mdl in MetalPainted_White_Glossy_A Wood_Laminate_Gray_A Metal_Glossy_A Wood_Pressed_A; do
  wget -q -O "$ASSETS/table/materials/${mdl}.mdl" "${TABLE_BASE}/materials/${mdl}.mdl"
done
cp "$ASSETS/table/materials/MetalPainted_White_Glossy_A.mdl" "$ASSETS/table/materials/MetalPainted_Gray_Glossy_A.mdl"

for mat in Metal_Glossy_A MetalPainted_White_Glossy_A; do
  for suf in Albedo Normal ORM; do
    wget -q -O "$ASSETS/table/materials/textures/$mat/T_${mat}_${suf}.png" "${TABLE_BASE}/materials/textures/$mat/T_${mat}_${suf}.png"
  done
done
for tex in T_Wood_Laminate_Gray_A_Albedo.png T_Wood_Laminate_Gray_A_ORM.png T_Wood_Laminate_Gray_A_Normal.png; do
  wget -q -O "$ASSETS/table/materials/textures/Wood_Laminate_Gray_A/$tex" "${TABLE_BASE}/materials/textures/Wood_Laminate_Gray_A/$tex"
done
for tex in T_Wood_Pressed_A1_Albedo.png T_Wood_Pressed_A1_Normal.png T_Wood_Pressed_A1_ORM.png; do
  wget -q -O "$ASSETS/table/materials/textures/Wood_Pressed_A/$tex" "${TABLE_BASE}/materials/textures/Wood_Pressed_A/$tex"
done

mkdir -p $ASSETS/crate/materials/textures/Plastic_Yellow_A
wget -q -O "$ASSETS/crate/materials/Plastic_Yellow_A.mdl" "${CRATE_BASE}/materials/Plastic_Yellow_A.mdl"
for suf in Albedo Normal ORM; do
  wget -q -O "$ASSETS/crate/materials/textures/Plastic_Yellow_A/T_Plastic_Yellow_A_${suf}.png" "${CRATE_BASE}/materials/textures/Plastic_Yellow_A/T_Plastic_Yellow_A_${suf}.png"
done

mkdir -p $ASSETS/plate/Textures
for tex in BaseColor Metallic Normal Roughness; do
  wget -q -O "$ASSETS/plate/Textures/Plates_Small_${tex}.png" \
    "$ARCHVIS/Residential/Kitchen/Kitchenware/Dinnerware/Textures/Plates_Small_${tex}.png"
done
```

#### Step 5 — Install Python packages into isaac_packages

Isaac Sim uses Python 3.10. Do **not** copy packages from another user — reinstall fresh inside the container. Takes ~10 minutes.

```bash
APPTAINERENV_ACCEPT_EULA=Y \
APPTAINERENV_HTTP_PROXY=http://proxy.ethz.ch:3128 \
APPTAINERENV_HTTPS_PROXY=http://proxy.ethz.ch:3128 \
apptainer exec \
  --bind /cluster/scratch/$USER/isaac_packages:/target \
  /cluster/scratch/$USER/pi0_test/isaac-sim_4.5.0.sif \
  /isaac-sim/python.sh -m pip install --no-cache-dir --target /target \
  "jax[cuda12]==0.5.3" "flax==0.10.2" "jaxtyping==0.2.36" \
  "orbax-checkpoint==0.11.13" "numpy==1.26.4" "beartype==0.19.0" \
  "ml_collections==1.0.0" "chex>=0.1.86" "augmax>=0.3.4" \
  "dm-tree>=0.1.8" "einops>=0.8.0" "equinox>=0.11.8" \
  "flatbuffers>=24.3.25" "gcsfs>=2024.6.0" "imageio>=2.36.1" \
  "numpydantic>=1.6.6" "pillow>=11.0.0" "sentencepiece>=0.2.0" \
  "tqdm-loggable>=0.2" "tyro>=0.9.5" "wandb>=0.19.1" \
  "filelock>=3.16.1" "treescope>=0.1.7" "polars>=1.30.0" \
  "transformers==4.53.2" "draccus" "pytest>=8.3.4" "rich>=14.0.0"
```

**Critical version pins — changing these breaks the pipeline:**

| Package | Version | Why |
|---------|---------|-----|
| `jax` | `0.5.3` | Locked in uv.lock; 0.4.x silently computes wrong outputs |
| `jaxtyping` | `0.2.36` | 0.3.x removed `_check_dataclass_annotations` used in `array_typing.py` |
| `orbax-checkpoint` | `0.11.13` | Checkpoint format compatibility |
| `numpy` | `1.26.4` | numpy 2.x breaks Isaac Sim's internal numba usage |
| `transformers` | `4.53.2` | API compatibility with openpi's tokenizer code |

**Verify the install:**
```bash
APPTAINERENV_ACCEPT_EULA=Y apptainer exec \
  --bind /cluster/scratch/$USER/isaac_packages:/isaac_packages \
  /cluster/scratch/$USER/pi0_test/isaac-sim_4.5.0.sif \
  /isaac-sim/python.sh -c "
import sys; sys.path.insert(0, '/isaac_packages')
import jax, jaxtyping, orbax.checkpoint
print('jax', jax.__version__)
print('jaxtyping', jaxtyping.__version__)
"
```

### 7.3 Running an Evaluation

#### Standard submit command

Pull the latest eval script from the repo and submit:

```bash
cd /cluster/scratch/$USER/openpi && git pull && \
cp 3dvision-experiments/isaac-sim/eval_script_1.py /cluster/scratch/$USER/pi0_test/eval_script_1.py && \
sed -i 's/\r//' /cluster/scratch/$USER/submit.sh && \
sbatch --partition=gpu.4h --time=00:30:00 --mem-per-cpu=8G --cpus-per-task=8 --gpus=rtx_3090:1 \
  /cluster/scratch/$USER/submit.sh
```

`submit.sh` uses `$USER` throughout — no path edits needed for any user.

#### Monitoring

```bash
squeue -u $USER                                          # check status, note NODELIST
tail -f /cluster/scratch/$USER/slurm-<JOBID>.out         # live log
```

#### Download results

```bash
rsync -avP $USER@euler.ethz.ch:/cluster/scratch/$USER/pi0_test/evaluation.mp4 ./
rsync -avP $USER@euler.ethz.ch:/cluster/scratch/$USER/pi0_test/results.csv ./
```

### 7.4 How eval_script_1.py Works

#### Startup sequence

1. `SimulationApp` is instantiated **before any other import** — Isaac Sim requires this as the absolute first line.
2. After app init, `sys.path` is extended with `/workspace/openpi/src`, `/workspace/openpi/packages/openpi-client/src`, `/isaac_packages`.
3. `typing_extensions` is force-reloaded from `/isaac_packages`. Isaac Sim caches an ancient version in `sys.modules` at startup; without this fix, `from typing_extensions import NoDefault` fails.
4. The `pi05_egoverse` config is loaded and checkpoint `29999` is read from `/checkpoints/pi05_egoverse/test/29999` in orbax format.

#### Scene loading and asset patching

`kitchen_scene_1.usd` contains the Franka FR3, table, plate, and crate. However, the USD payload references for the table, plate, crate, and robot all point to Omniverse S3 URLs — compute nodes have no internet. `eval_script_1.py` patches each USD prim's payload at runtime to point to local files under `/workspace/assets/`.

#### Joint permutation

Isaac Sim's Franka FR3 reports joints in a different order than the Egoverse training data. Specifically, joint indices 3 and 5 are swapped. A self-inverse permutation `[0,1,2,5,4,3,6]` is applied in both directions:

```python
_PERM = [0, 1, 2, 5, 4, 3, 6]

# Reading state for the policy (sim → training frame):
obs_joints = sim_joints[_PERM]

# Writing action back to sim (training → sim frame):
sim_targets = policy_action[_PERM]
```

Without this, the policy receives out-of-distribution joint states and outputs meaningless actions.

#### Home position warmup

Before the main policy loop, the robot is driven to the training home position over 100 warmup steps:
```python
HOME_TRAIN = [0.476, 0.120, 0.292, 0.994, 0.105, -0.025, -0.009]
```
Joint j3 (sim idx 3, FR3 j4) has target -0.025 which is outside the sim's joint limit [-3.04, -0.152] and gets clamped to -0.152. Joint j5 only reaches ~0.689 in 100 steps but the policy drives it further within the first 50 policy steps.

#### EMA smoothing

Actions from the policy are smoothed before application to reduce chunk-boundary jitter:
```python
smoothed_cmd = 0.4 * full_cmd + 0.6 * smoothed_cmd
franka.apply_action(ArticulationAction(joint_positions=smoothed_cmd))
```

#### Camera setup

- `ExternalCamera` (224×224): positioned at `(0.7, -1.5, 2.0)` with 20° downward tilt, oriented toward the workspace. Applied via USD `pxr.UsdGeom` at runtime using `Gf.Rotation` / `Gf.Quatd`. Policy receives this frame every step.
- `RecordingCamera` (1280×720): defined statically in the USD scene. Writes every frame to `evaluation.mp4` via OpenCV.

Diagnostic images are saved at init: `policy_cam_init.png` and `policy_cam_step0.png`. Inspect these to confirm the camera has a sensible view of the scene.

#### Bind mounts (from submit.sh)

| Host path | Container path | Purpose |
|-----------|----------------|---------|
| `/cluster/scratch/$USER/pi0_test` | `/workspace` | Scene, assets, outputs |
| `/cluster/scratch/$USER/openpi` | `/workspace/openpi` | Source code + norm stats |
| `/cluster/work/cvg/data/rytsui/checkpoints` | `/checkpoints` | Trained checkpoint |
| `$ISAAC_SIM_CACHE_DIR/kit` | `/isaac-sim/kit/cache` | Writable shader cache (required) |
| `$ISAAC_SIM_CACHE_DIR/ov_home` | `/cluster/home/$USER` | Overrides read-only home for Warp/texture cache |
| `/cluster/scratch/$USER/isaac_packages` | `/isaac_packages` | Python 3.10 deps |

The `ov_home` bind is critical: Isaac Sim writes Warp kernel caches and texture data using the C library `getpwuid()` (not `$HOME`), which resolves to `/cluster/home/$USER`. That path is read-only on compute nodes. Binding a writable scratch directory over it prevents crashes.

### 7.5 Recovering After Scratch Purge

Euler auto-deletes files not accessed in 15 days. When this happens, run these recovery commands on the login node in order:

```bash
export HTTP_PROXY=http://proxy.ethz.ch:3128
export HTTPS_PROXY=http://proxy.ethz.ch:3128

# 1. Norm stats (from persistent work storage)
mkdir -p /cluster/scratch/$USER/openpi/assets/pi05_egoverse/egoverse/all
cp /cluster/work/cvg/data/rytsui/checkpoints/pi05_egoverse/test/29999/assets/egoverse/all/norm_stats.json \
   /cluster/scratch/$USER/openpi/assets/pi05_egoverse/egoverse/all/

# 2. USD scene (from repo)
cp /cluster/scratch/$USER/openpi/3dvision-experiments/isaac-sim/kitchen_scene_1.usd \
   /cluster/scratch/$USER/pi0_test/

# 3. USD mesh assets (robot, table, crate, plate)
ASSETS=/cluster/scratch/$USER/pi0_test/assets
BASE5=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac
ARCHVIS=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/ArchVis
mkdir -p $ASSETS/fr3_full/configuration $ASSETS/crate $ASSETS/plate $ASSETS/table
wget -q -O $ASSETS/fr3_full/fr3.usd "$BASE5/Robots/FrankaRobotics/FrankaFR3/fr3.usd"
wget -q -O $ASSETS/fr3_full/configuration/fr3_robot_schema.usd "$BASE5/Robots/FrankaRobotics/FrankaFR3/configuration/fr3_robot_schema.usd"
wget -q -O $ASSETS/crate/SM_Crate_A07_Yellow_01_physics.usd "$BASE5/IsaacLab/Mimic/pick_place_task/pick_place_assets/PackingTable/props/SM_Crate_A07_Yellow_01/SM_Crate_A07_Yellow_01_physics.usd"
wget -q -O $ASSETS/crate/SM_Crate_A07_Yellow_01.usd "$BASE5/IsaacLab/Mimic/pick_place_task/pick_place_assets/PackingTable/props/SM_Crate_A07_Yellow_01/SM_Crate_A07_Yellow_01.usd"
wget -q -O $ASSETS/plate/plate_small.usd "$ARCHVIS/Residential/Kitchen/Kitchenware/Dinnerware/plate_small.usd"
wget -q -O $ASSETS/table/SM_HeavyDutyPackingTable_C02_01.usd "$BASE5/IsaacLab/Mimic/g1_squatting_task/PackingTable/props/SM_HeavyDutyPackingTable_C02_01/SM_HeavyDutyPackingTable_C02_01.usd"

# 4. MDL shaders and textures
TABLE_BASE="$BASE5/IsaacLab/Mimic/g1_squatting_task/PackingTable/props/SM_HeavyDutyPackingTable_C02_01"
CRATE_BASE="$BASE5/IsaacLab/Mimic/pick_place_task/pick_place_assets/PackingTable/props/SM_Crate_A07_Yellow_01"
mkdir -p $ASSETS/table/materials/textures/{Metal_Glossy_A,MetalPainted_White_Glossy_A,Wood_Laminate_Gray_A,Wood_Pressed_A}
for mdl in MetalPainted_White_Glossy_A Wood_Laminate_Gray_A Metal_Glossy_A Wood_Pressed_A; do wget -q -O "$ASSETS/table/materials/${mdl}.mdl" "${TABLE_BASE}/materials/${mdl}.mdl"; done
cp "$ASSETS/table/materials/MetalPainted_White_Glossy_A.mdl" "$ASSETS/table/materials/MetalPainted_Gray_Glossy_A.mdl"
for mat in Metal_Glossy_A MetalPainted_White_Glossy_A; do for suf in Albedo Normal ORM; do wget -q -O "$ASSETS/table/materials/textures/$mat/T_${mat}_${suf}.png" "${TABLE_BASE}/materials/textures/$mat/T_${mat}_${suf}.png"; done; done
for tex in T_Wood_Laminate_Gray_A_Albedo.png T_Wood_Laminate_Gray_A_ORM.png T_Wood_Laminate_Gray_A_Normal.png; do wget -q -O "$ASSETS/table/materials/textures/Wood_Laminate_Gray_A/$tex" "${TABLE_BASE}/materials/textures/Wood_Laminate_Gray_A/$tex"; done
for tex in T_Wood_Pressed_A1_Albedo.png T_Wood_Pressed_A1_Normal.png T_Wood_Pressed_A1_ORM.png; do wget -q -O "$ASSETS/table/materials/textures/Wood_Pressed_A/$tex" "${TABLE_BASE}/materials/textures/Wood_Pressed_A/$tex"; done
mkdir -p $ASSETS/crate/materials/textures/Plastic_Yellow_A
wget -q -O "$ASSETS/crate/materials/Plastic_Yellow_A.mdl" "${CRATE_BASE}/materials/Plastic_Yellow_A.mdl"
for suf in Albedo Normal ORM; do wget -q -O "$ASSETS/crate/materials/textures/Plastic_Yellow_A/T_Plastic_Yellow_A_${suf}.png" "${CRATE_BASE}/materials/textures/Plastic_Yellow_A/T_Plastic_Yellow_A_${suf}.png"; done
mkdir -p $ASSETS/plate/Textures
for tex in BaseColor Metallic Normal Roughness; do wget -q -O "$ASSETS/plate/Textures/Plates_Small_${tex}.png" "$ARCHVIS/Residential/Kitchen/Kitchenware/Dinnerware/Textures/Plates_Small_${tex}.png"; done

# 5. Reinstall Python packages (takes ~10 min)
# See §7.2 Step 5 above for the full pip install command
```

`isaac_packages/` may or may not have been purged — check with `ls /cluster/scratch/$USER/isaac_packages/`. If empty, reinstall (§7.2 Step 5). If present, verify with the version check command in §7.2 Step 5.

**Note:** `isaac_packages/` was not purged in the 2026-05-26 scratch purge because it had been recently accessed. Don't count on this.

### 7.6 Current Status and Open Problems

**As of 2026-05-26:** The pipeline runs end-to-end cleanly (3000 steps, exit 0, ~12 min). The robot reaches the correct starting joint distribution. However, **the arm barely moves during the policy loop** — joint positions stay within ±0.05 rad of the home position.

**Root cause (hypothesis):** Sim-to-real visual domain gap. The model was trained on real Aria egocentric images (480×640, human hand visible, real-world lighting). The sim camera produces rendered 3D geometry — a completely different visual distribution. For out-of-distribution inputs, the policy collapses to predicting training-mean actions, which look like "do almost nothing."

**What has been tried and verified working:**
- Joint permutation `[0,1,2,5,4,3,6]` — correct, confirmed by joint range matching
- Home position warmup — robot reaches training joint distribution before policy loop
- EMA smoothing — reduces jitter but doesn't fix the core issue
- Full scene textures (MDL shaders + PNG textures for all objects)
- Language command matches training label: `"put the object in the bowl"`
- Camera repositioned from bird's-eye to eye-level (0.7, -1.5, 2.0)

**Likely next directions:**
1. Match camera FoV to Aria RGB (76° hFoV) — currently using Isaac Sim default (~60°)
2. Domain randomization or photorealistic rendering in the sim
3. Evaluate on real Franka hardware to separate policy quality from domain gap
4. Add real Aria-style hand overlay in the sim camera frame

---

## 8. Shared File Paths Reference

### Datasets

| What | Path |
|------|------|
| Raw h5 data (object_in_bowl, 78 eps) | `/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/object_in_bowl_processed_50hz/` |
| Raw h5 data (bag_groceries, 300 eps) | `/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries/` |
| Converted LeRobot dataset (object_in_bowl, 5 eps) | `/cluster/work/cvg/data/Egoverse/lerobot_egoverse/egoverse/all/` |
| Converted LeRobot dataset (bag_grocery_human, 1683 eps) | `/cluster/work/cvg/data/Egoverse/lerobot_egoverse/egoverse/bag_grocery_human/` |
| Converted LeRobot dataset (oic_human, 2537 eps) | `/cluster/work/cvg/data/Egoverse/lerobot_egoverse/egoverse/oic_human/` |
| jiaqchen's raw LeRobot data | `/cluster/work/cvg/jiaqchen/EGOVERSE_DATA_3DV/` |

### Model weights and checkpoints

| What | Path |
|------|------|
| pi0.5 base weights (JAX/orbax) | `/cluster/work/cvg/data/Egoverse/pi05_base_jax/params` |
| Fine-tuned checkpoint (step 29999) | `/cluster/work/cvg/data/rytsui/checkpoints/pi05_egoverse/test/29999/` |
| Norm stats (backed up in checkpoint) | `/cluster/work/cvg/data/rytsui/checkpoints/pi05_egoverse/test/29999/assets/egoverse/all/norm_stats.json` |

### Per-user scratch paths (kdoman's current setup)

| What | Path |
|------|------|
| openpi repo | `/cluster/scratch/kdoman/openpi/` |
| Isaac Sim container | `/cluster/scratch/kdoman/pi0_test/isaac-sim_4.5.0.sif` |
| USD scene + assets | `/cluster/scratch/kdoman/pi0_test/` |
| Python packages (Isaac) | `/cluster/scratch/kdoman/isaac_packages/` |
| Shader cache | `/cluster/scratch/kdoman/isaac_cache/` |
| SLURM submit script | `/cluster/scratch/kdoman/submit.sh` |
| Eval outputs | `/cluster/scratch/kdoman/pi0_test/evaluation.mp4`, `results.csv` |

---

## 9. Troubleshooting

### Isaac Sim / container issues

| Symptom | Fix |
|---------|-----|
| Container hangs at startup (>10 min, no output) | Kit shader cache is read-only. `submit.sh` must bind `$ISAAC_SIM_CACHE_DIR/kit` → `/isaac-sim/kit/cache`. Already in `submit.sh`. |
| `Read-only file system: '/cluster/home'` in Warp or texture logs | Isaac Sim uses `getpwuid()` not `$HOME`. Bind a writable scratch dir over `/cluster/home/$USER`. Already in `submit.sh`. |
| `ModuleNotFoundError` after ~73 s (after Isaac boots successfully) | `isaac_packages/` was wiped or not reinstalled. The job runs fine for ~73 s while Isaac Sim boots, then crashes on the first openpi import. Reinstall immediately. |
| `cannot import name 'NoDefault' from 'typing_extensions'` | Isaac Sim caches an ancient `typing_extensions` in `sys.modules`. `eval_script_1.py` force-reloads the correct version at the top — don't remove that block. |
| `AttributeError: module 'datetime' has no attribute 'UTC'` | Python 3.10 compat issue. Fixed in `src/openpi/shared/download.py` — uses `datetime.timezone.utc`. |
| `franka.get_joint_positions()` returns `None` at step 0 | After `world.reset()` the articulation view is cleared. `eval_script_1.py` calls `franka.initialize()` immediately after reset. Don't remove this. |
| Objects render as plain white | MDL shaders and texture PNGs missing. Run recovery step 4 above. |
| Policy outputs near-constant actions, robot barely moves | Sim-to-real domain gap — see §7.6. Not a bug. |

### Python package issues

| Symptom | Fix |
|---------|-----|
| Wrong `jaxtyping` version despite reinstall | `chex` (transitive dep via flax → optax) pulls in latest jaxtyping. Wipe `$ASSETS/jaxtyping*` dirs and reinstall with the pinned version. |
| `jaxtyping` loads from wrong path | Check `import jaxtyping; print(jaxtyping.__file__)`. It should be under `/isaac_packages/`. If it's elsewhere, the wrong version is shadowing it. |
| `lerobot` import error | `lerobot` must not be installed in `isaac_packages`. The import was moved inside its function in `data_loader.py`. Don't add lerobot to the container install. |
| Packages present but wrong versions | Use the verify command in §7.2 Step 5. Wipe and reinstall if needed. |

### SLURM / cluster issues

| Symptom | Fix |
|---------|-----|
| `--mem` flag rejected | Euler only accepts `--mem-per-cpu`. Never use `--mem=`. |
| `#SBATCH` directives ignored | Euler quirk — pass ALL flags on the `sbatch` command line. Never use in-script `#SBATCH`. |
| SLURM log not found in `~/` | Log goes to CWD of the `sbatch` invocation. Check `/cluster/scratch/$USER/`. |
| Script has Windows line endings (`\r`) | Run `sed -i 's/\r//' submit.sh` before submitting. |
| `gpuhe.4h` job never starts | Partition is busy. Switch to `gpu.4h --gpus=rtx_3090:1`. 24 GB VRAM is sufficient for inference. |
| Job starts but `pip install` in container fails with DNS errors | Container has no internet. Add `APPTAINERENV_HTTP_PROXY` and `APPTAINERENV_HTTPS_PROXY` before `apptainer exec`. Already set in setup commands above. |

### Common git / code issues

| Symptom | Fix |
|---------|-----|
| `eval_script_1.py` crashes with undefined `CAMERA_RES`, `ext_img`, or `prepare_image` | A bad commit was merged. Check git log — commits d087bfb and bc49b04 introduced broken camera code. Verify variable names match constants in the CONFIG block after every pull. |
| Norm stats not found | Re-run the `cp` command in §7.2 Step 3 or the recovery step 1. |
