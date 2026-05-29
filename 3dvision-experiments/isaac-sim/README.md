# Isaac Sim Evaluation — pi0.5 on Franka FR3 (Euler HPC)

Run a trained pi0.5 checkpoint in closed-loop inside Isaac Sim 4.5.0 on ETH Euler, producing `evaluation.mp4` and `results.csv`.

---

## What this does

1. Loads `kitchen_scene_1.usd` — a Franka FR3 arm, table, plate, and yellow crate.
2. Runs `eval_script_1.py` in a closed loop at ~50 Hz for 3000 steps (60 s):
   - Grabs a 224×224 frame from `ExternalCamera` → feeds to pi0.5 → applies 7-DOF arm + gripper action.
   - Writes HD video from `RecordingCamera` (1280×720) to `evaluation.mp4`.
   - Logs joint positions to `results.csv`.
3. All of this runs inside an Apptainer container on a GPU node via SLURM.

---

## File layout

```
3dvision-experiments/isaac-sim/
├── eval_script_1.py      # main eval loop (edit this to change policy/scene behaviour)
├── submit.sh             # apptainer + SLURM wrapper
├── kitchen_scene_1.usd   # Isaac Sim scene (Franka FR3 + objects + cameras)
└── README.md             # this file
```

Files that live in `/cluster/scratch/$USER/` (not in the repo — too large or auto-generated):

| Path | Size | Purpose |
|------|------|---------|
| `pi0_test/isaac-sim_4.5.0.sif` | ~6 GB | Apptainer container |
| `pi0_test/kitchen_scene_1.usd` | small | copy from repo |
| `pi0_test/assets/` | ~50 MB | local USD mesh + texture files |
| `isaac_packages/` | ~2 GB | Python deps installed into Isaac Sim's Python 3.10 |
| `isaac_cache/` | auto | shader + Warp kernel cache |

---

## One-time setup (new user)

Run everything on the **login node** — compute nodes have no internet.

### 1. Clone repo and copy scene file

```bash
git clone <repo-url> /cluster/scratch/$USER/openpi
mkdir -p /cluster/scratch/$USER/pi0_test/assets
cp /cluster/scratch/$USER/openpi/3dvision-experiments/isaac-sim/kitchen_scene_1.usd /cluster/scratch/$USER/pi0_test/
```

### 2. Get the Isaac Sim container

Copy from an existing user (fast, same filesystem):

```bash
cp /cluster/scratch/kdoman/pi0_test/isaac-sim_4.5.0.sif /cluster/scratch/$USER/pi0_test/
```

### 3. Get USD scene assets

Copy from an existing user's scratch (saves re-downloading):

```bash
cp -r /cluster/scratch/kdoman/pi0_test/assets /cluster/scratch/$USER/pi0_test/
```

If the source has been purged, see [Recovering after scratch purge](#recovering-after-scratch-purge).

### 4. Install Python packages into `isaac_packages/`

Isaac Sim uses Python 3.10 — the repo's uv venv (Python 3.11) is ABI-incompatible. Install separately:

```bash
APPTAINERENV_ACCEPT_EULA=Y APPTAINERENV_HTTP_PROXY=http://proxy.ethz.ch:3128 APPTAINERENV_HTTPS_PROXY=http://proxy.ethz.ch:3128 apptainer exec --bind /cluster/scratch/$USER/isaac_packages:/target /cluster/scratch/$USER/pi0_test/isaac-sim_4.5.0.sif /isaac-sim/python.sh -m pip install --no-cache-dir --target /target "jax[cuda12]==0.5.3" "flax==0.10.2" "jaxtyping==0.2.36" "orbax-checkpoint==0.11.13" "numpy==1.26.4" "beartype==0.19.0" "ml_collections==1.0.0" "chex>=0.1.86" "augmax>=0.3.4" "dm-tree>=0.1.8" "einops>=0.8.0" "equinox>=0.11.8" "flatbuffers>=24.3.25" "gcsfs>=2024.6.0" "imageio>=2.36.1" "numpydantic>=1.6.6" "pillow>=11.0.0" "sentencepiece>=0.2.0" "tqdm-loggable>=0.2" "tyro>=0.9.5" "wandb>=0.19.1" "filelock>=3.16.1" "treescope>=0.1.7" "polars>=1.30.0" "transformers==4.53.2" "draccus" "pytest>=8.3.4" "rich>=14.0.0"
```

**Critical version pins** — do not change these:

| Package | Pin | Reason |
|---------|-----|--------|
| `jax` | `0.5.3` | uv.lock pin; 0.4.x silently wrong |
| `jaxtyping` | `0.2.36` | 0.3.x removed `_check_dataclass_annotations` |
| `orbax-checkpoint` | `0.11.13` | checkpoint format compatibility |
| `numpy` | `1.26.4` | numpy 2.x breaks Isaac Sim's numba |
| `beartype` | `0.19.0` | uv.lock pin |

**Do not copy `isaac_packages/` between users** — reinstall is faster and guaranteed correct.

### 5. Compute norm stats (uses uv venv, not isaac_packages)

```bash
cd /cluster/scratch/$USER/openpi && uv sync && HF_LEROBOT_HOME=/cluster/work/cvg/data/Egoverse/lerobot_egoverse uv run python scripts/compute_norm_stats.py --config-name pi05_egoverse
```

Writes to `assets/pi05_egoverse/egoverse/all/norm_stats.json` — required by the eval script.

---

## Running an evaluation

### Standard submit (pull latest, then submit)

```bash
cd /cluster/scratch/$USER/openpi && git pull && cp 3dvision-experiments/isaac-sim/eval_script_1.py /cluster/scratch/$USER/pi0_test/eval_script_1.py && sed -i 's/\r//' /cluster/scratch/$USER/pi0_test/../submit.sh 2>/dev/null; sbatch --partition=gpu.4h --time=00:30:00 --mem-per-cpu=8G --cpus-per-task=8 --gpus=rtx_3090:1 /cluster/scratch/$USER/openpi/3dvision-experiments/isaac-sim/submit.sh
```

Use `gpu.4h + rtx_3090:1` (24 GB VRAM, fast scheduling). Full 3000-step run takes ~12 min; `--time=00:30:00` is safe.

### Smoke test (5 min, confirms everything loads)

```bash
sbatch --partition=gpu.4h --time=00:05:00 --mem-per-cpu=8G --cpus-per-task=8 --gpus=rtx_3090:1 /cluster/scratch/$USER/openpi/3dvision-experiments/isaac-sim/submit.sh
```

### Monitor the job

```bash
squeue -u $USER                             # note JOBID and NODELIST
tail -f /cluster/scratch/$USER/slurm-<JOBID>.out   # SLURM log goes to CWD of sbatch
```

### Download results

```bash
rsync -avP $USER@euler.ethz.ch:/cluster/scratch/$USER/pi0_test/evaluation.mp4 ./
rsync -avP $USER@euler.ethz.ch:/cluster/scratch/$USER/pi0_test/results.csv ./
```

---

## Timing profile (RTX 3090, 3000 steps)

| Phase | Time |
|-------|------|
| Isaac Sim boot | ~72 s |
| Policy load + JAX backend init | ~30 s |
| Scene open + asset patches + warmup | ~30 s |
| Step 0 (JAX JIT compile) | ~49 s |
| Steps 1–2999 (~160 ms/step) | ~8 min |
| **Total** | **~12 min** |

---

## How `eval_script_1.py` works

1. `SimulationApp` starts first (Isaac Sim requirement — must be line 1).
2. `sys.path` is extended with openpi src, openpi-client src, `/isaac_packages`.
3. `typing_extensions` is force-reloaded from `/isaac_packages` — Isaac Sim caches an ancient version at startup that would otherwise shadow the correct one.
4. `pi05_egoverse` config loaded; checkpoint from `/checkpoints/pi05_egoverse/test/29999` (orbax format).
5. `kitchen_scene_1.usd` opened; S3 payload URLs for table/plate/crate/fr3 patched to local paths under `/workspace/assets/`.
6. `ExternalCamera` repositioned programmatically to eye-level (see [Current issues](#current-issues)).
7. Franka initialized; robot driven to training home position over 100 warmup steps.
8. Main loop: ExternalCamera frame → `build_observation` (with joint permutation) → `policy.infer()` → EMA-smoothed joint targets → `franka.apply_action()` → `world.step(render=True)`.
9. RecordingCamera frames written to `evaluation.mp4`; joint positions logged to `results.csv`.

### Joint permutation

Isaac Sim FR3 joint ordering differs from the Egoverse training data at indices 3 and 5. A self-inverse permutation `[0,1,2,5,4,3,6]` is applied in both directions:

```python
_SIM_TO_TRAIN = [0, 1, 2, 5, 4, 3, 6]   # reading state for the policy
_TRAIN_TO_SIM = [0, 1, 2, 5, 4, 3, 6]   # writing action back to sim
```

### Bind mounts inside the container

| Host path | Container path | Purpose |
|-----------|----------------|---------|
| `/cluster/scratch/$USER/pi0_test` | `/workspace` | scene, assets, outputs |
| `/cluster/scratch/$USER/openpi` | `/workspace/openpi` | source code + norm stats |
| `/cluster/work/cvg/data/rytsui/checkpoints` | `/checkpoints` | trained checkpoint |
| `$ISAAC_SIM_CACHE_DIR/kit` | `/isaac-sim/kit/cache` | writable shader cache |
| `$ISAAC_SIM_CACHE_DIR/ov_home` | `/cluster/home/$USER` | overrides read-only home for Warp/texture cache |
| `/cluster/scratch/$USER/isaac_packages` | `/isaac_packages` | Python 3.10 deps |

---

## Current issues

### Sim-to-real domain gap (primary open problem)

The pipeline runs cleanly but the **robot barely moves**. Joint positions stay within ±0.05 rad of the home position for all 3000 steps. The policy is outputting near-mean predictions.

**Root cause:** Training used real Aria egocentric RGB images (480×640, human hand visible in frame, real-world lighting and textures). The sim camera sees rendered 3D geometry — a completely different visual distribution. The model collapses to outputting training-mean actions for out-of-distribution inputs.

**What has been tried:**
- Repositioned `ExternalCamera` from bird's-eye `(0.5, 0, 4.2)` to eye-level `(0.7, -1.5, 2.5)` with 20° downward tilt.
- All scene assets properly textured (table MDL shaders + textures, crate Plastic_Yellow_A, plate textures).
- EMA smoothing on actions (`α=0.4`) to reduce chunk-boundary jitter.
- Warmup: robot driven to training home position before policy loop.
- Joint permutation and hand-state autoregression to keep observations in-distribution.

**Diagnostic outputs saved per run:** `policy_cam_init.png` (camera view at init) and `policy_cam_step0.png` (view at step 0) — inspect these to verify the camera perspective.

**Likely next steps to fix:**
1. Match camera FoV to Aria RGB (76° hFoV) — currently using Isaac Sim default.
2. Correct language command to exactly match training label (training task is `object_in_bowl`; command should be `"put the object in the bowl"` not `"put the plate into the yellow crate"`).
3. Domain randomization or sim-to-real transfer techniques.
4. Evaluate on real hardware rather than sim to separate policy quality from domain gap.

---

## Recovering after scratch purge

Euler auto-deletes scratch files not accessed in 15 days. Recovery order:

```bash
# 1. Norm stats (from persistent checkpoints — always available)
mkdir -p /cluster/scratch/$USER/openpi/assets/pi05_egoverse/egoverse/all && cp /cluster/work/cvg/data/rytsui/checkpoints/pi05_egoverse/test/29999/assets/egoverse/all/norm_stats.json /cluster/scratch/$USER/openpi/assets/pi05_egoverse/egoverse/all/

# 2. USD scene file (from repo)
cp /cluster/scratch/$USER/openpi/3dvision-experiments/isaac-sim/kitchen_scene_1.usd /cluster/scratch/$USER/pi0_test/

# 3. USD mesh assets
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

# 4. MDL shaders and textures (without these, objects render as plain white — breaks the policy)
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
```

Then reinstall Python packages (see [One-time setup](#one-time-setup-new-user) step 4).

---

## Gotchas reference

| Problem | Fix |
|---------|-----|
| Container hangs at startup (>10 min, no output) | Kit shader cache is read-only. Bind `$ISAAC_SIM_CACHE_DIR/kit` → `/isaac-sim/kit/cache`. Already in `submit.sh`. |
| Warp kernel / texture cache: `Read-only file system: '/cluster/home'` | Isaac Sim uses `getpwuid()` not `$HOME` — `APPTAINERENV_HOME` has no effect. Bind a writable scratch dir over `/cluster/home/$USER`. Already in `submit.sh`. |
| `ModuleNotFoundError` after ~73 s (after Isaac boots) | `isaac_packages/` was wiped and not reinstalled. Reinstall immediately after wiping. |
| `cannot import name 'NoDefault' from 'typing_extensions'` | Isaac Sim caches an old `typing_extensions` in `sys.modules` at startup. `eval_script_1.py` force-reloads the correct version at the top of the file. |
| `AttributeError: module 'datetime' has no attribute 'UTC'` | Python 3.10 compat — fixed in `src/openpi/shared/download.py` to use `datetime.timezone.utc`. |
| `franka.get_joint_positions()` returns `None` at step 0 | After `world.reset()` the articulation view is cleared. Call `franka.initialize()` again immediately after reset. Fixed in `eval_script_1.py`. |
| Wrong `jaxtyping` version despite pinning | `chex` (transitive dep) pulls in latest. Wipe and reinstall with pinned version. |
| `gpuhe.4h` stuck in queue | Switch to `gpu.4h --gpus=rtx_3090:1`. 24 GB VRAM is enough for inference; RTX 3090 nodes are much more available. |
| SLURM `--mem` flag rejected | Euler only accepts `--mem-per-cpu`. Never use `--mem=`. |
| `#SBATCH` directives ignored | Euler quirk — pass ALL flags on the `sbatch` command line, not as `#SBATCH` lines in the script. |
| SLURM log not in `~` | Log goes to the CWD of the `sbatch` invocation. Run `sbatch` from `/cluster/scratch/$USER/`. |
| Script has Windows line endings | `sed -i 's/\r//' submit.sh` before submitting. |
| Objects render as plain white | MDL shaders and texture PNGs missing — run the recovery commands in step 4 above. |
| Policy outputs near-constant actions | Sim-to-real domain gap — see [Current issues](#current-issues). |
