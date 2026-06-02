# Getting Set Up: Isaac Sim Eval Pipeline on Euler

This document is a complete setup guide for running the pi0.5 Isaac Sim evaluation on the Euler cluster. Follow every step in order on the **login node** unless noted otherwise.

---

## Prerequisites

- Euler cluster account (`ssh <username>@euler.ethz.ch`)
- Access to the CVG group storage (`/cluster/work/cvg/` readable)
- The `isaac-sim_4.5.0.sif` container file (6 GB) — obtained via shared link from the project owner

---

## Step 1 — Clone the repo to your scratch directory

Scratch has 2.5 TB quota and is fast. Never use `~` (home) for large files.

```bash
cd /cluster/scratch/$USER
git clone https://github.com/dalongbao/openpi.git
```

---

## Step 2 — Place the SIF container

Download the `isaac-sim_4.5.0.sif` file shared with you and put it in your scratch:

```bash
mkdir -p /cluster/scratch/$USER/pi0_test

# If you have a Polybox / direct wget link:
wget -O /cluster/scratch/$USER/pi0_test/isaac-sim_4.5.0.sif "<paste-link-here>"

# If you have a Google Drive file ID:
pip install --user gdown
gdown --id <FILE_ID> -O /cluster/scratch/$USER/pi0_test/isaac-sim_4.5.0.sif
```

---

## Step 3 — Create directory structure and copy scene files

```bash
mkdir -p /cluster/scratch/$USER/pi0_test/assets \
         /cluster/scratch/$USER/isaac_packages \
         /cluster/scratch/$USER/isaac_cache/kit \
         /cluster/scratch/$USER/isaac_cache/ov_home

cp /cluster/scratch/$USER/openpi/3dvision-experiments/isaac-sim/kitchen_scene_1.usd \
   /cluster/scratch/$USER/pi0_test/
cp /cluster/scratch/$USER/openpi/3dvision-experiments/isaac-sim/submit.sh \
   /cluster/scratch/$USER/submit.sh
```

---

## Step 4 — Restore norm stats

```bash
mkdir -p /cluster/scratch/$USER/openpi/assets/pi05_egoverse/egoverse/all
cp /cluster/work/cvg/data/rytsui/checkpoints/pi05_egoverse/test/29999/assets/egoverse/all/norm_stats.json \
   /cluster/scratch/$USER/openpi/assets/pi05_egoverse/egoverse/all/
```

---

## Step 5 — Download USD scene assets from Omniverse CDN

The login node has internet via ETH proxy. Run all of this on the login node.

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

### Step 5b — Download MDL shaders and textures

Without these all objects render as plain white, causing the VLA policy to fail (domain gap).

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

---

## Step 6 — Install Python packages into isaac_packages

Isaac Sim uses its own Python 3.10 interpreter. Run this inside the container so the ABI matches. Takes ~10 minutes. Do **not** copy from another user's scratch — reinstall fresh.

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

**Critical version pins — do not change these:**

| Package | Version | Why |
|---------|---------|-----|
| `jax` | `0.5.3` | locked in uv.lock |
| `jaxtyping` | `0.2.36` | 0.3.x removed `_check_dataclass_annotations` |
| `orbax-checkpoint` | `0.11.13` | checkpoint format compatibility |
| `numpy` | `1.26.4` | numpy 2.x breaks Isaac Sim's numba |
| `transformers` | `4.53.2` | API compatibility with openpi |

---

## Step 7 — Submit the eval job

```bash
cd /cluster/scratch/$USER/openpi && git pull && \
cp 3dvision-experiments/isaac-sim/eval_script_1.py /cluster/scratch/$USER/pi0_test/eval_script_1.py && \
sed -i 's/\r//' /cluster/scratch/$USER/submit.sh && \
sbatch --partition=gpu.4h --time=00:30:00 --mem-per-cpu=8G --cpus-per-task=8 --gpus=rtx_3090:1 \
  /cluster/scratch/$USER/submit.sh
```

`submit.sh` uses `$USER` throughout — no path edits needed.

Check status:
```bash
squeue -u $USER
tail -f /cluster/scratch/$USER/slurm-<JOBID>.out
```

---

## Verification — what success looks like

After ~12 minutes the job exits 0 and produces:
- `/cluster/scratch/$USER/pi0_test/evaluation.mp4` — HD video of the robot attempt
- `/cluster/scratch/$USER/pi0_test/results.csv` — joint positions over 3000 steps

Download to your laptop:
```bash
rsync -avP $USER@euler.ethz.ch:/cluster/scratch/$USER/pi0_test/evaluation.mp4 ./
rsync -avP $USER@euler.ethz.ch:/cluster/scratch/$USER/pi0_test/results.csv ./
```

---

## Cluster quirks (must know)

- Never use `--mem=` in SLURM — only `--mem-per-cpu=`
- Never put `#SBATCH` directives in the script body — pass all flags on the `sbatch` command line
- Compute nodes have no internet — all downloads (Steps 4–6) must run on the **login node**
- SLURM log goes to the CWD where `sbatch` was run — check `/cluster/scratch/$USER/`
- Scratch auto-purges after 15 days of no access — re-run Steps 4–5 after any purge
- `gpu.4h + rtx_3090:1` schedules faster than `gpuhe.4h + a100`
