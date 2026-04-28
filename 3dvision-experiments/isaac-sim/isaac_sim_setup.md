# Project Goal: Run a Robot Brain (pi0.5 VLA) in a Simulated World on the Euler Cluster

## What we are doing, in one paragraph

We want to test a smart robot "brain" (a Vision-Language-Action model called **pi0.5**) by letting it control a simulated **Franka FR3 robot arm** inside **NVIDIA Isaac Sim**. The simulation is too heavy for a laptop, so we run it on **ETH Zurich's Euler cluster** (a big shared computer with powerful GPUs). The robot watches through two cameras, hears a text command like "pick up the block," and decides how to move. We will watch it live in a web browser and save a video of the result.

---

## The 5 Big Phases

1. **Build the world** on a laptop (Isaac Sim → save as a `.usd` file).
2. **Write the Python script** that connects the brain to the robot.
3. **Move everything to Euler** and pack the brain into a "container" file.
4. **Ask Euler for a GPU** using a SLURM job script and run it.
5. **Watch it live** in your browser, then **download the results**.

---

## Phase 1 — Build the World on Your Laptop

**Goal:** Make a 3D scene file (`franka_vla_env.usd`) that contains the robot, a table, objects, and two cameras.

### Step-by-step

1. Open Isaac Sim on your laptop.
2. Add the **Franka FR3** arm, a table, and the objects you want it to manipulate.
3. Add **two cameras**:
   - **External camera** — placed over the robot's shoulder, looking down at the table.
   - **Wrist camera** — drag it under the robot's hand link (`panda_hand`) in the Stage tree, so it moves with the gripper.
4. Click the robot in the Stage tree → make sure **Articulation Root** is applied (this lets physics control it).
5. Click the table and objects → make sure they have **Rigid Body** and **Collider** properties (so the robot can't pass through them).
6. **File → Save As** → `franka_vla_env.usd`.

### Common pitfalls

| 🔴 Problem | 🟢 Fix |
|---|---|
| Camera resolution is huge (1080p) → model crashes or is super slow. | Don't worry about resolution in the GUI. Set it to 256x256 in the Python script later. |
| You forgot Articulation Root → robot is frozen like a statue. | Re-open the USD, click the robot, add Articulation Root, save again. |
| Objects fall through the table. | Add Collider properties to the table AND the objects. |

---

## Phase 2 — Write the Python Script (`vla_eval.py`)

**Goal:** A single Python file that loads the world, loads the brain, takes camera pictures, asks the brain what to do, and moves the robot — repeating ~50 times per second.

### The 5 things the script must do, in order

1. **Start Isaac Sim in headless mode** (no monitor needed) with livestreaming on.
2. **Load the world** from `franka_vla_env.usd`.
3. **Hook up the robot and both cameras** (paths must match the Stage tree exactly).
4. **Load the pi0.5 brain** onto the GPU.
5. **Run a loop** that: takes pictures → reads joint positions → asks the brain → moves the robot → steps the simulator forward.

### Skeleton script

```python
# vla_eval.py
from isaacsim import SimulationApp

# Headless = no monitor needed. livestream = 1 lets us watch in a browser.
simulation_app = SimulationApp({"headless": True, "livestream": 1})

import torch
from omni.isaac.core import World
from omni.isaac.franka import Franka
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.sensor import Camera

# Load the world
world = World(stage_units_in_meters=1.0)
world.stage.Open("/workspace/franka_vla_env.usd")

# Hook up the robot and cameras (paths MUST match the USD)
franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka_robot"))
external_cam = Camera(prim_path="/World/ExternalCamera", resolution=(256, 256))
wrist_cam = Camera(prim_path="/World/Franka/panda_hand/WristCamera", resolution=(256, 256))
external_cam.initialize()
wrist_cam.initialize()

# Load the pi0.5 brain
from pi0_fast import Pi0Model
vla_model = Pi0Model.load_weights("/workspace/pi0_weights.pt")
vla_model.eval()
vla_model.to("cuda")

language_command = "Pick up the block and place it in the bin."

# Main loop
world.reset()
for step in range(1000):
    # 1. Take pictures (drop the alpha channel)
    ext_img = external_cam.get_rgba()[:, :, :3]
    wrist_img = wrist_cam.get_rgba()[:, :, :3]

    # 2. Read robot's current joint positions
    current_joints = franka.get_joint_positions()

    # 3. Ask the brain what to do
    with torch.no_grad():
        predicted_action = vla_model.predict(
            external_image=ext_img,
            wrist_image=wrist_img,
            proprioception=current_joints,
            text=language_command,
        )

    # 4. Send the action to the robot
    joint_targets = predicted_action[:7]
    gripper_target = predicted_action[7]
    franka.apply_action(ArticulationAction(joint_positions=joint_targets))

    # 5. Step the simulator (render=True is REQUIRED for new camera frames)
    world.step(render=True)

simulation_app.close()
```

### Common pitfalls

| 🔴 Problem | 🟢 Fix |
|---|---|
| Script uses Windows-style paths like `C:/Users/...` → crashes on Linux cluster. | Use **relative paths** (`./franka_vla_env.usd`) or paths inside `/workspace/`. Keep all files in one folder. |
| Image shape mismatch — Isaac gives `[H, W, C]`, PyTorch wants `[C, H, W]`. | Transpose before feeding the model: `torch.from_numpy(img).permute(2, 0, 1)`. |
| Robot flails violently and "explodes." | The model probably outputs **deltas**, not absolute joint targets. Use `target = current_joints + predicted_action` instead. |
| Cameras return blank/black images. | Make sure `world.step(render=True)` runs **before** you call `.get_rgba()`. |

---

## Phase 3 — Move Files to Euler and Set Up the Container

**Goal:** Get the simulator (in a `.sif` container file), the script, the world, and the model weights onto Euler's scratch storage.

### Step-by-step

1. **Build the Isaac Sim container on your laptop** using Isaac Lab's tools. This produces a single big file like `isaac_sim.sif` (around 15 GB). Euler does **not** allow Docker — only Apptainer/Singularity, hence the `.sif`.
2. **Log into Euler** via SSH:
   ```bash
   ssh user@euler.ethz.ch
   ```
3. **Make a working folder in scratch space** (NOT in your home folder — home is too small):
   ```bash
   cd /cluster/scratch/$USER/
   mkdir pi0_test
   cd pi0_test
   ```
4. **From your laptop terminal**, send the files over with `rsync`:
   ```bash
   rsync -avP ./isaac_sim.sif        user@euler.ethz.ch:/cluster/scratch/$USER/pi0_test/
   rsync -avP ./franka_vla_env.usd   user@euler.ethz.ch:/cluster/scratch/$USER/pi0_test/
   rsync -avP ./vla_eval.py          user@euler.ethz.ch:/cluster/scratch/$USER/pi0_test/
   rsync -avP ./pi0_weights.pt       user@euler.ethz.ch:/cluster/scratch/$USER/pi0_test/
   ```

### Common pitfalls

| 🔴 Problem | 🟢 Fix |
|---|---|
| `scp` of the 15 GB `.sif` file fails halfway. | Always use `rsync -avP`. The `-P` flag shows progress and **resumes** if the connection drops. |
| You put files in `$HOME` and hit a quota error. | Move everything to `/cluster/scratch/$USER/`. Home is small; scratch is big. |
| Files transfer but Euler can't find them. | Double-check the path — `/cluster/scratch/$USER/pi0_test/` (use `$USER` literally; it auto-fills). |

---

## Phase 4 — Write and Submit the SLURM Job

**Goal:** A short shell script that asks Euler for a GPU and runs the simulator inside the container.

### Step-by-step

1. On Euler, in your `pi0_test` folder, create a file called `submit.sh`:

   ```bash
   #!/bin/bash
   #SBATCH --job-name=pi0_eval
   #SBATCH --time=04:00:00            # 4 hours max
   #SBATCH --nodes=1
   #SBATCH --ntasks=1
   #SBATCH --cpus-per-task=8
   #SBATCH --mem=64G                  # Isaac + pi0.5 needs lots of RAM
   #SBATCH --gpus=rtx_3090:1          # Or a100:1 for faster runs
   #SBATCH --output=eval_log.out      # All printed text lands here

   # Some Isaac Sim downloads need a proxy on Euler compute nodes
   export HTTP_PROXY=http://proxy.ethz.ch:3128
   export HTTPS_PROXY=http://proxy.ethz.ch:3128

   # Send Isaac's shader cache to scratch (NOT to home — home will fill up)
   export ISAAC_SIM_CACHE_DIR=/cluster/scratch/$USER/isaac_cache
   mkdir -p $ISAAC_SIM_CACHE_DIR

   # Run the simulator inside the container
   apptainer exec --nv \
       --bind /cluster/scratch/$USER/pi0_test:/workspace \
       /cluster/scratch/$USER/pi0_test/isaac_sim.sif \
       python /workspace/vla_eval.py
   ```

2. **Submit the job:**
   ```bash
   sbatch submit.sh
   ```

3. **Check it's running:**
   ```bash
   squeue -u $USER
   ```
   The `NODELIST` column tells you which compute node has your job, e.g. `eu-g4-005`. **Write this name down — you need it in Phase 5.**

### Common pitfalls

| 🔴 Problem | 🟢 Fix |
|---|---|
| Job dies instantly: "No space left on device." | Isaac Sim caches up to 10 GB of shaders. Set `ISAAC_SIM_CACHE_DIR` to a scratch path (shown above). |
| Job stuck in queue forever (`PD` = pending). | The GPU you asked for is busy. Try a different GPU type, or shorten `--time`. |
| Container can't read your files. | Always use `--bind /cluster/scratch/$USER/pi0_test:/workspace` so the container sees your folder as `/workspace`. |
| `apptainer: command not found`. | On Euler, you may need `module load eth_proxy` and/or `module load apptainer` before submitting. |
| Asset downloads fail inside the container. | The proxy lines (`HTTP_PROXY` / `HTTPS_PROXY`) are required on compute nodes — don't remove them. |

---

## Phase 5 — Watch It Live and Download the Results

**Goal:** See the robot move in real time, then pull the saved video and data back to your laptop.

### Step 1: Find the compute node

On Euler:
```bash
squeue -u $USER
```
Note the node name (e.g. `eu-g4-005`).

### Step 2: Open an SSH tunnel from your laptop

In a **new terminal on your laptop** (not on Euler):
```bash
ssh -L 8211:eu-g4-005:8211 -L 49100:eu-g4-005:49100 user@euler.ethz.ch
```
Replace `eu-g4-005` with the actual node name from Step 1. Leave this terminal open.

### Step 3: Watch in your browser

Open Chrome or Edge and go to:
```
http://localhost:8211/streaming/webrtc-demo/
```
You should see the robot moving live.

### Step 4: When the job finishes, download results

Back on your laptop terminal:
```bash
rsync -avP user@euler.ethz.ch:/cluster/scratch/$USER/pi0_test/results.csv ./
rsync -avP user@euler.ethz.ch:/cluster/scratch/$USER/pi0_test/evaluation_video.mp4 ./
```

### Common pitfalls

| 🔴 Problem | 🟢 Fix |
|---|---|
| Browser shows a black screen or "Connection Lost." | Euler's firewall is blocking the UDP video ports. Force TCP by adding `?tcp=true` to the URL: `http://localhost:8211/streaming/webrtc-demo/?tcp=true` |
| Tunnel command says "channel: open failed." | The job hasn't actually started yet. Run `squeue -u $USER` again — wait until status is `R` (running), not `PD` (pending). |
| Tunnel works but page won't load. | You used the wrong node name. Re-check `squeue -u $USER` and rebuild the tunnel. |
| `rsync` says "no such file." | The script crashed before saving outputs. Check `eval_log.out` on Euler for the error message. |

---

## Quick Sanity Checklist Before You Run a Long Job

Run through these every single time — they save hours of wasted GPU time:

- [ ] All file paths in `vla_eval.py` use `/workspace/...`, not laptop paths.
- [ ] `franka_vla_env.usd`, `vla_eval.py`, `pi0_weights.pt`, and `isaac_sim.sif` are all in `/cluster/scratch/$USER/pi0_test/`.
- [ ] The robot has Articulation Root; objects have Colliders.
- [ ] Cameras are 256x256 and have clear views.
- [ ] `submit.sh` requests enough RAM (≥ 64 GB) and a real GPU.
- [ ] `ISAAC_SIM_CACHE_DIR` points to scratch.
- [ ] You did a **5-minute test run** (`--time=00:05:00`) before launching the 4-hour job.

---

## What "Done" Looks Like

You can:
1. Type `sbatch submit.sh` on Euler.
2. Open a browser and watch the FR3 arm pick up a block based on a text command.
3. Download an MP4 video and a CSV of results to your laptop when it finishes.

That's the whole project.