# Model registry

One `.env` file per trained model. Each records everything needed to evaluate that
model: its training config, checkpoint dir, norm-stats dir, and prompt. Pick a model by
passing its name (the filename without `.env`) as the **2nd arg** to `submit.sh`, which
sources the file and forwards the values into the container.

| Preset | Config | Episodes | Notes |
|--------|--------|----------|-------|
| `egoverse_5ep` | `pi05_egoverse` | 5 | Original object_in_bowl finetune. 24-dim (arm+hand). Weak (real-frame cosine ~0.385). Run with `eval_script_object_in_bowl.py`. |
| `rid30` | `pi05_egoverse_n30` | 30 | R_ID object_in_bowl finetune, 30 robot episodes (step 20000). Same 24-dim arm+hand → `eval_script_object_in_bowl.py`, arm visible. Checkpoint is in THIS user's scratch (`/user_checkpoints`), bound by `submit.sh`. |
| `rid64` | `pi05_egoverse` | 64 | R_ID object_in_bowl finetune, FULL 64 episodes (step 30000). Strongest robot model. 24-dim → `eval_script_object_in_bowl.py`, arm visible. Teammate-owned (lichin), bound at `/shared_checkpoints` by `submit.sh`. |
| `oic_human_2537ep` | `pi05_ego_human_oic` | 2537 | Object-in-container, human (Aria) data. **6-dim single-arm cartesian (no hand)** → run with the dedicated `eval_script_oic.py`. Euler order/frame unresolved; the script sweeps `EULER_ORDER` at startup. |
| `base` | `pi05_egoverse` | 0 | Untrained pi0.5. Use the dedicated `eval_script_base_object_in_bowl.py`. |

**Each model has a matching eval script** — the action space differs (24-dim arm+hand vs 6-dim cartesian), so the script must match the preset.

## Run a model

```
# Full eval of the big model
sbatch --partition=gpu.4h --time=00:30:00 --mem-per-cpu=8G --cpus-per-task=8 --gpus=rtx_3090:1 submit.sh eval_script_object_in_bowl.py oic_human_2537ep

# Quick (400-step) smoke test of the big model, with the nicer scene
SCENE_FIDELITY=1 sbatch --partition=gpu.4h --time=00:15:00 --mem-per-cpu=8G --cpus-per-task=8 --gpus=rtx_3090:1 submit.sh eval_script_object_in_bowl_quick.py oic_human_2537ep
```

Outputs land in `results/<MODEL_NAME>/` (on the cluster, `/cluster/scratch/$USER/pi0_test/results/<MODEL_NAME>/`),
so different models never overwrite each other:
`results/oic_human_2537ep/evaluation.mp4`, `results.csv`, `policy_cam_init.png`, …

## Add a new model

Copy an existing `.env`, point it at the new checkpoint, and give it a unique
`MODEL_NAME`. Container paths: `/checkpoints` = rytsui's checkpoints dir,
`/base_weights` = pi05_base_jax, `/workspace/openpi` = the repo clone.
