# `merge_to_oic_mix.py` — design spec (fast image-linking merge)

**For the agent implementing this:** your job is to write & run `3dvision-experiments/merge_to_oic_mix.py`,
which builds the `egoverse/oic_mix` LeRobot dataset **fast** by *reusing* two already-converted
datasets (linking images instead of re-encoding them), rather than the slow `build_oic_mix.py`
(which re-reads raw h5 + re-encodes ~563k images → 5–8 h). The merge should take **minutes**.

> **FIRST, check if it's even needed:** if `build_oic_mix.py` already produced a valid `oic_mix`
> (`python 3dvision-experiments/verify_oic_mix.py --root <…>/egoverse/oic_mix` prints `PASS` with
> ~2601 episodes), **stop — you don't need this.** The merge is the fast path for *rebuilds*
> (e.g. a different held-out split) and a backup if the slow build failed.

---

## 1. What `oic_mix` is (the target)

A cross-embodiment co-training dataset: robot teleop (**R-ID**, `object_in_bowl`) + human video
(**H-ID**, `object_in_container`) in **one 24D action space**, so a single π0.5 head can train on
both. It feeds the `pi05_ego_mix_oic*` configs. **Read `3dvision-experiments/build_oic_mix.py`** —
it is the ground-truth definition of the target format; your output must be byte-equivalent to what
it produces (same `state`/`actions`/`action_mask` values), just built by linking instead of re-encoding.

**Target schema** (LeRobot v2.1, `fps=50`, `robot_type=franka`):
- `image` (480,640,3 uint8), `state` (24 f32), `actions` (24 f32), `action_mask` (24 f32),
  + standard LeRobot cols (`timestamp, frame_index, episode_index, index, task_index`), `task`.
- **Robot episodes** (`episode_index` 0..63): `state`/`actions` = native **24D base frame** (these are
  already correct in `egoverse/all` — **identity, no transform**); `action_mask = ones(24)`.
- **Human episodes** (`episode_index` 64..): `state`/`actions` = **6D→24D transformed** (see §3);
  `action_mask = [1]*7 + [0]*17` (human never supervises the 17 hand dims — R-ID is the only gripper source).

## 2. Sources (both already on disk)
- **Robot:** `/cluster/scratch/lichin/lerobot/egoverse/all` — 64 eps, `state`/`actions` already **24D base**.
  Only thing missing vs target: the `action_mask` column. (Must be COMPLETE: `meta/info.json` `total_episodes` ≈ 64.)
- **Human:** `/cluster/work/cvg/data/Egoverse/lerobot_egoverse/egoverse/oic_human` — 2537 eps, `state`/`actions` **6D** (euler/head frame).

## 3. The transform (human only; robot is identity)
Use `openpi.policies.egoverse_unify` (already on `main`, self-tested):
```python
from openpi.policies import egoverse_unify as U
arm7   = U.human6d_to_base_arm7(pose6d)      # 6D [x,y,z,yaw,pitch,roll] head -> 7D [pos,quat xyzw] base
act24, mask24 = U.to_unified(arm7, None)     # -> 24D action + mask [1]*7+[0]*17
```
Apply to **both** the human `state` and `actions` columns. Robot: keep `state`/`actions` as-is,
set `action_mask = np.ones(24, float32)`.
Convention (resolved, do not re-derive): human is **scipy intrinsic ZYX, radians, egocentric head frame**;
unified frame = **robot base** (so `egoverse/all` + `ablation_eval.py` are untouched). See `egoverse_unify.py` header.

## 4. MANDATORY first step — determine image storage
Run `python 3dvision-experiments/inspect_lerobot_format.py` and branch:
- **Case A — images embedded in parquet (bytes):** the parquet *is* the images. Merge = read each source
  parquet, (robot) add `action_mask` col / (human) rewrite `state`/`actions` to 24D + add mask, write to the
  target. Image bytes ride along for free. No separate files.
- **Case B — images are separate PNG files (`images/` dir):** **hard-link** them (`os.link`, instant,
  same filesystem) into the target's `images/` tree, and rewrite the parquets (which reference paths),
  preserving/renumbering the references. Do **not** copy bytes.

**Do not write the merge before confirming which case** — the file ops differ entirely.

## 5. Algorithm
1. Verify `egoverse/all` complete + `oic_human` exists. Create target dir (`meta/`, `data/chunk-*`, `images/` if case B).
2. **Robot eps → target index 0..63:** per source episode, add `action_mask=ones(24)`; keep `state`/`actions`/`image`;
   write parquet to the right chunk (`chunk = idx // chunks_size`, `chunks_size=1000`,
   path `data/chunk-{chunk:03d}/episode_{idx:06d}.parquet`); case B → hard-link its image files.
3. **Human eps → target index 64..64+Nh-1:** transform `state`/`actions` 6D→24D (§3), add mask `[1]*7+[0]*17`;
   **renumber** `episode_index` and the global `index` column (continue from the robot frame count); write parquet; case B → hard-link images.
4. **Merge `meta/`:**
   - `info.json`: `total_episodes = 64+Nh`, `total_frames = robot+human`, `total_chunks`, `splits={"train":"0:total"}`,
     `fps=50`, and **`features`** = the *target* schema (state/actions shape [24], **add `action_mask` [24] float32**).
   - `episodes.jsonl`: concat robot + human, renumbered, lengths preserved.
   - `tasks.jsonl`: single task `"put the object in the bowl"` (one `task_index`) if both sources share it.
   - stats (`episodes_stats.jsonl` or `stats.json`): match whatever format the inspector shows; simplest correct
     approach is to recompute per-episode stats over the merged data. (Training norm stats are computed *separately*
     by `compute_norm_stats.py`, so these dataset stats just need to be schema-valid.)

## 6. Validate (do NOT skip)
- Implement `--max-robot N --max-human M` smoke mode. Build a 2+2 merged set, then:
  `python 3dvision-experiments/verify_oic_mix.py --root <target>` → must print
  `robot(=24): 2  human(=7): 2`, `[human hand slots] max|val| = 0.000000`, shapes (24,), `PASS`.
- Cross-check against a `build_oic_mix.py --max_robot 2 --max_human 2 --low_mem` output on the **same**
  episodes: the `state`/`actions`/`action_mask` arrays must be numerically identical (the merge must not
  change values, only avoid re-encoding). Confirm a sample image opens and matches.
- Only run the **full** merge after the smoke matches.

## 6b. CRITICAL — fps / timestamps (a real bug the first merge hit)
`oic_human` is **30 Hz**; the merged dataset is **fps=50**. You **MUST regenerate the human
`timestamp` column as `frame_index / 50`** — do **NOT** copy `oic_human`'s 30 Hz timestamps
(0.0333 s spacing). Training builds 10-step action chunks at `delta_timestamps = [t/50 …]`
(0.02 s spacing); with 30 Hz human stamps the lookups miss LeRobot's tolerance and the dataset
**fails to load**. (`build_oic_mix` is correct because LeRobot's `add_frame` auto-stamps
`frame_index/fps`.) Robot timestamps from `egoverse/all` are already 50 Hz — leave them.
If a merge already shipped with this bug, patch it with `fix_mix_timestamps.py` (rewrites only the
human `timestamp` column, no image re-encode).

## 7. Gotchas
- LeRobot v2.1 metadata is strict — a wrong `total_frames`/`episode_index`/`index` breaks dataset loading.
  Counts must be exact and contiguous (`index` = 0..total_frames-1 across the whole merged set).
- Human `state`/`actions` schema **changes** (6D→24D) → the human parquets must be **rewritten**, not copied.
- Robot `state`/`actions` are already 24D base → only **add the mask column**.
- `action_mask` must appear in `info.json.features` (dtype float32, shape [24]).
- The human episode count (`Nh`, expected **2537**) sets `total_episodes=64+Nh`. After the merge, confirm
  it equals what `config.py`'s `_MIX_HUMAN = list(range(64, 2601))` assumes (2601 total); if not, update that.
- This needs `egoverse/all` **complete** first.

## 8. Key files to read
- `3dvision-experiments/build_oic_mix.py` — the canonical builder = the exact target format & values.
- `src/openpi/policies/egoverse_unify.py` — the 6D→24D transform + mask (self-test: `python -m openpi.policies.egoverse_unify`).
- `3dvision-experiments/verify_oic_mix.py` — the validator.
- `3dvision-experiments/inspect_lerobot_format.py` — run FIRST to pick case A vs B.
- `src/openpi/training/config.py` (`LeRobotEgoverseUnifiedDataConfig`, `pi05_ego_mix_oic*`) — the consumer.
