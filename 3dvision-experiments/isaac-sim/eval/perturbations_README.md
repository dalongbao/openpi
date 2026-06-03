# Perturbation harness

Mutates the USD kitchen scene before `world.reset()` to produce variant
rollouts of the pi0.5 policy. Invoked by `eval/core.py`. Stage-touching
code is in `perturbations.py`; sweep generators in `perturbation_sweeps.py`.

## Axes

| Axis            | Prim(s) touched                                  | Field on `PerturbationConfig`                                  |
|-----------------|--------------------------------------------------|----------------------------------------------------------------|
| `distractors`   | `/World/Distractor_<i>` (new prims)              | `distractors: list[DistractorSpec]`                            |
| `dome_intensity`| first `UsdLuxDomeLight` under `/World/...`       | `dome_intensity` (multiplicative scale), `dome_color`          |
| `plate_pose`    | `/World/plate_small`                             | `plate_pose_offset_m`, `plate_yaw_offset_rad`                  |
| `crate_pose`    | `/World/SM_Crate_A07_Yellow_01_physics`          | `crate_pose_offset_m`, `crate_yaw_offset_rad`                  |
| `viewpoint`     | `/World/ExternalCamera`                          | `external_camera_translation_offset_m` + `..._rotation_offset_deg` |
| `language`      | (none — metadata)                                | `language_prompt`                                              |

`/World/RecordingCamera` is the HD recorder and is **never** modified.
`/World/fr3` is **never** moved.

## Clamping

Plate, crate, and distractor positions are clamped to the table AABB
defined in `perturbations.TABLE_BOUNDS_XY = (-0.40, 0.40, 0.00, 0.60)` and
`TABLE_BOUNDS_Z = (0.70, 1.00)`. These were chosen to match the packing
table footprint used in the original eval. **Sanity-check on Euler**: open
the scene and confirm the table top really occupies that XY box; if the
table was moved in a later commit, widen these bounds.

There's no ground plane in the scene, so a bad offset that escapes the
clamp would cause the object to fall forever — keep the clamp in.

Pose offsets are **additive on top of the current USD pose**. Calling
`apply()` twice with the same config does NOT double the offset, because
the runner constructs a fresh stage per rollout. Do not call `apply()`
twice on the same stage.

## How to add a new axis

1. Add a field to `PerturbationConfig` in `perturbations.py`.
2. Extend `_config_from_dict` so JSON loads the field.
3. Add the mutation in `apply()` (new helper `_apply_<axis>` if non-trivial).
4. Add the axis to `single_axis_sweeps()` in `perturbation_sweeps.py`.
5. Decide whether `latin_hypercube_sample` should jointly vary it.
6. Add unit tests in `eval/tests/test_perturbations.py`.

## Distractor assets to procure on Euler

The sweep generator references the following USDs under
`/workspace/assets/distractors/` — these need to exist on the compute
node (compute has no internet). Recommended source: Omniverse public S3
(same approach as plate/crate/table caching documented in `CLAUDE.md`).

- `mug.usd`
- `apple.usd`
- `cereal_box.usd`
- `banana.usd`
- `can.usd`
- `bowl.usd`
- `spoon.usd`
- `sponge.usd`

Pull each on the login node, then `cp -r` into your scratch
`pi0_test/assets/distractors/`. Update `_DISTRACTOR_USD_POOL` in
`perturbation_sweeps.py` if you swap meshes.

## Running the unit tests

The tests do NOT require Isaac Sim. They use an in-memory `Usd.Stage`
via `pxr` (USD core); if `pxr` is not installed locally the
`apply()`-touching tests are skipped automatically.

```
cd 3dvision-experiments/isaac-sim && python -m pytest eval/tests -v
```
