"""Sweep generators for ``PerturbationConfig``.

Provides three flavors of sweep production:

- ``single_axis_sweeps(baseline)``: one axis at a time around a baseline.
- ``latin_hypercube_sample(baseline, n, seed)``: jointly perturb several
  axes using a Latin hypercube design.
- ``write_configs(configs, output_dir)``: serialize a list of configs to
  numbered JSON files under ``output_dir``.

Only stdlib + numpy + the ``perturbations`` module. No pxr / Isaac Sim.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

# Support being imported either as a subpackage (``eval.perturbation_sweeps``)
# or as a top-level module (``perturbation_sweeps``) — the latter is used by
# the test suite so it doesn't have to drag the rest of the ``eval`` package
# (which imports Isaac Sim deps via ``.core``) onto the import path.
try:
    from . import perturbations as pert
except ImportError:  # pragma: no cover
    import perturbations as pert  # type: ignore


# --------------------------------------------------------------------
# Language templates
# --------------------------------------------------------------------

def language_templates() -> List[str]:
    """Return the 4 language templates used by the language axis.

    Index 0 is the baseline; index 1 is a paraphrase; index 2 is a
    contradictory instruction (used to test instruction-following);
    index 3 is the empty string (no-language).
    """
    return [
        "put the plate in the crate",
        "place the small dish into the yellow box",
        "ignore the plate, do not touch it",
        "",
    ]


# --------------------------------------------------------------------
# Axis tables
# --------------------------------------------------------------------

# Number of distractors per step in the 'distractors' axis.
_DISTRACTOR_COUNTS = (0, 2, 4, 8)

# Intensity scale factors in the 'dome_intensity' axis.
_DOME_INTENSITIES = (0.3, 1.0, 3.0)

# Plate position offsets (m) in the 'plate_pose' axis.
_PLATE_OFFSETS = (
    (0.0,  0.0,  0.0),
    (0.10, 0.0,  0.0),
    (-0.10, 0.0, 0.0),
    (0.0,  0.10, 0.0),
    (0.0, -0.10, 0.0),
)

# Crate position offsets (m).
_CRATE_OFFSETS = (
    (0.0,  0.0,  0.0),
    (0.10, 0.0,  0.0),
    (-0.10, 0.0, 0.0),
    (0.0,  0.10, 0.0),
    (0.0, -0.10, 0.0),
)

# Viewpoint perturbations: (translate_offset_m, rotate_offset_deg)
_VIEWPOINT_OFFSETS = (
    ((0.0, 0.0, 0.0),   (0.0, 0.0, 0.0)),   # baseline
    ((0.05, 0.0, 0.0),  (0.0, 0.0, 0.0)),   # 5 cm right
    ((0.0, 0.0, 0.10),  (0.0, 0.0, 0.0)),   # 10 cm up
    ((0.0, 0.0, 0.0),   (5.0, 0.0, 0.0)),   # 5 deg pitch
)

# Distractor USD assets used by the distractor axis. These files must
# exist on Euler under /workspace/assets/distractors/ at runtime.
_DISTRACTOR_USD_POOL = (
    "/workspace/assets/distractors/mug.usd",
    "/workspace/assets/distractors/apple.usd",
    "/workspace/assets/distractors/cereal_box.usd",
    "/workspace/assets/distractors/banana.usd",
    "/workspace/assets/distractors/can.usd",
    "/workspace/assets/distractors/bowl.usd",
    "/workspace/assets/distractors/spoon.usd",
    "/workspace/assets/distractors/sponge.usd",
)


def _make_distractors(count: int, seed: int) -> List[pert.DistractorSpec]:
    """Sample ``count`` distractor specs in a deterministic, seeded order.

    Positions are sampled uniformly inside the table bounds (with a small
    margin so distractors don't sit on the plate). The clamp in
    ``apply()`` will further enforce the bounds at write time.
    """
    if count <= 0:
        return []
    rng = np.random.default_rng(seed)
    specs: List[pert.DistractorSpec] = []
    xmin, xmax, ymin, ymax = pert.TABLE_BOUNDS_XY
    # Shrink slightly to avoid the edge.
    xmin += 0.05; xmax -= 0.05; ymin += 0.05; ymax -= 0.05
    for i in range(count):
        usd_path = _DISTRACTOR_USD_POOL[i % len(_DISTRACTOR_USD_POOL)]
        specs.append(
            pert.DistractorSpec(
                prim_path=f"/World/Distractor_{i}",
                usd_path=usd_path,
                translate=(
                    float(rng.uniform(xmin, xmax)),
                    float(rng.uniform(ymin, ymax)),
                    0.78,
                ),
                rotate_yaw_rad=float(rng.uniform(-math.pi, math.pi)),
            )
        )
    return specs


# --------------------------------------------------------------------
# Sweeps
# --------------------------------------------------------------------

def _copy_with(baseline: pert.PerturbationConfig, **overrides) -> pert.PerturbationConfig:
    """Return a deep copy of ``baseline`` with selected fields overridden."""
    new = copy.deepcopy(baseline)
    for k, v in overrides.items():
        setattr(new, k, v)
    return new


def single_axis_sweeps(
    baseline: pert.PerturbationConfig,
) -> Dict[str, List[pert.PerturbationConfig]]:
    """Build one list of configs per axis, varying ONLY that axis.

    Axes:
        - ``distractors``       — 4 configs (0, 2, 4, 8 distractors).
        - ``dome_intensity``    — 3 configs (0.3x, 1.0x, 3.0x).
        - ``plate_pose``        — 5 plate position offsets.
        - ``crate_pose``        — 5 crate position offsets.
        - ``language``          — 4 instructions.
        - ``viewpoint``         — 4 ExternalCamera poses.
    """
    out: Dict[str, List[pert.PerturbationConfig]] = {}

    out["distractors"] = [
        _copy_with(
            baseline,
            distractors=_make_distractors(n, seed=baseline.seed + 100 + n),
        )
        for n in _DISTRACTOR_COUNTS
    ]

    out["dome_intensity"] = [
        _copy_with(baseline, dome_intensity=float(v))
        for v in _DOME_INTENSITIES
    ]

    out["plate_pose"] = [
        _copy_with(baseline, plate_pose_offset_m=tuple(v))
        for v in _PLATE_OFFSETS
    ]

    out["crate_pose"] = [
        _copy_with(baseline, crate_pose_offset_m=tuple(v))
        for v in _CRATE_OFFSETS
    ]

    out["language"] = [
        _copy_with(baseline, language_prompt=p)
        for p in language_templates()
    ]

    out["viewpoint"] = [
        _copy_with(
            baseline,
            external_camera_translation_offset_m=tuple(t),
            external_camera_rotation_offset_deg=tuple(r),
        )
        for (t, r) in _VIEWPOINT_OFFSETS
    ]

    return out


# --------------------------------------------------------------------
# Latin hypercube sampling
# --------------------------------------------------------------------

def _lhs_unit(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Plain Latin hypercube sample in [0,1)^d with ``n`` rows.

    Each column independently shuffles the centered strata so each
    1D projection has exactly one sample per strata.
    """
    out = np.empty((n, d), dtype=np.float64)
    for k in range(d):
        strata = (np.arange(n) + rng.random(n)) / n   # n points, one per stratum
        rng.shuffle(strata)
        out[:, k] = strata
    return out


def latin_hypercube_sample(
    baseline: pert.PerturbationConfig,
    n: int,
    seed: int,
) -> List[pert.PerturbationConfig]:
    """Sample ``n`` joint perturbations via a Latin hypercube design.

    Axes jointly varied (one column each):
        0: plate_x_offset   in [-0.10, 0.10] m
        1: plate_y_offset   in [-0.10, 0.10] m
        2: plate_yaw        in [-pi/4, pi/4] rad
        3: crate_x_offset   in [-0.10, 0.10] m
        4: crate_y_offset   in [-0.10, 0.10] m
        5: crate_yaw        in [-pi/4, pi/4] rad
        6: dome_intensity   in [0.3, 3.0]
        7: cam_translate_x  in [-0.05, 0.05] m
        8: cam_rotate_y_deg in [-5, 5] deg
        9: n_distractors    discrete in {0, 2, 4, 8}

    Language axis is NOT jointly varied (it's the most disruptive axis;
    we sweep it separately via ``single_axis_sweeps``). All ``n`` configs
    inherit ``baseline.language_prompt``.
    """
    if n <= 0:
        return []
    rng = np.random.default_rng(seed)
    u = _lhs_unit(n, 10, rng)

    def lerp(col: int, lo: float, hi: float) -> np.ndarray:
        return lo + (hi - lo) * u[:, col]

    plate_x = lerp(0, -0.10, 0.10)
    plate_y = lerp(1, -0.10, 0.10)
    plate_yaw = lerp(2, -math.pi / 4, math.pi / 4)
    crate_x = lerp(3, -0.10, 0.10)
    crate_y = lerp(4, -0.10, 0.10)
    crate_yaw = lerp(5, -math.pi / 4, math.pi / 4)
    dome = lerp(6, 0.3, 3.0)
    cam_tx = lerp(7, -0.05, 0.05)
    cam_ry = lerp(8, -5.0, 5.0)
    # Map column 9 in [0,1) to one of {0, 2, 4, 8}.
    distractor_bin = np.clip((u[:, 9] * len(_DISTRACTOR_COUNTS)).astype(int), 0, len(_DISTRACTOR_COUNTS) - 1)

    configs: List[pert.PerturbationConfig] = []
    for i in range(n):
        n_dist = int(_DISTRACTOR_COUNTS[distractor_bin[i]])
        cfg = _copy_with(
            baseline,
            plate_pose_offset_m=(float(plate_x[i]), float(plate_y[i]), 0.0),
            plate_yaw_offset_rad=float(plate_yaw[i]),
            crate_pose_offset_m=(float(crate_x[i]), float(crate_y[i]), 0.0),
            crate_yaw_offset_rad=float(crate_yaw[i]),
            dome_intensity=float(dome[i]),
            external_camera_translation_offset_m=(float(cam_tx[i]), 0.0, 0.0),
            external_camera_rotation_offset_deg=(0.0, float(cam_ry[i]), 0.0),
            distractors=_make_distractors(n_dist, seed=seed + 1000 + i),
            seed=int(seed + i),
        )
        configs.append(cfg)
    return configs


# --------------------------------------------------------------------
# Serialization helper
# --------------------------------------------------------------------

def write_configs(configs: List[pert.PerturbationConfig], output_dir: Path) -> List[Path]:
    """Write each config to ``output_dir/config_<idx>.json``.

    Returns the list of written paths in input order.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    width = max(3, len(str(max(0, len(configs) - 1))))
    for i, cfg in enumerate(configs):
        p = output_dir / f"config_{i:0{width}d}.json"
        pert.save_config(cfg, p)
        paths.append(p)
    return paths
