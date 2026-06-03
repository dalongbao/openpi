"""Perturbations harness for the Isaac Sim kitchen scene.

Mutates the USD stage in-place to produce scene variants for pi0.5
evaluation. Designed to be invoked by ``eval/core.py`` BEFORE
``world.reset()``.

Public API
----------
- ``PerturbationConfig`` — dataclass describing one perturbation.
- ``DistractorSpec`` — dataclass describing one additional object.
- ``apply(stage, config) -> dict`` — mutate the stage; return realized values.
- ``load_config(path) -> PerturbationConfig`` — JSON deserializer.
- ``save_config(config, path) -> None`` — JSON serializer.
- ``baseline_config(language_prompt) -> PerturbationConfig`` — no-op preset.

Design rules
------------
- ``pxr`` is imported LAZILY inside ``apply()`` so the module is importable
  on a laptop without Isaac Sim. The dataclasses and JSON IO have no pxr
  dependency.
- All pose offsets are ADDITIVE on top of the current world pose. Never
  chain multiplicatively across calls.
- Positions on the table are clamped to ``TABLE_BOUNDS_XY`` to keep things
  on the table (the scene has no ground plane).
- ``apply()`` is safe to call once per stage. Calling twice with the same
  config is idempotent for everything except distractor placement (which
  is keyed by ``DistractorSpec.prim_path`` — reusing the same path
  overwrites the previous one).

Prim paths touched (kept in sync with eval_script_1.py)
-------------------------------------------------------
- ``/World/plate_small`` — translate/yaw offset.
- ``/World/SM_Crate_A07_Yellow_01_physics`` — translate/yaw offset.
- ``/World/ExternalCamera`` — translate/rotate offset (policy input cam).
- ``/World/RecordingCamera`` — NEVER touched (HD recorder must stay fixed).
- ``/World/fr3`` — NEVER moved.
- Any ``UsdLuxDomeLight`` under ``/World/Environment/`` — intensity/color.
- ``/World/Distractor_<i>`` — newly created distractor objects.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# --------------------------------------------------------------------
# Scene constants — kept consistent with the original eval_script_1.py.
# Read by both apply() and the sweep generator.
# --------------------------------------------------------------------

# Table-top XY bounds (meters, world frame). Anything placed on the table
# is clamped to this AABB so it does not fall off (no ground plane in the
# scene). Tuned to the packing table footprint as used by the original
# eval; widen here if the scene changes.
TABLE_BOUNDS_XY: Tuple[float, float, float, float] = (-0.40, 0.40, 0.00, 0.60)
# Vertical clamp for object placement (table surface to ~30 cm above).
TABLE_BOUNDS_Z: Tuple[float, float] = (0.70, 1.00)

# Prim paths.
PLATE_PRIM = "/World/plate_small"
CRATE_PRIM = "/World/SM_Crate_A07_Yellow_01_physics"
EXTERNAL_CAMERA_PRIM = "/World/ExternalCamera"
RECORDING_CAMERA_PRIM = "/World/RecordingCamera"
FR3_PRIM = "/World/fr3"
ENVIRONMENT_PRIM = "/World/Environment"


# --------------------------------------------------------------------
# Public dataclasses
# --------------------------------------------------------------------

@dataclass
class DistractorSpec:
    """One additional object placed on the table.

    The ``usd_path`` should point to a local USD on the compute node, e.g.
    ``/workspace/assets/distractors/mug.usd``. Spawning a distractor that
    references S3 will hang on Euler compute nodes (no internet).
    """
    prim_path: str
    usd_path: str
    translate: Tuple[float, float, float] = (0.0, 0.0, 0.75)
    rotate_yaw_rad: float = 0.0


@dataclass
class PerturbationConfig:
    """A single perturbation to apply to the scene.

    All offsets are deltas from the baseline pose loaded from USD. The
    runner calls ``apply(stage, config)`` once, before ``world.reset()``.
    """
    distractors: List[DistractorSpec] = field(default_factory=list)
    dome_intensity: Optional[float] = None
    dome_color: Optional[Tuple[float, float, float]] = None
    plate_pose_offset_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    plate_yaw_offset_rad: float = 0.0
    crate_pose_offset_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    crate_yaw_offset_rad: float = 0.0
    external_camera_translation_offset_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    external_camera_rotation_offset_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    language_prompt: str = "put the plate in the crate"
    seed: int = 0


# --------------------------------------------------------------------
# Baseline / serialization
# --------------------------------------------------------------------

def baseline_config(language_prompt: str = "put the plate in the crate") -> PerturbationConfig:
    """Return a no-op perturbation config.

    ``apply()`` on this config does not change object poses, lighting or
    cameras, and adds no distractors.
    """
    return PerturbationConfig(language_prompt=language_prompt)


def _config_to_dict(config: PerturbationConfig) -> Dict[str, Any]:
    d = dataclasses.asdict(config)
    # dataclasses.asdict turns tuples into lists; that's fine for JSON.
    return d


def save_config(config: PerturbationConfig, path: Union[str, Path]) -> Path:
    """Serialize a PerturbationConfig to JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(_config_to_dict(config), f, indent=2)
    return p


def load_config(path: Union[str, Path]) -> PerturbationConfig:
    """Load a PerturbationConfig from JSON.

    Unknown keys are ignored (forward compatibility). Missing keys fall
    back to dataclass defaults.
    """
    p = Path(path)
    with open(p, "r") as f:
        raw = json.load(f)
    return _config_from_dict(raw)


def _config_from_dict(raw: Dict[str, Any]) -> PerturbationConfig:
    distractors_raw = raw.get("distractors", []) or []
    distractors: List[DistractorSpec] = []
    for d in distractors_raw:
        distractors.append(
            DistractorSpec(
                prim_path=d["prim_path"],
                usd_path=d["usd_path"],
                translate=tuple(d.get("translate", (0.0, 0.0, 0.75))),
                rotate_yaw_rad=float(d.get("rotate_yaw_rad", 0.0)),
            )
        )

    def _tup3(key: str, default: Tuple[float, float, float]) -> Tuple[float, float, float]:
        v = raw.get(key, default)
        if v is None:
            return default
        return (float(v[0]), float(v[1]), float(v[2]))

    dome_color = raw.get("dome_color")
    if dome_color is not None:
        dome_color = (float(dome_color[0]), float(dome_color[1]), float(dome_color[2]))

    dome_intensity = raw.get("dome_intensity")
    if dome_intensity is not None:
        dome_intensity = float(dome_intensity)

    return PerturbationConfig(
        distractors=distractors,
        dome_intensity=dome_intensity,
        dome_color=dome_color,
        plate_pose_offset_m=_tup3("plate_pose_offset_m", (0.0, 0.0, 0.0)),
        plate_yaw_offset_rad=float(raw.get("plate_yaw_offset_rad", 0.0)),
        crate_pose_offset_m=_tup3("crate_pose_offset_m", (0.0, 0.0, 0.0)),
        crate_yaw_offset_rad=float(raw.get("crate_yaw_offset_rad", 0.0)),
        external_camera_translation_offset_m=_tup3("external_camera_translation_offset_m", (0.0, 0.0, 0.0)),
        external_camera_rotation_offset_deg=_tup3("external_camera_rotation_offset_deg", (0.0, 0.0, 0.0)),
        language_prompt=str(raw.get("language_prompt", "put the plate in the crate")),
        seed=int(raw.get("seed", 0)),
    )


# --------------------------------------------------------------------
# Internal helpers (pxr only used inside apply() / its callees)
# --------------------------------------------------------------------

def _clamp_xy(x: float, y: float) -> Tuple[float, float]:
    xmin, xmax, ymin, ymax = TABLE_BOUNDS_XY
    return (max(xmin, min(xmax, x)), max(ymin, min(ymax, y)))


def _clamp_z(z: float) -> float:
    zmin, zmax = TABLE_BOUNDS_Z
    return max(zmin, min(zmax, z))


def _find_dome_light(stage) -> Optional[str]:
    """Walk the stage and return the path of the first DomeLight prim found.

    Returns the prim path as a string (so callers don't need pxr types).
    """
    from pxr import UsdLux  # type: ignore

    for prim in stage.Traverse():
        if prim.IsA(UsdLux.DomeLight):
            return str(prim.GetPath())
    return None


def _get_xform_translate(prim) -> Tuple[float, float, float]:
    """Return the current local translate of an Xform prim, or (0,0,0)."""
    from pxr import UsdGeom  # type: ignore

    xform = UsdGeom.Xformable(prim)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            t = op.Get()
            if t is None:
                return (0.0, 0.0, 0.0)
            return (float(t[0]), float(t[1]), float(t[2]))
    return (0.0, 0.0, 0.0)


def _get_xform_rotate_xyz_deg(prim) -> Tuple[float, float, float]:
    """Return the current local rotateXYZ (deg) of an Xform prim."""
    from pxr import UsdGeom  # type: ignore

    xform = UsdGeom.Xformable(prim)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            r = op.Get()
            if r is None:
                return (0.0, 0.0, 0.0)
            return (float(r[0]), float(r[1]), float(r[2]))
    return (0.0, 0.0, 0.0)


def _set_xform_translate_rotate(prim, translate, rotate_deg) -> None:
    """Set translate and rotateXYZ on a prim via XformCommonAPI.

    Preserves the prim's scale/pivot via XformCommonAPI semantics.
    """
    from pxr import UsdGeom, Gf  # type: ignore

    api = UsdGeom.XformCommonAPI(prim)
    api.SetTranslate(Gf.Vec3d(float(translate[0]), float(translate[1]), float(translate[2])))
    api.SetRotate(
        Gf.Vec3f(float(rotate_deg[0]), float(rotate_deg[1]), float(rotate_deg[2])),
        UsdGeom.XformCommonAPI.RotationOrderXYZ,
    )


def _apply_pose_offset(
    stage,
    prim_path: str,
    translate_offset: Tuple[float, float, float],
    yaw_offset_rad: float,
    clamp_to_table: bool,
) -> Optional[Dict[str, Any]]:
    """Apply an additive translate + yaw offset to a prim.

    Reads the current local pose, adds the offset, optionally clamps the
    XY position to the table bounds, and writes back via XformCommonAPI.
    Returns a dict of realized values, or None if the prim is missing.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None

    cur_t = _get_xform_translate(prim)
    cur_r = _get_xform_rotate_xyz_deg(prim)

    new_x = cur_t[0] + float(translate_offset[0])
    new_y = cur_t[1] + float(translate_offset[1])
    new_z = cur_t[2] + float(translate_offset[2])

    if clamp_to_table:
        new_x, new_y = _clamp_xy(new_x, new_y)
        new_z = _clamp_z(new_z)

    yaw_deg_offset = math.degrees(float(yaw_offset_rad))
    new_r = (cur_r[0], cur_r[1], cur_r[2] + yaw_deg_offset)

    _set_xform_translate_rotate(prim, (new_x, new_y, new_z), new_r)
    return {
        "prim_path": prim_path,
        "previous_translate": list(cur_t),
        "previous_rotate_xyz_deg": list(cur_r),
        "new_translate": [new_x, new_y, new_z],
        "new_rotate_xyz_deg": list(new_r),
    }


def _apply_camera_offset(
    stage,
    prim_path: str,
    translate_offset: Tuple[float, float, float],
    rotate_offset_deg: Tuple[float, float, float],
) -> Optional[Dict[str, Any]]:
    """Apply an additive translate + per-axis rotation offset to a camera prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None

    cur_t = _get_xform_translate(prim)
    cur_r = _get_xform_rotate_xyz_deg(prim)

    new_t = (
        cur_t[0] + float(translate_offset[0]),
        cur_t[1] + float(translate_offset[1]),
        cur_t[2] + float(translate_offset[2]),
    )
    new_r = (
        cur_r[0] + float(rotate_offset_deg[0]),
        cur_r[1] + float(rotate_offset_deg[1]),
        cur_r[2] + float(rotate_offset_deg[2]),
    )

    _set_xform_translate_rotate(prim, new_t, new_r)
    return {
        "prim_path": prim_path,
        "previous_translate": list(cur_t),
        "previous_rotate_xyz_deg": list(cur_r),
        "new_translate": list(new_t),
        "new_rotate_xyz_deg": list(new_r),
    }


def _apply_dome_light(
    stage,
    intensity: Optional[float],
    color: Optional[Tuple[float, float, float]],
) -> Optional[Dict[str, Any]]:
    """Scale the dome light intensity and/or set its color."""
    if intensity is None and color is None:
        return None

    from pxr import UsdLux, Gf  # type: ignore

    dome_path = _find_dome_light(stage)
    if dome_path is None:
        return {"prim_path": None, "warning": "no UsdLuxDomeLight found in stage"}

    prim = stage.GetPrimAtPath(dome_path)
    dome = UsdLux.DomeLight(prim)

    realized: Dict[str, Any] = {"prim_path": dome_path}

    if intensity is not None:
        attr = dome.GetIntensityAttr()
        # Multiplicative scaling on top of the baseline intensity stored
        # in USD. A factor of 1.0 leaves the scene unchanged.
        baseline = attr.Get()
        if baseline is None:
            baseline = 1000.0  # USD default for DomeLight
        new_intensity = float(baseline) * float(intensity)
        attr.Set(new_intensity)
        realized["previous_intensity"] = float(baseline)
        realized["intensity_scale"] = float(intensity)
        realized["new_intensity"] = new_intensity

    if color is not None:
        attr = dome.GetColorAttr()
        attr.Set(Gf.Vec3f(float(color[0]), float(color[1]), float(color[2])))
        realized["new_color"] = list(color)

    return realized


def _add_distractor(stage, spec: DistractorSpec) -> Dict[str, Any]:
    """Create / overwrite a distractor prim on the stage.

    Uses ``Sdf.CreatePrimInLayer`` + payload reference rather than the
    Isaac Sim ``add_reference_to_stage`` helper, so this module remains
    importable without Isaac Sim. The result is functionally equivalent:
    a defined Xform prim with a payload to ``spec.usd_path``.

    Position is clamped to the table bounds. RigidBody / Collision API
    is applied if not already present on the new prim so the distractor
    behaves as a physical object.
    """
    from pxr import UsdGeom, UsdPhysics, Sdf, Gf  # type: ignore

    prim_path = spec.prim_path

    # Clamp the spawn position to the table.
    tx, ty = _clamp_xy(spec.translate[0], spec.translate[1])
    tz = _clamp_z(spec.translate[2])

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        # DefinePrim creates an Xform. We then attach a payload.
        prim = UsdGeom.Xform.Define(stage, Sdf.Path(prim_path)).GetPrim()

    # Refresh payload to point at the local USD asset.
    prim.GetPayloads().ClearPayloads()
    prim.GetPayloads().AddPayload(spec.usd_path)

    # Set translate + yaw via XformCommonAPI.
    yaw_deg = math.degrees(float(spec.rotate_yaw_rad))
    _set_xform_translate_rotate(prim, (tx, ty, tz), (0.0, 0.0, yaw_deg))

    # Physics: attach RigidBody + Collision API if not already present.
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(prim)
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(prim)

    return {
        "prim_path": prim_path,
        "usd_path": spec.usd_path,
        "translate": [tx, ty, tz],
        "rotate_yaw_rad": float(spec.rotate_yaw_rad),
        "clamped": (tx, ty) != (spec.translate[0], spec.translate[1]) or tz != spec.translate[2],
    }


# --------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------

def apply(stage, config: Union[PerturbationConfig, Dict[str, Any]]) -> Dict[str, Any]:
    """Mutate ``stage`` in-place per ``config``.

    Must be called AFTER the payload patches in ``eval_script_1.py`` and
    BEFORE ``World(...)`` / ``world.reset()``. Returns a dict of realized
    values (some inputs may be clamped) suitable for logging.

    ``config`` may also be a dict (the dict that came out of ``json.load``)
    for backwards compatibility with the old runner stub.
    """
    if isinstance(config, dict):
        config = _config_from_dict(config)

    realized: Dict[str, Any] = {
        "language_prompt": config.language_prompt,
        "seed": int(config.seed),
        "plate": None,
        "crate": None,
        "external_camera": None,
        "dome_light": None,
        "distractors": [],
    }

    # 1. Plate pose offset.
    if any(v != 0.0 for v in config.plate_pose_offset_m) or config.plate_yaw_offset_rad != 0.0:
        realized["plate"] = _apply_pose_offset(
            stage,
            PLATE_PRIM,
            config.plate_pose_offset_m,
            config.plate_yaw_offset_rad,
            clamp_to_table=True,
        )

    # 2. Crate pose offset.
    if any(v != 0.0 for v in config.crate_pose_offset_m) or config.crate_yaw_offset_rad != 0.0:
        realized["crate"] = _apply_pose_offset(
            stage,
            CRATE_PRIM,
            config.crate_pose_offset_m,
            config.crate_yaw_offset_rad,
            clamp_to_table=True,
        )

    # 3. External (policy) camera offset. RecordingCamera is untouched.
    if (
        any(v != 0.0 for v in config.external_camera_translation_offset_m)
        or any(v != 0.0 for v in config.external_camera_rotation_offset_deg)
    ):
        realized["external_camera"] = _apply_camera_offset(
            stage,
            EXTERNAL_CAMERA_PRIM,
            config.external_camera_translation_offset_m,
            config.external_camera_rotation_offset_deg,
        )

    # 4. Dome light.
    if config.dome_intensity is not None or config.dome_color is not None:
        realized["dome_light"] = _apply_dome_light(stage, config.dome_intensity, config.dome_color)

    # 5. Distractors.
    for spec in config.distractors:
        realized["distractors"].append(_add_distractor(stage, spec))

    return realized
