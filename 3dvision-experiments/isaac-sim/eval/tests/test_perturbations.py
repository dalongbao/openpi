"""Tests for the perturbation harness.

These tests intentionally avoid Isaac Sim. The dataclass/JSON layer is
exercised directly; ``apply()`` is exercised against a MockStage that
implements the small slice of the pxr API we touch — IF pxr is not
available the apply()-touching tests are skipped.

Run with:

    cd 3dvision-experiments/isaac-sim
    python -m pytest eval/tests -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Allow running from the repo root or from 3dvision-experiments/isaac-sim/.
# We deliberately do NOT import the ``eval`` package — its __init__.py pulls
# in ``core``/``runner`` (Isaac Sim deps) that aren't needed for these tests.
# Instead we add the ``eval/`` directory itself to sys.path so the modules
# resolve as top-level imports.
_HERE = Path(__file__).resolve().parent
_EVAL_DIR = _HERE.parent                          # .../isaac-sim/eval
_ISAAC_DIR = _EVAL_DIR.parent                     # .../isaac-sim
sys.path.insert(0, str(_EVAL_DIR))

import perturbations as pert            # noqa: E402
import perturbation_sweeps as sweeps    # noqa: E402


# --------------------------------------------------------------------
# baseline_config / dataclass behavior
# --------------------------------------------------------------------

def test_baseline_is_noop():
    cfg = pert.baseline_config()
    assert cfg.distractors == []
    assert cfg.dome_intensity is None
    assert cfg.dome_color is None
    assert cfg.plate_pose_offset_m == (0.0, 0.0, 0.0)
    assert cfg.plate_yaw_offset_rad == 0.0
    assert cfg.crate_pose_offset_m == (0.0, 0.0, 0.0)
    assert cfg.crate_yaw_offset_rad == 0.0
    assert cfg.external_camera_translation_offset_m == (0.0, 0.0, 0.0)
    assert cfg.external_camera_rotation_offset_deg == (0.0, 0.0, 0.0)
    assert cfg.language_prompt == "put the plate in the crate"
    assert cfg.seed == 0


def test_baseline_custom_prompt():
    cfg = pert.baseline_config("place the small dish into the yellow box")
    assert cfg.language_prompt == "place the small dish into the yellow box"


# --------------------------------------------------------------------
# JSON IO round-trip
# --------------------------------------------------------------------

def test_load_config_round_trip(tmp_path):
    original = pert.PerturbationConfig(
        distractors=[
            pert.DistractorSpec(
                prim_path="/World/Distractor_0",
                usd_path="/workspace/assets/distractors/mug.usd",
                translate=(0.1, 0.2, 0.78),
                rotate_yaw_rad=0.5,
            ),
        ],
        dome_intensity=0.3,
        dome_color=(1.0, 0.9, 0.8),
        plate_pose_offset_m=(0.05, -0.02, 0.0),
        plate_yaw_offset_rad=0.1,
        crate_pose_offset_m=(-0.05, 0.0, 0.0),
        crate_yaw_offset_rad=-0.1,
        external_camera_translation_offset_m=(0.0, 0.0, 0.05),
        external_camera_rotation_offset_deg=(2.0, 0.0, 0.0),
        language_prompt="place the small dish into the yellow box",
        seed=42,
    )
    p = tmp_path / "round_trip.json"
    pert.save_config(original, p)
    loaded = pert.load_config(p)

    assert loaded.distractors[0].prim_path == "/World/Distractor_0"
    assert loaded.distractors[0].usd_path == "/workspace/assets/distractors/mug.usd"
    assert loaded.distractors[0].translate == (0.1, 0.2, 0.78)
    assert loaded.distractors[0].rotate_yaw_rad == pytest.approx(0.5)
    assert loaded.dome_intensity == pytest.approx(0.3)
    assert loaded.dome_color == (1.0, 0.9, 0.8)
    assert loaded.plate_pose_offset_m == (0.05, -0.02, 0.0)
    assert loaded.plate_yaw_offset_rad == pytest.approx(0.1)
    assert loaded.crate_pose_offset_m == (-0.05, 0.0, 0.0)
    assert loaded.crate_yaw_offset_rad == pytest.approx(-0.1)
    assert loaded.external_camera_translation_offset_m == (0.0, 0.0, 0.05)
    assert loaded.external_camera_rotation_offset_deg == (2.0, 0.0, 0.0)
    assert loaded.language_prompt == "place the small dish into the yellow box"
    assert loaded.seed == 42


def test_load_config_ignores_unknown_keys(tmp_path):
    p = tmp_path / "future.json"
    p.write_text(json.dumps({
        "language_prompt": "ok",
        "future_key": "ignored",
        "seed": 7,
    }))
    cfg = pert.load_config(p)
    assert cfg.language_prompt == "ok"
    assert cfg.seed == 7
    # Defaults filled in for missing keys:
    assert cfg.distractors == []
    assert cfg.dome_intensity is None


def test_load_sample_configs_in_repo():
    """The hand-written sample configs in eval/configs/ should all parse."""
    configs_dir = _ISAAC_DIR / "eval" / "configs"
    json_files = sorted(configs_dir.glob("*.json"))
    assert len(json_files) >= 5, f"Expected >=5 sample configs, got {json_files}"
    for jf in json_files:
        cfg = pert.load_config(jf)
        # All sample configs should at least have a prompt.
        assert isinstance(cfg.language_prompt, str)


# --------------------------------------------------------------------
# Sweep generators
# --------------------------------------------------------------------

def test_single_axis_sweeps_counts():
    base = pert.baseline_config()
    out = sweeps.single_axis_sweeps(base)
    assert set(out.keys()) == {
        "distractors", "dome_intensity", "plate_pose",
        "crate_pose", "language", "viewpoint",
    }
    assert len(out["distractors"]) == 4
    assert len(out["dome_intensity"]) == 3
    assert len(out["plate_pose"]) == 5
    assert len(out["crate_pose"]) == 5
    assert len(out["language"]) == 4
    assert len(out["viewpoint"]) == 4


def test_single_axis_sweeps_only_change_one_axis():
    """Every config along an axis differs from baseline ONLY on that axis."""
    base = pert.baseline_config()
    out = sweeps.single_axis_sweeps(base)

    # 'dome_intensity' axis: only dome_intensity should differ.
    for cfg in out["dome_intensity"]:
        assert cfg.distractors == base.distractors
        assert cfg.plate_pose_offset_m == base.plate_pose_offset_m
        assert cfg.crate_pose_offset_m == base.crate_pose_offset_m
        assert cfg.language_prompt == base.language_prompt
        # dome_intensity itself should be set (not None) on every entry.
        assert cfg.dome_intensity is not None

    # 'plate_pose' axis: only plate_pose_offset_m should differ.
    for cfg in out["plate_pose"]:
        assert cfg.dome_intensity == base.dome_intensity
        assert cfg.crate_pose_offset_m == base.crate_pose_offset_m
        assert cfg.distractors == base.distractors
        assert cfg.language_prompt == base.language_prompt


def test_language_templates_count():
    t = sweeps.language_templates()
    assert len(t) == 4
    assert t[0] != ""           # baseline non-empty
    assert t[3] == ""           # no-language is the empty string
    assert len({*t}) == 4       # all distinct


def test_latin_hypercube_distinct():
    base = pert.baseline_config()
    configs = sweeps.latin_hypercube_sample(base, n=20, seed=123)
    assert len(configs) == 20
    # Distinct by their plate offset (LHS guarantees uniqueness per stratum
    # in continuous columns).
    plate_offsets = {tuple(c.plate_pose_offset_m) for c in configs}
    assert len(plate_offsets) == 20


def test_latin_hypercube_zero():
    base = pert.baseline_config()
    assert sweeps.latin_hypercube_sample(base, n=0, seed=0) == []


def test_latin_hypercube_seeded():
    """Same seed -> same configs."""
    base = pert.baseline_config()
    a = sweeps.latin_hypercube_sample(base, n=5, seed=7)
    b = sweeps.latin_hypercube_sample(base, n=5, seed=7)
    for ca, cb in zip(a, b):
        assert ca.plate_pose_offset_m == cb.plate_pose_offset_m
        assert ca.crate_pose_offset_m == cb.crate_pose_offset_m
        assert ca.dome_intensity == cb.dome_intensity


def test_write_configs_round_trip(tmp_path):
    base = pert.baseline_config()
    configs = sweeps.latin_hypercube_sample(base, n=5, seed=11)
    paths = sweeps.write_configs(configs, tmp_path)
    assert len(paths) == 5
    for p, cfg in zip(paths, configs):
        loaded = pert.load_config(p)
        assert loaded.plate_pose_offset_m == cfg.plate_pose_offset_m
        assert loaded.dome_intensity == pytest.approx(cfg.dome_intensity)


# --------------------------------------------------------------------
# apply() against a MockStage. Skipped when pxr is missing.
# --------------------------------------------------------------------

try:
    import pxr  # noqa: F401
    _HAS_PXR = True
except ImportError:
    _HAS_PXR = False

pxr_only = pytest.mark.skipif(not _HAS_PXR, reason="pxr (USD core) not installed")


class _MockStage:
    """Tiny stage substitute that owns an in-memory USD via Sdf.Layer."""

    def __init__(self):
        from pxr import Usd, UsdGeom, UsdLux, Gf
        self._stage = Usd.Stage.CreateInMemory()
        # Define the prims we need.
        UsdGeom.Xform.Define(self._stage, "/World")
        UsdGeom.Xform.Define(self._stage, "/World/plate_small")
        UsdGeom.Xform.Define(self._stage, "/World/SM_Crate_A07_Yellow_01_physics")
        UsdGeom.Xform.Define(self._stage, "/World/ExternalCamera")
        UsdGeom.Xform.Define(self._stage, "/World/RecordingCamera")
        UsdGeom.Xform.Define(self._stage, "/World/Environment")
        dome = UsdLux.DomeLight.Define(self._stage, "/World/Environment/DomeLight")
        dome.CreateIntensityAttr(500.0)

        # Give plate and crate an initial translate so offsets test add-on semantics.
        api = UsdGeom.XformCommonAPI(self._stage.GetPrimAtPath("/World/plate_small"))
        api.SetTranslate(Gf.Vec3d(0.2, 0.3, 0.78))
        api = UsdGeom.XformCommonAPI(self._stage.GetPrimAtPath("/World/SM_Crate_A07_Yellow_01_physics"))
        api.SetTranslate(Gf.Vec3d(0.0, 0.5, 0.78))

    def GetPrimAtPath(self, path):
        return self._stage.GetPrimAtPath(path)

    def Traverse(self):
        return self._stage.Traverse()


@pxr_only
def test_apply_baseline_noop():
    s = _MockStage()
    realized = pert.apply(s, pert.baseline_config())
    assert realized["plate"] is None
    assert realized["crate"] is None
    assert realized["external_camera"] is None
    assert realized["dome_light"] is None
    assert realized["distractors"] == []
    assert realized["language_prompt"] == "put the plate in the crate"


@pxr_only
def test_apply_plate_offset():
    s = _MockStage()
    cfg = pert.baseline_config()
    cfg.plate_pose_offset_m = (0.05, -0.02, 0.0)
    realized = pert.apply(s, cfg)
    assert realized["plate"] is not None
    # Plate started at (0.2, 0.3, 0.78); offset (0.05, -0.02, 0).
    nt = realized["plate"]["new_translate"]
    assert nt[0] == pytest.approx(0.25)
    assert nt[1] == pytest.approx(0.28)
    assert nt[2] == pytest.approx(0.78)


@pxr_only
def test_apply_plate_offset_clamp():
    s = _MockStage()
    cfg = pert.baseline_config()
    # Push way past table bounds.
    cfg.plate_pose_offset_m = (5.0, 5.0, 0.0)
    realized = pert.apply(s, cfg)
    nt = realized["plate"]["new_translate"]
    xmin, xmax, ymin, ymax = pert.TABLE_BOUNDS_XY
    assert nt[0] == pytest.approx(xmax)
    assert nt[1] == pytest.approx(ymax)


@pxr_only
def test_apply_dome_intensity_scales():
    s = _MockStage()
    cfg = pert.baseline_config()
    cfg.dome_intensity = 0.3
    realized = pert.apply(s, cfg)
    assert realized["dome_light"]["previous_intensity"] == pytest.approx(500.0)
    assert realized["dome_light"]["new_intensity"] == pytest.approx(150.0)
    assert realized["dome_light"]["intensity_scale"] == pytest.approx(0.3)


@pxr_only
def test_apply_recording_camera_untouched():
    """RecordingCamera must NEVER be modified."""
    from pxr import UsdGeom
    s = _MockStage()
    cfg = pert.baseline_config()
    cfg.external_camera_translation_offset_m = (0.1, 0.0, 0.0)
    pert.apply(s, cfg)
    rec = s.GetPrimAtPath(pert.RECORDING_CAMERA_PRIM)
    xform = UsdGeom.Xformable(rec)
    # No xform ops added (or all zero).
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            t = op.Get()
            if t is not None:
                assert tuple(t) == (0.0, 0.0, 0.0)


@pxr_only
def test_apply_distractor_creates_prim():
    s = _MockStage()
    cfg = pert.baseline_config()
    cfg.distractors = [
        pert.DistractorSpec(
            prim_path="/World/Distractor_test",
            usd_path="/workspace/assets/distractors/mug.usd",
            translate=(0.15, 0.20, 0.78),
            rotate_yaw_rad=0.0,
        )
    ]
    realized = pert.apply(s, cfg)
    assert len(realized["distractors"]) == 1
    prim = s.GetPrimAtPath("/World/Distractor_test")
    assert prim.IsValid()


@pxr_only
def test_apply_idempotent_no_distractor_duplication():
    """Calling apply twice with the same config doesn't multiply distractors."""
    s = _MockStage()
    cfg = pert.baseline_config()
    cfg.distractors = [
        pert.DistractorSpec(
            prim_path="/World/Distractor_x",
            usd_path="/workspace/assets/distractors/mug.usd",
            translate=(0.1, 0.1, 0.78),
        )
    ]
    pert.apply(s, cfg)
    pert.apply(s, cfg)
    # Same prim path: not duplicated.
    prim = s.GetPrimAtPath("/World/Distractor_x")
    assert prim.IsValid()
    # There should be exactly one child under /World matching this name.
    world = s.GetPrimAtPath("/World")
    matching = [c for c in world.GetChildren() if c.GetName() == "Distractor_x"]
    assert len(matching) == 1
