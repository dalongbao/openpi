"""Eval core: EvalConfig, EvalResult, EvalSim.

This module owns the Isaac Sim lifecycle for a single pi0.5 rollout. It is
a direct refactor of the legacy ``eval_script_1.py`` — behavior is
preserved bit-for-bit when invoked with defaults. The new wrapper merely
adds:

  * dataclass-based config so the CLI can populate it.
  * seed plumbing (Python random, NumPy, JAX) so sweeps are reproducible.
  * optional perturbation + probe hooks (deferred imports — no overhead on
    parity runs).
  * a structured ``EvalResult`` so downstream metrics code doesn't have to
    re-read CSVs.

Critical Isaac Sim quirks preserved from the legacy script (do NOT remove
unless verifying on Euler):

  * ``SimulationApp`` must be the first Isaac-related import. We do this
    inside ``EvalSim.setup()`` so that simply importing this module on a
    laptop (for static checks / metrics analysis) does NOT spin up Isaac.
  * ``typing_extensions`` is force-reloaded from ``/isaac_packages``
    because Isaac Sim caches an ancient copy at startup.
  * All scene payloads are repatched from S3 URLs to local USDs in
    ``/workspace/assets/``.
  * ``franka.initialize()`` is called AGAIN after ``world.reset()`` (the
    articulation view is cleared by reset).
  * Cameras need a 20-step warmup before they yield non-empty buffers.
  * ``franka.get_joint_positions()`` can return ``None`` at step 0 — guard.
"""

from __future__ import annotations

import csv
import dataclasses
import importlib.util as _ilu
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# NOTE: we intentionally do NOT import Isaac Sim, torch, cv2, openpi, pxr,
# omni, etc. at module load. They are imported lazily inside EvalSim.setup
# so that this file is importable on a laptop for offline analysis.


# --------------------------------------------------------------------
# Defaults (mirror the legacy script verbatim).
# --------------------------------------------------------------------
DEFAULT_USD_PATH       = "/workspace/kitchen_scene_1.usd"
DEFAULT_CHECKPOINT_DIR = "/checkpoints/pi05_egoverse/test/29999"
DEFAULT_OUTPUT_DIR     = "/workspace"
DEFAULT_NUM_STEPS      = 3000
DEFAULT_LANG_PROMPT    = "place the plate into the yellow crate"

NUM_ARM_JOINTS = 7
POLICY_CAM_RES = (224, 224)
HD_VIDEO_RES   = (1280, 720)

# Map of prim path -> local USD file. Verbatim from legacy script.
_PAYLOAD_PATCHES = {
    "/World/fr3":                             "/workspace/assets/fr3_full/fr3.usd",
    "/World/SM_HeavyDutyPackingTable_C02_01": "/workspace/assets/table/SM_HeavyDutyPackingTable_C02_01.usd",
    "/World/plate_small":                     "/workspace/assets/plate/plate_small.usd",
    "/World/SM_Crate_A07_Yellow_01_physics":  "/workspace/assets/crate/SM_Crate_A07_Yellow_01_physics.usd",
}


# --------------------------------------------------------------------
# Config + result dataclasses
# --------------------------------------------------------------------
@dataclasses.dataclass
class EvalConfig:
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR
    scene_usd: str = DEFAULT_USD_PATH
    output_dir: str = DEFAULT_OUTPUT_DIR
    num_steps: int = DEFAULT_NUM_STEPS
    seed: int = 42
    language_prompt: str = DEFAULT_LANG_PROMPT
    perturbation_config: Optional[Dict[str, Any]] = None
    probe_config: Optional[Dict[str, Any]] = None
    record_external_camera: bool = False
    record_recording_camera: bool = True


@dataclasses.dataclass
class EvalResult:
    success: bool = False
    progress_fraction: float = 0.0
    trajectory_smoothness: float = 0.0
    per_step_joint_positions: Optional[np.ndarray] = None
    per_step_observations: Optional[List[Any]] = None
    per_step_actions: Optional[np.ndarray] = None
    per_step_probe_outputs: Optional[Dict[str, List[Any]]] = None
    success_metric_details: Dict[str, Any] = dataclasses.field(default_factory=dict)
    runtime_seconds: float = 0.0
    num_steps_completed: int = 0
    seed: int = 0
    checkpoint_dir: str = ""
    scene_usd: str = ""


# --------------------------------------------------------------------
# EvalSim
# --------------------------------------------------------------------
class EvalSim:
    """Encapsulates a single rollout. Use as a context manager-ish object:

        sim = EvalSim(config)
        sim.setup()
        result = sim.run()
        sim.close()
    """

    def __init__(self, config: EvalConfig):
        self.config = config
        self._seeded = False
        self._sim_app = None       # SimulationApp
        self._world = None         # omni.isaac.core.World
        self._franka = None        # Articulation
        self._external_cam = None  # Camera (policy)
        self._recording_cam = None # Camera (HD video)
        self._policy = None
        self._stage = None
        self._video_writer = None
        self._video_writer_ext = None
        self._cv2 = None           # imported lazily
        self._ArticulationAction = None

    # ----- public API -----
    def setup(self) -> None:
        self._seed_everything()
        self._start_simulation_app()
        self._patch_sys_path_and_typing_extensions()
        self._load_policy()
        self._open_scene_and_patch_payloads()
        self._maybe_apply_perturbations()
        self._init_world_and_robot()
        self._init_cameras()

    def close(self) -> None:
        try:
            if self._video_writer is not None:
                self._video_writer.release()
            if self._video_writer_ext is not None:
                self._video_writer_ext.release()
        except Exception:
            traceback.print_exc()
        try:
            if self._sim_app is not None:
                self._sim_app.close()
        except Exception:
            traceback.print_exc()

    def step(self, observation):
        """One inference step. Returns the np.ndarray action."""
        if self._policy is None:
            raise RuntimeError("EvalSim.step() called before setup().")
        import torch  # noqa: lazy

        with torch.no_grad():
            result = self._policy.infer(observation)
        actions = np.asarray(result["actions"])
        return actions

    def run(self) -> EvalResult:
        """Main 50 Hz loop. Mirrors legacy script step-for-step."""
        if self._policy is None:
            raise RuntimeError("EvalSim.run() called before setup().")

        cfg = self.config
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        results_csv_path = out_dir / "results.csv"
        video_path       = out_dir / "evaluation.mp4"
        video_ext_path   = out_dir / "evaluation_external.mp4"

        import cv2  # noqa: lazy
        from omni.isaac.core.utils.types import ArticulationAction  # noqa: lazy
        self._cv2 = cv2
        self._ArticulationAction = ArticulationAction

        csv_file = open(results_csv_path, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["step", "infer_ms"] + [f"j{i}" for i in range(9)])

        if cfg.record_recording_camera:
            self._video_writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                50,
                HD_VIDEO_RES,
            )
        if cfg.record_external_camera:
            self._video_writer_ext = cv2.VideoWriter(
                str(video_ext_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                50,
                POLICY_CAM_RES,
            )

        # Per-step buffers
        joint_pos_log: List[np.ndarray] = []
        action_log: List[np.ndarray] = []
        probe_log: Dict[str, List[Any]] = {}

        # Probe attach (after policy load, before loop)
        if cfg.probe_config is not None:
            from . import probes  # deferred
            probes.attach(self._policy)

        last_action_chunk: Optional[np.ndarray] = None
        chunk_idx = 0
        step = 0
        t_start = time.time()
        num_completed = 0

        try:
            # Mirror legacy: extra reset + re-init right before the loop.
            self._world.reset()
            self._franka.initialize()
            for _ in range(20):
                self._world.step(render=True)

            print("[run] Starting evaluation...")
            for step in range(cfg.num_steps):
                # ---- LOOK ----
                policy_img = self._get_frame(self._external_cam, POLICY_CAM_RES)
                hd_img = (
                    self._get_frame(self._recording_cam, HD_VIDEO_RES)
                    if self._recording_cam is not None
                    else None
                )
                joint_pos = self._franka.get_joint_positions()
                if joint_pos is None:
                    joint_pos = np.zeros(9, dtype=np.float32)
                joint_pos = np.asarray(joint_pos, dtype=np.float32)

                # ---- THINK ----
                t0 = time.time()
                if last_action_chunk is None or chunk_idx >= len(last_action_chunk):
                    obs = self.build_observation(policy_img, joint_pos)
                    action_chunk = self.step(obs)
                    last_action_chunk = action_chunk
                    chunk_idx = 0

                    # Probe record on each new chunk (cheap and stable).
                    if cfg.probe_config is not None:
                        from . import probes  # deferred
                        out = probes.record(self._policy, obs, action_chunk, step)
                        if out is not None:
                            for k, v in out.items():
                                probe_log.setdefault(k, []).append(v)

                action = last_action_chunk[chunk_idx]
                chunk_idx += 1
                infer_ms = (time.time() - t0) * 1000

                # ---- ACT ----
                target_joints = action[:NUM_ARM_JOINTS]
                hand_action = action[NUM_ARM_JOINTS:]
                assert hand_action.shape[-1] >= 3, (
                    f"hand_action must have at least 3 dims (gripper command "
                    f"uses mean of first 3), got shape {hand_action.shape}"
                )
                gripper_cmd = float(np.mean(hand_action[:3]))

                finger_l, finger_r = self._to_gripper_positions(gripper_cmd)
                full_cmd = np.zeros(9, dtype=np.float32)
                full_cmd[:NUM_ARM_JOINTS] = target_joints
                full_cmd[7] = finger_l
                full_cmd[8] = finger_r
                self._franka.apply_action(
                    self._ArticulationAction(joint_positions=full_cmd)
                )

                # ---- STEP ----
                self._world.step(render=True)

                # ---- RECORD ----
                if self._video_writer is not None and hd_img is not None:
                    self._video_writer.write(
                        cv2.cvtColor(hd_img, cv2.COLOR_RGB2BGR)
                    )
                if self._video_writer_ext is not None:
                    self._video_writer_ext.write(
                        cv2.cvtColor(policy_img, cv2.COLOR_RGB2BGR)
                    )

                # ---- LOG ----
                writer.writerow(
                    [step, f"{infer_ms:.1f}"] + joint_pos.tolist()
                )
                joint_pos_log.append(joint_pos)
                action_log.append(np.asarray(action, dtype=np.float32))

                if step % 50 == 0:
                    print(
                        f"[run] step {step:4d} | infer {infer_ms:5.1f}ms | "
                        f"j0-j2 {joint_pos[:3].round(2)}"
                    )

                num_completed = step + 1

        except Exception as e:
            print(f"[FATAL] Crashed at step {step}: {e}")
            traceback.print_exc()
        finally:
            csv_file.close()
            if self._video_writer is not None:
                self._video_writer.release()
                self._video_writer = None
                print(f"[exit] HD video saved to {video_path}")
            if self._video_writer_ext is not None:
                self._video_writer_ext.release()
                self._video_writer_ext = None
                print(f"[exit] External-cam video saved to {video_ext_path}")

        runtime = time.time() - t_start
        joint_pos_arr = (
            np.stack(joint_pos_log, axis=0) if joint_pos_log else np.zeros((0, 9))
        )
        action_arr = (
            np.stack(action_log, axis=0) if action_log else np.zeros((0, 1))
        )

        # Compute metrics here so the result is self-contained. Object
        # poses are not yet tracked, so success uses the heuristic
        # fallback. The metrics module documents this.
        from . import metrics
        smoothness = metrics.compute_trajectory_smoothness(joint_pos_arr)
        progress = metrics.compute_progress_fraction(joint_pos_arr)
        success = metrics.compute_success_heuristic(joint_pos_arr)

        return EvalResult(
            success=success["success"],
            progress_fraction=progress,
            trajectory_smoothness=smoothness,
            per_step_joint_positions=joint_pos_arr,
            per_step_observations=None,
            per_step_actions=action_arr,
            per_step_probe_outputs=probe_log if probe_log else None,
            success_metric_details=success["details"],
            runtime_seconds=runtime,
            num_steps_completed=num_completed,
            seed=self.config.seed,
            checkpoint_dir=self.config.checkpoint_dir,
            scene_usd=self.config.scene_usd,
        )

    # ----- observation construction -----
    def build_observation(self, ext_img_uint8: np.ndarray, joint_pos: np.ndarray) -> dict:
        # TODO(team): the policy expects a 24-dim state (7 arm + 17 hand)
        # — currently we only fill the arm joints and zero-pad the hand.
        # Open question: should the hand joints from the FR3 sim (last 2
        # finger DOFs + zero-padded fingers) be projected into the 17-dim
        # hand space, or kept as zeros? Until resolved this matches the
        # legacy script's behavior verbatim.
        if not getattr(self, "_warned_state_padding", False):
            print(
                "[obs] WARNING: state vector zero-pads hand joints "
                "(arm only). See TODO in core.py build_observation."
            )
            self._warned_state_padding = True

        state = np.zeros(24, dtype=np.float32)
        state[:7] = joint_pos[:7]
        return {
            "observation/image": ext_img_uint8,
            "observation/state": state,
            "prompt": self.config.language_prompt,
        }

    # ----- internals -----
    def _seed_everything(self) -> None:
        if self._seeded:
            return
        s = int(self.config.seed)
        random.seed(s)
        np.random.seed(s)
        os.environ.setdefault("PYTHONHASHSEED", str(s))
        # JAX is loaded lazily by openpi. Seed via env vars where possible
        # and re-seed any visible jax.random key at policy-load time.
        os.environ.setdefault("JAX_DEFAULT_PRNG_IMPL", "threefry2x32")
        try:
            import jax  # noqa
            # Touch the key so any subsequent jax.random.PRNGKey users get
            # a deterministic starting point.
            _ = jax.random.PRNGKey(s)
        except Exception:
            pass
        self._seeded = True
        print(f"[seed] Seeded random/numpy/jax with seed={s}")

    def _start_simulation_app(self) -> None:
        # Must be the first Isaac-related import.
        from isaacsim import SimulationApp

        sim_config = {
            "headless": True,
            "livestream": 0,
            "width": HD_VIDEO_RES[0],
            "height": HD_VIDEO_RES[1],
        }
        self._sim_app = SimulationApp(sim_config)
        print("[init] SimulationApp started")

    def _patch_sys_path_and_typing_extensions(self) -> None:
        for p in (
            "/workspace/openpi/src",
            "/workspace/openpi/packages/openpi-client/src",
            "/isaac_packages",
        ):
            if p not in sys.path:
                sys.path.insert(0, p)

        # Flush stale typing_extensions that Isaac Sim cached at startup,
        # then load the version from /isaac_packages.
        for mod_name in list(sys.modules):
            if mod_name == "typing_extensions" or mod_name.startswith("typing_extensions."):
                del sys.modules[mod_name]
        te_path = "/isaac_packages/typing_extensions.py"
        if os.path.exists(te_path):
            spec = _ilu.spec_from_file_location("typing_extensions", te_path)
            mod = _ilu.module_from_spec(spec)
            sys.modules["typing_extensions"] = mod
            spec.loader.exec_module(mod)
            print("[init] Reloaded typing_extensions from /isaac_packages")

    def _load_policy(self) -> None:
        import torch  # noqa
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[init] device = {device}")

        from openpi.policies import policy_config
        from openpi.shared import normalize
        from openpi.training import config as _config

        cfg = _config.get_config("pi05_egoverse")
        cfg = dataclasses.replace(cfg, assets_base_dir="/workspace/openpi/assets")
        data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
        norm_stats = normalize.load(cfg.assets_dirs / data_cfg.repo_id)

        self._policy = policy_config.create_trained_policy(
            cfg,
            self.config.checkpoint_dir,
            norm_stats=norm_stats,
            default_prompt=self.config.language_prompt,
        )
        print(f"[init] Loaded pi0.5 from {self.config.checkpoint_dir}")

    def _open_scene_and_patch_payloads(self) -> None:
        from omni.isaac.core.utils.stage import open_stage
        import omni.usd

        print(f"[init] Opening stage {self.config.scene_usd}")
        open_stage(usd_path=self.config.scene_usd)
        self._stage = omni.usd.get_context().get_stage()

        for prim_path, local_usd in _PAYLOAD_PATCHES.items():
            prim = self._stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                prim.GetPayloads().ClearPayloads()
                prim.GetPayloads().AddPayload(local_usd)
                print(f"[init] Patched {prim_path} -> {local_usd}")
            else:
                print(f"[WARN] {prim_path} not found in stage — skipping patch")

    def _maybe_apply_perturbations(self) -> None:
        if self.config.perturbation_config is None:
            return
        # Deferred import: do not touch perturbations at all on parity runs.
        from . import perturbations
        print("[init] Applying perturbations...")
        perturbations.apply(self._stage, self.config.perturbation_config)

    def _init_world_and_robot(self) -> None:
        from omni.isaac.core import World
        from omni.isaac.core.articulations import Articulation

        self._world = World(stage_units_in_meters=1.0)
        self._world.reset()

        self._franka = Articulation(prim_path="/World/fr3", name="franka")
        self._franka.initialize()
        print(f"[init] Franka has {self._franka.num_dof} DOF")

    def _init_cameras(self) -> None:
        from omni.isaac.sensor import Camera

        self._external_cam = Camera(
            prim_path="/World/ExternalCamera", resolution=POLICY_CAM_RES
        )
        self._external_cam.initialize()

        self._recording_cam = Camera(
            prim_path="/World/RecordingCamera", resolution=HD_VIDEO_RES
        )
        self._recording_cam.initialize()

        print("[init] Warming up cameras...")
        for _ in range(20):
            self._world.step(render=True)
        print(
            "[init] Cameras ready: ExternalCamera (policy, 224x224), "
            "RecordingCamera (HD, 1280x720)"
        )

    @staticmethod
    def _get_frame(cam, expected_res):
        if cam is None:
            return np.zeros((expected_res[1], expected_res[0], 3), dtype=np.uint8)
        rgba = cam.get_rgba()
        if rgba is None or rgba.size == 0:
            return np.zeros((expected_res[1], expected_res[0], 3), dtype=np.uint8)
        return rgba[:, :, :3]

    @staticmethod
    def _to_gripper_positions(gripper_cmd: float):
        gripper_cmd = float(np.clip(gripper_cmd, 0.0, 1.0))
        finger_pos = 0.04 * (1.0 - gripper_cmd)
        return finger_pos, finger_pos
