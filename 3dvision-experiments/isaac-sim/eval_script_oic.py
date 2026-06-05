"""
eval_script_oic.py — Isaac eval for the oic_human (object-in-container) model.

DIFFERENT from eval_script_object_in_bowl.py: oic is single-arm CARTESIAN with a 6-dim
state/action = [x,y,z, e1,e2,e3] (3 position + 3 EULER angles, NO hand). So:
  observation state = FK(sim joints) -> [pos, euler]            (6-dim, via oic_kinematics)
  action execution  = IK([pos, euler]) -> joint targets          (6-dim)
The gripper has no policy signal here (no hand dims), so it's held at a fixed opening.

UNRESOLVED conventions (this script helps crack them):
  EULER_ORDER  scipy order string; UPPER=intrinsic, lower=extrinsic. Default XYZ.
               A startup SWEEP IKs the action-mean pose under several orders and prints
               reachability + posture so you can pick the winner (judge in the video).
  POSE frame   oic is human (Aria) data, so positions may not be in the FR3 base frame.
               If ALL orders give IK failures / wild joints, the frame is wrong, not the order.

Model/paths come from env (set by submit.sh's model preset, e.g. oic_human_2537ep):
  CONFIG_NAME CHECKPOINT_DIR NORM_STATS_DIR PROMPT MODEL_NAME RUN_TAG NUM_STEPS SCENE_FIDELITY
"""
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True, "livestream": 0, "width": 1280, "height": 720})

import dataclasses
import os
import sys
import csv
import math
import time
import traceback
import numpy as np
import cv2

from omni.isaac.core import World
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.sensor import Camera

# --------------------------------------------------------------------
# CONFIG (env-overridable; defaults target the oic_human model)
# --------------------------------------------------------------------
USD_PATH       = "/workspace/kitchen_scene_1.usd"
CONFIG_NAME    = os.environ.get("CONFIG_NAME") or "pi05_ego_human_oic"
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR") or "/checkpoints/pi05_ego_human_oic/human_oic/29999"
NORM_STATS_DIR = os.environ.get("NORM_STATS_DIR") or "/checkpoints/pi05_ego_human_oic/human_oic/29999/assets/egoverse/oic_human"
LANGUAGE_COMMAND = os.environ.get("PROMPT") or "put the object in the container"
MODEL_NAME     = os.environ.get("MODEL_NAME") or "oic_human_2537ep"
RUN_TAG        = os.environ.get("RUN_TAG") or ""
EULER_ORDER    = os.environ.get("EULER_ORDER") or "XYZ"
EE_FRAME       = os.environ.get("EE_FRAME") or "panda_hand"

NUM_STEPS      = int(os.environ.get("NUM_STEPS") or "3000")
NUM_ARM_JOINTS = 7
POLICY_CAM_RES = (224, 224)
HD_VIDEO_RES   = (1280, 720)
GRIPPER_OPEN   = 0.04   # fixed finger opening (no hand signal in oic)

RESULTS_DIR    = f"/workspace/results/{MODEL_NAME}"
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_CSV    = f"{RESULTS_DIR}/results_oic{RUN_TAG}.csv"
VIDEO_PATH     = f"{RESULTS_DIR}/evaluation_oic{RUN_TAG}.mp4"

# The action-mean pose from the oic norm stats — used as the home posture and the
# canonical pose for the euler-order sweep.
OIC_MEAN_POSE6 = np.array([0.174, 0.211, 0.469, -1.445, -0.531, -0.892], dtype=np.float64)

print(f"[init] OIC eval | model={MODEL_NAME} config={CONFIG_NAME} euler={EULER_ORDER}")
print(f"[init] outputs -> {RESULTS_DIR}/ (tag='{RUN_TAG}')  steps={NUM_STEPS}")

# --------------------------------------------------------------------
# LOAD POLICY
# --------------------------------------------------------------------
sys.path.insert(0, "/workspace/openpi/src")
sys.path.insert(0, "/workspace/openpi/packages/openpi-client/src")
sys.path.insert(0, "/isaac_packages")
sys.path.insert(0, "/workspace")

import importlib.util as _ilu
_te_spec = _ilu.spec_from_file_location("typing_extensions", "/isaac_packages/typing_extensions.py")
_te_mod  = _ilu.module_from_spec(_te_spec)
sys.modules["typing_extensions"] = _te_mod
_te_spec.loader.exec_module(_te_mod)
del _ilu, _te_spec, _te_mod

try:
    import pathlib as _pl
    from openpi.policies import policy_config
    from openpi.shared import normalize
    from openpi.training import config as _config

    cfg        = _config.get_config(CONFIG_NAME)
    cfg        = dataclasses.replace(cfg, assets_base_dir="/workspace/openpi/assets")
    data_cfg   = cfg.data.create(cfg.assets_dirs, cfg.model)
    _ns_dir    = _pl.Path(NORM_STATS_DIR) if NORM_STATS_DIR else (cfg.assets_dirs / data_cfg.repo_id)
    print(f"[init] norm stats from {_ns_dir}")
    norm_stats = normalize.load(_ns_dir)

    policy = policy_config.create_trained_policy(
        cfg, CHECKPOINT_DIR, norm_stats=norm_stats, default_prompt=LANGUAGE_COMMAND,
    )
    print(f"[init] Loaded {CONFIG_NAME} from {CHECKPOINT_DIR}")
except Exception as e:
    print(f"[FATAL] Could not load policy: {e}")
    traceback.print_exc()
    simulation_app.close()
    sys.exit(1)

# --------------------------------------------------------------------
# SCENE (same object/bowl/table/camera as the object_in_bowl eval)
# --------------------------------------------------------------------
print(f"[init] Opening stage {USD_PATH}")
open_stage(usd_path=USD_PATH)
import omni.usd
from pxr import UsdGeom, Gf, UsdPhysics, UsdShade, Sdf

_stage = omni.usd.get_context().get_stage()
for prim_path, local_usd in {
    "/World/fr3": "/workspace/assets/fr3_full/fr3.usd",
    "/World/SM_HeavyDutyPackingTable_C02_01": "/workspace/assets/table/SM_HeavyDutyPackingTable_C02_01.usd",
}.items():
    p = _stage.GetPrimAtPath(prim_path)
    if p.IsValid():
        p.GetPayloads().ClearPayloads(); p.GetPayloads().AddPayload(local_usd)
for _old in ("/World/plate_small", "/World/SM_Crate_A07_Yellow_01_physics"):
    _p = _stage.GetPrimAtPath(_old)
    if _p.IsValid():
        _p.SetActive(False)

_OBJECT_POS = (0.527, -0.405, 1.85)
_BOWL_POS   = (1.463, -0.020, 1.807)


def _add_sphere(stage, path, pos, radius=0.04):
    s = UsdGeom.Sphere.Define(stage, path); s.CreateRadiusAttr(radius)
    UsdGeom.Xformable(s).AddTranslateOp().Set(Gf.Vec3d(*pos))
    pr = s.GetPrim()
    UsdPhysics.CollisionAPI.Apply(pr); UsdPhysics.RigidBodyAPI.Apply(pr)
    UsdPhysics.MassAPI.Apply(pr).CreateMassAttr(0.03)
    s.CreateDisplayColorAttr([Gf.Vec3f(0.80, 0.15, 0.15)])


def _add_bowl(stage, path, pos, Rb=0.10, Rt=0.18, H=0.13, wall=0.025, n=32):
    pts = []

    def ring(r, z):
        b = len(pts)
        for j in range(n):
            a = 2 * math.pi * j / n
            pts.append(Gf.Vec3f(r * math.cos(a), r * math.sin(a), z))
        return b

    ob = ring(Rb, 0.0); ot = ring(Rt, H); it = ring(Rt - wall, H); ib = ring(Rb - wall, wall)
    oc = len(pts); pts.append(Gf.Vec3f(0, 0, 0.0)); ic = len(pts); pts.append(Gf.Vec3f(0, 0, wall))
    counts, idx = [], []

    def quad(a, b, c, d): counts.append(4); idx.extend([a, b, c, d])
    def tri(a, b, c): counts.append(3); idx.extend([a, b, c])

    for j in range(n):
        k = (j + 1) % n
        quad(ob + j, ob + k, ot + k, ot + j); quad(ot + j, ot + k, it + k, it + j)
        quad(it + j, it + k, ib + k, ib + j); tri(ic, ib + k, ib + j); tri(oc, ob + j, ob + k)
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(pts); m.CreateFaceVertexCountsAttr(counts); m.CreateFaceVertexIndicesAttr(idx)
    m.CreateSubdivisionSchemeAttr("none"); m.CreateDoubleSidedAttr(True)
    UsdGeom.Xformable(m).AddTranslateOp().Set(Gf.Vec3d(*pos))
    pr = m.GetPrim()
    UsdPhysics.CollisionAPI.Apply(pr); UsdPhysics.MeshCollisionAPI.Apply(pr).CreateApproximationAttr("none")
    m.CreateDisplayColorAttr([Gf.Vec3f(0.45, 0.22, 0.55)])


_add_sphere(_stage, "/World/object", _OBJECT_POS)
_add_bowl(_stage, "/World/bowl", _BOWL_POS)

# Optional scene realism (shared with the object_in_bowl eval).
if os.environ.get("SCENE_FIDELITY", "0").lower() in ("1", "true", "yes", "y"):
    import scene_fidelity
    scene_fidelity.apply_fidelity(_stage)

# Camera — same egocentric-ish placement as the object_in_bowl eval.
_CAM_POS    = Gf.Vec3d(-0.2, -0.8, 2.6)
_CAM_TARGET = Gf.Vec3d(0.8, -0.15, 1.82)
_cp = _stage.GetPrimAtPath("/World/ExternalCamera")
if _cp.IsValid():
    _xf = UsdGeom.Xformable(_cp); _xf.ClearXformOpOrder(); _xf.AddTranslateOp().Set(_CAM_POS)
    _look = (_CAM_TARGET - _CAM_POS).GetNormalized(); _q = Gf.Rotation(Gf.Vec3d(0, 0, -1), _look).GetQuat()
    _xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(_q.GetReal(), *_q.GetImaginary()))
    _c = UsdGeom.Camera(_cp); _ha = 36.0
    _c.CreateHorizontalApertureAttr(_ha); _c.CreateVerticalApertureAttr(_ha)
    _c.CreateFocalLengthAttr(_ha / (2 * math.tan(math.radians(90.0) / 2)))

world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 50.0, rendering_dt=1.0 / 50.0)
world.reset()
franka = Articulation(prim_path="/World/fr3", name="franka"); franka.initialize()
print(f"[init] Franka has {franka.num_dof} DOF")
external_cam  = Camera(prim_path="/World/ExternalCamera", resolution=POLICY_CAM_RES); external_cam.initialize()
recording_cam = Camera(prim_path="/World/RecordingCamera", resolution=HD_VIDEO_RES); recording_cam.initialize()
for _ in range(20):
    world.step(render=True)


def get_frame(cam, res):
    rgba = cam.get_rgba()
    if rgba is None or rgba.size == 0:
        return np.zeros((res[1], res[0], 3), np.uint8)
    return rgba[:, :, :3]


cv2.imwrite(f"{RESULTS_DIR}/policy_cam_init.png", cv2.cvtColor(get_frame(external_cam, POLICY_CAM_RES), cv2.COLOR_RGB2BGR))

# --------------------------------------------------------------------
# 6-DIM KINEMATICS + EULER-ORDER SWEEP
# --------------------------------------------------------------------
import oic_kinematics
kin = oic_kinematics.OicKinematics(ee_frame=EE_FRAME, euler_order=EULER_ORDER)

# Sweep: IK the action-mean pose under several euler orders. The right one is reachable
# AND yields a sensible (non-extreme) posture. If ALL fail, the position frame is wrong.
print(f"[sweep] IK of oic action-mean pose {np.round(OIC_MEAN_POSE6, 3)} under euler orders:")
for _order in ("XYZ", "ZYX", "xyz", "zyx", "ZYZ", "YXZ"):
    try:
        _k = oic_kinematics.OicKinematics(ee_frame=EE_FRAME, euler_order=_order)
        _j, _ok = _k.ik6(OIC_MEAN_POSE6, None)
        print(f"[sweep]   {_order:4s} ik_ok={bool(_ok)}  joints={None if _j is None else np.round(_j, 3)}")
    except Exception as _se:
        print(f"[sweep]   {_order:4s} ERROR {_se}")
print(f"[sweep] (set EULER_ORDER=<name> to run the policy with that order)")

# --------------------------------------------------------------------
# OBSERVATION + LOOP
# --------------------------------------------------------------------
def build_observation(img_uint8, joint_pos):
    state6 = kin.fk6(joint_pos[:7]).astype(np.float32)   # [pos, euler]
    return {
        "observation/image": img_uint8,
        "observation/state": state6,
        "prompt": LANGUAGE_COMMAND,
    }


csv_file = open(RESULTS_CSV, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["step", "infer_ms", "ik_ok", "tx", "ty", "tz", "e1", "e2", "e3"] + [f"j{i}" for i in range(9)])
video_writer = cv2.VideoWriter(VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), 50, HD_VIDEO_RES)

last_chunk = None
chunk_idx  = 0
step       = 0
_warm      = None

try:
    world.reset(); franka.initialize()
    for _ in range(20):
        world.step(render=True)

    # Home: IK the action-mean pose so the first observation is in-distribution.
    home_j, home_ok = kin.ik6(OIC_MEAN_POSE6, None)
    if home_ok and home_j is not None:
        _warm = np.asarray(home_j, dtype=np.float64)
        home_cmd = np.zeros(9, dtype=np.float32); home_cmd[:7] = home_j
        home_cmd[7] = home_cmd[8] = GRIPPER_OPEN
        for _ in range(100):
            franka.apply_action(ArticulationAction(joint_positions=home_cmd))
            world.step(render=True)
    print(f"[init] Home IK ok={home_ok}")

    print(f"[run] Starting oic eval for {NUM_STEPS} steps...")
    for step in range(NUM_STEPS):
        policy_img = get_frame(external_cam, POLICY_CAM_RES)
        hd_img     = get_frame(recording_cam, HD_VIDEO_RES)
        joint_pos  = franka.get_joint_positions()
        if joint_pos is None:
            joint_pos = np.zeros(9, dtype=np.float32)

        t0 = time.time()
        _requery = step < 200
        if last_chunk is None or chunk_idx >= len(last_chunk) or _requery:
            result = policy.infer(build_observation(policy_img, joint_pos))
            last_chunk = np.asarray(result["actions"])
            chunk_idx = 0
        action = last_chunk[chunk_idx]          # 6-dim [pos, euler]
        chunk_idx += 1
        infer_ms = (time.time() - t0) * 1000

        if step == 0:
            print(f"[diag] chunk shape={last_chunk.shape}  first 3 action poses:")
            for _ci in range(min(3, len(last_chunk))):
                print(f"  chunk[{_ci}] pose6={np.round(last_chunk[_ci][:6], 3)}")
        if step % 500 == 0 and step > 0:
            print(f"[diag] step {step} action pose6={np.round(action[:6], 3)}")

        arm_joints, ik_ok = kin.ik6(action[:6], _warm)
        if ik_ok and arm_joints is not None:
            _warm = np.asarray(arm_joints, dtype=np.float64)
        cmd = np.zeros(9, dtype=np.float32)
        cmd[:7] = _warm if _warm is not None else 0.0
        cmd[7] = cmd[8] = GRIPPER_OPEN
        franka.apply_action(ArticulationAction(joint_positions=cmd))

        world.step(render=True)
        video_writer.write(cv2.cvtColor(hd_img, cv2.COLOR_RGB2BGR))
        a = action
        writer.writerow([step, f"{infer_ms:.1f}", int(ik_ok)]
                        + [f"{a[i]:.3f}" for i in range(6)] + joint_pos.round(4).tolist())
        if step % 50 == 0:
            print(f"[run] step {step:4d} | infer {infer_ms:6.1f}ms | ik_ok={ik_ok} | "
                  f"pose6 {np.round(a[:6], 3)}")

except Exception as e:
    print(f"[FATAL] Crashed at step {step}: {e}")
    traceback.print_exc()

finally:
    print("[exit] Closing...")
    csv_file.close(); video_writer.release()
    print(f"[exit] Video saved to {VIDEO_PATH}")
    simulation_app.close(); print("[exit] Done.")
