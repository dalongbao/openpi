"""
eval_replay_demo.py — ground-truth action replay (NO policy).

Feeds a recorded demo's actions directly to the FR3 in Isaac Sim to sanity-check
the action pipeline (joint convention, scale, sim physics) independent of the
policy AND the visual gap. If the arm reproduces the demonstrated motion, the
plumbing is correct; if not, there's a deeper action-mapping bug.

Reads /workspace/demo_actions.npz (pre-extracted from an h5 demo on the login
node, where h5py is available) containing:
  actions_arm (N,7), actions_hand (N,17), qpos_arm (N,7)  -- all training convention.
Outputs /workspace/evaluation_replay.mp4 + /workspace/results_replay.csv.

No openpi / checkpoint / isaac_packages needed — just Isaac core + numpy + cv2.
"""

from isaacsim import SimulationApp

CONFIG = {"headless": True, "livestream": 0, "width": 1280, "height": 720}
simulation_app = SimulationApp(CONFIG)

import csv
import math
import traceback
import numpy as np
import cv2

from omni.isaac.core import World
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.sensor import Camera

USD_PATH       = "/workspace/kitchen_scene_1.usd"
DEMO_NPZ       = "/workspace/demo_actions.npz"
RESULTS_CSV    = "/workspace/results_replay.csv"
VIDEO_PATH     = "/workspace/evaluation_replay.mp4"
NUM_ARM_JOINTS = 7
POLICY_CAM_RES = (224, 224)
HD_VIDEO_RES   = (1280, 720)

# Isaac Sim FR3 vs Egoverse training data: dims 3 and 5 are swapped (self-inverse).
_TRAIN_TO_SIM = [0, 1, 2, 5, 4, 3, 6]

# --------------------------------------------------------------------
# LOAD THE DEMO
# --------------------------------------------------------------------
_demo = np.load(DEMO_NPZ)
demo_arm  = np.asarray(_demo["actions_arm"], dtype=np.float32)    # (N,7) train convention
demo_hand = np.asarray(_demo["actions_hand"], dtype=np.float32)   # (N,17)
demo_qpos = np.asarray(_demo["qpos_arm"], dtype=np.float32)       # (N,7)
N = len(demo_arm)
print(f"[init] Loaded demo: {N} frames | arm {demo_arm.shape} | hand {demo_hand.shape}")

# --------------------------------------------------------------------
# SCENE (same matched scene as the eval: sphere object, purple bowl, wooden table)
# --------------------------------------------------------------------
print(f"[init] Opening stage {USD_PATH}")
open_stage(usd_path=USD_PATH)

import omni.usd
from pxr import UsdGeom, Gf, UsdPhysics, UsdShade, Sdf

_stage = omni.usd.get_context().get_stage()

_PAYLOAD_PATCHES = {
    "/World/fr3":                             "/workspace/assets/fr3_full/fr3.usd",
    "/World/SM_HeavyDutyPackingTable_C02_01": "/workspace/assets/table/SM_HeavyDutyPackingTable_C02_01.usd",
}
for prim_path, local_usd in _PAYLOAD_PATCHES.items():
    prim = _stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        prim.GetPayloads().ClearPayloads()
        prim.GetPayloads().AddPayload(local_usd)
        print(f"[init] Patched {prim_path}")

for _old in ("/World/plate_small", "/World/SM_Crate_A07_Yellow_01_physics"):
    _p = _stage.GetPrimAtPath(_old)
    if _p.IsValid():
        _p.SetActive(False)

_OBJECT_POS = (0.527, -0.405, 1.847)
_BOWL_POS   = (1.463, -0.020, 1.807)


def _add_sphere_object(stage, path, pos, radius=0.04):
    sph = UsdGeom.Sphere.Define(stage, path)
    sph.CreateRadiusAttr(radius)
    UsdGeom.Xformable(sph).AddTranslateOp().Set(Gf.Vec3d(*pos))
    p = sph.GetPrim()
    UsdPhysics.CollisionAPI.Apply(p)
    UsdPhysics.RigidBodyAPI.Apply(p)
    UsdPhysics.MassAPI.Apply(p).CreateMassAttr(0.03)
    sph.CreateDisplayColorAttr([Gf.Vec3f(0.80, 0.15, 0.15)])


def _build_bowl_mesh(Rb=0.10, Rt=0.18, H=0.13, wall=0.025, n=32):
    pts = []

    def ring(r, z):
        base = len(pts)
        for j in range(n):
            a = 2.0 * math.pi * j / n
            pts.append(Gf.Vec3f(r * math.cos(a), r * math.sin(a), z))
        return base

    ob = ring(Rb, 0.0); ot = ring(Rt, H); it = ring(Rt - wall, H); ib = ring(Rb - wall, wall)
    oc = len(pts); pts.append(Gf.Vec3f(0, 0, 0.0))
    ic = len(pts); pts.append(Gf.Vec3f(0, 0, wall))
    counts, idx = [], []

    def quad(a, b, c, d):
        counts.append(4); idx.extend([a, b, c, d])

    def tri(a, b, c):
        counts.append(3); idx.extend([a, b, c])

    for j in range(n):
        k = (j + 1) % n
        quad(ob + j, ob + k, ot + k, ot + j)
        quad(ot + j, ot + k, it + k, it + j)
        quad(it + j, it + k, ib + k, ib + j)
        tri(ic, ib + k, ib + j)
        tri(oc, ob + j, ob + k)
    return pts, counts, idx


def _add_bowl(stage, path, pos):
    m = UsdGeom.Mesh.Define(stage, path)
    pts, counts, idx = _build_bowl_mesh()
    m.CreatePointsAttr(pts)
    m.CreateFaceVertexCountsAttr(counts)
    m.CreateFaceVertexIndicesAttr(idx)
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateDoubleSidedAttr(True)
    UsdGeom.Xformable(m).AddTranslateOp().Set(Gf.Vec3d(*pos))
    p = m.GetPrim()
    UsdPhysics.CollisionAPI.Apply(p)
    UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("none")
    m.CreateDisplayColorAttr([Gf.Vec3f(0.45, 0.22, 0.55)])


def _make_table_wooden(stage, table_path="/World/SM_HeavyDutyPackingTable_C02_01"):
    tprim = stage.GetPrimAtPath(table_path)
    if not tprim.IsValid():
        return
    mat = UsdShade.Material.Define(stage, table_path + "/WoodenMat")
    sh = UsdShade.Shader.Define(stage, table_path + "/WoodenMat/PBR")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.40, 0.26, 0.13))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.75)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(tprim).Bind(
        mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)


_add_sphere_object(_stage, "/World/object", _OBJECT_POS)
_add_bowl(_stage, "/World/bowl", _BOWL_POS)
_make_table_wooden(_stage)

# Camera matched to Aria FoV + egocentric pose.
ARIA_HFOV_DEG = 76.0
_CAM_POS    = Gf.Vec3d(0.90, -0.70, 2.90)
_CAM_TARGET = Gf.Vec3d(0.98, -0.20, 1.81)
_cam_prim = _stage.GetPrimAtPath("/World/ExternalCamera")
if _cam_prim.IsValid():
    _xf = UsdGeom.Xformable(_cam_prim)
    _xf.ClearXformOpOrder()
    _xf.AddTranslateOp().Set(_CAM_POS)
    _look = (_CAM_TARGET - _CAM_POS).GetNormalized()
    _q = Gf.Rotation(Gf.Vec3d(0, 0, -1), _look).GetQuat()
    _xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(_q.GetReal(), *_q.GetImaginary()))
    _cam = UsdGeom.Camera(_cam_prim)
    _ha = 36.0
    _cam.CreateHorizontalApertureAttr(_ha)
    _cam.CreateVerticalApertureAttr(_ha)
    _cam.CreateFocalLengthAttr(_ha / (2.0 * math.tan(math.radians(ARIA_HFOV_DEG) / 2.0)))

world = World(stage_units_in_meters=1.0)
world.reset()
franka = Articulation(prim_path="/World/fr3", name="franka")
franka.initialize()
print(f"[init] Franka has {franka.num_dof} DOF")
recording_cam = Camera(prim_path="/World/RecordingCamera", resolution=HD_VIDEO_RES)
recording_cam.initialize()
external_cam = Camera(prim_path="/World/ExternalCamera", resolution=POLICY_CAM_RES)
external_cam.initialize()
for _ in range(20):
    world.step(render=True)


def get_frame(cam, res):
    rgba = cam.get_rgba()
    if rgba is None or rgba.size == 0:
        return np.zeros((res[1], res[0], 3), dtype=np.uint8)
    return rgba[:, :, :3]


cv2.imwrite("/workspace/policy_cam_init.png", cv2.cvtColor(get_frame(external_cam, POLICY_CAM_RES), cv2.COLOR_RGB2BGR))


def permute_to_sim(arm7):
    cmd = np.zeros(9, dtype=np.float32)
    for train_dim, sim_idx in enumerate(_TRAIN_TO_SIM):
        cmd[sim_idx] = arm7[train_dim]
    return cmd


def to_finger(gripper_cmd):
    return 0.04 * (1.0 - float(np.clip(gripper_cmd, 0.0, 1.0)))


# --------------------------------------------------------------------
# REPLAY LOOP
# --------------------------------------------------------------------
csv_file = open(RESULTS_CSV, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["step"] + [f"cmd{i}" for i in range(7)] + [f"jp{i}" for i in range(9)])
video_writer = cv2.VideoWriter(VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), 50, HD_VIDEO_RES)

step = 0
try:
    world.reset()
    franka.initialize()
    for _ in range(20):
        world.step(render=True)

    # Teleport the arm to the demo's first joint configuration so the replay starts in-place.
    start = permute_to_sim(demo_qpos[0])
    start[7] = start[8] = 0.04
    franka.set_joint_positions(start)
    for _ in range(20):
        world.step(render=True)
    print(f"[init] Set to demo start qpos (sim): {start[:7].round(3)}")

    print(f"[run] Replaying {N} demo frames (no policy)...")
    for step in range(N):
        cmd = permute_to_sim(demo_arm[step])
        cmd[7] = cmd[8] = to_finger(float(np.mean(demo_hand[step][:3])))
        franka.apply_action(ArticulationAction(joint_positions=cmd))
        world.step(render=True)

        video_writer.write(cv2.cvtColor(get_frame(recording_cam, HD_VIDEO_RES), cv2.COLOR_RGB2BGR))
        jp = franka.get_joint_positions()
        if jp is None:
            jp = np.zeros(9, dtype=np.float32)
        writer.writerow([step] + cmd[:7].round(4).tolist() + jp.round(4).tolist())
        if step % 50 == 0:
            print(f"[run] step {step:4d} | demo arm(train) {demo_arm[step].round(3)} | jp {jp[:7].round(3)}")

except Exception as e:
    print(f"[FATAL] Crashed at step {step}: {e}")
    traceback.print_exc()

finally:
    print("[exit] Closing...")
    csv_file.close()
    video_writer.release()
    print(f"[exit] Video saved to {VIDEO_PATH}")
    simulation_app.close()
    print("[exit] Done.")
