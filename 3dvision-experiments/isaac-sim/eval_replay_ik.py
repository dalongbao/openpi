"""
eval_replay_ik.py — replay the demo's end-effector POSES via inverse kinematics.

KEY FIX: actions_arm is a 7-D Cartesian end-effector pose [x, y, z, qw, qx, qy, qz]
(dims 3-6 are a unit quaternion), NOT joint angles. The earlier replay/eval fed it
straight into joint targets, which is a category error. Here we solve IK each step:
target EE pose -> 7 arm joint angles -> apply to the FR3.

Reads /workspace/demo_actions.npz (actions_arm (N,7), actions_hand (N,17), qpos_arm (N,7)).
Outputs /workspace/evaluation_replay_ik.mp4 + /workspace/results_replay_ik.csv.

TUNABLES (first run prints diagnostics to set these):
  EE_FRAME      - end-effector frame name in the Lula Franka descriptor
  QUAT_WXYZ     - True if action[3:7] is [qw,qx,qy,qz] (our data: dim3≈1 => qw-first)
  POSE_IN_BASE  - True if the action pose is in the robot base frame (Lula expects base frame)
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

# Lula IK (try deprecated then new namespace)
try:
    from omni.isaac.motion_generation import interface_config_loader
    from omni.isaac.motion_generation.lula import LulaKinematicsSolver
except Exception:
    from isaacsim.robot_motion.motion_generation import interface_config_loader
    from isaacsim.robot_motion.motion_generation.lula import LulaKinematicsSolver

USD_PATH    = "/workspace/kitchen_scene_1.usd"
DEMO_NPZ    = "/workspace/demo_actions.npz"
RESULTS_CSV = "/workspace/results_replay_ik.csv"
VIDEO_PATH  = "/workspace/evaluation_replay_ik.mp4"
POLICY_CAM_RES = (224, 224)
HD_VIDEO_RES   = (1280, 720)

# --- tunables (verify from the first run's diagnostics) ---
EE_FRAME     = "right_gripper"   # fallback candidates tried automatically if this is missing
QUAT_WXYZ    = True              # action[3:7] = [qw,qx,qy,qz]
POSE_IN_BASE = True              # action pose is in the FR3 base frame

_demo = np.load(DEMO_NPZ)
demo_arm  = np.asarray(_demo["actions_arm"], dtype=np.float64)    # (N,7) EE pose
demo_hand = np.asarray(_demo["actions_hand"], dtype=np.float64)   # (N,17)
N = len(demo_arm)
print(f"[init] Loaded demo: {N} frames | arm(pose) {demo_arm.shape} | hand {demo_hand.shape}")

# --------------------------------------------------------------------
# SCENE (matched: sphere object, purple bowl, wooden table)
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

_OBJECT_POS = (0.527, -0.405, 1.847)
_BOWL_POS   = (1.463, -0.020, 1.807)

def _add_sphere(stage, path, pos, radius=0.04):
    s = UsdGeom.Sphere.Define(stage, path); s.CreateRadiusAttr(radius)
    UsdGeom.Xformable(s).AddTranslateOp().Set(Gf.Vec3d(*pos))
    pr = s.GetPrim()
    UsdPhysics.CollisionAPI.Apply(pr); UsdPhysics.RigidBodyAPI.Apply(pr)
    UsdPhysics.MassAPI.Apply(pr).CreateMassAttr(0.03)
    s.CreateDisplayColorAttr([Gf.Vec3f(0.80, 0.15, 0.15)])

def _bowl_mesh(Rb=0.10, Rt=0.18, H=0.13, wall=0.025, n=32):
    pts=[]
    def ring(r,z):
        b=len(pts)
        for j in range(n):
            a=2*math.pi*j/n; pts.append(Gf.Vec3f(r*math.cos(a), r*math.sin(a), z))
        return b
    ob=ring(Rb,0.0); ot=ring(Rt,H); it=ring(Rt-wall,H); ib=ring(Rb-wall,wall)
    oc=len(pts); pts.append(Gf.Vec3f(0,0,0.0)); ic=len(pts); pts.append(Gf.Vec3f(0,0,wall))
    counts=[]; idx=[]
    def quad(a,b,c,d): counts.append(4); idx.extend([a,b,c,d])
    def tri(a,b,c): counts.append(3); idx.extend([a,b,c])
    for j in range(n):
        k=(j+1)%n
        quad(ob+j,ob+k,ot+k,ot+j); quad(ot+j,ot+k,it+k,it+j); quad(it+j,it+k,ib+k,ib+j)
        tri(ic,ib+k,ib+j); tri(oc,ob+j,ob+k)
    return pts,counts,idx

def _add_bowl(stage, path, pos):
    m=UsdGeom.Mesh.Define(stage, path); pts,counts,idx=_bowl_mesh()
    m.CreatePointsAttr(pts); m.CreateFaceVertexCountsAttr(counts); m.CreateFaceVertexIndicesAttr(idx)
    m.CreateSubdivisionSchemeAttr("none"); m.CreateDoubleSidedAttr(True)
    UsdGeom.Xformable(m).AddTranslateOp().Set(Gf.Vec3d(*pos))
    pr=m.GetPrim()
    UsdPhysics.CollisionAPI.Apply(pr); UsdPhysics.MeshCollisionAPI.Apply(pr).CreateApproximationAttr("none")
    m.CreateDisplayColorAttr([Gf.Vec3f(0.45, 0.22, 0.55)])

def _table_wooden(stage, tp="/World/SM_HeavyDutyPackingTable_C02_01"):
    tprim=stage.GetPrimAtPath(tp)
    if not tprim.IsValid(): return
    mat=UsdShade.Material.Define(stage, tp+"/WoodenMat"); sh=UsdShade.Shader.Define(stage, tp+"/WoodenMat/PBR")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.40,0.26,0.13))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.75)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(tprim).Bind(mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)

_add_sphere(_stage, "/World/object", _OBJECT_POS)
_add_bowl(_stage, "/World/bowl", _BOWL_POS)
_table_wooden(_stage)

# camera (matched)
_CAM_POS=Gf.Vec3d(0.90,-0.70,2.90); _CAM_TARGET=Gf.Vec3d(0.98,-0.20,1.81)
_cp=_stage.GetPrimAtPath("/World/ExternalCamera")
if _cp.IsValid():
    _xf=UsdGeom.Xformable(_cp); _xf.ClearXformOpOrder(); _xf.AddTranslateOp().Set(_CAM_POS)
    _look=(_CAM_TARGET-_CAM_POS).GetNormalized(); _q=Gf.Rotation(Gf.Vec3d(0,0,-1),_look).GetQuat()
    _xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(_q.GetReal(), *_q.GetImaginary()))
    _c=UsdGeom.Camera(_cp); _ha=36.0
    _c.CreateHorizontalApertureAttr(_ha); _c.CreateVerticalApertureAttr(_ha)
    _c.CreateFocalLengthAttr(_ha/(2*math.tan(math.radians(76.0)/2)))

world = World(stage_units_in_meters=1.0); world.reset()
franka = Articulation(prim_path="/World/fr3", name="franka"); franka.initialize()
print(f"[init] Franka has {franka.num_dof} DOF; joint names: {franka.dof_names}")
recording_cam = Camera(prim_path="/World/RecordingCamera", resolution=HD_VIDEO_RES); recording_cam.initialize()
external_cam  = Camera(prim_path="/World/ExternalCamera", resolution=POLICY_CAM_RES); external_cam.initialize()
for _ in range(20): world.step(render=True)

def get_frame(cam,res):
    rgba=cam.get_rgba()
    if rgba is None or rgba.size==0: return np.zeros((res[1],res[0],3),np.uint8)
    return rgba[:,:,:3]

cv2.imwrite("/workspace/policy_cam_init.png", cv2.cvtColor(get_frame(external_cam,POLICY_CAM_RES), cv2.COLOR_RGB2BGR))

# --------------------------------------------------------------------
# IK SOLVER
# --------------------------------------------------------------------
_cfg = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
solver = LulaKinematicsSolver(**_cfg)
_frames = list(solver.get_all_frame_names())
_jnames = list(solver.get_joint_names())
print(f"[ik] Lula joint names: {_jnames}")
print(f"[ik] Lula frame names: {_frames}")
if EE_FRAME not in _frames:
    for cand in ("right_gripper", "panda_hand", "panda_rightfinger", "tool0", "fr3_hand", "panda_link8"):
        if cand in _frames:
            EE_FRAME = cand; break
print(f"[ik] Using EE frame: {EE_FRAME}")


def solve_ik(pose7, warmstart):
    pos = np.asarray(pose7[:3], dtype=np.float64)
    q = np.asarray(pose7[3:7], dtype=np.float64)
    quat_wxyz = q if QUAT_WXYZ else np.array([q[3], q[0], q[1], q[2]])
    joints, ok = solver.compute_inverse_kinematics(EE_FRAME, pos, quat_wxyz, warmstart)
    return joints, bool(ok)


def to_finger(g):
    return 0.04 * (1.0 - float(np.clip(g, 0.0, 1.0)))


# --------------------------------------------------------------------
# REPLAY LOOP
# --------------------------------------------------------------------
csv_file = open(RESULTS_CSV, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["step", "ik_ok", "tx", "ty", "tz"] + [f"jp{i}" for i in range(9)])
video_writer = cv2.VideoWriter(VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), 50, HD_VIDEO_RES)

step = 0
warm = None
n_ok = 0
try:
    world.reset(); franka.initialize()
    for _ in range(20):
        world.step(render=True)

    # diagnostic: solve IK for the first pose and report
    j0, ok0 = solve_ik(demo_arm[0], None)
    print(f"[ik] first-frame IK success={ok0}  joints={None if j0 is None else np.round(j0,3)}")

    print(f"[run] Replaying {N} EE-pose frames via IK...")
    for step in range(N):
        joints, ok = solve_ik(demo_arm[step], warm)
        if ok and joints is not None:
            warm = joints
            n_ok += 1
            cmd = np.zeros(9, dtype=np.float32)
            cmd[:7] = joints
            cmd[7] = cmd[8] = to_finger(float(np.mean(demo_hand[step][:3])))
            franka.apply_action(ArticulationAction(joint_positions=cmd))
        # if IK fails, hold previous command (don't move)
        world.step(render=True)

        video_writer.write(cv2.cvtColor(get_frame(recording_cam, HD_VIDEO_RES), cv2.COLOR_RGB2BGR))
        jp = franka.get_joint_positions()
        if jp is None:
            jp = np.zeros(9, dtype=np.float32)
        t = demo_arm[step][:3]
        writer.writerow([step, int(ok), f"{t[0]:.3f}", f"{t[1]:.3f}", f"{t[2]:.3f}"] + jp.round(4).tolist())
        if step % 50 == 0:
            print(f"[run] step {step:4d} | ik_ok={ok} | target xyz {np.round(t,3)} | jp {jp[:7].round(3)}")

    print(f"[run] IK success on {n_ok}/{N} frames ({100*n_ok/N:.1f}%)")

except Exception as e:
    print(f"[FATAL] Crashed at step {step}: {e}")
    traceback.print_exc()

finally:
    print("[exit] Closing...")
    csv_file.close(); video_writer.release()
    print(f"[exit] Video saved to {VIDEO_PATH}")
    simulation_app.close(); print("[exit] Done.")
