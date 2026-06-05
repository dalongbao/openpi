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
import os
import traceback
import numpy as np
import cv2

from omni.isaac.core import World
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.sensor import Camera

# Shared FK/IK (imported after SimulationApp; lives at /workspace/ik_fk_helpers.py)
import sys
sys.path.insert(0, "/workspace")
import ik_fk_helpers

USD_PATH    = "/workspace/kitchen_scene_1.usd"
DEMO_NPZ    = "/workspace/demo_actions.npz"
RESULTS_CSV = "/workspace/results_replay_ik.csv"
VIDEO_PATH  = "/workspace/evaluation_replay_ik.mp4"
POLICY_CAM_RES = (224, 224)
HD_VIDEO_RES   = (1280, 720)

# --- tunables ---
# (1) FRAME and (2) QUAT ORDER and (3) ABS/DELTA confirmed by infer_ee_convention.py:
#     base frame, scalar-first wxyz, absolute targets. Only the control-point (EE_FRAME)
#     is unresolved by data — sweep it here. Override per-run with EE_FRAME=<name> env var
#     so you don't edit code between sweep runs, e.g.:
#       EE_FRAME=panda_link8 .../python.sh eval_replay_ik.py
# Franka gotcha: panda_hand/right_gripper is rotated -45° about z vs the panda_link8 flange,
# so the wrong frame shows up as a constant wrist twist (the bug we're chasing).
def _envbool(name, default):
    v = os.environ.get(name)
    return default if v is None or v == "" else v.lower() in ("1", "true", "yes", "y")

EE_FRAME     = os.environ.get("EE_FRAME", "right_gripper")  # candidates auto-tried if missing
# QUAT order is genuinely ambiguous from data (a grasp held ~pointing-down looks like a
# constant ~180°-about-x pose in xyzw, which mimics near-identity wxyz in sign-stability).
# Decided by the VIDEO: wxyz made the gripper face UP -> wrong; xyzw points it DOWN. Override
# per-run with QUAT_WXYZ=0/1.
QUAT_WXYZ    = _envbool("QUAT_WXYZ", False)   # action[3:7] = [qx,qy,qz,qw]  (xyzw, scalar-last)
POSE_IN_BASE = _envbool("POSE_IN_BASE", True) # action pose is in the FR3 base frame (CONFIRMED)

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

# Camera: approximate Aria egocentric view — positioned front-left, angled ~35° down.
# Robot appears upper-right, sphere+bowl fill the center of the 224x224 frame.
_CAM_POS    = Gf.Vec3d(-0.2, -0.8, 2.6)
_CAM_TARGET = Gf.Vec3d(0.8, -0.15, 1.82)
_cp=_stage.GetPrimAtPath("/World/ExternalCamera")
if _cp.IsValid():
    _xf=UsdGeom.Xformable(_cp); _xf.ClearXformOpOrder(); _xf.AddTranslateOp().Set(_CAM_POS)
    _look=(_CAM_TARGET-_CAM_POS).GetNormalized(); _q=Gf.Rotation(Gf.Vec3d(0,0,-1),_look).GetQuat()
    _xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(_q.GetReal(), *_q.GetImaginary()))
    _c=UsdGeom.Camera(_cp); _ha=36.0
    _c.CreateHorizontalApertureAttr(_ha); _c.CreateVerticalApertureAttr(_ha)
    _c.CreateFocalLengthAttr(_ha/(2*math.tan(math.radians(90.0)/2)))

# Data is 50 Hz; step physics at 1/50 s so one world.step() == one demo tick == one 50fps
# video frame (real time). Default would be 1/60, desyncing data/physics/video.
world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 50.0, rendering_dt=1.0 / 50.0)
world.reset()
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
# IK SOLVER (shared with the policy eval via ik_fk_helpers)
# --------------------------------------------------------------------
kin = ik_fk_helpers.FrankaKinematics(ee_frame=EE_FRAME, quat_wxyz=QUAT_WXYZ)
solver = kin.solver          # for the frame-sweep diagnostic below
_frames = kin.frames
EE_FRAME = kin.ee_frame
solve_ik = kin.ik            # (pose7, warmstart) -> (joints, ok), stored quat convention


# Hand: demos use a 17-DOF ORCA hand; the sim FR3 has a 2-finger gripper. Coarse proxy:
# overall hand flexion -> finger width, normalized to THIS episode's own range so it
# actually modulates. HAND_INVERT=1 flips it if open/closed comes out reversed.
_hand_sig = demo_hand.mean(axis=1)                       # (N,) overall flexion proxy
_hlo, _hhi = np.percentile(_hand_sig, 5), np.percentile(_hand_sig, 95)
_HAND_INVERT = _envbool("HAND_INVERT", False)

def closure_frac(step):
    c = float(np.clip((_hand_sig[step] - _hlo) / (_hhi - _hlo + 1e-9), 0.0, 1.0))
    return 1.0 - c if _HAND_INVERT else c

def to_finger(frac):
    return 0.04 * (1.0 - float(frac))                    # frac=1 (closed) -> 0 m width


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

    # FRAME SWEEP DIAGNOSTIC: try every plausible control-point on the first pose in ONE boot.
    # Report IK success + posture so you can compare frames without N full renders.
    # The right frame is the one whose replayed gripper matches the real demo (judge in the mp4);
    # this table just tells you which frames are even viable and how different their postures are.
    _sweep = [f for f in ("panda_link8", "fr3_link8", "panda_hand", "fr3_hand",
                          "right_gripper", "panda_rightfinger", "tool0") if f in _frames]
    print(f"[sweep] Testing {len(_sweep)} candidate EE frames on demo_arm[0] "
          f"(pos={np.round(demo_arm[0][:3],3)}, quat_wxyz={np.round(demo_arm[0][3:7],3)}):")
    _p0 = np.asarray(demo_arm[0][:3], np.float64)
    _q0 = np.asarray(demo_arm[0][3:7], np.float64)
    _q0 = _q0 if QUAT_WXYZ else np.array([_q0[3], _q0[0], _q0[1], _q0[2]])
    for _f in _sweep:
        _j, _ok = solver.compute_inverse_kinematics(_f, _p0, _q0, None)
        print(f"[sweep]   {_f:18s} ik_ok={bool(_ok)}  "
              f"joints={None if _j is None else np.round(_j,3)}")
    print(f"[sweep] (set EE_FRAME=<name> to render the full replay for the winner)")

    print(f"[run] Replaying {N} EE-pose frames via IK...")
    for step in range(N):
        joints, ok = solve_ik(demo_arm[step], warm)
        if ok and joints is not None:
            warm = joints
            n_ok += 1
            cmd = np.zeros(9, dtype=np.float32)
            cmd[:7] = joints
            cmd[7] = cmd[8] = to_finger(closure_frac(step))
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
