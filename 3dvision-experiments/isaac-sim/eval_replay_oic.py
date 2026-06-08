"""
eval_replay_oic.py — GT replay of an oic_human demo to PIN the Euler-order convention.

The oic action is [x,y,z, e1,e2,e3] (3 pos + 3 Euler angles). Position is order-independent,
but the ORIENTATION depends on the Euler order (XYZ vs ZYX, intrinsic vs extrinsic). IK
success alone can't disambiguate (we learned this with the quaternion) — so this script:

  1. MONTAGE: IK one representative demo frame under ALL 12 Tait-Bryan orders in one boot,
     render each, and tile into euler_order_montage.png. Compare each panel's gripper
     orientation against the demo's hand frames (oic_demo_ref_*.png from extract_oic_demo.py)
     and pick the order that matches.
  2. REPLAY: a full video under EULER_ORDER (env, default XYZ) for the chosen order.

Reads /workspace/oic_demo.npz (actions6 (N,6)). Outputs into /workspace/results/oic_replay/.
Tunables (env): EULER_ORDER (replay video order), REP_FRAME (montage frame, default 3N/4).
"""
from isaacsim import SimulationApp

CONFIG = {"headless": True, "livestream": 0, "width": 1280, "height": 720}
simulation_app = SimulationApp(CONFIG)

import os
import sys
import math
import traceback
import numpy as np
import cv2

from omni.isaac.core import World
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.sensor import Camera

sys.path.insert(0, "/workspace")
import oic_kinematics

USD_PATH   = "/workspace/kitchen_scene_1.usd"
DEMO_NPZ   = "/workspace/oic_demo.npz"
RESULTS    = f"/workspace/results/oic_replay"
os.makedirs(RESULTS, exist_ok=True)
HD = (1280, 720)
EE_FRAME     = os.environ.get("EE_FRAME") or "panda_hand"
EULER_ORDER  = os.environ.get("EULER_ORDER") or "XYZ"
POS_MAP      = os.environ.get("OIC_POS_MAP") or "x,y,z"   # base<->model position remap (frame sweep)
# All 12 Tait-Bryan orders (3 distinct axes): UPPER=intrinsic, lower=extrinsic (scipy).
ORDERS = ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX",
          "xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]

_d = np.load(DEMO_NPZ)
demo = np.asarray(_d["actions6"], dtype=np.float64)   # (N,6)
N = len(demo)
REP = int(os.environ.get("REP_FRAME") or (3 * N // 4))
REP = max(0, min(N - 1, REP))
print(f"[init] demo {demo.shape} | rep frame {REP} pose6={np.round(demo[REP],3).tolist()}")

# --------------------------------------------------------------------
# SCENE (object_in_container: sphere + bowl + wooden table, egocentric cam)
# --------------------------------------------------------------------
open_stage(usd_path=USD_PATH)
import omni.usd
from pxr import UsdGeom, Gf, UsdPhysics, UsdShade, Sdf

_stage = omni.usd.get_context().get_stage()
for path, local in {
    "/World/fr3": "/workspace/assets/fr3_full/fr3.usd",
    "/World/SM_HeavyDutyPackingTable_C02_01": "/workspace/assets/table/SM_HeavyDutyPackingTable_C02_01.usd",
}.items():
    p = _stage.GetPrimAtPath(path)
    if p.IsValid():
        p.GetPayloads().ClearPayloads(); p.GetPayloads().AddPayload(local)
for old in ("/World/plate_small", "/World/SM_Crate_A07_Yellow_01_physics"):
    pp = _stage.GetPrimAtPath(old)
    if pp.IsValid():
        pp.SetActive(False)

_OBJECT_POS = (0.527, -0.405, 1.85)
_BOWL_POS   = (1.463, -0.020, 1.807)


def _add_sphere(stage, path, pos, r=0.04):
    s = UsdGeom.Sphere.Define(stage, path); s.CreateRadiusAttr(r)
    UsdGeom.Xformable(s).AddTranslateOp().Set(Gf.Vec3d(*pos))
    pr = s.GetPrim(); UsdPhysics.CollisionAPI.Apply(pr)
    s.CreateDisplayColorAttr([Gf.Vec3f(0.80, 0.15, 0.15)])


def _add_bowl(stage, path, pos, Rb=0.10, Rt=0.18, H=0.13, wall=0.025, n=32):
    pts = []
    def ring(rad, z):
        b = len(pts)
        for j in range(n):
            a = 2 * math.pi * j / n; pts.append(Gf.Vec3f(rad * math.cos(a), rad * math.sin(a), z))
        return b
    ob = ring(Rb, 0.0); ot = ring(Rt, H); it = ring(Rt - wall, H); ib = ring(Rb - wall, wall)
    ic = len(pts); pts.append(Gf.Vec3f(0, 0, wall))
    counts, idx = [], []
    for j in range(n):
        k = (j + 1) % n
        for qa, qb, qc, qd in [(ob+j, ob+k, ot+k, ot+j), (ot+j, ot+k, it+k, it+j), (it+j, it+k, ib+k, ib+j)]:
            counts.append(4); idx.extend([qa, qb, qc, qd])
        counts.append(3); idx.extend([ic, ib + k, ib + j])
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(pts); m.CreateFaceVertexCountsAttr(counts); m.CreateFaceVertexIndicesAttr(idx)
    m.CreateDoubleSidedAttr(True); UsdGeom.Xformable(m).AddTranslateOp().Set(Gf.Vec3d(*pos))
    m.CreateDisplayColorAttr([Gf.Vec3f(0.45, 0.22, 0.55)])


_add_sphere(_stage, "/World/object", _OBJECT_POS)
_add_bowl(_stage, "/World/bowl", _BOWL_POS)

# egocentric camera (match eval_script_oic)
_CAM_POS = Gf.Vec3d(-0.2, -0.8, 2.6); _CAM_TARGET = Gf.Vec3d(0.8, -0.15, 1.82)
_cp = _stage.GetPrimAtPath("/World/ExternalCamera")
if _cp.IsValid():
    _xf = UsdGeom.Xformable(_cp); _xf.ClearXformOpOrder(); _xf.AddTranslateOp().Set(_CAM_POS)
    _look = (_CAM_TARGET - _CAM_POS).GetNormalized(); _q = Gf.Rotation(Gf.Vec3d(0, 0, -1), _look).GetQuat()
    _xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(_q.GetReal(), *_q.GetImaginary()))
    _c = UsdGeom.Camera(_cp); _ha = 36.0
    _c.CreateHorizontalApertureAttr(_ha); _c.CreateVerticalApertureAttr(_ha)
    _c.CreateFocalLengthAttr(_ha / (2 * math.tan(math.radians(90.0) / 2)))

# Re-aim the HD RecordingCamera at the workspace (its USD pose otherwise stares at empty sky).
import ego_view
ego_view.place_recording_camera(_stage)

# data is 30 Hz
world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 30.0, rendering_dt=1.0 / 30.0)
world.reset()
franka = Articulation(prim_path="/World/fr3", name="franka"); franka.initialize()
cam = Camera(prim_path="/World/RecordingCamera", resolution=HD); cam.initialize()
for _ in range(20):
    world.step(render=True)


def frame():
    rgba = cam.get_rgba()
    if rgba is None or rgba.size == 0:
        return np.zeros((HD[1], HD[0], 3), np.uint8)
    return rgba[:, :, :3]


def pose_arm(joints, settle=18):
    cmd = np.zeros(9, dtype=np.float32)
    cmd[:7] = joints
    cmd[7] = cmd[8] = 0.04   # open gripper (oic has no hand dims)
    for _ in range(settle):
        franka.apply_action(ArticulationAction(joint_positions=cmd)); world.step(render=True)


# --------------------------------------------------------------------
# 1) EULER-ORDER MONTAGE on the representative frame
# --------------------------------------------------------------------
try:
    world.reset(); franka.initialize()
    for _ in range(20):
        world.step(render=True)

    print(f"[montage] rendering rep frame {REP} under {len(ORDERS)} euler orders...")
    panels = []
    for order in ORDERS:
        try:
            k = oic_kinematics.OicKinematics(ee_frame=EE_FRAME, euler_order=order, pos_map=POS_MAP)
            j, ok = k.ik6(demo[REP], None)
            label = f"{order}  ik_ok={bool(ok)}"
            if ok and j is not None:
                pose_arm(j)
            img = frame().copy()
            if not ok or j is None:
                cv2.putText(img, "IK FAIL", (40, HD[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)
        except Exception as se:
            img = np.zeros((HD[1], HD[0], 3), np.uint8); label = f"{order} ERROR"
            print(f"[montage]   {order}: {se}")
        cv2.putText(img, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (60, 255, 60), 4)
        panels.append(cv2.resize(img, (480, 270)))
        print(f"[montage]   {label}")

    # tile 3 rows x 4 cols
    rows = [np.hstack(panels[r * 4:(r + 1) * 4]) for r in range(3)]
    montage = np.vstack(rows)
    mpath = f"{RESULTS}/euler_order_montage.png"
    cv2.imwrite(mpath, cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    print(f"[montage] saved {mpath}  (compare panels vs oic_demo_ref_*.png)")

    # --------------------------------------------------------------------
    # 2) FULL REPLAY video under EULER_ORDER
    # --------------------------------------------------------------------
    print(f"[replay] full replay under EULER_ORDER={EULER_ORDER}")
    kin = oic_kinematics.OicKinematics(ee_frame=EE_FRAME, euler_order=EULER_ORDER, pos_map=POS_MAP)
    vid = cv2.VideoWriter(f"{RESULTS}/replay_{EULER_ORDER}.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"), 30, HD)
    world.reset(); franka.initialize()
    for _ in range(20):
        world.step(render=True)
    warm = None; n_ok = 0
    for t in range(N):
        j, ok = kin.ik6(demo[t], warm)
        if ok and j is not None:
            warm = j; n_ok += 1
            cmd = np.zeros(9, dtype=np.float32); cmd[:7] = j; cmd[7] = cmd[8] = 0.04
            franka.apply_action(ArticulationAction(joint_positions=cmd))
        world.step(render=True)
        vid.write(cv2.cvtColor(frame(), cv2.COLOR_RGB2BGR))
    vid.release()
    print(f"[replay] IK ok {n_ok}/{N} ({100*n_ok/N:.0f}%) -> {RESULTS}/replay_{EULER_ORDER}.mp4")

except Exception as e:
    print(f"[FATAL] {e}"); traceback.print_exc()
finally:
    simulation_app.close(); print("[exit] done")
