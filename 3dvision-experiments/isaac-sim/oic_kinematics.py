"""
oic_kinematics.py — 6-dim EE-pose convention for the oic_human (object-in-container)
model. Its state/action is [x, y, z, e1, e2, e3]: 3 position (metres) + 3 EULER angles
(radians), with NO hand dims (single-arm cartesian).

Why Euler (not a rotation vector): in the oic norm stats the middle angle (dim 4) is
clamped to ~[-pi/2, pi/2] while dims 3 and 5 span ~[-pi, pi] — the classic Euler gimbal
signature. The exact ORDER (XYZ vs ZYX, intrinsic vs extrinsic) is NOT resolvable from
the stats, so it's a tunable: set EULER_ORDER and sweep it in sim (uppercase = intrinsic,
lowercase = extrinsic, per scipy convention).

Wraps the shared Lula solver in ik_fk_helpers (import after SimulationApp(...)):
  fk6(joints) -> [pos, euler]   builds the 6-dim OBSERVATION state
  ik6(pose6)  -> (joints, ok)   executes the 6-dim ACTION
"""
import numpy as np

import ik_fk_helpers

try:
    from scipy.spatial.transform import Rotation as _R
except Exception as _e:  # pragma: no cover
    raise ImportError(
        "oic_kinematics needs scipy. If missing in the container, install into "
        "/isaac_packages: /isaac-sim/python.sh -m pip install --target /isaac_packages scipy"
    ) from _e


def _wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def _xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


class OicKinematics:
    """6-dim [pos, euler] convention over the shared Franka Lula solver."""

    def __init__(self, ee_frame="panda_hand", euler_order="XYZ"):
        # Drive the underlying solver in wxyz so we control the quat<->euler step here.
        self.kin = ik_fk_helpers.FrankaKinematics(ee_frame=ee_frame, quat_wxyz=True)
        self.euler_order = euler_order
        self.ee_frame = self.kin.ee_frame
        self.solver = self.kin.solver
        self.frames = self.kin.frames
        print(f"[oic-kin] euler_order={euler_order}  ee_frame={self.ee_frame}")

    def fk6(self, arm_joints):
        """7 arm joints -> [x,y,z, e1,e2,e3] (euler in self.euler_order)."""
        pose7 = self.kin.fk(arm_joints)                 # [pos, wxyz]
        pos = np.asarray(pose7[:3], dtype=np.float64)
        euler = _R.from_quat(_wxyz_to_xyzw(pose7[3:7])).as_euler(self.euler_order)
        return np.concatenate([pos, euler])

    def ik6(self, pose6, warm=None):
        """[x,y,z, e1,e2,e3] -> (7 arm joints, success)."""
        pos = np.asarray(pose6[:3], dtype=np.float64)
        euler = np.asarray(pose6[3:6], dtype=np.float64)
        quat_wxyz = _xyzw_to_wxyz(_R.from_euler(self.euler_order, euler).as_quat())
        return self.kin.ik(np.concatenate([pos, quat_wxyz]), warm)
