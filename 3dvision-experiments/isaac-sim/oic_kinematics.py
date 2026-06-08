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


_AXES = {"x": 0, "y": 1, "z": 2}


def _parse_pos_map(s):
    """'x,-y,z' -> (perm, sign) mapping a BASE-frame position to the MODEL-frame position.
    Used to calibrate the (unknown) transform between the Lula base frame the sim runs in and
    the human/Aria frame the oic model was trained in. Identity is 'x,y,z'."""
    perm, sign = [], []
    for tok in s.split(","):
        tok = tok.strip()
        sign.append(-1.0 if tok.startswith("-") else 1.0)
        perm.append(_AXES[tok.lstrip("+-")])
    return np.asarray(perm, dtype=int), np.asarray(sign, dtype=np.float64)


class OicKinematics:
    """6-dim [pos, euler] convention over the shared Franka Lula solver."""

    def __init__(self, ee_frame="panda_hand", euler_order="XYZ", pos_map="x,y,z"):
        # Drive the underlying solver in wxyz so we control the quat<->euler step here.
        self.kin = ik_fk_helpers.FrankaKinematics(ee_frame=ee_frame, quat_wxyz=True)
        self.euler_order = euler_order
        self.ee_frame = self.kin.ee_frame
        self.solver = self.kin.solver
        self.frames = self.kin.frames
        self.perm, self.sign = _parse_pos_map(pos_map)
        print(f"[oic-kin] euler_order={euler_order}  ee_frame={self.ee_frame}  pos_map={pos_map}")

    def fk6(self, arm_joints):
        """7 arm joints -> [x,y,z, e1,e2,e3] in the MODEL frame (euler in self.euler_order)."""
        pose7 = self.kin.fk(arm_joints)                 # [base_pos, wxyz]
        base_pos = np.asarray(pose7[:3], dtype=np.float64)
        model_pos = self.sign * base_pos[self.perm]     # base -> model
        euler = _R.from_quat(_wxyz_to_xyzw(pose7[3:7])).as_euler(self.euler_order)
        return np.concatenate([model_pos, euler])

    def ik6(self, pose6, warm=None):
        """[x,y,z, e1,e2,e3] in the MODEL frame -> (7 arm joints, success)."""
        model_pos = np.asarray(pose6[:3], dtype=np.float64)
        base_pos = np.empty(3, dtype=np.float64)
        base_pos[self.perm] = model_pos * self.sign     # invert: model -> base
        euler = np.asarray(pose6[3:6], dtype=np.float64)
        quat_wxyz = _xyzw_to_wxyz(_R.from_euler(self.euler_order, euler).as_quat())
        return self.kin.ik(np.concatenate([base_pos, quat_wxyz]), warm)
