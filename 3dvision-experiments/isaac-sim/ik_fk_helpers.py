"""
ik_fk_helpers.py — shared FK/IK for the Egoverse EE-pose convention, used by both
eval_replay_ik.py (GT replay) and eval_script_object_in_bowl.py (policy eval) so the
solver setup lives in ONE place.

IMPORTANT: import this module only AFTER SimulationApp(...) has started — the Lula
imports require the running Isaac app.

The Egoverse demo convention (resolved 2026-06-03, see eval_replay_ik.py header):
  - poses are in the FR3 BASE frame (Lula computes base-relative, so no base transform)
  - control point = `panda_hand`
  - quaternion stored order = `xyzw` (scalar-LAST)  -> QUAT_WXYZ=False
  - absolute targets

FK (sim joints -> EE pose) builds the policy OBSERVATION.
IK (EE pose -> joints) executes the policy ACTION (and the GT replay).
Both speak the same stored convention so one FrankaKinematics instance serves both.
"""
import numpy as np

# Lula (try deprecated then new namespace) — needs the Isaac app already running.
try:
    from omni.isaac.motion_generation import interface_config_loader
    from omni.isaac.motion_generation.lula import LulaKinematicsSolver
except Exception:
    from isaacsim.robot_motion.motion_generation import interface_config_loader
    from isaacsim.robot_motion.motion_generation.lula import LulaKinematicsSolver


def _mat_to_quat_wxyz(R):
    """3x3 rotation matrix -> unit quaternion [w,x,y,z] (Shepperd's method)."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0.0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])


# preference order when the requested EE frame name isn't in the Lula descriptor
_FRAME_FALLBACKS = ("panda_hand", "right_gripper", "panda_rightfinger",
                    "panda_link8", "fr3_hand", "tool0")


class FrankaKinematics:
    """Wraps the Lula Franka solver in the Egoverse stored pose convention.

    pose7 everywhere is [x, y, z, q...] where q is in the *stored* order
    (xyzw if quat_wxyz=False, wxyz if True). Lula internally uses wxyz; we convert.
    """

    def __init__(self, ee_frame="panda_hand", quat_wxyz=False):
        cfg = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
        self.solver = LulaKinematicsSolver(**cfg)
        self.frames = list(self.solver.get_all_frame_names())
        self.joint_names = list(self.solver.get_joint_names())
        self.quat_wxyz = bool(quat_wxyz)
        self.ee_frame = ee_frame if ee_frame in self.frames else self._fallback()
        print(f"[kin] Lula joints: {self.joint_names}")
        print(f"[kin] Lula frames: {self.frames}")
        print(f"[kin] EE frame: {self.ee_frame} | stored quat order: "
              f"{'wxyz' if self.quat_wxyz else 'xyzw'}")

    def _fallback(self):
        for c in _FRAME_FALLBACKS:
            if c in self.frames:
                return c
        return self.frames[-1]

    # --- quaternion order conversions: stored <-> Lula(wxyz) ---
    def _stored_to_wxyz(self, q):
        q = np.asarray(q, dtype=np.float64)
        return q if self.quat_wxyz else np.array([q[3], q[0], q[1], q[2]])

    def _wxyz_to_stored(self, q_wxyz):
        w, x, y, z = q_wxyz
        return np.array([w, x, y, z]) if self.quat_wxyz else np.array([x, y, z, w])

    def ik(self, pose7, warmstart=None):
        """EE pose (base frame, stored convention) -> (7 arm joints, success)."""
        pos = np.asarray(pose7[:3], dtype=np.float64)
        quat_wxyz = self._stored_to_wxyz(pose7[3:7])
        joints, ok = self.solver.compute_inverse_kinematics(self.ee_frame, pos, quat_wxyz, warmstart)
        return joints, bool(ok)

    def fk(self, arm_joints):
        """7 arm joints -> EE pose7 (base frame, stored convention)."""
        j = np.asarray(arm_joints, dtype=np.float64)[:7]
        pos, rot = self.solver.compute_forward_kinematics(self.ee_frame, j)
        rot = np.asarray(rot, dtype=np.float64)
        if rot.shape == (3, 3):
            q_wxyz = _mat_to_quat_wxyz(rot)
        elif rot.size == 4:                       # already a quaternion (assume wxyz)
            q_wxyz = rot.reshape(4)
        else:
            raise ValueError(f"unexpected FK rotation shape {rot.shape}")
        q_stored = self._wxyz_to_stored(q_wxyz)
        return np.concatenate([np.asarray(pos, dtype=np.float64), q_stored])
