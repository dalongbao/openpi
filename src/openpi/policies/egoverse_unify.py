"""egoverse_unify.py — put robot (R-ID) and human (H-ID) EE actions in ONE action space.

Why: co-training a single pi0.5 head needs every sample to be the same fixed-width vector
in a CONSISTENT frame/representation. The two sources differ:

  ROBOT object_in_bowl (R-ID): 24D = [pos(3), quat xyzw(4), hand(17)], robot BASE frame,
                               +Z up. (validated: base / xyzw / panda_hand)
  HUMAN object_in_container (H-ID): 6D = [pos(3), yaw,pitch,roll], egocentric HEAD frame,
                               +X right / +Y down / +Z forward, scipy intrinsic "ZYX", rad.
                               No hand channel. (resolved from EgoMimic source pose_utils.py)

Unification: express BOTH in the egocentric CANONICAL frame (+X right, +Y down, +Z forward),
keep the robot's 24D layout (so the gripper survives), and emit a per-dim LOSS MASK so human
samples never supervise the 17 hand dims (the robot is the only grasp teacher).

  human -> [pos, euler->quat, 17 zeros]      mask = [1]*7 + [0]*17   (arm only)
  robot -> [base->canonical pos+quat, hand]  mask = [1]*24           (everything)

CONFIDENCE: the euler->quat math, the 24D placement, and the mask are exact. The robot
base->canonical *position* remap is high-confidence (a signed axis permutation). The
*orientation* remap convention is the one bit to validate — sweep ROBOT_BASE_TO_CANONICAL
(and its sign variants) with the R-ID+H-ID smoke test; reach metrics will pick the right one.

Pure numpy, no scipy (so it's safe to import in the training path). Run directly to self-test:
    python -m openpi.policies.egoverse_unify
"""
import numpy as np

ARM_POS = 3
ARM_QUAT = 4
ARM_DIM = ARM_POS + ARM_QUAT      # 7
HAND_DIM = 17
ACTION_DIM = ARM_DIM + HAND_DIM   # 24

# Base (Franka: +X forward, +Y left, +Z up)  ->  canonical (+X right, +Y down, +Z forward).
#   canonical_X(right)   = -base_Y(left)
#   canonical_Y(down)    = -base_Z(up)
#   canonical_Z(forward) = +base_X(forward)
# det = +1 (proper rotation). SWEEP this (and sign flips) if the smoke test underperforms.
ROBOT_BASE_TO_CANONICAL = np.array([[0.0, -1.0, 0.0],
                                    [0.0, 0.0, -1.0],
                                    [1.0, 0.0, 0.0]])

# CHOSEN unified frame = ROBOT BASE (so robot data AND ablation_eval.py are untouched; only the
# human is moved into base). head(=canonical) -> base is the inverse of base->canonical.
HEAD_TO_BASE = ROBOT_BASE_TO_CANONICAL.T


def euler_zyx_to_quat_xyzw(angles: np.ndarray) -> np.ndarray:
    """Intrinsic ZYX euler [yaw, pitch, roll] (radians) -> quaternion xyzw.

    Matches scipy R.from_euler("ZYX", angles).as_quat() (the EgoMimic convention)."""
    angles = np.asarray(angles, dtype=np.float64).reshape(-1, 3)
    y, p, r = angles[:, 0] / 2, angles[:, 1] / 2, angles[:, 2] / 2     # yaw, pitch, roll
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return np.stack([qx, qy, qz, qw], axis=-1)


def _matrix_to_quat_xyzw(m: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> quaternion xyzw (Shepperd's method, single matrix)."""
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a ⊗ b for xyzw quaternions, batched on a."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[0], b[1], b[2], b[3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def human6d_to_arm7(pose6d: np.ndarray) -> np.ndarray:
    """Human [pos(3), yaw,pitch,roll] (head frame) -> [pos(3), quat xyzw(4)] (canonical = head)."""
    pose6d = np.asarray(pose6d, dtype=np.float64).reshape(-1, 6)
    quat = euler_zyx_to_quat_xyzw(pose6d[:, 3:6])
    return np.concatenate([pose6d[:, :3], quat], axis=-1)


def human6d_to_base_arm7(pose6d: np.ndarray, R_hb: np.ndarray = HEAD_TO_BASE) -> np.ndarray:
    """Human [pos(3), yaw,pitch,roll] (head frame) -> [pos(3), quat xyzw(4)] in ROBOT BASE frame.

    This is the function the mix converter uses (unified frame = robot base). Position:
    p_b = R_hb @ p_head (high confidence). Orientation: left-multiply by quat(R_hb) (validate
    with the smoke test; sweep R_hb sign variants if reach underperforms)."""
    arm7_head = human6d_to_arm7(pose6d)
    pos_b = arm7_head[:, :3] @ R_hb.T
    q_hb = _matrix_to_quat_xyzw(R_hb)
    quat_b = _quat_mul_xyzw(arm7_head[:, 3:7], q_hb)
    return np.concatenate([pos_b, quat_b], axis=-1)


def robot_arm7_base_to_canonical(arm7: np.ndarray, R_cb: np.ndarray = ROBOT_BASE_TO_CANONICAL) -> np.ndarray:
    """Robot [pos(3), quat xyzw(4)] in BASE frame -> canonical frame.

    Position: p_c = R_cb @ p_b (high confidence). Orientation: left-multiply by quat(R_cb)
    (the convention to validate with the smoke test)."""
    arm7 = np.asarray(arm7, dtype=np.float64).reshape(-1, 7)
    pos_c = arm7[:, :3] @ R_cb.T
    q_cb = _matrix_to_quat_xyzw(R_cb)
    quat_c = _quat_mul_xyzw(arm7[:, 3:7], q_cb)
    return np.concatenate([pos_c, quat_c], axis=-1)


def to_unified(arm7: np.ndarray, hand17: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Pack a 7D canonical arm pose (+ optional 17D hand) into the 24D layout + per-dim loss mask.

    hand17 None (human)  -> hand slots zero,  mask = [1]*7 + [0]*17  (arm supervised only)
    hand17 given (robot) -> hand slots filled, mask = [1]*24         (all supervised)
    Returns (action[..,24], mask[..,24])."""
    arm7 = np.asarray(arm7, dtype=np.float32).reshape(-1, ARM_DIM)
    n = arm7.shape[0]
    act = np.zeros((n, ACTION_DIM), dtype=np.float32)
    mask = np.zeros((n, ACTION_DIM), dtype=np.float32)
    act[:, :ARM_DIM] = arm7
    mask[:, :ARM_DIM] = 1.0
    if hand17 is not None:
        act[:, ARM_DIM:] = np.asarray(hand17, dtype=np.float32).reshape(-1, HAND_DIM)
        mask[:, ARM_DIM:] = 1.0
    return act, mask


def _selftest():
    # euler->quat: identity
    q0 = euler_zyx_to_quat_xyzw(np.zeros((1, 3)))[0]
    assert np.allclose(q0, [0, 0, 0, 1], atol=1e-9), q0
    # euler->quat: pure yaw 90deg about Z -> qz=sin45, qw=cos45
    qz = euler_zyx_to_quat_xyzw(np.array([[np.pi / 2, 0, 0]]))[0]
    assert np.allclose(qz, [0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)], atol=1e-9), qz
    # quat always unit norm
    rng = np.random.default_rng(0)
    q = euler_zyx_to_quat_xyzw(rng.uniform(-np.pi, np.pi, (50, 3)))
    assert np.allclose(np.linalg.norm(q, axis=-1), 1.0, atol=1e-9)
    # base->canonical position: base +X(forward) -> canonical +Z(forward); base +Z(up) -> canonical -Y(down)
    arm = robot_arm7_base_to_canonical(np.array([[1, 0, 0, 0, 0, 0, 1.0]]))[0]
    assert np.allclose(arm[:3], [0, 0, 1], atol=1e-9), arm[:3]
    arm = robot_arm7_base_to_canonical(np.array([[0, 0, 1, 0, 0, 0, 1.0]]))[0]
    assert np.allclose(arm[:3], [0, -1, 0], atol=1e-9), arm[:3]
    # human head->base (the path the converter uses): head +Z(fwd) -> base +X(fwd); head +Y(down) -> base -Z(up)
    hb = human6d_to_base_arm7(np.array([[0, 0, 1.0, 0, 0, 0]]))[0]
    assert np.allclose(hb[:3], [1, 0, 0], atol=1e-9), hb[:3]
    hb = human6d_to_base_arm7(np.array([[0, 1.0, 0, 0, 0, 0]]))[0]
    assert np.allclose(hb[:3], [0, 0, -1], atol=1e-9), hb[:3]
    # both remaps are proper rotations and mutual inverses
    assert np.isclose(np.linalg.det(ROBOT_BASE_TO_CANONICAL), 1.0)
    assert np.allclose(ROBOT_BASE_TO_CANONICAL @ HEAD_TO_BASE, np.eye(3), atol=1e-12)
    # packing + mask
    a_h, m_h = to_unified(human6d_to_arm7(np.array([[0.1, 0.2, 0.5, 0.1, 0.2, 0.3]])), None)
    assert a_h.shape == (1, 24) and m_h.shape == (1, 24)
    assert m_h[0, :7].sum() == 7 and m_h[0, 7:].sum() == 0, "human must mask the 17 hand dims"
    a_r, m_r = to_unified(np.zeros((1, 7)), np.ones((1, 17)))
    assert m_r.sum() == 24, "robot must supervise all 24 dims"
    print("egoverse_unify self-test OK  (euler->quat, base->canonical, 24D pack, mask)")


if __name__ == "__main__":
    _selftest()
