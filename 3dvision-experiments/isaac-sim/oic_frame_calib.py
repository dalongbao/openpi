"""oic_frame_calib.py — calibrate the rigid transform (R, t, s) that maps the oic model's
egocentric EE frame to the Franka base frame.

The oic model's positions live in a small egocentric cube (the Aria/headset frame), not the
robot base frame, so feeding its actions straight into IK reaches nowhere near the sim ball
(see the demo pos ranges vs the ball's base-frame location). This solves the full transform
by anchoring 3 demo keypoints to 3 sim base-frame targets:

    demo START  (hand ready)            <-> robot HOME EE
    demo GRASP  (dwell over the object) <-> BALL position
    demo RELEASE(dwell over container)  <-> BOWL position

dst ~= s * R @ src + t   (Umeyama / Kabsch, closed-form SVD). Used by eval_script_oic.py when
OIC_CALIBRATE=1. Standalone inspection:  python oic_frame_calib.py /workspace/oic_demo.npz
"""
import sys
import numpy as np


def umeyama(src, dst, with_scale=True):
    """Least-squares rigid (+scale) transform s,R,t with dst ~= s*R@src + t. src,dst: (N,3)."""
    src = np.asarray(src, float); dst = np.asarray(dst, float); n = len(src)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    Sigma = (dc.T @ sc) / n
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    s = float((D * np.diag(S)).sum() / ((sc ** 2).sum() / n)) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return R, t, s


def pick_anchor_frames(pos, override=None):
    """pos (N,3) model-frame EE path -> (i_start, i_grasp, i_release).
    grasp/release = the two main DWELLs (smoothed-speed minima) in the first/second half;
    falls back to quartile frames. Pass override='0,75,99' to set them by hand."""
    n = len(pos)
    if override:
        if isinstance(override, str):
            override = [int(x) for x in override.split(",")]
        i0, ig, ir = (int(x) for x in override)
        return max(0, min(n - 1, i0)), max(0, min(n - 1, ig)), max(0, min(n - 1, ir))
    speed = np.r_[0.0, np.linalg.norm(np.diff(pos, axis=0), axis=1)]
    k = max(1, n // 25)
    sm = np.convolve(speed, np.ones(k) / k, mode="same")
    lo = max(1, n // 6); half = max(lo + 1, n // 2)
    i_grasp = int(np.argmin(sm[lo:half]) + lo)
    i_release = int(np.argmin(sm[half:n - 1]) + half) if n - 1 > half else n - 1
    return 0, i_grasp, i_release


def build_transform(demo_pos, base_anchors, anchor_frames=None, with_scale=True):
    """demo_pos (N,3) model frame; base_anchors=(home3, ball3, bowl3) in base frame.
    Returns (R, t, s, (i_start,i_grasp,i_release), residual_per_anchor_m)."""
    i0, ig, ir = pick_anchor_frames(demo_pos, anchor_frames)
    model_anchors = np.array([demo_pos[i0], demo_pos[ig], demo_pos[ir]], float)
    base = np.asarray(base_anchors, float)
    R, t, s = umeyama(model_anchors, base, with_scale)
    pred = s * (model_anchors @ R.T) + t
    resid = np.linalg.norm(pred - base, axis=1)
    return R, t, s, (i0, ig, ir), resid


if __name__ == "__main__":
    npz = sys.argv[1] if len(sys.argv) > 1 else "/workspace/oic_demo.npz"
    d = np.load(npz)
    pos = np.asarray(d["actions6"], float)[:, :3]
    i0, ig, ir = pick_anchor_frames(pos)
    print(f"[calib] {npz}: {len(pos)} frames")
    print(f"[calib] auto anchor frames: start={i0} grasp={ig} release={ir}")
    for nm, i in [("start", i0), ("grasp", ig), ("release", ir)]:
        print(f"  {nm:8s} f{i:3d}  model-pos={np.round(pos[i], 3).tolist()}")
    print("Pair these with (home, ball, bowl) base-frame targets to solve the transform.")
