"""Build the pi0.5 object_in_bowl evaluation scene for Isaac Sim.

Assembles a single-arm scene: Franka arm, ORCA right hand (not yet welded to
flange — see TODO below), table, bowl, ball, Aria-matched camera, lighting.
Saves as a USD file that eval_pi05.py will load headlessly.

Run on cvg-desktop-52 after `source ~/isaacsim_work/activate.sh`:
    python ~/openpi/3dvision-experiments/build_scene.py

To preview the result:
    Open ~/isaacsim_work/scenes/object_in_bowl.usd in the Isaac Sim GUI.
    Compare the camera view against a real h5 frame and adjust the TUNE
    constants below until the viewpoints match.
"""

import argparse
import math
import pathlib

import numpy as np
from scipy.spatial.transform import Rotation

# SimulationApp must be the very first Isaac import.
from isaacsim import SimulationApp

# ── Layout ───────────────────────────────────────────────────────────────────
# Franka base at world origin, arm reaches in +X. Y is up (ISO standard).
# Camera is positioned at the near edge (large-Z side) looking toward the
# far end (Franka side, which is also the lab-wall/door background side).

TABLE_X_CENTER = 0.55   # table centre in X
TABLE_X_HALF   = 0.55   # half-length in X  →  table spans [0.0, 1.1]
TABLE_Z_HALF   = 0.50   # half-width in Z
TABLE_H        = 0.80   # top-surface height (m)
TABLE_THICK    = 0.04   # slab thickness

BOWL_POS = np.array([0.60, TABLE_H + 0.04, -0.08])   # (x, y, z)
BALL_POS = np.array([0.62, TABLE_H + 0.025, 0.12])

# ── Camera ───────────────────────────────────────────────────────────────────
# TUNE: adjust CAM_EYE until the rendered view matches real h5 frames.
# From visual inspection of training frames: camera is ~0.7 m above the table
# surface, near the far (non-Franka) edge, looking down-and-forward at ~38°.
CAM_EYE    = np.array([0.60, 1.50,  0.70])   # world position (m)
CAM_TARGET = np.array([0.60, 0.85, -0.10])   # point camera looks at
CAM_HFOV   = 110.0                            # Aria RGB horizontal FoV (deg)
IMG_W, IMG_H = 640, 480                       # matches training-data resolution

ORCA_USD = pathlib.Path.home() / "isaacsim_work/assets/orcahand_right.usd"
DEFAULT_OUT = pathlib.Path.home() / "isaacsim_work/scenes/object_in_bowl.usd"


def look_at_euler_xyz(eye: np.ndarray, target: np.ndarray,
                      up: np.ndarray = np.array([0.0, 1.0, 0.0])) -> np.ndarray:
    """Euler XYZ angles (degrees) for a USD camera at `eye` looking at `target`.
    USD convention: camera looks along local -Z, local +Y is up.
    """
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    n = np.linalg.norm(right)
    right = right / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
    up_cam = np.cross(right, fwd)
    # Rotation matrix whose columns are the camera's X, Y, Z axes in world frame
    # (camera -Z = fwd, so last column is -fwd).
    R = np.column_stack([right, up_cam, -fwd])
    return Rotation.from_matrix(R).as_euler("XYZ", degrees=True)


def main(output: pathlib.Path, headless: bool = True) -> None:
    app = SimulationApp({"headless": headless, "renderer": "RayTracedLighting"})

    # All omni/isaac imports must come after SimulationApp.
    import omni.usd
    from omni.isaac.core import World
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from pxr import Gf, UsdGeom, UsdLux, UsdPhysics

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    assets_root = get_assets_root_path()
    if not assets_root:
        raise RuntimeError(
            "Isaac Sim assets root not found. "
            "Make sure Nucleus is running (or start it via Isaac Sim GUI once)."
        )

    # ── Physics scene ─────────────────────────────────────────────────────────
    phys = UsdPhysics.Scene.Define(stage, "/physicsScene")
    phys.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, -1.0, 0.0))
    phys.CreateGravityMagnitudeAttr().Set(9.81)

    # ── Ground plane ──────────────────────────────────────────────────────────
    world.scene.add_default_ground_plane()

    # ── Table ─────────────────────────────────────────────────────────────────
    table = UsdGeom.Cube.Define(stage, "/World/Table")
    table.CreateSizeAttr(1.0)   # unit cube, scaled below
    t_xf = UsdGeom.Xformable(table.GetPrim())
    t_xf.AddTranslateOp().Set(Gf.Vec3d(TABLE_X_CENTER, TABLE_H - TABLE_THICK / 2, 0.0))
    t_xf.AddScaleOp().Set(Gf.Vec3d(TABLE_X_HALF, TABLE_THICK / 2, TABLE_Z_HALF))
    UsdPhysics.CollisionAPI.Apply(table.GetPrim())

    # ── Franka ────────────────────────────────────────────────────────────────
    franka_usd = f"{assets_root}/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    add_reference_to_stage(usd_path=franka_usd, prim_path="/World/Franka")
    # Base at origin; default orientation has the arm reaching upward/forward (+X).

    # ── ORCA right hand ───────────────────────────────────────────────────────
    # TODO (next step): use Robot Assembler GUI to find the Franka flange →
    # ORCA base transform, then replace this with a FixedJoint so the hand
    # moves with the arm during eval.
    add_reference_to_stage(usd_path=str(ORCA_USD), prim_path="/World/OrcaRight")
    orca_xf = UsdGeom.Xformable(stage.GetPrimAtPath("/World/OrcaRight"))
    orca_xf.AddTranslateOp().Set(Gf.Vec3d(0.40, TABLE_H + 0.25, 0.0))

    # ── Bowl ──────────────────────────────────────────────────────────────────
    bowl = UsdGeom.Cylinder.Define(stage, "/World/Bowl")
    bowl.CreateRadiusAttr(0.075)
    bowl.CreateHeightAttr(0.055)
    b_xf = UsdGeom.Xformable(bowl.GetPrim())
    b_xf.AddTranslateOp().Set(Gf.Vec3d(*BOWL_POS.tolist()))
    UsdPhysics.CollisionAPI.Apply(bowl.GetPrim())

    # ── Ball (the object to pick and place) ───────────────────────────────────
    ball = UsdGeom.Sphere.Define(stage, "/World/Ball")
    ball.CreateRadiusAttr(0.025)
    ba_xf = UsdGeom.Xformable(ball.GetPrim())
    ba_xf.AddTranslateOp().Set(Gf.Vec3d(*BALL_POS.tolist()))
    UsdPhysics.RigidBodyAPI.Apply(ball.GetPrim())
    UsdPhysics.CollisionAPI.Apply(ball.GetPrim())
    UsdPhysics.MassAPI.Apply(ball.GetPrim()).CreateMassAttr(0.05)

    # ── Aria-matched camera ───────────────────────────────────────────────────
    cam_prim = stage.DefinePrim("/World/AriaCamera", "Camera")
    cam = UsdGeom.Camera(cam_prim)
    # Horizontal aperture 24 mm (standard); focal length derived from HFOV.
    h_ap = 24.0
    focal = h_ap / (2.0 * math.tan(math.radians(CAM_HFOV / 2.0)))
    v_ap = h_ap * IMG_H / IMG_W   # maintain 480/640 aspect ratio
    cam.CreateHorizontalApertureAttr(h_ap)
    cam.CreateVerticalApertureAttr(v_ap)
    cam.CreateFocalLengthAttr(focal)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 50.0))

    euler = look_at_euler_xyz(CAM_EYE, CAM_TARGET)
    cam_xf = UsdGeom.Xformable(cam_prim)
    cam_xf.AddTranslateOp().Set(Gf.Vec3d(*CAM_EYE.tolist()))
    cam_xf.AddRotateXYZOp().Set(Gf.Vec3d(*euler.tolist()))
    print(f"Camera position : {CAM_EYE}")
    print(f"Camera target   : {CAM_TARGET}")
    print(f"Camera Euler XYZ: {euler.round(1)} deg")
    elev = math.degrees(math.asin((CAM_EYE[1] - CAM_TARGET[1]) / np.linalg.norm(CAM_EYE - CAM_TARGET)))
    print(f"Elevation angle : {elev:.1f} deg below horizontal")

    # ── Lighting ──────────────────────────────────────────────────────────────
    sun = stage.DefinePrim("/World/SunLight", "DistantLight")
    UsdLux.DistantLight(sun).CreateIntensityAttr(800.0)
    lxf = UsdGeom.Xformable(sun)
    lxf.AddRotateXYZOp().Set(Gf.Vec3d(-55.0, 30.0, 0.0))

    fill = stage.DefinePrim("/World/FillLight", "SphereLight")
    UsdLux.SphereLight(fill).CreateIntensityAttr(300.0)
    UsdLux.SphereLight(fill).CreateRadiusAttr(0.3)
    fxf = UsdGeom.Xformable(fill)
    fxf.AddTranslateOp().Set(Gf.Vec3d(0.5, 2.0, 0.8))

    # ── Save ──────────────────────────────────────────────────────────────────
    output.parent.mkdir(parents=True, exist_ok=True)
    omni.usd.get_context().save_as_stage(str(output))
    print(f"\nSaved scene: {output}")

    app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--no-headless", dest="headless", action="store_false",
                        help="Open GUI instead of running headlessly.")
    args = parser.parse_args()
    main(output=args.output.expanduser().resolve(), headless=args.headless)
