"""
scene_fidelity.py — push the Isaac scene closer to the real Aria training frames so
pi0.5's SigLIP vision encoder isn't out-of-distribution (the confirmed cause of the
frozen sim policy: the same checkpoint moves on real frames, freezes on sim frames).

Native USD only — no external textures/assets — so it runs offline on the GPU node.

Shared by eval_script_object_in_bowl.py and scene_preview.py so the visual setup we
actively iterate on lives in ONE place. Call apply_fidelity(stage) AFTER the
object/bowl/table prims exist (it tints the table and fills the empty void around them).

Everything is parametric: tweak the constants below, then re-render with scene_preview.py
(a ~3-4 min job, no policy) to see the result before spending a full eval.
"""
import math
from pxr import UsdGeom, UsdLux, UsdShade, Gf, Sdf

# --- tunables: eyeball against a real Aria frame, re-render, repeat ---------------
TABLE_TOP_Z    = 1.807               # scene table surface height (world Z)
WALL_COLOR     = (0.52, 0.31, 0.26)  # brick-ish reddish brown (Aria background)
FLOOR_COLOR    = (0.62, 0.45, 0.28)  # warm wood — continues the table out to the edges
DOME_INTENSITY = 800.0               # soft ambient fill (also colors the background)
KEY_INTENSITY  = 2500.0              # directional key light for shadows/shape
KEY_ROT_XYZ    = (-45.0, 0.0, -30.0) # key DistantLight orientation (degrees)
ADD_WALLS      = True                # set False if the walls show up in odd places


def _flat_material(stage, path, rgb, rough=0.85):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/S")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _slab(stage, path, center, halfextents, rgb):
    """A purely-visual colored box (UsdGeom.Cube scaled) for floor/walls. No collider,
    so the backdrop never interferes with the manipulation physics."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(2.0)                       # unit cube spans [-1,1] -> scale below
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddScaleOp().Set(Gf.Vec3f(*halfextents))
    mat = _flat_material(stage, path + "/Mat", rgb)
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
        mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    return cube


def apply_fidelity(stage, root="/World/Fidelity",
                   table_path="/World/SM_HeavyDutyPackingTable_C02_01"):
    # 1) Lighting — replace the flat look with soft ambient + a directional key.
    dome = UsdLux.DomeLight.Define(stage, root + "/DomeLight")
    dome.CreateIntensityAttr(DOME_INTENSITY)
    dome.CreateColorAttr(Gf.Vec3f(0.90, 0.90, 0.95))   # also tints the empty background
    key = UsdLux.DistantLight.Define(stage, root + "/KeyLight")
    key.CreateIntensityAttr(KEY_INTENSITY)
    key.CreateAngleAttr(1.0)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(*KEY_ROT_XYZ))

    # 2) Floor — extend a wood surface far past the table so the camera sees a continuous
    #    tabletop/floor (like the real photo) instead of a black void around the objects.
    _slab(stage, root + "/Floor",
          center=(0.8, 0.0, TABLE_TOP_Z - 0.01),
          halfextents=(4.0, 4.0, 0.005), rgb=FLOOR_COLOR)

    # 3) Backdrop walls — brick-ish surfaces beyond the far table edges (camera looks +X/+Y).
    if ADD_WALLS:
        _slab(stage, root + "/BackWall",
              center=(3.5, 1.5, TABLE_TOP_Z + 1.0),
              halfextents=(0.02, 4.0, 1.5), rgb=WALL_COLOR)
        _slab(stage, root + "/SideWall",
              center=(0.8, 3.5, TABLE_TOP_Z + 1.0),
              halfextents=(4.0, 0.02, 1.5), rgb=WALL_COLOR)

    # 4) Table — warm wood tint so it reads as a wooden table, not flat tan.
    tprim = stage.GetPrimAtPath(table_path)
    if tprim.IsValid():
        mat = _flat_material(stage, table_path + "/FidelityWood", (0.45, 0.30, 0.16))
        UsdShade.MaterialBindingAPI.Apply(tprim).Bind(
            mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)

    print(f"[fidelity] applied: dome({DOME_INTENSITY}) + key({KEY_INTENSITY}) light, "
          f"floor, walls={ADD_WALLS}, table wood tint")
