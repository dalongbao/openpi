"""
scene_fidelity.py — push the Isaac scene toward the real Aria training frames so pi0.5's
SigLIP encoder isn't out-of-distribution (the confirmed cause of the frozen sim policy).

Realism, native + procedural only (no external asset files -> offline-safe on the GPU node):
  - PROCEDURAL TEXTURES generated at runtime (numpy+cv2 -> PNG): wood (table+floor),
    brick (walls), red/green split (ball). Written to /workspace, bound via UsdUVTexture.
  - A FULL enclosing room: large wood floor + brick back/side walls that fill the camera
    background (no black void), instead of the old corner-only patch.
  - Dome + key lighting for soft, photo-like shading.
  - Bigger, two-tone ball to match the real red/green ball.

Shared by eval_script_object_in_bowl.py and scene_preview.py. Call apply_fidelity(stage)
AFTER the object/bowl/table exist. Tune the constants below and re-render with
scene_preview.py (~3-4 min, no policy) before spending a full eval.
"""
import os
import numpy as np
import cv2
from pxr import UsdGeom, UsdLux, UsdShade, Gf, Sdf

# --- tunables -------------------------------------------------------------------
TABLE_TOP_Z    = 1.807
TEX_DIR        = "/workspace"          # where generated texture PNGs are written
WOOD_TILES     = 2.5                   # larger plywood grain (was 6 -> too busy/orange)
BACKDROP_TILES = 1.0                   # backdrop maps ONCE (solid bands; must not repeat)
DOME_INTENSITY = 800.0                # neutral daylight; lower so the plywood isn't washed white
KEY_INTENSITY  = 1000.0
KEY_ROT_XYZ    = (-50.0, 0.0, -35.0)
BALL_RADIUS    = 0.055                 # bigger than the eval default (0.04) -> easier to see
ROOM_HALF      = 4.0                   # floor extends +/- this (m) around the workspace
WALL_DIST      = 1.6                   # backdrop walls this far from workspace centre (closer = fills more)
WALL_HEIGHT    = 3.0


# ---------------------------------------------------------------- procedural textures
# Colours below are sampled from a real Aria training frame (so they already bake in the
# scene lighting). Stored as RGB in the comments; cv2.imwrite writes BGR.
def _wood_png(path, w=512, h=512, seed=0):
    """Light blonde/greige plywood: low-contrast straight grain (matches the real table)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grain = 0.5 + 0.5 * np.sin(xx * 0.06 + 0.7 * np.sin(yy * 0.010))
    fine = rng.standard_normal((h, w)) * 0.03
    bright = np.clip(0.92 + 0.10 * (grain - 0.5) + fine, 0.7, 1.12)
    base = np.array([0.62, 0.57, 0.545], np.float32)            # greige RGB ~ (158,146,139)
    rgb = np.clip(base[None, None, :] * bright[..., None], 0, 1)
    cv2.imwrite(path, (rgb[:, :, ::-1] * 255).astype(np.uint8))  # RGB -> BGR


def _backdrop_png(path, w=512, h=512):
    """Two flat bands (no tiling): greenish-gray WALL on top, single dark-reddish FLOOR below.
    On the vertical backdrop quad V=0 (bottom) is floor, V=1 (top) is wall."""
    img = np.empty((h, w, 3), np.uint8)
    img[:, :] = (105, 81, 38)                       # wall  greenish-gray   BGR of RGB(38,81,105)
    floor_top = int(h * 0.35)                        # top 35% wall, bottom 65% floor
    img[floor_top:, :] = (68, 52, 57)               # floor dark purplish   BGR of RGB(57,52,68)
    cv2.imwrite(path, img)


def _ball_png(path, w=256, h=256):
    """Four-colour ball: quadrants blue / yellow / green / red (sampled RGB; written BGR)."""
    img = np.empty((h, w, 3), np.uint8)
    hw, hh = w // 2, h // 2
    img[:hh, :hw] = (173, 78, 10)     # blue   RGB(10,78,173)
    img[:hh, hw:] = (36, 171, 167)    # yellow RGB(167,171,36)
    img[hh:, :hw] = (99, 101, 8)      # green  RGB(8,101,99)
    img[hh:, hw:] = (60, 21, 112)     # red    RGB(112,21,60)
    cv2.imwrite(path, img)


# ---------------------------------------------------------------- materials / geometry
def _textured_material(stage, path, tex_png, rough=0.8):
    mat = UsdShade.Material.Define(stage, path)
    pbr = UsdShade.Shader.Define(stage, path + "/PBR"); pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    reader = UsdShade.Shader.Define(stage, path + "/StReader"); reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    tex = UsdShade.Shader.Define(stage, path + "/Tex"); tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex_png)
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    return mat


def _textured_quad(stage, path, corners, tex_png, st_max):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*c) for c in corners])
    mesh.CreateFaceVertexCountsAttr([4]); mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDoubleSidedAttr(True)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(st_max, 0), Gf.Vec2f(st_max, st_max), Gf.Vec2f(0, st_max)])
    mat = _textured_material(stage, path + "/Mat", tex_png)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    return mesh


def apply_fidelity(stage, root="/World/Fidelity",
                   table_path="/World/SM_HeavyDutyPackingTable_C02_01",
                   object_path="/World/object"):
    wood_png     = os.path.join(TEX_DIR, "tex_wood.png")
    backdrop_png = os.path.join(TEX_DIR, "tex_backdrop.png")
    ball_png     = os.path.join(TEX_DIR, "tex_ball.png")
    _wood_png(wood_png); _backdrop_png(backdrop_png); _ball_png(ball_png)

    cx, cy, z = 0.8, 0.0, TABLE_TOP_Z
    R, H = ROOM_HALF, WALL_HEIGHT

    # 1) Lights.
    dome = UsdLux.DomeLight.Define(stage, root + "/DomeLight")
    dome.CreateIntensityAttr(DOME_INTENSITY); dome.CreateColorAttr(Gf.Vec3f(0.95, 0.96, 1.0))
    key = UsdLux.DistantLight.Define(stage, root + "/KeyLight")
    key.CreateIntensityAttr(KEY_INTENSITY); key.CreateAngleAttr(1.0)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(*KEY_ROT_XYZ))

    # 2) Wood floor — large, at table-top height, so the background reads as table/floor.
    _textured_quad(stage, root + "/Floor",
                   [(cx - R, cy - R, z - 0.01), (cx + R, cy - R, z - 0.01),
                    (cx + R, cy + R, z - 0.01), (cx - R, cy + R, z - 0.01)],
                   wood_png, WOOD_TILES)

    # 3) Backdrop walls — back (+X) and side (+Y), brought close (WALL_DIST) so they fill the
    #    camera background: reddish floor-tile band low, teal wall strip on top.
    W = WALL_DIST
    _textured_quad(stage, root + "/BackWall",
                   [(cx + W, cy - R, z), (cx + W, cy + R, z),
                    (cx + W, cy + R, z + H), (cx + W, cy - R, z + H)],
                   backdrop_png, BACKDROP_TILES)
    _textured_quad(stage, root + "/SideWall",
                   [(cx - R, cy + W, z), (cx + R, cy + W, z),
                    (cx + R, cy + W, z + H), (cx - R, cy + W, z + H)],
                   backdrop_png, BACKDROP_TILES)

    # 4) Wood material on the actual table too (over its broken offline material).
    tprim = stage.GetPrimAtPath(table_path)
    if tprim.IsValid():
        wmat = _textured_material(stage, table_path + "/FidelityWood", wood_png)
        UsdShade.MaterialBindingAPI.Apply(tprim).Bind(wmat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)

    # 5) Ball — bigger + red/green like the real one.
    bprim = stage.GetPrimAtPath(object_path)
    if bprim.IsValid():
        sph = UsdGeom.Sphere(bprim)
        if sph:
            sph.CreateRadiusAttr(BALL_RADIUS)
        bmat = _textured_material(stage, object_path + "/FidelityBall", ball_png, rough=0.5)
        UsdShade.MaterialBindingAPI.Apply(bprim).Bind(bmat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)

    print(f"[fidelity] applied: wood floor+table, brick walls, dome+key light, red/green ball "
          f"(textures in {TEX_DIR})")
