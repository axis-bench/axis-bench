"""The overlay check: a frame of the real exterior camera with the outlines of the rendered table and robot.

Render tier (mujoco, numpy, PyAV). The render is the workspace profile's arena and its exterior camera,
with the robot of a recorded scene posed at the joint positions the photograph was taken at and the
recorded objects left out. `compose` puts the real frame with both outlines beside a 50/50 blend of the
two images: where the outlines follow the real table edge and the real arm, the profile's camera pose and
field of view are the real camera's. Colours and lighting are not compared.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from axis_qvla.realrobot.render import rig
from axis_qvla.realrobot.render.assemble import build_target_scene
from axis_qvla.realrobot.render.profile import WorkspaceProfile

COLOURS = {"table": (255, 64, 64), "robot": (64, 255, 64)}


def read_image(path: Path) -> np.ndarray:
    """The first frame of an image or video file, RGB uint8."""
    import av
    with av.open(str(path)) as c:
        return next(c.decode(video=0)).to_ndarray(format="rgb24")


def write_png(path: Path, rgb: np.ndarray) -> None:
    import av
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w", format="image2") as c:
        s = c.add_stream("png")
        s.width, s.height, s.pix_fmt = int(rgb.shape[1]), int(rgb.shape[0]), "rgb24"
        for packet in s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")):
            c.mux(packet)
        for packet in s.encode():
            c.mux(packet)


def fit(image: np.ndarray) -> np.ndarray:
    """`image` centre-cropped to 16:9 and area-resized to the render's 640 x 360."""
    h, w = image.shape[:2]
    if w * 9 > h * 16:
        cw = h * 16 // 9
        image = image[:, (w - cw) // 2:(w - cw) // 2 + cw]
    elif w * 9 < h * 16:
        ch = w * 9 // 16
        image = image[(h - ch) // 2:(h - ch) // 2 + ch]
    h, w = image.shape[:2]
    rows = (np.arange(rig.HEIGHT + 1) * h) // rig.HEIGHT
    cols = (np.arange(rig.WIDTH + 1) * w) // rig.WIDTH
    acc = np.add.reduceat(np.add.reduceat(image.astype(np.float64), rows[:-1], axis=0), cols[:-1], axis=1)
    area = np.diff(rows)[:, None, None] * np.diff(cols)[None, :, None]
    return np.clip(np.rint(acc / area), 0, 255).astype(np.uint8)


def outline(mask: np.ndarray) -> np.ndarray:
    """The pixels of `mask` with a 4-neighbour outside it."""
    m = np.pad(mask, 1)
    inner = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    return mask & ~inner


def compose(real: np.ndarray, rendered: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    left = real.copy()
    for name, m in masks.items():
        left[outline(m)] = COLOURS[name]
    blend = ((real.astype(np.uint16) + rendered.astype(np.uint16)) // 2).astype(np.uint8)
    return np.concatenate([left, blend], axis=1)


def render_view(scene_xml: Path, gripper_xml: Path, profile: WorkspaceProfile,
                joints) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """The exterior camera's image of the arena and the posed robot, and the table and robot masks."""
    scene = build_target_scene(scene_xml, gripper_xml, objects=False, arena=True, profile=profile)
    m, d = scene.model, mujoco.MjData(scene.model)
    d.qpos[scene.arm_qpos] = np.asarray(joints, np.float64)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, height=rig.HEIGHT, width=rig.WIDTH)
    try:
        r.update_scene(d, camera=rig.EXTERIOR_NAME)
        rgb = r.render().copy()
        r.enable_segmentation_rendering()
        r.update_scene(d, camera=rig.EXTERIOR_NAME)
        seg = r.render().copy()
    finally:
        r.close()
    geom = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    table = geom & (seg[..., 0] == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "qvla_table"))
    robot_geoms = [g for g in range(m.ngeom)
                   if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "")
                   .startswith((rig.SOURCE_PREFIX, rig.GRIPPER_PREFIX))]
    robot = geom & np.isin(seg[..., 0], robot_geoms)
    return rgb, {"table": table, "robot": robot}
