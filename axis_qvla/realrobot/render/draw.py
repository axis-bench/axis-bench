"""Camera poses, end-effector state and frames of a rendered episode.

Render tier. `poses` needs no OpenGL (forward kinematics and camera placement only); `frames` renders
the scene's two cameras offscreen and needs a GL backend (MUJOCO_GL=egl on Mesa llvmpipe, or osmesa).
camera_to_world is the MuJoCo/OpenGL camera frame (+x right, +y up, -z forward) in world coordinates,
row-major 4x4, as the published render stores it; ee_state is the 2F-85 base pose (xyz, quat wxyz).
"""
from __future__ import annotations

from collections.abc import Iterator

import mujoco
import numpy as np

from axis_qvla.realrobot.render import rig
from axis_qvla.realrobot.render.assemble import TargetScene


def camera_to_world(data: mujoco.MjData, cam_id: int) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = data.cam_xmat[cam_id].reshape(3, 3)
    T[:3, 3] = data.cam_xpos[cam_id]
    return T


def poses(scene: TargetScene, qpos_T: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    m, d = scene.model, mujoco.MjData(scene.model)
    c2w = {c.key: np.zeros((len(qpos_T), 16), np.float32) for c in scene.camera_specs}
    ee = np.zeros((len(qpos_T), 7), np.float32)
    for t, q in enumerate(qpos_T):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)
        for c in scene.camera_specs:
            c2w[c.key][t] = camera_to_world(d, scene.cameras[c.key]).reshape(-1)
        ee[t] = np.r_[d.xpos[scene.base_body], d.xquat[scene.base_body]]
    return c2w, ee


def frames(scene: TargetScene, qpos_T: np.ndarray) -> Iterator[dict[str, np.ndarray]]:
    m, d = scene.model, mujoco.MjData(scene.model)
    renderer = mujoco.Renderer(m, height=rig.HEIGHT, width=rig.WIDTH)
    try:
        for q in qpos_T:
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            out = {}
            for c in scene.camera_specs:
                renderer.update_scene(d, camera=c.name)
                out[c.key] = renderer.render().copy()
            yield out
    finally:
        renderer.close()
