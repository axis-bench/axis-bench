"""The 2F-85's two settled extremes, and the gripper state of every knot between them.

Render tier (mujoco, numpy). The 2F-85 is a closed four-bar held by equality constraints, so its joints
cannot be written from a ctrl value; the linkage is settled once at each end of the actuator range
(gravity off, arm pinned, 1500 steps) and every knot interpolates the settled joint positions AND the
ctrl linearly by the source finger openness u in [0, 1]: ctrl = closed + u * (open - closed), i.e.
255 * (1 - u), which is action[7] of the render.
"""
from __future__ import annotations

import dataclasses

import mujoco
import numpy as np

from axis_qvla.realrobot.render import rig
from axis_qvla.realrobot.render.assemble import TargetScene


@dataclasses.dataclass(frozen=True)
class GripperExtremes:
    open_qpos: np.ndarray
    closed_qpos: np.ndarray
    open_ctrl: float
    closed_ctrl: float
    open_separation: float
    closed_separation: float


def settle_extremes(scene: TargetScene, arm_q: np.ndarray, steps: int = rig.GRIPPER_SETTLE_STEPS) -> GripperExtremes:
    m = scene.model
    lo, hi = (float(v) for v in m.actuator_ctrlrange[scene.gripper_actuator])
    gravity = m.opt.gravity.copy()
    m.opt.gravity[:] = 0.0
    out = []
    try:
        for ctrl in (lo, hi):
            d = mujoco.MjData(m)
            d.qpos[:] = m.qpos0
            d.qpos[scene.arm_qpos] = arm_q
            d.ctrl[scene.gripper_actuator] = ctrl
            for _ in range(steps):
                d.qpos[scene.arm_qpos] = arm_q
                d.qvel[scene.arm_dof] = 0.0
                mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
            sep = float(np.linalg.norm(d.xpos[scene.finger_bodies[0]] - d.xpos[scene.finger_bodies[1]]))
            out.append((d.qpos[scene.gripper_qpos].copy(), ctrl, sep))
    finally:
        m.opt.gravity[:] = gravity
    o, c = (out[0], out[1]) if out[0][2] >= out[1][2] else (out[1], out[0])
    return GripperExtremes(o[0], c[0], o[1], c[1], o[2], c[2])


def gripper_knots(ext: GripperExtremes, openness: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(openness, np.float64)[:, None]
    qpos = ext.closed_qpos[None, :] + u * (ext.open_qpos - ext.closed_qpos)[None, :]
    ctrl = ext.closed_ctrl + u[:, 0] * (ext.open_ctrl - ext.closed_ctrl)
    return qpos, ctrl


def check_extremes(ext: GripperExtremes, *, open_separation: float, closed_separation: float,
                   tol: float = rig.GRIPPER_SEPARATION_TOL_M) -> list[str]:
    problems = []
    if abs(ext.open_separation - open_separation) > tol:
        problems.append(f"open separation {ext.open_separation:.6f} m, expected {open_separation:.6f}")
    if abs(ext.closed_separation - closed_separation) > tol:
        problems.append(f"closed separation {ext.closed_separation:.6f} m, expected {closed_separation:.6f}")
    if (ext.open_ctrl, ext.closed_ctrl) != (0.0, 255.0):
        problems.append(f"ctrl range {(ext.open_ctrl, ext.closed_ctrl)}, expected (0.0, 255.0)")
    return problems
