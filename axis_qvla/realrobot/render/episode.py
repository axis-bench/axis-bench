"""Knot states of the render model, and their 15 fps frames and actions.

Render tier. Shared object joints are copied from the recording (free-joint positions lifted by the
table height), the arm comes from the IK with its velocity a finite difference over the recorded sample
times, the gripper from the openness. Frames follow resample.py; actions are the causal ZOH of [knot arm
joints, gripper ctrl].
"""
from __future__ import annotations

import dataclasses

import numpy as np

from axis_qvla.realrobot.render import resample, source
from axis_qvla.realrobot.render.assemble import TargetScene
from axis_qvla.realrobot.render.gripper import GripperExtremes, gripper_knots


@dataclasses.dataclass(frozen=True)
class RenderStates:
    grid_t: np.ndarray      # (T,)
    qpos: np.ndarray        # (T, nq) of the render model
    action: np.ndarray      # (T, 8): arm joint targets, gripper ctrl in 0..255
    knot_rows: np.ndarray   # (K,) frame of each source row
    methods: dict


def knot_states(scene: TargetScene, packed: source.PackedState, arm_q: np.ndarray, ext: GripperExtremes,
                openness: np.ndarray, height: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m, K = scene.model, packed.n_steps
    qpos, qvel = np.tile(m.qpos0, (K, 1)), np.zeros((K, m.nv))
    for src_q, tgt_q, nq, src_d, tgt_d, nv, free in scene.shared:
        qpos[:, tgt_q:tgt_q + nq] = packed.qpos[:, src_q:src_q + nq]
        qvel[:, tgt_d:tgt_d + nv] = packed.qvel[:, src_d:src_d + nv]
        if free:
            qpos[:, tgt_q + 2] += height
    qpos[:, scene.arm_qpos] = arm_q
    sample_t = packed.row_times
    grip_q, grip_ctrl = gripper_knots(ext, openness)
    qpos[:, scene.gripper_qpos] = grip_q
    if K > 1:
        qvel[:, scene.arm_dof] = np.gradient(arm_q, sample_t, axis=0, edge_order=1)
        qvel[:, scene.gripper_dof] = np.gradient(grip_q, sample_t, axis=0, edge_order=1)
    return qpos, qvel, grip_ctrl


def compose(scene: TargetScene, packed: source.PackedState, arm_q: np.ndarray, ext: GripperExtremes,
            openness: np.ndarray, height: float) -> RenderStates:
    qk, vk, ck = knot_states(scene, packed, arm_q, ext, openness, height)
    kt = packed.row_times
    grid, rows = resample.frame_grid(kt)
    t_all, q_all, v_all, a_all, c_all = resample.with_initial(packed.lead_time, kt, qk, vk, arm_q, ck[:, None])
    qpos, methods = resample.resample_qpos(scene.model, t_all, q_all, v_all, grid)
    action = np.concatenate([resample.zoh(t_all, a_all, grid), resample.zoh(t_all, c_all, grid)], axis=1)
    return RenderStates(grid, qpos, action, rows, methods)
