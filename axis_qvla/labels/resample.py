"""Put 5 Hz trace state onto 15 fps render rows, and mark which rows are real.

A trace step at sim_time t belongs at render row round((t - t0) * fps) + RENDER_FRAME_OFFSET, where
t0 is the attempt's first sim_time; the +2 is the measured render start offset, the one the
training-side samplers align with. Poses are interpolated between anchor rows (slerp for
orientations) and clamped outside the anchor span; `reward_valid_mask` marks the rows a clipped
reward curve really covers.

Offline tier: numpy, scipy.
"""
from __future__ import annotations

import numpy as np

# See the module docstring: measured, and shared with the training-side samplers.
RENDER_FRAME_OFFSET = 2


def trace_anchor_rows(sim_times, fps: float, offset: int = RENDER_FRAME_OFFSET) -> np.ndarray:
    """Render row each trace step maps to. These are the anchors everything interpolates between."""
    t = np.asarray(sim_times, dtype=float)
    if t.size == 0:
        return np.asarray([], dtype=np.int64)
    return np.round((t - t[0]) * fps).astype(np.int64) + offset


def _dedup_anchors(anchor_rows, values):
    """Collapse duplicate anchor rows -- sim_time jitter can round two steps onto one row.

    Keeps the first value per unique row. Assumes `anchor_rows` is non-decreasing, as
    `trace_anchor_rows` returns.
    """
    a = np.asarray(anchor_rows)
    uniq, idx = np.unique(a, return_index=True)
    return uniq, np.asarray(values)[idx]


def resample_poses_to_rows(anchor_rows, poses_wxyz, n_rows: int) -> np.ndarray:
    """Interpolate one body's `(K, 7)` [pos, quat wxyz] anchors onto `n_rows` render rows.

    Position lerps, orientation slerps. Rows outside the anchor span clamp to the nearest
    anchor, because object state is only known over the trace -- extrapolating would invent
    motion. Returns `(n_rows, 7)`, quats still wxyz.
    """
    from scipy.spatial.transform import Rotation, Slerp

    anchors, poses = _dedup_anchors(anchor_rows, poses_wxyz)
    anchors = anchors.astype(float)
    rows = np.clip(np.arange(n_rows, dtype=float), anchors[0], anchors[-1])

    pos = np.column_stack([np.interp(rows, anchors, poses[:, i]) for i in range(3)])
    # trace quats are wxyz (cols 3..6); scipy wants xyzw.
    quat_xyzw = poses[:, [4, 5, 6, 3]]
    slerped = Slerp(anchors, Rotation.from_quat(quat_xyzw))(rows).as_quat()  # xyzw
    return np.hstack([pos, slerped[:, [3, 0, 1, 2]]])


def reward_to_rows(values, anchor_rows, n_rows: int) -> np.ndarray:
    """Map a trace-resolution reward curve onto render rows (interp, clamped at the span)."""
    v = np.asarray(values, dtype=float)
    anchors, v = _dedup_anchors(np.asarray(anchor_rows)[: v.size], v)
    anchors = anchors.astype(float)
    rows = np.clip(np.arange(n_rows, dtype=float), anchors[0], anchors[-1])
    return np.interp(rows, anchors, v)


def success_from_pass_frame(pass_frame, anchor_rows, n_rows: int) -> np.ndarray:
    """Per-row success step: True from the render row the checker first passes.

    `pass_frame` is a trace step index, as recorded in the goal leaves; None means never passed.
    """
    out = np.zeros(n_rows, dtype=bool)
    if pass_frame is None:
        return out
    anchors = np.asarray(anchor_rows, dtype=np.int64)
    k = int(pass_frame)
    if k < 0 or k >= anchors.size:
        return out
    out[int(anchors[k]):] = True
    return out


def anchor_mask(anchor_rows, n_rows: int) -> np.ndarray:
    """True on rows whose object poses are EXACT, False where they were interpolated.

    The trace is 5 Hz and the render 15 fps, so only every third row carries a measured pose.
    Consumers that care about exactness -- contact events are threshold crossings and can shift
    a frame under interpolation -- filter on this.
    """
    out = np.zeros(n_rows, dtype=bool)
    a = np.asarray(anchor_rows, dtype=np.int64)
    a = a[(a >= 0) & (a < n_rows)]
    out[a] = True
    return out


def reward_valid_mask(anchor_rows, n_values: int, n_rows: int) -> np.ndarray:
    """True on rows the reward curve actually covers; False where the stored value is a held
    constant (not covered by the reward curve).

    The curve is clipped at the success frame and `reward_to_rows` clamps outside the anchor
    span, so every row past the clip inherits a CONSTANT max reward. Left unmarked, that tail
    silently over-weights post-success frames in any reward-weighted training.
    """
    out = np.zeros(n_rows, dtype=bool)
    a = np.asarray(anchor_rows, dtype=np.int64)
    if n_values <= 0 or a.size == 0:
        return out
    lo = max(int(a[0]), 0)
    hi = min(int(a[min(n_values, a.size) - 1]), n_rows - 1)
    if hi >= lo:
        out[lo : hi + 1] = True
    return out
