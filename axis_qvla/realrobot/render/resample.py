"""The render's 15 fps time base and its interpolants.

Render tier (numpy, scipy, mujoco). Hinge and slide joints are cubic Hermite where the recorded
velocity is reliable and PCHIP otherwise, free joints Hermite position with slerp, controls a causal
zero-order hold. The float arithmetic of the grid (end - k / fps) is part of the result -- which knot a
frame that falls exactly on a knot time holds depends on it.
"""
from __future__ import annotations

from collections import Counter

import mujoco
import numpy as np
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp

from axis_qvla.realrobot.render import rig

TIME_EPS = 1e-9
VELOCITY_RELIABILITY_P90_MAX = 0.5
HERMITE_ENVELOPE_FRACTION = 0.05


def frame_grid(knot_t, fps: int = rig.FPS) -> tuple[np.ndarray, np.ndarray]:
    """(frame times, the frame of each knot). Knot k is on frame rint((t_k - t_0) * fps) + KNOT_ROW_OFFSET,
    the label pipeline's rule (axis_qvla.labels.resample.trace_anchor_rows); the last knot ends the episode
    and frames run back from it 1 / fps apart."""
    kt = np.asarray(knot_t, np.float64)
    rows = np.rint((kt - kt[0]) * fps).astype(np.int64) + rig.KNOT_ROW_OFFSET
    n = int(rows[-1]) + 1
    return float(kt[-1]) - np.arange(n - 1, -1, -1, dtype=np.float64) * (1.0 / float(fps)), rows


def with_initial(t0: float, knot_t: np.ndarray, *arrays: np.ndarray):
    """Prepend the initial snapshot as a knot at t0. The retargeted snapshot is knot 0 itself."""
    return (np.concatenate([[float(t0)], knot_t]), *(np.concatenate([a[:1], a]) for a in arrays))


def interpolate_coordinate(knot_t, values, velocity, grid_t, *, bounds=None, speed_floor: float):
    values = np.asarray(values, np.float64)
    if len(knot_t) == 2:
        return np.interp(grid_t, knot_t, values), "linear_two_knots"
    reliable = False
    if velocity is not None and np.all(np.isfinite(velocity)):
        velocity = np.asarray(velocity, np.float64)
        ref = np.gradient(values, knot_t, edge_order=2)
        scale = max(float(np.quantile(np.abs(ref), 0.9)), float(speed_floor))
        err = np.abs(velocity - ref) / (np.abs(ref) + scale)
        moving = np.abs(ref) > float(speed_floor)
        if np.any(moving):
            ratio = np.abs(velocity[moving]) / (np.abs(ref[moving]) + float(speed_floor))
            mag_ok = bool(np.quantile(ratio, 0.5) >= 0.2 and np.quantile(ratio, 0.9) <= 5.0)
            dir_ok = bool(np.mean(velocity[moving] * ref[moving] >= 0.0) >= 0.9)
        else:
            mag_ok = dir_ok = True
        reliable = bool(np.quantile(err, 0.9) <= VELOCITY_RELIABILITY_P90_MAX and mag_ok and dir_ok)
    reason = "missing_or_unreliable_velocity"
    if reliable:
        h = CubicHermiteSpline(knot_t, values, velocity, extrapolate=False)
        fr = np.array([0.25, 0.5, 0.75])
        probe = np.asarray(h((knot_t[:-1, None] + np.diff(knot_t)[:, None] * fr[None, :]).reshape(-1)),
                           np.float64).reshape(len(knot_t) - 1, 3)
        lo, hi = np.minimum(values[:-1], values[1:]), np.maximum(values[:-1], values[1:])
        env = np.maximum(np.abs(values[1:] - values[:-1]) * HERMITE_ENVELOPE_FRACTION, 1e-8)
        env_ok = bool(np.all(probe >= (lo - env)[:, None]) and np.all(probe <= (hi + env)[:, None]))
        bounds_ok = bool(bounds is None or (np.all(probe >= bounds[0] - 1e-7) and np.all(probe <= bounds[1] + 1e-7)))
        if env_ok and bounds_ok:
            return np.asarray(h(grid_t), np.float64), "cubic_hermite_reliable_qvel"
        reason = "hermite_overshoot_guard"
    return np.asarray(PchipInterpolator(knot_t, values, extrapolate=False)(grid_t), np.float64), f"pchip_{reason}"


def slerp(knot_t, quat_wxyz, grid_t) -> np.ndarray:
    q = np.asarray(quat_wxyz, np.float64).copy()
    n = np.linalg.norm(q, axis=1)
    if np.any(~np.isfinite(n)) or np.any(n <= 1e-12):
        raise ValueError("quaternion interpolation received an invalid quaternion")
    q /= n[:, None]
    for i in range(1, len(q)):
        if float(np.dot(q[i - 1], q[i])) < 0.0:
            q[i] *= -1.0
    if len(knot_t) == 1:
        return np.repeat(q, len(grid_t), axis=0)
    xyzw = np.concatenate([q[:, 1:], q[:, :1]], axis=1)
    out = Slerp(knot_t, Rotation.from_quat(xyzw))(grid_t).as_quat()
    wxyz = np.concatenate([out[:, 3:], out[:, :3]], axis=1)
    return wxyz / np.linalg.norm(wxyz, axis=1, keepdims=True)


def resample_qpos(model: mujoco.MjModel, knot_t, qpos, qvel, grid_t) -> tuple[np.ndarray, dict]:
    out = np.empty((len(grid_t), model.nq))
    covered = np.zeros(model.nq, bool)
    methods: Counter[str] = Counter()
    for j in range(model.njnt):
        jt, qa, da = int(model.jnt_type[j]), int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
        if jt == int(mujoco.mjtJoint.mjJNT_FREE):
            for c in range(3):
                out[:, qa + c], m = interpolate_coordinate(knot_t, qpos[:, qa + c], qvel[:, da + c], grid_t,
                                                           speed_floor=1e-4)
            out[:, qa + 3:qa + 7] = slerp(knot_t, qpos[:, qa + 3:qa + 7], grid_t)
            covered[qa:qa + 7] = True
            methods["free"] += 1
            continue
        if jt == int(mujoco.mjtJoint.mjJNT_BALL):
            out[:, qa:qa + 4] = slerp(knot_t, qpos[:, qa:qa + 4], grid_t)
            covered[qa:qa + 4] = True
            methods["ball"] += 1
            continue
        label = "slide" if jt == int(mujoco.mjtJoint.mjJNT_SLIDE) else "hinge"
        values = qpos[:, qa].copy()
        if label == "hinge" and not bool(model.jnt_limited[j]):
            values = np.unwrap(values, period=2.0 * np.pi)
        bounds = None
        if bool(model.jnt_limited[j]):
            bounds = (min(float(model.jnt_range[j, 0]), float(values.min())),
                      max(float(model.jnt_range[j, 1]), float(values.max())))
        out[:, qa], m = interpolate_coordinate(knot_t, values, qvel[:, da], grid_t, bounds=bounds,
                                               speed_floor=1e-4 if label == "slide" else 1e-3)
        covered[qa] = True
        methods[label] += 1
        methods[f"{label}:{m}"] += 1
    if not np.all(covered) or not np.all(np.isfinite(out)):
        raise ValueError("state interpolation left qpos entries unset or non-finite")
    return out, dict(sorted(methods.items()))


def zoh(knot_t, values, grid_t) -> np.ndarray:
    idx = np.clip(np.searchsorted(knot_t, grid_t, side="right") - 1, 0, len(knot_t) - 1)
    return np.asarray(values, np.float64)[idx]
