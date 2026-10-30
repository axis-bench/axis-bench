"""FR3 idle filter: cut the arm's stationary stretches out of a teleoperated recording.

HOW IT RELATES TO openpi's DROID FILTER
    openpi's DROID filter (axis_qvla.labels.idle_filter) calls a frame idle when the COMMANDED joint
    velocity stops changing; the FR3 bench's GELLO command never settles. This filter keeps openpi's
    segment rule and its constants and differs in two things: the velocity is MEASURED,
    v = diff(q) * fps over the 7 arm joints (the last row repeated), and frames within +-5 of a
    gripper event are never idle -- an arm holding still while the gripper closes is the grasp, not
    a pause.

THE RULE (every boundary is a unit test)
    * frame t >= 1 is idle when EVERY joint has |v[t] - v[t-1]| < 1e-3: a change of velocity, not a
      low velocity, so steady motion is idle too; frame 0 is never idle;
    * a frame within 5 of a gripper event (|g[t] - g[t-1]| > 0.02 on the gripper command) is never idle;
    * idle runs of >= 7 frames are removed; shorter pauses stay inside their segment;
    * the frames left form segments; a segment of < 16 frames is dropped;
    * each kept segment becomes one episode (axis_qvla.realrobot.realdata.export).

WHERE IT RAN
    At EXPORT time, by the data producer: the published 392-episode FR3 dataset is this filter's
    output on the 250-episode recording (66,463 -> 62,680 frames), i.e. 392 idle-filtered SEGMENTS
    of 250 recordings. Training does not re-run it; the one frame rule the stage-2 configs apply is
    the uniform tail-10 (axis_qvla.realrobot.realdata.tail).

THE GRIPPER SIGNAL IS THE COMMAND
    Events are read from the gripper command (actions[:, 7] on FR3), not the measured gripper
    (state[0]): the published split reproduces with the command, and with the measurement it is
    391 episodes / 62,845 frames.

Rewritten from the dataset producers' FR3 data pipeline (credited in NOTICE); the segment rule is openpi's
(Physical Intelligence), constant for constant. Offline tier, numpy only.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from axis_qvla.labels.idle_filter import idle_mask

Segment = tuple[int, int]


@dataclasses.dataclass(frozen=True)
class IdleRule:
    """The filter's constants. `grip_guard = 0` turns gripper protection off."""

    idle_threshold: float = 1e-3   # on |v[t] - v[t-1]| per joint, rad/s
    min_idle_len: int = 7          # idle runs at least this long are removed
    min_segment_len: int = 16      # kept segments shorter than this are dropped
    grip_guard: int = 5            # frames protected on each side of a gripper event
    grip_delta: float = 0.02       # |gripper command change| above this is an event


FR3_RULE = IdleRule()


def measured_velocity(q: np.ndarray, fps: float) -> np.ndarray:
    """(T, J) joint velocity diff(q) * fps, the last row repeated to keep T rows.

    v[t] is the motion from t to t+1, the response to the command at t. A recording of one frame
    has no motion: zeros.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2:
        raise ValueError(f"expected (T, joints) positions, got shape {q.shape}")
    v = np.diff(q, axis=0) * fps
    return np.vstack([v, v[-1:]]) if len(v) else np.zeros_like(q)


def gripper_protect(grip_cmd: np.ndarray, guard: int, delta: float) -> np.ndarray:
    """(T,) frames within `guard` of a gripper event; frame t is one when |g[t] - g[t-1]| > delta."""
    g = np.asarray(grip_cmd, dtype=np.float64).reshape(-1)
    event = np.zeros(len(g), dtype=bool)
    event[1:] = np.abs(np.diff(g)) > delta
    protect = event.copy()
    for k in range(1, guard + 1):
        protect[k:] |= event[:-k]
        protect[:-k] |= event[k:]
    return protect


def idle_frames(v: np.ndarray, threshold: float = FR3_RULE.idle_threshold) -> np.ndarray:
    """(T,) frame t >= 1 is idle when every joint's velocity changed by less than `threshold` since t-1.

    openpi's test, shared with the DROID filter (axis_qvla.labels.idle_filter.idle_mask).
    """
    v = np.asarray(v, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError(f"expected (T, joints) velocities, got shape {v.shape}")
    return idle_mask(v, threshold)


def _runs(mask: np.ndarray) -> list[Segment]:
    """Half-open [start, end) spans of True in `mask`."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def kept_segments(v: np.ndarray, protect: np.ndarray | None = None, rule: IdleRule = FR3_RULE) -> list[Segment]:
    """The [start, end) segments of a (T, J) velocity that survive the rule."""
    idle = idle_frames(v, rule.idle_threshold)
    if protect is not None:
        protect = np.asarray(protect, dtype=bool)
        if protect.shape != idle.shape:
            raise ValueError(f"protect mask has shape {protect.shape}, the velocity {len(idle)} frames")
        idle &= ~protect
    keep = np.ones(len(idle), dtype=bool)
    for start, end in _runs(idle):
        if end - start >= rule.min_idle_len:
            keep[start:end] = False
    return [(s, e) for s, e in _runs(keep) if e - s >= rule.min_segment_len]


def segments_from_recording(q: np.ndarray, grip_cmd: np.ndarray, fps: float,
                            rule: IdleRule = FR3_RULE) -> list[Segment]:
    """The kept segments of one recording: (T, 7) joint positions, (T,) gripper command, its fps."""
    q = np.asarray(q, dtype=np.float64)
    grip = np.asarray(grip_cmd, dtype=np.float64).reshape(-1)
    if len(grip) != len(q):
        raise ValueError(f"{len(grip)} gripper commands for {len(q)} joint positions")
    protect = gripper_protect(grip, rule.grip_guard, rule.grip_delta) if rule.grip_guard > 0 else None
    return kept_segments(measured_velocity(q, fps), protect, rule)
