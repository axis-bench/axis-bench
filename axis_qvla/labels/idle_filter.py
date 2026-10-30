"""Non-idle frame ranges, the filter openpi uses for every production pi-DROID model.

WHY THIS MATTERS FOR US
    Every VLA loader surveyed -- openpi, LeRobot, Octo, OpenVLA, GR00T, RoboVLM -- builds
    action chunks at stride 1, i.e. one training sample starting at EVERY timestep. At 15 fps
    with a 16-step horizon, consecutive samples therefore overlap by 15/16 frames. During a
    dwell (robot holding still) those samples are near-duplicates of each other AND the
    action label is ~constant, so they cost full compute for almost no gradient signal.
    Dropping idle frames is the only lever any of those codebases actually pulls against
    this, and openpi is the only one that implements it.

    Reference implementation: openpi/examples/droid/compute_droid_nonidle_ranges.py, whose
    output (droid_sample_ranges_v1_0_1.json) we measured at 25.77M kept frames over 95 658
    episodes -- 269.4 kept frames/episode, median 1 contiguous range, 94.8% retention inside
    the span it covers. So on DROID it behaves mostly as a tail trim, not aggressive dedup.

THRESHOLD IS NOT PORTABLE
    openpi's 1e-3 is tuned to DROID's VR-teleop noise floor and controller quantisation, and
    is applied to CONSECUTIVE DIFFERENCES of the COMMANDED joint velocity -- so "idle" means
    the command is not CHANGING, not that the robot is stationary. Our rendered episodes come
    from a planner, not a human holding a VR controller, so the idle distribution is
    different in kind. Calibrate `idle_eps` against a real signal histogram instead of
    copying 1e-3 -- measured on our converted corpus (task 1644, delta of the droid8d
    velocity action, which IS openpi's criterion since that action == 15*diff(state)):
    retention 82.9% / 78.4% / 71.4% / 61.4% at idle_eps 1e-4 / 1e-3 / 1e-2 / 5e-2.

Offline (numpy only). Runs under the offline environment.
"""

from __future__ import annotations

import numpy as np

from axis_qvla.core.canonical import canonical_name

# openpi's published defaults, kept verbatim so our behaviour is comparable to the
# pi-DROID recipe when the same signal/threshold are supplied.
MIN_IDLE_LEN = 7
MIN_NON_IDLE_LEN = 16
TRIM_LAST_N = 10
DROID_IDLE_EPS = 1e-3

# The filter the corpus ranges were built with (axis_qvla.data.index.ranges): openpi's defaults above,
# applied FORWARD to the 8-D state, so a frame is idle when the action trained at it is ~zero.
# Every row space records this, so it is part of every row-space id: the function names keep the
# development root in every tree (axis_qvla.core.canonical), whatever root the package imports under.
IDLE_FILTER = {
    "function": canonical_name("labels.idle_filter.nonidle_ranges"),
    "signal": ("observation.state as the 8-D idle signal ("
               + canonical_name("data.lerobot_local.state_to_idle_signal") + ")"),
    "forward": True,
    "idle_eps": DROID_IDLE_EPS,
    "min_idle_len": MIN_IDLE_LEN,
    "min_non_idle_len": MIN_NON_IDLE_LEN,
    "trim_last_n": TRIM_LAST_N,
}


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Half-open [start, end) spans of True in `mask`.

    Padded with False on both sides so a run touching either end is still detected as a
    transition rather than silently dropped.
    """
    padded = np.concatenate([[False], mask, [False]])
    d = np.diff(padded.astype(np.int8))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def idle_mask(signal: np.ndarray, idle_eps: float = DROID_IDLE_EPS, *, forward: bool = False) -> np.ndarray:
    """Per-frame idle flag: True where the state does not CHANGE across one frame.

    ``forward=False`` (default, generic): backward diff -- frame t is idle iff
    ``|x[t]-x[t-1]|<eps``. Frame 0 is never idle (no predecessor).

    ``forward=True`` (frame-matched, used by the pretraining ranges): frame t is idle iff
    ``|x[t+1]-x[t]|<eps`` -- i.e. iff the DROID-8D action label actually regressed at row t
    (``fps*(state[t+1]-state[t])``) is ~zero. This is what "this frame teaches the policy to
    hold still" means for the sample drawn at row t; the backward form tests action[t-1]
    instead, one frame late. The last frame has no successor (its action is zeroed by the
    converter), so it is idle.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    out = np.zeros(len(x), dtype=bool)
    if len(x) > 1:
        moved = np.all(np.abs(np.diff(x, axis=0)) < idle_eps, axis=1)  # moved[i] = |x[i+1]-x[i]|<eps
        if forward:
            out[:-1] = moved  # frame t idle iff |x[t+1]-x[t]|<eps
            out[-1] = True  # no successor -> action zeroed -> idle
        else:
            out[1:] = moved  # frame t idle iff |x[t]-x[t-1]|<eps
    return out


def nonidle_ranges(
    signal: np.ndarray,
    *,
    idle_eps: float = DROID_IDLE_EPS,
    min_idle_len: int = MIN_IDLE_LEN,
    min_non_idle_len: int = MIN_NON_IDLE_LEN,
    trim_last_n: int = TRIM_LAST_N,
    forward: bool = False,
) -> list[tuple[int, int]]:
    """Half-open [start, end) ranges of frames worth training on.

    Faithful to openpi's algorithm, including two details that are easy to get wrong:
      * Only idle runs of length >= min_idle_len are removed. Shorter pauses are TOLERATED
        inside a kept range -- otherwise every brief grasp dwell would shatter one clean
        episode into dozens of fragments, each then failing the min_non_idle_len test.
      * The min_non_idle_len test is applied to the UNTRIMMED run, and trim_last_n is
        subtracted only when the range is emitted. A run of exactly min_non_idle_len
        therefore yields (min_non_idle_len - trim_last_n) frames, not zero.
    The tail trim exists because a chunk starting near the end of a range looks ahead into
    the idle frames that follow it, so its action label is mostly the idle it was meant to
    exclude.
    """
    idle = idle_mask(signal, idle_eps, forward=forward)
    keep = np.ones(len(idle), dtype=bool)
    starts, ends = _runs(idle)
    long_enough = (ends - starts) >= min_idle_len
    for s, e in zip(starts[long_enough], ends[long_enough]):
        keep[s:e] = False

    out: list[tuple[int, int]] = []
    ks, ke = _runs(keep)
    long_enough = (ke - ks) >= min_non_idle_len
    for s, e in zip(ks[long_enough], ke[long_enough]):
        end = int(e) - trim_last_n
        if end > s:
            out.append((int(s), end))
    return out


