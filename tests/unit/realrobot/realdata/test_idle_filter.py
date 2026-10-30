"""Every boundary of the FR3 idle rule, on synthetic velocities and recordings."""
from __future__ import annotations

import numpy as np
import pytest

from axis_qvla.realrobot.realdata.idle_filter import (FR3_RULE, IdleRule, gripper_protect, idle_frames, kept_segments, measured_velocity,
                                   segments_from_recording)


def _velocity(n: int, idle: tuple[tuple[int, int], ...] = ()) -> np.ndarray:
    """(n, 7) velocity whose idle frames are exactly the `idle` spans.

    A moving frame flips every joint between 0 and 5 (|dv| = 5); an idle frame repeats the row
    before it (dv = 0). Frame 0 is never idle, so spans start at >= 1.
    """
    held = {t for a, b in idle for t in range(a, b)}
    v = np.zeros((n, 7))
    level = 0.0
    for t in range(1, n):
        if t not in held:
            level = 5.0 - level
        v[t] = level
    return v


def _recording(n: int, hold: tuple[int, int] | None = None) -> np.ndarray:
    """(n, 7) joint positions stepping 0.01 / 0.02 rad alternately (v alternates 0.15 / 0.30 rad/s at
    15 fps, so no moving frame is idle); over `hold` = [a, b) the arm does not move: q[a-1..b-1] equal."""
    q = np.zeros((n, 7))
    for t in range(1, n):
        moving = hold is None or not hold[0] <= t < hold[1]
        q[t] = q[t - 1] + (0.01 * (1 + t % 2) if moving else 0.0)
    return q


def test_the_constants_are_the_exports():
    assert (FR3_RULE.idle_threshold, FR3_RULE.min_idle_len, FR3_RULE.min_segment_len, FR3_RULE.grip_guard, FR3_RULE.grip_delta) == (
        1e-3, 7, 16, 5, 0.02)


def test_the_helper_makes_exactly_the_idle_frames_asked_for():
    np.testing.assert_array_equal(np.flatnonzero(idle_frames(_velocity(30, ((10, 13),)))), [10, 11, 12])


def test_frame_zero_is_never_idle():
    assert idle_frames(np.zeros((4, 7))).tolist() == [False, True, True, True]


def test_a_velocity_change_of_exactly_the_threshold_is_not_idle():
    v = np.zeros((3, 7))
    v[2, 3] = 1e-3                                   # 1e-3 - 0.0 is exactly 1e-3
    assert idle_frames(v).tolist() == [False, True, False]
    v[2, 3] = np.nextafter(1e-3, 0.0)
    assert idle_frames(v).tolist() == [False, True, True]


def test_one_moving_joint_keeps_the_frame():
    v = np.zeros((3, 7))
    v[2, 6] = 0.5
    assert idle_frames(v).tolist() == [False, True, False]


def test_an_idle_run_of_six_stays_and_of_seven_is_cut():
    assert kept_segments(_velocity(60, ((20, 26),))) == [(0, 60)]
    assert kept_segments(_velocity(60, ((20, 27),))) == [(0, 20), (27, 60)]


def test_a_threshold_change_inside_a_run_breaks_it():
    # frames 1..20 move (20 flips: back at 0.0), so 21..27 hold exactly 0.0 and a bump at 24 is
    # exactly 1e-3 away from both neighbours: 24 and 25 stay non-idle and the run splits 3 + 2.
    v = _velocity(60, ((21, 28),))
    assert kept_segments(v) == [(0, 21), (28, 60)]
    v[24, 0] = 1e-3
    assert kept_segments(v) == [(0, 60)]
    v[24, 0] = np.nextafter(1e-3, 0.0)
    assert kept_segments(v) == [(0, 21), (28, 60)]


def test_a_segment_of_fifteen_is_dropped_and_of_sixteen_kept():
    assert kept_segments(_velocity(60, ((10, 17), (32, 39)))) == [(39, 60)]            # [17, 32) is 15
    assert kept_segments(_velocity(60, ((10, 17), (33, 40)))) == [(17, 33), (40, 60)]  # [17, 33) is 16


def test_a_gripper_event_protects_five_frames_each_side():
    g = np.zeros(30)
    g[15:] = 1.0
    np.testing.assert_array_equal(np.flatnonzero(gripper_protect(g, 5, 0.02)), np.arange(10, 21))


def test_a_gripper_change_of_exactly_the_delta_is_no_event():
    g = np.zeros(10)
    g[5:] = 0.02                                     # 0.02 - 0.0 is exactly 0.02
    assert not gripper_protect(g, 5, 0.02).any()
    g[5:] = np.nextafter(0.02, 1.0)
    assert gripper_protect(g, 5, 0.02).any()


@pytest.mark.parametrize(("event", "kept"), [
    (31, [(0, 60)]),               # protects 26..36: frame 26 leaves 20..25, a run of 6
    (32, [(0, 20), (27, 60)]),     # protects 27..37: the run 20..26 stays 7 and is cut
    (15, [(0, 60)]),               # protects 10..20: frame 20 leaves 21..26, a run of 6
    (14, [(0, 20), (27, 60)]),     # protects 9..19: the run is untouched
])
def test_the_protect_window_reaches_exactly_five_frames(event, kept):
    g = np.zeros(60)
    g[event:] = 1.0
    assert kept_segments(_velocity(60, ((20, 27),)), gripper_protect(g, 5, 0.02)) == kept


def test_measured_velocity_repeats_the_last_row():
    q = np.cumsum(np.full((5, 7), 0.1), axis=0)
    v = measured_velocity(q, 15.0)
    assert v.shape == (5, 7)
    np.testing.assert_allclose(v, 1.5)
    np.testing.assert_array_equal(v[-1], v[-2])
    assert measured_velocity(np.zeros((1, 7)), 15.0).tolist() == [[0.0] * 7]


def test_a_held_pose_is_cut_from_a_recording():
    q = _recording(60, hold=(21, 30))     # q[20..29] equal: v is 0 over 20..28, idle over 21..28
    assert segments_from_recording(q, np.zeros(60), 15.0) == [(0, 21), (29, 60)]


def test_steady_motion_counts_as_idle():
    # a change of velocity, not a low velocity: a constant-velocity sweep is one long idle run
    q = np.outer(np.arange(60) * 0.01, np.ones(7))
    assert segments_from_recording(q, np.zeros(60), 15.0) == []


def test_a_gripper_event_keeps_a_held_pose():
    g = np.zeros(60)
    g[25:] = 1.0                           # protects 20..30, the whole hold
    assert segments_from_recording(_recording(60, hold=(21, 30)), g, 15.0) == [(0, 60)]


def test_grip_guard_zero_turns_protection_off():
    g = np.zeros(60)
    g[25:] = 1.0
    off = IdleRule(grip_guard=0)
    assert segments_from_recording(_recording(60, hold=(21, 30)), g, 15.0, off) == [(0, 21), (29, 60)]


def test_a_recording_too_short_for_a_segment_yields_nothing():
    assert segments_from_recording(np.zeros((4, 7)), np.zeros(4), 15.0) == []
    assert segments_from_recording(np.zeros((0, 7)), np.zeros(0), 15.0) == []


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="gripper"):
        segments_from_recording(np.zeros((5, 7)), np.zeros(4), 15.0)
    with pytest.raises(ValueError, match="protect"):
        kept_segments(np.zeros((5, 7)), np.zeros(4, dtype=bool))
