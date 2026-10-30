"""State dims 6-7 must be the SAME quantity in the corpus and at eval.

MEASURED over 5,949 frames of the same episodes, under the default closedness encoding:
    closedness the policy TRAINS on   mean 0.408
    closedness it RECEIVES at eval    mean 0.138
    correlation 0.971, mean |difference| 0.270  ->  1.14 sigma of that dim's own std (0.2375)

Cause: the corpus normalises by the ROBOSUITE finger range (open 0.0468 m, closed 0.0170 m, fitted
at R^2 0.96); `encode_state_eef` normalised by the SIM ctrlrange (0.0400 / 0.0000). Two different
physical spans producing two different numbers for one grasp.

`AXIS_EVAL_GRIP_STATE=fingers` stores the RAW finger joint positions instead -- robosuite's
`robot0_gripper_qpos` form -- which removes the difference by construction: no derived constant is
left for the two sides to disagree about. It also keeps both measurements, since the two fingers
correlate only 0.281.
"""

from __future__ import annotations

import pytest

from axis_qvla.bench.sim import action_mapping as AM


def test_default_is_closedness_so_existing_checkpoints_are_unaffected(monkeypatch):
    monkeypatch.delenv("AXIS_EVAL_GRIP_STATE", raising=False)
    assert AM._raw_finger_state_requested() is False


@pytest.mark.parametrize("value", ["fingers", "FINGERS", "  Fingers  "])
def test_fingers_opts_in(monkeypatch, value):
    monkeypatch.setenv("AXIS_EVAL_GRIP_STATE", value)
    assert AM._raw_finger_state_requested() is True


@pytest.mark.parametrize("value", ["closedness", "", "raw", "finger", "0"])
def test_anything_else_keeps_the_default(monkeypatch, value):
    """A typo must not silently change the state contract a checkpoint was trained under."""
    monkeypatch.setenv("AXIS_EVAL_GRIP_STATE", value)
    assert AM._raw_finger_state_requested() is False


def test_the_two_branches_are_documented_as_mutually_exclusive():
    """The raw branch must return early, never fall through into the closedness computation."""
    import inspect

    src = inspect.getsource(AM.encode_state_eef)
    raw_idx = src.index("_raw_finger_state_requested()")
    closed_idx = src.index("FINGER_OPEN_CTRL - FINGER_CLOSE_CTRL")
    assert raw_idx < closed_idx, "the raw branch must be checked first"
    assert "return state" in src[raw_idx:closed_idx], "the raw branch must return before falling through"


def test_raw_branch_uses_both_fingers_not_a_mean():
    """Collapsing to a mean would lose the second finger's measurement."""
    import inspect

    src = inspect.getsource(AM.encode_state_eef)
    start = src.index("_raw_finger_state_requested()")
    # bound the slice at the raw branch's OWN return, not at the closedness block that follows --
    # otherwise this reads the fall-through path's mean and fails on code that is correct
    end = src.index("return state", start) + len("return state")
    raw = src[start:end]
    assert "finger_qadr[0]" in raw and "finger_qadr[1]" in raw
    assert "0.5 * (" not in raw, "the raw branch must not average the fingers"
