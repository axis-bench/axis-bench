"""Non-idle range extraction (offline venv).

Pins the two behaviours that are easy to get subtly wrong when reimplementing openpi's
filter: short pauses must NOT fragment a range, and the length test must run on the
untrimmed span.
"""

from __future__ import annotations

import numpy as np
import pytest

from axis_qvla.labels.idle_filter import idle_mask, nonidle_ranges


def _moving(n: int, step: float = 1.0) -> np.ndarray:
    return (np.arange(n, dtype=float) * step)[:, None]


def _still(n: int, at: float = 0.0) -> np.ndarray:
    return np.full((n, 1), at, dtype=float)


def test_idle_mask_forward_aligns_with_the_action_label():
    # frame t's trained action label is fps*(state[t+1]-state[t]) (FORWARD diff). The default
    # backward mask tests action[t-1] instead -- one frame late. `forward=True` makes
    # idle_mask[t] test the action actually regressed at row t.
    s = np.concatenate([np.arange(20.0), np.full(20, 19.0)])[:, None]  # move 0..19, then hold
    bwd = idle_mask(s, idle_eps=1e-3)
    fwd = idle_mask(s, idle_eps=1e-3, forward=True)
    # frame 19: state[20]-state[19]=0 -> its action IS idle (forward), but state[19]-state[18]=1
    # so backward wrongly marks it moving.
    assert not bwd[19]
    assert fwd[19]
    # last frame has no successor (action is zeroed) -> idle under forward alignment.
    assert fwd[-1]
    # a moving first frame has a real action -> NOT idle under forward (backward can't see it).
    assert not fwd[0]


def test_frame_zero_is_never_idle():
    """It has no predecessor; calling it idle would let a run start before any motion."""
    assert not idle_mask(_still(10))[0]


def test_all_motion_keeps_everything_but_the_trim():
    r = nonidle_ranges(_moving(100), idle_eps=1e-3)
    assert r == [(0, 90)]  # 100 frames minus the 10-frame tail trim


def test_short_pause_does_not_fragment_a_range():
    """A 5-frame dwell (< min_idle_len=7) must be tolerated INSIDE one range.

    Fragmenting here would shatter every grasp dwell into pieces that then fail the
    min_non_idle_len test and vanish entirely.
    """
    sig = np.vstack([_moving(40), _still(5, 39.0), _moving(40) + 39.0])
    assert len(nonidle_ranges(sig, idle_eps=1e-3)) == 1


def test_long_idle_run_splits_into_two_ranges():
    sig = np.vstack([_moving(40), _still(30, 39.0), _moving(40) + 39.0])
    r = nonidle_ranges(sig, idle_eps=1e-3)
    assert len(r) == 2
    assert r[0][1] <= 40 and r[1][0] >= 69


def test_length_test_uses_the_untrimmed_span():
    """A run of exactly min_non_idle_len survives and yields (len - trim) frames.

    If the trim were applied BEFORE the length test, this range would be dropped.
    """
    sig = np.vstack([_still(20), _moving(16, 1.0) + 1.0, _still(20, 16.0)])
    r = nonidle_ranges(sig, idle_eps=1e-3, min_non_idle_len=16, trim_last_n=10)
    assert len(r) == 1
    assert (r[0][1] - r[0][0]) == 6


def test_too_short_a_motion_burst_is_dropped():
    sig = np.vstack([_still(20), _moving(8, 1.0) + 1.0, _still(20, 8.0)])
    assert nonidle_ranges(sig, idle_eps=1e-3, min_non_idle_len=16) == []


def test_fully_idle_episode_yields_nothing():
    assert nonidle_ranges(_still(200), idle_eps=1e-3) == []


def test_ranges_are_disjoint_and_ordered():
    rng = np.random.default_rng(0)
    sig = np.cumsum(rng.normal(size=(600, 3)) * (rng.random((600, 1)) > 0.5), axis=0)
    r = nonidle_ranges(sig, idle_eps=1e-3)
    assert all(a < b for a, b in r)
    assert all(r[i][1] <= r[i + 1][0] for i in range(len(r) - 1))



@pytest.mark.parametrize("n", [0, 1, 2])
def test_degenerate_lengths_do_not_raise(n):
    assert nonidle_ranges(_still(n)) == []


def test_accepts_1d_signal():
    assert nonidle_ranges(np.arange(100, dtype=float), idle_eps=1e-3) == [(0, 90)]
