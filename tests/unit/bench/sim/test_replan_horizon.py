"""How many chunk actions run OPEN-LOOP before re-observing must be configurable, and default to 8.

Why it matters for a delta action space: eef7 commands per-step position deltas, so a longer
open-loop window accumulates directional error unobserved; joint-velocity commands tolerate it well.

The default is 8, the value every reported evaluation used.
"""

from __future__ import annotations

import pytest

from axis_qvla.bench.sim.replan import effective_replan as _effective_replan


def test_unset_keeps_the_callers_default(monkeypatch):
    monkeypatch.delenv("AXIS_EVAL_REPLAN", raising=False)
    assert _effective_replan(8) == 8
    assert _effective_replan(4) == 4


def test_blank_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv("AXIS_EVAL_REPLAN", "   ")
    assert _effective_replan(8) == 8


@pytest.mark.parametrize("value,expected", [("1", 1), ("2", 2), ("16", 16), (" 4 ", 4)])
def test_override_is_honoured(monkeypatch, value, expected):
    monkeypatch.setenv("AXIS_EVAL_REPLAN", value)
    assert _effective_replan(8) == expected


def test_zero_is_refused_rather_than_silently_hanging(monkeypatch):
    """replan=0 would extend the plan with nothing and re-infer forever on the same observation."""
    monkeypatch.setenv("AXIS_EVAL_REPLAN", "0")
    with pytest.raises(ValueError, match="must be >= 1"):
        _effective_replan(8)


def test_negative_is_refused(monkeypatch):
    monkeypatch.setenv("AXIS_EVAL_REPLAN", "-3")
    with pytest.raises(ValueError, match="must be >= 1"):
        _effective_replan(8)


def test_non_integer_is_refused_not_silently_ignored(monkeypatch):
    """A typo must fail loudly; silently reverting to 8 would make an A/B report a null result."""
    monkeypatch.setenv("AXIS_EVAL_REPLAN", "two")
    with pytest.raises(ValueError, match="not an integer"):
        _effective_replan(8)
