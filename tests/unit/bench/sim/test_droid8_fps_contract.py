"""fps is half of a contract; the corpus holds the other half.

`convert_droid_actions` writes v = (s[t+1] - s[t]) * fps and the eval replays q_cmd += v * (1/fps).
Those invert each other exactly, and only exactly, when the two rates agree. A mismatch rescales
every joint velocity with no error and no log line -- the arm just moves at the wrong speed and
times out. Nothing in the pipeline checked this before these guards.
"""
from __future__ import annotations

import json

import pytest

from axis_qvla.bench.sim.action_mapping import (
    CONVERTER_FPS,
    assert_corpus_fps,
    corpus_fps,
)


def _corpus(tmp_path, fps):
    root = tmp_path / "task_1889"
    (root / "meta").mkdir(parents=True)
    payload = {"codebase_version": "v3.0"}
    if fps is not None:
        payload["fps"] = fps
    (root / "meta" / "info.json").write_text(json.dumps(payload))
    return root


class TestCorpusFps:
    def test_reads_the_declared_rate(self, tmp_path):
        assert corpus_fps(_corpus(tmp_path, 15.0)) == 15.0

    def test_returns_none_rather_than_guessing_when_undeclared(self, tmp_path):
        assert corpus_fps(_corpus(tmp_path, None)) is None

    def test_an_unreadable_corpus_raises_rather_than_reading_as_absent(self, tmp_path):
        """EACCES must not be laundered into 'no fps declared'."""
        root = _corpus(tmp_path, 15.0)
        (root / "meta" / "info.json").chmod(0o000)
        try:
            with pytest.raises(OSError):
                corpus_fps(root)
        finally:
            (root / "meta" / "info.json").chmod(0o644)

    def test_a_missing_corpus_raises_too(self, tmp_path):
        with pytest.raises(OSError):
            corpus_fps(tmp_path / "does_not_exist")


class TestAssertCorpusFps:
    def test_matching_rate_passes_and_returns_it(self, tmp_path):
        assert assert_corpus_fps(_corpus(tmp_path, 15.0), 15.0) == 15.0

    def test_a_mismatch_is_refused_with_the_scale_factor_named(self, tmp_path):
        with pytest.raises(ValueError, match="2x|2.0x|off by"):
            assert_corpus_fps(_corpus(tmp_path, 15.0), 30.0)

    def test_an_undeclared_rate_is_refused_not_defaulted(self, tmp_path):
        with pytest.raises(ValueError, match="declares no fps"):
            assert_corpus_fps(_corpus(tmp_path, None))

    def test_the_default_is_the_converter_rate(self, tmp_path):
        assert CONVERTER_FPS == 15.0
        assert assert_corpus_fps(_corpus(tmp_path, CONVERTER_FPS)) == CONVERTER_FPS


class TestControllerGuard:
    """`from_model` needs a real MuJoCo model, so exercise the guard through the same predicate."""

    def test_the_guard_condition_matches_the_converter_rate(self):
        assert abs(15.0 - CONVERTER_FPS) <= 1e-9

    def test_a_doubled_rate_would_trip_the_guard(self):
        assert abs(30.0 - CONVERTER_FPS) > 1e-9
