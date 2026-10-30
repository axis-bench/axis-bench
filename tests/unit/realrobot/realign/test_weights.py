"""The curve-input weights must be the pinned builder's weights, byte for byte.

`build_awr_weights` computes the phase curve from corpus columns and then weights it; REALALIGN's curves
arrive with the corpus. Running the pinned builder and the curve path on ONE fixture, with the curves the
pinned builder itself forms, is what makes "the same tool code as the sim benchmark" a tested claim.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from axis_qvla.signals.methods import awr_chunk as baw
from axis_qvla.signals.blocks.awr import AwrWeightError
from axis_qvla.realrobot.realign.curves import PhaseCurve
from axis_qvla.realrobot.realign.weights import balanced_weights, rank_weights, write_weights
from tests.golden.signals.test_paper_signal_rules import META, _phase_corpus


def _fixture_curves(rows) -> dict:
    out = {}
    for e, g in enumerate(rows):
        curve, t0 = baw.phase_reward(g.drop(columns=["episode_index", "frame_index"]), META)
        out[(900, e)] = PhaseCurve(np.asarray(curve, float), g["reward_valid"].to_numpy(bool), int(t0))
    return out


def _pinned(tmp_path, monkeypatch, *extra):
    idx, rows = _phase_corpus(tmp_path)
    out = tmp_path / "pinned.json"
    monkeypatch.setattr("sys.argv", ["x", "--reward", "phase", "--binding", "names", "--index", str(idx),
                                     "--out", str(out), *extra])
    assert baw.main() == 0
    return json.loads(out.read_text()), rows


def _assert_same(eps, doc):
    got = {e.key: e.weights for e in eps}
    assert set(got) == set(doc["weights"])
    for k, w in doc["weights"].items():
        np.testing.assert_array_equal(got[k], np.asarray(w, np.float32))


def test_balanced_weights_equal_the_pinned_phase_balanced_builder(tmp_path, monkeypatch):
    doc, rows = _pinned(tmp_path, monkeypatch, "--phase-balanced")
    eps, scales = balanced_weights(_fixture_curves(rows))
    _assert_same(eps, doc)
    assert scales == doc["provenance"]["phase_balance"]


def test_rank_weights_equal_the_pinned_unbalanced_builder(tmp_path, monkeypatch):
    doc, rows = _pinned(tmp_path, monkeypatch)
    _assert_same(rank_weights(_fixture_curves(rows)), doc)


def test_a_neutral_episode_is_all_ones_in_both_files(tmp_path):
    _, rows = _phase_corpus(tmp_path)
    curves = _fixture_curves(rows)
    curves[(900, 999)] = PhaseCurve(np.zeros(50, np.float32), np.zeros(50, bool), None)
    for eps in (rank_weights(curves), balanced_weights(curves)[0]):
        n = {e.key: e for e in eps}["task_900--999"]
        assert (n.weights == 1.0).all() and n.n_neutral == 50


def test_float32_progress_semantics_are_kept():
    """A positive Delta below ~6e-9 rounds to w == 1.0f: progress is read on the float32 weight."""
    curve = np.r_[np.zeros(20), np.cumsum(np.full(30, 1e-12))]
    (w,) = rank_weights({(1, 0): PhaseCurve(curve, np.ones(50, bool), 20)})
    assert w.weights.dtype == np.float32 and (w.weights == 1.0).all()


def test_too_few_frames_to_fit_the_phase_scales_are_refused():
    curves = {(1, e): PhaseCurve(np.linspace(0, 1, 40), np.ones(40, bool), 20) for e in range(5)}
    with pytest.raises(AwrWeightError, match="pools too small"):
        balanced_weights(curves)


def test_write_weights_stamps_the_curve_binding_and_the_scales(tmp_path, monkeypatch):
    doc, rows = _pinned(tmp_path, monkeypatch, "--phase-balanced")
    eps, scales = balanced_weights(_fixture_curves(rows))
    out = tmp_path / "w.json"
    write_weights(out, eps, corpus=tmp_path / "corpus", balanced=True, scales=scales)
    got = json.loads(out.read_text())
    assert got["reward_id"] == "phase" and got["n_rows"] == doc["n_rows"]
    assert got["provenance"]["params"]["binding"] == "curves"
    assert got["provenance"]["params"]["phase_balanced"] is True
    assert got["provenance"]["phase_balance"] == scales
    assert got["provenance"]["corpus_fingerprint"] == doc["provenance"]["corpus_fingerprint"]
    with pytest.raises(ValueError, match="scales"):
        write_weights(out, eps, corpus=tmp_path / "corpus", balanced=True, scales=None)


def test_a_small_corpus_builds_with_an_explicit_lower_fit_pool_and_says_so():
    # 85 fit frames per phase, pooled capped rate 0.47 (checked against the pinned helpers)
    curve = np.r_[np.linspace(0.0, 0.03, 20), 0.03 + 0.5 * np.linspace(0.0, 1.0, 20) ** 3]
    curves = {(1, e): PhaseCurve(curve, np.ones(40, bool), 20) for e in range(5)}
    with pytest.raises(AwrWeightError, match="pools too small"):
        balanced_weights(curves)
    eps, scales = balanced_weights(curves, min_pool=50)
    assert len(eps) == 5 and scales["min_fit_pool"] == 50


def test_the_default_fit_pool_leaves_the_scales_unchanged(tmp_path, monkeypatch):
    doc, rows = _pinned(tmp_path, monkeypatch, "--phase-balanced")
    _eps, scales = balanced_weights(_fixture_curves(rows))
    assert "min_fit_pool" not in scales
