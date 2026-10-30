"""Fig. 4(b): Spearman's rho with average ranks for ties, and a paired-episode bootstrap whose
generator stream includes the auxiliary draw of the original analysis."""
from __future__ import annotations

import math

import pytest

from axis_qvla.paper import fig4


def test_spearman_uses_average_ranks():
    assert fig4.spearman([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(4.5 / math.sqrt(4.5 * 5), abs=1e-15)
    assert fig4.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert math.isnan(fig4.spearman([1, 1, 1], [1, 2, 3]))


def test_the_bootstrap_is_deterministic_and_the_auxiliary_draw_matters():
    p = {"r": [0.1 * i + 0.05 * (i % 3) for i in range(20)]}
    h = [0.07 * i + 0.1 * (i % 4) for i in range(20)]
    a = fig4.paired_bootstrap(p, h, n_boot=300, seed=42, aux_draw=18)
    assert a == fig4.paired_bootstrap(p, h, n_boot=300, seed=42, aux_draw=18)
    assert a != fig4.paired_bootstrap(p, h, n_boot=300, seed=42, aux_draw=0)
    lo, hi, na = a["r"]
    assert lo < fig4.spearman(p["r"], h) < hi and na == 0


def test_the_bootstrap_stream_is_pinned_for_the_fixed_seed():
    """Golden over numpy's `Generator` stream itself, not just this module's arithmetic: NEP 19
    does not guarantee `default_rng(seed).integers(...)` across numpy feature releases, so a future
    numpy could silently move every published interval while every self-consistency test here
    stays green (a reader's `--check` would be the first signal, off by about 1e-2). Holds on numpy
    1.26.4 (the `offline` extra's floor and the public CI's pin) and 2.4.4; if a numpy upgrade ever
    changes these, this is the test that must fail, not `fig4.expected.json`."""
    import numpy as np

    rng = np.random.default_rng(42)
    assert rng.integers(0, 20, 20)[:10].tolist() == [1, 15, 13, 8, 8, 17, 1, 13, 4, 1]
    assert rng.integers(0, 18, 18)[:10].tolist() == [9, 6, 3, 16, 14, 11, 7, 14, 9, 7]

    # The same two draws, consumed by `paired_bootstrap` itself: with n_boot=1 there is exactly one
    # resample, so its rho is pinned directly, through the real code path.
    p = {"r": [0.1 * i + 0.05 * (i % 3) for i in range(20)]}
    h = [0.07 * i + 0.1 * (i % 4) for i in range(20)]
    assert fig4.paired_bootstrap(p, h, n_boot=1, seed=42, aux_draw=18) == {"r": (0.928978954689393, 0.928978954689393, 0)}

    # A single resample passes even when the auxiliary draw's shape is wrong (e.g.
    # `rng.integers(0, aux_draw, n)` instead of `(0, aux_draw, aux_draw)`): the bug only shows once
    # a SECOND resample's `ic` starts at the wrong stream position. Pin n_boot=2 to catch it.
    assert fig4.paired_bootstrap(p, h, n_boot=2, seed=42, aux_draw=18) == \
        {"r": (0.9293229450691188, 0.9423945794987008, 0)}

    # And the toy interval `test_the_bootstrap_is_deterministic_and_the_auxiliary_draw_matters`
    # builds (n_boot=300), so that test's own assertions (that it reproduces, and that aux_draw=0
    # moves it) are themselves pinned against a numpy that moved the stream silently.
    assert fig4.paired_bootstrap(p, h, n_boot=300, seed=42, aux_draw=18) == \
        {"r": (0.8630393447053768, 0.986728205847778, 0)}


def test_fig4b_reads_only_the_published_study_fields():
    pe = {"object_centric": {(806, 0): 0.2, (806, 1): 0.5, (807, 0): 0.9},
          "approach_aware": {(806, 0): 0.3, (806, 1): 0.4, (807, 0): 0.8}}
    study = {"schema": "qvla.human_study/1",
             "params": {"n_boot": 50, "seed": 42, "aux_draws_per_resample": 0},
             "episodes": [{"task_id": 806, "episode_index": 0, "H": 0.1, "in_correlation_set": True},
                          {"task_id": 806, "episode_index": 1, "H": 0.6, "in_correlation_set": True},
                          {"task_id": 807, "episode_index": 0, "H": 0.95, "in_correlation_set": True}]}
    out = fig4.fig4b(pe, study)
    assert out["object_centric"]["N"] == 3 and out["object_centric"]["rho"] == pytest.approx(1.0)
