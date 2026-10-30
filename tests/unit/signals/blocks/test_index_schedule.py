"""Tests for the filtered index-schedule maths, and the quality ranking CFG conditions on."""
from __future__ import annotations

import numpy as np

from axis_qvla.signals.blocks.schedule import (
    DEFAULT_KEEP_TOP_FRAC,
    FILTER_MODES,
    MODES,
    broadcast_episode_bins,
    random_fraction_mask,
    top_fraction_mask,
    delta_quality_bins,
    draw_schedule,
    N_BINS,
    NEUTRAL_Q,
    quality_bins,
)


def test_global_quintiles_cover_all_five_bins_with_balanced_counts():
    # n=100 is evenly divisible by N_BINS=5 with all-unique values, so counts are exact.
    progress_ratio = np.linspace(0.0, 1.0, 100)
    unlabelled = np.zeros(100, dtype=bool)
    q = quality_bins(progress_ratio, unlabelled)
    assert set(q.tolist()) == {1, 2, 3, 4, 5}
    counts = [int((q == b).sum()) for b in range(1, N_BINS + 1)]
    assert counts == [20, 20, 20, 20, 20]


def test_single_episode_is_not_forced_to_the_worst_bin():
    # Ranking is global, not within-task: a lone high-ratio episode lands in a high bin.
    progress_ratio = np.concatenate([np.full(19, 0.1), np.array([0.99])])
    unlabelled = np.zeros(20, dtype=bool)
    q = quality_bins(progress_ratio, unlabelled)
    assert q[-1] == 5


def test_ties_are_broken_by_position_not_value():
    # Ties are broken deterministically by stable sort (array position), not by value equality.
    # This means a tied pair CAN straddle a bin boundary. The guarantee is DETERMINISM
    # (reproducible byte-for-byte for a fixed input array), not tie-consistency.
    # Counter-example: [0.01, 0.01, 0.3, 0.4, 0.5] with n=5 bins gives ranks [0,1,2,3,4]
    # and bins [1,2,3,4,5], so the two tied 0.01 values get bins 1 and 2 respectively.
    progress_ratio = np.array([0.01, 0.01, 0.3, 0.4, 0.5])
    unlabelled = np.zeros(5, dtype=bool)
    q = quality_bins(progress_ratio, unlabelled)
    assert q[0] == 1
    assert q[1] == 2


def test_binning_is_deterministic_across_repeated_calls():
    # Regression guard: quality_bins is a pure function with no RNG or global state.
    # This test protects against a future edit introducing in-place sorting or hidden state.
    # It holds trivially today, but its value is as a regression detector.
    rng = np.random.default_rng(0)
    progress_ratio = rng.random(37)
    unlabelled = np.zeros(37, dtype=bool)
    q1 = quality_bins(progress_ratio, unlabelled)
    q2 = quality_bins(progress_ratio, unlabelled)
    assert np.array_equal(q1, q2)


def test_unlabelled_does_not_consume_a_quantile_slot():
    # Assert the full labelled bin set: a ranking that also ranks the unlabelled entry drops one
    # bin and duplicates another, which a weaker check passes.
    progress_ratio = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    unlabelled = np.array([False, False, True, False, False, False])
    q = quality_bins(progress_ratio, unlabelled)
    assert q[2] == NEUTRAL_Q
    labelled_bins = q[~unlabelled]
    assert set(labelled_bins.tolist()) == {1, 2, 3, 4, 5}


import pytest

from axis_qvla.signals.blocks.schedule import draw_schedule


def _toy(n_ep: int = 20, rows_per_ep: int = 50):
    """n_ep episodes, contiguous disjoint row blocks, q_ep cycling 1..5.

    Every episode has the SAME row count. That makes this fixture blind to the
    episode-vs-row tilt distinction: `draw_schedule` tilts PER ROW, but with equal-length
    episodes an episode-level tilt would produce the identical row probabilities. A test that
    needs to separate the two schemes (see `test_terminal_tilt_is_per_row_not_per_episode`
    below) must build episodes of DIFFERENT lengths instead.
    """
    episode_rows = [np.arange(i * rows_per_ep, (i + 1) * rows_per_ep) for i in range(n_ep)]
    q_ep = np.array([(i % 5) + 1 for i in range(n_ep)], dtype=np.int64)
    return episode_rows, q_ep


def test_schedule_shape_and_dtype():
    rows, _ = _toy()
    s = draw_schedule(rows, mode="vanilla", total_steps=100, batch=8, seed=0)
    assert s.shape == (100, 8) and s.dtype == np.int64


def test_every_index_is_inside_a_trainable_range():
    rows, _ = _toy()
    allowed = set(np.concatenate(rows).tolist())
    s = draw_schedule(rows, mode="vanilla", total_steps=200, batch=8, seed=0)
    assert set(np.unique(s).tolist()) <= allowed


def test_same_seed_is_byte_identical_and_different_seed_is_not():
    rows, _ = _toy()
    a = draw_schedule(rows, mode="vanilla", total_steps=50, batch=4, seed=7)
    b = draw_schedule(rows, mode="vanilla", total_steps=50, batch=4, seed=7)
    c = draw_schedule(rows, mode="vanilla", total_steps=50, batch=4, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_a_full_pass_over_the_drawable_rows_repeats_nothing():
    """The confound this pins: a with-replacement draw instead of a permutation.

    A permutation-based pass sees each drawable row exactly once; a with-replacement draw of the
    same size sees only ~63% of them. The arms are compared against a round-1 control that samples
    with `torch.randperm` (`slb_variant_sampler.RowSampler`), so a with-replacement schedule makes
    the arms differ from the control in distinct-frames-seen as well as in supervision -- which is
    exactly what an underperforming arm would then be explainable by. 20 x 50 = 1,000 drawable
    rows and 125 x 8 = 1,000 draws, so a correct sampler emits a permutation and nothing else.
    """
    rows, _ = _toy(n_ep=20, rows_per_ep=50)
    flat = np.concatenate(rows)
    s = draw_schedule(rows, mode="vanilla", total_steps=125, batch=8, seed=0)
    drawn = s.ravel()
    assert len(np.unique(drawn)) == drawn.size == flat.size
    assert set(drawn.tolist()) == set(flat.tolist())


def test_drop_covers_its_whole_drawable_set_over_a_pass_and_repermutes_after():
    """`drop` draws over a smaller set, so its budget is >1 pass -- the second pass must be a
    fresh permutation, not with-replacement padding, or the coverage confound returns for exactly
    the two arms whose budget exceeds one epoch (1.21 and 1.16 epochs on the real corpus)."""
    rows, _ = _toy(n_ep=10, rows_per_ep=30)
    keep = [r[r % 2 == 0] for r in rows]              # 150 drawable rows
    s = draw_schedule(keep, mode="drop", total_steps=30, batch=8, seed=0)   # 240 draws
    drawn = s.ravel()
    assert set(drawn.tolist()) == set(np.concatenate(keep).tolist())   # 100% coverage
    first_pass = drawn[:150]
    assert len(np.unique(first_pass)) == 150
    counts = np.bincount(drawn)
    assert counts.max() == 2, "a row was drawn 3x in 1.6 passes: this is not a permutation stream"


def test_drop_mode_never_emits_a_filtered_row():
    rows, _ = _toy(n_ep=10, rows_per_ep=30)
    keep = [r[r % 2 == 0] for r in rows]          # keep only even rows
    s = draw_schedule(keep, mode="drop", total_steps=100, batch=8, seed=0)
    assert (s % 2 == 0).all()


def test_unknown_mode_raises():
    rows, _ = _toy()
    with pytest.raises(ValueError, match="unknown mode"):
        draw_schedule(rows, mode="nonsense", total_steps=10, batch=2, seed=0)


# --- sample-level bins (WVM Eq E.4 is per SAMPLE, so the CFG quality tags rank rows, not episodes) ---


def test_delta_bins_populations_are_exact_even_with_a_huge_atom():
    """Rank, not threshold. A `>= percentile` test lands ON an atom and admits every tie with it.

    Ranking keeps every bin's population exact, whatever the ties do.
    """
    # 60% of rows sit at exactly 0.0 -- a flat reward horizon, which is the common case here.
    delta = [np.array([0.0] * 60 + [0.1 * i for i in range(1, 41)], dtype=np.float64)]
    bins = delta_quality_bins(delta)[0]
    counts = [int((bins == b).sum()) for b in range(1, N_BINS + 1)]
    assert counts == [20, 20, 20, 20, 20], counts
    assert set(np.unique(bins).tolist()) == {1, 2, 3, 4, 5}


def test_the_cap_atom_lands_wholly_inside_the_top_bin():
    """Why the row-level rank needs no reward recomputation to be sample-level.

    Ranking rows by `w` equals ranking by `Delta` except inside the `w == cap` atom, where the cap
    destroyed the magnitudes. That only matters if the atom STRADDLES a bin edge. On the real
    corpus the atom is the top 13.56% of rows and the top bin is the top 20%, so it cannot -- and
    no bin assignment depends on breaking those unbreakable ties.
    """
    n = 10_000
    n_cap = 1356  # 13.56%, the measured share of rows pinned at w == cap == 2.0
    w = np.concatenate([np.linspace(0.5, 1.99, n - n_cap), np.full(n_cap, 2.0)])
    bins = delta_quality_bins([w])[0]
    # every capped row is in the top bin ...
    assert set(np.unique(bins[-n_cap:]).tolist()) == {N_BINS}
    # ... and the top bin is strictly larger than the atom, so the edge is decided by real values
    assert int((bins == N_BINS).sum()) == n // N_BINS > n_cap


def test_two_episodes_with_equal_episode_quality_get_different_row_bins():
    """The tilt discriminates WITHIN a trajectory.

    Both episodes have the same per-episode progress ratio (half their rows advance), so an
    episode-level score gives them one identical bin. Their advantage MAGNITUDES differ sharply,
    which is what a per-row ranking sees and an episode-level one cannot.
    """
    mild = np.array([0.0] * 5 + [0.01] * 5, dtype=np.float64)
    steep = np.array([0.0] * 5 + [5.00] * 5, dtype=np.float64)
    got = delta_quality_bins([mild, steep])
    assert got[0].max() < got[1].max(), (got[0], got[1])
    # the episode-level view really is blind here: identical progress ratios
    assert float((mild > 0).mean()) == float((steep > 0).mean()) == 0.5


def test_delta_bins_refuse_a_non_finite_advantage():
    import pytest

    with pytest.raises(ValueError, match="non-finite"):
        delta_quality_bins([np.array([0.0, np.nan, 1.0])])


def test_broadcast_reproduces_the_episode_level_tilt_exactly():
    """The episode-level broadcast is behaviour-preserving against hand-derived output."""
    rows = [np.arange(0, 4), np.arange(4, 10), np.arange(10, 12)]
    q_ep = np.array([1, 5, 3], dtype=np.int64)
    got = broadcast_episode_bins(q_ep, rows)
    assert [g.tolist() for g in got] == [[1] * 4, [5] * 6, [3] * 2]


def test_broadcast_episode_bins_must_match_episode_rows_one_for_one():
    """A length mismatch would shift every later episode's bin with no error -- so it must raise."""
    import pytest

    rows = [np.arange(0, 4), np.arange(4, 10)]
    with pytest.raises(ValueError, match="1:1 by position"):
        broadcast_episode_bins(np.array([1], dtype=np.int64), rows)


# --- the selective cut (WVM Eq E.6 percentile variant, at the fraction this corpus can express) ---


def test_top_fraction_keeps_exactly_the_asked_share_despite_a_huge_atom():
    """Ranked, not thresholded -- the same trap as `delta_quality_bins`, on the keep rule."""
    # 700 rows tied at exactly 1.0, then 300 strictly greater. `range(1, 301)` on purpose: at i=0
    # the row would also be exactly 1.0, making the atom 701 wide, and the 300th kept row would
    # then legitimately come out of the atom by tie-break rather than by value.
    w = [np.array([1.0] * 700 + [1.0 + 0.001 * i for i in range(1, 301)], dtype=np.float64)]
    keep = top_fraction_mask(w, 0.30)[0]
    assert int(keep.sum()) == 300
    # a `w >= quantile(w, 70)` test would admit the whole 700-row atom at 1.0 instead
    assert keep[:700].sum() == 0
    assert keep[700:].all()


def test_top_fraction_is_global_not_per_episode():
    """WVM ranks over the TRAINING SET, so a uniformly-weak episode can be emptied entirely.

    This is the documented cost of following the paper on a 731-task corpus, and it is asserted
    rather than left implicit: a per-episode cut would keep the best 30% of BOTH episodes.
    """
    strong = np.full(100, 1.9, dtype=np.float64)
    weak = np.full(100, 1.01, dtype=np.float64)
    keep = top_fraction_mask([strong, weak], 0.30)
    assert int(keep[0].sum()) == 60   # 60 of the 60 kept rows come from the strong episode
    assert int(keep[1].sum()) == 0    # the weak episode is emptied -- global rank, not per episode


def test_top_fraction_refuses_a_non_finite_weight_and_a_bad_fraction():
    import pytest

    with pytest.raises(ValueError, match="non-finite"):
        top_fraction_mask([np.array([1.0, np.nan])], 0.5)
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="keep_frac"):
            top_fraction_mask([np.array([1.0, 2.0])], bad)


def test_top_fraction_of_one_keeps_everything():
    w = [np.array([1.0, 2.0]), np.array([1.5])]
    assert [m.tolist() for m in top_fraction_mask(w, 1.0)] == [[True, True], [True]]


def test_drop_top_is_a_known_mode_and_a_filter_mode():
    assert "drop_top" in MODES
    assert "drop_top" in FILTER_MODES
    # 30%, not WVM's 70%: at 75.7% advantaged, a top-70% cut is nearly the binary arm.
    # 0.50: equal coverage between rewards (659,892 rows each,
    # 50.00%) and Jaccard(kept_v2, kept_phase) 0.887 -> 0.604, so the two rewards stop
    # being nearly the same intervention. See index_schedule.DEFAULT_KEEP_TOP_FRAC.
    assert DEFAULT_KEEP_TOP_FRAC == 0.50


# --- the control that makes drop_top interpretable -------------------------------------------


def test_random_keeps_exactly_as_many_rows_as_top_for_the_same_fraction():
    """THE property that makes it a control. If these ever diverge it stops controlling, silently."""
    w = [np.linspace(0.5, 2.0, 700), np.linspace(0.5, 2.0, 300)]
    for frac in (0.1, 0.3, 0.5, 0.7, 1.0):
        n_top = sum(int(m.sum()) for m in top_fraction_mask(w, frac))
        n_rnd = sum(int(m.sum()) for m in random_fraction_mask(w, frac, seed=0))
        assert n_top == n_rnd == round(frac * 1000), (frac, n_top, n_rnd)


def test_random_selection_ignores_the_weights():
    """It must not accidentally prefer high-advantage rows, or it is a second drop_top."""
    n = 4000
    w = [np.concatenate([np.full(n // 2, 0.6), np.full(n // 2, 1.9)])]
    keep = random_fraction_mask(w, 0.3, seed=1)[0]
    high = int(keep[n // 2:].sum())
    # a uniform 30% draw puts ~600 of its 1200 picks in each half; drop_top would put ALL 1200 high
    assert 500 < high < 700, high
    assert int(top_fraction_mask(w, 0.3)[0][n // 2:].sum()) == 1200


def test_random_is_deterministic_for_a_seed_and_varies_across_seeds():
    w = [np.linspace(0.5, 2.0, 1000)]
    a = random_fraction_mask(w, 0.3, seed=5)[0]
    b = random_fraction_mask(w, 0.3, seed=5)[0]
    c = random_fraction_mask(w, 0.3, seed=6)[0]
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_random_rejects_a_bad_fraction():
    import pytest

    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="keep_frac"):
            random_fraction_mask([np.ones(10)], bad, seed=0)


def test_drop_random_is_a_known_filter_mode():
    assert "drop_random" in MODES and "drop_random" in FILTER_MODES
