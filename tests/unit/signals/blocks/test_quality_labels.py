"""Tests for the π0.7 quality-tag maths.

Two properties are load-bearing and neither is visible in any training log.

The FIRST is that dropout is TWO draws, not one. π0.7's scheme is 15% on the whole metadata
block plus 5% on the component, marginal 1 - 0.85*0.95 = 0.1925. A collapsed implementation --
one draw at 0.15, or one draw at 0.1925 -- produces a plausible tagged fraction and a
plausible-looking arm, and the only way to tell is to pin the draw ORDER against the raw
generator stream. Both draws are consumed unconditionally so that adding a `Mistake` component
later does not perturb the first component's stream.

The SECOND is `H(q|task)/H(q)`, measured separately at 0.729 (v2)
and 0.737 (phase) on the real corpus; this file pins the ESTIMATOR against cases whose answer is
known exactly, so its agreement with that measurement is a real cross-check rather than two
runs of the same possibly-wrong code.
"""
from __future__ import annotations

import numpy as np
import pytest

from axis_qvla.signals.blocks.tags import (
    DROP_COMPONENT,
    DROP_WHOLE,
    NO_TAG,
    NOT_TRAINABLE,
    TAGGED_FRACTION,
    dense_tags,
    q_given_task_entropy_ratio,
    two_level_dropout,
)


def test_the_two_levels_are_two_independent_draws_in_stream_order():
    """Catches: a collapsed one-draw scheme, or the component draw reading the whole draw's slot.

    Pinned against the raw generator stream rather than against a rate, because a single draw at
    0.1925 reproduces the marginal exactly and every rate test would pass.
    """
    n = 5000
    whole, component = two_level_dropout(n, seed=7)
    u = np.random.default_rng(7).random(2 * n)
    assert np.array_equal(whole, u[0::2] < DROP_WHOLE)
    assert np.array_equal(component, u[1::2] < DROP_COMPONENT)


def test_both_draws_are_consumed_even_where_the_block_was_already_dropped():
    """Catches: `if not drop_whole: drop_component = rng.random() < 0.05`, which makes the
    component stream depend on the whole stream and silently moves every later row."""
    whole, component = two_level_dropout(20_000, seed=3)
    assert whole.shape == component.shape == (20_000,)
    # The component draw must be statistically independent of the whole draw. Under the
    # conditional implementation, component is identically False wherever whole is True.
    assert component[whole].mean() > 0.02, "component draw was skipped where the block was dropped"


def test_realized_rates_match_the_two_levels_separately():
    """Catches: a single draw at the 0.1925 marginal (whole would then read 0.1925, not 0.15)."""
    whole, component = two_level_dropout(1_000_000, seed=0)
    assert abs(whole.mean() - DROP_WHOLE) < 0.002
    assert abs(component.mean() - DROP_COMPONENT) < 0.002
    assert abs((~(whole | component)).mean() - TAGGED_FRACTION) < 0.002


def test_dropout_is_deterministic_in_the_seed_and_sensitive_to_it():
    """Catches: a builder that ignores --seed. Plan A shipped a determinism test that ran one
    seed twice and would have passed with the seed argument deleted entirely."""
    a = two_level_dropout(1000, seed=11)
    b = two_level_dropout(1000, seed=11)
    c = two_level_dropout(1000, seed=12)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[0], c[0])


def test_the_rates_are_parameters_over_the_same_stream_and_zero_drops_nothing():
    """Catches: a rate that is accepted but not applied (the masks would still be the paper's), or
    one that shifts the stream (another rate would then move which rows the paper's rate drops)."""
    from axis_qvla.signals.blocks.tags import dense_tags_rowwise

    u = np.random.default_rng(5).random((1000, 2))
    whole, component = two_level_dropout(1000, seed=5, drop_whole=0.3, drop_component=0.1)
    assert np.array_equal(whole, u[:, 0] < 0.3) and np.array_equal(component, u[:, 1] < 0.1)
    whole, component = two_level_dropout(1000, seed=5, drop_whole=0.0, drop_component=0.0)
    assert not whole.any() and not component.any()
    for bad in (-0.1, 1.0):
        with pytest.raises(ValueError, match="must be in"):
            two_level_dropout(10, seed=0, drop_component=bad)
    blocks, q_ep, n_rows = _toy_blocks()
    flat = np.concatenate(blocks)
    tag, meta = dense_tags(blocks, q_ep, n_rows=n_rows, seed=0, drop_whole=0.0, drop_component=0.0)
    assert not (tag[flat] == NO_TAG).any() and meta["realized_tagged_fraction"] == 1.0
    tag, _ = dense_tags_rowwise(blocks, _toy_row_bins(blocks), n_rows=n_rows, seed=0, drop_whole=0.0,
                                drop_component=0.0)
    assert not (tag[flat] == NO_TAG).any()


def _toy_blocks(n_ep: int = 8, rows_per_ep: int = 10, stride: int = 25):
    """Disjoint ASCENDING blocks with gaps -- the gaps are the non-idle rows the corpus drops."""
    blocks = [np.arange(i * stride, i * stride + rows_per_ep, dtype=np.int64) for i in range(n_ep)]
    q_ep = np.array([(i % 5) + 1 for i in range(n_ep)], dtype=np.int64)
    return blocks, q_ep, n_ep * stride


def test_untrainable_rows_carry_the_sentinel_not_a_default():
    """Catches: `np.zeros(n_rows)` as the base array, which makes an untrainable row
    indistinguishable from a dropped-out one -- the 0.0-default pattern StrictWeightedRowDataset
    exists to refuse."""
    blocks, q_ep, n_rows = _toy_blocks()
    tag, _ = dense_tags(blocks, q_ep, n_rows=n_rows, seed=0)
    assert tag.dtype == np.uint8 and tag.shape == (n_rows,)
    trainable = np.concatenate(blocks)
    untrainable = np.setdiff1d(np.arange(n_rows), trainable)
    assert (tag[untrainable] == NOT_TRAINABLE).all()
    assert (tag[trainable] != NOT_TRAINABLE).all()


def test_every_tagged_row_carries_its_own_episode_quintile():
    """Catches: a per-ROW q, or an off-by-one in the block -> q_ep repeat."""
    blocks, q_ep, n_rows = _toy_blocks()
    tag, _ = dense_tags(blocks, q_ep, n_rows=n_rows, seed=0)
    for block, q in zip(blocks, q_ep, strict=True):
        vals = set(np.unique(tag[block]).tolist())
        assert vals <= {NO_TAG, int(q)}, f"episode with q={q} emitted {vals}"


def test_meta_reports_both_realized_rates_and_the_tagged_fraction():
    # stride=60 (> rows_per_ep=50): at the default stride=25 consecutive blocks would OVERLAP
    # (block i ends at i*25+49, block i+1 starts at i*25+25) and trip dense_tags' own
    # ascending-block guard. The wider stride keeps the blocks disjoint, per the fixture's own
    # docstring contract ("Disjoint ASCENDING blocks with gaps"), at the larger scale this checks.
    blocks, q_ep, n_rows = _toy_blocks(n_ep=400, rows_per_ep=50, stride=60)
    tag, meta = dense_tags(blocks, q_ep, n_rows=n_rows, seed=0)
    for key in ("realized_drop_whole", "realized_drop_component", "realized_tagged_fraction"):
        assert key in meta
    assert abs(meta["realized_tagged_fraction"] - TAGGED_FRACTION) < 0.01
    n_trainable = sum(len(b) for b in blocks)
    assert int((tag[np.concatenate(blocks)] != NO_TAG).sum()) == round(
        meta["realized_tagged_fraction"] * n_trainable
    )


def test_out_of_order_or_overlapping_blocks_raise():
    """Catches: an enumeration whose blocks are not ascending. The dropout stream is consumed in
    flat row order, so a re-ordered enumeration silently produces a DIFFERENT artifact from the
    same seed -- reproducibility that quietly is not."""
    blocks, q_ep, n_rows = _toy_blocks(n_ep=3)
    with pytest.raises(ValueError, match="ascending"):
        dense_tags(blocks[::-1], q_ep[:3], n_rows=n_rows, seed=0)
    with pytest.raises(ValueError, match="ascending"):
        dense_tags([blocks[0], blocks[0]], q_ep[:2], n_rows=n_rows, seed=0)


def test_a_bin_outside_one_to_five_raises():
    """Catches: a sentinel or a 0-based quintile leaking in. `Quality: 0` would collide with the
    NO_TAG code and turn a tagged row into an untagged one with no error."""
    blocks, _, n_rows = _toy_blocks(n_ep=3)
    with pytest.raises(ValueError, match="quality bin"):
        dense_tags(blocks, np.array([0, 1, 2]), n_rows=n_rows, seed=0)
    with pytest.raises(ValueError, match="quality bin"):
        dense_tags(blocks, np.array([1, 2, 6]), n_rows=n_rows, seed=0)


def test_a_row_beyond_the_corpus_raises():
    """Catches: a tag array built against a different corpus size -- the artifact would load and
    every index would mean a different frame."""
    blocks, q_ep, _ = _toy_blocks(n_ep=3)
    with pytest.raises(ValueError, match="outside the pretraining data"):
        dense_tags(blocks, q_ep[:3], n_rows=10, seed=0)


def test_entropy_ratio_is_one_when_quality_is_independent_of_task():
    q = np.array([1, 2, 3, 4, 5] * 4)
    task = np.repeat([10, 20, 30, 40], 5)
    assert abs(q_given_task_entropy_ratio(q, task) - 1.0) < 1e-12


def test_entropy_ratio_is_zero_when_quality_is_a_function_of_task():
    """The degenerate case the diagnostic exists to detect: the tag IS a task label."""
    q = np.array([1, 1, 1, 5, 5, 5])
    task = np.array([10, 10, 10, 20, 20, 20])
    assert q_given_task_entropy_ratio(q, task) == 0.0


def test_entropy_ratio_matches_a_hand_computed_case():
    """H(q) = 1 bit; task 10 holds {1,2} and task 20 holds {1,2}, so H(q|task) = 1 bit."""
    q = np.array([1, 2, 1, 2])
    task = np.array([10, 10, 20, 20])
    assert abs(q_given_task_entropy_ratio(q, task) - 1.0) < 1e-12
    # Half-explained: task 10 is pure, task 20 is balanced -> H(q|task) = 0.5 of H(q).
    q2 = np.array([1, 1, 1, 2])
    task2 = np.array([10, 10, 20, 20])
    h_q = -sum(p * np.log2(p) for p in (0.75, 0.25))
    assert abs(q_given_task_entropy_ratio(q2, task2) - (0.5 / h_q)) < 1e-12


def test_entropy_ratio_of_a_constant_quality_is_defined_and_raises_nothing_useful_silently():
    """H(q) = 0 makes the ratio 0/0. Returning 0.0 or NaN silently would be reported as
    'the tag is pure task identity', which is the opposite of what a constant tag means."""
    with pytest.raises(ValueError, match="constant"):
        q_given_task_entropy_ratio(np.array([3, 3, 3]), np.array([1, 1, 2]))


def test_only_a_constant_quality_raises_the_typed_refusal():
    """A consumer that accepts a constant-quality corpus recognises the H(q) = 0 refusal by its
    type, so that type must mark exactly that refusal: the function's other ValueErrors (a length
    mismatch, an empty population) are not it."""
    from axis_qvla.signals.blocks.tags import ConstantQualityError

    assert issubclass(ConstantQualityError, ValueError)
    with pytest.raises(ConstantQualityError, match="H\\(q\\) is zero"):
        q_given_task_entropy_ratio(np.array([3, 3, 3]), np.array([1, 1, 2]))
    for q, t, msg in ((np.array([1, 2]), np.array([1]), "quality bins but"), (np.array([]), np.array([]), "no episodes")):
        with pytest.raises(ValueError, match=msg) as exc:
            q_given_task_entropy_ratio(q, t)
        assert not isinstance(exc.value, ConstantQualityError)


# ---------------------------------------------------------------------------
# Row (segment) granularity -- the CFG granularity ablation's offline half.


def _toy_row_bins(blocks):
    """Bins that VARY within every episode, so a broadcast bug cannot hide."""
    return [((np.arange(len(b)) % 5) + 1).astype(np.int64) for b in blocks]


def test_rowwise_tags_carry_each_rows_own_bin():
    """Catches: a rowwise entry point that quietly aggregates to per-episode and broadcasts."""
    from axis_qvla.signals.blocks.tags import dense_tags_rowwise

    blocks, _, n_rows = _toy_blocks()
    row_bins = _toy_row_bins(blocks)
    tag, _ = dense_tags_rowwise(blocks, row_bins, n_rows=n_rows, seed=0)
    for block, bins in zip(blocks, row_bins, strict=True):
        got = tag[block]
        keep = got != NO_TAG
        assert (got[keep] == np.asarray(bins, dtype=np.uint8)[keep]).all()
    multi = [len(set(tag[b][tag[b] != NO_TAG].tolist())) > 1 for b in blocks]
    assert any(multi), "no episode carries more than one bin -- the rowwise path broadcast"


def test_rowwise_and_episode_streams_drop_exactly_the_same_rows():
    """The ablation's identity: at one seed the two granularities tag and drop the SAME rows and
    differ only in WHICH bin a tagged row carries. A rowwise path that drew its own stream would
    change the tagged-row population too, making the comparison two experiments in one name."""
    from axis_qvla.signals.blocks.tags import dense_tags_rowwise

    blocks, q_ep, n_rows = _toy_blocks()
    row_bins = _toy_row_bins(blocks)
    ep_tag, ep_meta = dense_tags(blocks, q_ep, n_rows=n_rows, seed=7)
    row_tag, row_meta = dense_tags_rowwise(blocks, row_bins, n_rows=n_rows, seed=7)
    assert ((ep_tag == NO_TAG) == (row_tag == NO_TAG)).all()
    assert ((ep_tag == NOT_TRAINABLE) == (row_tag == NOT_TRAINABLE)).all()
    assert ep_meta["realized_tagged_fraction"] == row_meta["realized_tagged_fraction"]


def test_rowwise_block_count_and_length_mismatches_raise():
    from axis_qvla.signals.blocks.tags import dense_tags_rowwise

    blocks, _, n_rows = _toy_blocks()
    row_bins = _toy_row_bins(blocks)
    with pytest.raises(ValueError, match="bin blocks"):
        dense_tags_rowwise(blocks, row_bins[:-1], n_rows=n_rows, seed=0)
    slipped = [b.copy() for b in row_bins]
    slipped[2] = slipped[2][:-1]
    with pytest.raises(ValueError, match="rows but"):
        dense_tags_rowwise(blocks, slipped, n_rows=n_rows, seed=0)


def test_rowwise_bin_outside_one_to_five_raises():
    from axis_qvla.signals.blocks.tags import dense_tags_rowwise

    blocks, _, n_rows = _toy_blocks()
    for bad_value in (0, 6):
        row_bins = _toy_row_bins(blocks)
        row_bins[0][0] = bad_value
        with pytest.raises(ValueError, match="quality bin outside"):
            dense_tags_rowwise(blocks, row_bins, n_rows=n_rows, seed=0)
