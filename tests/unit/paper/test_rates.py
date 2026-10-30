"""The arithmetic behind every paper number: SR, SE, exact differences and ratios, mean and SD."""
from __future__ import annotations

import math
from fractions import Fraction

from axis_qvla.paper.pool import pool_cell
from axis_qvla.paper.rates import diff, exact, fmt, mean, norm_diff, rel, sample_sd, scaled, se, se_diff, shown, sign, sr


def _P() -> dict:
    # A and B have the same success count but their per-task rates are summed in a different
    # order, so the float aggregates differ in the last bit; every exact quantity still ties.
    return {"a|bare": pool_cell("ra", {1: (1, 10), 2: (2, 10), 3: (3, 10)}),
            "b|bare": pool_cell("rb", {1: (3, 10), 2: (2, 10), 3: (1, 10)}),
            "c|bare": pool_cell("rc", {1: (4, 10), 2: (2, 10), 3: (1, 10)})}


def test_sr_is_the_exact_rate_rounded_once_and_se_the_pooled_cells():
    """SR is exact (the pooled float `agg` carries a last-bit residue, 20.000000000000004
    here); SE is the pooled cell's task-stratified SE, and a difference's SE combines two."""
    P = _P()
    assert sr(P, "a|bare") == float(100 * exact(P, "a|bare")) == 20.0 != P["a|bare"]["agg"]
    assert se(P, "a|bare") == P["a|bare"]["se"]
    assert se_diff(P, "a|bare", "c|bare") == math.sqrt(P["a|bare"]["se"] ** 2 + P["c|bare"]["se"] ** 2)
    assert se_diff(P, "a|bare", "absent|bare") is None
    assert sr(P, "absent|bare") is None and se(P, "absent|bare") is None and exact(P, "absent|bare") is None
    assert exact(P, "c|bare") == Fraction(7, 30)


def test_differences_ratios_and_signs_are_exact_where_the_floats_are_not():
    P = _P()
    assert P["a|bare"]["agg"] != P["b|bare"]["agg"]       # 0.6000000000000001 vs 0.6 (x 100 / 3)
    assert sr(P, "a|bare") == sr(P, "b|bare")            # the exact rates tie, and so do their floats
    assert sign(P, "a|bare", "b|bare") == 0 and diff(P, "a|bare", "b|bare") == 0.0
    assert rel(P, "a|bare", "b|bare") == 0.0 and norm_diff(P, "a|bare", "b|bare", "c|bare") == 0
    assert sign(P, "c|bare", "a|bare") == 1 and sign(P, "a|bare", "c|bare") == -1
    assert diff(P, "c|bare", "a|bare") == float(Fraction(100, 30))
    assert rel(P, "c|bare", "a|bare") == float(100 * (Fraction(7, 6) - 1))
    assert norm_diff(P, "c|bare", "a|bare", "b|bare") == 100 * Fraction(1, 30) / Fraction(1, 5)
    assert sign(P, "a|bare", "absent|bare") is None and diff(P, "a|bare", "absent|bare") is None
    assert rel(P, "absent|bare", "a|bare") is None and norm_diff(P, "a|bare", "b|bare", "absent|bare") is None


def test_mean_and_sample_sd_are_exact_until_the_square_root():
    xs = [Fraction(1), Fraction(2), Fraction(4)]
    assert mean(xs) == Fraction(7, 3)
    assert sample_sd([Fraction(k) for k in (1, 2, 3, 4)]) == math.sqrt(5 / 3)
    assert mean([]) is None and sample_sd([Fraction(1)]) is None
    assert mean([Fraction(1), None]) is None and sample_sd([Fraction(1), None, Fraction(2)]) is None
    assert scaled(1.5, 50.0) == 3.0 and scaled(None, 1.0) is None and scaled(1.0, 0.0) is None


def test_fmt_prints_the_papers_way():
    assert fmt(1.0166, 2, signed=True) == "+1.02" and fmt(-3.916, 2, signed=True) == "-3.92"
    assert fmt(-1e-13, 1, signed=True) == "0.0" and fmt(0.04, 1, signed=True) == "0.0"
    assert fmt(57.4166, 2) == "57.42" and fmt(None, 2) == "—"
    assert shown("-3.92") == "−3.92" and shown("+1.02") == "+1.02" and shown("0.0") == "0.0" and shown("—") == "—"
