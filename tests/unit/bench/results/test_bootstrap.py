"""Per-task Wilson score intervals, exact at 0/n and n/n; a deterministic seed bootstrap of the
all-24 mean that narrows with more seeds."""
from __future__ import annotations

import pytest

from axis_qvla.bench.results.bootstrap import LEVEL, N_BOOT, SEED, Z95, bootstrap, wilson, z_for

# The roots of the Wilson score-test quadratic (n + z^2) p^2 - (2k + z^2) p + k^2/n = 0 at n = 50,
# z = 1.959963984540054, solved in 50-digit decimal arithmetic: an independent derivation of the
# closed form `wilson` evaluates. Clopper-Pearson for 1/50 is [0.05, 10.65]: Wilson is not exact,
# but neither collapses.
WILSON_50 = {0: (0.0, 7.134759913336), 1: (0.353925927165, 10.495443589638),
             25: (36.644514316829, 63.355485683171), 50: (92.865240086664, 100.0)}


def test_the_defaults_are_the_published_ones():
    assert (N_BOOT, SEED, LEVEL) == (10_000, 7, 0.95) and Z95 == pytest.approx(1.959964, abs=1e-6)


@pytest.mark.parametrize("k", sorted(WILSON_50))
def test_the_wilson_interval_matches_the_score_test_roots(k):
    lo, hi = wilson(k, 50)
    assert (lo, hi) == pytest.approx(WILSON_50[k], abs=1e-9)


def test_a_task_at_the_floor_or_the_ceiling_keeps_an_interval():
    assert wilson(0, 50)[0] == 0.0 and wilson(0, 50)[1] > 7.0
    assert wilson(50, 50)[1] == 100.0 and wilson(50, 50)[0] < 93.0
    iv = bootstrap({806: (50, 50), 1050: (0, 50), 1459: (1, 50)}, n_boot=200)
    assert iv.per_task == {1050: wilson(0, 50), 1459: wilson(1, 50), 806: wilson(50, 50)}


def test_the_interval_is_deterministic_and_contains_the_estimate():
    t = {806: (20, 50), 1050: (35, 50), 1223: (2, 50)}
    a = bootstrap(t, n_boot=500)
    assert a == bootstrap(t, n_boot=500) and a != bootstrap(t, n_boot=500, seed=8)
    est = 100 * (20 / 50 + 35 / 50 + 2 / 50) / 3
    assert a.pooled[0] < est < a.pooled[1]
    assert all(a.per_task[k][0] <= 100 * s / n <= a.per_task[k][1] for k, (s, n) in t.items())
    assert list(a.per_task) == [1050, 1223, 806]          # pooled (task-id string) order
    assert (a.n_boot, a.seed, a.level) == (500, SEED, 0.95)


def test_more_seeds_give_a_narrower_interval():
    narrow, wide = bootstrap({1: (100, 200)}, n_boot=500), bootstrap({1: (10, 20)}, n_boot=500)
    assert narrow.pooled[1] - narrow.pooled[0] < wide.pooled[1] - wide.pooled[0]
    assert narrow.per_task[1][1] - narrow.per_task[1][0] < wide.per_task[1][1] - wide.per_task[1][0]


def test_another_level_takes_its_own_quantile():
    assert z_for(0.95) == Z95 and z_for(0.90) == pytest.approx(1.6448536269514722, abs=1e-12)
    assert bootstrap({1: (5, 10)}, n_boot=50, level=0.9).per_task[1] == wilson(5, 10, z=z_for(0.9))


@pytest.mark.parametrize("tallies, kw", [({}, {}), ({1: (0, 0)}, {}), ({1: (3, 2)}, {}), ({1: (-1, 2)}, {}),
                                         ({1: (1, 2)}, {"n_boot": 0}), ({1: (1, 2)}, {"level": 1.0})])
def test_bad_input_is_refused(tallies, kw):
    with pytest.raises(ValueError):
        bootstrap(tallies, **{"n_boot": 10, **kw})


@pytest.mark.parametrize("k, n", [(0, 0), (3, 2), (-1, 2)])
def test_a_wilson_interval_needs_a_proportion(k, n):
    with pytest.raises(ValueError):
        wilson(k, n)
