"""Intervals for one read's success rate: a Wilson score interval per task, and a seed bootstrap of
the all-24 mean.

Per task, a success rate is one binomial proportion, k successes over n seeds. Its interval is the
Wilson score interval: the p whose score test does not reject at the level (z = 1.959964 at 95 %),

    (p_hat + z^2/2n -+ z sqrt(p_hat (1 - p_hat)/n + z^2/4n^2)) / (1 + z^2/n),

which stays an interval at 0/n and n/n (0/50 gives [0, 7.13] %), where several benchmark tasks
sit. The binomial SE and a percentile bootstrap both collapse to a point there: every resample of
0/n is 0/n. So no per-task interval here is a bootstrap.

Pooled, the reported number is the mean of the per-task rates. Its interval is a seed bootstrap:
each task's seeds are resampled with replacement (each task keeps its own n), the mean is
recomputed, and the bounds are the percentiles at indices floor((1 - level) / 2 * B) and
round((1 + level) / 2 * B) - 1 of the B sorted resamples (250 and 9,749 for B = 10,000). Resampling
n outcomes of which k succeeded draws Binomial(n, k/n) successes, so one resample of a task is n
Bernoulli(k/n) draws -- the same distribution without materialising the outcomes. It needs no
normal approximation for the mean of unequal rates. Like the binomial SE, it takes each task's
observed rate as that task's distribution, so a task at 0 % or 100 % adds no spread to it.
Deterministic: random.Random(seed), tasks in the pooled (string) order, so one file always gives
one interval.

Both measure the evaluation uncertainty of ONE read of one checkpoint over the benchmark's seeds.
They say nothing about training-seed variation, and neither is a band to apply to any other read
or contrast (docs/reference/reporting-rules.md).

Core tier: stdlib only.
"""
from __future__ import annotations

import dataclasses
import math
import random
from collections.abc import Mapping

N_BOOT = 10_000
SEED = 7
LEVEL = 0.95
Z95 = 1.959963984540054      # the two-sided 95 % normal quantile


@dataclasses.dataclass(frozen=True)
class Interval:
    pooled: tuple[float, float]                  # pct, seed bootstrap of the mean over tasks
    per_task: dict[int, tuple[float, float]]     # pct, Wilson score, in pooled order
    n_boot: int
    seed: int
    level: float


def z_for(level: float) -> float:
    """The two-sided normal quantile of `level` (Z95 at 0.95), by bisection on math.erf."""
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), got {level!r}")
    if level == LEVEL:
        return Z95
    lo, hi = 0.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if math.erf(mid / math.sqrt(2.0)) < level:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def wilson(successes: int, n: int, *, z: float = Z95) -> tuple[float, float]:
    """The Wilson score interval of `successes` out of `n`, in percent."""
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError(f"a Wilson interval needs n >= 1 and 0 <= successes <= n, got {successes}/{n}")
    p, z2 = successes / n, z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z / denom * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return 100.0 * max(0.0, centre - half), 100.0 * min(1.0, centre + half)


def _bounds(values: list[float], level: float) -> tuple[float, float]:
    n = len(values)
    lo = int((1 - level) / 2 * n)
    hi = max(lo, int(round((1 + level) / 2 * n)) - 1)
    return values[lo], values[hi]


def bootstrap(tallies: Mapping[int, tuple[int, int]], *, n_boot: int = N_BOOT, seed: int = SEED,
              level: float = LEVEL) -> Interval:
    """`tallies`: task -> (successes, rollouts). The pooled interval is the seed bootstrap of the
    mean per-task rate; `per_task` holds each task's Wilson interval at the same level. Raises
    ValueError for no tasks, a task with no rollouts, successes outside [0, n], n_boot < 1 or a
    level outside (0, 1)."""
    order = sorted(tallies, key=str)
    if n_boot < 1 or not 0 < level < 1 or not order or \
            any(not 0 <= tallies[t][0] <= tallies[t][1] or tallies[t][1] <= 0 for t in order):
        raise ValueError("a bootstrap needs n_boot >= 1, 0 < level < 1, and for every task at least one "
                         "rollout and 0 <= successes <= rollouts")
    rng = random.Random(seed)
    pooled: list[float] = []
    for _ in range(n_boot):
        acc = 0.0
        for t in order:
            s, n = tallies[t]
            k = s
            if 0 < s < n:
                p, k = s / n, 0
                for _ in range(n):          # a counting loop: no builtin sum() on the T0 path
                    if rng.random() < p:    # (tests/golden/paper/test_t0_version_independence.py)
                        k += 1
            acc += k / n
        pooled.append(100.0 * acc / len(order))
    pooled.sort()
    z = z_for(level)
    return Interval(_bounds(pooled, level), {t: wilson(*tallies[t], z=z) for t in order}, n_boot, seed, level)
