"""Score results files and contrast arms under the release's reporting rules, enforced in code.

Why: the reporting rules were prose, not enforced; here they are
behaviour. all-24 is the headline and every other view is flagged `appendix`; `informative_20` and
`usable_12` are the benchmark's declared subsets. A declared (task, seed) with no row counts as a
failure, so a partial read can never score higher than the full one. Which benchmark version, seed
set, execution and serving condition a file may carry is decided by axis_qvla.bench.results.checks; a
development-seed read is scored on the development seeds, and never beside a test read.

A contrast reports, by default, the raw difference in percentage points. With
a BC side (`bc`, the CLI's `--relative --bc`) it reports the paper's Table I statistic instead:
per replicate 100 (SR_A - SR_B) / SR_BC with that replicate's plain-BC rate, computed exactly from
the per-task tallies, then the mean and sample SD over replicates (Appendix A).

A contrast pairs arms by TRAINING REPLICATE, not the full (pretrain, FT) tuple -- the key is the
pretrain seed, or the FT seed for an arm with no pretrain -- because FT seeds are recorded per
run and are not guaranteed to match across a pretrain-keyed replicate (a replicate pairs on the
pretrain seed: seed43_bc ran FT seed 43 while seed43_cfgpt ran FT seed 42 on the same pretrain-43
replicate); a differing FT seed -- on any side, the BC normaliser's included -- is kept as
metadata and disclosed on the Contrast, never silently dropped. It reports the mean and sample
SD of the per-replicate differences, and keeps each replicate's difference and that pair's own
eval SE, sqrt(se_a² + se_b²) (in the relative mode the delta-method SE of the ratio,
`relative_se`), so both are reported per replicate; the eval SE belongs to that pair and is never
a band for any other. It refuses re-reads (the same checkpoint twice on one side is a
reproducibility check, never a replicate), pairs only reads of one seed set executed alike, and
states a claim only with >= 3 paired replicates.

Per-task success rates are averaged in task-id STRING order, the order of pool.py, so a full
arm's all-24 score equals its paper cell bit for bit -- `score` builds the per-task tallies in
that order and hands them to `pool.pool_cell` rather than re-deriving the agg/SE
formula, so there is exactly one implementation of the arithmetic.

Core tier: stdlib only.
"""
from __future__ import annotations

import dataclasses
import math
import statistics
from collections.abc import Sequence
from fractions import Fraction

from axis_qvla.bench.spec import Benchmark
from axis_qvla.paper.pool import pool_cell
from axis_qvla.core.ltr import ltr_sum
from axis_qvla.bench.results.checks import (check_alike, check_bench, check_execution, check_rows, check_serve,
                                       seed_set_of)
from axis_qvla.bench.results.format import Results

VIEWS = ("all24", "informative_20", "usable_12")
MIN_CLAIM_SEEDS = 3


@dataclasses.dataclass(frozen=True)
class ArmScore:
    served: str
    view: str
    pct: float
    se: float
    n_tasks: int
    rollouts: int
    missing: int
    appendix: bool
    rate: Fraction | None = None       # the exact mean per-task success rate (pct / 100)
    seed_set: str = "test"
    off_protocol: bool = False
    tallies: tuple[tuple[int, int, int], ...] = ()   # (task, successes, n) in pooled (string) order


@dataclasses.dataclass(frozen=True)
class Contrast:
    a: str
    b: str
    n_pairs: int
    mean: float
    sd: float | None
    eval_se: tuple[float, ...]
    claim_allowed: bool
    ft_seed_differs: tuple[int, ...]
    label: str
    relative: bool = False             # True: 100 (SR_A - SR_B) / SR_BC per replicate (Table I)
    replicates: tuple[int, ...] = ()   # the replicate keys, ascending; diffs and eval_se follow them
    diffs: tuple[float, ...] = ()      # the paired difference at each replicate


def score(r: Results, bench: Benchmark, *, view: str = "all24", allow_off_protocol: bool = False) -> ArmScore:
    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}")
    check_bench(r, bench)
    off = check_execution(r, bench, allow_off_protocol=allow_off_protocol)
    check_serve(r)
    check_rows(r)
    seed_set = seed_set_of(r)
    tasks = bench.task_ids if view == "all24" else bench.subsets[view]
    # Declared (task, seed) pairs come from the benchmark's own per-task seed lists, never a
    # formula: two tasks need not share a seed-numbering convention. A run evaluates the first
    # n_seeds of its set (the units every run manifest digests), so those are the declared ones.
    seeds_of = {t: bench.seeds_for(t, seed_set)[:bench.n_seeds] for t in bench.task_ids}
    declared = {(t, s) for t in bench.task_ids for s in seeds_of[t]}
    got = {(x.task, x.seed): x.success for x in r.rows}
    stray = sorted(set(got) - declared)
    if stray:
        raise ValueError(f"{r.header.served}: rows for undeclared (task, seed) of the {seed_set} seeds: {stray[:3]}")
    tallies: dict[int, tuple[int, int]] = {}
    found = 0
    for t in sorted(tasks, key=str):
        seeds = seeds_of[t]
        found += sum(1 for s in seeds if (t, s) in got)
        tallies[t] = (sum(1 for s in seeds if got.get((t, s)) is True), len(seeds))
    pooled = pool_cell(r.header.served, tallies)
    expected = sum(n for _, n in tallies.values())
    rate = ltr_sum((Fraction(s, n) for s, n in tallies.values()), Fraction(0)) / len(tallies)
    return ArmScore(r.header.served, view, pooled["agg"], pooled["se"], len(tasks), found,
                    expected - found, view != "all24", rate, seed_set, off,
                    tuple((t, s, n) for t, (s, n) in tallies.items()))


def _same_read(p: Results, q: Results) -> bool:
    """Two files of the same seed files are one read, not two: one eval fingerprint and the same
    rows (the rows too, so a placeholder fingerprint cannot make two reads one)."""
    return bool(p.header.eval_fingerprint) and p.header.eval_fingerprint == q.header.eval_fingerprint \
        and p.rows == q.rows


def relative_se(a: tuple[ArmScore, Results], b: tuple[ArmScore, Results], bc: tuple[ArmScore, Results]) -> float:
    """The eval SE of R = 100 (A - B) / C, C the plain-BC read, by the delta method:
    var R = g^2 (var_A + var_B - 2 cov_AB) + h^2 var_C + 2 g h (cov_AC - cov_BC), with
    g = 100 / C and h = -100 (A - B) / C^2, rates and SEs in pct. Separate reads are independent
    (covariance 0); a side that IS the BC read (B is the plain-BC file, say) moves with it, so its
    covariance with C is C's variance. Leaving C's own error out understates the SE."""
    x, y, z = a[0], b[0], bc[0]
    g, h = 100.0 / z.pct, -100.0 * (x.pct - y.pct) / z.pct ** 2

    def cov(p: tuple[ArmScore, Results], q: tuple[ArmScore, Results]) -> float:
        return p[0].se ** 2 if _same_read(p[1], q[1]) else 0.0

    var = (g * g * (x.se ** 2 + y.se ** 2 - 2 * cov(a, b)) + h * h * z.se ** 2
           + 2 * g * h * (cov(a, bc) - cov(b, bc)))
    return math.sqrt(max(var, 0.0))


def _seed_key(ts: object) -> int | None:
    """The training replicate key: the pretrain seed, or the FT seed when there is no pretrain
    (an arm with no pretraining has only its FT seed to identify the replicate). None when the
    header carries no FT seed at all."""
    if not isinstance(ts, dict) or not isinstance(ts.get("ft"), int):
        return None
    pre = ts.get("pretrain")
    return pre if isinstance(pre, int) else ts["ft"]


def _by_seed(side: Sequence[Results], name: str) -> dict[int, Results]:
    out: dict[int, Results] = {}
    # A None pub_digest can't be caught as a re-read here (excluded from `dup` below); the
    # per-replicate-key uniqueness check in the loop is what still refuses two entries at the
    # same replicate regardless of digest, so together the two guards cover a side with unknown
    # digests too -- only a distinct digest AND a distinct replicate key passes.
    # A published file names its checkpoint by pub_digest, a file built from a run by eval_digest
    # (the evaluated directory's content digest, in the same format over the same files when the
    # directory is the published file set), so either identifies a re-read.
    digests = [x.header.checkpoint.get("pub_digest") or x.header.checkpoint.get("eval_digest") for x in side]
    dup = sorted({d for d in digests if digests.count(d) > 1 and d is not None})
    if dup:
        raise ValueError(f"side {name}: the same checkpoint appears twice ({dup[0]}); re-reads are "
                         "reproducibility checks, not replicates")
    for x in side:
        s = _seed_key(x.header.training_seed)
        if s is None or s in out:
            raise ValueError(f"side {name}: {x.header.served} has training seed "
                             f"{x.header.training_seed!r} (missing or repeated)")
        out[s] = x
    return out


def contrast(a: Sequence[Results], b: Sequence[Results], bench: Benchmark, *,
             bc: Sequence[Results] | None = None, allow_off_protocol: bool = False) -> Contrast:
    every = [*a, *b, *(bc or ())]
    sets = {seed_set_of(x) for x in every}
    if len(sets) > 1:
        raise ValueError(f"a contrast pairs reads of one seed set; got {sorted(sets)}")
    sa, sb = _by_seed(a, "a"), _by_seed(b, "b")
    sbc = _by_seed(bc, "bc") if bc is not None else None
    if set(sa) != set(sb) or not sa or (sbc is not None and set(sbc) != set(sa)):
        raise ValueError(f"training seeds do not pair: a {sorted(sa)} vs b {sorted(sb)}"
                         + ("" if sbc is None else f" vs bc {sorted(sbc)}"))
    diffs, eval_se, ft_differs, off = [], [], [], False
    for s in sorted(sa):
        x = score(sa[s], bench, allow_off_protocol=allow_off_protocol)
        y = score(sb[s], bench, allow_off_protocol=allow_off_protocol)
        off = off or x.off_protocol or y.off_protocol
        se = math.sqrt(x.se ** 2 + y.se ** 2)
        if sbc is None:
            diffs.append(x.pct - y.pct)
            eval_se.append(se)
        else:
            z = score(sbc[s], bench, allow_off_protocol=allow_off_protocol)
            off = off or z.off_protocol
            if not z.rate:
                raise ValueError(f"bc {z.served} has no success: the BC-normalized difference is undefined")
            diffs.append(float(100 * (x.rate - y.rate) / z.rate))
            eval_se.append(relative_se((x, sa[s]), (y, sb[s]), (z, sbc[s])))
        # The replicate key pairs a, b (and bc); their FT seeds were recorded independently per
        # run and are not guaranteed to match even on a paired replicate (a replicate pairs on
        # the pretrain seed, never the full (pretrain, ft) tuple).
        sides = (sa[s], sb[s]) if sbc is None else (sa[s], sb[s], sbc[s])
        if len({x.header.training_seed.get("ft") for x in sides}) > 1:
            ft_differs.append(s)
    check_alike(every)          # after scoring: an off-protocol file is refused by name first
    k = len(diffs)
    ok = k >= MIN_CLAIM_SEEDS
    label = f"paired, n_seeds={k}" if ok else f"descriptive, n_seeds={k}"
    if ft_differs:
        label += f"; ft seeds differ in replicate(s) {', '.join(str(s) for s in ft_differs)}"
    if sbc is not None:
        label += "; BC-normalized: 100 (SR_A - SR_B) / SR_BC per replicate (the main results)"
    if off:
        label += "; OFF-PROTOCOL"
    if sets == {"dev"}:
        label += "; development seeds"
    return Contrast("+".join(sa[s].header.arm for s in sorted(sa)), "+".join(sb[s].header.arm for s in sorted(sb)),
                    k, ltr_sum(diffs) / k, statistics.stdev(diffs) if k >= 2 else None, tuple(eval_se), ok,
                    tuple(ft_differs), label, sbc is not None, tuple(sorted(sa)), tuple(diffs))
