"""Figure data: label, cells used, SR, SR_BC, 100(SR/SR_BC - 1) from unrounded rates, SE x 100/SR_BC."""
from __future__ import annotations

import math

from axis_qvla.paper.figures import COLUMNS, fig1_rows, figure_rows, to_csv
from axis_qvla.paper.rates import exact, rel, scaled, se, sr
from axis_qvla.paper.spec import load_spec
from axis_qvla.paper.table1 import compute
from tests.unit.paper.paper_world import lineage, pooled

SPEC = load_spec()


def _succ(cell: str) -> int:
    return 600 + (len(cell) * 7) % 97          # distinct, deterministic rates


def _fig(key: str):
    return next(f for f in SPEC.figures if f.key == key)


def test_every_bar_is_relative_to_its_vlas_plain_bc():
    P = pooled(_succ, SPEC)
    rows = figure_rows(P, SPEC, _fig("fig5"))
    assert len(rows) == 42 and [r.vla for r in rows[:2]] == ["π0.5", "π0.5"]
    bc, ref = rows[0], sr(P, "bc|bare")
    assert (bc.group, bc.cells, bc.sr, bc.sr_bc, bc.rel) == ("plain BC", ("bc|bare",), ref, ref, 0.0)
    for r, b in zip(rows, _fig("fig5").bars):
        ref = sr(P, SPEC.vla(b.vla).bc)
        assert r.rel == rel(P, b.cell, SPEC.vla(b.vla).bc) and r.se_rel == scaled(se(P, b.cell), ref)
        assert abs(r.rel - 100 * (sr(P, b.cell) / ref - 1)) < 1e-9     # the float formula, to rounding
    assert sum(r.group in ("weighting", "filtering", "conditioning") for r in rows) == 36


def test_a_missing_cell_is_an_empty_bar():
    P = pooled(_succ, SPEC)
    del P["svla_rand_ep|bare"]
    r = figure_rows(P, SPEC, _fig("fig4c"))[0]
    assert r.cells == ("svla_rand_ep|bare",) and (r.sr, r.rel, r.se_rel) == (None, None, None)


def test_fig1_is_table_i_first_comparison_per_seed_and_its_mean():
    P = pooled(_succ, SPEC)
    t = compute(P, SPEC, lineage(SPEC))
    rows = fig1_rows(P, t, SPEC)
    assert len(rows) == 12
    pi = [r for r in rows if r.vla == "π0.5"]
    e = t.entries[("cond_minus_bc", "pi05")]
    assert [r.label for r in pi] == ["seed 42", "seed 43", "seed 44", "mean over seeds"]
    assert [r.rel for r in pi] == list(e.values) + [e.mean]
    assert pi[0].cells == ("cfg_ph_ep|cond", "bc|bare") and pi[3].sr is None


def test_sr_is_the_exact_rate_rounded_once_like_rel():
    """SR and SR_BC are the exact mean of the per-task rates rounded to a float once (the
    pooled `agg` is a float sum and can carry a last-bit residue), so a reader recomputing
    100(sr/sr_bc - 1) from the CSV gets the CSV's own tie."""
    P = pooled(_succ, SPEC)
    for r, b in zip(figure_rows(P, SPEC, _fig("fig5")), _fig("fig5").bars):
        assert r.sr == float(100 * exact(P, b.cell)) and r.sr_bc == float(100 * exact(P, SPEC.vla(b.vla).bc))


def test_fig1_se_is_the_se_of_the_difference_on_that_seeds_bc_scale():
    """A Fig. 1 seed row is SR_A - SR_B over that seed's SR_BC, so its SE is the marginal
    SE of the difference, 100 sqrt(SE_A^2 + SE_B^2) / SR_BC -- not cell A's SE alone."""
    P = pooled(_succ, SPEC)
    rows = [r for r in fig1_rows(P, compute(P, SPEC, lineage(SPEC)), SPEC) if r.label.startswith("seed")]
    assert len(rows) == 9
    for r in rows:
        a, b = r.cells
        assert r.se_rel == 100 * math.sqrt(se(P, a) * se(P, a) + se(P, b) * se(P, b)) / r.sr_bc


def test_csv_bytes_are_exact_and_stable():
    P = pooled(_succ, SPEC)
    data = to_csv(figure_rows(P, SPEC, _fig("fig6")))
    lines = data.decode("utf-8").split("\n")
    assert lines[0] == ",".join(COLUMNS) and lines[-1] == "" and len(lines) == 18 + 2
    first = lines[1].split(",")
    assert first[:4] == ["π0.5", "plain BC", "1K", "s1000bc|bare"]
    assert float(first[4]) == sr(P, "s1000bc|bare") and first[4] == repr(sr(P, "s1000bc|bare"))
    assert to_csv(figure_rows(P, SPEC, _fig("fig6"))) == data
    del P["s1000bc|bare"]
    assert to_csv(figure_rows(P, SPEC, _fig("fig6"))).decode().split("\n")[1].split(",")[4:8] == [
        "", repr(sr(P, "bc|bare")), "", ""]
