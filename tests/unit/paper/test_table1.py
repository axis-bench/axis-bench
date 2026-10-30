"""Table I: per seed 100(SR_A - SR_B)/SR_BC with that seed's BC, then mean and sample SD over seeds."""
from __future__ import annotations

import math

from axis_qvla.paper.spec import load_spec
from axis_qvla.paper.table1 import compute, render_html, render_md, render_tex
from tests.unit.paper.paper_world import lineage, pooled

SPEC = load_spec()


IDX = {c: (i, role) for reps in SPEC.table1["replicates"].values() for i, r in enumerate(reps)
       for role, c in r["cells"].items()}


def _succ(cell: str) -> int:
    """BC 600 at every seed; episode conditioning beats it by 12, 24 and 36 trials at the three seeds."""
    if cell not in IDX:
        return 500
    i, role = IDX[cell]
    return {"bc": 600, "cond_ep": 612 + 12 * i}.get(role, 600)


def test_each_entry_is_normalised_by_its_own_seeds_bc():
    t = compute(pooled(_succ, SPEC), SPEC, lineage(SPEC))
    assert t.problems == ()
    e = t.entries[("cond_minus_bc", "pi05")]
    assert e.seeds == (42, 43, 44)
    sr_bc = 100 * 600 / 1200
    want = [100 * (100 * k / 1200 - sr_bc) / sr_bc for k in (612, 624, 636)]
    assert all(math.isclose(a, b, rel_tol=1e-12) for a, b in zip(e.values, want))
    assert math.isclose(e.mean, sum(want) / 3, rel_tol=1e-12)
    assert math.isclose(e.sd, math.sqrt(sum((w - sum(want) / 3) ** 2 for w in want) / 2), rel_tol=1e-12)
    assert e.signs == (1, 1, 1)
    tie = t.entries[("filter_chunk_minus_episode", "xvla")]
    assert tie.values == (0.0, 0.0, 0.0) and tie.signs == (0, 0, 0) and tie.sd == 0.0


def test_a_missing_cell_leaves_the_entry_empty():
    P = pooled(_succ, SPEC)
    del P["svla_cfgph_sd1|cond"]
    e = compute(P, SPEC, lineage(SPEC)).entries[("cond_minus_bc", "smolvla")]
    assert e.values[1] is None and e.mean is None and e.sd is None and e.signs[1] is None


def test_a_cell_pretrained_with_another_seed_is_a_problem():
    lin = lineage(SPEC)
    lin["cfgpt43|cond"] = {"pretrain": 42, "ft": 42}
    t = compute(pooled(_succ, SPEC), SPEC, lin)
    assert t.problems == ("Table I: π0.5 replicate 43 (role cond_ep) is cell cfgpt43|cond, "
                          "whose pretrain seed is 42",)
    lin["cfgpt43|cond"] = {"pretrain": 43, "ft": 42}         # a different FT seed is metadata only
    assert compute(pooled(_succ, SPEC), SPEC, lin).problems == ()


def test_a_table_i_cell_with_no_recorded_lineage_is_a_problem():
    """A null training_seed cannot pass the pretrain-seed check silently. A
    cell the cell map does not hold at all stays an empty entry (a missing cell), not a problem."""
    lin = lineage(SPEC)
    lin["cfgpt43|cond"] = None
    assert compute(pooled(_succ, SPEC), SPEC, lin).problems == (
        "Table I: π0.5 replicate 43 (role cond_ep) is cell cfgpt43|cond, whose lineage is not recorded",)
    del lin["cfgpt43|cond"]
    assert compute(pooled(_succ, SPEC), SPEC, lin).problems == ()


def test_negatives_print_with_the_papers_minus_sign_and_the_caption_spells_normalized():
    """The paper prints U+2212 and "BC-normalized"; the .tex keeps a math-mode minus."""
    def lose(cell: str) -> int:
        return {"bc": 600, "cond_ep": 588}.get(IDX[cell][1], 600) if cell in IDX else 500

    t = compute(pooled(lose, SPEC), SPEC, lineage(SPEC))
    e = t.entries[("cond_minus_bc", "pi05")]
    md, tex, html = (f(t, SPEC).decode() for f in (render_md, render_tex, render_html))
    cell = f"−{abs(e.mean):.2f} ± {e.sd:.2f}"
    assert e.mean < 0 and f"| Episode cond. − BC | {cell} |" in md and f"<td>{cell}</td>" in html
    assert f"${e.mean:+.2f} \\pm {e.sd:.2f}$" in tex and "−" not in tex
    assert "BC-normalized" in md + tex + html and "normalised" not in md + tex + html


def test_renderings_carry_every_entry_in_the_papers_format():
    t = compute(pooled(_succ, SPEC), SPEC, lineage(SPEC))
    md, tex, html = render_md(t, SPEC).decode(), render_tex(t, SPEC).decode(), render_html(t, SPEC).decode()
    e = t.entries[("cond_minus_bc", "pi05")]
    cell = f"{e.mean:+.2f} ± {e.sd:.2f}"
    assert f"| Episode cond. − BC | {cell} |" in md
    assert f"${e.mean:+.2f} \\pm {e.sd:.2f}$" in tex and "\\pi_{0.5}" in tex
    assert f"<td>{cell}</td>" in html and "<th>SmolVLA</th>" in html
    assert md.count("\n|") == 2 + len(SPEC.table1["rows"])
