"""render_paper: the paper's outputs, in the paper's terms, the same bytes every time."""
from __future__ import annotations

import re

from axis_qvla.paper.render import OUTPUTS, render_paper
from axis_qvla.paper.spec import load_spec
from tests.unit.paper.paper_world import lineage, pooled

SPEC = load_spec()
# Words the paper's prose does not use. Cell and run ids keep them inside identifiers
# (`awr_v2_ep|bare`), where a word boundary never falls, so whole-word matches are real prose.
INTERNAL = re.compile(r"\b(v2|phase|phases|row|rows|CFG|AWR|drop-top|guided|anneal|R1)\b", re.IGNORECASE)


def _render(P=None):
    return render_paper(P if P is not None else pooled(lambda c: 400 + (len(c) * 17) % 101, SPEC), lineage(SPEC), SPEC)


def test_the_output_set_is_the_papers():
    out, problems = _render()
    assert problems == () and sorted(out) == sorted(OUTPUTS)
    assert {"figures/fig1.csv", "figures/fig4c.csv", "figures/fig5.csv", "figures/fig6.csv", "figures/fig7.csv",
            "tables/table1.tex", "tables/table1.md", "tables/table1.html", "claims.json",
            "results_page.html", "results_page.md"} == set(OUTPUTS)


def test_no_internal_vocabulary_outside_identifiers():
    out, _ = _render()
    for name, data in out.items():
        hits = INTERNAL.findall(data.decode("utf-8"))
        assert hits == [], (name, hits[:5])


def test_the_ratchet_sees_internal_words_in_prose():
    assert INTERNAL.findall("the v2 reward, CFG arms, drop-top filtering, row granularity, R1 reads") == [
        "v2", "CFG", "drop-top", "row", "R1"]
    assert INTERNAL.findall("awr_v2_ep|bare cfg_ph_row|cond r1cfgphep_c_w0_dualsim") == []


def test_no_dates_and_identical_bytes_on_a_rerun():
    a, _ = _render()
    b, _ = _render()
    assert a == b
    assert not any(re.search(rb"20\d\d-\d\d-\d\d", d) for d in a.values())


def test_a_lineage_mismatch_is_a_problem_not_an_output():
    lin = lineage(SPEC)
    lin["svla_bc_sd2|bare"] = {"pretrain": 1, "ft": 1}
    out, problems = render_paper(pooled(lambda c: 600, SPEC), lin, SPEC)
    assert len(problems) == 1 and "svla_bc_sd2|bare" in problems[0]
