"""The results page (Markdown and HTML): the paper's sections, in the paper's terms, from T0's data."""
from __future__ import annotations

import re

from axis_qvla.paper.render import render_paper
from axis_qvla.paper.spec import load_spec
from tests.unit.paper.paper_world import lineage, pooled

SPEC = load_spec()


def _outputs(P=None, lin=None):
    P = P if P is not None else pooled(lambda c: 500 + (len(c) * 11) % 83, SPEC)
    out, problems = render_paper(P, lineage(SPEC) if lin is None else lin, SPEC)
    assert problems == ()
    return out


def test_the_page_follows_the_papers_sections_in_order():
    md = _outputs()["results_page.md"].decode()
    heads = [line[3:] for line in md.split("\n") if line.startswith("## ")]
    assert heads == [s["title"] for s in SPEC.sections]
    html = _outputs()["results_page.html"].decode()
    assert [m for m in re.findall(r"<h2>(.*?)</h2>", html)] == [s["title"].replace("&", "&amp;") for s in SPEC.sections]


def test_every_claim_and_table_i_row_is_on_the_page():
    out = _outputs()
    md = out["results_page.md"].decode()
    for c in SPEC.claims:
        assert c["label"].replace("|", "\\|") in md, c["id"]
    for r in SPEC.table1["rows"]:
        assert f"| {r['label']} |" in md
    assert "**" + SPEC.table1["title"] + "**" in md


def test_cell_ids_are_escaped_in_markdown_and_html():
    out = _outputs()
    md, html = out["results_page.md"].decode(), out["results_page.html"].decode()
    assert "awr_v2_ep\\|bare" in md and "| awr_v2_ep|bare" not in md
    assert "<td>awr_v2_ep|bare</td>" in html and "<script" not in html


def test_the_intro_defines_every_quantity_the_page_prints():
    """The SE formula, the per-seed BC of Fig. 1 and Table I, and Fig. 1's SE are stated;
    the HTML shows no Markdown backticks."""
    out = _outputs()
    intro = out["results_page.md"].decode().split("\n")[2]
    for phrase in ("task-stratified binomial SE", "that seed's own plain BC", "sqrt(SE_A² + SE_B²)", "SR(A) − SR(B)"):
        assert phrase in intro, phrase
    assert "`" not in out["results_page.html"].decode()


def test_negative_numbers_on_the_page_use_the_papers_minus_sign():
    md = _outputs()["results_page.md"].decode()
    assert "| −" in md and not re.search(r"(^|[ (|])-\d", md, re.M)


def test_a_ft_seed_that_differs_from_its_replicate_is_disclosed():
    lin = lineage(SPEC)
    lin["cfgpt43|cond"] = {"pretrain": 43, "ft": 42}
    md = _outputs(lin=lin)["results_page.md"].decode()
    assert "| π0.5 | 43 | episode conditioning (tagged FT, tag-5 inference) | cfgpt43\\|cond | 43 | 42 |" in md
