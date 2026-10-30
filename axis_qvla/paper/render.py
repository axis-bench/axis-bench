"""Every paper output T0 writes, from the pooled cells and the recorded lineage.

Why: one entry point for t0.rebuild and the golden tests. Given the pooled cells (pooled_r1.json),
each cell's training lineage from cells.json and the paper layout, it returns the figure data
(figures/fig*.csv), Table I (tables/table1.{md,tex,html}), the in-text numbers with Table I's
per-seed values (claims.json) and the results page (results_page.{md,html}), plus any integrity
problem (a Table I cell pretrained with another seed than its replicate's), which T0 reports as
exit 2 instead of writing the outputs.

Core tier: stdlib only.
"""
from __future__ import annotations

from collections.abc import Mapping

from axis_qvla.paper import claims as claims_mod, figures, page, table1
from axis_qvla.paper.spec import PaperSpec

OUTPUTS = ("figures/fig1.csv", "figures/fig4c.csv", "figures/fig5.csv", "figures/fig6.csv", "figures/fig7.csv",
           "tables/table1.md", "tables/table1.tex", "tables/table1.html", "claims.json",
           "results_page.md", "results_page.html")


def render_paper(P: Mapping[str, dict], lineage: Mapping[str, object],
                 spec: PaperSpec) -> tuple[dict[str, bytes], tuple[str, ...]]:
    t = table1.compute(P, spec, lineage)
    if t.problems:
        return {}, t.problems
    rows = {"fig1": figures.fig1_rows(P, t, spec)} | {f.key: figures.figure_rows(P, spec, f) for f in spec.figures}
    claims = claims_mod.compute(P, t, spec)
    out = {f"figures/{k}.csv": figures.to_csv(v) for k, v in rows.items()}
    t1_md = table1.render_md(t, spec)
    out |= {"tables/table1.md": t1_md, "tables/table1.tex": table1.render_tex(t, spec),
            "tables/table1.html": table1.render_html(t, spec),
            "claims.json": claims_mod.dump(claims, t, spec, n_cells=len(P))}
    bs = page.blocks(rows, t, claims, spec, table1_md=t1_md.decode("utf-8"), n_cells=len(P))
    out["results_page.md"] = page.to_md(bs)
    out["results_page.html"] = page.to_html(bs, table1.html_table(t, spec))
    return {k: out[k] for k in OUTPUTS}, ()
