"""The results page, in Markdown and HTML, structured like the paper's Secs. IV.B and V.B-V.F.

Why: a reader checks the paper against the release on one page. Each section of the layout lists
its figures as tables (SR, the BC-relative value, the rescaled SE and the cells read), then its
in-text numbers (claims.json), in the paper's terms; V.F adds Table I, its per-seed values, and
every cell whose fine-tuning seed differs from its replicate's pretraining seed. Both renderings
come from one list of blocks, so they cannot disagree; negatives carry the paper's minus sign
(U+2212). No date, path or host is written.

Core tier: stdlib only.
"""
from __future__ import annotations

import html
from collections.abc import Mapping, Sequence

from axis_qvla.paper.figures import Row
from axis_qvla.paper.rates import fmt, shown
from axis_qvla.paper.spec import PaperSpec
from axis_qvla.paper.table1 import Table1

INTRO = ("Every number on this page is recomputed by qvla paper tables (T0) from the per-seed rollout files "
         "of the {n} cells in qvla-v1-results. A cell is one fine-tuned checkpoint served one way and evaluated "
         "on the 24 tasks with the same 50 layouts each (1,200 trials). SR is the mean task success (%), and "
         "its SE is the task-stratified binomial SE, 100 sqrt(Σ_t p_t(1 − p_t)/50)/24 over the per-task success "
         "rates p_t. A relative value is 100(SR/SR_BC − 1) from unrounded rates, where SR_BC is the VLA's 5K "
         "plain BC (plain FT, untagged inference), and its SE is the SE scaled by 100/SR_BC. Fig. 1 and Table I "
         "instead compare two cells of one training seed and divide by that seed's own plain BC, "
         "100(SR(A) − SR(B))/SR_BC; the SE of a Fig. 1 per-seed value is that of the difference, "
         "100 sqrt(SE_A² + SE_B²)/SR_BC. A difference in percentage points (pp) is SR(A) − SR(B).")
FIG_HEAD = ("VLA", "group", "bar", "SR (%)", "relative (%)", "SE (%)", "note", "cells")


def _fig_rows(rows: Sequence[Row]) -> list[tuple[str, ...]]:
    return [(r.vla, r.group, r.label, fmt(r.sr, 2), shown(fmt(r.rel, 1, signed=True)), fmt(r.se_rel, 1), r.note,
             "; ".join(r.cells)) for r in rows]


def _claim_text(c: Mapping) -> tuple[str, str]:
    """(value, details) of one claim for the page."""
    if "text" in c:
        return shown(c["text"]), ", ".join(c.get("cells", ()))
    if "groups" in c:
        return ", ".join(f"{g} {shown(x['text'])}" for g, x in c["groups"].items()), ""
    counts = f"{c['positive']} positive, {c['zero']} equal, {c['negative']} negative"
    if c["missing"]:
        counts += f", {c['missing']} missing"
    if "pairs" in c:
        return counts, "; ".join(f"{p['label']} {shown(fmt(p['diff'], 2, signed=True))}" for p in c["pairs"])
    return counts, ""


def blocks(fig_rows: Mapping[str, Sequence[Row]], t: Table1, claims: Mapping[str, Mapping], spec: PaperSpec,
           *, table1_md: str, n_cells: int) -> list[tuple]:
    out: list[tuple] = [("h1", f"Simulation results: {spec.paper}"), ("p", INTRO.format(n=n_cells))]
    vla_label = {v.key: v.label for v in spec.vlas}
    for s in spec.sections:
        out.append(("h2", s["title"]))
        for f in s["figures"]:
            if f == "table1":
                out.append(("md", table1_md, t))
                seeds = [(vla_label[v], r["label"], str(seed), shown(fmt(x, 2, signed=True)))
                         for r in spec.table1["rows"] for v in t.replicates for seed, x in
                         zip(t.entries[(r["id"], v)].seeds, t.entries[(r["id"], v)].values)]
                out.append(("table", "Table I per training seed (%)", ("VLA", "comparison", "seed", "value"), seeds))
                moved = [(vla_label[v], str(rep["seed"]), spec.table1["roles"][role], x["cell"],
                          str(x["lineage"]["pretrain"]), str(x["lineage"]["ft"]))
                         for v, reps in t.replicates.items() for rep in reps for role, x in rep["cells"].items()
                         if isinstance(x["lineage"], dict) and x["lineage"].get("ft") != rep["seed"]]
                out.append(("p", "A training seed is a replicate of the pretraining run, keyed by its pretraining "
                                 "seed; each cell's fine-tuning seed is recorded in cells.json. Cells whose "
                                 "fine-tuning seed differs from the replicate's seed:"))
                out.append(("table", "", ("VLA", "seed", "role", "cell", "pretraining seed", "fine-tuning seed"), moved))
            else:
                title = spec.fig1["title"] if f == "fig1" else next(x.title for x in spec.figures if x.key == f)
                out.append(("table", title, FIG_HEAD, _fig_rows(fig_rows[f])))
        rows = [(c["label"], *_claim_text(claims[c["id"]])) for c in spec.claims if c["section"] == s["id"]]
        if rows:
            out.append(("table", "In-text numbers", ("claim", "value", "cells / per comparison"), rows))
    return out


def _md_cell(s: str) -> str:
    return s.replace("|", "\\|")


def to_md(bs: Sequence[tuple]) -> bytes:
    lines: list[str] = []
    for b in bs:
        if b[0] == "h1":
            lines += [f"# {b[1]}", ""]
        elif b[0] == "h2":
            lines += [f"## {b[1]}", ""]
        elif b[0] == "p":
            lines += [b[1], ""]
        elif b[0] == "md":
            lines += [b[1].rstrip("\n"), ""]
        else:
            _, cap, head, rows = b
            lines += ([f"**{cap}**", ""] if cap else []) + ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
            lines += ["| " + " | ".join(_md_cell(x) for x in r) + " |" for r in rows] + [""]
    return ("\n".join(lines).rstrip("\n") + "\n").encode("utf-8")


STYLE = ("body{font-family:system-ui,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;color:#222}"
         "table{border-collapse:collapse;margin:.5em 0 1.5em}td,th{border:1px solid #ccc;padding:.2em .5em;"
         "font-size:.9em}th{background:#f3f3f3}caption{text-align:left;font-weight:600;padding:.3em 0}")


def to_html(bs: Sequence[tuple], table1_html: str) -> bytes:
    e = html.escape
    parts = ['<!doctype html>', '<html lang="en">', '<head><meta charset="utf-8">',
             f"<title>{e(bs[0][1])}</title>", f"<style>{STYLE}</style>", "</head>", "<body>"]
    for b in bs:
        if b[0] in ("h1", "h2", "p"):
            parts.append(f"<{b[0]}>{e(b[1])}</{b[0]}>")
        elif b[0] == "md":
            parts.append(table1_html)
        else:
            _, cap, head, rows = b
            parts.append("<table>" + (f"<caption>{e(cap)}</caption>" if cap else "")
                         + "<thead><tr>" + "".join(f"<th>{e(h)}</th>" for h in head) + "</tr></thead><tbody>")
            parts += ["<tr>" + "".join(f"<td>{e(x)}</td>" for x in r) + "</tr>" for r in rows]
            parts.append("</tbody></table>")
    parts += ["</body>", "</html>"]
    return ("\n".join(parts) + "\n").encode("utf-8")

