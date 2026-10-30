"""Table I, exactly as the paper defines it, and its three renderings (.md, .tex, .html).

Why: Table I reports, for each comparison A - B and each VLA, the mean and SAMPLE SD over three
training seeds of 100 (SR_A - SR_B) / SR_BC, where SR_BC is plain BC (plain FT, untagged inference)
AT THAT SEED (Appendix A). The layout names, per VLA and seed, the cell playing each role. A seed
is a training replicate keyed by the PRETRAIN seed: 42/43/44 for π0.5,
0/1/2 for SmolVLA and X-VLA. Each cell's recorded lineage {pretrain, ft} (cells.json) must carry
that pretrain seed, or T0 reports an integrity problem (so does a null lineage); its FT seed is
metadata, kept in `replicates` (and disclosed by the results page), never a reason to drop a
pair. Values and means are exact (axis_qvla.paper.rates) until they are rounded to floats once;
`exact` keeps the per-seed fractions for the claims that average them.

Core tier: stdlib only.
"""
from __future__ import annotations

import dataclasses
import html
from collections.abc import Mapping

from fractions import Fraction

from axis_qvla.paper.rates import as_float, fmt, mean, norm_diff, sample_sd, shown, sign
from axis_qvla.paper.spec import PaperSpec


@dataclasses.dataclass(frozen=True)
class Entry:
    seeds: tuple[int, ...]
    values: tuple[float | None, ...]
    signs: tuple[int | None, ...]
    mean: float | None
    sd: float | None
    exact: tuple[Fraction | None, ...] = ()

    def as_json(self) -> dict:
        return {"seeds": list(self.seeds), "values": list(self.values), "signs": list(self.signs),
                "mean": self.mean, "sd": self.sd}


@dataclasses.dataclass(frozen=True)
class Table1:
    entries: Mapping[tuple[str, str], Entry]          # (row key, vla key) -> entry
    replicates: Mapping[str, tuple[dict, ...]]          # vla -> ({seed, cells: {role: {cell, lineage}}}, ...)
    problems: tuple[str, ...]


def compute(P: Mapping[str, dict], spec: PaperSpec, lineage: Mapping[str, object]) -> Table1:
    t1, entries, reps, problems = spec.table1, {}, {}, []
    for v in spec.vlas:
        rs = t1["replicates"][v.key]
        for r in rs:
            for role, cell in r["cells"].items():
                if cell not in lineage:            # not in the cell map: an empty entry, not a problem
                    continue
                lin = lineage[cell]
                what = (f"whose pretrain seed is {lin.get('pretrain')}" if isinstance(lin, dict)
                        else "whose lineage is not recorded")
                if not isinstance(lin, dict) or lin.get("pretrain") != r["seed"]:
                    problems.append(f"Table I: {v.label} replicate {r['seed']} (role {role}) is cell {cell}, {what}")
        reps[v.key] = tuple({"seed": r["seed"], "cells": {role: {"cell": c, "lineage": lineage.get(c)}
                                                          for role, c in r["cells"].items()}} for r in rs)
        for row in t1["rows"]:
            pairs = [(r["cells"][row["a"]], r["cells"][row["b"]], r["cells"]["bc"]) for r in rs]
            ex = tuple(norm_diff(P, a, b, bc) for a, b, bc in pairs)
            entries[(row["id"], v.key)] = Entry(tuple(r["seed"] for r in rs), tuple(as_float(x) for x in ex),
                                                 tuple(sign(P, a, b) for a, b, _ in pairs),
                                                 as_float(mean(ex)), sample_sd(ex), ex)
    return Table1(entries, reps, tuple(problems))


def _cell(e: Entry) -> str:
    if e.mean is None or e.sd is None:
        return "—"
    return f"{shown(fmt(e.mean, 2, signed=True))} ± {fmt(e.sd, 2)}"


def caption(spec: PaperSpec) -> str:
    return spec.table1["title"]


def render_md(t: Table1, spec: PaperSpec) -> bytes:
    head = "| Comparison | " + " | ".join(v.label for v in spec.vlas) + " |"
    lines = [f"**{caption(spec)}**", "", head, "|---|" + "---:|" * len(spec.vlas)]
    for row in spec.table1["rows"]:
        lines.append(f"| {row['label']} | " + " | ".join(_cell(t.entries[(row['id'], v.key)]) for v in spec.vlas) + " |")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _tex(s: str) -> str:
    s = s.replace("\\", "\\textbackslash{}")
    for a, b in (("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#")):
        s = s.replace(a, b)
    return s.replace("π0.5", "$\\pi_{0.5}$").replace(" − ", " $-$ ").replace("−", "$-$")


def render_tex(t: Table1, spec: PaperSpec) -> bytes:
    lines = [f"% {_tex(caption(spec))}", "\\begin{tabular}{l" + "r" * len(spec.vlas) + "}", "\\toprule",
             "Comparison & " + " & ".join(_tex(v.label) for v in spec.vlas) + " \\\\", "\\midrule"]
    for row in spec.table1["rows"]:
        cells = []
        for v in spec.vlas:
            e = t.entries[(row["id"], v.key)]
            cells.append("--" if e.mean is None or e.sd is None else f"${fmt(e.mean, 2, signed=True)} \\pm {fmt(e.sd, 2)}$")
        lines.append(f"{_tex(row['label'])} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return ("\n".join(lines) + "\n").encode("utf-8")


def html_table(t: Table1, spec: PaperSpec) -> str:
    e = html.escape
    head = "<tr><th>Comparison</th>" + "".join(f"<th>{e(v.label)}</th>" for v in spec.vlas) + "</tr>"
    body = ["<tr><td>" + e(row["label"]) + "</td>" + "".join(f"<td>{e(_cell(t.entries[(row['id'], v.key)]))}</td>"
                                                            for v in spec.vlas) + "</tr>" for row in spec.table1["rows"]]
    return (f'<table class="table1">\n<caption>{e(caption(spec))}</caption>\n<thead>{head}</thead>\n<tbody>\n'
            + "\n".join(body) + "\n</tbody>\n</table>")


def render_html(t: Table1, spec: PaperSpec) -> bytes:
    return (html_table(t, spec) + "\n").encode("utf-8")


def as_json(t: Table1, spec: PaperSpec) -> dict:
    """Every per-seed value behind the table, with each cell's recorded lineage (claims.json)."""
    return {"comparisons": [{"id": r["id"], "label": r["label"], "a": r["a"], "b": r["b"]} for r in spec.table1["rows"]],
            "vlas": {v.key: {"label": v.label, "replicates": list(t.replicates[v.key]),
                             "entries": {r["id"]: t.entries[(r["id"], v.key)].as_json()
                                         for r in spec.table1["rows"]}} for v in spec.vlas}}
