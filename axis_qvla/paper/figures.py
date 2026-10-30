"""The data of Figs. 1, 4(c), 5, 6 and 7, one CSV per figure.

Why: the figures are drawn elsewhere, so T0 verifies each
figure through its data. A row is one bar: the VLA, its group and label, the cells it reads, SR,
the reference SR_BC (the VLA's 5K plain BC with plain FT and untagged inference), the relative
improvement 100(SR/SR_BC - 1) from the unrounded rates (exact, axis_qvla.paper.rates), and the
marginal SE scaled by 100/SR_BC (Appendix A). Fig. 1 is Table I's first comparison: one row per
training seed, 100(SR_A - SR_B)/SR_BC with that seed's BC and the SE of that difference,
100 sqrt(SE_A^2 + SE_B^2)/SR_BC, then the mean over seeds -- the value Fig. 1 prints. SR is the
exact rate rounded once (axis_qvla.paper.rates), so 100(sr/sr_bc - 1) recomputed from a row gives its
`rel` to the last bits, a tie exactly.

Floats are written with `repr` (the shortest round-trip form) and a missing value as an empty
field, so the bytes are the same on every run, Python version and PYTHONHASHSEED. Labels are
checked for commas and quotes when the layout loads, so no field needs quoting.

Core tier: stdlib only.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

from axis_qvla.paper.rates import rel, scaled, se, se_diff, sr
from axis_qvla.paper.spec import Figure, PaperSpec
from axis_qvla.paper.table1 import Table1

COLUMNS = ("vla", "group", "label", "cells", "sr", "sr_bc", "rel", "se_rel", "note")


@dataclasses.dataclass(frozen=True)
class Row:
    vla: str
    group: str
    label: str
    cells: tuple[str, ...]
    sr: float | None
    sr_bc: float | None
    rel: float | None
    se_rel: float | None
    note: str = ""


def figure_rows(P: Mapping[str, dict], spec: PaperSpec, fig: Figure) -> list[Row]:
    out = []
    for b in fig.bars:
        v = spec.vla(b.vla)
        ref = sr(P, v.bc)
        out.append(Row(v.label, b.group, b.label, (b.cell,), sr(P, b.cell), ref, rel(P, b.cell, v.bc),
                       scaled(se(P, b.cell), ref), b.note))
    return out


def fig1_rows(P: Mapping[str, dict], t: Table1, spec: PaperSpec) -> list[Row]:
    row = next(r for r in spec.table1["rows"] if r["id"] == spec.fig1["row"])
    out = []
    for v in spec.vlas:
        e = t.entries[(row["id"], v.key)]
        for rep, value in zip(spec.table1["replicates"][v.key], e.values):
            a, bc = rep["cells"][row["a"]], rep["cells"]["bc"]
            b = rep["cells"][row["b"]]
            out.append(Row(v.label, row["label"], f"seed {rep['seed']}", (a, b),
                           sr(P, a), sr(P, bc), value, scaled(se_diff(P, a, b), sr(P, bc))))
        out.append(Row(v.label, row["label"], "mean over seeds", (), None, None, e.mean, None, "the value Fig. 1 prints"))
    return out


def _field(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return repr(x)
    if isinstance(x, tuple):
        return ";".join(x)
    return str(x)


def to_csv(rows: Sequence[Row]) -> bytes:
    lines = [",".join(COLUMNS)] + [",".join(_field(getattr(r, c)) for c in COLUMNS) for r in rows]
    return ("\n".join(lines) + "\n").encode("utf-8")
