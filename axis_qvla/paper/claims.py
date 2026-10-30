"""claims.json: the in-text numbers and counts of Secs. IV.B and V.B-V.F, recomputed.

Why: besides its figures and Table I, the paper states percentage-point differences ("by 1.08 and
5.75 pp"), relative gains ("by 1.02%, 73.77%, and 8.09% relative"), success rates ("58.00% versus
57.42%") and counts ("six of nine", "15 of 18"). Each is a claim of experiments/paper/paper.json, computed
here by kind:

  diff                    SR(a) - SR(b), percentage points
  rel                     100 (SR(a) / SR(ref) - 1)
  sr                      SR(a)
  means                   the mean SR of each named group of cells
  signs                   per (a, b) pair: the difference and its exact sign, plus the counts
  replicate_signs         the same over one VLA's training seeds, for two Table I roles
  table1_mean             the equal average of one Table I comparison over all VLA-seed values
  table1_signs            the exact signs of every seed-level value of the listed comparisons
  table1_vla_mean_signs   the per-VLA means of the listed comparisons, with exact signs

Values are unrounded floats; `text` is the value at the paper's printed precision. A sign is
decided on exact fractions (axis_qvla.paper.rates), so a tie is a tie. A missing cell gives a None value
and is counted as `missing`.

Core tier: stdlib only.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction

from axis_qvla.paper.rates import as_float, diff, exact, fmt, mean, rel, sign, sign_of, sr
from axis_qvla.paper.spec import PaperSpec
from axis_qvla.paper.table1 import Table1, as_json
from axis_qvla.core.canon import dumps_canonical

SCHEMA = "qvla.paper_claims/1"


def _num(value: float | None, dp: int, signed: bool) -> dict:
    return {"value": value, "dp": dp, "text": fmt(value, dp, signed=signed)}


def _counts(signs: Sequence[int | None]) -> dict:
    return {"positive": sum(1 for s in signs if s == 1), "zero": sum(1 for s in signs if s == 0),
            "negative": sum(1 for s in signs if s == -1), "missing": sum(1 for s in signs if s is None)}


def _pairs(P: Mapping[str, dict], pairs: Sequence[tuple[str, str, str]]) -> dict:
    out = [{"label": lab, "cells": [a, b], "diff": diff(P, a, b), "sign": sign(P, a, b)} for lab, a, b in pairs]
    return {"pairs": out, **_counts([p["sign"] for p in out])}


def _seed_values(t: Table1, spec: PaperSpec, rows: Sequence[str]) -> list[dict]:
    return [{"comparison": r, "vla": v.key, "seed": s, "value": x, "sign": g}
            for r in rows for v in spec.vlas
            for s, x, g in zip(t.entries[(r, v.key)].seeds, t.entries[(r, v.key)].values, t.entries[(r, v.key)].signs)]


def _mean_sr(P: Mapping[str, dict], cells: Sequence[str]) -> float | None:
    m = mean([exact(P, x) for x in cells])
    return None if m is None else float(100 * m)


def _vla_mean_sign(t_exact: Mapping[tuple[str, str], Sequence[Fraction | None]], row: str, vla: str) -> int | None:
    """The exact sign of a Table I mean (of its exact per-seed values)."""
    m = mean(list(t_exact[(row, vla)]))
    return sign_of(m)


def _one(P: Mapping[str, dict], t: Table1, spec: PaperSpec, c: Mapping) -> dict:
    k = c["kind"]
    if k == "diff":
        return {"cells": [c["a"], c["b"]], **_num(diff(P, c["a"], c["b"]), c["dp"], True)}
    if k == "rel":
        return {"cells": [c["a"], c["ref"]], **_num(rel(P, c["a"], c["ref"]), c["dp"], True)}
    if k == "sr":
        return {"cells": [c["a"]], **_num(sr(P, c["a"]), c["dp"], False)}
    if k == "means":
        return {"groups": {g: {"cells": list(cs), **_num(_mean_sr(P, cs), c["dp"], False)}
                           for g, cs in c["groups"].items()}}
    if k == "signs":
        return _pairs(P, [(p["label"], p["a"], p["b"]) for p in c["pairs"]])
    if k == "replicate_signs":
        return _pairs(P, [(f"seed {r['seed']}", r["cells"][c["a"]], r["cells"][c["b"]])
                          for r in spec.table1["replicates"][c["vla"]]])
    if k == "table1_mean":
        vals = [x for v in spec.vlas for x in t.entries[(c["row"], v.key)].exact]
        return _num(as_float(mean(vals)), c["dp"], True)
    if k == "table1_signs":
        vals = _seed_values(t, spec, c["rows"])
        return {"values": vals, **_counts([v["sign"] for v in vals])}
    t_exact = {key: e.exact for key, e in t.entries.items()}
    vals = [{"comparison": r, "vla": v.key, "value": t.entries[(r, v.key)].mean, "sign": _vla_mean_sign(t_exact, r, v.key)}
            for r in c["rows"] for v in spec.vlas]
    return {"values": vals, **_counts([v["sign"] for v in vals])}      # table1_vla_mean_signs


def compute(P: Mapping[str, dict], t: Table1, spec: PaperSpec) -> dict[str, dict]:
    return {c["id"]: {"section": c["section"], "kind": c["kind"], "label": c["label"], **_one(P, t, spec, c)}
            for c in spec.claims}


def dump(claims: Mapping[str, dict], t: Table1, spec: PaperSpec, *, n_cells: int) -> bytes:
    return dumps_canonical({"schema": SCHEMA, "paper": spec.paper, "n_cells": n_cells, "claims": dict(claims),
                            "table1": as_json(t, spec)})
