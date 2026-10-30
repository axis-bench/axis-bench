"""T0 gives the same bytes on every Python >= 3.11: no version-sensitive float arithmetic on its path.

Python 3.12 made the builtin sum() over floats compensated (Neumaier), which moved T0's outputs
off their 3.11 goldens. Every float sum on the
T0 path -- `paper tables` and the results scorer, found by walking their import closure here --
goes through ltr_sum, a plain left-to-right loop (exactly 3.11's sum). This ratchet keeps it so:
a builtin sum() is allowed only at the listed call sites, each provably over ints, and the other
version-sensitive numeric helpers only where listed with the reason they are stable.
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import axis_qvla.paper.cli as _entry

ROOT = _entry.__name__.split(".")[0]
PKG = Path(importlib.import_module(ROOT).__file__).parent
ENTRIES = ("paper.cli", "bench.results.cli", "bench.results.score")
# Where the walk stops, with why: `paper reeval`'s runner and host checks, which paper.cli imports
# inside `paper reeval`'s handler only. They start and time `bench eval` children and compute no
# T0 output (the re-evaluation's tables go through paper.reeval_tables, which IS walked).
OFF_T0 = {"paper.reeval_run"}

# Builtin sum() call sites on the T0 path, by module and exact source: each sums ints.
INT_SUMS = {
    ("paper/pool.py", "sum(n for _, n in vals)"),               # seed counts
    ("paper/pool.py", "sum(s for s, _ in vals)"),               # success counts
    ("paper/claims.py", "sum(1 for s in signs if s == 1)"),     # sign counts
    ("paper/claims.py", "sum(1 for s in signs if s == 0)"),
    ("paper/claims.py", "sum(1 for s in signs if s == -1)"),
    ("paper/claims.py", "sum(1 for s in signs if s is None)"),
    ("bench/results/score.py", "sum(1 for s in seeds if (t, s) in got)"),
    ("bench/results/score.py", "sum(1 for s in seeds if got.get((t, s)) is True)"),
    ("bench/results/score.py", "sum(n for _, n in tallies.values())"),
    ("bench/results/seedfiles.py", 'sum(1 for f in files if f.record().get("success") is True)'),
}
# Other numeric helpers whose result could differ across versions, with why each listed use is
# stable on 3.11-3.13: statistics.stdev computes in exact fractions and rounds the square root
# correctly (identical source in 3.11, 3.12 and 3.13 bar a 3.13 inf/nan guard).
REVIEWED = {("bench/results/score.py", "statistics.stdev(diffs)"),
            # The label study runs as its own command: `paper label-study` imports numpy inside its
            # rank-correlation functions only, so `paper tables` never loads it.
            ("paper/fig4.py", "import numpy as np")}
SENSITIVE_ATTRS = {("math", "fsum"), ("math", "fmean"), ("math", "sumprod"), ("math", "prod")}


def _closure() -> dict[str, ast.Module]:
    seen: dict[str, ast.Module] = {}
    todo = list(ENTRIES)
    while todo:
        mod = todo.pop()
        if mod in seen or mod in OFF_T0:
            continue
        rel = Path(*mod.split("."))
        path = next((p for p in (PKG / rel.with_suffix(".py"), PKG / rel / "__init__.py") if p.is_file()), None)
        if path is None:
            continue
        tree = seen[mod] = ast.parse(path.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            names = ([a.name for a in n.names] if isinstance(n, ast.Import) else
                     [n.module] + [f"{n.module}.{a.name}" for a in n.names]
                     if isinstance(n, ast.ImportFrom) and n.module and n.level == 0 else [])
            todo += [m[len(ROOT) + 1:] for m in names if m.startswith(ROOT + ".")]
    return seen


def _rel(mod: str) -> str:
    rel = Path(*mod.split("."))
    return (rel.with_suffix(".py") if (PKG / rel.with_suffix(".py")).is_file() else rel / "__init__.py").as_posix()


def findings(rel: str, src: str) -> set[tuple[str, str, str, int]]:
    """(kind, module, source, line) of every version-sensitive construct in one module, before any
    allow-list. "sum": a builtin sum() call (allowed only at INT_SUMS). Everything else must be in
    REVIEWED: a statistics.* or math.fsum-like call; `sum` used as a value (map(sum, rows), f = sum);
    builtins.sum or any builtins import; `from statistics/math import ...`; a statistics/math alias
    (`import statistics as st`); any numpy import."""
    tree = ast.parse(src)
    funcs = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    out = set()

    def add(kind: str, n: ast.AST) -> None:
        out.add((kind, rel, ast.get_source_segment(src, n), n.lineno))

    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name) and f.id == "sum":
                add("sum", n)
            elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and (
                    f.value.id == "statistics" or (f.value.id, f.attr) in SENSITIVE_ATTRS):
                add("numeric", n)
        elif isinstance(n, ast.Name) and n.id == "sum" and isinstance(n.ctx, ast.Load) and id(n) not in funcs:
            add("sum-value", n)
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "builtins":
            add("builtins", n)
        elif isinstance(n, ast.Import):
            for al in n.names:
                base = al.name.split(".")[0]
                if base in ("numpy", "builtins") or (base in ("statistics", "math") and al.asname):
                    add(f"import-{base}", n)
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            if n.module.split(".")[0] in ("numpy", "builtins", "statistics", "math"):
                add(f"from-{n.module.split('.')[0]}", n)
    return out


def _all_findings() -> set[tuple[str, str, str, int]]:
    out = set()
    for mod in _closure():
        out |= findings(_rel(mod), (PKG / _rel(mod)).read_text(encoding="utf-8"))
    return out


def test_the_closure_is_the_t0_path():
    mods = set(_closure())
    assert {"paper.pool", "paper.rates", "paper.table1", "paper.claims", "paper.figures", "paper.render",
            "bench.results.score", "paper.t0"} <= mods
    assert len(mods) >= 25
    assert "paper.reeval_tables" in mods and not mods & OFF_T0


def test_no_builtin_sum_on_the_t0_path_except_listed_int_sums():
    found = {(rel, seg) for kind, rel, seg, _ in _all_findings() if kind == "sum"}
    assert found - INT_SUMS == set(), "a builtin sum() on the T0 path: use ltr_sum, or list a provably-int sum"
    assert INT_SUMS - found == set(), "a listed int sum no longer exists: drop it from INT_SUMS"


def test_no_other_version_sensitive_construct_unless_reviewed():
    assert {(rel, seg) for kind, rel, seg, _ in _all_findings() if kind != "sum"} == REVIEWED


# Follow-up 4: every way to reach a version-sensitive sum or mean without writing `sum(` or
# `statistics.` -- each line from 2 on must be flagged.
SCRATCH = """x = 1
from statistics import fmean
import statistics as st
from math import fsum
import builtins
import numpy as np
from numpy import mean
t = builtins.sum([0.1])
rows = list(map(sum, [[0.1]]))
f = sum
from builtins import sum as sum_alias
import math as m
"""


def test_the_ratchet_flags_every_form_of_reaching_a_version_sensitive_numeric():
    lines = {line for *_, line in findings("scratch.py", SCRATCH)}
    assert lines == set(range(2, len(SCRATCH.splitlines()) + 1))


def test_ltr_sum_is_the_311_float_sum_on_every_version():
    from axis_qvla.core.ltr import ltr_sum

    assert ltr_sum([0.1] * 10) == 0.9999999999999999      # 3.12's builtin sum gives 1.0
    assert ltr_sum([1e16, 1.0, -1e16]) == 0.0             # compensated summation gives 1.0
    assert ltr_sum(x / 10 for x in range(4)) == ((0 + 0.0) + 0.1) + 0.2 + 0.3   # generators too
    acc = 0
    for x in [0.7, 0.1, 0.2, 1e-17, 3.3]:
        acc = acc + x
    assert ltr_sum([0.7, 0.1, 0.2, 1e-17, 3.3]) == acc
    assert ltr_sum([]) == 0 and ltr_sum([2, 3]) == 5 and ltr_sum([], 1.5) == 1.5
