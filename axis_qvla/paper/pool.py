"""Pooled-cell arithmetic and the exact bytes of `pooled_r1.json`.

Why: the pooled file T0 writes (pooled_r1.json) is the source of every number the paper prints.
The arithmetic, one run per result: the per-task success rates are averaged in the order the
tallies were inserted (task-id STRING order), so the float bits of `agg` and `se` match only if
that order is kept. Never sort tasks numerically here.
A cell whose run is missing or does not cover exactly `n_tasks` tasks is an error, not a
partial cell: every paper cell is one complete 24 x 50 read.

Core tier: stdlib only.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence

from axis_qvla.core.ltr import ltr_sum


class PoolError(ValueError):
    """A cell cannot be pooled: its run is absent or incomplete."""


def pool_cell(run: str, tallies: Mapping[int, tuple[int, int]]) -> dict:
    vals = list(tallies.values())
    ps = [s / n for s, n in vals]
    agg = 100 * ltr_sum(ps) / len(ps)
    var = ltr_sum(p * (1 - p) / n for p, (s, n) in zip(ps, vals))
    se = 100 * math.sqrt(var) / len(ps)
    return {"agg": agg, "se": se, "n_tasks": len(ps), "tot": sum(n for _, n in vals),
            "suc": sum(s for s, _ in vals), "reads": 1, "runs": [run], "partial": [],
            "per_task": {str(t): [s, n] for t, (s, n) in tallies.items()}}


def pool_cells(cells: Sequence[tuple[str, str]], data: Mapping[str, Mapping[int, tuple[int, int]]],
               *, n_tasks: int = 24) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for cell, run in cells:
        if run not in data:
            raise PoolError(f"cell {cell}: run {run} has no rows")
        if len(data[run]) != n_tasks:
            raise PoolError(f"cell {cell}: run {run} covers {len(data[run])} tasks, need {n_tasks}")
        out[cell] = pool_cell(run, data[run])
    return out


def dump_pooled(pooled: Mapping[str, dict]) -> bytes:
    return json.dumps(pooled, indent=1).encode("utf-8")
