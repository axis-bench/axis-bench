"""The paper-layout outputs of a re-evaluation, from the user's own results files.

Why: a re-evaluation reports what the user measured, laid out as the paper lays it out, and nothing
else -- no published value is read or compared. Each re-evaluated
result is read back from the results file `bench eval` wrote for its run (axis_qvla.bench.finish),
scored under the reporting rules (all-24), pooled exactly as T0 pools a published run
(axis_qvla.paper.pool: tasks in string order, the order `score` keeps), and rendered by T0's own
renderer (axis_qvla.paper.render). A cell that was not re-evaluated is absent from the pooled cells, so
it is an empty CSV field and a dash in Table I.

Core tier: stdlib only.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from axis_qvla.bench.finish import results_path
from axis_qvla.bench.results.format import load_results
from axis_qvla.bench.results.score import score
from axis_qvla.paper.cells import CellEntry
from axis_qvla.paper.pool import dump_pooled, pool_cell
from axis_qvla.paper.rates import se, sr
from axis_qvla.paper.reeval import ReevalError, Unit
from axis_qvla.paper.render import render_paper
from axis_qvla.paper.spec import PaperSpec


def collect(sel: Sequence[CellEntry], run_root: Path, bench) -> tuple[dict[str, dict], list[str]]:
    """(the pooled cells of every result whose results file is there, the cells still missing one).
    A file that is there but not a scorable results file of `bench` is refused with ValueError; so is
    one that lacks rollouts (it would score them as failures) or records other training seeds than
    the results map (Table I pairs replicates by the recorded seeds, not the file's). Every file is
    read before refusing, so one error names every bad file, not only the first."""
    pooled, missing, bad = {}, [], []
    for c in sel:
        path = results_path(run_root, c.run)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            missing.append(c.cell)
            continue
        try:
            r = load_results(data)
            s = score(r, bench)
        except ValueError as exc:
            bad.append(f"{path}: {exc}")
            continue
        if s.missing:
            bad.append(f"{path}: {s.missing} of {s.rollouts + s.missing} rollouts are missing; "
                       f"finish the evaluation of {c.run} or remove the file")
            continue
        got = dict(r.header.training_seed) if r.header.training_seed is not None else None
        want = dict(c.training_seed) if c.training_seed is not None else None
        if got != want:
            bad.append(f"{path}: records training seeds {got}, the results map {want} for {c.cell}")
            continue
        pooled[c.cell] = pool_cell(c.run, {t: (k, n) for t, k, n in s.tallies})
    if bad:
        raise ValueError("; ".join(bad))
    return pooled, missing


def tables(P: Mapping[str, dict], sel: Sequence[CellEntry], spec: PaperSpec) -> dict[str, bytes]:
    """T0's outputs (figures/*.csv, tables/table1.*, claims.json, results_page.*) over `P`, plus the
    pooled cells as pooled_reeval.json."""
    outputs, problems = render_paper(P, {c.cell: c.training_seed for c in sel if c.cell in P}, spec)
    if problems:
        raise ReevalError("; ".join(problems))
    return {**outputs, "pooled_reeval.json": dump_pooled(P)}


def summary(unit: Unit, outputs: Mapping[str, bytes], P: Mapping[str, dict]) -> str:
    """What the command prints at the end: Table I, a figure's data, or a result's all-24 SR."""
    if unit.kind == "table":
        return outputs["tables/table1.md"].decode("utf-8")
    if unit.kind == "figure":
        return outputs[f"figures/{unit.key}.csv"].decode("utf-8")
    return "\n".join(f"{cell}\tall-24 SR {sr(P, cell):.2f} ±{se(P, cell):.2f}" for cell in P)
