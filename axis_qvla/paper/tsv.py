"""Per-task TSV rows: the one table between seed files and the pooled results.

Why: the pooled numbers depend on the ORDER tasks are added (float summation), and the paper's
order is the string order of the task ids ('1050' < '806'). This module writes and reads the
6-column format (64-hex fingerprint).

Core tier: stdlib only.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from pathlib import Path

TSV_HEADER = "\t".join(("run", "served", "task", "succ", "tot", "fingerprint"))


@dataclasses.dataclass(frozen=True)
class TsvRow:
    run: str
    served: str
    task: int
    succ: int
    tot: int
    fingerprint: str


def rows_for_run(run: str, served: str, tallies: Mapping[int, tuple[int, int]],
                 fingerprint: str) -> list[TsvRow]:
    return [TsvRow(run, served, int(t), int(tallies[t][0]), int(tallies[t][1]), fingerprint)
            for t in sorted(tallies, key=str)]


def write_tsv(rows: Iterable[TsvRow]) -> bytes:
    lines = [TSV_HEADER] + [f"{r.run}\t{r.served}\t{r.task}\t{r.succ}\t{r.tot}\t{r.fingerprint}"
                            for r in rows]
    return ("\n".join(lines) + "\n").encode("utf-8")


def read_tsv(path: Path) -> list[TsvRow]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != TSV_HEADER:
        raise ValueError(f"{path}: unknown TSV header {lines[:1]!r}")
    out: list[TsvRow] = []
    for n, line in enumerate(lines[1:], 2):
        p = line.split("\t")
        if len(p) != 6:
            raise ValueError(f"{path}:{n}: expected 6 columns, got {len(p)}")
        out.append(TsvRow(p[0], p[1], int(p[2]), int(p[3]), int(p[4]), p[5]))
    return out


def tallies_by_run(rows: Iterable[TsvRow]) -> dict[str, dict[int, tuple[int, int]]]:
    out: dict[str, dict[int, tuple[int, int]]] = {}
    for r in rows:
        out.setdefault(r.run, {})[r.task] = (r.succ, r.tot)
    return out
