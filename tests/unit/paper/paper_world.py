"""Synthetic pooled cells over the real paper layout, for the renderer unit tests."""
from __future__ import annotations

from collections.abc import Callable

from axis_qvla.paper.pool import pool_cell
from axis_qvla.paper.spec import PaperSpec, load_spec

TASKS = [806, 807, 808, 809, 855, 878, 882, 888, 890, 945, 953, 966, 1050, 1223, 1231, 1426, 1435, 1436,
         1438, 1459, 1525, 1527, 1632, 1889]


def tallies(successes: int, n: int = 50) -> dict[int, tuple[int, int]]:
    """`successes` spread over the 24 tasks (front-loaded), inserted in task-id string order."""
    out, left = {}, successes
    for t in sorted(TASKS, key=str):
        k = min(n, left)
        out[t] = (k, n)
        left -= k
    assert left == 0
    return out


def pooled(successes: Callable[[str], int | None], spec: PaperSpec | None = None) -> dict[str, dict]:
    """Every cell of the layout, pooled from `successes(cell)` of 1,200 trials (None: cell absent)."""
    spec = spec or load_spec()
    out = {}
    for cell in sorted(spec.cells()):
        k = successes(cell)
        if k is not None:
            out[cell] = pool_cell(f"r_{cell}", tallies(k))
    return out


def lineage(spec: PaperSpec | None = None) -> dict[str, dict]:
    """{cell: training_seed}: each Table I cell pretrained with its replicate's seed."""
    spec = spec or load_spec()
    out = {}
    for reps in spec.table1["replicates"].values():
        for r in reps:
            for cell in r["cells"].values():
                out[cell] = {"pretrain": r["seed"], "ft": r["seed"]}
    return out
