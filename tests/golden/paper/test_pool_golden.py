"""The committed pooled file is rebuilt byte-for-byte from the golden TSVs and the cell-run map."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from axis_qvla.paper.pool import PoolError, dump_pooled, pool_cell, pool_cells
from axis_qvla.paper.tsv import read_tsv, tallies_by_run

REPO = Path(__file__).resolve().parents[3]
GOLD = Path(__file__).resolve().parent
POOLED_R1_SHA256 = "7269540a6bdb0f515762438356f8e6c1114dd8e4a947320466dceaca4264e7dd"


def _data() -> tuple[dict, dict]:
    data, box = {}, {}
    for b in ("s2", "s3"):
        t = tallies_by_run(read_tsv(GOLD / f"pt_r1_{b}.tsv"))
        box.update({run: b for run in t})
        data.update(t)
    return data, box


def _cells() -> list[dict]:
    doc = json.loads((REPO / "experiments" / "paper" / "cells_runs.json").read_text())
    assert doc["schema"] == "qvla.cell_runs/1"
    return doc["cells"]


def test_golden_pooled_bytes():
    data, _ = _data()
    out = pool_cells([(c["cell"], c["run"]) for c in _cells()], data)
    assert hashlib.sha256(dump_pooled(out)).hexdigest() == POOLED_R1_SHA256
    assert dump_pooled(out) == (GOLD / "pooled_r1.json").read_bytes()


def test_cell_run_map_matches_the_committed_pool():
    cells = _cells()
    golden = json.loads((GOLD / "pooled_r1.json").read_text())
    assert [c["cell"] for c in cells] == list(golden)
    assert [[c["run"]] for c in cells] == [v["runs"] for v in golden.values()]
    _, box = _data()
    assert set(box) == {c["run"] for c in cells}
    assert all(c["box"] == box[c["run"]] for c in cells)
    assert (sum(c["box"] == "s2" for c in cells), sum(c["box"] == "s3" for c in cells)) == (62, 61)
    assert len({c["run"] for c in cells}) == 123


def test_task_order_changes_the_float_bits_so_it_is_never_sorted():
    a = pool_cell("r", {1050: (48, 50), 806: (1, 50), 953: (7, 50)})
    assert list(a["per_task"]) == ["1050", "806", "953"]


def test_missing_or_incomplete_runs_raise():
    with pytest.raises(PoolError, match="has no rows"):
        pool_cells([("c|bare", "absent")], {})
    with pytest.raises(PoolError, match="covers 1 tasks"):
        pool_cells([("c|bare", "r")], {"r": {806: (1, 2)}})
