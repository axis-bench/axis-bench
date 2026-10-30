"""qvla.cells/1 validation: every malformed cell map is refused with a named problem."""
from __future__ import annotations

import copy
import json

import pytest

from axis_qvla.paper.cells import load_cells, validate_cells
from axis_qvla.core.canon import dumps_canonical
from tests.golden.paper.t0_fixture import cells_doc

GOOD = {"cell": "bc|bare", "run": "r1fbc", "box": "s2", "served": "r1fbc_dualsim", "backbone": "pi05",
        "training_seed": {"pretrain": None, "ft": 42}, "serve": {"fields": {}, "uncond_fields": None, "beta": 1.0},
        "eval_fingerprint": "a" * 64, "rollouts": {"path": "rollouts/s2/r1fbc_dualsim.tar", "sha256": "b" * 64, "size": 10},
        "results": None,
        "checkpoint": {"ref": "axisrobotics/qvla-v1-pi05@v1.0:ft/x/13579"},
        "eval_command": "qvla bench eval --run r1fbc",
        "lineage": {"train_config": None, "train_overrides": [], "step": "13579", "norm_stats_sha256": None,
                    "base": None, "init": None, "arm": "pi05.finetune.x"}}


def test_good_map_loads_and_keeps_list_order():
    other = dict(GOOD, cell="bct|cond", run="r1fbct", served="r1fbct_dualsim")
    doc = cells_doc([other, GOOD])
    cells = load_cells(dumps_canonical(doc))
    assert [c.cell for c in cells.cells] == ["bct|cond", "bc|bare"]
    assert validate_cells(json.loads(dumps_canonical(doc))) == []


@pytest.mark.parametrize("mutate,needle", [
    (lambda c: c.update(eval_fingerprint="abc"), "eval_fingerprint"),
    (lambda c: c["rollouts"].update(path="/abs/x.tar"), "safe relative path"),
    (lambda c: c["rollouts"].update(path="rollouts/../x.tar"), "safe relative path"),
    (lambda c: c["rollouts"].update(path="elsewhere/x.tar"), "under rollouts/"),
    (lambda c: c.pop("eval_command"), "fields must be exactly"),
    (lambda c: c.update(training_seed="42"), "training_seed"),
    (lambda c: c.update(training_seed={"ft": 42}), "training_seed"),
    (lambda c: c["serve"].pop("beta"), "serve"),
    (lambda c: c.update(manifest={"path": "runs/x.json"}), "manifest"),
    (lambda c: c.update(manifest=None), "manifest"),
    (lambda c: c.update(extra=1), "fields must be exactly"),
    (lambda c: c.update(box="s4"), "box"),
])
def test_bad_cells_are_named(mutate, needle):
    c = copy.deepcopy(GOOD)
    mutate(c)
    assert any(needle in p for p in validate_cells(cells_doc([c])))


def test_duplicates_schema_and_bench_are_checked():
    assert "cells: duplicate cell values" in validate_cells(cells_doc([GOOD, dict(GOOD)]))
    assert validate_cells(dict(cells_doc([GOOD]), schema="x")) == ["schema must be qvla.cells/1"]
    bad = cells_doc([GOOD]); bad["bench"]["units_sha256"] = "z"
    assert "bench.units_sha256: must be 64 hex" in validate_cells(bad)
    with pytest.raises(ValueError, match="invalid cells.json"):
        load_cells(json.dumps(bad).encode())
