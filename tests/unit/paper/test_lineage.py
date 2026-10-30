"""The lineage block of a cell is validated like every other field."""
from __future__ import annotations

import copy

import pytest

from axis_qvla.paper.cells import validate_cells
from axis_qvla.paper.lineage import lineage_problems, replicate_table
from tests.golden.paper.t0_fixture import cells_doc
from tests.unit.paper.test_cells import GOOD

INIT = {"id": "pi05/pretrain/p", "ref": "axisrobotics/qvla-v1-pi05@v1.0:pretrain/p/20604", "pub_digest": "e" * 64,
        "training_seed": 42, "train_config": "cfg_pre", "train_overrides": ["--lr-schedule.warmup-steps=2060"],
        "step": "20604", "norm_stats_sha256": "f" * 64, "base": "pi05_droid"}


def _cell(**lineage):
    c = copy.deepcopy(GOOD)
    c["lineage"].update(lineage)
    return c


def test_a_cell_with_and_without_an_init_checkpoint_validates():
    assert validate_cells(cells_doc([GOOD])) == []
    assert validate_cells(cells_doc([_cell(init=INIT)])) == []


@pytest.mark.parametrize("lin, needle", [
    ({"init": {**INIT, "pub_digest": "x"}}, "pub_digest"),
    ({"init": {k: v for k, v in INIT.items() if k != "ref"}}, "init"),
    ({"train_overrides": "--seed=43"}, "train_overrides"),
    ({"step": 13579}, "step"),
    ({"norm_stats_sha256": "abc"}, "norm_stats_sha256"),
    ({"arm": None}, "lineage.arm"),
    ({"arm": "pi05.pretrain.p"}, "lineage.arm"),
])
def test_a_bad_lineage_is_named(lin, needle):
    assert any(needle in p for p in validate_cells(cells_doc([_cell(**lin)])))


def test_a_missing_lineage_is_refused():
    c = copy.deepcopy(GOOD)
    del c["lineage"]
    assert any("fields must be exactly" in p for p in validate_cells(cells_doc([c])))
    assert lineage_problems(0, None) == ["cells[0].lineage: needs exactly ['train_config', 'train_overrides', 'step', 'norm_stats_sha256', 'base', 'init', 'arm']"]


def test_the_replicate_table_lists_each_pi05_cell_with_its_seeds():
    cells = [dict(GOOD, cell="a|bare", lineage={**GOOD["lineage"], "train_config": "cfg_a", "init": INIT},
                  training_seed={"pretrain": 42, "ft": 42}),
             dict(GOOD, cell="b|bare", backbone="smolvla")]
    table = replicate_table(cells)
    assert "| `a\\|bare` | 42 | 42 | `cfg_a` | `pi05/pretrain/p` |" in table and "b\\|bare" not in table
