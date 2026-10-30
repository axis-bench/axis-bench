"""The cell map lives beside the paper layout that reads it."""
from __future__ import annotations

from pathlib import Path

from axis_qvla.paper import cli as paper_cli

REPO = Path(__file__).resolve().parents[3]


def test_the_cell_map_is_with_the_paper_spec():
    assert paper_cli.DEFAULT_CELLS == REPO / "experiments" / "paper" / "cells.json"
    assert (REPO / "experiments" / "paper" / "cells_runs.json").is_file()
    assert not (REPO / "conf" / "cells").exists()
