"""T0 runs from a checkout. Every default input -- the cell map, the table
layout, the page template, the benchmark declaration and the release lock -- resolves against the
source tree, the lock through ONE helper shared by `paper tables` and `release verify`; a tree
without conf/ (what a non-editable `pip install .` leaves) exits 3 and says how to run T0.
"""
from __future__ import annotations

import pathlib

from axis_qvla.cli import build_parser, main
from axis_qvla.core import lock as lk
from axis_qvla.paper import cli as pcli


def test_paper_tables_and_release_verify_default_to_the_same_source_tree_lock(tmp_path, monkeypatch):
    assert lk.SOURCE_TREE == pathlib.Path(pcli.__file__).resolve().parents[2]
    assert lk.default_lock() == lk.SOURCE_TREE / "conf" / "release" / "v1.0.lock.json"
    monkeypatch.chdir(tmp_path)                                   # never the working directory
    p = build_parser()
    tables = p.parse_args(["paper", "tables", "--results", "x"])
    verify = p.parse_args(["data", "verify"])
    assert pathlib.Path(tables.lock) == pathlib.Path(verify.lock) == lk.default_lock()


def test_outside_a_checkout_t0_exits_3_and_says_how_to_run_it(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pcli, "REPO", tmp_path / "site-packages")
    assert main(["paper", "tables", "--results", str(tmp_path)]) == 3
    out = capsys.readouterr().out
    assert out.startswith("FETCH  ") and "checkout" in out and "pip install -e ." in out
    assert "paper tables" in out and "not `pip install .`" in out
