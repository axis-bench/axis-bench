"""The fixture pack's results files are accepted by `results score` and `results contrast`: they
carry the fixture declaration's real digest, so the scorer's identity check passes on them."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from axis_qvla.cli import main

FIX = Path(__file__).resolve().parent / "fixture_pack"


def _files() -> tuple[str, str]:
    found = {p.name: str(p) for p in (FIX / "pack" / "results").rglob("*.json")}
    return found["r1fbc_dualsim.json"], found["r1fcfg_dualsim.json"]


def test_the_fixture_results_carry_the_fixture_declarations_digest():
    want = hashlib.sha256((FIX / "conf" / "fixture_bench.json").read_bytes()).hexdigest()
    for f in _files():
        assert json.loads(Path(f).read_bytes())["header"]["bench"]["declaration_sha256"] == want
    for cells in (FIX / "pack" / "cells.json", FIX / "paper_fixture_cells.json"):
        assert json.loads(cells.read_bytes())["bench"]["r1_declaration_sha256"] == want


def test_the_fixture_results_score_and_contrast(capsys):
    fbc, fcfg = _files()
    common = ["--benchmark", "fixture_bench", "--bench-conf-dir", str(FIX / "conf")]
    assert main(["results", "score", fbc, fcfg, *common]) == 0
    assert main(["results", "contrast", "--a", fcfg, "--b", fbc, *common]) == 0
    out = capsys.readouterr().out
    assert "descriptive, n_seeds=1" in out and "no claim" in out
