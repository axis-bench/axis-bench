"""The committed paper outputs are rebuilt byte-for-byte from the golden per-task tallies.

The golden TSVs are T0's own tallies of the 123 cells, read from the per-seed rollout files of
qvla-v1-results; pooling them with the committed cell map and rendering the layout
experiments/paper/paper.json must give exactly these files, under any PYTHONHASHSEED. The digests are
the v1.0 baseline.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from axis_qvla.paper.pool import dump_pooled, pool_cells
from axis_qvla.paper.render import OUTPUTS, render_paper
from axis_qvla.paper.spec import load_spec
from axis_qvla.paper.tsv import read_tsv, tallies_by_run

REPO = Path(__file__).resolve().parents[3]
GOLD = Path(__file__).resolve().parent
GOLDEN_SHA256 = {
    "claims.json": "ec8f19585c39d318a5e9b64e9917f922c62fc698e9d1ca9587d9a0ff68d5218a",
    "figures/fig1.csv": "a53d4600ad40f48a172c1a74d4604e20aa4e98ea44ec8823596be1d75030d320",
    "figures/fig4c.csv": "2fd372cc927d804076fb5d031d59f4b0fd7180af1792548351544f0209289ede",
    "figures/fig5.csv": "d394dcccc493ed48eafa9e50d8bd1fdd4e4e6c2b45e388c5d93486f602ab6e12",
    "figures/fig6.csv": "d1a89aa077389a9263d519106e6f7e473badc195d410ea2fda1978901d5965d1",
    "figures/fig7.csv": "6257bd61aae9ebc1b5019805c6d6da6fe3a15ba8d7f008cef819f0c052ccdebb",
    "results_page.html": "14dab3842558395ee1e74261e60cef78d8ac683fe000a751a50f0b603699102d",
    "results_page.md": "8f8e488887e962e5c9dbc7d82ab1475ee01eff297945f6efc51a3ca9a88a5a15",
    "tables/table1.html": "90e3534d0231cb131089a42af109842c901beecac675bfde1a5be06623833af8",
    "tables/table1.md": "21690de303a615d00d4272f530c366fd54b1544207028ce98549daf559f60de1",
    "tables/table1.tex": "a0048b0946dc077372b6cb6e9a2f0cd43248248ef9591a330fed7304d9f130c7",
}
_SCRIPT = (
    "import hashlib, json, sys\n"
    "from tests.golden.paper.test_paper_golden import rebuild\n"
    "print(json.dumps({k: hashlib.sha256(v).hexdigest() for k, v in rebuild().items()}, sort_keys=True))\n"
)


def rebuild() -> dict[str, bytes]:
    data = {}
    for box in ("s2", "s3"):
        data.update(tallies_by_run(read_tsv(GOLD / f"pt_r1_{box}.tsv")))
    cells = json.loads((REPO / "experiments" / "paper" / "cells.json").read_text())["cells"]
    P = json.loads(dump_pooled(pool_cells([(c["cell"], c["run"]) for c in cells], data)))
    out, problems = render_paper(P, {c["cell"]: c["training_seed"] for c in cells}, load_spec())
    assert problems == ()
    return out


def test_the_rebuild_is_byte_identical_to_the_committed_goldens():
    out = rebuild()
    assert sorted(out) == sorted(OUTPUTS) == sorted(GOLDEN_SHA256)
    for name, data in out.items():
        assert data == (GOLD / name).read_bytes(), name
        assert hashlib.sha256(data).hexdigest() == GOLDEN_SHA256[name], name


@pytest.mark.parametrize("hashseed", ["0", "1"])
def test_golden_bytes_under_any_hash_seed(hashseed: str):
    env = {**os.environ, "PYTHONHASHSEED": hashseed, "PYTHONPATH": str(REPO)}
    out = subprocess.run([sys.executable, "-c", _SCRIPT], capture_output=True, text=True, env=env, cwd=REPO, check=True)
    assert json.loads(out.stdout) == GOLDEN_SHA256


def test_the_committed_files_match_the_recorded_digests():
    for name, sha in GOLDEN_SHA256.items():
        assert hashlib.sha256((GOLD / name).read_bytes()).hexdigest() == sha, name
