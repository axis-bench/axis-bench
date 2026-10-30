"""eval_fingerprint equals campaign.fingerprint; units_sha256 equals every published run
manifest's."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from axis_qvla.bench.spec import load_benchmark
from axis_qvla.bench.sim.campaign import fingerprint as reference_fingerprint
from axis_qvla.bench.results.fingerprint import FingerprintError, eval_fingerprint, units_sha256
from axis_qvla.bench.results.seedfiles import read_served_dir, read_served_tar, write_served_tar
from tests.unit.bench.results.fixtures import make_served

OUTCOMES = {806: [True, False, True], 1050: [False, False, True]}
R1_UNITS_SHA256 = "b3f14d19894fb3e50e932329cfe98d991049c05387476313af9a9b24215388a9"


def test_matches_the_campaign_reference_from_dir_and_tar(tmp_path: Path):
    make_served(tmp_path / "logs", "armx_dualsim", OUTCOMES)
    want = reference_fingerprint(tmp_path / "logs", "armx_dualsim", [1050, 806])
    run = read_served_dir(tmp_path / "logs", "armx_dualsim")
    assert eval_fingerprint(run, [1050, 806]) == want
    write_served_tar(run, tmp_path / "a.tar")
    assert eval_fingerprint(read_served_tar(tmp_path / "a.tar", "armx_dualsim"), [806, 1050]) == want


def test_diag_and_meta_do_not_count(tmp_path: Path):
    make_served(tmp_path / "logs", "armx_dualsim", OUTCOMES)
    before = eval_fingerprint(read_served_dir(tmp_path / "logs", "armx_dualsim"), [806, 1050])
    d = tmp_path / "logs" / "armx_dualsim" / "task_806"
    (d / "meta.json").write_text('{"changed": true}')
    (d / "seed_806001.diag.json").write_text('{"changed": true}')
    assert eval_fingerprint(read_served_dir(tmp_path / "logs", "armx_dualsim"), [806, 1050]) == before


def test_missing_task_fails_closed(tmp_path: Path):
    make_served(tmp_path / "logs", "armx_dualsim", {806: [True]})
    run = read_served_dir(tmp_path / "logs", "armx_dualsim")
    with pytest.raises(FingerprintError, match="task 1050 is missing"):
        eval_fingerprint(run, [806, 1050])


def test_units_sha256_of_the_declared_benchmark():
    b = load_benchmark()
    assert units_sha256(b.task_ids, b.n_seeds) == R1_UNITS_SHA256
    assert units_sha256(list(reversed(b.task_ids)), b.n_seeds) != R1_UNITS_SHA256


# What the harness writes beside a result under --diagnostics (axis_qvla.bench.sim.domain.diagnostic_path
# and the render / dual-sim backends that take it as their trace path).
SIDECARS = ("seed_806000.diag.json", "seed_806000.diag_objz.json", "seed_806000.diag.mp4", "seed_806000.mp4",
            "seed_806000.diag_frame0.png", "seed_806000.diag_frameN.png", "seed_806000.diag_base_ab12cd.png")


def test_a_diagnostics_run_fingerprints_the_same_here_and_in_the_harness(tmp_path: Path):
    """The harness selects exactly `seed_<int>.json` while this
    package still took `seed_*.json` minus `*.diag.json`. On an arm run with --diagnostics the
    dual-sim backend's seed_<n>.diag_objz.json then counted here and not there, so the two
    fingerprints of one arm disagreed (and its PNG frames made the read fail)."""
    make_served(tmp_path / "logs", "armx_dualsim", OUTCOMES)
    want = reference_fingerprint(tmp_path / "logs", "armx_dualsim", [806, 1050])
    d = tmp_path / "logs" / "armx_dualsim" / "task_806"
    for name in SIDECARS:
        (d / name).write_text('{"seed": 806000, "success": true}')
    assert reference_fingerprint(tmp_path / "logs", "armx_dualsim", [806, 1050]) == want
    run = read_served_dir(tmp_path / "logs", "armx_dualsim")
    assert eval_fingerprint(run, [806, 1050]) == want
    assert run.tallies() == {806: (2, 3), 1050: (1, 3)}


FIXTURE_PACK = Path(__file__).resolve().parents[3] / "golden" / "paper" / "fixture_pack" / "pack"


def test_the_fixture_packs_recorded_fingerprints_are_unchanged():
    """The published fingerprints were stamped before the rule became exact. Every published tar
    holds only `seed_<int>.json` members, so the exact rule reproduces each recorded digest; T0
    recomputes all 123 of the real pack's, and the committed fixture pack is checked here."""
    results = sorted((FIXTURE_PACK / "results").glob("*/*.json"))
    assert len(results) == 2
    for res in results:
        header = json.loads(res.read_bytes())["header"]
        run = read_served_tar(FIXTURE_PACK / "rollouts" / res.parent.name / f"{header['served']}.tar",
                              header["served"])
        assert eval_fingerprint(run, sorted(run.tasks)) == header["eval_fingerprint"], res.name
