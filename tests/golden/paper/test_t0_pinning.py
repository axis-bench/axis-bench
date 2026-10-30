"""Every byte T0 reads from a results pack is pinned.

Each cell's `results` and `manifest` references in cells.json are verified like the rollout
tars -- present in the source, at the recorded sha256 (and size where recorded), fetched through
the same Source, so over hf:// the lock pins every path -- and a mismatch or a missing file is an
integrity failure (exit 2). A cell that names no run manifest has none to check. A local pack's
expected/MANIFEST.json is pinned too, by the committed cell map it belongs to, as hf://'s is by
the release lock. And the committed fixture pack the public CI runs `python -S` on is exactly what
the builder builds, so CI exercises the reference checks.
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import http.server
import json
import shutil
import threading
from pathlib import Path

import pytest

from axis_qvla.paper import cli as paper_cli
from axis_qvla.paper.expected import write_expected
from axis_qvla.paper.fetch import LocalSource, ShaMismatch
from axis_qvla.paper.t0 import check
from axis_qvla.core.canon import dumps_canonical
from tests.golden.paper.t0_fixture import build_pack
from tests.golden.paper.test_t0_fixture import COMMIT, _cli, _lock, _QuietHandler, _rebuild, pack  # noqa: F401

REPO = Path(__file__).resolve().parents[3]
FIXTURE = Path(__file__).resolve().parent / "fixture_pack"
GOLD = Path(__file__).resolve().parent
RESULTS0 = "results/s2/r1fbc_dualsim.json"      # bc|bare's results file
RESULTS1 = "results/s2/r1fcfg_dualsim.json"     # cfg_ph_ep|cond's (it names no run manifest)
MANIFEST0 = "runs/s2/r1fbc/run_manifest.json"   # bc|bare's run manifest


def _flip(f: Path, at: int = 40) -> None:
    data = bytearray(f.read_bytes())
    data[at] ^= 1
    f.write_bytes(bytes(data))


def _write_cells(doc: dict, ref: Path, pack: Path) -> None:
    for f in (ref, pack / "cells.json"):
        f.write_bytes(dumps_canonical(doc))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- the results and manifest references, local pack -----------------------------------

def test_the_fixture_names_a_results_file_per_cell_and_one_run_manifest(pack):
    _, p, ref = pack
    cells = json.loads(ref.read_bytes())["cells"]
    assert [c["results"]["path"] for c in cells] == [RESULTS0, RESULTS1]
    assert [c.get("manifest", {}).get("path") for c in cells] == [MANIFEST0, None]
    assert all((p / r).is_file() for r in (RESULTS0, RESULTS1, MANIFEST0))


def test_a_flipped_byte_in_a_referenced_results_file_is_an_integrity_failure(pack, capsys):
    conf, p, ref = pack
    _flip(p / RESULTS0)
    assert _cli(conf, p, ref) == 2
    assert f"INTEGRITY  bc|bare: {RESULTS0}: sha256" in capsys.readouterr().out


def test_a_flipped_byte_in_a_referenced_manifest_is_an_integrity_failure(pack, capsys):
    conf, p, ref = pack
    _flip(p / MANIFEST0, at=5)
    assert _cli(conf, p, ref) == 2
    assert f"INTEGRITY  bc|bare: {MANIFEST0}: sha256" in capsys.readouterr().out


def test_a_missing_referenced_manifest_is_an_integrity_failure(pack, capsys):
    conf, p, ref = pack
    (p / MANIFEST0).unlink()
    assert _cli(conf, p, ref) == 2
    assert (f"INTEGRITY  bc|bare: manifest {MANIFEST0} is named in cells.json but the results source "
            "does not hold it") in capsys.readouterr().out


def test_a_missing_referenced_results_file_is_an_integrity_failure(pack, capsys):
    conf, p, ref = pack
    (p / RESULTS1).unlink()
    assert _cli(conf, p, ref) == 2
    assert f"INTEGRITY  cfg_ph_ep|cond: results {RESULTS1} is named in cells.json" in capsys.readouterr().out


def test_a_cell_without_a_run_manifest_passes(pack, capsys):
    """cfg_ph_ep|cond names no run manifest and nothing under runs/ belongs to it: nothing to verify."""
    conf, p, ref = pack
    assert not (p / "runs" / "s2" / "r1fcfg").exists()
    assert _cli(conf, p, ref) == 0
    assert "IDENTICAL  13 outputs match expected/" in capsys.readouterr().out


def test_every_bad_reference_is_reported_together(pack):
    conf, p, ref = pack
    _flip(p / RESULTS0)
    (p / RESULTS1).unlink()
    result = _rebuild(conf, LocalSource(p), ref)
    assert result.outputs == {}
    assert [x.split(":")[0] for x in result.problems] == ["bc|bare", "cfg_ph_ep|cond"]


def test_a_recorded_size_is_checked(pack, capsys):
    conf, p, ref = pack
    doc = json.loads(ref.read_bytes())
    size = (p / RESULTS0).stat().st_size
    doc["cells"][0]["results"]["size"] = size + 1
    _write_cells(doc, ref, p)
    assert _cli(conf, p, ref) == 2
    assert f"INTEGRITY  bc|bare: {RESULTS0} is {size} bytes, cells.json says {size + 1}" in capsys.readouterr().out
    doc["cells"][0]["results"]["size"] = size
    _write_cells(doc, ref, p)
    assert _cli(conf, p, ref) == 0


def test_a_recorded_size_must_be_an_integer(pack, capsys):
    conf, p, ref = pack
    doc = json.loads(ref.read_bytes())
    doc["cells"][0]["manifest"]["size"] = "12"
    _write_cells(doc, ref, p)
    assert _cli(conf, p, ref) == 2
    assert "cells[0].manifest: a recorded size must be a non-negative integer" in capsys.readouterr().out


def test_an_unreadable_reference_is_a_fetch_error_not_a_missing_file(pack, capsys):
    """Only "no such file" is a missing reference. A path T0 cannot read (here a
    directory where the file should be: unreadable for any user, root included) is exit 3."""
    conf, p, ref = pack
    (p / RESULTS0).unlink()
    (p / RESULTS0).mkdir()
    assert _cli(conf, p, ref) == 3
    assert capsys.readouterr().out.startswith("FETCH  ")


# --- over hf://: the lock pins every referenced path ------------------------------------

@contextlib.contextmanager
def _hub(served: Path, tmp_path: Path, monkeypatch):
    """Serve `served` as the fixture dataset at COMMIT; never the real Hub."""
    root = tmp_path / "srv"
    shutil.copytree(served, root / "datasets" / "axisrobotics" / "qvla-fixture" / "resolve" / COMMIT)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_QuietHandler, directory=str(root)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{httpd.server_address[1]}")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    try:
        yield
    finally:
        httpd.shutdown()
        httpd.server_close()


HF = "hf://axisrobotics/qvla-fixture@v1.0"


def test_over_hf_a_clean_pack_with_references_checks_identical(pack, tmp_path, monkeypatch, capsys):
    conf, p, ref = pack
    lock = _lock(p, tmp_path)
    with _hub(p, tmp_path, monkeypatch):
        assert _cli(conf, HF, ref, "--cache", str(tmp_path / "c"), "--checksums", str(lock)) == 0
    assert "IDENTICAL" in capsys.readouterr().out
    assert (tmp_path / "c" / "axisrobotics__qvla-fixture" / COMMIT / MANIFEST0).is_file()


def test_over_hf_a_reference_the_lock_does_not_list_is_an_integrity_failure(pack, tmp_path, monkeypatch, capsys):
    conf, p, ref = pack
    doc = json.loads(_lock(p, tmp_path).read_text())
    doc["files"] = [f for f in doc["files"] if f["path"] != RESULTS1]
    lock = tmp_path / "lock_without_results1.json"
    lock.write_text(json.dumps(doc))
    with _hub(p, tmp_path, monkeypatch):
        assert _cli(conf, HF, ref, "--cache", str(tmp_path / "c"), "--checksums", str(lock)) == 2
    assert f"INTEGRITY  cfg_ph_ep|cond: axisrobotics/qvla-fixture:{RESULTS1} not in v1.0.lock.json" \
        in capsys.readouterr().out


def test_over_hf_served_bytes_that_differ_from_the_lock_are_an_integrity_failure(pack, tmp_path, monkeypatch, capsys):
    conf, p, ref = pack
    lock = _lock(p, tmp_path)           # rows from the clean pack ...
    _flip(p / MANIFEST0, at=5)          # ... the Hub serves a tampered manifest
    with _hub(p, tmp_path, monkeypatch):
        assert _cli(conf, HF, ref, "--cache", str(tmp_path / "c"), "--checksums", str(lock)) == 2
    out = capsys.readouterr().out
    assert "INTEGRITY  bc|bare: " in out and MANIFEST0 in out


def test_over_hf_the_cell_map_not_the_lock_is_the_authority_for_a_reference(pack, tmp_path, monkeypatch, capsys):
    """A results file republished consistently (the lock and the Hub agree on new bytes) is still
    not the file the cell map names."""
    conf, p, ref = pack
    _flip(p / RESULTS0)
    lock = _lock(p, tmp_path)
    with _hub(p, tmp_path, monkeypatch):
        assert _cli(conf, HF, ref, "--cache", str(tmp_path / "c"), "--checksums", str(lock)) == 2
    assert f"INTEGRITY  bc|bare: {RESULTS0}: sha256" in capsys.readouterr().out


# --- the local expected/MANIFEST.json pin --------------------------------------------------

def _golden_outputs() -> dict[str, bytes]:
    """The 14 frozen outputs, by their expected/ path (the goldens keep the TSVs at top level).
    They share tests/golden/paper/ with the golden tests and the fixture pack, which are not
    outputs: only the top level's data files and figures/ and tables/ are."""
    out = {}
    top = [x for x in GOLD.iterdir() if x.is_file() and x.suffix != ".py"]
    for f in sorted(top + [x for d in ("figures", "tables") for x in (GOLD / d).rglob("*") if x.is_file()]):
        rel = f.relative_to(GOLD).as_posix()
        out[f"tsv/{rel}" if rel.startswith("pt_r1_") else rel] = f.read_bytes()
    return out


def test_the_committed_cell_map_pins_the_frozen_expected_manifest(tmp_path):
    """The pin is keyed by the committed cell map's bytes, and equals the MANIFEST.json that
    write_expected gives for the 14 golden outputs (themselves pinned by test_paper_golden): the
    frozen expected/ of qvla-v1-results, 9f1b03a0."""
    key = _sha((REPO / "experiments" / "paper" / "cells.json").read_bytes())
    assert list(paper_cli.FROZEN_EXPECTED) == [key]
    outputs = _golden_outputs()
    assert len(outputs) == 14
    manifest = write_expected(outputs, tmp_path / "expected") / "MANIFEST.json"
    assert paper_cli.FROZEN_EXPECTED[key] == _sha(manifest.read_bytes())
    assert paper_cli.FROZEN_EXPECTED[key].startswith("9f1b03a0")


def test_a_local_manifest_that_differs_from_its_pin_is_an_integrity_failure(pack, monkeypatch, capsys):
    conf, p, ref = pack
    good = _sha((p / "expected" / "MANIFEST.json").read_bytes())
    monkeypatch.setattr(paper_cli, "FROZEN_EXPECTED", {_sha(ref.read_bytes()): "0" * 64})
    assert _cli(conf, p, ref) == 2
    assert "INTEGRITY  expected/MANIFEST.json: sha256" in capsys.readouterr().out
    monkeypatch.setattr(paper_cli, "FROZEN_EXPECTED", {_sha(ref.read_bytes()): good})
    assert _cli(conf, p, ref) == 0
    assert "IDENTICAL" in capsys.readouterr().out


def test_check_reads_the_manifest_through_its_pin(pack):
    conf, p, ref = pack
    result = _rebuild(conf, LocalSource(p), ref)
    good = _sha((p / "expected" / "MANIFEST.json").read_bytes())
    assert check(result, LocalSource(p), manifest_sha256=good) == []
    with pytest.raises(ShaMismatch, match="expected/MANIFEST.json"):
        check(result, LocalSource(p), manifest_sha256="0" * 64)


def test_a_cell_map_without_a_pin_reads_its_manifest_unpinned(pack):
    _, _, ref = pack
    assert _sha(ref.read_bytes()) not in paper_cli.FROZEN_EXPECTED


# --- the committed fixture pack is the builder's ------------------------------------------

def _tree(root: Path) -> dict[str, bytes]:
    return {f.relative_to(root).as_posix(): f.read_bytes() for f in sorted(root.rglob("*")) if f.is_file()}


def test_the_committed_fixture_pack_is_what_the_builder_builds(tmp_path):
    """CI's `python -S` step runs on tests/golden/paper/fixture_pack; it must be the builder's pack
    (references included) and the expected/ T0 writes for it, byte for byte."""
    conf, p, ref = build_pack(tmp_path)
    write_expected(_rebuild(conf, LocalSource(p), ref).outputs, p / "expected")
    assert _tree(FIXTURE / "conf") == _tree(conf)
    assert _tree(FIXTURE / "pack") == _tree(p)
    assert (FIXTURE / "paper_fixture_cells.json").read_bytes() == ref.read_bytes()
