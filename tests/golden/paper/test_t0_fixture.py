"""T0 end to end on a two-cell pack: identical, differs (1), integrity (2), fetch (3), hf://."""
from __future__ import annotations

import functools
import http.server
import json
import shutil
import threading
from pathlib import Path

import pytest

from axis_qvla.bench.spec import load_benchmark
from axis_qvla.cli import main
from axis_qvla.paper import cli as paper_cli
from axis_qvla.paper.expected import write_expected
from axis_qvla.paper.fetch import LocalSource, open_source
from axis_qvla.paper.spec import load_spec
from axis_qvla.paper.t0 import check, rebuild
from axis_qvla.core.canon import dumps_canonical
from axis_qvla.bench.results.fingerprint import eval_fingerprint
from axis_qvla.bench.results.seedfiles import ServedRun, read_served_tar, write_served_tar
from tests.golden.paper.t0_fixture import TASKS, build_pack
from tests.unit.bench.results.fixtures import make_bench



class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass


def _rebuild(conf: Path, source, ref: Path):
    """Exactly the inputs `paper tables` uses, so a pack built here checks clean from the CLI."""
    return rebuild(source, cells_ref=ref.read_bytes(), bench=load_benchmark("fixture_bench", conf_dir=conf),
                   spec=load_spec())


@pytest.fixture()
def pack(tmp_path: Path):
    conf, pack, ref = build_pack(tmp_path)
    result = _rebuild(conf, LocalSource(pack), ref)
    assert result.problems == ()
    write_expected(result.outputs, pack / "expected")
    return conf, pack, ref


def _cli(conf: Path, pack: Path | str, ref: Path, *extra: str) -> int:
    return main(["paper", "tables", "--results", str(pack), "--results-map", str(ref), "--check",
                 "--benchmark", "fixture_bench", "--bench-conf-dir", str(conf), *extra])


def test_outputs_and_identical_check(pack, capsys):
    conf, p, ref = pack
    result = _rebuild(conf, LocalSource(p), ref)
    assert sorted(result.outputs)[:3] == ["claims.json", "figures/fig1.csv", "figures/fig4c.csv"]
    assert "tsv/pt_r1_s2.tsv" in result.outputs and "tables/table1.tex" in result.outputs
    assert len(result.outputs) == 13
    pooled = json.loads(result.outputs["pooled_r1.json"])
    assert list(pooled) == ["bc|bare", "cfg_ph_ep|cond"] and pooled["bc|bare"]["suc"] == 3
    assert check(result, LocalSource(p)) == []
    assert _cli(conf, p, ref) == 0
    assert "IDENTICAL" in capsys.readouterr().out


def test_a_flipped_tar_byte_is_an_integrity_failure(pack, capsys):
    conf, p, ref = pack
    tar = p / "rollouts" / "s2" / "r1fbc_dualsim.tar"
    data = bytearray(tar.read_bytes()); data[600] ^= 1; tar.write_bytes(bytes(data))
    assert _cli(conf, p, ref) == 2
    assert "INTEGRITY  bc|bare: rollouts/s2/r1fbc_dualsim.tar: sha256" in capsys.readouterr().out


def test_a_wrong_fingerprint_is_an_integrity_failure(pack, capsys):
    conf, p, ref = pack
    doc = json.loads(ref.read_bytes()); doc["cells"][1]["eval_fingerprint"] = "0" * 64
    for f in (ref, p / "cells.json"):
        f.write_bytes(dumps_canonical(doc))
    assert _cli(conf, p, ref) == 2
    assert "cfg_ph_ep|cond: eval_fingerprint" in capsys.readouterr().out


def test_a_pack_cell_map_that_differs_from_the_committed_one_is_refused(pack, capsys):
    conf, p, ref = pack
    (p / "cells.json").write_bytes(ref.read_bytes() + b" ")
    assert _cli(conf, p, ref) == 2
    assert "differs from the committed results map" in capsys.readouterr().out


def test_a_changed_expected_exits_1_and_names_the_first_cell(pack, capsys):
    conf, p, ref = pack
    result = _rebuild(conf, LocalSource(p), ref)
    P = json.loads(result.outputs["pooled_r1.json"]); P["cfg_ph_ep|cond"]["suc"] += 1
    shutil.rmtree(p / "expected")
    write_expected(dict(result.outputs, **{"pooled_r1.json": json.dumps(P, indent=1).encode()}), p / "expected")
    assert _cli(conf, p, ref) == 1
    assert "DIFFERS  pooled_r1.json differs: first differing cell 'cfg_ph_ep|cond'" in capsys.readouterr().out


def test_a_missing_tar_or_unreadable_pack_is_a_fetch_error(pack):
    conf, p, ref = pack
    (p / "rollouts" / "s2" / "r1fcfg_dualsim.tar").unlink()
    assert _cli(conf, p, ref) == 3
    # chmod(0) is not unreadable for root (CI/containers often run as root, and
    # `skip` is forbidden here), so use a mechanism that fails for ANY user: a directory where a
    # file is expected. `open(dir, "rb")` always raises IsADirectoryError, root included.
    cells_json = p / "cells.json"
    data = cells_json.read_bytes()
    cells_json.unlink()
    cells_json.mkdir()
    try:
        assert _cli(conf, p, ref) == 3
    finally:
        cells_json.rmdir()
        cells_json.write_bytes(data)


COMMIT = "c" * 40   # the fixture lock pins the tag to this commit; the server serves /resolve/<COMMIT>/


def _lock(pack: Path, tmp_path: Path, *, bad: str | None = None) -> Path:
    """A qvla.release_lock/1 fixture over every file of the pack."""
    import hashlib
    files = []
    for f in sorted(x for x in pack.rglob("*") if x.is_file()):
        rel = f.relative_to(pack).as_posix()
        sha = "0" * 64 if rel == bad else hashlib.sha256(f.read_bytes()).hexdigest()
        files.append({"repo": "axisrobotics/qvla-fixture", "path": rel, "size": f.stat().st_size,
                      "sha256": sha, "git_oid": "0" * 40})
    doc = {"schema": "qvla.release_lock/1", "release": "v1.0",
           "repos": {"axisrobotics/qvla-fixture": {"type": "dataset", "tag": "v1.0", "commit": COMMIT}},
           "files": files}
    out = tmp_path / ("lock_bad.json" if bad else "lock.json")
    out.write_text(json.dumps(doc))
    return out


def test_hf_source_over_http(pack, tmp_path: Path, monkeypatch, capsys):
    conf, p, ref = pack
    srv_root = tmp_path / "srv"
    shutil.copytree(p, srv_root / "datasets" / "axisrobotics" / "qvla-fixture" / "resolve" / COMMIT)
    handler = functools.partial(_QuietHandler, directory=str(srv_root))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    lock, bad = _lock(p, tmp_path), _lock(p, tmp_path, bad="cells.json")
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        src = open_source("hf://axisrobotics/qvla-fixture@v1.0", cache=tmp_path / "cache", endpoint=url,
                          lock=json.loads(lock.read_text()))
        result = _rebuild(conf, src, ref)
        assert result.problems == () and check(result, src) == []
        assert (tmp_path / "cache" / "axisrobotics__qvla-fixture" / COMMIT / "cells.json").is_file()
        monkeypatch.setenv("HF_ENDPOINT", url)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        hf = "hf://axisrobotics/qvla-fixture@v1.0"
        assert _cli(conf, hf, ref, "--cache", str(tmp_path / "cache2"), "--checksums", str(lock)) == 0
        # no --cache: defaults to ./.qvla-cache in the working directory
        monkeypatch.chdir(tmp_path)
        assert _cli(conf, hf, ref, "--checksums", str(lock)) == 0
        assert (tmp_path / ".qvla-cache").is_dir() and "cache: " in capsys.readouterr().out
        # a lock row that disagrees with the served bytes is an integrity failure
        assert _cli(conf, hf, ref, "--cache", str(tmp_path / "cache3"), "--checksums", str(bad)) == 2
        assert "cells.json: sha256" in capsys.readouterr().out
        # a moved tag cannot matter: the lock pins the commit, and paths outside the lock are refused
        assert _cli(conf, "hf://axisrobotics/qvla-other@v1.0", ref, "--cache", str(tmp_path / "cache4"),
                    "--checksums", str(lock)) == 2
        assert "not in v1.0.lock.json" in capsys.readouterr().out
    finally:
        httpd.shutdown()
        httpd.server_close()   # avoid a ResourceWarning under -W error


# ---------------------------------------------------------------------------------------------
# Load-bearing guards that no existing test pinned.
# ---------------------------------------------------------------------------------------------

def test_a_missing_expected_file_is_a_fetch_error(pack):
    """Pins the per-file prefetch/verify loop at t0.py's `check` (deleting it would let a
    silently-missing expected/ file go undetected, since compare_expected only diffs against
    the in-memory rebuild and MANIFEST.json's claims, never the real bytes on disk)."""
    conf, p, ref = pack
    (p / "expected" / "tables" / "table1.tex").unlink()
    assert _cli(conf, p, ref) == 3


def test_a_truncated_expected_file_is_an_integrity_failure(pack, capsys):
    """Same guard as above, from the other side: a REAL file that no longer matches its own
    MANIFEST.json entry must be caught even though the rebuild's own output still matches the
    manifest's (unchanged) claimed digest."""
    conf, p, ref = pack
    f = p / "expected" / "pooled_r1.json"
    data = f.read_bytes()
    f.write_bytes(data[: len(data) // 2])
    assert _cli(conf, p, ref) == 2
    out = capsys.readouterr().out
    assert "pooled_r1.json" in out and "sha256" in out


def test_a_benchmark_n_seeds_mismatch_is_an_integrity_failure_naming_units_sha256(pack, tmp_path, capsys):
    """Pins `bench_problems` (cells.py). The committed cells.json still declares n_seeds=3; the
    benchmark passed to `rebuild` here declares 4, so cells.json and the benchmark disagree on
    the very work-unit digest that is supposed to make the two mutually load-bearing."""
    conf, p, ref = pack
    bad_conf = make_bench(tmp_path / "bad_conf", TASKS, 4)
    assert _cli(bad_conf, p, ref) == 2
    assert "units_sha256" in capsys.readouterr().out


def test_a_cell_with_a_missing_seed_is_an_integrity_failure_naming_the_seed_gap(pack, capsys):
    """Pins `run_problems` (cells.py): a served run whose fingerprint and task set both check
    out, but which holds only 2 of the declared 3 seeds for one task, must still be refused."""
    conf, p, ref = pack
    tar = p / "rollouts" / "s2" / "r1fbc_dualsim.tar"
    run = read_served_tar(tar, "r1fbc_dualsim")
    trimmed = ServedRun("r1fbc_dualsim", {**run.tasks, 806: run.tasks[806][:2]})
    sha = write_served_tar(trimmed, tar)
    fp = eval_fingerprint(trimmed, TASKS)
    doc = json.loads(ref.read_bytes())
    doc["cells"][0]["rollouts"] = dict(doc["cells"][0]["rollouts"], sha256=sha, size=tar.stat().st_size)
    doc["cells"][0]["eval_fingerprint"] = fp
    for f in (ref, p / "cells.json"):
        f.write_bytes(dumps_canonical(doc))
    assert _cli(conf, p, ref) == 2
    assert "does not hold exactly seeds" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------------
# `--out` never lets a rebuild check itself: --check compares before --out writes anything.
# ---------------------------------------------------------------------------------------------

def test_out_inside_the_pack_is_written_after_the_check(pack, capsys):
    """--check runs before --out writes, so a tampered expected/ is reported even when --out points at it."""
    conf, p, ref = pack
    result = _rebuild(conf, LocalSource(p), ref)
    P = json.loads(result.outputs["pooled_r1.json"]); P["cfg_ph_ep|cond"]["suc"] += 1
    shutil.rmtree(p / "expected")
    write_expected(dict(result.outputs, **{"pooled_r1.json": json.dumps(P, indent=1).encode()}), p / "expected")
    assert _cli(conf, p, ref, "--out", str(p / "expected")) == 1
    assert "DIFFERS" in capsys.readouterr().out


def test_out_outside_the_pack_still_writes_after_a_clean_check(pack, tmp_path, capsys):
    conf, p, ref = pack
    out = tmp_path / "elsewhere"
    assert _cli(conf, p, ref, "--out", str(out)) == 0
    assert (out / "pooled_r1.json").is_file()
    assert "wrote" in capsys.readouterr().out


def test_an_unexpected_exception_is_reported_as_integrity_not_a_bare_exit_1(pack, monkeypatch, capsys):
    """A raw traceback exits 1 by Python's own convention -- indistinguishable, at the shell,
    from this CLI's OWN exit 1 ("differs"). Any exception this CLI does not already have a named
    handler for must still come out as exit 2 with a message, never a bare traceback."""
    conf, p, ref = pack

    def _boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(paper_cli, "rebuild", _boom)
    assert _cli(conf, p, ref) == 2
    assert "boom" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------------
# A lock row's declared size, not just its sha256, is checked against the served
# bytes (a typed HfDigestMismatch from hf_http.download, not message-sniffing).
# ---------------------------------------------------------------------------------------------

def test_a_lock_size_mismatch_over_hf_is_an_integrity_failure(pack, tmp_path, monkeypatch, capsys):
    conf, p, ref = pack
    srv_root = tmp_path / "srv"
    shutil.copytree(p, srv_root / "datasets" / "axisrobotics" / "qvla-fixture" / "resolve" / COMMIT)
    handler = functools.partial(_QuietHandler, directory=str(srv_root))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        lock_doc = json.loads(_lock(p, tmp_path).read_text())
        for row in lock_doc["files"]:
            if row["path"] == "cells.json":
                row["size"] += 1   # correct sha256, wrong size: a corrupt/inconsistent lock row
        bad_size_lock = tmp_path / "lock_bad_size.json"
        bad_size_lock.write_text(json.dumps(lock_doc))
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        monkeypatch.setenv("HF_ENDPOINT", url)   # never hit the real Hub
        monkeypatch.delenv("HF_TOKEN", raising=False)
        assert _cli(conf, "hf://axisrobotics/qvla-fixture@v1.0", ref, "--cache", str(tmp_path / "cachesz"),
                    "--checksums", str(bad_size_lock)) == 2
        out = capsys.readouterr().out
        assert "size" in out and "cells.json" in out
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_a_table_i_cell_pretrained_with_another_seed_is_an_integrity_failure(pack, capsys):
    """Table I pairs on the pretraining seed: a cell whose recorded lineage
    names another pretrain seed than its replicate's cannot enter the table."""
    conf, p, ref = pack
    doc = json.loads(ref.read_bytes()); doc["cells"][1]["training_seed"] = {"pretrain": 43, "ft": 42}
    for f in (ref, p / "cells.json"):
        f.write_bytes(dumps_canonical(doc))
    assert _cli(conf, p, ref) == 2
    assert "INTEGRITY  Table I: π0.5 replicate 42 (role cond_ep) is cell cfg_ph_ep|cond, whose pretrain seed is 43" \
        in capsys.readouterr().out


def test_the_committed_fixture_pack_checks_identical_as_ci_runs_it(capsys):
    """The exact `python -S` step of the public CI: the committed two-cell pack and its expected/."""
    fx = Path(__file__).resolve().parent / "fixture_pack"
    assert main(["paper", "tables", "--results", str(fx / "pack"), "--results-map", str(fx / "paper_fixture_cells.json"),
                 "--benchmark", "fixture_bench", "--bench-conf-dir", str(fx / "conf"), "--check"]) == 0
    assert "IDENTICAL  13 outputs match expected/" in capsys.readouterr().out
