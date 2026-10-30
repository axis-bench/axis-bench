"""A benchmark version is a directory identified by the hash of everything in it, frozen by
VERSION.json; a frozen version cannot change without `bench check` saying so."""
from __future__ import annotations

import json
import shutil

import pytest

from axis_qvla.bench import spec, versions
from axis_qvla.cli import main

V1 = "b1d47d1d379da64353ebdd2c65249745584dd6c8f034af9b58651f25deb40cfe"


def test_the_shipped_version_is_frozen_and_intact():
    d = versions.version_dir(versions.DEFAULT_VERSION)
    assert versions.check_version(d) == []
    rec = versions.frozen_record(d)
    assert rec["sha256"] == V1 and rec["version"] == "axis-heldout-v1"
    assert set(rec["files"]) == {"ASSETS.lock.json", "execution.json", "franka_heldout24.json",
                                 "franka_heldout24.layouts.json", "franka_heldout24.tasks.json"}
    assert rec["files"]["franka_heldout24.json"].startswith("6982b5f0")   # the declaration this version froze


def test_a_qualified_name_and_a_bare_name_load_the_same_benchmark():
    a, b = spec.load_benchmark("axis-heldout-v1/franka_heldout24"), spec.load_benchmark()
    assert a.sha256 == b.sha256 and a.version_sha256 == b.version_sha256 == V1
    assert a.identity()["version_sha256"] == V1 and a.frozen_sha256 == V1
    assert a.execution["replan"] == 8 and a.execution["render"] == {"width": 640, "height": 360}
    assert spec.check(a) == []


def test_unknown_versions_and_names_are_refused():
    with pytest.raises(spec.BenchmarkError, match="no benchmark version 'nope'"):
        spec.load_benchmark("nope/franka_heldout24")
    with pytest.raises(spec.BenchmarkError):
        spec.load_benchmark("axis-heldout-v1/other")


def _copy(tmp_path):
    root = tmp_path / "benchmarks"
    shutil.copytree(versions.version_dir(versions.DEFAULT_VERSION), root / "axis-heldout-v1")
    return root / "axis-heldout-v1"


def test_an_edit_to_a_frozen_version_is_named(tmp_path):
    d = _copy(tmp_path)
    (d / "franka_heldout24.layouts.json").write_text("{}")
    (d / "extra.json").write_text("{}")
    problems = versions.check_version(d)
    assert any("franka_heldout24.layouts.json: edited after the version was frozen" in p for p in problems)
    assert any("extra.json: added after the version was frozen" in p for p in problems)


def test_freeze_writes_the_record_once(tmp_path):
    d = _copy(tmp_path)
    (d / versions.VERSION_FILE).unlink()
    rec = versions.freeze(d)
    assert rec["sha256"] == V1
    assert json.loads((d / versions.VERSION_FILE).read_text()) == rec
    with pytest.raises(versions.VersionError, match="already frozen"):
        versions.freeze(d)


def test_the_hash_covers_names_and_bytes(tmp_path):
    files = {"a.json": "1" * 64, "b.json": "2" * 64}
    assert versions.version_sha256(files) != versions.version_sha256({"a.json": "1" * 64, "c.json": "2" * 64})
    assert versions.version_sha256(files) == versions.version_sha256(dict(reversed(list(files.items()))))


def test_a_symlink_in_a_version_is_refused(tmp_path):
    d = tmp_path / "v"
    d.mkdir()
    (d / "real.json").write_text("{}")
    (d / "link.json").symlink_to(d / "real.json")
    with pytest.raises(versions.VersionError, match="symlink"):
        versions.content_files(d)


def test_execution_constants_are_validated(tmp_path):
    d = _copy(tmp_path)
    ex = json.loads((d / "execution.json").read_text())
    ex["replan"] = 0
    (d / "execution.json").write_text(json.dumps(ex))
    with pytest.raises(spec.BenchmarkError, match="replan"):
        spec.load_benchmark(conf_dir=d)


def test_bench_check_prints_the_version_and_versions_lists_it(capsys):
    assert main(["bench", "check"]) == 0
    out = capsys.readouterr().out
    assert f"version_sha256   {V1}" in out and "frozen           yes" in out
    assert main(["bench", "versions"]) == 0
    assert f"axis-heldout-v1  {V1}  frozen" in capsys.readouterr().out


# --- check()'s released-version failure branches (spec.py's frozen block) --------------------

def test_check_flags_a_released_version_whose_declaration_names_the_wrong_directory(tmp_path, monkeypatch):
    d = _copy(tmp_path)
    decl = json.loads((d / "franka_heldout24.json").read_text())
    decl["version"] = "not-this-directory"
    (d / "franka_heldout24.json").write_text(json.dumps(decl))
    monkeypatch.setattr(versions, "BENCHMARKS_ROOT", d.parent)
    b = spec.load_benchmark(conf_dir=d)
    problems = spec.check(b)
    assert any("declares version 'not-this-directory'" in p and "'axis-heldout-v1'" in p for p in problems)


def test_check_flags_a_released_version_with_no_execution_json(tmp_path, monkeypatch):
    d = _copy(tmp_path)
    (d / "execution.json").unlink()
    (d / versions.VERSION_FILE).unlink()
    versions.freeze(d)   # re-freeze without execution.json, so check_version itself stays clean
    monkeypatch.setattr(versions, "BENCHMARKS_ROOT", d.parent)
    b = spec.load_benchmark(conf_dir=d)
    assert b.execution == {}
    problems = spec.check(b)
    assert any("no execution.json" in p for p in problems)
    assert any(p.startswith("axis-heldout-v1: no execution.json") for p in problems)


# --- `bench versions`' EDITED / not-frozen / malformed-record rows --------------------------

def _renamed_copy(root, name):
    d = root / name
    shutil.copytree(versions.version_dir(versions.DEFAULT_VERSION), d)
    decl = json.loads((d / "franka_heldout24.json").read_text())
    decl["version"] = name
    (d / "franka_heldout24.json").write_text(json.dumps(decl))
    (d / versions.VERSION_FILE).unlink()
    return d


def test_bench_versions_shows_edited_not_frozen_and_a_bad_record_without_stopping(tmp_path, monkeypatch, capsys):
    root = tmp_path / "benchmarks"

    edited = _renamed_copy(root, "edited-v1")
    versions.freeze(edited)
    (edited / "franka_heldout24.layouts.json").write_text("{}")   # edited after freeze

    unfrozen = _renamed_copy(root, "unfrozen-v1")   # VERSION.json never written

    broken = _renamed_copy(root, "broken-v1")
    versions.freeze(broken)
    (broken / versions.VERSION_FILE).write_text("not json")

    monkeypatch.setattr(versions, "BENCHMARKS_ROOT", root)
    assert main(["bench", "versions"]) == 0
    lines = {ln.split()[0]: ln for ln in capsys.readouterr().out.splitlines()}
    assert len(lines) == 3   # one row per directory, including the malformed one -- never stops
    assert lines["edited-v1"].endswith("EDITED")
    assert lines["unfrozen-v1"].endswith("not frozen")
    assert lines["broken-v1"].split()[1] == "ERROR"


def test_frozen_record_wraps_malformed_json_and_a_missing_sha256_as_version_error(tmp_path):
    """A malformed VERSION.json must raise VersionError, never a bare JSONDecodeError or
    KeyError -- load_benchmark/bench check then print one clear line, not a traceback."""
    d = tmp_path / "v"
    d.mkdir()
    (d / versions.VERSION_FILE).write_text("not json")
    with pytest.raises(versions.VersionError, match="not valid JSON"):
        versions.frozen_record(d)
    assert any("not valid JSON" in p for p in versions.check_version(d))

    (d / versions.VERSION_FILE).write_text(json.dumps(
        {"schema": versions.VERSION_SCHEMA, "files": {}}))   # no "sha256" key
    with pytest.raises(versions.VersionError, match="not a .* record"):
        versions.frozen_record(d)
    assert any("not a" in p and "record" in p for p in versions.check_version(d))
