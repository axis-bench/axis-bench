"""`bench eval` reads its inputs from where `qvla data get bench` put them and links the texture bank
into the render tree itself: the flags remain, as overrides."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from axis_qvla.bench import inputs
from axis_qvla.cli import main
from tests.unit.bench.eval_stubs import fake_driver
from tests.unit.bench.test_eval_roots import BANK, PROFILE

BANK_SRC_REL = Path(inputs.BANK_LOCK_PREFIX[len("assets/"):])   # textures/polyhaven/downloaded_256


def _lock(path: Path, monkeypatch) -> Path:
    rows = [{"path": inputs.BANK_LOCK_PREFIX + n, "sha256": hashlib.sha256(b).hexdigest(), "size": len(b)}
            for n, b in sorted(BANK.items())]
    path.write_text(json.dumps({"schema": "qvla.assets_lock/1", "files": rows}))
    monkeypatch.setattr(inputs, "ASSETS_LOCK", path)
    return path


@pytest.fixture
def got(tmp_path, monkeypatch):
    """A host whose <corpus>/qvla-v1-bench holds assets/ (with the bank) and tasks/, and a render tree
    with no bank yet."""
    (tmp_path / "hosts").mkdir()
    (tmp_path / "hosts" / "demo.toml").write_text(PROFILE.format(d=tmp_path))
    monkeypatch.setenv("AXIS_HOSTS_DIR", str(tmp_path / "hosts"))
    monkeypatch.setattr(os, "environ", dict(os.environ))  # restored after the test
    bench = tmp_path / "corpus" / "qvla-v1-bench"
    (bench / "tasks").mkdir(parents=True)
    src = bench / "assets" / BANK_SRC_REL
    src.mkdir(parents=True)
    for n, b in BANK.items():
        (src / n).write_bytes(b)
    (tmp_path / "render").mkdir()
    (tmp_path / "ck").mkdir()
    _lock(tmp_path / "lock.json", monkeypatch)
    return tmp_path


def _argv(d: Path, *extra: str) -> list[str]:
    return ["bench", "eval", "--host", "demo", "--checkpoint", str(d / "ck"), "--config", "cfg", "--run", "probe",
            "--render-root", str(d / "render"), *extra]


def test_the_inputs_default_to_the_data_get_location(got, monkeypatch):
    seen = fake_driver(monkeypatch, got)
    assert main(_argv(got)) == 0
    bench = got / "corpus" / "qvla-v1-bench"
    assert f"--asset-dir={bench / 'assets'}" in seen["argv"]
    assert f"--task-data-root={bench / 'tasks'}" in seen["argv"]


def test_an_explicit_flag_still_wins(got, monkeypatch):
    seen = fake_driver(monkeypatch, got)
    (got / "mytasks").mkdir()
    assert main(_argv(got, "--task-data-root", str(got / "mytasks"))) == 0
    assert f"--task-data-root={got / 'mytasks'}" in seen["argv"]


@pytest.mark.parametrize("missing, flag", [("tasks", "--task-data-root"), ("assets", "--asset-dir")])
def test_an_absent_default_names_data_get_bench(got, monkeypatch, capsys, missing, flag):
    seen = fake_driver(monkeypatch, got)
    shutil.rmtree(got / "corpus" / "qvla-v1-bench" / missing)
    assert main(_argv(got)) == 2
    out = capsys.readouterr().out
    assert flag in out and "does not exist" in out and "qvla data get bench" in out and "before any server" in out
    assert seen == {}


def test_the_bank_is_linked_into_the_render_tree_and_then_checked(got, monkeypatch, capsys):
    seen = fake_driver(monkeypatch, got)
    assert main(_argv(got)) == 0
    bank = got / "render" / inputs.BANK_RENDER_REL
    assert sorted(p.name for p in bank.iterdir()) == sorted(BANK)
    assert all((bank / n).is_symlink() and (bank / n).read_bytes() == b for n, b in BANK.items())
    assert f"linked {len(BANK)} Polyhaven texture files" in capsys.readouterr().out and seen


def test_link_bank_keeps_what_is_there_and_links_only_listed_files(tmp_path, monkeypatch):
    lock = _lock(tmp_path / "lock.json", monkeypatch)
    assets = tmp_path / "assets"
    (assets / BANK_SRC_REL).mkdir(parents=True)
    first, second = sorted(BANK)
    (assets / BANK_SRC_REL / first).write_bytes(BANK[first])        # the second is absent at the source
    (assets / BANK_SRC_REL / "stray.jpg").write_bytes(b"x")        # not in the checksum manifest
    bank = tmp_path / "render" / inputs.BANK_RENDER_REL
    bank.mkdir(parents=True)
    (bank / first).write_bytes(b"a copy made by hand")
    assert inputs.link_bank(tmp_path / "render", assets, lock) == []
    assert (bank / first).read_bytes() == b"a copy made by hand"    # never replaced
    assert sorted(p.name for p in bank.iterdir()) == [first]
    (bank / first).unlink()
    assert inputs.link_bank(tmp_path / "render", assets, lock) == [first]
    assert inputs.bank_problems(tmp_path / "render", json.loads(lock.read_text())) == [
        f"{second}: in the checksum manifest, not under the render root"]
