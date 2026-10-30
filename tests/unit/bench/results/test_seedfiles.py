"""Seed files are read byte-exact from a directory or a tar, and the tar is deterministic."""
from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from axis_qvla.bench.results.seedfiles import (SeedFileError, read_served_dir, read_served_tar,
                                    write_served_tar)
from tests.unit.bench.results.fixtures import make_served

OUTCOMES = {806: [True, False, True], 1050: [False, False, True]}


def test_dir_and_tar_readers_agree(tmp_path: Path):
    make_served(tmp_path / "logs", "armx_dualsim", OUTCOMES)
    a = read_served_dir(tmp_path / "logs", "armx_dualsim")
    assert sorted(a.tasks) == [806, 1050]
    assert [f.name for f in a.tasks[806]] == ["seed_806000.json", "seed_806001.json", "seed_806002.json"]
    assert a.tallies() == {806: (2, 3), 1050: (1, 3)}
    sha = write_served_tar(a, tmp_path / "armx.tar")
    b = read_served_tar(tmp_path / "armx.tar", "armx_dualsim")
    assert b == a and len(sha) == 64


def test_write_is_byte_stable(tmp_path: Path):
    make_served(tmp_path / "logs", "armx_dualsim", OUTCOMES)
    run = read_served_dir(tmp_path / "logs", "armx_dualsim")
    sha_one = write_served_tar(run, tmp_path / "one.tar")
    sha_two = write_served_tar(run, tmp_path / "two.tar")
    assert sha_one == sha_two
    assert (tmp_path / "one.tar").read_bytes() == (tmp_path / "two.tar").read_bytes()
    mt = {"armx_dualsim/task_806/seed_806000.json": 1758500000}
    assert write_served_tar(run, tmp_path / "three.tar", mtimes=mt) != sha_one
    with tarfile.open(tmp_path / "three.tar") as tf:
        ti = tf.getmember("armx_dualsim/task_806/seed_806000.json")
        assert (ti.mtime, ti.uid, ti.gid, ti.uname, ti.mode) == (1758500000, 0, 0, "", 0o644)


def _tar_with(tmp_path: Path, members: list[tarfile.TarInfo]) -> Path:
    p = tmp_path / "bad.tar"
    with tarfile.open(p, "w") as tf:
        for ti in members:
            data = b"{}" if ti.isreg() else None
            if data is not None:
                ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data) if data is not None else None)
    return p


def _reg(name: str) -> tarfile.TarInfo:
    return tarfile.TarInfo(name)


@pytest.mark.parametrize("names,match", [
    (["/abs/task_806/seed_806000.json"], "unsafe"),
    (["armx_dualsim/../task_806/seed_806000.json"], "unsafe"),
    (["armx_dualsim/task_806/seed_806000.json", "armx_dualsim/task_806/seed_806000.json"], "duplicate"),
    (["other_dualsim/task_806/seed_806000.json"], "not under"),
    (["armx_dualsim/task_806/notes.txt"], "unexpected file"),
])
def test_tar_reader_rejects_unsafe_members(tmp_path: Path, names, match):
    p = _tar_with(tmp_path, [_reg(n) for n in names])
    with pytest.raises(SeedFileError, match=match):
        read_served_tar(p, "armx_dualsim")


def test_tar_reader_rejects_links(tmp_path: Path):
    ti = tarfile.TarInfo("armx_dualsim/task_806/seed_806000.json")
    ti.type, ti.linkname = tarfile.SYMTYPE, "seed_806001.json"
    with pytest.raises(SeedFileError, match="not a regular file"):
        read_served_tar(_tar_with(tmp_path, [ti]), "armx_dualsim")


def test_unreadable_task_dir_is_fatal(tmp_path: Path):
    make_served(tmp_path / "logs", "armx_dualsim", OUTCOMES)
    d = tmp_path / "logs" / "armx_dualsim" / "task_806"
    d.chmod(0)
    try:
        with pytest.raises(SeedFileError, match="cannot read"):
            read_served_dir(tmp_path / "logs", "armx_dualsim")
    finally:
        d.chmod(0o755)


def test_the_harness_and_the_results_package_share_one_seed_file_rule():
    """The harness's rule (axis_qvla.bench.sim.record) and this package's were two copies that
    had to agree, and stopped agreeing when the harness's became exact. Now there is one function."""
    from axis_qvla.bench.sim import record
    from axis_qvla.bench.results import seedfiles

    assert record.is_seed_file is seedfiles.is_seed_file
    assert seedfiles.is_seed_file("seed_806000.json") and seedfiles.is_seed_file("seed_0.json")
    # ASCII digits only: `\d` also matches other scripts' digits (Arabic-Indic, fullwidth).
    assert not any(seedfiles.is_seed_file(n) for n in (
        "seed_1.diag.json", "seed_1.diag_objz.json", "seed_-1.json", "seed_x.json", "seed_.json",
        "seed_1.json.tmp", "xseed_1.json", "seed_\u0661\u0662.json", "seed_\uff11.json"))


@pytest.mark.parametrize("name", ["seed_x.json", "seed_-1.json", "seed_\u0661.json", "seed_1.json.bak",
                                  ".seed_806001.json.tmp99", "notes.txt"])
def test_a_name_neither_a_seed_nor_a_known_sidecar_is_refused(tmp_path: Path, name: str):
    """The old rule read seed_x.json, seed_-1.json and seed_<Arabic-Indic digit>.json as results: a
    phantom seed. Under the exact rule they are not results, and not a known sidecar either, so a
    reader of a served run refuses them by name instead of skipping them silently."""
    make_served(tmp_path / "logs", "armx_dualsim", OUTCOMES)
    (tmp_path / "logs" / "armx_dualsim" / "task_806" / name).write_text("{}")
    with pytest.raises(SeedFileError, match="unexpected file"):
        read_served_dir(tmp_path / "logs", "armx_dualsim")


def test_the_seed_file_pattern_is_spelled_in_one_module():
    """Every reader (the campaign tally and fingerprint, this package's fingerprint and results
    files, T0's per-task seed check) imports SEED_FILE / is_seed_file; a second spelling is how the
    two rules drift apart. The import root is read from this package, so the check holds whatever
    it is called."""
    import ast

    from axis_qvla.bench.results import seedfiles

    root = Path(seedfiles.__file__).resolve().parents[2]

    def spells(p: Path) -> bool:
        return any(isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith("seed_")
                   and ("[0-9]" in n.value or "\\d" in n.value) for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))))

    assert sorted(p.relative_to(root).as_posix() for p in root.rglob("*.py") if spells(p)) == ["bench/results/seedfiles.py"]
