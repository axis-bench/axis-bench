"""A user's qvla.toml is the machine profile when no --host names one."""
from __future__ import annotations

import pytest

from axis_qvla.core.hosts import HostProfileError
from axis_qvla.data.userconf import load_profile, user_paths

TOML = """launcher = "local"
gpus = 1
requires_secrets = []

[roots]
corpus = "{d}/corpus"
cache = "{d}/cache"
scratch = "{d}/scratch"
runs = "{d}/runs"
"""


def test_qvla_toml_in_the_working_directory_is_found(tmp_path, monkeypatch):
    monkeypatch.delenv("AXIS_HOST", raising=False)
    (tmp_path / "qvla.toml").write_text(TOML.format(d=tmp_path))
    assert user_paths(cwd=tmp_path).corpus == tmp_path / "corpus"


def test_an_explicit_toml_path_wins_and_a_missing_one_is_refused(tmp_path):
    (tmp_path / "mine.toml").write_text(TOML.format(d=tmp_path / "x"))
    assert load_profile("mine.toml", cwd=tmp_path).roots["cache"] == tmp_path / "x" / "cache"
    with pytest.raises(HostProfileError, match="no such file"):
        load_profile("absent.toml", cwd=tmp_path)


def test_an_explicit_path_wins_over_a_competing_qvla_toml(tmp_path):
    """--host <path>.toml is resolution's first match (docstring), so it must win even when a
    ./qvla.toml sits right there in the same directory -- not merely when there is nothing else."""
    (tmp_path / "qvla.toml").write_text(TOML.format(d=tmp_path / "local"))
    (tmp_path / "mine.toml").write_text(TOML.format(d=tmp_path / "explicit"))
    assert load_profile("mine.toml", cwd=tmp_path).roots["cache"] == tmp_path / "explicit" / "cache"


def test_qvla_toml_wins_over_axis_host(tmp_path, monkeypatch):
    """./qvla.toml beats $AXIS_HOST (docstring order), even when AXIS_HOST names a profile that
    resolves fine on its own -- a stray env var must not silently override the file sitting in
    the working directory."""
    hosts_dir = tmp_path / "hosts"
    hosts_dir.mkdir()
    (hosts_dir / "envhost.toml").write_text(TOML.format(d=tmp_path / "env"))
    monkeypatch.setenv("AXIS_HOSTS_DIR", str(hosts_dir))
    monkeypatch.setenv("AXIS_HOST", "envhost")
    cwd = tmp_path / "work"
    cwd.mkdir()
    (cwd / "qvla.toml").write_text(TOML.format(d=tmp_path / "local"))
    assert load_profile(None, cwd=cwd).roots["cache"] == tmp_path / "local" / "cache"


def test_nothing_to_go_on_says_what_to_write(tmp_path, monkeypatch):
    monkeypatch.delenv("AXIS_HOST", raising=False)
    with pytest.raises(HostProfileError, match=r"run `qvla init` here to write one \(or copy qvla.example.toml"):
        load_profile(None, cwd=tmp_path)


def test_an_explicit_path_in_an_untraversable_directory_raises_rather_than_reading_absent(tmp_path):
    """Path.is_file() raises PermissionError (not a quiet False) when a parent directory is
    untraversable -- load_profile's --host <path>.toml check must turn that into a named
    HostProfileError, never let a raw PermissionError escape as 'no such file'."""
    import os
    import stat

    if os.geteuid() == 0:
        pytest.skip("root bypasses directory permissions")
    d = tmp_path / "locked"
    d.mkdir()
    (d / "mine.toml").write_text(TOML.format(d=tmp_path / "x"))
    d.chmod(0o000)
    try:
        with pytest.raises(HostProfileError, match="cannot access"):
            load_profile(str(d / "mine.toml"), cwd=tmp_path)
    finally:
        d.chmod(stat.S_IRWXU)


def test_qvla_toml_in_an_untraversable_directory_raises_rather_than_reading_absent(tmp_path):
    """Same guard for the ./qvla.toml check: an untraversable working directory must not read as
    'no qvla.toml here', which would then (wrongly) fall through to $AXIS_HOST or the 'run
    `qvla init`' refusal."""
    import os
    import stat

    if os.geteuid() == 0:
        pytest.skip("root bypasses directory permissions")
    d = tmp_path / "locked"
    d.mkdir()
    (d / "qvla.toml").write_text(TOML.format(d=tmp_path / "x"))
    d.chmod(0o000)
    try:
        with pytest.raises(HostProfileError, match="cannot access"):
            load_profile(None, cwd=d)
    finally:
        d.chmod(stat.S_IRWXU)


def test_a_missing_queue_is_added_in_the_file_the_profile_came_from(tmp_path, monkeypatch):
    """A qvla.toml user is told to edit their own qvla.toml, never a profile directory the public
    repository does not ship; a profile built by hand, which has no file, is named instead."""
    from axis_qvla.core.hosts import HostProfile

    monkeypatch.delenv("AXIS_HOST", raising=False)
    (tmp_path / "qvla.toml").write_text(TOML.format(d=tmp_path))
    with pytest.raises(HostProfileError) as exc:
        load_profile(None, cwd=tmp_path).queue("long")
    assert f"Add it under [queues] in {tmp_path / 'qvla.toml'} " in str(exc.value)
    assert str(exc.value).count(".toml") == 1                 # that file and no other
    assert HostProfile(name="lab", launcher="local", roots={}, venvs={}).source() == "the 'lab' host profile"


def test_a_profile_without_a_launcher_runs_locally(tmp_path, monkeypatch):
    monkeypatch.delenv("AXIS_HOST", raising=False)
    (tmp_path / "qvla.toml").write_text(TOML.format(d=tmp_path).replace('launcher = "local"\n', "")
                                        .replace("requires_secrets = []\n", ""))
    assert load_profile(None, cwd=tmp_path).launcher == "local"
