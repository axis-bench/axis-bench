"""`qvla init` writes the ./qvla.toml every other command reads: data roots under one directory,
checkout paths under this checkout, nothing to hand-edit."""
from __future__ import annotations

import tomllib

from axis_qvla.cli import main
from axis_qvla.data.userconf import REPO, init_text, load_profile, user_paths


def test_init_writes_a_profile_every_command_reads(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["init"]) == 0
    assert capsys.readouterr().out.startswith("OK    ")
    paths = user_paths(None, cwd=tmp_path)
    assert paths.runs == tmp_path / "srv" / "runs" and paths.cache.is_dir() and paths.corpus.is_dir()
    profile = load_profile(None, cwd=tmp_path)
    assert profile.roots["openpi"] == REPO / "third_party" / "openpi"
    assert profile.roots["smolvla_venv"] == REPO / "third_party" / "lerobot" / ".venv"
    assert profile.venvs["online"] == REPO / "third_party" / "openpi" / ".venv"
    assert profile.gpus == 1
    assert "@" not in (tmp_path / "qvla.toml").read_text()


def test_init_never_replaces_a_profile_without_force(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "qvla.toml").write_text("mine\n")
    assert main(["init"]) == 2
    assert "--force" in capsys.readouterr().out
    assert (tmp_path / "qvla.toml").read_text() == "mine\n"
    assert main(["init", "--force", "--root", str(tmp_path / "data"), "--gpus", "8"]) == 0
    assert user_paths(None, cwd=tmp_path).corpus == tmp_path / "data" / "corpus"
    assert load_profile(None, cwd=tmp_path).gpus == 8


def test_init_text_quotes_every_path(tmp_path):
    odd = tmp_path / 'a "b"'
    doc = tomllib.loads(init_text(odd, tmp_path / "co", gpus=2))
    assert doc["roots"]["corpus"] == str(odd / "corpus") and doc["roots"]["bench"] == str(tmp_path / "co")
    assert doc["gpus"] == 2
