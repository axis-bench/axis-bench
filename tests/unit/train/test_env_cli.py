"""`qvla env build <name>` runs an environment's pinned build lines in its checkout, then checks it;
`--print` only prints them. Names are backbones (pi05, smolvla, xvla) or tiers (eval, offline,
droid_convert); SmolVLA and X-VLA share one environment."""
from __future__ import annotations

import shutil

import pytest

from axis_qvla.cli import main
from axis_qvla.core.hosts import load_host
from axis_qvla.train import envs, launch
from axis_qvla.train.envs import REPO, VENV_DIRS, env_python, load_env, tier_of


@pytest.fixture
def venvs(tmp_path, monkeypatch):
    """Every environment's venv under tmp_path (none built), so no test reads or removes a real one."""
    monkeypatch.setattr(envs, "venv_dir", lambda tier, profile=None: tmp_path / "venvs" / tier)
    return tmp_path / "venvs"


def _built(venvs, tier):
    python = venvs / tier / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    return python


@pytest.fixture
def fake_build(monkeypatch, venvs):
    ran = []
    monkeypatch.setattr(launch, "RUNNER", lambda l: ran.append(l) or 0)
    monkeypatch.setattr(envs, "check_lock", lambda spec, checkout=None: [])
    monkeypatch.setattr(envs, "check_built", lambda spec, python: [])
    monkeypatch.setattr(envs, "check_closure", lambda spec, python, checkout=None: [])
    monkeypatch.setattr(shutil, "which", lambda name: f"bin/{name}")
    return ran


def test_print_lists_the_build_in_its_checkout(capsys):
    assert main(["env", "build", "pi05", "--print"]) == 0
    out = capsys.readouterr().out
    assert f"cd {REPO / 'third_party' / 'openpi'} && sh -c 'GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --python 3.11'" in out


def test_smolvla_and_xvla_share_one_environment():
    assert tier_of("smolvla") == tier_of("xvla") == "lerobot" and tier_of("pi05") == "openpi"
    assert tier_of("render") == "render" and tier_of("droid_convert") == "droid_convert"
    with pytest.raises(envs.EnvError, match="pi05"):
        tier_of("openpi0")


def test_the_real_robot_renderer_builds_from_its_own_lock(capsys):
    spec = load_env("render")
    assert spec.checkout == "." and spec.lock == "axis_qvla/realrobot/render/render.lock.txt"
    assert envs.check_lock(spec) == []                                  # the pinned digest is the lock's
    assert main(["env", "build", "render", "--print"]) == 0
    assert "uv venv --python 3.11 .venv-render" in capsys.readouterr().out


def test_build_runs_every_line_in_the_checkout_then_checks(fake_build, capsys):
    assert main(["env", "build", "smolvla"]) == 0
    assert [l.argv for l in fake_build] == [("sh", "-c", line) for line in load_env("lerobot").build]
    assert {l.cwd for l in fake_build} == {REPO / "third_party" / "lerobot"}
    assert f"OK    lerobot environment at {env_python('lerobot')}" in capsys.readouterr().out


def test_an_environment_that_is_built_and_passes_its_check_is_left_as_it_is(fake_build, venvs, capsys):
    python = _built(venvs, "eval")
    assert main(["env", "build", "eval"]) == 0
    assert fake_build == [] and capsys.readouterr().out.splitlines()[0] == f"OK    eval already built at {python}"


def test_a_built_environment_that_differs_from_its_pins_is_repaired_in_place(fake_build, venvs, monkeypatch, capsys):
    """`uv venv` refuses a directory that already holds one, so the repair runs every line but it."""
    _built(venvs, "eval")
    monkeypatch.setattr(envs, "check_built", lambda spec, python: ["numpy 2.0 != 1.26"] if not fake_build else [])
    assert main(["env", "build", "eval"]) == 0
    lines = load_env("eval").build
    assert [l.argv[-1] for l in fake_build] == [x for x in lines if not x.startswith("uv venv")] != list(lines)
    assert "SKIP  " in capsys.readouterr().out and (venvs / "eval").is_dir()


def test_rebuild_removes_a_uv_venv_environment_and_runs_every_line(fake_build, venvs, capsys):
    _built(venvs, "eval")
    assert main(["env", "build", "eval", "--rebuild"]) == 0
    assert [l.argv[-1] for l in fake_build] == list(load_env("eval").build)
    assert f"RM    {venvs / 'eval'}" in capsys.readouterr().out and not (venvs / "eval").exists()


def test_rebuild_refuses_a_symlinked_venv_and_removes_nothing(fake_build, venvs, tmp_path, capsys):
    target = tmp_path / "elsewhere"
    (target / "bin").mkdir(parents=True)
    (target / "bin" / "python").write_text("")
    venvs.mkdir()
    (venvs / "eval").symlink_to(target)
    assert main(["env", "build", "eval", "--rebuild"]) == 2
    assert f"FAIL  {venvs / 'eval'} is a symlink; remove it yourself" in capsys.readouterr().out
    assert fake_build == [] and (target / "bin" / "python").exists() and (venvs / "eval").is_symlink()


def test_a_venv_whose_interpreter_link_dangles_is_built_and_failing(fake_build, venvs, monkeypatch, capsys):
    """Not "not built": `uv venv` would refuse the directory, so it is repaired in place."""
    (venvs / "eval" / "bin").mkdir(parents=True)
    (venvs / "eval" / "bin" / "python").symlink_to(venvs / "gone" / "python3")
    monkeypatch.setattr(envs, "check_built", lambda spec, python: ["cannot start"])
    assert main(["env", "build", "eval"]) == 1
    out = capsys.readouterr().out
    assert "SKIP  " in out and not any(l.argv[-1].startswith("uv venv") for l in fake_build)
    assert "run `qvla env build eval --rebuild`" in out.splitlines()[-1] and out.splitlines()[-1].startswith("FAIL")


def test_a_venv_with_only_its_pyvenv_cfg_is_built(fake_build, venvs, capsys):
    (venvs / "eval").mkdir(parents=True)
    (venvs / "eval" / "pyvenv.cfg").write_text("home = /nowhere\n")
    assert main(["env", "build", "eval", "--rebuild"]) == 0
    assert f"RM    {venvs / 'eval'}" in capsys.readouterr().out


def test_rebuild_of_a_uv_sync_environment_syncs_it_in_place(fake_build, venvs):
    _built(venvs, "lerobot")
    assert main(["env", "build", "smolvla", "--rebuild"]) == 0
    assert [l.argv[-1] for l in fake_build] == list(load_env("lerobot").build) and (venvs / "lerobot").is_dir()


def test_a_failing_line_stops_the_build_and_its_exit_code_is_returned(fake_build, monkeypatch, capsys):
    monkeypatch.setattr(launch, "RUNNER", lambda l: fake_build.append(l) or 3)
    assert main(["env", "build", "eval"]) == 3
    assert len(fake_build) == 1 and "FAIL  exit 3" in capsys.readouterr().out


def test_without_uv_the_build_names_how_to_get_it(fake_build, monkeypatch, capsys):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert main(["env", "build", "eval"]) == 2
    assert "pip install uv" in capsys.readouterr().out and fake_build == []


def test_the_interpreter_is_the_profiles_when_it_names_one(tmp_path):
    d = tmp_path / "hosts"
    d.mkdir()
    roots = "\n".join(f'{k} = "{tmp_path / k}"' for k in ("corpus", "cache", "scratch", "runs"))
    (d / "qvla.toml").write_text(f'[roots]\n{roots}\nsmolvla_venv = "{tmp_path / "lv"}"\n\n'
                                 f'[venv]\nonline = "{tmp_path / "ov"}"\n')
    profile = load_host("qvla", hosts_dir=d)
    assert env_python("openpi", profile) == tmp_path / "ov" / "bin" / "python"
    assert env_python("lerobot", profile) == tmp_path / "lv" / "bin" / "python"
    assert env_python("eval", None) == REPO / ".venv-eval" / "bin" / "python"
    assert "qvla env build pi05" in envs.missing_problem("openpi", profile)


def test_every_venv_dir_is_where_its_build_puts_it():
    for tier, rel in VENV_DIRS.items():
        spec = load_env(tier)
        if spec.build[0].startswith("uv venv"):
            assert spec.checkout == "." and spec.build[0].split()[-1] == rel, tier
        else:
            assert rel == f"{spec.checkout}/.venv", tier


def test_qvla_init_names_each_environment_where_env_build_puts_it(tmp_path):
    """`qvla bench eval` and `qvla train` read the environments from qvla.toml: what `qvla init`
    writes must be exactly where `qvla env build` builds them."""
    from axis_qvla.data.userconf import init_text

    d = tmp_path / "hosts"
    d.mkdir()
    (d / "qvla.toml").write_text(init_text(tmp_path / "srv", REPO))
    profile = load_host("qvla", hosts_dir=d)
    for tier in ("openpi", "lerobot", "eval"):
        assert envs.venv_dir(tier, profile) == REPO / VENV_DIRS[tier], tier
    assert profile.roots["openpi"] == envs.checkout("openpi")
