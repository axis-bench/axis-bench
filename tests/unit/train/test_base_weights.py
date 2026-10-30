"""A run needs its base model in the base-weights copy: `qvla train` fetches a missing one before it
binds (verified against conf/recipes/bases.lock.json); `--render` refuses, naming the command. A run
from a published checkpoint still needs the base files its policy loads (the pi0.5 tokenizer, the
SmolVLM2 processor, the X-VLA base): fetched the same way, or named as `--serving <backbone>`."""
from __future__ import annotations

import pathlib

import pytest

from axis_qvla.cli import main
from axis_qvla.data import bases
from axis_qvla.data.hf import HfHttpError
from axis_qvla.train import ids, launch, runs
from axis_qvla.train.armspec import dumps_armspec, loads_armspec
from tests.unit.train._arms import H, arm_toml, host

PRETRAIN = dict(id="pi05.pretrain.b", stage="pretrain", preset="pi05.pretrain.full.paper",
                rowspace="libero5k_v2_d8", init="base:pi05_droid")


def _fill(paths, name):
    for p in runs.missing_base_files(name, paths):
        pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(p).write_bytes(b"")


def _machine(tmp_path):
    hosts, paths = host(tmp_path)
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text("")
    hosts.write_text(hosts.read_text() + f'\n[venv]\nonline = "{venv}"\n')
    return hosts, paths


def test_the_missing_files_are_the_locks(tmp_path):
    _, paths = host(tmp_path)
    want = sum(len(b["files"]) for b in bases.load()["bases"]["pi05_droid"])
    assert len(runs.missing_base_files("pi05_droid", paths)) == want
    _fill(paths, "pi05_droid")
    assert runs.missing_base_files("pi05_droid", paths) == []


def test_a_finetune_from_a_run_needs_that_runs_base(tmp_path):
    _, paths = host(tmp_path)
    pre = loads_armspec(arm_toml(**PRETRAIN))
    (ids.run_root(paths, pre.id)).mkdir(parents=True)
    (ids.run_root(paths, pre.id) / "run_spec.toml").write_text(dumps_armspec(pre))
    ft = loads_armspec(arm_toml(init="run:pi05.pretrain.b"))
    assert runs.init_base(pre, paths) == runs.init_base(ft, paths) == "pi05_droid"
    assert runs.init_base(loads_armspec(arm_toml()), paths) is None          # a published checkpoint
    assert runs.init_base(loads_armspec(arm_toml(init="run:nobody.ran.this")), paths) is None


def test_training_fetches_a_missing_base_first(tmp_path, monkeypatch, capsys):
    hosts, paths = _machine(tmp_path)
    fetched, ran = [], []
    monkeypatch.setattr(runs, "fetch_base", lambda name, p: fetched.append(name) or _fill(p, name))
    monkeypatch.setattr(launch, "RUNNER", lambda l: ran.append(l) or 0)
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml(**PRETRAIN))
    assert main(["train", str(arm), "--host", str(hosts), "--no-bind"]) == 0
    assert fetched == ["pi05_droid"] and len(ran) == 1
    assert "GET   base weights pi05_droid" in capsys.readouterr().out


def test_a_base_that_is_there_is_not_fetched_again(tmp_path, monkeypatch):
    hosts, paths = _machine(tmp_path)
    _fill(paths, "pi05_droid")
    monkeypatch.setattr(runs, "fetch_base", lambda name, p: pytest.fail("fetched a base that is there"))
    monkeypatch.setattr(launch, "RUNNER", lambda l: 0)
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml(**PRETRAIN))
    assert main(["train", str(arm), "--host", str(hosts), "--no-bind"]) == 0


def test_a_base_that_cannot_be_fetched_names_the_command_and_trains_nothing(tmp_path, monkeypatch, capsys):
    hosts, paths = _machine(tmp_path)

    def offline(name, p):
        raise HfHttpError(f"{name}: <urlopen error [Errno -3] Temporary failure in name resolution>")

    monkeypatch.setattr(runs, "fetch_base", offline)
    monkeypatch.setattr(launch, "RUNNER", lambda l: pytest.fail("trained without its base"))
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml(**PRETRAIN))
    assert main(["train", str(arm), "--host", str(hosts), "--no-bind"]) == 3
    out = capsys.readouterr().out
    assert "FAIL  pi05.pretrain.b: fetching the base weights pi05_droid" in out
    assert "`qvla data get base-weights --include pi05_droid`" in out
    assert not ids.run_root(paths, "pi05.pretrain.b").exists()


def test_render_fetches_nothing_and_binding_names_the_command(tmp_path, monkeypatch, capsys):
    hosts, _ = host(tmp_path)
    monkeypatch.setattr(runs, "fetch_base", lambda name, p: pytest.fail("--render fetched"))
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml(**PRETRAIN))
    assert main(["train", str(arm), "--host", str(hosts), "--render"]) == 2      # bound: inputs are missing
    assert "`qvla data get base-weights --include pi05_droid`" in capsys.readouterr().out


def test_print_never_looks_for_the_base(tmp_path, monkeypatch, capsys):
    hosts, _ = host(tmp_path)
    monkeypatch.setattr(runs, "missing_base_files", lambda name, p: pytest.fail("--print looked"))
    monkeypatch.setattr(runs, "fetch_base", lambda name, p: pytest.fail("--print fetched"))
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml(**PRETRAIN))
    assert main(["train", str(arm), "--host", str(hosts), "--print"]) == 0


def _fake_fetch(calls):
    """bases.fetch without a network: records what it was asked for and writes each file (empty)."""
    def fetch(lock, names, dest, **kw):
        calls.append({n: [b["dest"] for b in lock["bases"][n]] for n in names})
        for n in names:
            for block in lock["bases"][n]:
                for row in block["files"]:
                    out = pathlib.Path(dest) / block["dest"] / row["name"]
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(b"")
        return 0, 0
    return fetch


@pytest.mark.parametrize("family, want", [
    ("pi05", {"pi05_base": ["openpi/big_vision"]}),          # the 4 MB tokenizer, not the 12 GB params
    ("smolvla", {"smolvlm2_500m": [b["dest"] for b in bases.load()["bases"]["smolvlm2_500m"]]}),
    ("xvla", {"xvla_base": [b["dest"] for b in bases.load()["bases"]["xvla_base"]]}),   # bart included
])
def test_the_files_a_policy_loads_beside_a_published_checkpoint_are_fetched(tmp_path, monkeypatch, family, want):
    _, paths = host(tmp_path)
    calls = []
    monkeypatch.setattr(bases, "fetch", _fake_fetch(calls))
    assert runs.missing_serving_files(family, paths)
    runs.fetch_serving(family, paths)
    assert calls == [want] and runs.missing_serving_files(family, paths) == []


def test_the_subset_is_the_data_commands_one_table(tmp_path, monkeypatch):
    """`qvla data get base-weights --serving` and `qvla train` fetch by one table, axis_qvla.data.bases.SERVING:
    train keeps no copy of it, so a change there reaches both."""
    _, paths = host(tmp_path)
    assert not hasattr(runs, "_SERVING") and not hasattr(runs, "serving_lock")
    assert runs.missing_serving_files("pi05", paths) == [
        str(ids.base_weights_root(paths) / "openpi" / "big_vision" / "paligemma_tokenizer.model")]
    monkeypatch.setitem(bases.SERVING, "pi05", (("pi05_droid", "openpi/big_vision"),))
    assert runs.missing_serving_files("pi05", paths) == [
        str(ids.base_weights_root(paths) / "openpi" / "big_vision" / "paligemma_tokenizer.model")]   # same block
    monkeypatch.setitem(bases.SERVING, "pi05", (("pi05_droid", None),))
    assert len(runs.missing_serving_files("pi05", paths)) > 1                           # the whole droid base


def test_the_serving_hint_names_a_backbone_the_data_command_takes():
    """Every short name `init_serving` returns is a SERVING key and a `--serving` choice, so the FAIL line's
    `qvla data get base-weights --serving <backbone>` is a command that runs."""
    import argparse

    from axis_qvla.cli import build_parser

    def sub(parser, name):
        return next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction)).choices[name]

    get = sub(sub(build_parser(), "data"), "get")
    choices = next(a for a in get._actions if a.dest == "serving").choices
    assert set(runs.SHORT_NAMES.values()) == set(bases.SERVING) == set(choices)


def test_a_run_from_a_published_checkpoint_or_a_run_that_started_from_one_needs_the_serving_files(tmp_path):
    _, paths = host(tmp_path)
    pre = loads_armspec(arm_toml(id="pi05.pretrain.p", stage="pretrain", preset="pi05.pretrain.full.paper",
                                 rowspace="libero5k_v2_d8"))                       # init: a published checkpoint
    ids.run_root(paths, pre.id).mkdir(parents=True)
    (ids.run_root(paths, pre.id) / "run_spec.toml").write_text(dumps_armspec(pre))
    assert runs.init_serving(pre, paths) == runs.init_serving(loads_armspec(arm_toml(init="run:pi05.pretrain.p")),
                                                              paths) == "pi05"
    assert runs.init_serving(loads_armspec(arm_toml(**PRETRAIN)), paths) is None    # its base holds them
    lerobot = loads_armspec(arm_toml(id="xvla.finetune.x", backbone="xvla-lerobot", preset="xvla.finetune.paper",
                                     init=f"axisrobotics/qvla-v1-xvla:pretrain/x/020621#sha256:{H}", norm=False))
    assert runs.init_serving(lerobot, paths) == "xvla"


def test_training_from_a_published_checkpoint_fetches_its_serving_files_first(tmp_path, monkeypatch, capsys):
    hosts, paths = _machine(tmp_path)
    calls, ran = [], []
    monkeypatch.setattr(bases, "fetch", _fake_fetch(calls))
    monkeypatch.setattr(launch, "RUNNER", lambda l: ran.append(l) or 0)
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml())
    assert main(["train", str(arm), "--host", str(hosts), "--no-bind"]) == 0
    assert calls == [{"pi05_base": ["openpi/big_vision"]}] and len(ran) == 1
    assert "GET   base weights --serving pi05" in capsys.readouterr().out


def test_serving_files_that_cannot_be_fetched_name_the_command_and_train_nothing(tmp_path, monkeypatch, capsys):
    hosts, paths = _machine(tmp_path)

    def offline(lock, names, dest, **kw):
        raise HfHttpError("pi05_base: <urlopen error [Errno -3] Temporary failure in name resolution>")

    monkeypatch.setattr(bases, "fetch", offline)
    monkeypatch.setattr(launch, "RUNNER", lambda l: pytest.fail("trained without its tokenizer"))
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml())
    assert main(["train", str(arm), "--host", str(hosts), "--no-bind"]) == 3
    out = capsys.readouterr().out
    assert "FAIL  pi05.finetune.example: fetching the pi05 base files its policy loads" in out
    assert "`qvla data get base-weights --serving pi05`" in out
