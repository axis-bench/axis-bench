"""Steps 5 and 7: norm statistics, the run specs of a stage, and `qvla train` of each."""
from __future__ import annotations

import json
import os
import sys

from axis_qvla.cli import main
from axis_qvla.realrobot.layout import REPO
from axis_qvla.realrobot.steps import train_steps as step
from tests.unit.realrobot.steps.hosts import local_layout, qvla_toml

STAND_IN = {"norm_stats": {k: {"mean": [0.0] * 8, "std": [1.0] * 8, "q01": [-1.0] * 8, "q99": [1.0] * 8}
                           for k in ("state", "actions")}}


def _record(monkeypatch, rc=0):
    calls = []
    monkeypatch.setattr(step, "_train", lambda spec, *, host, print_only: calls.append((spec, print_only)) or rc)
    return calls


def test_pretrain_writes_both_stage1_specs_and_trains_each(tmp_path, monkeypatch, capsys):
    toml, lay = local_layout(tmp_path)
    calls = _record(monkeypatch)
    assert main(["realrobot", "pretrain", "--host", toml, "--print"]) == 0
    assert calls == [(lay.run_spec("pi05.pretrain.realrobot_bc"), True), (lay.run_spec("pi05.pretrain.realrobot_cond"), True)]
    assert all(spec.is_file() for spec, _ in calls)
    out = capsys.readouterr().out
    assert "norm statistics of d8 (realalign_sim): kept" in out
    assert f"OK    pi05.pretrain.realrobot_bc -> {lay.run_spec('pi05.pretrain.realrobot_bc')}" in out
    assert "training " not in out                       # --print trains nothing


def test_finetune_runs_a_subset_by_short_name(tmp_path, monkeypatch, capsys):
    toml, lay = local_layout(tmp_path)
    calls = _record(monkeypatch)
    assert main(["realrobot", "finetune", "--runs", "only", "--host", toml]) == 0
    assert calls == [(lay.run_spec("pi05.finetune.realrobot_only"), False)]
    assert "training pi05.finetune.realrobot_only (1 of 1)" in capsys.readouterr().out


def test_finetune_defaults_to_the_papers_three_runs(tmp_path, monkeypatch):
    toml, lay = local_layout(tmp_path)
    calls = _record(monkeypatch)
    assert main(["realrobot", "finetune", "--host", toml, "--print"]) == 0
    assert [spec for spec, _ in calls] == [lay.run_spec(f"pi05.finetune.realrobot_{r}") for r in ("bc", "cond", "only")]


def test_stale_norm_statistics_are_computed_again(tmp_path, monkeypatch, capsys):
    toml, lay = local_layout(tmp_path)
    lay.norm_file("pretrain").write_text(json.dumps(STAND_IN))
    _record(monkeypatch)
    assert main(["realrobot", "pretrain", "--runs", "cond", "--host", toml]) == 0
    assert f"norm statistics of d8 (realalign_sim) -> {lay.norm_file('pretrain')}" in capsys.readouterr().out


def test_a_failed_training_stops_the_stage(tmp_path, monkeypatch):
    toml, _lay = local_layout(tmp_path)
    calls = _record(monkeypatch, rc=3)
    assert main(["realrobot", "pretrain", "--host", toml]) == 3 and len(calls) == 1


def test_refusals_name_what_to_run(tmp_path, capsys):
    toml = str(qvla_toml(tmp_path))
    assert main(["realrobot", "pretrain", "--host", toml]) == 2
    assert "qvla realrobot rerender" in capsys.readouterr().err
    assert main(["realrobot", "finetune", "--host", toml]) == 2
    assert "qvla realrobot demos" in capsys.readouterr().err
    assert main(["realrobot", "pretrain", "--runs", "only", "--host", toml]) == 2
    assert "not a pretrain run" in capsys.readouterr().err


def test_train_is_stream_bs_command_in_a_child_process(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(step, "_run_child", lambda argv: seen.append(argv) or 0)
    assert step._train(tmp_path / "r.toml", host="q.toml", print_only=True) == 0
    assert step._train(tmp_path / "r.toml", host=None, print_only=False) == 0
    assert seen == [[sys.executable, "-m", "axis_qvla.cli", "train", str(tmp_path / "r.toml"), "--host", "q.toml", "--print"],
                    [sys.executable, "-m", "axis_qvla.cli", "train", str(tmp_path / "r.toml")]]


def test_the_child_inherits_this_checkout_and_passes_its_exit_code_through(monkeypatch):
    seen = {}

    class Done:
        returncode = 7

    monkeypatch.setattr(step.subprocess, "run", lambda argv, **kw: seen.update(argv=argv, **kw) or Done())
    assert step._run_child(["python", "-m", "x"]) == 7
    assert seen["argv"] == ["python", "-m", "x"] and seen["check"] is False
    assert str(REPO) in seen["env"]["PYTHONPATH"].split(os.pathsep)


def test_print_reaches_qvla_trains_own_print_in_the_child(tmp_path, capfd):
    """No stand-in for `_train`: each stage's `--print` runs the real `qvla train <spec> --host H --print`
    in a child, which accepts the run specs `fr3 local-runs` wrote and prints the one command per run."""
    toml, lay = local_layout(tmp_path)
    for stage, runs in (("pretrain", step.DEFAULTS["pretrain"]), ("finetune", step.DEFAULTS["finetune"])):
        assert main(["realrobot", stage, "--host", toml, "--print"]) == 0
        lines = capfd.readouterr().out.splitlines()
        for r in runs:
            spec = lay.run_spec(f"pi05.{stage}.realrobot_{r}")
            assert f"OK    pi05.{stage}.realrobot_{r}: {spec}" in lines
            assert f"qvla train {spec} --host {toml}" in lines
        assert not lay.paths.runs.exists()                  # --print writes no run directory
