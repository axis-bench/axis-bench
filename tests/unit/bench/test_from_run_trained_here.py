"""`bench eval --from-run` over runs laid out as `qvla train` lays them out, through the real
axis_qvla.train.serve.serve_spec: the two streams' halves of the method author's last step, joined.

Why: tests/unit/bench/test_local_run.py replaces the training package's serve interface with a stub
of its declared contract, and tests/unit/train/test_serve.py checks that interface without the
benchmark. Neither sees the other's actual objects: a field the stub has and the real ServeSpec lacks,
or an eval.json the benchmark reads otherwise than the training package writes it, passes both. Here
three finished runs are built as the training tests build them (a pi0.5 fine-tune, a SmolVLA fine-tune,
and a `--cond` pi0.5 fine-tune whose eval.json records ["bare", "cond"]) and `--print` and `--dry-run`
plan each one end to end.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from axis_qvla.cli import main
from axis_qvla.core.hosts import load_host
from axis_qvla.core.paths import Paths
from axis_qvla.train.serve import write_eval_record
from tests.unit.bench.eval_stubs import lerobot_host, make_envs
from tests.unit.bench.test_openpi_serve import host  # noqa: F401  (a fixture)
from axis_qvla.train import ids
from axis_qvla.train.armspec import dumps_armspec, loads_armspec
from tests.unit.train._arms import H, arm_toml

PI05 = "pi05.finetune.mine"
COND = "pi05.finetune.mine.cond"
SMOLVLA = "smolvla.finetune.mine"
SMOLVLA_FT = dict(id=SMOLVLA, backbone="smolvla-lerobot", preset="smolvla.finetune.paper", norm=False,
                  init=f"axisrobotics/qvla-v1-smolvla:pretrain/example/020621#sha256:{H}")


def _run(paths, text: str) -> Path:
    """<runs>/<id>/run_spec.toml, as `qvla train` renders it."""
    spec = loads_armspec(text)
    run = ids.run_root(paths, spec.id)
    run.mkdir(parents=True)
    (run / "run_spec.toml").write_text(dumps_armspec(spec))
    return run


def _finished_pi05(paths, text: str) -> Path:
    """A pi0.5 run that trained its whole budget: train/final/{params,assets} and its summary."""
    run = _run(paths, text)
    (run / "train" / "final" / "params").mkdir(parents=True)
    (run / "train" / "final" / "assets").mkdir()
    (run / "train" / "train_summary.json").write_text(json.dumps(
        {"trained_steps": 13580, "declared_steps": 13580, "smoke": False}))
    return run


def _finished_smolvla(paths) -> Path:
    """A SmolVLA run whose last step is its whole budget, with its pretrained_model/ written."""
    run = _run(paths, arm_toml(**SMOLVLA_FT))
    last = run / "train" / "checkpoints" / "last"
    (last / "training_state").mkdir(parents=True)
    (last / "pretrained_model").mkdir()
    (last / "training_state" / "training_step.json").write_text('{"step": 6790}')
    (last / "pretrained_model" / "train_config.json").write_text('{"steps": 6790}')
    (last / "pretrained_model" / "model.safetensors").write_bytes(b"")
    write_eval_record(run, pretrain="smolvla.pretrain.mine", pretrain_seed=0, cond=False)
    return run


@pytest.fixture
def trained(host):
    """The `host` fixture's demo profile (with the LeRobot servers' roots) and three finished runs under
    its runs root, as tests/unit/train/test_serve.py lays them out
    (built here: the bench tests share no other test module's privates)."""
    lerobot_host(host)
    make_envs(host)
    paths = Paths(load_host("demo"))
    # A generated pi0.5 fine-tune names no compiled labels; --cond tags every frame instead.
    pi05 = _finished_pi05(paths, arm_toml(id=PI05, init="run:pi05.pretrain.mine", bundle=False))
    write_eval_record(pi05, pretrain="pi05.pretrain.mine", pretrain_seed=7, cond=False)
    cond = _finished_pi05(paths, arm_toml(id=COND, init="run:pi05.pretrain.mine", bundle=False,
                                             overrides='"data.quality_tag" = 5'))
    write_eval_record(cond, pretrain="pi05.pretrain.mine", pretrain_seed=7, cond=True)
    _finished_smolvla(paths)
    assert json.loads((cond / "eval.json").read_text())["serving"] == ["bare", "cond"]
    return host


def _tree(d: Path) -> dict[str, bytes]:
    return {p.relative_to(d).as_posix(): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("run_id, arms", [(PI05, [PI05]), (SMOLVLA, [SMOLVLA]), (COND, [f"{COND}_b", f"{COND}_c"])])
def test_print_plans_every_mode_the_run_records_and_writes_nothing(trained, capsys, run_id, arms):
    before = _tree(trained)
    assert main(["bench", "eval", "--host", "demo", "--from-run", run_id, "--print"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert _tree(trained) == before                                        # not even serve_config.json
    assert [ln.rsplit(" as run ", 1)[1] for ln in lines if ln.startswith("# serves:")] == arms
    assert [ln for ln in lines if ln.startswith("# results file:")] == [
        f"# results file: {trained / 'r1' / f'out_{a}' / 'results.json'}" for a in arms]
    assert not any("does not resolve" in ln for ln in lines)              # a finished run, resolved
    seeds = "pretraining seed 0" if run_id == SMOLVLA else "pretraining seed 7"
    assert f"# from run: {run_id} (fine-tune seed 42, {seeds})" in lines   # eval.json's, read by serve_spec
    assert lines[-1].endswith(f"bench eval --from-run {run_id} --host demo")


@pytest.mark.parametrize("run_id, arms", [(PI05, [PI05]), (SMOLVLA, [SMOLVLA]), (COND, [f"{COND}_b", f"{COND}_c"])])
def test_dry_run_plans_each_mode_on_the_runs_backbone(trained, capsys, run_id, arms):
    assert main(["bench", "eval", "--host", "demo", "--from-run", run_id, "--dry-run"]) == 0
    out = capsys.readouterr().out
    for a in arms:
        assert f"--run={a}" in out
    if run_id == COND:
        assert "--serving=bare" in out and "--serving=conditioned" in out
    if run_id == SMOLVLA:
        ckpt = trained / "runs" / SMOLVLA / "train" / "checkpoints" / "last" / "pretrained_model"
        assert "--backend=smolvla" in out and f"--checkpoint={ckpt}" in out
    else:
        assert f"--checkpoint={trained / 'runs' / run_id / 'train' / 'final'}" in out
