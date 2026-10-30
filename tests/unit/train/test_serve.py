"""What `qvla bench eval --from-run` serves for a run trained here (axis_qvla.train.serve.serve_spec)."""
from __future__ import annotations

import json
import os

import pytest

from axis_qvla.train import ids
from axis_qvla.train.armspec import dumps_armspec, loads_armspec
from axis_qvla.train.serve import RunNotReady, ServeSpecError, serve_spec, write_eval_record
from tests.unit.train._arms import H, arm_toml, host

SMOLVLA_FT = dict(id="smolvla.finetune.example", backbone="smolvla-lerobot", preset="smolvla.finetune.paper",
                  init=f"axisrobotics/qvla-v1-smolvla:pretrain/example/020621#sha256:{H}", norm=False)


def _run(paths, text: str):
    spec = loads_armspec(text)
    run = ids.run_root(paths, spec.id)
    run.mkdir(parents=True)
    (run / "run_spec.toml").write_text(dumps_armspec(spec))
    return spec, run


def _finished_pi05(paths, text=None, *, trained=13580, smoke=False):
    spec, run = _run(paths, text or arm_toml())
    (run / "train" / "final" / "params").mkdir(parents=True)
    (run / "train" / "final" / "assets").mkdir()
    (run / "train" / "train_summary.json").write_text(json.dumps(
        {"trained_steps": trained, "declared_steps": 13580, "smoke": smoke}))
    return spec, run


def test_a_finished_pi05_finetune_serves_bare_and_cond_from_its_final_step(tmp_path):
    _, paths = host(tmp_path)
    spec, run = _finished_pi05(paths)
    write_eval_record(run, pretrain="pi05.pretrain.mine", pretrain_seed=7, cond=True)
    s = serve_spec(run, paths)
    assert (s.run_id, s.backbone, s.backend, s.stage) == (spec.id, "pi05", "openpi", "finetune")
    assert s.checkpoint == run / "train" / "final"
    cfg = json.loads(s.config.read_text())
    assert s.config == run / "serve_config.json" and cfg["exp_name"] == "serve" and cfg["data"]["bundle"] is None
    assert [(m.name, m.serving, dict(m.fields)) for m in s.modes] == [
        ("bare", "bare", {}), ("cond", "conditioned", {"quality": "5"})]
    assert (s.ft_seed, s.pretrain_seed, s.pretrain) == (42, 7, "pi05.pretrain.mine")


def test_a_run_without_an_eval_record_serves_bare(tmp_path):
    _, paths = host(tmp_path)
    _, run = _finished_pi05(paths)
    assert [m.name for m in serve_spec(run, paths).modes] == ["bare"]


def test_a_malformed_eval_record_is_refused_not_read_as_bare(tmp_path):
    _, paths = host(tmp_path)
    _, run = _finished_pi05(paths)
    (run / "eval.json").write_text(json.dumps({"schema": "qvla.train_eval/1", "serving": ["cond"]}))
    with pytest.raises(ServeSpecError, match="bare first"):
        serve_spec(run, paths)


@pytest.mark.parametrize("kw, why, fix", [
    (dict(trained=10), "it trained 10 of 13580 steps", "continue it with `qvla train {spec} --resume`"),
    (dict(smoke=True), "it is a smoke run", "train it without --smoke: `qvla train {spec}`")])
def test_an_unfinished_or_smoke_run_is_refused_naming_the_fix(tmp_path, kw, why, fix):
    _, paths = host(tmp_path)
    spec, run = _finished_pi05(paths, **kw)
    with pytest.raises(ServeSpecError) as err:
        serve_spec(run, paths)
    msg = str(err.value)
    assert msg.startswith(f"{spec.id} cannot be served: {why}") and f"run:{spec.id}" not in msg   # one id
    assert fix.format(spec=run / "run_spec.toml") in msg


def test_a_run_whose_final_step_lacks_a_file_is_trained_again(tmp_path):
    """A pi0.5 step without its assets/ (the norm stats the server reads) never serves."""
    _, paths = host(tmp_path)
    spec, run = _finished_pi05(paths)
    (run / "train" / "final" / "assets").rmdir()
    (run / "spec.toml").write_text(dumps_armspec(spec))         # planned by `qvla train pretrain|finetune`
    with pytest.raises(ServeSpecError) as err:
        serve_spec(run, paths)
    assert str(err.value) == (f"{spec.id} cannot be served: {run / 'train' / 'final' / 'assets'} is not there; "
                              f"remove {run / 'train'} yourself (qvla train never deletes a directory) and train "
                              f"it again with `qvla train {run / 'spec.toml'}`")


@pytest.mark.parametrize("field, value", [("pretrain_seed", "42"), ("pretrain_seed", True), ("pretrain_seed", 1.5),
                                          ("pretrain", 7), ("pretrain", ["pi05.pretrain.mine"])])
def test_an_eval_record_with_a_malformed_seed_or_pretrain_is_refused(tmp_path, field, value):
    _, paths = host(tmp_path)
    _, run = _finished_pi05(paths)
    write_eval_record(run, pretrain="pi05.pretrain.mine", pretrain_seed=7, cond=True)
    doc = json.loads((run / "eval.json").read_text())
    (run / "eval.json").write_text(json.dumps({**doc, field: value}))
    with pytest.raises(ServeSpecError, match=f"{field} is "):
        serve_spec(run, paths)
    (run / "eval.json").write_text(json.dumps({**doc, field: None}))      # null: not known, which is valid
    assert getattr(serve_spec(run, paths), field) is None


def test_a_directory_qvla_train_did_not_write_is_refused(tmp_path):
    _, paths = host(tmp_path)
    (paths.runs / "x").mkdir(parents=True)
    with pytest.raises(RunNotReady, match="not a run"):          # e.g. only planned (`--print`): not trained yet
        serve_spec(paths.runs / "x", paths)


def test_a_run_spec_copied_into_another_directory_is_refused(tmp_path):
    _, paths = host(tmp_path)
    _, run = _finished_pi05(paths)
    other = paths.runs / "copy"
    other.mkdir()
    (other / "run_spec.toml").write_bytes((run / "run_spec.toml").read_bytes())
    with pytest.raises(ServeSpecError, match="whose directory is"):
        serve_spec(other, paths)


def test_a_finished_smolvla_finetune_serves_its_last_pretrained_model(tmp_path):
    _, paths = host(tmp_path)
    spec, run = _run(paths, arm_toml(**SMOLVLA_FT))
    last = run / "train" / "checkpoints" / "last"
    (last / "training_state").mkdir(parents=True)
    (last / "pretrained_model").mkdir()
    (last / "training_state" / "training_step.json").write_text('{"step": 10}')
    (last / "pretrained_model" / "train_config.json").write_text('{"steps": 6790}')
    with pytest.raises(ServeSpecError, match=r"it trained 10 of 6790 steps; .*\(a LeRobot run is not resumed\)"):
        serve_spec(run, paths)
    (last / "training_state" / "training_step.json").write_text('{"step": 6790}')
    with pytest.raises(ServeSpecError, match="model.safetensors is not there; remove .* and train it again"):
        serve_spec(run, paths)                                    # a step LeRobot never finished writing
    (last / "pretrained_model" / "model.safetensors").write_bytes(b"")
    s = serve_spec(run, paths)
    assert (s.backbone, s.backend, s.checkpoint, s.config) == ("smolvla", "smolvla", last / "pretrained_model", None)
    assert (s.ft_seed, s.pretrain_seed) == (42, None)


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads through a 000 directory")
def test_a_checkpoint_file_that_cannot_be_stated_is_named_as_such_not_as_missing(tmp_path):
    from axis_qvla.train.runs import not_a_dir, not_a_file

    d = tmp_path / "pretrained_model"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"")
    os.chmod(d, 0o000)
    try:
        assert not_a_file(d / "model.safetensors") == not_a_dir(d / "assets") == "cannot be read (PermissionError)"
    finally:
        os.chmod(d, 0o755)
    assert not_a_file(d / "model.safetensors") is None and not_a_file(d) == "is not a file"


def test_write_false_renders_the_pi05_config_without_writing_anything(tmp_path):
    _, paths = host(tmp_path)
    _, run = _finished_pi05(paths)
    before = sorted(p.relative_to(run) for p in run.rglob("*"))
    s = serve_spec(run, paths, write=False)
    assert sorted(p.relative_to(run) for p in run.rglob("*")) == before and not s.config.exists()
    assert s.config == run / "serve_config.json" and json.loads(s.config_bytes)["exp_name"] == "serve"
    assert serve_spec(run, paths).config.read_bytes() == s.config_bytes          # write=True: the same bytes


def test_a_lerobot_run_has_no_config_bytes(tmp_path):
    _, paths = host(tmp_path)
    spec, run = _run(paths, arm_toml(**SMOLVLA_FT))
    last = run / "train" / "checkpoints" / "last"
    (last / "training_state").mkdir(parents=True)
    (last / "pretrained_model").mkdir()
    (last / "training_state" / "training_step.json").write_text('{"step": 6790}')
    (last / "pretrained_model" / "train_config.json").write_text('{"steps": 6790}')
    (last / "pretrained_model" / "model.safetensors").write_bytes(b"")
    s = serve_spec(run, paths, write=False)
    assert (s.config, s.config_bytes) == (None, None)


def test_without_an_eval_record_the_pretraining_comes_from_the_init_run(tmp_path):
    _, paths = host(tmp_path)
    _run(paths, arm_toml(id="pi05.pretrain.mine", stage="pretrain", preset="pi05.pretrain.full.paper",
                         rowspace="libero5k_v2_d8", init="base:pi05_droid", seed=7))
    _, run = _finished_pi05(paths, arm_toml(id="pi05.finetune.mine", init="run:pi05.pretrain.mine"))
    s = serve_spec(run, paths)
    assert (s.pretrain, s.pretrain_seed, s.ft_seed) == ("pi05.pretrain.mine", 7, 42)
    _, other = _finished_pi05(paths, arm_toml(id="pi05.finetune.gone", init="run:pi05.pretrain.gone"))
    assert (serve_spec(other, paths).pretrain, serve_spec(other, paths).pretrain_seed) == ("pi05.pretrain.gone", None)


@pytest.mark.parametrize("dest, seed", [
    ("axisrobotics/qvla-v1-pi05:pretrain/pi05_axis_cfg_d8__libero5k_d8_cfg_phase_s43/20604", 43),  # its paper run: seed 43
    ("axisrobotics/qvla-v1-pi05:ft/pi05_axis_heldout_qual_v2_bc__seed44_bc/13579", 44),           # a fine-tune: its pretrain_seed
    ("axisrobotics/qvla-v1-pi05:pretrain/example/20604", None)])                                  # no paper run trained it
def test_a_fine_tune_of_a_published_checkpoint_takes_the_releases_seed(tmp_path, dest, seed):
    _, paths = host(tmp_path)
    _, run = _finished_pi05(paths, arm_toml(init=f"{dest}#sha256:{H}"))
    s = serve_spec(run, paths)
    assert (s.pretrain, s.pretrain_seed) == (None, seed)


def test_an_eval_record_wins_over_the_init(tmp_path):
    _, paths = host(tmp_path)
    _, run = _finished_pi05(paths, arm_toml(id="pi05.finetune.mine", init="run:pi05.pretrain.mine"))
    write_eval_record(run, pretrain="pi05.pretrain.mine", pretrain_seed=3, cond=False)
    assert serve_spec(run, paths).pretrain_seed == 3


def test_not_ready_is_told_apart_from_a_defect(tmp_path):
    _, paths = host(tmp_path)
    with pytest.raises(RunNotReady, match="holds no run_spec.toml; train it first"):
        serve_spec(paths.runs / "pi05.finetune.nobody", paths)            # not there at all
    _, run = _finished_pi05(paths, trained=10)
    with pytest.raises(RunNotReady, match="trained 10 of 13580"):
        serve_spec(run, paths)
    (run / "train" / "train_summary.json").write_text(json.dumps({"trained_steps": 13580, "declared_steps": 13580}))
    (run / "train" / "final" / "assets").rmdir()
    with pytest.raises(RunNotReady, match="assets is not there"):
        serve_spec(run, paths)
    (run / "train" / "final" / "assets").mkdir()
    (run / "eval.json").write_text("{}")
    with pytest.raises(ServeSpecError) as err:
        serve_spec(run, paths)
    assert not isinstance(err.value, RunNotReady)                         # a malformed record is a defect
    (run / "eval.json").unlink()
    other = paths.runs / "copy"
    other.mkdir()
    (other / "run_spec.toml").write_bytes((run / "run_spec.toml").read_bytes())
    with pytest.raises(ServeSpecError) as err:
        serve_spec(other, paths)
    assert not isinstance(err.value, RunNotReady)


@pytest.mark.parametrize("pi05, rel, text", [
    (True, "train_summary.json", "not json"),
    (True, "train_summary.json", '{"trained_steps": "13580", "declared_steps": 13580}'),
    (False, "checkpoints/last/training_state/training_step.json", "not json"),
    (False, "checkpoints/last/pretrained_model/train_config.json", "{}")])
def test_a_step_file_that_is_there_but_malformed_is_a_defect_not_a_run_to_resume(tmp_path, pi05, rel, text):
    _, paths = host(tmp_path)
    if pi05:
        _, run = _finished_pi05(paths)
    else:
        _, run = _run(paths, arm_toml(**SMOLVLA_FT))
        last = run / "train" / "checkpoints" / "last"
        (last / "training_state").mkdir(parents=True)
        (last / "pretrained_model").mkdir()
        (last / "training_state" / "training_step.json").write_text('{"step": 6790}')
        (last / "pretrained_model" / "train_config.json").write_text('{"steps": 6790}')
        (last / "pretrained_model" / "model.safetensors").write_bytes(b"")
    (run / "train" / rel).write_text(text)
    with pytest.raises(ServeSpecError) as err:
        serve_spec(run, paths)
    assert not isinstance(err.value, RunNotReady) and "--resume" not in str(err.value)
    assert "repair it, or remove" in str(err.value)


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads through a 000 file")
def test_an_unreadable_summary_is_a_defect(tmp_path):
    _, paths = host(tmp_path)
    _, run = _finished_pi05(paths)
    summary = run / "train" / "train_summary.json"
    os.chmod(summary, 0o000)
    try:
        with pytest.raises(ServeSpecError, match="cannot be read") as err:
            serve_spec(run, paths)
        assert not isinstance(err.value, RunNotReady)
    finally:
        os.chmod(summary, 0o644)
