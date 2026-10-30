"""An arm renders to one backbone config, from the arm, its recipe and the host's roots alone."""
from __future__ import annotations

import dataclasses
import json
import pathlib
import shutil

import pytest

from axis_qvla.data.layout import BENCH, CORPUS, checkpoint_dest, default_dest, norm_stats_dest

from axis_qvla.train import ids
from axis_qvla.train.armspec import loads_armspec
from axis_qvla.train.generate import GenerateError, openpi_config, render
from axis_qvla.train.lerobot_cfg import lerobot_argv, lerobot_config
from axis_qvla.train.recipes import RECIPES_DIR, load_recipe
from tests.unit.train._arms import H, arm_toml, host

REPO = pathlib.Path(__file__).resolve().parents[3]

OPENPI_TOP_KEYS = {"schema", "name", "exp_name", "seed", "batch_size", "num_train_steps", "ema_decay",
                   "model", "lr_schedule", "optimizer", "freeze", "fsdp_devices", "num_workers",
                   "save_interval", "keep_period", "log_interval", "wandb_enabled", "overwrite",
                   "resume", "checkpoint_base_dir", "assets_base_dir", "weight_loader", "data", "identity"}
OPENPI_DATA_KEYS = {"repo_id", "asset_id", "image_center_crop", "default_prompt", "prompt_from_task",
                    "roots_index", "ranges_path", "corpus_base", "bundle", "bundle_id", "norm_stats_dir", "smoke"}

LEGACY_ENV = ("AXIS_PRETRAIN_ROOTS_INDEX", "AXIS_PRETRAIN_RANGES", "AXIS_PRETRAIN_AWR_WEIGHTS",
              "AXIS_PRETRAIN_CORPUS_ROOT", "HELDOUT_ROOTS_ALL", "HELDOUT_RANGES_ALL", "SLB_CFG_METADATA",
              "AXIS_DATALOADER_ROOT", "AXIS_EEF_INIT_CKPT")


def _ft(**kw):
    s = loads_armspec(arm_toml(**kw))
    return s, load_recipe(s.preset)


def test_a_pi05_finetune_renders_the_launched_recipe(tmp_path):
    _, paths = host(tmp_path)
    spec, recipe = _ft()
    c = openpi_config(spec, recipe, paths)
    assert (c["num_train_steps"], c["batch_size"], c["seed"], c["ema_decay"]) == (13580, 32, 42, None)
    assert c["lr_schedule"] == {"warmup_steps": 500, "peak_lr": 2.5e-05, "decay_steps": 13580, "decay_lr": 2.5e-05}
    assert c["model"]["paligemma_variant"] == "gemma_2b_lora" and c["freeze"] == "lora"
    # Every path is where the data layout (`qvla data get`) puts it.
    assert c["weight_loader"]["params_path"] == str(
        checkpoint_dest("axisrobotics/qvla-v1-pi05:pretrain/example/20604", paths) / "params")
    assert c["data"]["bundle"] == str(default_dest(CORPUS, paths, f"bundles/{H}"))
    assert c["data"]["roots_index"] == str(default_dest(BENCH, paths, "ft_demos/splits/qual_v2.roots.json"))
    assert c["data"]["ranges_path"] == str(default_dest(BENCH, paths, "ft_demos/splits/qual_v2.ranges.json"))
    assert c["data"]["corpus_base"] == str(default_dest(BENCH, paths))
    assert c["data"]["norm_stats_dir"] == str(norm_stats_dest("heldout24_qual_v2", paths))
    assert c["identity"]["bundle"] == f"sha256:{H}" and c["exp_name"] == "train" and not c["data"]["smoke"]


def test_overrides_reach_the_config(tmp_path):
    _, paths = host(tmp_path)
    spec, recipe = _ft(overrides='"lr_schedule.warmup_steps" = 100\nnum_train_steps = 4168')
    c = openpi_config(spec, recipe, paths)
    assert c["lr_schedule"]["warmup_steps"] == 100 and c["num_train_steps"] == 4168


def test_smoke_is_20_steps_and_saves_once(tmp_path):
    _, paths = host(tmp_path)
    c = openpi_config(*_ft(), paths, smoke=True)
    assert (c["num_train_steps"], c["save_interval"], c["exp_name"], c["overwrite"], c["data"]["smoke"]) == (
        20, 20, "smoke", True, True)


def test_a_real_run_below_100_steps_is_refused(tmp_path):
    _, paths = host(tmp_path)
    with pytest.raises(GenerateError, match="100-step floor"):
        openpi_config(*_ft(overrides="num_train_steps = 50"), paths)


def test_the_openpi_config_has_exactly_the_keys_task_8_allows(tmp_path):
    _, paths = host(tmp_path)
    c = openpi_config(*_ft(), paths)
    assert sorted(c) == sorted(OPENPI_TOP_KEYS)
    assert sorted(c["data"]) == sorted(OPENPI_DATA_KEYS)


def test_data_quality_tag_is_consumed_at_bind_not_dropped(tmp_path):
    """`data.quality_tag` is a declared, null-by-default field of every pi0.5 finetune recipe
    that `openpi_config` never reads -- the pre-launch binding step bakes it into the signal
    bundle it compiles for an arm that names none -- so overriding it there must render cleanly,
    not be caught by the "unused field" refusal below."""
    _, paths = host(tmp_path)
    c = openpi_config(*_ft(bundle=False, overrides='"data.quality_tag" = 5'), paths, bundle=f"sha256:{H}",
                      bundle_dir="bound")
    assert c["num_train_steps"] == 13580   # rendered fine; nothing about quality_tag appears here
    assert openpi_config(*_ft(bundle=False, overrides='"data.quality_tag" = 5'), paths, serve=True)["data"]["bundle"] is None


def test_a_quality_tag_beside_a_named_bundle_is_refused_without_binding(tmp_path):
    """With a bundle named the tag would change nothing (the bundle's codes are its own), so the
    config generator refuses it too, not only binding: `--no-bind` cannot slip past it."""
    _, paths = host(tmp_path)
    with pytest.raises(GenerateError, match="data.quality_tag is set but the run names compiled labels"):
        openpi_config(*_ft(overrides='"data.quality_tag" = 5'), paths)


def test_an_override_of_a_field_neither_emitted_nor_bind_consumed_is_refused(tmp_path):
    _, paths = host(tmp_path)
    shutil.copytree(RECIPES_DIR, tmp_path / "r")
    p = tmp_path / "r" / "pi05.finetune.lora.paper.toml"
    p.write_text(p.read_text().replace(
        '"data.image_center_crop" = true', '"data.image_center_crop" = true\n"data.mystery" = 1'))
    recipe = load_recipe("pi05.finetune.lora.paper", root=tmp_path / "r")
    spec, _ = _ft(overrides='"data.mystery" = 2')
    with pytest.raises(GenerateError, match="data.mystery"):
        openpi_config(spec, recipe, paths)


@pytest.mark.parametrize("missing", ["norm", "init"])
def test_a_pi05_arm_needs_its_inputs(tmp_path, missing):
    _, paths = host(tmp_path)
    with pytest.raises(GenerateError, match="needs"):
        openpi_config(*_ft(**{missing: None if missing == "init" else False}), paths)


def test_a_finetune_without_a_bundle_renders_only_once_bound(tmp_path):
    _, paths = host(tmp_path)
    spec, recipe = _ft(bundle=False)
    with pytest.raises(GenerateError, match="drop --no-bind"):
        openpi_config(spec, recipe, paths)
    bound = "sha256:" + "d" * 64
    c = openpi_config(spec, recipe, paths, bundle=bound)
    assert c["data"]["bundle_id"] == bound and c["identity"]["bundle"] == bound
    assert c["data"]["bundle"] == str(default_dest(CORPUS, paths, "bundles/" + "d" * 64))
    with pytest.raises(GenerateError, match="but the run names"):
        openpi_config(*_ft(), paths, bundle=bound)


def test_legacy_environment_variables_change_nothing(tmp_path, monkeypatch):
    _, paths = host(tmp_path)
    spec, recipe = _ft()
    before = render(spec, recipe, paths)
    for k in LEGACY_ENV:
        monkeypatch.setenv(k, "/elsewhere/x.json")
    assert render(spec, recipe, paths) == before


def _lerobot(backbone="xvla-lerobot", stage="pretrain", init="base:xvla_base", **kw):
    preset = f"{backbone.split('-')[0]}.{stage}.paper"
    s = loads_armspec(arm_toml(id=f"{backbone.split('-')[0]}.{stage}.example", backbone=backbone, stage=stage,
                               preset=preset, rowspace="libero5k_v2_d8", init=init, norm=False, seed=1, **kw))
    return s, load_recipe(preset)


def test_an_xvla_pretrain_renders_its_template(tmp_path):
    _, paths = host(tmp_path)
    spec, recipe = _lerobot()
    c = lerobot_config(spec, recipe, paths, signal_files={"axis_rows_path": "/r/rows.npz"})
    assert (c["steps"], c["batch_size"], c["parallelism"]["dp_replicate"], c["seed"]) == (20621, 8, 8, 1)
    assert c["policy"]["pretrained_path"] == "lerobot/xvla-base" and c["policy"]["domain_id"] == 20
    assert c["axis_rows_path"] == "/r/rows.npz"
    assert c["dataset"]["root"] == str(paths.corpus / "qvla-v1-corpus" / "lerobot_merged")
    assert c["output_dir"] == str(paths.runs / spec.id / "train") and c["job_name"] == spec.id


def test_a_smolvla_pretrain_loads_its_vlm_itself(tmp_path):
    _, paths = host(tmp_path)
    c = lerobot_config(*_lerobot("smolvla-lerobot", init="base:smolvlm2_500m"), paths, signal_files={})
    assert c["policy"]["pretrained_path"] is None and c["policy"]["load_vlm_weights"] is True


def test_lerobot_overrides_build_null_tables(tmp_path):
    _, paths = host(tmp_path)
    over = '"sample_weighting.type" = "axis_flat"\n"sample_weighting.extra_params.expected_frames" = 1906469'
    c = lerobot_config(*_lerobot(overrides=over), paths,
                       signal_files={"sample_weighting.extra_params.weights_path": "/w.npz"})
    assert c["sample_weighting"] == {"type": "axis_flat",
                                     "extra_params": {"expected_frames": 1906469, "weights_path": "/w.npz"}}


def test_lerobot_refusals(tmp_path):
    _, paths = host(tmp_path)
    with pytest.raises(GenerateError, match="not a field of the LeRobot train config"):
        lerobot_config(*_lerobot(overrides="no_such = 1"), paths, signal_files={})
    with pytest.raises(GenerateError, match="per-rank batch 16 x world 8"):
        lerobot_config(*_lerobot(overrides="batch_size = 16"), paths, signal_files={})
    with pytest.raises(GenerateError, match="not LeRobot label fields"):
        lerobot_config(*_lerobot(), paths, signal_files={"axis_rows": "/x"})


def test_pure_conditioning_is_declared_by_an_override_and_reaches_the_config(tmp_path):
    _, paths = host(tmp_path)
    c = lerobot_config(*_lerobot(overrides="axis_quality_unconditional = false"), paths, signal_files={})
    assert c["axis_quality_unconditional"] is False
    assert "axis_quality_unconditional" not in lerobot_config(*_lerobot(), paths, signal_files={})


def test_a_line_field_override_must_have_the_fields_type(tmp_path):
    _, paths = host(tmp_path)
    with pytest.raises(GenerateError, match="'axis_quality_unconditional' is a bool"):
        lerobot_config(*_lerobot(overrides='axis_quality_unconditional = "false"'), paths, signal_files={})


def test_a_lerobot_run_below_100_steps_is_refused(tmp_path):
    _, paths = host(tmp_path)
    with pytest.raises(GenerateError, match="100-step floor"):
        lerobot_config(*_lerobot(overrides="steps = 50"), paths, signal_files={})


def test_render_writes_the_backbone_config_and_its_command(tmp_path):
    _, paths = host(tmp_path)
    spec, recipe = _ft()
    files, argv = render(spec, recipe, paths)
    assert sorted(files) == ["openpi_config.json", "run.json", "run_spec.toml"]
    bw = paths.cache / "checkpoints" / "qvla-v1-base-weights"
    assert argv[:5] == ["env", f"OPENPI_DATA_HOME={bw / 'openpi'}", "python", "scripts/train.py", "--config-json"]
    assert json.loads(files["run.json"])["argv"] == argv
    files, argv = render(*_lerobot(), paths, signal_files={"axis_rows_path": "/r.npz"})
    assert "train_config.json" in files
    assert argv == ["env", f"HF_HOME={bw / 'hf'}", "HF_HUB_OFFLINE=1",
                    *lerobot_argv(paths.runs / "xvla.pretrain.example" / "train_config.json", 8)]


def _openpi_checkpoint_dir(cfg: dict) -> pathlib.Path:
    """Where a backbone's own openpi convention would write this config's checkpoints
    (`checkpoint_base_dir/name/exp_name`), read from the rendered config itself rather than
    standing in a LeRobot-named helper for a pi0.5 path."""
    return pathlib.Path(cfg["checkpoint_base_dir"]) / cfg["name"] / cfg["exp_name"]


def test_a_smoke_render_keys_all_its_files_under_rendered_smoke(tmp_path):
    """A smoke render must never collide with a real run's recorded files, and must never write
    into the smoke's own output/checkpoint directory: LeRobot refuses an existing `output_dir`
    (`ids.lerobot_output_dir`), and openpi's smoke mode deletes and recreates its checkpoint
    directory, read from the rendered config's own `checkpoint_base_dir`/`name`/`exp_name`
    (`_openpi_checkpoint_dir`), on every launch -- either way a config sitting there would be
    refused or destroyed before training starts. All three files, including run_spec.toml this
    time, key under the sibling `rendered-smoke/` -- the two modes share nothing."""
    _, paths = host(tmp_path)
    spec, recipe = _ft()
    run = paths.runs / spec.id
    files, argv = render(spec, recipe, paths, smoke=True)
    assert sorted(files) == ["rendered-smoke/openpi_config.json", "rendered-smoke/run.json",
                             "rendered-smoke/run_spec.toml"]
    assert argv[-1] == str(run / "rendered-smoke" / "openpi_config.json")
    assert json.loads(files["rendered-smoke/run.json"])["argv"] == argv
    cfg = json.loads(files["rendered-smoke/openpi_config.json"])
    assert not pathlib.Path(argv[-1]).is_relative_to(_openpi_checkpoint_dir(cfg))

    lr_files, lr_argv = render(*_lerobot(), paths, smoke=True, signal_files={"axis_rows_path": "/r.npz"})
    lr_run = paths.runs / "xvla.pretrain.example"
    assert sorted(lr_files) == ["rendered-smoke/run.json", "rendered-smoke/run_spec.toml",
                                "rendered-smoke/train_config.json"]
    lr_cfg_path = lr_run / "rendered-smoke" / "train_config.json"
    assert f"--config_path={lr_cfg_path}" in lr_argv
    assert not lr_cfg_path.is_relative_to(ids.lerobot_output_dir(paths, "xvla.pretrain.example", smoke=True))


def test_a_real_render_still_keys_its_files_at_the_run_root(tmp_path):
    """The real run's files sit directly at `<runs>/<arm id>/`, not inside `<runs>/<arm id>/train/`
    (LeRobot's real `output_dir`, which no earlier render call has reason to collide with since
    training runs once for real)."""
    _, paths = host(tmp_path)
    spec, recipe = _ft()
    run = paths.runs / spec.id
    files, _ = render(spec, recipe, paths)
    assert sorted(files) == ["openpi_config.json", "run.json", "run_spec.toml"]
    cfg = json.loads(files["openpi_config.json"])
    assert not (run / "openpi_config.json").is_relative_to(_openpi_checkpoint_dir(cfg))

    lr_files, lr_argv = render(*_lerobot(), paths, signal_files={"axis_rows_path": "/r.npz"})
    assert sorted(lr_files) == ["run.json", "run_spec.toml", "train_config.json"]
    lr_run = paths.runs / "xvla.pretrain.example"
    assert f"--config_path={lr_run / 'train_config.json'}" in lr_argv
    assert not (lr_run / "train_config.json").is_relative_to(
        ids.lerobot_output_dir(paths, "xvla.pretrain.example", smoke=False))


def test_a_serve_config_needs_no_bundle(tmp_path):
    """What the benchmark's reference server renders for a published pi0.5 checkpoint."""
    _, paths = host(tmp_path)
    c = openpi_config(*_ft(bundle=False), paths, serve=True)
    assert (c["exp_name"], c["data"]["bundle"], c["data"]["bundle_id"], c["data"]["smoke"]) == ("serve", None, None, False)
    assert c["model"]["paligemma_variant"] == "gemma_2b_lora"
    with pytest.raises(GenerateError, match="not a smoke run"):
        openpi_config(*_ft(), paths, serve=True, smoke=True)


# sha256 over the 59 pi0.5 paper arms' serve configs (canonical bytes, arm-file order, the host's
# root spelled <root>): what `bench eval --checkpoint-id` serves each published checkpoint with.
PAPER_SERVE_SHA256 = "214be4c052ece6f85e71692bac991eceb71373969d83fa8f32a27d0ee1aef51f"


def test_a_serve_config_trains_nothing_so_a_smoke_checkpoint_is_served_too(tmp_path):
    """The real-run step floor applies to training only: a smoke arm (20 steps) renders its serve
    config, and every paper arm's serve bytes are the ones it rendered before serving skipped it."""
    import hashlib

    from axis_qvla.core.canon import dumps_canonical
    from axis_qvla.train import smoke
    from axis_qvla.train.armspec import load_armspec
    from axis_qvla.train.paper import paper_arm_files

    _, paths = host(tmp_path)
    rs = f"{smoke.SMOKE_ROWSPACE}#sha256:{H}"
    pi = smoke.fixture_arms(rs, {"pi05": f"sha256:{H}", "lerobot": f"sha256:{H}"}, rs, 609)["pi05"]
    with pytest.raises(GenerateError, match="below the 100-step floor"):
        openpi_config(pi, load_recipe(pi.preset), paths)
    c = openpi_config(pi, load_recipe(pi.preset), paths, serve=True, fsdp_devices=1)
    assert (c["exp_name"], c["num_train_steps"], c["batch_size"], c["data"]["bundle"], c["data"]["smoke"]) == (
        "serve", 20, 8, None, False)
    h, n = hashlib.sha256(), 0
    for f in paper_arm_files():
        spec = load_armspec(f)
        if spec.backbone == "pi05-openpi":
            cfg = openpi_config(spec, load_recipe(spec.preset), paths, serve=True)
            h.update(dumps_canonical(cfg).replace(str(tmp_path).encode(), b"<root>"))
            n += 1
    assert (n, h.hexdigest()) == (59, PAPER_SERVE_SHA256)


def test_a_local_input_is_refused_and_a_real_world_run_is_not(tmp_path):
    from axis_qvla.train.generate import GenerateError, check_renderable
    from axis_qvla.train.armspec import load_armspec
    shipped = load_armspec(REPO / "experiments" / "realrobot" / "runs" / "pi05.pretrain.realrobot_cond.toml")
    with pytest.raises(GenerateError, match="fr3 local-runs"):
        check_renderable(shipped)
    check_renderable(dataclasses.replace(shipped, rowspace="realworld_pretrain#sha256:" + "a" * 64,
                                         bundle="sha256:" + "b" * 64, norm="norm_realalign_sim#sha256:" + "c" * 64))


def test_fsdp_devices_shards_the_state_and_changes_nothing_else(tmp_path):
    """A launch option: the preset's sharding (8 for pi0.5) unless given, and the rendered field the
    only difference. The config hash parity compares (axis_qvla.train.parity.project_openpi) is a
    function of the arm and its recipe alone, so no render option reaches it."""
    _, paths = host(tmp_path)
    spec, recipe = _ft()
    default, one = openpi_config(spec, recipe, paths), openpi_config(spec, recipe, paths, fsdp_devices=1)
    assert (default["fsdp_devices"], one["fsdp_devices"]) == (8, 1)
    assert {k for k in default if default[k] != one[k]} == {"fsdp_devices"}
    assert default["batch_size"] == one["batch_size"] == 32
    files, argv = render(spec, recipe, paths, fsdp_devices=2)
    assert json.loads(files["openpi_config.json"])["fsdp_devices"] == 2
    assert json.loads(files["run.json"])["fsdp_devices"] == 2
    assert json.loads(render(spec, recipe, paths)[0]["run.json"])["fsdp_devices"] == 8
    for bad, match in ((3, "does not divide the global batch 32"), (0, "positive integer"), (True, "positive integer")):
        with pytest.raises(GenerateError, match=match):
            openpi_config(spec, recipe, paths, fsdp_devices=bad)


def test_resume_is_a_command_option_and_the_config_is_unchanged(tmp_path):
    _, paths = host(tmp_path)
    spec, recipe = _ft()
    fresh, resumed = render(spec, recipe, paths), render(spec, recipe, paths, resume=True)
    assert fresh[0]["openpi_config.json"] == resumed[0]["openpi_config.json"]
    assert resumed[1] == [*fresh[1], "--resume"]
    assert json.loads(resumed[0]["run.json"])["resume"] is True and json.loads(fresh[0]["run.json"])["resume"] is False
    with pytest.raises(GenerateError, match="a smoke run starts over at every launch"):
        render(spec, recipe, paths, smoke=True, resume=True)
    for kw in ({"resume": True}, {"fsdp_devices": 1}):
        with pytest.raises(GenerateError, match="are pi0.5 launch options"):
            render(*_lerobot(), paths, signal_files={"axis_rows_path": "/r.npz"}, **kw)


OWNED = ["axis_rows_path", "axis_schedule_path", "axis_quality_path", "sample_weighting.extra_params.weights_path",
         "seed", "output_dir", "job_name", "dataset.root", "policy.pretrained_path"]


@pytest.mark.parametrize("key", [*OWNED, "dataset", "policy", "sample_weighting.extra_params"])
def test_an_override_of_a_field_qvla_train_writes_is_refused(tmp_path, key):
    """Each would be replaced without a word, or put a file binding never checked into the run."""
    _, paths = host(tmp_path)
    value = '{ x = 1 }' if key in ("dataset", "policy", "sample_weighting.extra_params") else (
        "9" if key == "seed" else '"my/own_rows.npy"')
    with pytest.raises(GenerateError, match=rf"the override '{key}' sets "):
        lerobot_config(*_lerobot(overrides=f'"{key}" = {value}'), paths, signal_files={})


def test_exported_weights_and_the_axis_flat_weighter_come_together(tmp_path):
    _, paths = host(tmp_path)
    weights = {"sample_weighting.extra_params.weights_path": "/w.npz"}
    with pytest.raises(GenerateError, match="applies only with sample_weighting.type = 'axis_flat'"):
        lerobot_config(*_lerobot(), paths, signal_files=weights)
    with pytest.raises(GenerateError, match="applies only with sample_weighting.type = 'axis_flat' .* 'rabc'"):
        lerobot_config(*_lerobot(overrides='"sample_weighting.type" = "rabc"'), paths, signal_files=weights)
    for over in ('"sample_weighting.type" = "axis_flat"',
                 '"sample_weighting.type" = "rabc"\n"sample_weighting.progress_path" = "/anywhere/progress.parquet"',
                 '"sample_weighting.type" = "uniform"'):
        with pytest.raises(GenerateError, match=r"sets sample_weighting \(type '\w+'\) but the compiled labels export no"):
            lerobot_config(*_lerobot(overrides=over), paths, signal_files={"axis_rows_path": "/r.npz"})
    assert lerobot_config(*_lerobot(), paths, signal_files={"axis_rows_path": "/r.npz"})["sample_weighting"] is None
    c = lerobot_config(*_lerobot(overrides='"sample_weighting.type" = "axis_flat"'), paths, signal_files=weights)
    assert c["sample_weighting"] == {"type": "axis_flat", "extra_params": {"weights_path": "/w.npz"}}


def test_the_parity_check_reports_an_owned_override_of_a_paper_arm():
    import dataclasses
    import types

    from axis_qvla.train.armspec import load_armspec
    from axis_qvla.train.paper import RUNS_DIR
    from axis_qvla.train.parity import check_arm, load_lock, load_signals_lock

    index = json.loads((RUNS_DIR / "index.json").read_text())
    arm = next(a for a, row in sorted(index["arms"].items()) if a.startswith("smolvla.finetune."))
    spec = load_armspec(RUNS_DIR / index["arms"][arm]["file"])
    lock, signals = load_lock(), load_signals_lock()
    assert check_arm(spec, lock, signals) == []
    bad = dataclasses.replace(spec, overrides=types.MappingProxyType({**spec.overrides, "output_dir": "elsewhere"}))
    problems = check_arm(bad, lock, signals)
    assert len(problems) == 1 and "the override 'output_dir' sets output_dir" in problems[0]
