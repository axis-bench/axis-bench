"""`qvla train` writes the run directory and prints the command; it refuses rather than guesses."""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

import axis_qvla.train.cli as traincli
from axis_qvla.cli import main
from axis_qvla.data.layout import CORPUS, default_dest
from axis_qvla.train import ids
from axis_qvla.train.armspec import loads_armspec
from axis_qvla.train.generate import GenerateError, render
from axis_qvla.train.recipes import load_recipe
from tests.unit.train._arms import H, arm_toml, host
from tests.unit.train._bundles import (finetune_corpus, finetune_digests, finetune_inputs, ft_rowspace,
                                       lerobot_rows_legacy, rowspace_ref, write_fixture_bundle)


def test_qvla_train_renders_a_pi05_arm(tmp_path, capsys):
    hosts, paths = host(tmp_path)
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml())
    assert main(["train", "--render", str(arm), "--host", str(hosts), "--no-bind"]) == 0
    run = paths.runs / "pi05.finetune.example"
    assert sorted(p.name for p in run.iterdir()) == ["openpi_config.json", "run.json", "run_spec.toml"]
    out = capsys.readouterr().out
    assert "scripts/train.py --config-json" in out and str(run / "openpi_config.json") in out


def test_qvla_train_accepts_options_before_the_positional(tmp_path, capsys):
    """Options before the positional parse as options after it do (argparse's `nargs=REMAINDER`
    would refuse an option-looking token before the positional): every argument is registered on
    the subparser itself, like every other `_GROUPS` group, so order does not matter."""
    hosts, paths = host(tmp_path)
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml())
    assert main(["train", "--render", "--host", str(hosts), "--no-bind", str(arm)]) == 0
    run = paths.runs / "pi05.finetune.example"
    assert sorted(p.name for p in run.iterdir()) == ["openpi_config.json", "run.json", "run_spec.toml"]


def test_qvla_train_help():
    with pytest.raises(SystemExit) as exc:
        main(["train", "--help"])
    assert exc.value.code == 0


def test_qvla_train_env_plan_prints_the_build(capsys):
    assert main(["train", "env", "plan", "openpi"]) == 0
    out = capsys.readouterr().out
    assert "cd third_party/openpi" in out and "uv sync --frozen --python 3.11" in out


def test_qvla_train_smoke_fixture_reports_missing_episodes(tmp_path, capsys):
    hosts, paths = host(tmp_path)
    assert main(["train", "smoke-fixture", "--host", str(hosts), "--episodes", str(tmp_path / "absent")]) == 2
    out = capsys.readouterr().out
    assert out.startswith("FAIL  ") and "task_2013/meta/info.json" in out
    assert not paths.cache.exists()                                  # refused before anything is written


def test_qvla_train_refuses_an_unbound_lerobot_arm(tmp_path, capsys):
    hosts, _ = host(tmp_path)
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml(id="xvla.pretrain.example", backbone="xvla-lerobot", stage="pretrain",
                            preset="xvla.pretrain.paper", rowspace="libero5k_v2_d8", init="base:xvla_base",
                            norm=False))
    assert main(["train", "--render", str(arm), "--host", str(hosts), "--no-bind"]) == 2
    assert "must be bound" in capsys.readouterr().out


def test_qvla_train_reports_a_bad_arm(tmp_path, capsys):
    hosts, _ = host(tmp_path)
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml().replace('seed = 42', 'seed = 42\nsteps = 1'))
    assert main(["train", "--render", str(arm), "--host", str(hosts), "--no-bind"]) == 2
    assert "FAIL" in capsys.readouterr().out


def _published_bundle(paths, **kw):
    return write_fixture_bundle(default_dest(CORPUS, paths, "bundles"), **kw)


def test_qvla_train_binds_by_default(tmp_path, capsys):
    hosts, paths = host(tmp_path)
    norm_sha = finetune_inputs(paths)
    rs = ft_rowspace()
    bid = _published_bundle(paths, plan=(np.arange(13580 * 32) % 100).reshape(13580, 32), rowspace=rs,
                            inputs=finetune_digests(paths))
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml(bundle=bid, rowspace=rowspace_ref(rs)).replace(
        f"heldout24_qual_v2#sha256:{H}", f"heldout24_qual_v2#sha256:{norm_sha}"))
    assert main(["train", "--render", str(arm), "--host", str(hosts)]) == 0
    run = paths.runs / "pi05.finetune.example"
    assert sorted(p.name for p in run.iterdir()) == ["binding.json", "openpi_config.json", "run.json", "run_spec.toml"]
    cfg = json.loads((run / "openpi_config.json").read_text())
    binding = json.loads((run / "binding.json").read_text())
    assert (cfg["data"]["bundle"], cfg["data"]["bundle_id"]) == (binding["bundle_dir"], bid)


def test_qvla_train_renders_a_finetune_whose_signal_binding_compiles(tmp_path, capsys):
    hosts, paths = host(tmp_path)
    norm_sha = finetune_inputs(paths)
    rs = finetune_corpus(paths)
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml(bundle=False, track="small", preset="pi05.finetune.lora.small", rowspace=rowspace_ref(rs),
                            overrides='"data.quality_tag" = 5').replace(
        f"heldout24_qual_v2#sha256:{H}", f"heldout24_qual_v2#sha256:{norm_sha}"))
    assert main(["train", "--render", str(arm), "--host", str(hosts)]) == 0
    run = paths.runs / "pi05.finetune.example"
    cfg = json.loads((run / "openpi_config.json").read_text())
    binding = json.loads((run / "binding.json").read_text())
    assert cfg["data"]["bundle"] == binding["bundle_dir"] == str(ids.bundle_dirs(paths, binding["bundle"])[1])
    assert cfg["data"]["bundle_id"] == cfg["identity"]["bundle"] == binding["bundle"]
    assert (pathlib.Path(binding["bundle_dir"]) / "cond.npy").is_file()
    assert main(["train", "--render", str(arm), "--host", str(hosts), "--no-bind"]) == 2       # nothing to render without it
    assert "drop --no-bind" in capsys.readouterr().out


def test_qvla_train_refuses_a_quality_tag_beside_a_named_bundle_even_unbound(tmp_path, capsys):
    hosts, _ = host(tmp_path)
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml(overrides='"data.quality_tag" = 5'))
    assert main(["train", "--render", str(arm), "--host", str(hosts), "--no-bind"]) == 2
    assert "data.quality_tag is set but the run names compiled labels" in capsys.readouterr().out


def test_qvla_train_lists_every_refusal_and_writes_nothing(tmp_path, capsys):
    hosts, paths = host(tmp_path)
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml())
    assert main(["train", "--render", str(arm), "--host", str(hosts)]) == 2
    out = capsys.readouterr().out
    for part in ("FAIL", "refused before launch", "bundle: cannot load", "init:", "norm:", "rowspace:"):
        assert part in out
    assert not (paths.runs / "pi05.finetune.example").exists()


def test_qvla_train_exports_a_lerobot_arms_signal_files(tmp_path, capsys):
    hosts, paths = host(tmp_path)
    default_dest(CORPUS, paths, "lerobot_merged").mkdir(parents=True)
    rs = ft_rowspace(name="libero5k_v2_d8")
    bid = _published_bundle(paths, plan=(np.arange(20 * 64) % 100).reshape(20, 64), rowspace=rs,
                            **lerobot_rows_legacy(np.arange(100)))
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml(id="xvla.pretrain.example", backbone="xvla-lerobot", stage="pretrain", track="small",
                            preset="xvla.pretrain.small", rowspace=rowspace_ref(rs), init="base:xvla_base",
                            norm=False, bundle=bid, seed=0))
    assert main(["train", "--render", str(arm), "--host", str(hosts), "--smoke"]) == 0
    smoke = paths.runs / "xvla.pretrain.example" / "rendered-smoke"
    cfg = json.loads((smoke / "train_config.json").read_text())
    assert cfg["axis_rows_path"] == str(smoke / "signals/test" / f"{bid[7:19]}-lerobot_rows.npz")
    assert (smoke / "binding.json").is_file() and cfg["steps"] == 20
    capsys.readouterr()
    ids.lerobot_output_dir(paths, "xvla.pretrain.example", smoke=True).mkdir(parents=True)   # it launched
    assert main(["train", "--render", str(arm), "--host", str(hosts), "--smoke"]) == 2
    assert "already exists" in capsys.readouterr().out


def test_a_smoke_render_never_touches_the_real_runs_files(tmp_path, capsys):
    hosts, paths = host(tmp_path)
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml())
    assert main(["train", "--render", str(arm), "--host", str(hosts), "--no-bind"]) == 0
    run = paths.runs / "pi05.finetune.example"
    before = {p.name: p.read_bytes() for p in run.iterdir() if p.is_file()}
    capsys.readouterr()
    assert main(["train", "--render", str(arm), "--host", str(hosts), "--no-bind", "--smoke"]) == 0
    capsys.readouterr()
    after = {p.name: p.read_bytes() for p in run.iterdir() if p.is_file()}
    assert after == before
    assert sorted(p.name for p in (run / "rendered-smoke").iterdir()) == [
        "openpi_config.json", "run.json", "run_spec.toml"]
    # A smoke render must not itself create the smoke output/checkpoint directory
    # (ids.lerobot_output_dir): that directory existing is what `_check_output_dir` (below) takes
    # as "a previous run already launched here".
    assert not ids.lerobot_output_dir(paths, "pi05.finetune.example", smoke=True).exists()


def test_a_second_lerobot_render_is_refused_before_a_third_write(tmp_path):
    """Upstream LeRobot refuses to train into an existing output_dir when resume is false; that
    would otherwise surface only at launch, after GPU allocation. This calls `_check_output_dir`
    directly, per mode; test_qvla_train_exports_a_lerobot_arms_signal_files runs the same refusal
    through `qvla train`."""
    _, paths = host(tmp_path)
    spec = loads_armspec(arm_toml(id="xvla.pretrain.example", backbone="xvla-lerobot", stage="pretrain",
                                  preset="xvla.pretrain.paper", rowspace="libero5k_v2_d8",
                                  init="base:xvla_base", norm=False))
    traincli._check_output_dir(spec, paths, smoke=True)   # nothing exists yet: fine
    out_dir = ids.lerobot_output_dir(paths, spec.id, smoke=True)
    out_dir.mkdir(parents=True)
    with pytest.raises(GenerateError, match="already exists"):
        traincli._check_output_dir(spec, paths, smoke=True)
    assert out_dir.is_dir()   # never deleted
    # The real (non-smoke) output_dir is a different, still-untouched directory.
    traincli._check_output_dir(spec, paths, smoke=False)


def test_rendering_a_lerobot_smoke_does_not_itself_trip_the_output_dir_guard(tmp_path):
    """Actually writing a LeRobot smoke's rendered files must not trip `_check_output_dir` on the
    very next call. Only a real launch (which creates the checkpoint directory `qvla train` itself
    never touches) does."""
    _, paths = host(tmp_path)
    spec = loads_armspec(arm_toml(id="xvla.pretrain.example", backbone="xvla-lerobot", stage="pretrain",
                                  preset="xvla.pretrain.paper", rowspace="libero5k_v2_d8",
                                  init="base:xvla_base", norm=False))
    recipe = load_recipe(spec.preset)
    traincli._check_output_dir(spec, paths, smoke=True)   # nothing exists yet
    files, _ = render(spec, recipe, paths, smoke=True, signal_files={})
    run = paths.runs / spec.id
    for name, data in files.items():
        dest = run / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    assert not ids.lerobot_output_dir(paths, spec.id, smoke=True).exists()
    traincli._check_output_dir(spec, paths, smoke=True)   # still fine: rendering wrote
    out_dir = ids.lerobot_output_dir(paths, spec.id, smoke=True)         # rendered-smoke/, not here
    out_dir.mkdir(parents=True)   # simulate the backbone having actually launched
    with pytest.raises(GenerateError, match="already exists"):
        traincli._check_output_dir(spec, paths, smoke=True)


def test_a_launched_pi05_run_is_refused_a_fresh_render_and_continues_with_resume(tmp_path, capsys):
    hosts, paths = host(tmp_path)
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml())
    train = ["train", "--render", str(arm), "--host", str(hosts), "--no-bind"]
    assert main(["train", "--render", str(arm), "--host", str(hosts), "--no-bind", "--resume"]) == 2   # nothing to resume
    assert "which is not there; render without --resume" in capsys.readouterr().out
    assert main(train) == 0
    run = paths.runs / "pi05.finetune.example"
    config = (run / "openpi_config.json").read_bytes()
    capsys.readouterr()
    (run / "train").mkdir()                     # a launch that failed inside its first step left this
    assert main(train) == 2
    out = capsys.readouterr().out
    assert f"{run / 'train'} already exists and holds no saved step" in out
    assert f"remove {run / 'train'} yourself" in out and "Pass --resume" not in out
    (run / "train" / "2").mkdir()                                 # the run saved step 2, then stopped
    assert main(train) == 2
    out = capsys.readouterr().out
    assert f"{run / 'train'} already exists. The pinned openpi fork refuses" in out and "Pass --resume" in out
    assert main([*train, "--resume"]) == 0
    command = capsys.readouterr().out.splitlines()[-1]
    assert command.endswith(f"--config-json {run / 'openpi_config.json'} --resume")
    assert (run / "openpi_config.json").read_bytes() == config    # the config the run started with
    record = json.loads((run / "run.json").read_text())
    assert record["resume"] is True and record["argv"][-1] == "--resume"
    assert (run / "train").is_dir()                               # never deleted
    # A resume renders exactly the recorded config: another sharding, or another arm, is refused.
    assert main([*train, "--resume", "--fsdp-devices", "4"]) == 2
    out = capsys.readouterr().out
    assert "with another config than it recorded (differs in ['fsdp_devices'])" in out
    assert f"remove {run / 'train'} yourself and render without --resume" in out
    arm.write_text(arm_toml(seed=43))
    assert main([*train, "--resume"]) == 2
    assert "differs in ['identity', 'seed']" in capsys.readouterr().out
    assert (run / "openpi_config.json").read_bytes() == config


def test_a_resume_binds_without_rewriting_the_runs_binding(tmp_path, capsys):
    hosts, paths = host(tmp_path)
    norm_sha = finetune_inputs(paths)
    rs = ft_rowspace()
    bid = _published_bundle(paths, plan=(np.arange(13580 * 32) % 100).reshape(13580, 32), rowspace=rs,
                            inputs=finetune_digests(paths))
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml(bundle=bid, rowspace=rowspace_ref(rs)).replace(
        f"heldout24_qual_v2#sha256:{H}", f"heldout24_qual_v2#sha256:{norm_sha}"))
    assert main(["train", "--render", str(arm), "--host", str(hosts)]) == 0
    run = paths.runs / "pi05.finetune.example"
    binding = run / "binding.json"
    binding.write_bytes(binding.read_bytes() + b" ")             # marks the first launch's record
    marked = binding.read_bytes()
    (run / "train" / "100").mkdir(parents=True)
    assert main(["train", "--render", str(arm), "--host", str(hosts), "--resume"]) == 0
    assert binding.read_bytes() == marked                         # checked again, not rewritten
    capsys.readouterr()


def test_the_launch_options_are_pi05_options(tmp_path, capsys):
    hosts, _ = host(tmp_path)
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml(id="xvla.pretrain.example", backbone="xvla-lerobot", stage="pretrain",
                            preset="xvla.pretrain.paper", rowspace="libero5k_v2_d8", init="base:xvla_base",
                            norm=False))
    for option in (["--resume"], ["--fsdp-devices", "1"]):
        assert main(["train", "--render", str(arm), "--host", str(hosts), *option]) == 2
        assert "are pi0.5 launch options" in capsys.readouterr().out
    pi05 = tmp_path / "pi05.toml"
    pi05.write_text(arm_toml())
    assert main(["train", "--render", str(pi05), "--host", str(hosts), "--no-bind", "--smoke", "--resume"]) == 2
    assert "a smoke run starts over at every launch" in capsys.readouterr().out
    assert main(["train", "--render", str(pi05), "--host", str(hosts), "--no-bind", "--fsdp-devices", "3"]) == 2
    assert "does not divide the global batch 32" in capsys.readouterr().out


def test_a_real_robot_declaration_is_refused_before_its_launch_options_or_run_directory(tmp_path, capsys):
    """A declaration is refused as one, whatever else is wrong: --resume or a leftover train/ must not
    answer first with a refusal that tells the user the arm can be started."""
    hosts, paths = host(tmp_path)
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml(id="realrobot.finetune.bc", track="realrobot", preset="pi05.finetune.lora.realrobot",
                            init="run:realrobot.pretrain.bc", rowspace="local:realalign_droid8d",
                            bundle="local:realalign.bc", norm=False))
    train = ["train", "--render", str(arm), "--host", str(hosts), "--no-bind"]
    assert main([*train, "--resume"]) == 2
    out = capsys.readouterr().out
    assert "by a local: name" in out and "render without --resume" not in out
    (paths.runs / "realrobot.finetune.bc" / "train").mkdir(parents=True)
    assert main(train) == 2
    out = capsys.readouterr().out
    assert "by a local: name" in out and "already exists" not in out
