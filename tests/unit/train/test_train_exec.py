"""`qvla train <spec>` trains in the backbone's environment: the rendered command runs with that
environment's interpreter from the backbone's checkout, its exit code is the command's, and a LeRobot
run's merged copy is built first. `--print` reads nothing but the spec (no data, GPU or network) and
prints the one command that trains it; `--render` binds and renders without training."""
from __future__ import annotations

import json

import pytest

from axis_qvla.core.canon import sha256_file
from axis_qvla.cli import main
from axis_qvla.train import ids, launch, runs
from axis_qvla.train.armspec import dumps_armspec, loads_armspec
from axis_qvla.train.envs import REPO
from axis_qvla.train.names import PUBLISHED_ROWSPACES
from tests.unit.train._arms import H, arm_toml, host

LEROBOT_FT = dict(id="smolvla.finetune.example", backbone="smolvla-lerobot", preset="smolvla.finetune.paper",
                  init=f"axisrobotics/qvla-v1-smolvla:pretrain/example/020621#sha256:{H}", norm=False)
EXAMPLE = REPO / "experiments" / "examples" / "runs" / "pi05.pretrain.keep_top.toml"


@pytest.fixture
def machine(tmp_path):
    """A qvla.toml whose pi0.5 and LeRobot environments are built (an interpreter file each)."""
    hosts, paths = host(tmp_path)
    venvs = tmp_path / "venvs"
    for key in ("openpi", "lerobot"):
        (venvs / key / "bin").mkdir(parents=True)
        (venvs / key / "bin" / "python").write_text("")
    text = hosts.read_text().replace("[roots]\n", f'[roots]\nsmolvla_venv = "{venvs / "lerobot"}"\n')
    hosts.write_text(text + f'\n[venv]\nonline = "{venvs / "openpi"}"\n')
    return hosts, paths, venvs


@pytest.fixture
def ran(monkeypatch):
    out = []
    monkeypatch.setattr(launch, "RUNNER", lambda l: out.append(l) or 0)
    return out


def test_a_pi05_run_trains_with_the_environments_python_from_its_checkout(machine, ran, tmp_path, capsys):
    hosts, paths, venvs = machine
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml())
    assert main(["train", str(arm), "--host", str(hosts), "--no-bind"]) == 0
    (run,) = ran
    # argv[1] is openpi's training script, relative to the checkout it runs from
    assert run.argv[0] == str(venvs / "openpi" / "bin" / "python") and run.argv[1].endswith("/train.py")
    assert run.argv[2:] == ("--config-json", str(paths.runs / "pi05.finetune.example" / "openpi_config.json"))
    assert run.cwd == REPO / "third_party" / "openpi"
    assert dict(run.env)["OPENPI_DATA_HOME"] == str(ids.openpi_home(paths))
    assert "RUN   cd " in capsys.readouterr().out


def test_the_training_exit_code_is_passed_through(machine, monkeypatch, tmp_path):
    hosts, _, _ = machine
    monkeypatch.setattr(launch, "RUNNER", lambda l: 5)
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml())
    assert main(["train", str(arm), "--host", str(hosts), "--no-bind"]) == 5


def test_print_needs_no_data_and_prints_the_one_command_that_trains(tmp_path, monkeypatch, capsys):
    """The example spec names labels nobody compiled here, and no environment is built: --print still
    plans it, writes nothing and runs nothing."""
    hosts, paths = host(tmp_path)
    monkeypatch.setattr(launch, "RUNNER", lambda l: pytest.fail("--print ran a command"))
    assert main(["train", str(EXAMPLE), "--host", str(hosts), "--print", "--fsdp-devices", "4"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == f"OK    pi05.pretrain.keep_top: {EXAMPLE}"
    assert out[1] == f"qvla train {EXAMPLE} --host {hosts} --fsdp-devices 4"
    assert not paths.runs.exists()


def test_print_refuses_what_would_be_refused_anyway(tmp_path, capsys):
    hosts, _ = host(tmp_path)
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml(id="xvla.pretrain.x", backbone="xvla-lerobot", stage="pretrain", preset="xvla.pretrain.paper",
                            rowspace="libero5k_v2_d8", init="base:xvla_base", norm=False))
    assert main(["train", str(arm), "--host", str(hosts), "--print", "--resume"]) == 2
    assert "pi0.5 launch options" in capsys.readouterr().out


def test_render_trains_nothing_and_prints_the_backbone_command(machine, monkeypatch, tmp_path, capsys):
    hosts, paths, venvs = machine
    monkeypatch.setattr(launch, "RUNNER", lambda l: pytest.fail("--render ran a command"))
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml())
    assert main(["train", str(arm), "--host", str(hosts), "--no-bind", "--render"]) == 0
    line = capsys.readouterr().out.splitlines()[-1]
    assert line.startswith(f"cd {REPO / 'third_party' / 'openpi'} && env OPENPI_DATA_HOME=")
    script, flag = line.split(f"{venvs / 'openpi' / 'bin' / 'python'} ", 1)[1].split()[:2]
    assert script.endswith("/train.py") and flag == "--config-json"
    assert json.loads((paths.runs / "pi05.finetune.example" / "openpi_config.json").read_text())["fsdp_devices"] == 8


def test_an_unbuilt_environment_is_refused_with_the_command_that_builds_it(tmp_path, ran, capsys):
    hosts, paths = host(tmp_path)
    hosts.write_text(hosts.read_text() + f'\n[venv]\nonline = "{tmp_path / "absent"}"\n')
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml())
    assert main(["train", str(arm), "--host", str(hosts), "--no-bind"]) == 2
    assert "run `qvla env build pi05`" in capsys.readouterr().out
    assert ran == [] and not (paths.runs / "pi05.finetune.example").exists()


def test_fsdp_defaults_to_the_visible_gpus_when_training_here(machine, ran, monkeypatch, tmp_path):
    hosts, paths, _ = machine
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml())
    assert main(["train", str(arm), "--host", str(hosts), "--no-bind"]) == 0
    cfg = json.loads((paths.runs / "pi05.finetune.example" / "openpi_config.json").read_text())
    assert cfg["fsdp_devices"] == 2                                   # the preset's is 8
    other = tmp_path / "other.toml"
    other.write_text(arm_toml(id="pi05.finetune.other"))
    assert main(["train", str(other), "--host", str(hosts), "--no-bind", "--render"]) == 0
    # --render renders for the machine its printed command runs on: the preset's sharding
    assert json.loads((paths.runs / "pi05.finetune.other" / "openpi_config.json").read_text())["fsdp_devices"] == 8


def test_the_merged_finetune_copy_is_cut_before_binding(machine, ran, tmp_path, capsys):
    hosts, paths, venvs = machine
    spec = loads_armspec(arm_toml(**LEROBOT_FT))
    rs = ids.rowspace_dirs(paths, spec.rowspace)
    rs["roots"].parent.mkdir(parents=True)
    rs["roots"].write_text(json.dumps({"10": "ft_demos/task_10", "2": "ft_demos/task_2"}))
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml(**LEROBOT_FT))
    assert main(["train", str(arm), "--host", str(hosts)]) == 2       # the fake merge built nothing
    (merge,) = ran
    assert merge.argv == (str(venvs / "lerobot" / "bin" / "python"), "-m", "lerobot.scripts.axis_merge_datasets",
                          "--out", str(rs["lerobot_dataset"]), "--ranges", str(rs["ranges"]),
                          "--action-source", "action", str(rs["corpus_base"] / "ft_demos/task_2"),
                          str(rs["corpus_base"] / "ft_demos/task_10"))
    assert merge.cwd == REPO / "third_party" / "lerobot"
    assert "is not there" in capsys.readouterr().out


def test_no_merge_once_the_copy_is_there_or_for_pi05(tmp_path):
    _, paths = host(tmp_path)
    spec = loads_armspec(arm_toml(**LEROBOT_FT))
    ids.rowspace_dirs(paths, spec.rowspace)["lerobot_dataset"].mkdir(parents=True)
    assert launch.merge_launch(spec, paths, python=tmp_path / "py", checkout=tmp_path) is None
    assert launch.merge_launch(loads_armspec(arm_toml()), paths, python=tmp_path / "py", checkout=tmp_path) is None


def test_a_run_naming_an_input_by_name_records_what_it_resolved_to(machine, ran, tmp_path):
    hosts, paths, _ = machine
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml().replace(f'rowspace = "heldout24_ft#sha256:{H}"', 'rowspace = "heldout24_ft"'))
    assert main(["train", str(arm), "--host", str(hosts), "--no-bind", "--render"]) == 0
    run = paths.runs / "pi05.finetune.example"
    assert json.loads((run / "names.json").read_text()) == {
        "rowspace": {"name": "heldout24_ft", "id": f"heldout24_ft#sha256:{PUBLISHED_ROWSPACES['heldout24_ft']}"}}
    assert json.loads((run / "openpi_config.json").read_text()) != {}
    other = tmp_path / "other.toml"
    other.write_text(arm_toml(id="pi05.finetune.other"))
    assert main(["train", str(other), "--host", str(hosts), "--no-bind", "--render"]) == 0
    assert not (paths.runs / "pi05.finetune.other" / "names.json").exists()


def test_an_unknown_label_name_is_refused_before_anything_is_written(machine, ran, tmp_path, capsys):
    hosts, paths, _ = machine
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml(bundle="name:nobody.compiled.this"))
    assert main(["train", str(arm), "--host", str(hosts)]) == 2
    assert "no compiled labels named 'nobody.compiled.this'" in capsys.readouterr().out
    assert ran == [] and not paths.runs.exists()


def test_a_lerobot_run_without_binding_is_refused_before_its_merge_and_by_print(machine, ran, tmp_path, capsys):
    hosts, paths, _ = machine
    spec = loads_armspec(arm_toml(**LEROBOT_FT))
    rs = ids.rowspace_dirs(paths, spec.rowspace)
    rs["roots"].parent.mkdir(parents=True)
    rs["roots"].write_text(json.dumps({"2": "ft_demos/task_2"}))
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml(**LEROBOT_FT))
    for how in ([], ["--print"]):
        assert main(["train", str(arm), "--host", str(hosts), "--no-bind", *how]) == 2
        assert "a smolvla-lerobot run must be bound" in capsys.readouterr().out
    assert ran == []                                                  # no merge ran


def test_an_empty_cuda_visible_devices_is_refused_naming_it(machine, ran, monkeypatch, tmp_path, capsys):
    hosts, paths, _ = machine
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml())
    assert main(["train", str(arm), "--host", str(hosts), "--no-bind"]) == 2
    assert "FAIL  CUDA_VISIBLE_DEVICES is set to '', which names no GPU" in capsys.readouterr().out
    assert ran == [] and not (paths.runs / "pi05.finetune.example").exists()
    assert main(["train", str(arm), "--host", str(hosts), "--no-bind", "--render"]) == 0   # renders for elsewhere


@pytest.mark.parametrize("visible, preset, want", [("0,1", 8, 2), ("0,1,2,3,4,5,6,7", 8, None),
                                                   (",".join(map(str, range(16))), 8, None),
                                                   ("0,1", 1, None), ("0,1,2", 8, "FAIL"), ("0,1,2", 1, "FAIL")])
def test_the_visible_gpus_replace_a_presets_sharding_only_when_it_cannot_run_on_them(tmp_path, visible, preset, want):
    """A launch's device count must be a multiple of its sharding: a preset that already is (the smoke
    presets' explicit 1 always; 16 GPUs over a sharding of 8) keeps its own. 3 GPUs do not divide the
    batch of 32, so no sharding launches on them: refused, naming --fsdp-devices."""
    import dataclasses

    from axis_qvla.core.hosts import load_host
    from axis_qvla.train.recipes import load_recipe

    hosts, paths = host(tmp_path)
    spec = loads_armspec(arm_toml())
    recipe = load_recipe(spec.preset)
    recipe = dataclasses.replace(recipe, structure={**recipe.structure, "fsdp_devices": preset})
    profile = load_host("qvla", hosts_dir=hosts.parent)
    if want == "FAIL":
        with pytest.raises(launch.LaunchError, match=r"3 visible GPUs do not divide its global batch of 32.*--fsdp-devices N"):
            launch.default_fsdp_devices(spec, recipe, paths, profile, {"CUDA_VISIBLE_DEVICES": visible}, resume=False)
        return
    assert launch.default_fsdp_devices(spec, recipe, paths, profile, {"CUDA_VISIBLE_DEVICES": visible},
                                       resume=False) == want


def test_visible_gpus_no_sharding_can_launch_on_are_refused_before_rendering(machine, ran, monkeypatch, tmp_path, capsys):
    hosts, paths, _ = machine
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml())
    assert main(["train", str(arm), "--host", str(hosts), "--no-bind"]) == 2
    assert "FAIL  pi05.finetune.example: the 3 visible GPUs do not divide" in capsys.readouterr().out
    assert ran == [] and not (paths.runs / "pi05.finetune.example").exists()


def test_a_plain_copy_merged_here_records_its_roots_index(machine, monkeypatch, tmp_path):
    from axis_qvla.train import plain_copy, smoke

    hosts, paths, _ = machine
    rowspace = f"{smoke.SMOKE_ROWSPACE}#sha256:{H}"
    arms = smoke.fixture_arms(rowspace, {"pi05": f"sha256:{H}", "lerobot": f"sha256:{H}"}, rowspace, 609)
    rs = ids.rowspace_dirs(paths, rowspace)
    rs["roots"].parent.mkdir(parents=True)
    rs["roots"].write_text(json.dumps({"version": 2, "base": "../corpus_d8", "roots": {"2013": "task_2013"}}))
    merged = []

    def merge(l):          # the merge builds the copy; the training would run next
        merged.append(l)
        rs["lerobot_dataset"].mkdir(parents=True)
        return 0

    monkeypatch.setattr(launch, "RUNNER", merge)
    monkeypatch.setattr(runs, "missing_base_files", lambda name, p: [])     # the smoke's base is there
    arm = tmp_path / "arm.toml"
    arm.write_text(dumps_armspec(arms["smolvla"]))
    main(["train", str(arm), "--host", str(hosts), "--smoke"])       # binding then refuses the fake labels
    assert len(merged) == 1
    record = json.loads((rs["lerobot_dataset"] / plain_copy.PLAIN_RECORD).read_text())
    assert record["roots_sha256"] == sha256_file(rs["roots"])


def test_a_failed_merge_names_the_partial_copy_it_left(machine, monkeypatch, tmp_path, capsys):
    hosts, paths, _ = machine
    spec = loads_armspec(arm_toml(**LEROBOT_FT))
    rs = ids.rowspace_dirs(paths, spec.rowspace)
    rs["roots"].parent.mkdir(parents=True)
    rs["roots"].write_text(json.dumps({"2": "ft_demos/task_2"}))
    monkeypatch.setattr(launch, "RUNNER", lambda l: rs["lerobot_dataset"].mkdir(parents=True) or 1)
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml(**LEROBOT_FT))
    assert main(["train", str(arm), "--host", str(hosts)]) == 1
    out = capsys.readouterr().out
    assert f"FAIL  smolvla.finetune.example: the merged LeRobot copy was not built (exit 1); remove the partial " \
           f"{rs['lerobot_dataset']} yourself" in out


def test_a_render_naming_nothing_by_name_removes_a_stale_names_record(machine, ran, tmp_path):
    hosts, paths, _ = machine
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml().replace(f'rowspace = "heldout24_ft#sha256:{H}"', 'rowspace = "heldout24_ft"'))
    assert main(["train", str(arm), "--host", str(hosts), "--no-bind", "--render"]) == 0
    names = paths.runs / "pi05.finetune.example" / "names.json"
    assert names.is_file()
    arm.write_text(arm_toml())
    assert main(["train", str(arm), "--host", str(hosts), "--no-bind", "--render"]) == 0
    assert not names.exists()


def test_a_resume_rewrites_neither_names_nor_binding_records(machine, ran, tmp_path):
    hosts, paths, _ = machine
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml().replace(f'rowspace = "heldout24_ft#sha256:{H}"', 'rowspace = "heldout24_ft"'))
    assert main(["train", str(arm), "--host", str(hosts), "--no-bind", "--render"]) == 0
    run = paths.runs / "pi05.finetune.example"
    (run / "names.json").write_bytes((run / "names.json").read_bytes() + b" ")      # marks the first launch's
    marked = (run / "names.json").read_bytes()
    (run / "train" / "100").mkdir(parents=True)
    assert main(["train", str(arm), "--host", str(hosts), "--no-bind", "--render", "--resume"]) == 0
    assert (run / "names.json").read_bytes() == marked


def test_a_failed_training_is_reported_with_the_fail_padding(machine, monkeypatch, tmp_path, capsys):
    hosts, _, _ = machine
    monkeypatch.setattr(launch, "RUNNER", lambda l: 5)
    arm = tmp_path / "arm.toml"
    arm.write_text(arm_toml())
    assert main(["train", str(arm), "--host", str(hosts), "--no-bind"]) == 5
    assert capsys.readouterr().out.splitlines()[-1] == "FAIL  pi05.finetune.example: training exited 5"
