"""Run specs of the real-world line that name this machine's inputs by name, resolve to their content, and render."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from axis_qvla.realrobot.fr3 import local_runs
from axis_qvla.realrobot.fr3.cli import main
from axis_qvla.realrobot.fr3.line import load_line
from axis_qvla.train.armspec import load_armspec
from axis_qvla.train.generate import openpi_config
from axis_qvla.train.recipes import load_recipe
from tests.unit.realrobot.fr3.local_fixture import build_local

# Statistics of no particular data: what a norm file of another dataset looks like to the check.
PRETRAIN_STAND_IN = {"norm_stats": {k: {"mean": [0.0] * 8, "std": [1.0] * 8, "q01": [-1.0] * 8, "q99": [1.0] * 8}
                                    for k in ("state", "actions")}}


def test_local_runs_name_the_built_inputs_by_name_and_the_pretrain_renders(tmp_path):
    """The written spec carries names only (no 64-hex digest); they resolve to the built inputs' ids."""
    import re

    _toml, paths = build_local(tmp_path)
    line = load_line(local_runs.LINE)
    written = local_runs.write_local_runs(paths, line, runs=["pi05.pretrain.realrobot_cond"])
    path = written["pi05.pretrain.realrobot_cond"]
    assert not re.search(r"[0-9a-f]{64}", path.read_text())
    named = load_armspec(path)
    assert (named.rowspace, named.bundle, named.norm) == ("realworld_pretrain", "name:realrobot.conditioning",
                                                         "norm_realalign_sim")
    spec = local_runs.load_local_run(path, paths)
    manifest = json.loads((paths.cache / "bundles" / "realign_compiled.json").read_text())["runs"]
    assert spec.rowspace.startswith("realworld_pretrain#sha256:")
    assert spec.bundle == f"sha256:{manifest['realalign_conditioning']}"
    norm = paths.cache / "derived" / "norm" / "norm_realalign_sim" / "qvla-v1-realworld" / "norm_stats.json"
    assert spec.norm == f"norm_realalign_sim#sha256:{hashlib.sha256(norm.read_bytes()).hexdigest()}"
    cfg = openpi_config(spec, load_recipe(spec.preset), paths, smoke=True)
    assert cfg["freeze"] == "lora_frozen_vision" and cfg["data"]["sim_image_aug"] is True
    assert cfg["data"]["action_units"] == "joint_velocity_rad_s_clipped_1"
    assert cfg["data"]["roots_index"].endswith("realworld/built/roots.json")


def test_a_finetune_run_names_the_demonstrations_tail_frames_and_its_labels_compile_over_them(tmp_path):
    from axis_qvla.train.bind import compile_finetune_bundle

    _toml, paths = build_local(tmp_path)
    written = local_runs.write_local_runs(paths, load_line(local_runs.LINE), runs=["pi05.finetune.realrobot_cond"])
    spec = local_runs.load_local_run(written["pi05.finetune.realrobot_cond"], paths)
    assert spec.rowspace.startswith("realworld_finetune#sha256:") and spec.bundle is None
    assert spec.norm.startswith("norm_fr3_droid#sha256:") and spec.init == "run:pi05.pretrain.realrobot_cond"
    bundle = compile_finetune_bundle(spec, load_recipe(spec.preset), paths)
    assert bundle.rowspace.n_rows == (60 - 10) + (50 - 10) + (40 - 10)
    cfg = openpi_config(spec, load_recipe(spec.preset), paths, smoke=True, bundle="sha256:" + "b" * 64)
    assert cfg["data"]["action_units"] == "droid_normalized_joint_velocity_command"
    assert "sim_image_aug" not in cfg["data"] and cfg["data"]["repo_id"] == "fr3_demonstrations"


def test_a_run_whose_compiled_labels_were_not_built_is_named(tmp_path):
    _toml, paths = build_local(tmp_path, runs=("bc",))
    with pytest.raises(ValueError, match="realalign_conditioning"):
        local_runs.write_local_runs(paths, load_line(local_runs.LINE), runs=["pi05.pretrain.realrobot_cond"])


def test_the_cli_writes_every_run_of_the_line(tmp_path, capsys):
    toml, paths = build_local(tmp_path, runs=("bc", "conditioning", "awr", "filtering"))
    capsys.readouterr()
    assert main(["local-runs", "--host", str(toml)]) == 0
    out = capsys.readouterr().out.splitlines()
    line = load_line(local_runs.LINE)
    runs_dir = paths.cache / "derived" / local_runs.LOCAL / "runs"
    assert out == [f"OK    {r} -> {runs_dir / (r + '.toml')}" for r in sorted(line.runs)]


def test_the_cli_names_a_missing_input_and_writes_nothing_for_it(tmp_path, capsys):
    toml, _paths = build_local(tmp_path, runs=("bc",))
    capsys.readouterr()
    assert main(["local-runs", "--host", str(toml), "--runs", "pi05.pretrain.realrobot_awr"]) == 2
    assert "realalign_awr" in capsys.readouterr().err


def _scale_demo_actions(paths, factor):
    """Rewrite the demonstrations' actions in place: other content, the same lengths."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from axis_qvla.realrobot.realdata.training_set import _fsl
    for f in sorted((paths.cache / "derived" / local_runs.LOCAL / "demos").glob("task_*/data/*/*.parquet")):
        t = pq.read_table(f)
        a = np.asarray(t.column("action").combine_chunks().flatten()).reshape(len(t), -1) * factor
        pq.write_table(t.set_column(t.schema.get_field_index("action"), "action", _fsl(a)), f)


def _norm(paths, stage):
    from axis_qvla.realrobot.fr3 import norm
    name, dataset, embodiment = local_runs.NORMS[stage]
    asset = {"pretrain": "qvla-v1-realworld", "finetune": "fr3_demonstrations"}[stage]
    out = paths.cache / "derived" / "norm" / name / asset / "norm_stats.json"
    norm.write_norm(out, norm.compute_norm(paths.cache / "derived" / local_runs.LOCAL / dataset, embodiment))


def test_a_finetune_run_names_its_demonstrations_by_content(tmp_path):
    _toml, paths = build_local(tmp_path)
    run = "pi05.finetune.realrobot_only"
    before = local_runs.load_local_run(local_runs.write_local_runs(paths, load_line(local_runs.LINE), runs=[run])[run], paths)
    _scale_demo_actions(paths, 0.5)
    _norm(paths, "finetune")
    after = local_runs.load_local_run(local_runs.write_local_runs(paths, load_line(local_runs.LINE), runs=[run])[run], paths)
    assert before.rowspace != after.rowspace and before.norm != after.norm


@pytest.mark.parametrize("run, stage", [("pi05.finetune.realrobot_only", "finetune"),
                                        ("pi05.pretrain.realrobot_cond", "pretrain")])
def test_norm_statistics_of_other_data_are_refused_before_any_run_spec_is_written(tmp_path, run, stage):
    _toml, paths = build_local(tmp_path)
    if stage == "finetune":
        _scale_demo_actions(paths, 0.5)          # the demonstrations rebuilt; the old statistics kept
    else:
        f = paths.cache / "derived" / "norm" / "norm_realalign_sim" / "qvla-v1-realworld" / "norm_stats.json"
        f.write_text(json.dumps(PRETRAIN_STAND_IN))
    with pytest.raises(ValueError, match=r"do not describe .* re-run `qvla realrobot fr3 norm --dataset"):
        local_runs.write_local_runs(paths, load_line(local_runs.LINE), runs=[run])
    assert not (paths.cache / "derived" / local_runs.LOCAL / "runs").exists()


@pytest.mark.parametrize("missing, hint", [("demos", "qvla realrobot demos"),
                                           ("built", "qvla realrobot labels")])
def test_a_missing_training_layout_names_the_command_that_writes_it(tmp_path, capsys, missing, hint):
    import shutil
    toml, paths = build_local(tmp_path)
    shutil.rmtree(paths.cache / "derived" / local_runs.LOCAL / missing)
    capsys.readouterr()
    run = {"demos": "pi05.finetune.realrobot_only", "built": "pi05.pretrain.realrobot_bc"}[missing]
    assert main(["local-runs", "--host", str(toml), "--runs", run]) == 2
    err = capsys.readouterr().err
    assert err.startswith("FAIL  ") and hint in err


def test_an_unknown_run_and_missing_norm_statistics_are_refused_by_the_cli(tmp_path, capsys):
    toml, paths = build_local(tmp_path)
    capsys.readouterr()
    assert main(["local-runs", "--host", str(toml), "--runs", "pi05.finetune.nope"]) == 2
    assert "pi05.finetune.nope: not a run of the line record" in capsys.readouterr().err
    (paths.cache / "derived" / "norm" / "norm_fr3_droid" / "fr3_demonstrations" / "norm_stats.json").unlink()
    assert main(["local-runs", "--host", str(toml), "--runs", "pi05.finetune.realrobot_only"]) == 2
    assert "no norm statistics at" in capsys.readouterr().err


def test_the_norm_files_are_at_the_asset_id_each_run_renders(tmp_path):
    from axis_qvla.train.generate import openpi_asset_id
    from axis_qvla.train.recipes import resolved_fields
    _toml, paths = build_local(tmp_path)
    for run, path in local_runs.write_local_runs(paths, load_line(local_runs.LINE),
                                                 runs=["pi05.pretrain.realrobot_cond", "pi05.finetune.realrobot_only"]).items():
        spec = local_runs.load_local_run(path, paths)
        cfg = openpi_config(spec, load_recipe(spec.preset), paths, serve=True)
        assert cfg["data"]["asset_id"] == openpi_asset_id(spec, resolved_fields(load_recipe(spec.preset), spec))


def test_the_shipped_declarations_still_name_local_inputs():
    files = sorted(Path(local_runs.LINE).parent.joinpath("runs").glob("*.toml"))
    assert files and all("local:" in f.read_text() for f in files)


def test_the_labels_step_records_each_run_s_labels_by_name(tmp_path):
    """`qvla realrobot labels` (realign.bundles.write_bundles) records realrobot.<run>, so a run spec can
    name them before local-runs ever ran; the name resolves to the manifest's id."""
    from axis_qvla.train.names import resolve_labels

    _toml, paths = build_local(tmp_path, runs=("bc", "awr"))
    manifest = json.loads((paths.cache / "bundles" / "realign_compiled.json").read_text())["runs"]
    for run in ("bc", "awr"):
        assert resolve_labels(paths, f"name:realrobot.{run}") == f"sha256:{manifest[f'realalign_{run}']}"


def test_qvla_train_resolves_the_names_and_records_their_ids_in_names_json(tmp_path, capsys):
    """`qvla train <written spec> --render` binds the named inputs: names.json beside the rendered files
    holds each name with the id it resolved to, and the spec itself still holds no digest."""
    from axis_qvla.cli import main

    toml, paths = build_local(tmp_path)
    run = "pi05.pretrain.realrobot_cond"
    path = local_runs.write_local_runs(paths, load_line(local_runs.LINE), runs=[run])[run]
    # The base weights the run starts from, as `qvla data get base-weights` lays them out (not read here).
    (paths.cache / "checkpoints/qvla-v1-base-weights/openpi/openpi-assets/checkpoints/pi05_base/params").mkdir(parents=True)
    assert main(["train", str(path), "--host", str(toml), "--render", "--smoke"]) == 0, capsys.readouterr().out
    named = json.loads(next((paths.runs / run).rglob("names.json")).read_text())
    resolved = local_runs.load_local_run(path, paths)
    assert named == {"bundle": {"name": "name:realrobot.conditioning", "id": resolved.bundle},
                     "norm": {"name": "norm_realalign_sim", "id": resolved.norm},
                     "rowspace": {"name": "realworld_pretrain", "id": resolved.rowspace}}
