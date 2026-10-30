"""The real-robot line record, its nine ArmSpecs and the presets they name agree, and none of them trains."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from axis_qvla.realrobot.fr3 import checks, line

REPO = Path(__file__).resolve().parents[4]
LINE = REPO / "experiments" / "realrobot" / "line.toml"
RUNS = REPO / "experiments" / "realrobot" / "runs"


def test_the_line_record_has_nine_training_runs_and_two_serve_conditions():
    ln = line.load_line(LINE)
    assert len(ln.runs) == 9 and {s["stage"] for s in ln.runs.values()} == {"pretrain", "finetune"}
    assert set(ln.serve) == {"bare", "quality5"} and ln.units["declared_transfer"] is True
    assert ln.stage2["image"] == ln.stage1["image"] == "center_crop"
    assert ln.stage2["dataset"] == "ZhixuLi/tasl-fr3-10task-pbc-v2"


def test_every_run_has_an_armspec_the_loader_accepts_and_that_agrees_with_the_record():
    assert checks.check_runs(line.load_line(LINE), RUNS) == []


def test_an_armspec_that_disagrees_with_the_record_is_named(tmp_path):
    shutil.copytree(RUNS, tmp_path / "runs")
    p = tmp_path / "runs" / "pi05.finetune.realrobot_cond.toml"
    p.write_text(p.read_text().replace('"data.quality_tag" = 5', '"data.quality_tag" = 4'))
    assert any("quality_tag" in x for x in checks.check_runs(line.load_line(LINE), tmp_path / "runs"))
    (tmp_path / "runs" / "pi05.pretrain.extra.toml").write_text(p.read_text())
    assert any("not in the line record" in x for x in checks.check_runs(line.load_line(LINE), tmp_path / "runs"))


def test_a_missing_armspec_and_a_missing_runs_directory_are_named(tmp_path):
    shutil.copytree(RUNS, tmp_path / "runs")
    (tmp_path / "runs" / "pi05.pretrain.realrobot_drop.toml").unlink()
    assert checks.check_runs(line.load_line(LINE), tmp_path / "runs") == ["pi05.pretrain.realrobot_drop: no ArmSpec"]
    problems = checks.check_runs(line.load_line(LINE), tmp_path / "nowhere")
    assert len(problems) == 1 and "cannot list" in problems[0]


def test_the_run_files_are_the_canonical_form_of_the_record():
    """Each file is exactly what the record declares, written by the ArmSpec writer, so a hand edit
    that the field checks happen not to read still shows."""
    from axis_qvla.train.armspec import dumps_armspec

    ln = line.load_line(LINE)
    assert sorted(p.stem for p in RUNS.glob("*.toml")) == sorted(ln.runs)
    for run_id in ln.runs:
        assert (RUNS / f"{run_id}.toml").read_text(encoding="utf-8") == dumps_armspec(line.run_spec(ln, run_id)), run_id


def test_no_run_of_the_line_names_a_paper_config():
    ln = line.load_line(LINE)
    for run_id, s in ln.runs.items():
        assert "paper_ref" not in s and line.run_spec(ln, run_id).paper_ref is None, run_id


def test_the_record_checks_its_pretrain_and_finetune_runs():
    ln = line.load_line(LINE)

    def problems(run_id, **edit):
        spec = {k: v for k, v in {**ln.runs[run_id], **edit}.items() if v is not None}
        return checks.check_record(line.Line(ln.stage1, ln.stage2, ln.units, ln.serve, {**ln.runs, run_id: spec}))

    assert problems("pi05.finetune.realrobot_awr") == []
    assert problems("pi05.pretrain.realrobot_drop", signal=None) == [
        "pi05.pretrain.realrobot_drop: a pretrain run names no signal"]
    assert problems("pi05.finetune.realrobot_bc", serve="nope") == [
        "pi05.finetune.realrobot_bc: serve condition 'nope' is not in [serve]"]


def test_only_the_tagged_finetune_overrides_the_quality_tag():
    ln = line.load_line(LINE)
    tagged = {r for r in ln.runs if line.run_spec(ln, r).overrides}
    assert tagged == {"pi05.finetune.realrobot_cond"}
    assert dict(line.run_spec(ln, "pi05.finetune.realrobot_cond").overrides) == {"data.quality_tag": 5}


def test_qvla_train_refuses_every_run_of_the_line():
    """Declarations, not runs: the check every `qvla train` route makes first refuses each one and names
    the command that writes the run specs that train."""
    from axis_qvla.train.armspec import load_armspec
    from axis_qvla.train.generate import GenerateError, check_renderable

    for p in sorted(RUNS.glob("*.toml")):
        with pytest.raises(GenerateError, match="by a local: name.*fr3 local-runs"):
            check_renderable(load_armspec(p))


@pytest.mark.parametrize("run_id", ["pi05.pretrain.realrobot_cond", "pi05.finetune.realrobot_cond"])
@pytest.mark.parametrize("bind", [True, False])
def test_qvla_train_exits_2_naming_the_declaration_and_writes_nothing(tmp_path, capsys, run_id, bind):
    from axis_qvla.cli import main
    from tests.unit.train._arms import host

    qvla_toml, paths = host(tmp_path)
    argv = (["train", str(RUNS / f"{run_id}.toml"), "--host", str(qvla_toml), "--render"]
            + ([] if bind else ["--no-bind"]))
    assert main(argv) == 2
    out = capsys.readouterr().out
    assert out.startswith(f"FAIL  {run_id}: names ") and "`qvla realrobot pretrain` and `qvla realrobot finetune`" in out
    assert not paths.runs.exists() and not paths.cache.exists()


def test_an_undeclared_unit_change_is_refused():
    ln = line.load_line(LINE)
    bad = line.Line(ln.stage1, ln.stage2, {**ln.units, "declared_transfer": False}, ln.serve, ln.runs)
    assert any("unit" in p for p in checks.check_units(bad, "pi05.finetune.realrobot_cond"))
    assert checks.check_units(ln, "pi05.finetune.realrobot_cond") == []
    assert checks.check_units(bad, "pi05.finetune.realrobot_only") == []      # from the base: no unit change


def test_the_record_names_units_the_embodiments_and_the_fr3_declaration_agree_on():
    ln = line.load_line(LINE)
    wrong = line.Line(ln.stage1, ln.stage2, {**ln.units, "stage2": "rad_s"}, ln.serve, ln.runs)
    assert any("embodiment units" in p for p in checks.check_units(wrong, "pi05.finetune.realrobot_bc"))


def test_an_fr3_embodiment_that_disagrees_with_the_recordings_declaration_is_named(monkeypatch):
    """The unit check compares the FR3 embodiment with state_order.ACTION_UNITS, the record check with
    its declared layouts, so a declaration edited on one side only fails."""
    import dataclasses

    real = checks.load_embodiment

    def edited(name):
        e = real(name)
        return dataclasses.replace(e, action_units="rad_s", state_stored=e.state_model) if e.id == "fr3" else e

    monkeypatch.setattr(checks, "load_embodiment", edited)
    ln = line.load_line(LINE)
    assert ("fr3: embodiment action.units 'rad_s', the recording declares 'droid_normalized_joint_velocity_command'"
            in checks.check_units(ln, "pi05.finetune.realrobot_only"))
    assert any(p.startswith("fr3: embodiment state.stored") for p in checks.check_record(ln))


def test_the_record_is_consistent_with_itself_and_its_embodiments():
    ln = line.load_line(LINE)
    assert checks.check_record(ln) == []
    runs = {**ln.runs, "pi05.finetune.realrobot_bc": {**ln.runs["pi05.finetune.realrobot_bc"], "serve": "tagged"}}
    assert checks.check_record(line.Line(ln.stage1, ln.stage2, ln.units, ln.serve, runs)) == [
        "pi05.finetune.realrobot_bc: serve condition 'tagged' is not in [serve]"]
    runs = {**ln.runs, "pi05.finetune.realrobot_bc": {**ln.runs["pi05.finetune.realrobot_bc"],
                                                       "init": "run:pi05.pretrain.realrobot_none"}}
    assert checks.check_record(line.Line(ln.stage1, ln.stage2, ln.units, ln.serve, runs)) == [
        "pi05.finetune.realrobot_bc: init 'run:pi05.pretrain.realrobot_none' is neither a base nor a pretrain run "
        "of the record"]


def test_a_stage_whose_frames_carry_their_geometry_must_name_its_dataset():
    """The fr3 embodiment declares no geometry (it is measured from the set's pixels), so the record's
    stage 2 must say which set its image describes; an image the record cannot map is named."""
    ln = line.load_line(LINE)
    st2 = {k: v for k, v in ln.stage2.items() if k != "dataset"}
    assert checks.check_record(line.Line(ln.stage1, st2, ln.units, ln.serve, ln.runs)) == [
        "stage2: image 'center_crop' is a property of the fr3 frames' stored pixels; the stage names no dataset"]
    st1 = {**ln.stage1, "image": "raw"}
    assert checks.check_record(line.Line(st1, ln.stage2, ln.units, ln.serve, ln.runs)) == [
        "stage1: image 'raw' is none of center_crop, letterbox"]
    st1 = {**ln.stage1, "image": "letterbox"}
    assert checks.check_record(line.Line(st1, ln.stage2, ln.units, ln.serve, ln.runs)) == [
        "stage1: image 'letterbox', the realalign_sim frames are 'center_crop224'"]


def test_the_presets_carry_the_record_through_the_recipe_loader():
    assert checks.check_presets(line.load_line(LINE)) == []


def test_a_preset_that_disagrees_with_the_record_is_named():
    ln = line.load_line(LINE)
    bad = line.Line({**ln.stage1, "warmup": 500}, ln.stage2, ln.units, ln.serve, ln.runs)
    assert checks.check_presets(bad) == ["pi05.pretrain.lora.realrobot: warmup = 1000, the record says 500"]
    crop = line.Line(ln.stage1, {**ln.stage2, "image": "letterbox"}, ln.units, ln.serve, ln.runs)
    assert checks.check_presets(crop) == [
        f"{rid}: image_center_crop = True, the record says False"
        for rid in ("pi05.finetune.lora.realrobot", "pi05.finetune.lora.realrobot_only")]


def test_a_preset_whose_augmentation_or_units_are_not_its_stages_is_named():
    ln = line.load_line(LINE)
    no_aug = line.Line({**ln.stage1, "sim_image_aug": False}, ln.stage2, ln.units, ln.serve, ln.runs)
    assert checks.check_presets(no_aug) == [
        "pi05.pretrain.lora.realrobot: data.sim_image_aug True, the record's pretrain stage says False"]
    sim = line.Line(ln.stage1, {**ln.stage2, "embodiment": "realalign_sim"}, ln.units, ln.serve, ln.runs)
    assert checks.check_presets(sim) == [
        f"{rid}: data.action_units 'droid_normalized_joint_velocity_command', the {k} data's units are "
        f"'joint_velocity_rad_s_clipped_1'"
        for rid, k in (("pi05.finetune.lora.realrobot", "finetune"), ("pi05.finetune.lora.realrobot_only", "finetune_only"))]


def test_a_preset_without_a_model_key_of_the_paper_presets_is_named(tmp_path):
    """The openpi line requires every model field; the paper presets name them all."""
    import shutil

    from axis_qvla.train.recipes import RECIPES_DIR
    recipes = shutil.copytree(RECIPES_DIR, tmp_path / "recipes")
    f = recipes / "pi05.pretrain.lora.realrobot.toml"
    f.write_text("".join(x for x in f.read_text().splitlines(keepends=True)
                         if not x.startswith('"model.pytorch_compile_mode"')))
    assert checks.check_presets(line.load_line(LINE), recipes) == [
        "pi05.pretrain.lora.realrobot: no model.pytorch_compile_mode (pi05.finetune.lora.paper names "
        "every model field the openpi line requires)"]


def test_a_line_file_of_another_schema_is_refused(tmp_path):
    p = tmp_path / "line.toml"
    p.write_text(LINE.read_text().replace("qvla.realrobot_line/1", "qvla.realrobot_line/2"))
    with pytest.raises(line.LineError, match="schema"):
        line.load_line(p)


_CLOSEDNESS_FIRST = ("closedness", "q0", "q1", "q2", "q3", "q4", "q5", "q6")


@pytest.mark.parametrize("edit, check, named", [
    ({"fps": 10}, "units", "realalign_sim: embodiment fps 10, the renderer declares 15"),
    ({"action_range": (-3.0, 3.0)}, "units",
     "realalign_sim: embodiment action.range (-3.0, 3.0), the renderer declares (-1.0, 1.0)"),
    ({"state_stored": _CLOSEDNESS_FIRST, "state_model": _CLOSEDNESS_FIRST}, "record",
     f"realalign_sim: embodiment state.stored {_CLOSEDNESS_FIRST!r}, the renderer declares "
     f"('q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'closedness')"),
    ({"image_stored": (224, 224)}, "record", "realalign_sim: embodiment images.stored (224, 224), the renderer declares (640, 360)"),
])
def test_the_rendered_views_embodiment_is_held_to_its_producer(monkeypatch, edit, check, named):
    """realalign_sim is what axis_qvla/realrobot/render writes: rig.FPS, the +-1 rad/s clip of droid8d, its
    state and action layouts and frame size; an embodiment edited on one side only fails."""
    import dataclasses

    real = checks.load_embodiment
    monkeypatch.setattr(checks, "load_embodiment",
                        lambda name: dataclasses.replace(real(name), **edit) if name == "realalign_sim" else real(name))
    ln = line.load_line(LINE)
    got = checks.check_units(ln, "pi05.pretrain.realrobot_bc") if check == "units" else checks.check_record(ln)
    assert named in got, got
