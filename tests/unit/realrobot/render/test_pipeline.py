"""Rendering a selection of tasks: one task at a time, resumable, an IK refusal costing only its task, and
the command line naming every refusal. The GL path is replaced by a writer of stand-in frames, so these
run in the render environment without a GL backend."""
from __future__ import annotations

import json

import pytest

from axis_qvla.realrobot.render import cli, data, profile, source
from tests.unit.realrobot.render import fixtures
from tests.unit.realrobot.render.test_writer import _episode, _frames
from tests.unit.realrobot.realworld_fixture import write_realworld

pytest.importorskip("mink")         # the pipeline's IK needs the render environment
from axis_qvla.realrobot.render import ik, pipeline  # noqa: E402


def _stand_in(failing=(), log=None):
    """A render_episode that writes three stand-in frames, or raises the IK's refusal for an attempt in
    `failing`; each call is logged."""
    def render_episode(ep, _gripper_xml, task_writer, _profile, **_kw):
        if log is not None:
            log.append(("render", ep.task_id, ep.attempt_id))
        if ep.attempt_id in failing:
            raise ik.IkError("the retargeted arm reaches only 10.0% of rows within 2 cm")
        task_writer.add_episode(_episode(3, ep.attempt_id), _frames(3, ep.attempt_id))
        return {}
    return render_episode


def _sources(tmp_path, pairs):
    return {tid: [fixtures.write_episode(tmp_path / "src" / f"{tid}_{aid}", n=4, task_id=tid, attempt_id=aid)
                  for aid in aids] for tid, aids in pairs.items()}


def test_an_ik_refusal_costs_its_task_only_and_a_rerun_resumes(tmp_path, monkeypatch):
    log = []
    monkeypatch.setattr(pipeline, "render_episode", _stand_in(failing={7002}, log=log))
    tasks, out = _sources(tmp_path, {9001: [7001, 7002], 9002: [7003]}), tmp_path / "render"
    done = pipeline.render_tasks(tasks, tmp_path / "gripper.xml", out)
    assert done.rendered == [9002] and done.kept == []
    assert done.failed == {9001: "task 9001 attempt 7002: the retargeted arm reaches only 10.0% of rows within 2 cm"}
    assert sorted(p.name for p in out.iterdir()) == ["roots.json", "task_9002", profile.FILE]
    assert json.loads((out / "roots.json").read_text())["roots"] == {"9002": "task_9002"}

    log.clear()
    monkeypatch.setattr(pipeline, "render_episode", _stand_in(log=log))
    again = pipeline.render_tasks(tasks, tmp_path / "gripper.xml", out)
    assert (again.rendered, again.kept, again.failed) == ([9001], [9002], {})
    assert log == [("render", 9001, 7001), ("render", 9001, 7002)]          # task 9002 is not rendered again
    assert json.loads((out / "roots.json").read_text())["roots"] == {"9001": "task_9001", "9002": "task_9002"}


def test_an_interrupted_task_leaves_only_its_partial_directory_which_the_rerun_replaces(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "render_episode", _stand_in())
    tasks, out = _sources(tmp_path, {9001: [7001]}), tmp_path / "render"
    (out / "task_9001.partial" / "videos").mkdir(parents=True)          # what a killed process leaves
    assert pipeline.render_tasks(tasks, tmp_path / "gripper.xml", out).rendered == [9001]
    assert not (out / "task_9001.partial").exists() and (out / "task_9001" / "meta" / "info.json").is_file()


def test_any_other_error_removes_the_partial_task_and_propagates(tmp_path, monkeypatch):
    def broken(*_a, **_k):
        raise source.SourceError("'box_1' is not in the recorded scene")

    monkeypatch.setattr(pipeline, "render_episode", broken)
    out = tmp_path / "render"
    with pytest.raises(source.SourceError):
        pipeline.render_tasks(_sources(tmp_path, {9001: [7001]}), tmp_path / "gripper.xml", out)
    assert not list(out.glob("task_*"))


def test_episodes_are_read_one_task_at_a_time_and_a_kept_task_is_not_read(tmp_path, monkeypatch):
    log = []
    read = source.read_source_episode
    monkeypatch.setattr(pipeline.source, "read_source_episode", lambda p: log.append(("read", p.parent.name)) or read(p))
    monkeypatch.setattr(pipeline, "render_episode", _stand_in(log=log))
    tasks, out = _sources(tmp_path, {9001: [7001, 7002], 9002: [7003]}), tmp_path / "render"
    pipeline.render_tasks({9002: tasks[9002]}, tmp_path / "gripper.xml", out)
    log.clear()
    pipeline.render_tasks(tasks, tmp_path / "gripper.xml", out)
    assert log == [("read", "9001_7001"), ("read", "9001_7002"), ("render", 9001, 7001), ("render", 9001, 7002)]


def test_an_episode_filed_under_another_task_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "render_episode", _stand_in())
    tasks = _sources(tmp_path, {9001: [7001]})
    with pytest.raises(ValueError, match="task 9002: .* is an episode of task 9001"):
        pipeline.render_tasks({9002: tasks[9001]}, tmp_path / "gripper.xml", tmp_path / "render")


def _render_cli(tmp_path, *extra):
    return cli.main(["render", "--gripper-xml", str(tmp_path / "2f85.xml"), "--unlocked-gripper",
                     "--out", str(tmp_path / "render"), *extra])


def test_the_command_prints_the_failed_tasks_and_exits_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "render_episode", _stand_in(failing={7002}))
    eps = [p for ps in _sources(tmp_path, {9001: [7002], 9002: [7003]}).values() for p in ps]
    assert _render_cli(tmp_path, "--episodes", *map(str, eps)) == 1
    out, err = capsys.readouterr()
    assert f"rendered tasks [9002] -> {tmp_path / 'render'}" in out
    assert "FAIL task 9001 attempt 7002: the retargeted arm reaches only 10.0% of rows within 2 cm" in err
    assert "1 task not rendered (9001): their recordings are beyond the IK's acceptance; the rendered tasks are kept" in err
    monkeypatch.setattr(pipeline, "render_episode", _stand_in())
    assert _render_cli(tmp_path, "--episodes", *map(str, eps)) == 0
    assert "already rendered: tasks [9002]" in capsys.readouterr().out


def test_a_bad_profile_is_refused_before_any_episode_is_read(tmp_path, monkeypatch, capsys):
    write_realworld(tmp_path / "pub")
    data.prepare(tmp_path / "pub", tmp_path / "w")
    bad = tmp_path / "ws.toml"
    bad.write_text(profile.REFERENCE.read_text().replace("[camera]", "[camera]\nzoom = 2", 1))

    def no_read(_p):
        raise AssertionError("an episode was read before the profile")

    monkeypatch.setattr(pipeline.source, "read_source_episode", no_read)
    assert _render_cli(tmp_path, "--data", str(tmp_path / "w"), "--profile", str(bad)) == 2
    err = capsys.readouterr().err
    assert err.startswith("FAIL ") and "zoom" in err


@pytest.mark.parametrize("make, named", [
    ("bad_episode", "schema"),
    ("half_written_task", "exists; a rendered task is never overwritten"),
    ("other_profile", "holds a render of another workspace profile"),
])
def test_each_refusal_of_the_render_command_is_a_fail_line(tmp_path, monkeypatch, capsys, make, named):
    monkeypatch.setattr(pipeline, "render_episode", _stand_in())
    (ep,) = _sources(tmp_path, {9001: [7001]})[9001]
    out = tmp_path / "render"
    extra = []
    if make == "bad_episode":
        ep.write_text(json.dumps({**json.loads(ep.read_text()), "schema": "qvla.realrobot.source_episode/1"}))
    elif make == "half_written_task":
        (out / "task_9001" / "videos").mkdir(parents=True)                     # no meta/info.json
    else:
        pipeline.render_tasks({}, tmp_path / "gripper.xml", out)
        other = tmp_path / "ws.toml"
        other.write_text(profile.REFERENCE.read_text().replace("fovy = 71.5077745089", "fovy = 60.0"))
        extra = ["--profile", str(other)]
    assert _render_cli(tmp_path, "--episodes", str(ep), *extra) == 2
    err = capsys.readouterr().err
    assert err.startswith("FAIL ") and named in err and "Traceback" not in err


def test_the_render_command_keeps_the_profile_it_is_given(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "render_episode", _stand_in())
    (ep,) = _sources(tmp_path, {9001: [7001]})[9001]
    given = tmp_path / "mine.toml"
    given.write_text(profile.REFERENCE.read_text().replace("fovy = 71.5077745089", "fovy = 60.0"))
    assert _render_cli(tmp_path, "--episodes", str(ep), "--profile", str(given)) == 0
    assert (tmp_path / "render" / profile.FILE).read_text() == given.read_text()


def test_the_overlay_command_draws_the_profile_it_is_given_and_refuses_a_bad_one(tmp_path, monkeypatch, capsys):
    import numpy as np

    from axis_qvla.realrobot.render import overlay
    seen = []

    def render_view(_scene, _gripper, ws, _joints):
        seen.append(ws.text)
        return np.zeros((360, 640, 3), np.uint8), {"table": np.zeros((360, 640), bool), "robot": np.zeros((360, 640), bool)}

    monkeypatch.setattr(overlay, "render_view", render_view)
    given = tmp_path / "mine.toml"
    given.write_text(profile.REFERENCE.read_text().replace("fovy = 71.5077745089", "fovy = 60.0"))
    photo = tmp_path / "exterior.png"
    overlay.write_png(photo, np.zeros((720, 1280, 3), np.uint8))
    args = ["overlay", "--image", str(photo), "--scene", str(fixtures.write_scene(tmp_path / "s")),
            "--robots", str(tmp_path / "robots"), "--out", str(tmp_path / "o.png")]
    monkeypatch.setattr("axis_qvla.realrobot.render.assets.check_menagerie", lambda robots: tmp_path / "2f85.xml")
    assert cli.main(args + ["--profile", str(given)]) == 0
    assert seen == [given.read_text()]
    bad = tmp_path / "bad.toml"
    bad.write_text(profile.REFERENCE.read_text().replace("[camera]", "[camera]\nzoom = 2", 1))
    assert cli.main(args + ["--profile", str(bad)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("FAIL ") and "zoom" in err


def test_droid8d_into_a_view_that_exists_and_gates_on_a_half_written_task_are_fail_lines(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "render_episode", _stand_in())
    pipeline.render_tasks(_sources(tmp_path, {9001: [7001]}), tmp_path / "gripper.xml", tmp_path / "render")
    args = ["droid8d", "--src", str(tmp_path / "render"), "--out", str(tmp_path / "d8")]
    assert cli.main(args) == 0
    capsys.readouterr()
    assert cli.main(args) == 2
    err = capsys.readouterr().err
    assert err.startswith("FAIL ") and "task_9001 exists" in err
    (tmp_path / "render" / "task_9001" / "meta" / "info.json").unlink()      # a task directory without its info.json
    assert cli.main(["gates", "--out", str(tmp_path / "render")]) == 2
    err = capsys.readouterr().err
    assert err.startswith("FAIL ") and "info.json" in err and "Traceback" not in err
