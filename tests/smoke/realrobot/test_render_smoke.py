"""Renderer smoke on one episode, end to end, every hard gate green.

Render tier with a GL backend: the `render` CI job (render venv, MUJOCO_GL=egl or osmesa); locally
    srun ... env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl <render venv>/bin/python -m pytest tests/smoke/realrobot

The one test here that drives `pipeline.render_tasks` is the only one that reaches the GL path
(`draw.frames`), so it also checks that the decoded videos hold a drawn scene, not a blank buffer.
"""
from __future__ import annotations

import json

import pytest

from axis_qvla.realrobot import layout as layout_mod
from axis_qvla.realrobot.render import assets, cli, gates, ik, labels, pipeline, report, rig, source
from tests.unit.realrobot.render import fixtures


def test_one_episode_renders_and_passes_every_hard_gate(tmp_path):
    ep_path = fixtures.write_episode(tmp_path / "src", n=12)
    grip = fixtures.write_gripper(tmp_path / "gripper")
    assert pipeline.render_tasks({9001: [ep_path]}, grip, tmp_path / "render").rendered == [9001]
    assert cli.main(["droid8d", "--src", str(tmp_path / "render"), "--out", str(tmp_path / "d8")]) == 0
    labels.write_attempt_store(tmp_path / "attempts", fixtures.attempt_curves(36))
    assert cli.main(["labels", "--pretraining-data", str(tmp_path / "d8"), "--attempt-curves", str(tmp_path / "attempts"),
                     "--out", str(tmp_path / "curves")]) == 0
    report_ = gates.run_gates(tmp_path / "render", droid8d_root=tmp_path / "d8", render_store=tmp_path / "curves",
                              attempt_curves_dir=tmp_path / "attempts", open_separation=fixtures.OPEN_SEP,
                              closed_separation=fixtures.CLOSED_SEP)
    assert all(g["ok"] for g in report_["tasks"]["9001"].values()), report_
    assert report_["label_tag"]["ok"]
    side = json.loads((tmp_path / "render" / "task_9001" / "meta" / "axis_episodes" / "episode_000000.json").read_text())
    assert side["convergence_knot"] <= 8 and side["frames"] == 36
    assert report_["reported"]["droid8d_velocity"]["rows"] == 36
    for key in ("observation.images.third_person", "observation.images.wrist"):
        frames = report.episode_frames(tmp_path / "render" / "task_9001", key, 0)
        assert frames.shape == (36, rig.HEIGHT, rig.WIDTH, 3) and frames.std() > 1.0, key


def test_the_cli_render_uses_the_layout_default_robots_when_absent(tmp_path, monkeypatch):
    """No `--robots`, and the layout default (axis_qvla.realrobot.layout.ROBOTS) is a usable, locked Menagerie
    checkout: `render` uses it rather than refusing."""
    ep_path = fixtures.write_episode(tmp_path / "src", n=4)
    robots_root = tmp_path / "robots"
    fixtures.write_gripper(robots_root / "robotiq_2f85")
    lock_path = tmp_path / "menagerie.lock.json"
    lock_path.write_text(json.dumps(assets.build_lock(robots_root, "0" * 40)))
    monkeypatch.setattr(layout_mod, "ROBOTS", robots_root)
    monkeypatch.setattr(assets, "LOCK", lock_path)
    out = tmp_path / "r"
    assert cli.main(["render", "--episodes", str(ep_path), "--out", str(out)]) == 0
    assert (out / "task_9001").is_dir()


def test_the_cli_render_refuses_a_missing_layout_default_without_the_flag(tmp_path, monkeypatch):
    """No `--robots`, and the layout default does not exist on this machine: refuse by name, rather than
    rendering nothing or failing on an unrelated error."""
    ep_path = fixtures.write_episode(tmp_path / "src", n=4)
    monkeypatch.setattr(layout_mod, "ROBOTS", tmp_path / "no-such-robots-checkout")
    with pytest.raises(SystemExit, match="--robots"):
        cli.main(["render", "--episodes", str(ep_path), "--out", str(tmp_path / "r")])


def test_the_cli_writes_a_source_episode_from_a_trace(tmp_path):
    scene = fixtures.write_scene(tmp_path / "ep")
    trace = fixtures.write_trace(tmp_path / "t.jsonl", scene, n=4)
    assert cli.main(["source", "--trace", str(trace), "--scene-dir", str(tmp_path / "ep"), "--scene-xml", "scene.xml",
                     "--task-id", "9001", "--attempt-id", "7001", "--prompt", "push the red box",
                     "--out", str(tmp_path / "ep" / "source_episode.json")]) == 0
    assert source.read_source_episode(tmp_path / "ep" / "source_episode.json").packed.n_steps == 4


def test_an_ik_failure_names_its_task_and_attempt(tmp_path, monkeypatch):
    ep = source.read_source_episode(fixtures.write_episode(tmp_path / "ep", task_id=9001, attempt_id=7001))

    def fail(*_a, **_k):
        raise ik.IkError("the retargeted arm reaches only 10.0% of rows within 2 cm")

    monkeypatch.setattr(pipeline, "render_episode", fail)
    done = pipeline.render_tasks({9001: [ep]}, tmp_path / "gripper.xml", tmp_path / "out")
    assert done.failed[9001].startswith("task 9001 attempt 7001: the retargeted arm reaches only")
    assert not list((tmp_path / "out").glob("task_*"))
