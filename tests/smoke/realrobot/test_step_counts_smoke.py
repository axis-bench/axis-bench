"""A recording on step counts renders end to end on the label pipeline's frames, every hard gate green."""
from __future__ import annotations

import json

import numpy as np
import pytest

from axis_qvla.labels.resample import trace_anchor_rows
from axis_qvla.realrobot.render import cli, gates, labels, pipeline, source
from tests.unit.realrobot.render import fixtures

TIMES = [0.0, 0.2, 0.4, 0.6, 0.72, 0.76, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]


@pytest.mark.parametrize("times", [TIMES, [*TIMES[:-1], 1.82]], ids=["ends_on_a_frame", "ends_between_frames"])
def test_a_step_count_recording_renders_on_the_label_frames(tmp_path, times):
    scene = fixtures.write_scene(tmp_path / "src")
    trace = fixtures.write_trace(tmp_path / "src" / "trace.jsonl", scene, n=len(times), times=times)
    doc = source.from_trace(trace, task_id=9001, attempt_id=7001, prompt="push the red box", scene_xml="scene.xml",
                            scene_dir=tmp_path / "src")
    source.write_source_episode(tmp_path / "src" / "source_episode.json", doc)
    grip = fixtures.write_gripper(tmp_path / "gripper")
    pipeline.render_tasks({9001: [tmp_path / "src" / "source_episode.json"]}, grip, tmp_path / "render")
    frames = int(trace_anchor_rows(times, 15.0)[-1]) + 1
    assert cli.main(["droid8d", "--src", str(tmp_path / "render"), "--out", str(tmp_path / "d8")]) == 0
    labels.write_attempt_store(tmp_path / "attempts", fixtures.attempt_curves(frames))
    assert cli.main(["labels", "--pretraining-data", str(tmp_path / "d8"), "--attempt-curves", str(tmp_path / "attempts"),
                     "--out", str(tmp_path / "curves")]) == 0
    report = gates.run_gates(tmp_path / "render", droid8d_root=tmp_path / "d8", render_store=tmp_path / "curves",
                             attempt_curves_dir=tmp_path / "attempts", open_separation=fixtures.OPEN_SEP,
                             closed_separation=fixtures.CLOSED_SEP)
    assert all(g["ok"] for g in report["tasks"]["9001"].values()), report
    assert report["tasks"]["9001"]["structure"]["frames"] == frames
    side = json.loads((tmp_path / "render" / "task_9001" / "meta" / "axis_episodes" / "episode_000000.json").read_text())
    assert side["knot_rows"] == trace_anchor_rows(times, 15.0).tolist()
    assert np.isfinite(np.asarray(side["source_tcp"])).all()
