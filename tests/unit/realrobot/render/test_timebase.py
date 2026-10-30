"""Recorded rows land on the label pipeline's frames, whatever the recording's time base."""
from __future__ import annotations

import json

import numpy as np
import pytest

from axis_qvla.labels.resample import trace_anchor_rows
from axis_qvla.realrobot.render import resample, source
from tests.unit.realrobot.render import fixtures


def _packed(counts):
    n = len(counts) + 1
    z = np.zeros((n, 1))
    return source.PackedState(0.002, 100, n, 1, 1, 0, 0.0, np.zeros(1), np.zeros(1), None, z, z, np.zeros((n, 0)),
                              step_counts=tuple(counts))


def test_post_step_rows_are_frames_3k_plus_2_as_before():
    p = source.PackedState(0.001, 200, 81, 1, 1, 0, 0.0, np.zeros(1), np.zeros(1), None,
                           np.zeros((81, 1)), np.zeros((81, 1)), np.zeros((81, 0)))
    grid, rows = resample.frame_grid(p.row_times)
    assert rows.tolist() == (3 * np.arange(81) + 2).tolist() and len(grid) == 243 and p.lead_time == 0.0
    np.testing.assert_array_equal(grid, p.row_times[-1] - np.arange(242, -1, -1, dtype=np.float64) * (1.0 / 15.0))


def test_step_count_rows_start_at_the_initial_state_and_follow_the_counts():
    p = _packed([100, 100, 100, 60, 20, 20])
    np.testing.assert_allclose(p.row_times, [0.0, 0.2, 0.4, 0.6, 0.72, 0.76, 0.8], atol=1e-12)
    assert p.lead_time == pytest.approx(-0.2)
    grid, rows = resample.frame_grid(p.row_times)
    assert rows.tolist() == trace_anchor_rows(p.row_times, 15.0).tolist() == [2, 5, 8, 11, 13, 13, 14]
    assert len(grid) == 15 and grid[-1] == pytest.approx(0.8)


def test_step_count_times_accumulate_one_timestep_at_a_time_as_the_simulator_does():
    """150 steps of 0.002 s sum to 0.30000000000000024, not the product 0.3: on that exact half frame the
    accumulated time rounds to frame 7, as the label pipeline does with the trace's own times."""
    t = [0.0]
    for _ in range(150):
        t.append(t[-1] + 0.002)
    p = _packed([100, 50])
    np.testing.assert_array_equal(p.row_times, [t[0], t[100], t[150]])
    assert resample.frame_grid(p.row_times)[1].tolist() == trace_anchor_rows([t[0], t[100], t[150]], 15.0).tolist() \
        == [2, 5, 7]


def test_post_step_rows_off_the_5_hz_base_open_one_source_period_before_row_0():
    p = source.PackedState(0.001, 40, 12, 1, 1, 0, 0.0, np.zeros(1), np.zeros(1), None,
                           np.zeros((12, 1)), np.zeros((12, 1)), np.zeros((12, 0)))
    assert p.row_times[0] == pytest.approx(0.04) and p.lead_time == pytest.approx(0.04 - 0.2)
    assert resample.frame_grid(p.row_times)[0][0] > p.lead_time


def test_a_step_count_trajectory_round_trips_through_the_packed_format():
    p = _packed([100, 60, 20])
    q = source.decode_packed_state_v1(source.encode_packed_state_v1(p))
    assert q.step_counts == (100, 60, 20)
    np.testing.assert_allclose(q.row_times, p.row_times, atol=0)


def test_step_counts_of_the_wrong_length_are_refused():
    doc = source.encode_packed_state_v1(_packed([100, 60, 20]))
    doc["step_counts"] = [100, 60]
    with pytest.raises(source.SourceError, match="step_counts"):
        source.decode_packed_state_v1(doc)


def test_a_trace_on_another_time_base_becomes_step_counts(tmp_path):
    scene = fixtures.write_scene(tmp_path / "ep")
    times = [0.0, 0.2, 0.4, 0.46, 0.5, 0.7, 0.9, 1.1]
    trace = fixtures.write_trace(tmp_path / "t.jsonl", scene, n=len(times), times=times)
    doc = source.from_trace(trace, task_id=9001, attempt_id=7001, prompt="x", scene_xml="scene.xml",
                            scene_dir=tmp_path / "ep")
    assert doc["trajectory_data"]["step_counts"] == [200, 200, 60, 40, 200, 200, 200]
    p = source.decode_packed_state_v1(doc["trajectory_data"])
    np.testing.assert_allclose(p.row_times, times, atol=1e-9)


def test_a_trace_whose_rows_are_not_whole_simulator_steps_apart_is_refused(tmp_path):
    scene = fixtures.write_scene(tmp_path / "ep")
    trace = fixtures.write_trace(tmp_path / "t.jsonl", scene, n=4, times=[0.0, 0.2, 0.4005, 0.6])
    with pytest.raises(source.SourceError, match="whole number of simulator steps"):
        source.from_trace(trace, task_id=9001, attempt_id=7001, prompt="x", scene_xml="scene.xml",
                          scene_dir=tmp_path / "ep")


def test_the_post_step_trace_still_reads_as_the_post_step_base(tmp_path):
    scene = fixtures.write_scene(tmp_path / "ep")
    doc = source.from_trace(fixtures.write_trace(tmp_path / "t.jsonl", scene, n=4), task_id=9001, attempt_id=7001,
                            prompt="x", scene_xml="scene.xml", scene_dir=tmp_path / "ep")
    assert doc["trajectory_data"]["step_counts"] is None
    assert json.loads(json.dumps(doc))["trajectory_data"]["steps_per_ctrl"] == 200
