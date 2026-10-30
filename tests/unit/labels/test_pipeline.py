"""One episode's per-frame labels from its trace and goal record, resampled onto the 15 fps render rows."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from axis_qvla.labels import goals, pipeline, resample, reward
from tests.unit.labels.test_reward import LEAF, write_trace

GOALS = {"n_steps": 25, "top_first_pass_frame": 19, "leaves": [LEAF], "attempt_id": 2}


def test_labels_are_the_trace_reward_resampled_with_its_validity_and_success(tmp_path):
    trace = write_trace(tmp_path / "task_1_attempt_2.jsonl")
    states = pipeline.trace_states(trace)
    cols = pipeline.label_episode(states, GOALS, n_rows=80, trace_path=trace)
    anchors = resample.trace_anchor_rows([s["sim_time"] for s in states], pipeline.RENDER_FPS)
    curve, target = pipeline.reward_curve(GOALS, trace)
    assert target == "cup_0" and curve.shape == (20,)
    np.testing.assert_allclose(cols["reward"][anchors[:20]], curve)
    assert cols["reward_valid"].tolist() == [anchors[0] <= r <= anchors[19] for r in range(80)]
    assert cols["success"].tolist() == [r >= anchors[19] for r in range(80)]
    assert cols["is_anchor"].sum() == len(anchors) and cols["observation.body_pose"].shape == (80, 2, 7)
    assert cols["body_names"] == ["franka/panda_hand", "cup_0"]
    assert "contact_eef" not in cols                                   # no scene given


def test_an_attempt_without_a_reward_leaf_has_no_reward(tmp_path):
    trace = write_trace(tmp_path / "task_1_attempt_2.jsonl")
    cols = pipeline.label_episode(pipeline.trace_states(trace), {**GOALS, "leaves": []}, n_rows=80, trace_path=trace)
    assert not cols["reward"].any() and not cols["reward_valid"].any()


ROOT = pytest.mark.skipif(os.geteuid() == 0, reason="root reads a file whatever its mode, so it cannot be "
                                                   "made unreadable here")


def test_a_missing_trace_raises_and_never_reads_as_no_reward(tmp_path):
    states = pipeline.trace_states(write_trace(tmp_path / "task_1_attempt_2.jsonl"))
    with pytest.raises(FileNotFoundError):
        pipeline.label_episode(states, GOALS, n_rows=80, trace_path=tmp_path / "task_1_attempt_3.jsonl")
    with pytest.raises(FileNotFoundError):
        pipeline.reward_curve(GOALS, tmp_path / "task_1_attempt_3.jsonl")


@ROOT
def test_an_unreadable_trace_raises_and_never_reads_as_no_reward(tmp_path):
    trace = write_trace(tmp_path / "task_1_attempt_2.jsonl")
    states = pipeline.trace_states(trace)
    os.chmod(trace, 0)
    try:
        with pytest.raises(PermissionError):
            pipeline.label_episode(states, GOALS, n_rows=80, trace_path=trace)
        with pytest.raises(PermissionError):
            pipeline.reward_curve(GOALS, trace)
    finally:
        os.chmod(trace, 0o644)


def test_the_pipeline_and_the_goal_extraction_share_one_trace_reader():
    assert pipeline.trace_states is goals.read_trace_states


def test_an_empty_trace_fails_as_the_goal_extraction_does(tmp_path):
    empty = tmp_path / "task_1_attempt_2.jsonl"
    empty.write_text(json.dumps({"event": "final", "final_sim_time": 1.0}) + "\n")
    with pytest.raises(RuntimeError, match="empty trace"):
        goals._load_trace(empty)
    with pytest.raises(RuntimeError, match="empty trace"):
        pipeline.trace_states(empty)
    with pytest.raises(RuntimeError, match="empty trace"):
        pipeline.label_episode([], GOALS, n_rows=80, trace_path=empty)
    with pytest.raises(RuntimeError, match="empty trace"):
        pipeline.reward_curve(GOALS, empty)


def test_an_unformable_reward_is_no_reward(tmp_path):
    """A leaf of an unknown goal family raises ValueError in the reward: one of the named unformable
    cases, labelled reward 0 and reward_valid False, as the corpus labels it."""
    trace = write_trace(tmp_path / "task_1_attempt_2.jsonl")
    unknown = {**GOALS, "leaves": [{**LEAF, "goal_type": "teleport"}]}
    assert ValueError in pipeline.UNFORMABLE_REWARD
    assert pipeline.reward_curve(unknown, trace) == (None, "cup_0")
    cols = pipeline.label_episode(pipeline.trace_states(trace), unknown, n_rows=80, trace_path=trace)
    assert not cols["reward"].any() and not cols["reward_valid"].any() and cols["target_body"] == "cup_0"


def test_bounds_the_checker_configuration_cannot_supply_raise_through_the_pipeline(tmp_path):
    from tests.unit.labels.test_reward_families import PD

    (tmp_path / "traces").mkdir()
    trace = write_trace(tmp_path / "traces" / "task_7_attempt_1.jsonl")      # no data/task_7/checker_config.json
    with pytest.raises(reward.AxisBoundsError, match="does not exist"):
        pipeline.label_episode(pipeline.trace_states(trace), {**GOALS, "leaves": [PD]}, n_rows=80, trace_path=trace)
