"""Eq. (1)'s position-delta and directed-rotation progress on synthetic traces, and the per-axis
bounds a position-delta leaf without them takes from the task's checker configuration -- or a named
error when the configuration has none."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from axis_qvla.labels import reward
from tests.unit.labels.test_reward import write_trace

ROOT = pytest.mark.skipif(os.geteuid() == 0, reason="root reads a file whatever its mode, so it cannot be "
                                                   "made unreadable here")
CONTACT = 1.0 - np.tanh(5.0 * 0.05)                     # the hand rides 5 cm above the cup
BOUNDS = [{"axis": "x", "min": 0.1, "max": None}]       # lift-style one-sided bound: x moves >= 10 cm
PD = {"goal_type": "position_delta", "tracked_body_name": "cup_0", "goal_xyz": [0.3, 0.0, 0.0], "pass_frame": 19,
      "n_pass_frames": 6, "checker_type": "PositionDeltaChecker", "initial_object_position": [0.0, 0.0, 0.0],
      "delta_thresh": 0.05, "delta_axes": ["x"], "path": "0"}
CONFIG = {"checker": {"type": "PositionDeltaChecker", "sampleBodyName": "cup_0", "axes": ["x"], "minDeltaX": 0.1}}


def _x(n=25):
    """The cup's x per step in write_trace: 0 -> 0.3 m over the first n - 6 steps, then still."""
    return np.array([0.3 * min(i, n - 6) / (n - 6) for i in range(n)])


def _survey(tmp_path, config=CONFIG):
    """A trace in `traces` under tmp_path, with the task's checker configuration in `data` beside it
    (the layout the reward reads), unless config is None."""
    (tmp_path / "traces").mkdir()
    if config is not None:
        (tmp_path / "data" / "task_7").mkdir(parents=True)
        (tmp_path / "data" / "task_7" / "checker_config.json").write_text(json.dumps(config))
    return write_trace(tmp_path / "traces" / "task_7_attempt_1.jsonl")


def test_position_delta_progress_is_per_axis_toward_the_bounds(tmp_path):
    rewards, _hand, n = reward.compute_attempt_reward_curve({**PD, "axis_bounds": BOUNDS},
                                                            write_trace(tmp_path / "task_1_attempt_2.jsonl"))
    progress = np.clip(_x()[:20] / 0.1, 0.0, 1.0)
    np.testing.assert_allclose(rewards["progress"], progress, rtol=0, atol=1e-12)
    np.testing.assert_allclose(rewards["total"], 0.8 * progress + 0.2 * CONTACT, rtol=0, atol=1e-12)


def test_the_reward_dicts_carry_only_the_channels_a_label_reads(tmp_path):
    trace = write_trace(tmp_path / "task_1_attempt_2.jsonl")
    pnp = {"goal_type": "pnp", "tracked_body_name": "cup_0", "goal_xyz": [0.3, 0.0, 0.0], "pass_frame": 19}
    for leaf in (pnp, {**PD, "axis_bounds": BOUNDS}):
        rewards, _hand, _n = reward.compute_attempt_reward_curve(leaf, trace)
        assert sorted(rewards) == ["contact", "placed", "progress", "total"]


def test_bounds_missing_from_the_leaf_come_from_the_checker_configuration(tmp_path):
    trace = _survey(tmp_path)
    assert reward._maybe_recover_axis_bounds(PD, trace) == BOUNDS
    assert reward._maybe_recover_axis_bounds({**PD, "axis_bounds": BOUNDS}, trace) is None
    recovered, _, _ = reward.compute_attempt_reward_curve(PD, trace)
    stated, _, _ = reward.compute_attempt_reward_curve({**PD, "axis_bounds": BOUNDS}, trace)
    np.testing.assert_array_equal(recovered["total"], stated["total"])


@pytest.mark.parametrize("config,match", [
    (None, "checker_config.json: does not exist"),
    ("{", "checker_config.json: is not a JSON checker configuration"),
    ({"checker": {**CONFIG["checker"], "sampleBodyName": "plate_0"}},
     "checker_config.json: holds no position-delta checker with an x/y/z bound for 'cup_0'"),
])
def test_bounds_the_configuration_cannot_supply_are_a_named_error(tmp_path, config, match):
    trace = _survey(tmp_path, config=None if config is None else {})
    if config is not None:
        cfg = tmp_path / "data" / "task_7" / "checker_config.json"
        cfg.write_text(config if isinstance(config, str) else json.dumps(config))
    with pytest.raises(reward.AxisBoundsError, match=match):
        reward.compute_attempt_reward_curve(PD, trace)


CHECKER = {"type": "PositionDeltaChecker", "sampleBodyName": "cup_0", "axes": ["x"]}
UNUSABLE = {
    # A matching checker whose bounds the reward cannot use: each would otherwise reach the reward's own
    # ValueError ("no usable min/max", or float()) and be labelled "no reward".
    "no bound on its axes": ({"checker": CHECKER},
                             r"the position-delta checker for 'cup_0' \(goal leaf '0'\) has no numeric min or max "
                             r"bound on axes \['x'\]"),
    "snake_case bound keys": ({"checker": {**CHECKER, "min_delta_x": 0.1}},
                              r"spells its bounds \['min_delta_x'\], which the reward does not read"),
    "a bound that is not a number": ({"checker": {**CHECKER, "minDeltaX": "0.1m"}},
                                     r"\(goal leaf '0'\): its minDeltaX = '0.1m' is not a number"),
    "a list bound": ({"checker": {**CHECKER, "maxDeltaX": [0.1]}}, r"its maxDeltaX = \[0.1\] is not a number"),
    # float() accepts these, and a NaN bound scores the axis's progress as complete on every frame.
    "a NaN bound": ({"checker": {**CHECKER, "minDeltaX": float("nan")}},
                    r"\(goal leaf '0'\): its minDeltaX = nan is not a finite number"),
    "an infinite bound": ({"checker": {**CHECKER, "minDeltaX": 0.1, "maxDeltaX": float("-inf")}},
                          r"its maxDeltaX = -inf is not a finite number"),
    "an infinite string bound": ({"checker": {**CHECKER, "minDeltaX": "Infinity"}},
                                 r"its minDeltaX = 'Infinity' is not a finite number"),
    "a top-level list": ([CONFIG["checker"]], r"is a JSON list, not a checker configuration object"),
}


@pytest.mark.parametrize("case", sorted(UNUSABLE))
def test_a_matching_but_unusable_configuration_is_a_named_error_never_no_reward(tmp_path, case):
    from axis_qvla.labels import pipeline
    from tests.unit.labels.test_pipeline import GOALS

    config, match = UNUSABLE[case]
    trace = _survey(tmp_path, config=config)
    with pytest.raises(reward.AxisBoundsError, match=r"checker_config.json: .*" + match):
        reward.compute_attempt_reward_curve(PD, trace)
    with pytest.raises(reward.AxisBoundsError, match=match):
        pipeline.label_episode(pipeline.trace_states(trace), {**GOALS, "leaves": [PD]}, n_rows=80, trace_path=trace)


def test_a_numeric_string_bound_reads_as_float_does():
    """The recovery converts with float(), unchanged: a numeric string is its number."""
    assert reward._axis_bounds_from_checker_cfg({"checker": {**CHECKER, "minDeltaX": "0.1"}}, "cup_0", None) == BOUNDS


@ROOT
def test_an_unreadable_configuration_is_a_named_error(tmp_path):
    trace = _survey(tmp_path)
    cfg = tmp_path / "data" / "task_7" / "checker_config.json"
    os.chmod(cfg, 0)
    try:
        with pytest.raises(reward.AxisBoundsError, match=r"checker_config.json: cannot be read \(Permission denied\)"):
            reward.compute_attempt_reward_curve(PD, trace)
    finally:
        os.chmod(cfg, 0o644)


def test_a_trace_that_does_not_name_its_task_cannot_find_the_bounds(tmp_path):
    with pytest.raises(reward.AxisBoundsError, match="not named task_<id>_attempt_<id>.jsonl"):
        reward.compute_attempt_reward_curve(PD, write_trace(tmp_path / "recording.jsonl"))


def test_the_bounds_error_is_not_an_unformable_reward():
    from axis_qvla.labels.pipeline import UNFORMABLE_REWARD

    assert not issubclass(reward.AxisBoundsError, UNFORMABLE_REWARD)


def write_rotation_trace(path, n=25):
    """`cup_0` turns about z from 0 to 90 degrees over n - 6 steps at the origin, then holds; the hand
    rides 5 cm above it."""
    with open(path, "w") as fh:
        for i in range(n):
            half = np.radians(90.0 * min(i, n - 6) / (n - 6)) / 2
            fh.write(json.dumps({"event": "state", "sim_time": 0.2 * (i + 1),
                                 "body_pose": {"franka/panda_hand": {"pos": [0, 0, 0.05], "quat": [1, 0, 0, 0]},
                                               "cup_0": {"pos": [0, 0, 0],
                                                         "quat": [float(np.cos(half)), 0, 0, float(np.sin(half))]}},
                                 "joint_qpos": {}}) + "\n")
    return path


def test_directed_rotation_progress_is_the_angle_toward_the_target_within_the_tilt_limit(tmp_path):
    leaf = {"goal_type": "directed_rotation", "tracked_body_name": "cup_0", "goal_xyz": [0.0, 0.0, 0.0],
            "pass_frame": 19, "initial_object_quat_wxyz": [1.0, 0.0, 0.0, 0.0], "target_angle_deg": 90.0,
            "angle_tol_deg": 10.0, "max_tilt_deg": 15.0}
    rewards, hand, n = reward.compute_attempt_reward_curve(leaf, write_rotation_trace(tmp_path / "task_1_attempt_2.jsonl"))
    angle = np.array([90.0 * min(i, 19) / 19 for i in range(20)])
    assert hand == "franka/panda_hand" and n == 25
    assert sorted(rewards) == ["contact", "placed", "progress", "total"]
    np.testing.assert_allclose(rewards["progress"], angle / 90.0, rtol=0, atol=1e-6)    # no tilt: no penalty
    np.testing.assert_allclose(rewards["total"], 0.8 * rewards["progress"] + 0.2 * CONTACT, rtol=0, atol=1e-12)
    assert rewards["placed"].tolist() == [float(a >= 80.0) for a in angle]
