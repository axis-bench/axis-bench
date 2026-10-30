"""Eq. (1) on a synthetic pick-and-place trace: 0.8 * progress + 0.2 * (1 - tanh(5 * hand-object distance))."""
from __future__ import annotations

import json

import numpy as np
import pytest

from axis_qvla.labels import reward


def write_trace(path, n=25, offset=0.05):
    """`cup_0` moves from the origin to (0.3, 0, 0) over n steps at 5 Hz; the hand rides `offset` above it."""
    with open(path, "w") as fh:
        for i in range(n):
            x = 0.3 * min(i, n - 6) / (n - 6)
            fh.write(json.dumps({"event": "state", "sim_time": 0.2 * (i + 1),
                                 "body_pose": {"world": {"pos": [0, 0, 0], "quat": [1, 0, 0, 0]},
                                               "franka/panda_hand": {"pos": [x, 0, offset], "quat": [1, 0, 0, 0]},
                                               "cup_0": {"pos": [x, 0, 0], "quat": [1, 0, 0, 0]}},
                                 "joint_qpos": {"franka/panda_finger_joint1": 0.01, "franka/panda_finger_joint2": 0.01}}) + "\n")
        fh.write(json.dumps({"event": "final", "final_sim_time": 0.2 * n}) + "\n")
    return path


LEAF = {"goal_type": "pnp", "tracked_body_name": "cup_0", "goal_xyz": [0.3, 0.0, 0.0], "pass_frame": 19,
        "n_pass_frames": 6, "checker_type": "RelativePositionBoundsChecker"}


def test_the_reward_is_eq1_on_the_trace_clipped_at_success(tmp_path):
    rewards, hand, n = reward.compute_attempt_reward_curve(LEAF, write_trace(tmp_path / "task_1_attempt_2.jsonl"))
    assert hand == "franka/panda_hand" and n == 25 and len(rewards["total"]) == 20
    np.testing.assert_allclose(rewards["total"], 0.8 * rewards["progress"] + 0.2 * rewards["contact"])
    np.testing.assert_allclose(rewards["contact"], 1.0 - np.tanh(5.0 * 0.05))
    assert rewards["progress"][0] == 0.0 and rewards["progress"][19] == 1.0


def test_no_hold_tail_and_no_hold_bonus_are_constants_not_environment(monkeypatch):
    monkeypatch.setenv("CLIP_TAIL_FRAMES", "7")
    monkeypatch.setenv("HOLD_STABILITY_BONUS", "1.0")
    import importlib
    again = importlib.reload(reward)
    assert (again.CLIP_TAIL_FRAMES, again.HOLD_STABILITY_BONUS) == (0, 0.0)


def test_an_unknown_goal_family_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown goal_type"):
        reward.compute_attempt_reward_curve({**LEAF, "goal_type": "teleport"}, write_trace(tmp_path / "t.jsonl"))
