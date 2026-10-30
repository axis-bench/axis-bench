"""The declared FR3 state order: [gripper, q0..q6] in the dataset, [q0..q6, gripper] in the model."""
from __future__ import annotations

import numpy as np
import pytest

from axis_qvla.realrobot.realdata.state_order import (JOINTS, MODEL_STATE, MODEL_TO_STORED, STORED_ACTIONS, STORED_STATE,
                                       STORED_TO_MODEL, StateOrder, select)


def test_the_declared_layouts():
    assert STORED_STATE == ("gripper", "q0", "q1", "q2", "q3", "q4", "q5", "q6")
    assert MODEL_STATE == ("q0", "q1", "q2", "q3", "q4", "q5", "q6", "gripper")
    assert STORED_ACTIONS == ("v0", "v1", "v2", "v3", "v4", "v5", "v6", "gripper_cmd")


def test_stored_to_model_moves_the_gripper_last():
    stored = np.array([0.9, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    np.testing.assert_array_equal(STORED_TO_MODEL(stored), [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9])


def test_model_to_stored_moves_it_back_first():
    model = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9])
    np.testing.assert_array_equal(MODEL_TO_STORED(model), [0.9, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


def test_the_round_trip_is_exact_on_batches():
    x = np.random.default_rng(0).normal(size=(5, 3, 8)).astype(np.float32)
    y = STORED_TO_MODEL(x)
    assert y.dtype == np.float32 and y.shape == x.shape
    np.testing.assert_array_equal(MODEL_TO_STORED(y), x)
    assert MODEL_TO_STORED == STORED_TO_MODEL.inverse()


@pytest.mark.parametrize("width", [7, 9])
def test_a_state_of_the_wrong_width_is_refused(width):
    with pytest.raises(ValueError, match="expected"):
        STORED_TO_MODEL(np.zeros((2, width)))


def test_an_order_must_be_a_permutation():
    with pytest.raises(ValueError, match="permutation"):
        StateOrder(("a", "b"), ("a", "c"))
    with pytest.raises(ValueError, match="permutation"):
        StateOrder(("a", "a"), ("a", "a"))


def test_select_reads_named_columns():
    state = np.arange(16.0).reshape(2, 8)
    np.testing.assert_array_equal(select(state, STORED_STATE, JOINTS), state[:, 1:8])
    np.testing.assert_array_equal(select(state, STORED_ACTIONS, ("gripper_cmd",)), state[:, 7:8])
    with pytest.raises(KeyError):
        select(state, STORED_STATE, ("q7",))


def test_action_units_are_declared():
    from axis_qvla.realrobot.realdata.state_order import ACTION_UNITS

    assert ACTION_UNITS["arm"] == "droid_normalized_joint_velocity_command"
    assert ACTION_UNITS["arm_range"] == (-1.0, 1.0) and ACTION_UNITS["fps"] == 15
    assert ACTION_UNITS["gripper"] == "absolute_position_command_0_1"
