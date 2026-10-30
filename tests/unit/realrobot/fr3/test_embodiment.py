"""The embodiment declarations, read back, and the FR3 one against the recording's declared layouts."""
from __future__ import annotations

import pytest

from axis_qvla.realrobot.fr3.embodiment import EMBODIMENTS, EmbodimentError, load_embodiment
from axis_qvla.realrobot.realdata import state_order


def test_fr3_is_the_recordings_declared_layout_and_units():
    e = load_embodiment("fr3")
    assert (e.state_stored, e.state_model, e.action_stored) == (
        state_order.STORED_STATE, state_order.MODEL_STATE, state_order.STORED_ACTIONS)
    u = state_order.ACTION_UNITS
    assert (e.action_units, e.action_range, e.fps, e.gripper_units) == (u["arm"], u["arm_range"], u["fps"], u["gripper"])
    assert e.image_keys == ("image", "extra_view_image") and e.image_stored == (224, 224)
    assert e.geometry_default is None      # measured from each set's stored pixels, never declared


def test_the_rendered_view_is_in_model_order_in_rad_s():
    e = load_embodiment("realalign_sim")
    assert e.state_stored == e.state_model and e.action_units == "joint_velocity_rad_s_clipped_1"
    assert (e.fps, e.geometry_default, e.image_stored) == (15, "center_crop224", (640, 360))


def test_your_demonstrations_command_the_robot_in_the_fr3_recordings_units():
    mine, fr3 = load_embodiment("fr3_droid"), load_embodiment("fr3")
    assert (mine.action_units, mine.action_range, mine.gripper_units, mine.fps) == (
        fr3.action_units, fr3.action_range, fr3.gripper_units, fr3.fps)


def test_a_declaration_of_another_schema_or_a_non_permutation_is_refused(tmp_path):
    src = EMBODIMENTS / "fr3.toml"
    (tmp_path / "a.toml").write_text(src.read_text().replace("qvla.embodiment/1", "qvla.embodiment/0"))
    with pytest.raises(EmbodimentError, match="schema"):
        load_embodiment(tmp_path / "a.toml")
    (tmp_path / "b.toml").write_text(src.read_text().replace('model = ["q0"', 'model = ["q7"'))
    with pytest.raises(EmbodimentError, match="permutation"):
        load_embodiment(tmp_path / "b.toml")
    with pytest.raises(EmbodimentError, match="cannot read"):
        load_embodiment("no_such_robot")
