from __future__ import annotations

import numpy as np
import pytest

from axis_qvla.realrobot.render import assemble, gripper, rig
from tests.unit.realrobot.render import fixtures


@pytest.fixture
def scene(tmp_path):
    return assemble.build_target_scene(fixtures.write_scene(tmp_path / "s"), fixtures.write_gripper(tmp_path / "g"))


def test_settled_extremes_are_open_at_ctrl_0_and_closed_at_255(scene):
    ext = gripper.settle_extremes(scene, np.asarray(rig.PIXELMATCH_HOME))
    assert (ext.open_ctrl, ext.closed_ctrl) == (0.0, 255.0)
    assert ext.open_separation == pytest.approx(fixtures.OPEN_SEP, abs=1e-4)
    assert ext.closed_separation == pytest.approx(fixtures.CLOSED_SEP, abs=1e-4)
    assert gripper.check_extremes(ext, open_separation=fixtures.OPEN_SEP, closed_separation=fixtures.CLOSED_SEP,
                                  tol=1e-4) == []
    assert gripper.check_extremes(ext, open_separation=rig.GRIPPER_OPEN_SEPARATION_M,
                                  closed_separation=rig.GRIPPER_CLOSED_SEPARATION_M, tol=1e-4) != []


def test_settling_restores_gravity(scene):
    g = scene.model.opt.gravity.copy()
    gripper.settle_extremes(scene, np.asarray(rig.PIXELMATCH_HOME), steps=10)
    np.testing.assert_array_equal(scene.model.opt.gravity, g)


def test_openness_interpolates_linkage_and_ctrl_linearly(scene):
    ext = gripper.settle_extremes(scene, np.asarray(rig.PIXELMATCH_HOME))
    q, c = gripper.gripper_knots(ext, np.array([1.0, 0.5, 0.0]))
    np.testing.assert_allclose(c, [0.0, 127.5, 255.0])
    np.testing.assert_allclose(q[1], 0.5 * (ext.open_qpos + ext.closed_qpos))
