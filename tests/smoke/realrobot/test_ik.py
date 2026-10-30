"""Render tier: runs in the render venv (axis_qvla/realrobot/render/render.lock.txt), i.e. in the `render` CI job;
the `test` job ignores tests/smoke/realrobot. Nothing here skips."""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from axis_qvla.realrobot.render import assemble, ik, rig, source
from tests.unit.realrobot.render import fixtures


@pytest.fixture
def solved(tmp_path):
    ep = source.read_source_episode(fixtures.write_episode(tmp_path / "ep", n=12))
    stream = source.source_stream(mujoco.MjModel.from_xml_path(str(ep.scene_xml)), ep.packed, 0.48)
    scene = assemble.build_target_scene(ep.scene_xml, fixtures.write_gripper(tmp_path / "g"), objects=False)
    return scene, stream, ik.solve_knots(scene, stream)


def test_knot_0_is_the_pixelmatch_home(solved):
    _, _, r = solved
    np.testing.assert_array_equal(r.arm_q[0], rig.PIXELMATCH_HOME)


def test_the_tcp_converges_early_and_stays_within_3_mm(solved):
    _, _, r = solved
    k = r.convergence_knot
    assert k is not None and 1 <= k <= rig.CONVERGE_WITHIN_KNOTS
    assert np.all(r.perr[k:] <= rig.TCP_GATE_M) and r.reach_pct >= rig.IK.min_reach_pct


def test_the_gripper_base_takes_the_source_hand_orientation(solved):
    scene, stream, r = solved
    m, d = scene.model, mujoco.MjData(scene.model)
    for k in range(r.convergence_knot, len(r.arm_q)):
        d.qpos[scene.arm_qpos] = r.arm_q[k]
        mujoco.mj_kinematics(m, d)
        R = d.xmat[scene.base_body].reshape(3, 3)
        ang = np.degrees(np.arccos(np.clip((np.trace(R.T @ stream.hand_R[k]) - 1) / 2, -1, 1)))
        assert ang < 1.0, (k, ang)


def test_the_solve_is_deterministic(solved, tmp_path):
    scene, stream, r = solved
    again = ik.solve_knots(scene, stream)
    np.testing.assert_array_equal(r.arm_q, again.arm_q)


def test_an_unreachable_path_is_refused(solved):
    scene, stream, _ = solved
    far = source.SourceStream(stream.tcp + np.array([3.0, 0.0, 0.0]), stream.hand_R, stream.openness, stream.base_pos)
    with pytest.raises(ik.IkError, match="reaches"):
        ik.solve_knots(scene, far)
