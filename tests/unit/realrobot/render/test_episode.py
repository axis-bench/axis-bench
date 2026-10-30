from __future__ import annotations

import dataclasses

import numpy as np

from axis_qvla.labels.resample import trace_anchor_rows
from axis_qvla.realrobot.render import assemble, episode, gripper, rig, source
from tests.unit.realrobot.render import fixtures


def _setup(tmp_path, n=12):
    ep = source.read_source_episode(fixtures.write_episode(tmp_path / "ep", n=n))
    scene = assemble.build_target_scene(ep.scene_xml, fixtures.write_gripper(tmp_path / "g"), key=(9001, 7001))
    import mujoco
    stream = source.source_stream(mujoco.MjModel.from_xml_path(str(ep.scene_xml)), ep.packed, 0.48)
    arm = np.tile(np.asarray(rig.PIXELMATCH_HOME), (n, 1))
    arm[1:, 0] += np.linspace(0.0, 0.3, n - 1)
    ext = gripper.settle_extremes(scene, arm[0], steps=200)
    return ep, scene, stream, arm, ext


def test_frames_are_three_per_knot_and_the_first_three_hold_knot_0(tmp_path):
    ep, scene, stream, arm, ext = _setup(tmp_path)
    st = episode.compose(scene, ep.packed, arm, ext, stream.openness, 0.48)
    assert st.qpos.shape == (36, scene.model.nq) and st.action.shape == (36, 8)
    np.testing.assert_array_equal(st.knot_rows, 3 * np.arange(12) + 2)
    np.testing.assert_array_equal(st.qpos[:3, scene.arm_qpos], np.tile(arm[0], (3, 1)))
    np.testing.assert_allclose(st.qpos[st.knot_rows][:, scene.arm_qpos], arm, atol=1e-12)


def test_post_step_rows_0_04_s_apart_compose_on_the_label_frames(tmp_path):
    ep, scene, stream, arm, ext = _setup(tmp_path)
    packed = dataclasses.replace(ep.packed, steps_per_ctrl=40)          # 0.001 s x 40
    st = episode.compose(scene, packed, arm, ext, stream.openness, 0.48)
    assert np.all(np.isfinite(st.qpos)) and np.all(np.isfinite(st.action))
    assert st.knot_rows.tolist() == trace_anchor_rows(packed.row_times, 15.0).tolist() and len(st.grid_t) == 10
    before = st.grid_t <= packed.row_times[0]
    assert before[:2].all()
    np.testing.assert_array_equal(st.qpos[before][:, scene.arm_qpos], np.tile(arm[0], (int(before.sum()), 1)))


def test_actions_are_the_causal_zoh_of_knot_joints_and_255_times_one_minus_openness(tmp_path):
    ep, scene, stream, arm, ext = _setup(tmp_path)
    st = episode.compose(scene, ep.packed, arm, ext, stream.openness, 0.48)
    k = 5
    r = st.knot_rows[k]
    # frames strictly between knots; which knot a frame exactly ON a knot time holds depends on float
    # rounding of the two time bases, as in the published render
    np.testing.assert_array_equal(st.action[r + 1:r + 3, :7], np.tile(arm[k], (2, 1)))
    np.testing.assert_allclose(st.action[r + 1:r + 3, 7], 255.0 * (1.0 - stream.openness[k]))


def test_free_objects_are_lifted_onto_the_table(tmp_path):
    ep, scene, stream, arm, ext = _setup(tmp_path)
    qk, _vk, _ck = episode.knot_states(scene, ep.packed, arm, ext, stream.openness, 0.48)
    (src_q, tgt_q, nq, *_rest), = [s for s in scene.shared if s[6]]
    np.testing.assert_allclose(qk[:, tgt_q + 2], ep.packed.qpos[:, src_q + 2] + 0.48)
