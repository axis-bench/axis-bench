from __future__ import annotations

import base64
import gzip
import json

import mujoco
import numpy as np
import pytest

from axis_qvla.realrobot.render import rig, source
from tests.unit.realrobot.render import fixtures


def test_packed_state_round_trips_through_float32(tmp_path):
    scene = fixtures.write_scene(tmp_path)
    p = fixtures.fixture_packed(scene)
    q = source.decode_packed_state_v1(source.encode_packed_state_v1(p, task_id=9001))
    np.testing.assert_array_equal(q.qpos, p.qpos.astype(np.float32).astype(np.float64))
    assert (q.n_steps, q.nq, q.nv, q.nu, q.row_dt) == (12, 16, 15, 8, 0.001 * 200)


def test_rows_are_after_step_states_on_the_published_time_base():
    p = source.PackedState(0.001, 200, 81, 1, 1, 0, 0.0, np.zeros(1), np.zeros(1), None,
                           np.zeros((81, 1)), np.zeros((81, 1)), np.zeros((81, 0)))
    t = source.knot_times(p)
    assert t[0] == 0.2 * 1 and len(t) == 81 and abs(t[-1] - 16.2) < 1e-12


@pytest.mark.parametrize("edit, msg", [
    (lambda d: d.update(format="packed_state_v2"), "packed_state_v1"),
    (lambda d: d.update(step_counts=[1] * 5), "step_counts"),
    (lambda d: d.update(n_steps=13), "float32 values"),
    (lambda d: d.update(steps_per_ctrl=None), "steps_per_ctrl"),
])
def test_malformed_trajectories_are_refused(tmp_path, edit, msg):
    doc = source.encode_packed_state_v1(fixtures.fixture_packed(fixtures.write_scene(tmp_path)))
    edit(doc)
    with pytest.raises(source.SourceError, match=msg):
        source.decode_packed_state_v1(doc)


def test_an_attempt_record_loses_every_private_field_and_needs_its_joint_order(tmp_path):
    scene = fixtures.write_scene(tmp_path)
    traj = source.encode_packed_state_v1(fixtures.fixture_packed(scene), task_id=9001)
    record = {"attempt": {"id": 7001, "task_id": 9001, "user_id": 12345, "user_stage": "new", "is_completed": True},
              "task": {"id": 9001, "name": "push the red box", "description": "x", "mjcf_xml": "<mujoco/>"},
              "task_trajectory": {"trajectory_data": traj,
                                  "metadata": {"user_id": 12345, "submit_nonce": "1"},
                                  "trajectory_s3_bucket": "b", "trajectory_s3_key": "k"}}
    order = source.joint_names(mujoco.MjModel.from_xml_path(str(scene)))
    doc = source.from_attempt_record(record, scene_xml="scene.xml", joint_order=order)
    text = json.dumps(doc)
    for field in source.PRIVATE_FIELDS:
        assert field not in text
    assert (doc["task_id"], doc["attempt_id"], doc["prompt"], doc["joint_order"]) == (9001, 7001, "push the red box", order)
    traj["initial_snapshot"]["userdata"] = [1.0]
    traj["initial_snapshot"]["qacc_warmstart"] = [0.0] * 15
    doc = source.from_attempt_record(record, scene_xml="scene.xml", joint_order=order)
    assert sorted(doc["trajectory_data"]["initial_snapshot"]) == ["ctrl", "qpos", "qvel", "time"]
    with pytest.raises(TypeError):
        source.from_attempt_record(record, scene_xml="scene.xml")       # no positional guess, ever


def test_an_attempt_record_s_payload_loses_its_recording_time(tmp_path):
    """The recorded payload's gzip header holds the wall-clock time and OS it was written on."""
    traj = source.encode_packed_state_v1(fixtures.fixture_packed(fixtures.write_scene(tmp_path)), task_id=9001)
    raw = gzip.decompress(base64.b64decode(traj["data_b64_gzip"]))
    stamped = bytearray(gzip.compress(raw, mtime=1781007302))
    stamped[9] = 3                                                       # written on Unix
    traj["data_b64_gzip"] = base64.b64encode(bytes(stamped)).decode("ascii")
    record = {"attempt": {"id": 7001, "task_id": 9001}, "task": {"id": 9001, "name": "push the red box"},
              "task_trajectory": {"trajectory_data": traj}}
    doc = source.from_attempt_record(record, scene_xml="scene.xml", joint_order=["j"])
    out = base64.b64decode(doc["trajectory_data"]["data_b64_gzip"])
    assert out[:4] == b"\x1f\x8b\x08\x00" and out[4:8] == b"\x00\x00\x00\x00" and out[9] == 255
    assert gzip.decompress(out) == raw


def _permuted(names):
    """A recorded order with the object joint before the arm's."""
    return [names[-1], *names[:-1]]


def test_a_recorded_order_that_differs_from_the_scene_is_realigned_by_name(tmp_path):
    scene = fixtures.write_scene(tmp_path / "ep")
    m = mujoco.MjModel.from_xml_path(str(scene))
    p = fixtures.fixture_packed(scene)
    names = source.joint_names(m)
    rec = _permuted(names)
    cols = np.concatenate([np.arange(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)],
                                     m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                                     + (7 if n == "box_1_free" else 1)) for n in rec])
    vcols = np.concatenate([np.arange(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)],
                                      m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                                      + (6 if n == "box_1_free" else 1)) for n in rec])
    shuffled = source.PackedState(p.timestep, p.steps_per_ctrl, p.n_steps, p.nq, p.nv, p.nu, p.initial_time,
                                  p.initial_qpos[cols], p.initial_qvel[vcols], p.initial_ctrl,
                                  p.qpos[:, cols], p.qvel[:, vcols], p.ctrl)
    doc = source.source_episode_doc(task_id=9001, attempt_id=7001, prompt="x", scene_xml="scene.xml",
                                    trajectory_data=source.encode_packed_state_v1(shuffled), joint_order=rec)
    source.write_source_episode(tmp_path / "ep" / "source_episode.json", doc)
    got = source.read_source_episode(tmp_path / "ep" / "source_episode.json").packed
    np.testing.assert_array_equal(got.qpos, p.qpos.astype(np.float32).astype(np.float64))


def test_a_trace_is_read_by_joint_name_whatever_its_key_order(tmp_path):
    scene = fixtures.write_scene(tmp_path / "ep")
    m = mujoco.MjModel.from_xml_path(str(scene))
    trace = fixtures.write_trace(tmp_path / "trace.jsonl", scene, order=_permuted(source.joint_names(m)))
    doc = source.from_trace(trace, task_id=9001, attempt_id=7001, prompt="x", scene_xml="scene.xml",
                            scene_dir=tmp_path / "ep")
    source.write_source_episode(tmp_path / "ep" / "source_episode.json", doc)
    ep = source.read_source_episode(tmp_path / "ep" / "source_episode.json")
    p = fixtures.fixture_packed(scene)
    np.testing.assert_allclose(ep.packed.qpos, p.qpos, atol=1e-6)
    assert np.isnan(ep.packed.qvel).all()        # traces carry no velocities: never invented
    np.testing.assert_allclose(source.knot_times(ep.packed), (np.arange(12) + 1) * 0.2, atol=1e-12)


@pytest.mark.parametrize("edit, msg", [
    (lambda names: names[:-1], "missing"),
    (lambda names: [*names, "ghost_free"], "not in the scene"),
])
def test_a_joint_order_that_is_not_the_scenes_joint_set_is_refused(tmp_path, edit, msg):
    scene = fixtures.write_scene(tmp_path)
    m = mujoco.MjModel.from_xml_path(str(scene))
    with pytest.raises(source.SourceError, match=msg):
        source.align_to_scene(fixtures.fixture_packed(scene), edit(source.joint_names(m)), m)


def test_the_scene_path_must_stay_inside_the_episode_directory(tmp_path):
    path = fixtures.write_episode(tmp_path / "ep")
    doc = json.loads(path.read_text())
    doc["scene_xml"] = "../elsewhere/scene.xml"
    path.write_text(json.dumps(doc))
    with pytest.raises(source.SourceError, match="relative path inside"):
        source.read_source_episode(path)


def test_the_source_tcp_is_the_hand_plus_0_1084_lifted_onto_the_table(tmp_path):
    ep = source.read_source_episode(fixtures.write_episode(tmp_path / "ep"))
    m = mujoco.MjModel.from_xml_path(str(ep.scene_xml))
    s = source.source_stream(m, ep.packed, 0.48)
    d = mujoco.MjData(m)
    hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "franka/panda_hand")
    d.qpos[:] = ep.packed.qpos[5]
    mujoco.mj_kinematics(m, d)
    want = d.xpos[hand] + d.xmat[hand].reshape(3, 3) @ np.asarray(rig.SOURCE_TCP_IN_HAND) + [0, 0, 0.48]
    np.testing.assert_allclose(s.tcp[5], want, atol=1e-6)
    np.testing.assert_allclose(s.base_pos, [0.0, 0.0, 0.48], atol=1e-12)


def test_openness_is_the_mean_finger_position_over_the_joint_range(tmp_path):
    ep = source.read_source_episode(fixtures.write_episode(tmp_path / "ep"))
    s = source.source_stream(mujoco.MjModel.from_xml_path(str(ep.scene_xml)), ep.packed, 0.48)
    assert s.openness[0] == pytest.approx(1.0) and s.openness[-1] == pytest.approx(0.5)


def test_a_trace_that_lacks_a_scene_joint_is_refused_naming_it(tmp_path):
    scene = fixtures.write_scene(tmp_path / "ep")
    names = source.joint_names(mujoco.MjModel.from_xml_path(str(scene)))
    trace = fixtures.write_trace(tmp_path / "trace.jsonl", scene, order=names[:-1])
    with pytest.raises(source.SourceError, match=f"lacks the scene's joints \\['{names[-1]}'\\]"):
        source.from_trace(trace, task_id=9001, attempt_id=7001, prompt="x", scene_xml="scene.xml",
                          scene_dir=tmp_path / "ep")


def test_the_source_command_refuses_an_out_outside_the_scene_dir(tmp_path, capsys):
    """The written scene_xml is relative to the episode file's directory, so an episode file written
    anywhere but --scene-dir names a scene that does not resolve; it is refused before anything is written."""
    from axis_qvla.realrobot.render.cli import main
    scene = fixtures.write_scene(tmp_path / "ep")
    trace = fixtures.write_trace(tmp_path / "trace.jsonl", scene)
    argv = ["source", "--trace", str(trace), "--scene-dir", str(tmp_path / "ep"), "--scene-xml", "scene.xml",
            "--task-id", "9001", "--attempt-id", "7001", "--prompt", "x"]
    assert main([*argv, "--out", str(tmp_path / "elsewhere" / "source_episode.json")]) == 2
    assert "--scene-dir" in capsys.readouterr().err and not (tmp_path / "elsewhere").exists()
    assert main([*argv, "--out", str(tmp_path / "ep" / "source_episode.json")]) == 0
    assert source.read_source_episode(tmp_path / "ep" / "source_episode.json").attempt_id == 7001
