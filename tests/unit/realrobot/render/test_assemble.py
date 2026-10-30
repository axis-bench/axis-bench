from __future__ import annotations

import mujoco
import numpy as np
import pytest

from axis_qvla.realrobot.render import assemble, profile, rig
from tests.unit.realrobot.render import fixtures


@pytest.fixture
def scene(tmp_path):
    return assemble.build_target_scene(fixtures.write_scene(tmp_path / "s"), fixtures.write_gripper(tmp_path / "g"),
                                       key=(9001, 7001))


def _name(m, kind, i):
    return mujoco.mj_id2name(m, kind, i) or ""


def test_the_source_hand_actuators_lights_cameras_and_floor_are_gone(scene):
    m = scene.model
    bodies = {_name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)}
    assert not {"franka/panda_hand", "franka/panda_leftfinger"} & bodies
    assert [_name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)] == ["robotiq/fingers_actuator"]
    assert [_name(m, mujoco.mjtObj.mjOBJ_LIGHT, i) for i in range(m.nlight)] == ["qvla_light"]
    assert sorted(_name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)) == ["frontview", "wrist"]
    assert "floor" not in {_name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)}


def test_the_2f85_base_sits_where_the_published_render_put_it(scene):
    m, d = scene.model, mujoco.MjData(scene.model)
    d.qpos[scene.arm_qpos] = rig.PIXELMATCH_HOME
    mujoco.mj_kinematics(m, d)
    R7 = d.xmat[scene.ee_body].reshape(3, 3)
    np.testing.assert_allclose(R7.T @ (d.xpos[scene.base_body] - d.xpos[scene.ee_body]), [0, 0, 0.1178], atol=1e-9)
    q7inv = np.zeros(4)
    mujoco.mju_negQuat(q7inv, d.xquat[scene.ee_body])
    rel = np.zeros(4)
    mujoco.mju_mulQuat(rel, q7inv, d.xquat[scene.base_body])
    assert abs(abs(float(np.dot(rel, rig.GRIPPER_BASE_IN_EE.quat))) - 1.0) < 1e-9


def test_the_world_is_lifted_onto_the_table_top(scene):
    m, d = scene.model, mujoco.MjData(scene.model)
    mujoco.mj_kinematics(m, d)
    link0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "franka/panda_link0")
    assert d.xpos[link0][2] == pytest.approx(profile.load_profile().table.height)
    table = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "qvla_table")
    assert m.geom_pos[table][2] + m.geom_size[table][2] == pytest.approx(profile.load_profile().table.height)
    (free,) = [s for s in scene.shared if s[6]]
    assert m.jnt_type[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "box_1_free")] == mujoco.mjtJoint.mjJNT_FREE
    assert free[2] == 7


def test_one_directional_light_without_shadow_and_the_headlight(scene):
    m = scene.model
    assert m.light_castshadow[0] == 0 and m.light_type[0] == mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    np.testing.assert_allclose(m.light_ambient[0], rig.LIGHT.ambient)
    np.testing.assert_allclose(m.vis.headlight.ambient, rig.HEADLIGHT.ambient)
    assert (m.vis.global_.offwidth, m.vis.global_.offheight) == (rig.WIDTH, rig.HEIGHT)


def test_cameras_are_the_rig(scene):
    m = scene.model
    for cam in scene.camera_specs:
        i = scene.cameras[cam.key]
        assert m.cam_fovy[i] == pytest.approx(cam.fovy)
        np.testing.assert_allclose(m.cam_pos[i], cam.pos, atol=1e-12)


def test_the_ik_model_has_no_objects_and_no_arena(tmp_path):
    s = assemble.build_target_scene(fixtures.write_scene(tmp_path / "s"), fixtures.write_gripper(tmp_path / "g"),
                                    objects=False)
    assert s.shared == () and mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_GEOM, "qvla_table") < 0


def test_the_ik_model_drops_the_meshes_of_the_objects_it_drops(tmp_path):
    """A recorded object's mesh with inertia="convex", left in the spec with no geom once its body is
    deleted, crashes MuJoCo 3.3.7's compiler (task 1338's variant 23, lipstick_4): the IK model deletes
    the meshes of the bodies it deletes, and keeps every other mesh."""
    scene = fixtures.write_scene(tmp_path / "s")
    text = scene.read_text()
    mesh = '<mesh name="box_1_vis" inertia="convex" vertex="0 0 0  0.02 0 0  0 0.02 0  0 0 0.02"/>'
    text = text.replace("<worldbody>", f"<asset>{mesh}</asset>\n  <worldbody>", 1)
    text = text.replace('<freejoint name="box_1_free"/>',
                        '<freejoint name="box_1_free"/><geom type="mesh" mesh="box_1_vis" group="1" contype="0" conaffinity="0"/>', 1)
    scene.write_text(text)
    grip = fixtures.write_gripper(tmp_path / "g")
    ik_model = assemble.build_target_scene(scene, grip, objects=False).model
    assert mujoco.mj_name2id(ik_model, mujoco.mjtObj.mjOBJ_MESH, "box_1_vis") < 0
    render = assemble.build_target_scene(scene, grip).model
    assert mujoco.mj_name2id(render, mujoco.mjtObj.mjOBJ_MESH, "box_1_vis") >= 0
    assert ik_model.nmesh == render.nmesh - 1


def test_surface_jitter_is_deterministic_per_attempt(tmp_path):
    a = assemble.build_target_scene(fixtures.write_scene(tmp_path / "s"), fixtures.write_gripper(tmp_path / "g"), key=(1, 2))
    b = assemble.build_target_scene(fixtures.write_scene(tmp_path / "s"), fixtures.write_gripper(tmp_path / "g"), key=(1, 2))
    c = assemble.build_target_scene(fixtures.write_scene(tmp_path / "s"), fixtures.write_gripper(tmp_path / "g"), key=(1, 3))
    np.testing.assert_array_equal(a.model.mat_rgba, b.model.mat_rgba)
    assert not np.array_equal(a.model.mat_rgba, c.model.mat_rgba)
    i = mujoco.mj_name2id(a.model, mujoco.mjtObj.mjOBJ_MATERIAL, "qvla_table")
    assert a.model.mat_specular[i] <= profile.load_profile().table.surface.material_caps[2]
