from __future__ import annotations

import numpy as np

from axis_qvla.realrobot.render import assemble, draw, profile, rig
from tests.unit.realrobot.render import fixtures


def test_poses_without_gl_give_the_rig_frontview_and_a_moving_wrist(tmp_path):
    scene = assemble.build_target_scene(fixtures.write_scene(tmp_path / "s"), fixtures.write_gripper(tmp_path / "g"))
    q = np.tile(scene.model.qpos0, (3, 1))
    q[:, scene.arm_qpos] = rig.PIXELMATCH_HOME
    q[2, scene.arm_qpos[0]] += 0.3
    c2w, ee = draw.poses(scene, q)
    front, cam = c2w["third_person"].reshape(3, 4, 4), profile.load_profile().camera
    np.testing.assert_allclose(front[:, :3, 3], np.tile(cam.pos, (3, 1)), atol=1e-6)
    np.testing.assert_allclose(front[0, :3, :3], rig.quat_to_mat(cam.quat), atol=1e-6)
    wrist = c2w["wrist"].reshape(3, 4, 4)
    assert not np.allclose(wrist[0], wrist[2]) and np.allclose(wrist[0], wrist[1])
    assert ee.shape == (3, 7) and ee.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(ee[:, 3:], axis=1), 1.0, atol=1e-6)
