"""The pinned rig reproduces the published render's wrist camera, mount and home, number for number."""
from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np

from axis_qvla.realrobot.render import rig


def test_wrist_focal_is_the_published_float32():
    # published observation.cameras.wrist.intrinsics[0] (task 1000, every row)
    assert np.float32(rig.focal(rig.WRIST)) == np.float32("365.86197")


def test_gripper_base_mount_is_0_1178_up_link7_and_minus_135_degrees_about_z():
    pos, quat = rig.GRIPPER_BASE_IN_EE.pos, rig.GRIPPER_BASE_IN_EE.quat
    assert pos == (0.0, 0.0, 0.1178)
    R = rig.quat_to_mat(quat)
    np.testing.assert_allclose(R, [[-math.sqrt(0.5), math.sqrt(0.5), 0], [-math.sqrt(0.5), -math.sqrt(0.5), 0], [0, 0, 1]],
                               atol=1e-12)


def test_quaternion_round_trip():
    q = rig.mat_to_quat(rig.quat_to_mat(rig.WRIST.quat))
    assert np.allclose(q, rig.WRIST.quat, atol=1e-12) or np.allclose(q, -np.asarray(rig.WRIST.quat), atol=1e-12)


def test_home_and_posture_are_inside_panda_limits():
    lo = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973)
    hi = (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973)
    for pose in (rig.PIXELMATCH_HOME, rig.POSTURE_HOME):
        assert len(pose) == 7 and all(a <= x <= b for a, x, b in zip(lo, pose, hi))


def test_timing_constants_are_consistent():
    assert rig.FPS * rig.SOURCE_ROW_DT == 3 and rig.KNOT_ROW_OFFSET == 2


def test_rig_is_stdlib_only():
    tree = ast.parse(Path(rig.__file__).read_text(encoding="utf-8"))
    names = {a.name.split(".")[0] for n in tree.body if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module.split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom) and n.module}
    assert names <= {"__future__", "dataclasses", "math"}
