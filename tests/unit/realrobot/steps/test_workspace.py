"""Step 1: a workspace profile from the reference one and the camera's OpenCV calibration."""
from __future__ import annotations

import math
import tomllib

import pytest

from axis_qvla.cli import main
from axis_qvla.realrobot.render import profile
from axis_qvla.realrobot.steps import render_steps

REF = tomllib.loads(profile.REFERENCE.read_text(encoding="utf-8"))["camera"]


def _matrix(q):
    w, x, y, z = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def _opencv_extrinsic():
    """The reference camera as solvePnP would give it: x_cam = R x_base + t, OpenCV camera axes."""
    m = _matrix(REF["quat"])                                       # MuJoCo camera axes in the base frame
    cam_to_base = [[m[i][0], -m[i][1], -m[i][2]] for i in range(3)]   # OpenCV camera axes
    r = [[cam_to_base[j][i] for j in range(3)] for i in range(3)]
    t = [-sum(r[i][j] * REF["pos"][j] for j in range(3)) for i in range(3)]
    return [v for row in r for v in row], t


def _fy():
    return 720 / (2 * math.tan(math.radians(REF["fovy"]) / 2))


def test_the_reference_camera_round_trips_through_an_opencv_extrinsic():
    r, t = _opencv_extrinsic()
    pos, quat = render_steps.camera_pose_from_opencv(r, t)
    assert pos == pytest.approx(REF["pos"], abs=1e-12) and quat == pytest.approx(REF["quat"], abs=1e-12)
    assert render_steps.fovy_from_focal(_fy(), 720) == pytest.approx(REF["fovy"], abs=1e-9)


def test_workspace_writes_a_loadable_profile_and_keeps_an_existing_one(tmp_path, capsys):
    out = tmp_path / "my_workspace.toml"
    r, t = _opencv_extrinsic()
    argv = ["realrobot", "workspace", str(out), "--base-to-camera", *map(repr, r + t), "--fy", repr(_fy())]
    assert main(argv) == 0
    doc = tomllib.loads(out.read_text(encoding="utf-8"))
    assert doc["id"] == "my_workspace" and doc["camera"]["pos"] == pytest.approx(REF["pos"], abs=1e-12)
    assert doc["camera"]["fovy"] == pytest.approx(REF["fovy"], abs=1e-9)
    assert profile.load_profile(out).id == "my_workspace"
    assert "workspace my_workspace ->" in capsys.readouterr().out
    assert main(argv) == 2                                          # exists: refused without --force
    assert main(argv + ["--force"]) == 0


def test_workspace_without_calibration_is_the_reference_camera(tmp_path):
    out = tmp_path / "lab.toml"
    assert main(["realrobot", "workspace", str(out), "--id", "lab_a"]) == 0
    doc = tomllib.loads(out.read_text(encoding="utf-8"))
    assert doc["id"] == "lab_a" and doc["camera"] == REF
