from __future__ import annotations

import json

import pytest

from axis_qvla.realrobot.render import assets


def _menagerie(root):
    (root / "robotiq_2f85" / "assets").mkdir(parents=True)
    (root / "robotiq_2f85" / "2f85.xml").write_text("<mujoco/>")
    (root / "robotiq_2f85" / "assets" / "base.stl").write_bytes(b"solid x")
    return root


def test_a_locked_checkout_passes_and_names_the_gripper_mjcf(tmp_path):
    root = _menagerie(tmp_path / "m")
    lock = assets.build_lock(root, "0" * 40)
    assert set(lock["files"]) == {"robotiq_2f85/2f85.xml", "robotiq_2f85/assets/base.stl"}
    assert assets.check_menagerie(root, lock) == root / "robotiq_2f85" / "2f85.xml"


def test_an_edited_extra_or_missing_file_is_refused(tmp_path):
    root = _menagerie(tmp_path / "m")
    lock = assets.build_lock(root, "0" * 40)
    (root / "robotiq_2f85" / "2f85.xml").write_text("<mujoco model='x'/>")
    with pytest.raises(assets.AssetError, match="sha256 differs"):
        assets.check_menagerie(root, lock)
    (root / "robotiq_2f85" / "extra.xml").write_text("x")
    with pytest.raises(assets.AssetError, match="unlocked"):
        assets.check_menagerie(root, json.loads(json.dumps(lock)))


def test_a_directory_without_the_gripper_is_refused(tmp_path):
    with pytest.raises(assets.AssetError, match="not a Menagerie checkout"):
        assets.check_menagerie(tmp_path, {"commit": "0" * 40, "files": {}})
