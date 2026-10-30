"""The overlay renders the profile's table and the posed robot, and outlines both on a camera frame."""
from __future__ import annotations

import numpy as np

from axis_qvla.realrobot.render import cli, overlay, profile, rig
from tests.unit.realrobot.render import fixtures


def test_a_render_of_the_profile_overlays_itself_exactly(tmp_path):
    scene, grip = fixtures.write_scene(tmp_path / "s"), fixtures.write_gripper(tmp_path / "g")
    rgb, masks = overlay.render_view(scene, grip, profile.load_profile(), rig.PIXELMATCH_HOME)
    assert masks["table"].sum() > 10_000 and masks["robot"].sum() > 1_000 and rgb.std() > 1.0
    overlay.write_png(tmp_path / "frame.png", rgb)
    np.testing.assert_array_equal(overlay.fit(overlay.read_image(tmp_path / "frame.png")), rgb)


def test_the_overlay_command_writes_the_composite(tmp_path, monkeypatch):
    from axis_qvla.realrobot.render import assets
    scene, grip = fixtures.write_scene(tmp_path / "s"), fixtures.write_gripper(tmp_path / "g")
    monkeypatch.setattr(assets, "check_menagerie", lambda _root: grip)
    rgb, _ = overlay.render_view(scene, grip, profile.load_profile(), rig.PIXELMATCH_HOME)
    overlay.write_png(tmp_path / "frame.png", rgb)
    assert cli.main(["overlay", "--image", str(tmp_path / "frame.png"), "--scene", str(scene), "--robots",
                     str(tmp_path), "--out", str(tmp_path / "o.png")]) == 0
    assert overlay.read_image(tmp_path / "o.png").shape == (rig.HEIGHT, 2 * rig.WIDTH, 3)
