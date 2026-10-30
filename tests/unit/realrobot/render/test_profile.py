"""The workspace profile: the reference workspace's camera, table and background, and what a profile
must state."""
from __future__ import annotations

import ast
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from axis_qvla.realrobot.render import profile, rig


def _pipeline():
    """A render needs the render environment's mink (the IK)."""
    pytest.importorskip("mink")
    from axis_qvla.realrobot.render import pipeline
    return pipeline


def test_the_reference_camera_has_fx_fy_250_at_640x360():
    k = np.asarray(rig.intrinsics(profile.load_profile().camera), np.float32)
    np.testing.assert_array_equal(k, np.float32([250.0, 0.0, 319.5, 0.0, 250.0, 179.5, 0.0, 0.0, 1.0]))


def test_the_reference_camera_looks_from_its_eye_to_the_table_in_the_base_frame():
    p = profile.load_profile()
    lift = np.array([0.0, 0.0, p.table.height])
    eye, target = np.array([1.0, -0.13, 0.55]) + lift, np.array([0.35, -0.13, 0.33]) + lift
    assert p.camera.pos == (1.0, -0.13, 1.03)
    np.testing.assert_allclose(rig.lookat_quat(eye, target), p.camera.quat, atol=1e-7)


def test_the_reference_camera_axes_are_the_published_camera_to_world():
    published = np.array([[5.5511151e-17, -3.2059619e-01, 9.4721597e-01],
                          [1.0, 1.6653345e-16, 0.0],
                          [-1.6653345e-16, 9.4721597e-01, 3.2059619e-01]])
    np.testing.assert_allclose(rig.quat_to_mat(profile.load_profile().camera.quat), published, atol=1e-6)


def test_the_cameras_are_the_profiles_exterior_and_the_wrist_on_the_gripper():
    p = profile.load_profile()
    assert [c.key for c in p.cameras] == list(rig.CAMERA_KEYS) and p.cameras[1] is rig.WRIST
    assert [name for name, _ in p.surfaces()] == ["table", "floor", "wall"]


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "ws.toml"
    path.write_text(text)
    return path


def test_a_profile_names_the_field_it_lacks(tmp_path):
    text = profile.REFERENCE.read_text().replace("height = 0.48 ", "")
    with pytest.raises(profile.ProfileError, match="'height'"):
        profile.load_profile(_write(tmp_path, text))


def test_a_vector_of_the_wrong_length_is_refused_by_name(tmp_path):
    text = profile.REFERENCE.read_text().replace("pos = [1.0, -0.13, 0.55]", "pos = [1.0, -0.13]")
    with pytest.raises(profile.ProfileError, match="camera.pos must be 3 numbers"):
        profile.load_profile(_write(tmp_path, text))


def test_a_fixture_naming_an_undefined_material_is_refused(tmp_path):
    text = profile.REFERENCE.read_text().replace('material = "dark"', 'material = "black"', 1)
    with pytest.raises(profile.ProfileError, match=r"\['black'\]"):
        profile.load_profile(_write(tmp_path, text))


@pytest.mark.parametrize("old, new, message", [
    ("height = 0.48 ", 'height = "48cm" ', r"table\.height must be a number, got '48cm'"),
    ("height = 0.48 ", 'height = "0.48" ', r"table\.height must be a number"),
    ("wall_height = 3.0", "wall_height = true", r"background\.wall_height must be a number"),
    ("[table.surface] ", '[table.surface]\ntexure = "wood.png"\n', r"table\.surface has unknown keys \['texure'\]"),
    ("[materials.dark]\nrgba = [0.025, 0.025, 0.025, 1.0]", "[materials]\ndark = [0.025, 0.025, 0.025, 1.0]",
     r"materials\.dark must be a table"),
])
def test_a_mistyped_or_misspelled_field_is_refused_by_name(tmp_path, old, new, message):
    text = profile.REFERENCE.read_text()
    assert old in text
    with pytest.raises(profile.ProfileError, match=message):
        profile.load_profile(_write(tmp_path, text.replace(old, new, 1)))


def _with_texture(texture: str, extra: str = "") -> str:
    return profile.REFERENCE.read_text().replace("[table.surface] ",
                                                 f'[table.surface]\ntexture = "{texture}"\n{extra}')


@pytest.mark.parametrize("texture, message", [
    ("../wood.png", "must be a path inside the profile's directory"),
    ("{tmp}/wood.png", "must be a path inside the profile's directory"),
    ("task_1000/wood.png", "starts with 'task_1000', a name a render writes"),
    ("roots.json", "starts with 'roots.json', a name a render writes"),
    ("workspace.toml", "starts with 'workspace.toml', a name a render writes"),
    ("render_gates.json/wood.png", "starts with 'render_gates.json', a name a render writes"),
])
def test_a_texture_a_render_cannot_keep_beside_it_is_refused_by_name(tmp_path, texture, message):
    with pytest.raises(profile.ProfileError, match=r"table\.surface\.texture " + message):
        profile.load_profile(_write(tmp_path, _with_texture(texture.format(tmp=tmp_path))))


def _png(path: Path, rgb: np.ndarray) -> Path:
    """A minimal RGB PNG (zlib, no library): the table-top texture of the test."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[r].astype(np.uint8).tobytes() for r in range(h))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return path


def test_a_table_texture_resolves_against_the_profile_and_reaches_the_model(tmp_path):
    from axis_qvla.realrobot.render import assemble
    from tests.unit.realrobot.render import fixtures
    _png(tmp_path / "wood.png", np.full((8, 8, 3), 120))
    p = profile.load_profile(_write(tmp_path, _with_texture("wood.png", "texture_repeat = [2.0, 3.0]\n")))
    assert p.table.surface.texture == (tmp_path / "wood.png").resolve()
    s = assemble.build_target_scene(fixtures.write_scene(tmp_path / "s"), fixtures.write_gripper(tmp_path / "g"),
                                    key=(9001, 7001), profile=p)
    import mujoco
    mat = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_MATERIAL, "qvla_table")
    assert s.model.mat_texid[mat][int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] >= 0
    np.testing.assert_allclose(s.model.mat_texrepeat[mat], [2.0, 3.0])


def test_a_render_directory_keeps_the_one_profile_it_was_rendered_in(tmp_path):
    pipeline = _pipeline()
    out = tmp_path / "out"
    pipeline.render_tasks({}, tmp_path / "gripper.xml", out)
    assert (out / profile.FILE).read_text() == profile.REFERENCE.read_text()
    pipeline.render_tasks({}, tmp_path / "gripper.xml", out)             # the same profile again
    narrow = profile.load_profile(_write(tmp_path, profile.REFERENCE.read_text().replace("fovy = 71.5077745089",
                                                                                         "fovy = 60.0")))
    with pytest.raises(ValueError, match="another workspace profile"):
        pipeline.render_tasks({}, tmp_path / "gripper.xml", out, profile=narrow)


def test_a_kept_profile_carries_its_images_and_a_replaced_image_is_another_profile(tmp_path):
    pipeline = _pipeline()
    _png(tmp_path / "ws" / "img" / "wood.png", np.full((8, 8, 3), 120))
    p = profile.load_profile(_write(tmp_path / "ws", _with_texture("img/wood.png")))
    assert p.images() == (Path("img/wood.png"),)
    out = tmp_path / "out"
    pipeline.render_tasks({}, tmp_path / "gripper.xml", out, profile=p)
    kept = profile.load_profile(out / profile.FILE)
    assert kept.table.surface.texture == (out / "img" / "wood.png").resolve()
    assert kept.table.surface.texture.read_bytes() == (tmp_path / "ws" / "img" / "wood.png").read_bytes()
    pipeline.render_tasks({}, tmp_path / "gripper.xml", out, profile=p)
    _png(tmp_path / "ws" / "img" / "wood.png", np.full((8, 8, 3), 30))   # the same text, another photo
    with pytest.raises(ValueError, match="another workspace profile"):
        pipeline.render_tasks({}, tmp_path / "gripper.xml", out, profile=p)


def test_a_profile_given_by_a_relative_path_compiles_with_its_texture(tmp_path, monkeypatch):
    """MuJoCo resolves a relative texture file against the scene's directory, so the profile's images are
    absolute whatever path the profile was given by."""
    from axis_qvla.realrobot.render import assemble
    from tests.unit.realrobot.render import fixtures
    import mujoco
    _png(tmp_path / "ws" / "wood.png", np.full((8, 8, 3), 120))
    _write(tmp_path / "ws", _with_texture("wood.png"))
    scene, grip = fixtures.write_scene(tmp_path / "s"), fixtures.write_gripper(tmp_path / "g")
    monkeypatch.chdir(tmp_path)
    p = profile.load_profile(Path("ws") / "ws.toml")
    assert p.table.surface.texture == (tmp_path / "ws" / "wood.png").resolve()
    s = assemble.build_target_scene(scene, grip, key=(9001, 7001), profile=p)
    mat = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_MATERIAL, "qvla_table")
    assert s.model.mat_texid[mat][int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] >= 0


def test_a_render_into_the_profiles_own_directory_keeps_its_images_in_place(tmp_path):
    pipeline = _pipeline()
    _png(tmp_path / "ws" / "img" / "wood.png", np.full((8, 8, 3), 120))
    p = profile.load_profile(_write(tmp_path / "ws", _with_texture("img/wood.png")))
    pipeline.render_tasks({}, tmp_path / "gripper.xml", tmp_path / "ws", profile=p)
    assert profile.load_profile(tmp_path / "ws" / profile.FILE).table.surface.texture == p.table.surface.texture


def test_profile_is_stdlib_only():
    tree = ast.parse(Path(profile.__file__).read_text(encoding="utf-8"))
    names = {a.name.split(".")[0] for n in tree.body if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module.split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom) and n.module}
    # The package's own top-level name, read at runtime, so the check holds whatever it is called.
    assert names <= {"__future__", "dataclasses", "tomllib", "pathlib", profile.__name__.split(".")[0]}
