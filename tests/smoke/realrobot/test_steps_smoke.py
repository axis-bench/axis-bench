"""The real-world tutorial's render steps end to end on the stand-in data, through `qvla realrobot`, in
the renderer's environment (MuJoCo, mink, PyAV; MUJOCO_GL=egl or osmesa)."""
from __future__ import annotations

import pytest

from axis_qvla.cli import main
from axis_qvla.realrobot.layout import layout
from tests.unit.realrobot.realworld_fixture import write_realworld
from tests.unit.realrobot.steps.hosts import qvla_toml


@pytest.fixture
def realworld(tmp_path):
    toml = qvla_toml(tmp_path)
    lay = layout(str(toml))
    write_realworld(lay.data)
    return str(toml), lay


def test_check_camera_draws_the_overlay(realworld, tmp_path, capsys):
    from tests.docs.fixtures import workspace_photo
    toml, _lay = realworld
    workspace_photo(tmp_path / "exterior.png")
    assert main(["realrobot", "check-camera", "--image", str(tmp_path / "exterior.png"),
                 "--out", str(tmp_path / "overlay.png"), "--host", toml]) == 0
    assert (tmp_path / "overlay.png").is_file()
    assert "overlay of exterior.png on workspace fr3_reference" in capsys.readouterr().out


def test_rerender_renders_and_resumes(realworld, capsys):
    toml, lay = realworld
    assert main(["realrobot", "rerender", "--host", toml]) == 0
    assert (lay.d8 / "roots.json").is_file()
    assert main(["realrobot", "rerender", "--host", toml]) == 0
    out = capsys.readouterr().out
    assert "rendered tasks [9001]" in out and "training view: 0 tasks written, 1 kept" in out


def test_labels_on_the_stand_in(tmp_path, capsys):
    # Two episodes are fewer drawable rows than one batch of the compiled labels' row sampler, so the
    # stand-in here has six. Its episodes make no contact, so every one is neutral and its quality tag is
    # constant: only bc is labelled.
    toml = str(qvla_toml(tmp_path))
    write_realworld(layout(toml).data, attempts=(7001, 7002, 7003, 7004, 7005, 7006))
    assert main(["realrobot", "rerender", "--host", toml]) == 0
    assert main(["realrobot", "labels", "--runs", "bc", "--host", toml]) == 0
    out = capsys.readouterr().out
    assert "all hard gates pass" in out and "compiled labels realrobot.bc ->" in out
    assert main(["realrobot", "labels", "--runs", "bc", "--host", toml]) == 0      # resumes: phase keeps the task
    assert "0 tasks computed, 1 kept" in capsys.readouterr().out
