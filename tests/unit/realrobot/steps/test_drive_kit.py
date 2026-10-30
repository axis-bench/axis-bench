"""The drive kit beside a run: the client, its request module, a reference exterior frame, the instructions."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from axis_qvla.cli import main
from axis_qvla.realrobot.fr3.line import load_line
from axis_qvla.realrobot.fr3.local_runs import write_local_runs
from axis_qvla.realrobot.layout import LINE
from axis_qvla.realrobot.steps.drive_kit import write_drive_kit
from tests.unit.realrobot.steps.hosts import local_layout

ONLY = "pi05.finetune.realrobot_only"


def test_the_kit_holds_what_the_robot_computer_needs(tmp_path):
    _toml, lay = local_layout(tmp_path)
    kit = write_drive_kit(lay, ONLY)
    assert kit == lay.drive_kit(ONLY)
    assert sorted(p.name for p in kit.iterdir()) == ["drive.json", "droid_client.py", "droid_drive.py", "reference_exterior.png"]
    ref = np.asarray(Image.open(kit / "reference_exterior.png"))
    assert ref.shape == (180, 320, 3) and ref.dtype == np.uint8
    doc = json.loads((kit / "drive.json").read_text())
    assert doc == {"run": ONLY, "instructions": ["press the button", "stack the blocks"], "reference": "reference_exterior.png"}


def test_the_kit_runs_without_this_package(tmp_path):
    _toml, lay = local_layout(tmp_path)
    kit = write_drive_kit(lay, ONLY)
    got = subprocess.run([sys.executable, str(kit / "droid_drive.py"), "--help"], cwd=tmp_path,
                         env={"PATH": os.environ["PATH"]}, capture_output=True, text=True, check=False)
    assert got.returncode == 0 and "--remote-host" in got.stdout, got.stderr


def test_serve_print_writes_the_kit(tmp_path, capsys):
    toml, lay = local_layout(tmp_path)
    write_local_runs(lay.paths, load_line(LINE), runs=[ONLY])
    assert main(["realrobot", "serve", "only", "--host", toml, "--print"]) == 0
    assert f"drive kit -> {lay.drive_kit(ONLY)}" in capsys.readouterr().out


def test_a_re_serve_keeps_an_edited_kit_file_and_says_so(tmp_path, capsys):
    _toml, lay = local_layout(tmp_path)
    kit = write_drive_kit(lay, ONLY)
    edited = (kit / "drive.json").read_text()[:-1] + '  "edited": true}\n'
    (kit / "drive.json").write_text(edited)
    capsys.readouterr()

    got = write_drive_kit(lay, ONLY)

    assert got == kit
    assert (kit / "drive.json").read_text() == edited
    out = capsys.readouterr().out
    assert f"kept {kit / 'drive.json'}" in out and "--replace-kit" in out
    # the other three files are unchanged copies, so they are neither kept-and-printed nor replaced
    for name in ("droid_drive.py", "droid_client.py", "reference_exterior.png"):
        assert name not in out


def test_replace_kit_overwrites_an_edited_kit_file(tmp_path, capsys):
    _toml, lay = local_layout(tmp_path)
    kit = write_drive_kit(lay, ONLY)
    edited = (kit / "drive.json").read_text()[:-1] + '  "edited": true}\n'
    (kit / "drive.json").write_text(edited)
    capsys.readouterr()

    got = write_drive_kit(lay, ONLY, replace=True)

    assert got == kit
    doc = json.loads((kit / "drive.json").read_text())
    assert doc == {"run": ONLY, "instructions": ["press the button", "stack the blocks"], "reference": "reference_exterior.png"}
    assert f"replaced {kit / 'drive.json'}" in capsys.readouterr().out


def test_a_re_serve_over_an_unedited_kit_prints_nothing(tmp_path, capsys):
    _toml, lay = local_layout(tmp_path)
    write_drive_kit(lay, ONLY)
    capsys.readouterr()

    write_drive_kit(lay, ONLY)

    assert capsys.readouterr().out == ""


def test_serve_replace_kit_flag_overwrites_the_edit(tmp_path, capsys):
    toml, lay = local_layout(tmp_path)
    write_local_runs(lay.paths, load_line(LINE), runs=[ONLY])
    kit = write_drive_kit(lay, ONLY)
    edited = (kit / "drive.json").read_text()[:-1] + '  "edited": true}\n'
    (kit / "drive.json").write_text(edited)
    capsys.readouterr()

    assert main(["realrobot", "serve", "only", "--host", toml, "--print", "--replace-kit"]) == 0

    out = capsys.readouterr().out
    assert f"replaced {kit / 'drive.json'}" in out
    doc = json.loads((kit / "drive.json").read_text())
    assert doc == {"run": ONLY, "instructions": ["press the button", "stack the blocks"], "reference": "reference_exterior.png"}


def test_an_empty_roots_json_fails_with_a_clear_message(tmp_path):
    _toml, lay = local_layout(tmp_path)
    (lay.demos / "roots.json").write_text(json.dumps({"version": 1, "base": ".", "roots": {}}))
    with pytest.raises(ValueError, match="roots.json"):
        write_drive_kit(lay, ONLY)


def test_a_missing_camera_fails_with_a_clear_message(tmp_path, monkeypatch):
    _toml, lay = local_layout(tmp_path)
    import axis_qvla.realrobot.steps.drive_kit as drive_kit_module

    monkeypatch.setattr(drive_kit_module, "CAMERA", "observation.images.no_such_camera")
    with pytest.raises(ValueError, match="no_such_camera"):
        write_drive_kit(lay, ONLY)
