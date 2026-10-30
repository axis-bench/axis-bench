"""The three subcommands, end to end on a synthetic FR3 dataset."""
from __future__ import annotations

import json

import pytest

from axis_qvla.realrobot.realdata.cli import main
from tests.unit.realrobot.realdata.test_export import _world


def test_idle_ranges_writes_the_segments(tmp_path, capsys):
    ds = _world(tmp_path)
    assert main(["idle-ranges", "--dataset", str(ds.root), "--out", str(tmp_path / "seg.json")]) == 0
    prov = json.loads((tmp_path / "seg.json").read_text())
    assert prov["source"] == "rec_250" and len(prov["episodes"]) == 3
    assert "3 episodes / 160 frames -> 3 episodes / 112 frames" in capsys.readouterr().out
    assert not (tmp_path / "export").exists()


def test_idle_ranges_can_export_and_override_the_guard(tmp_path):
    ds = _world(tmp_path)
    assert main(["idle-ranges", "--dataset", str(ds.root), "--out", str(tmp_path / "seg.json"),
                 "--export", str(tmp_path / "export"), "--grip-guard", "0"]) == 0
    prov = json.loads((tmp_path / "export" / "meta" / "source_segments.json").read_text())
    assert "grip_guard=0" in prov["criterion"] and len(prov["episodes"]) == 4    # ep 2 is cut without the guard


def test_tail_writes_openpis_filter_file(tmp_path, capsys):
    ds = _world(tmp_path)
    assert main(["tail", "--dataset", str(ds.root), "--out", str(tmp_path / "f" / "tail10.json")]) == 0
    assert (tmp_path / "f" / "tail10.json").read_text() == '{"0": [[0, 50]], "1": [[0, 30]], "2": [[0, 50]]}'
    assert "3 episodes, 130 chunk starts" in capsys.readouterr().out


def test_a_subcommand_is_required():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


# --- refusals: one FAIL line, exit 2, nothing written ----------------------------------------------

def _with_video_feature(ds):
    info = json.loads((ds.root / "meta" / "info.json").read_text())
    info["features"]["observation.images.cam"] = {"dtype": "video", "shape": [180, 320, 3], "names": None}
    (ds.root / "meta" / "info.json").write_text(json.dumps(info, indent=4))


def test_export_of_a_video_copy_is_refused_before_anything_is_written(tmp_path, capsys):
    ds = _world(tmp_path)
    _with_video_feature(ds)
    argv = ["idle-ranges", "--dataset", str(ds.root), "--out", str(tmp_path / "seg.json"),
            "--export", str(tmp_path / "export")]
    assert main(argv) == 2
    err = capsys.readouterr().err
    assert err.startswith("FAIL  ") and "observation.images.cam" in err and "image-in-parquet" in err
    assert not (tmp_path / "seg.json").exists() and not (tmp_path / "export").exists()
    # Without --export the segments are still computed from the video copy's parquet.
    assert main(argv[:5]) == 0 and (tmp_path / "seg.json").exists()


def test_an_existing_export_is_refused_before_the_segments_are_written(tmp_path, capsys):
    ds = _world(tmp_path)
    (tmp_path / "export").mkdir()
    assert main(["idle-ranges", "--dataset", str(ds.root), "--out", str(tmp_path / "seg.json"),
                 "--export", str(tmp_path / "export")]) == 2
    assert "exists" in capsys.readouterr().err and not (tmp_path / "seg.json").exists()


@pytest.mark.parametrize("cmd", ["idle-ranges", "tail"])
def test_a_directory_that_is_not_a_dataset_is_refused_by_name(tmp_path, capsys, cmd):
    (tmp_path / "not_a_dataset").mkdir()
    assert main([cmd, "--dataset", str(tmp_path / "not_a_dataset"), "--out", str(tmp_path / "o.json")]) == 2
    err = capsys.readouterr().err
    assert err.startswith("FAIL  ") and "not_a_dataset" in err and "LeRobot v2.1 dataset" in err
    assert not (tmp_path / "o.json").exists()
