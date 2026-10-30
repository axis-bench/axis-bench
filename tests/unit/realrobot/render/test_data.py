"""The published real-world data: its layout, the prepared render tree, and a render from it."""
from __future__ import annotations

import json
import os

import pytest

from axis_qvla.realrobot.render import cli, data, source
from tests.unit.realrobot.realworld_fixture import LAYOUT, write_realworld


def _pipeline():
    """The render command imports the pipeline, which needs the render environment's mink."""
    pytest.importorskip("mink")
    from axis_qvla.realrobot.render import pipeline
    return pipeline


def test_prepare_links_the_tasks_and_extracts_their_meshes(tmp_path):
    write_realworld(tmp_path / "pub")
    done = data.prepare(tmp_path / "pub", tmp_path / "w")
    assert done == {"tasks": 1, "episodes": 2, "layouts": 1}
    assert (tmp_path / "w" / data.scene_rel(9001, 0)).is_file()
    assert (tmp_path / "w" / "tabletopgen" / LAYOUT / "box_1" / "README").is_file()
    assert os.stat(tmp_path / "w" / data.trace_rel(9001, 7001)).st_ino == os.stat(tmp_path / "pub" / data.trace_rel(9001, 7001)).st_ino
    assert not (tmp_path / "pub" / "tabletopgen" / LAYOUT).exists()          # the download stays as published


def test_prepare_refuses_a_task_whose_mesh_tar_was_not_fetched(tmp_path):
    write_realworld(tmp_path / "pub")
    (tmp_path / "pub" / data.mesh_tar_rel(LAYOUT)).unlink()
    with pytest.raises(data.DataError, match=f"tabletopgen/{LAYOUT}.tar"):
        data.prepare(tmp_path / "pub", tmp_path / "w")


def test_prepare_names_a_task_file_that_was_not_fetched(tmp_path):
    write_realworld(tmp_path / "pub")
    (tmp_path / "pub" / data.trace_rel(9001, 7002)).unlink()
    with pytest.raises(data.DataError, match=data.trace_rel(9001, 7002)):
        data.prepare(tmp_path / "pub", tmp_path / "w")
    assert not (tmp_path / "w").exists()


def test_prepare_into_the_downloaded_copy_is_refused(tmp_path):
    write_realworld(tmp_path / "pub")
    for out in (tmp_path / "pub", tmp_path / "pub" / "prepared"):
        with pytest.raises(data.DataError, match="inside the downloaded copy"):
            data.prepare(tmp_path / "pub", out)
    assert not (tmp_path / "pub" / "tabletopgen" / LAYOUT).exists() and not (tmp_path / "pub" / "prepared").exists()


def test_a_layout_whose_extraction_was_interrupted_is_extracted_again(tmp_path):
    write_realworld(tmp_path / "pub")
    left = tmp_path / "w" / "tabletopgen" / f".{LAYOUT}.partial" / LAYOUT / "box_1"
    left.mkdir(parents=True)
    data.prepare(tmp_path / "pub", tmp_path / "w")
    assert (tmp_path / "w" / "tabletopgen" / LAYOUT / "box_1" / "README").is_file()
    assert sorted(p.name for p in (tmp_path / "w" / "tabletopgen").iterdir()) == [LAYOUT]


def test_a_task_not_in_the_manifest_is_refused_by_number(tmp_path):
    write_realworld(tmp_path / "pub")
    with pytest.raises(data.DataError, match="1000"):
        data.select(data.load_manifest(tmp_path / "pub"), [1000])


def test_the_episode_files_of_a_selection_are_its_source_episodes(tmp_path):
    m = write_realworld(tmp_path / "pub")
    (tid, files), = data.task_episode_files(tmp_path / "pub", m, [9001]).items()
    assert tid == 9001 and [f.name for f in files] == ["attempt_7001.json", "attempt_7002.json"]
    assert json.loads(files[0].read_text())["scene_xml"] == "variant_0.xml"


def test_the_prepare_command_prints_what_it_did(tmp_path, capsys):
    write_realworld(tmp_path / "pub")
    assert cli.main(["prepare", "--data", str(tmp_path / "pub"), "--out", str(tmp_path / "w")]) == 0
    assert "1 task, 2 episodes, 1 mesh layout -> " in capsys.readouterr().out


def _render_args(tmp_path, tree):
    return ["render", "--data", str(tree), "--gripper-xml", str(tmp_path / "2f85.xml"), "--unlocked-gripper",
            "--out", str(tmp_path / "r")]


def test_a_render_from_the_prepared_tree_reads_its_selected_episodes(tmp_path, monkeypatch):
    pipeline = _pipeline()
    write_realworld(tmp_path / "pub")
    data.prepare(tmp_path / "pub", tmp_path / "w")
    seen = []

    def render_tasks(tasks, gripper_xml, out_root, **_kw):
        seen.extend((tid, e.attempt_id, e.scene_xml) for tid, files in tasks.items()
                    for e in map(source.read_source_episode, files))
        return pipeline.Rendered(sorted(tasks), [], {})

    monkeypatch.setattr(pipeline, "render_tasks", render_tasks)
    assert cli.main(_render_args(tmp_path, tmp_path / "w") + ["--tasks", "9001"]) == 0
    scene = tmp_path / "w" / data.scene_rel(9001, 0)
    assert seen == [(9001, 7001, scene), (9001, 7002, scene)]


def test_a_render_from_the_downloaded_copy_is_refused(tmp_path, capsys):
    _pipeline()
    write_realworld(tmp_path / "pub")
    assert cli.main(_render_args(tmp_path, tmp_path / "pub")) == 2
    assert f"tabletopgen/{LAYOUT}/ is not extracted" in capsys.readouterr().err


@pytest.mark.parametrize("args", [[], ["--episodes", "e.json", "--data", "w"], ["--episodes", "e.json", "--tasks", "1"]])
def test_a_render_reads_episodes_or_the_prepared_tree(tmp_path, args):
    with pytest.raises(SystemExit):
        cli.main(["render", *args, "--unlocked-gripper", "--gripper-xml", "g.xml", "--out", str(tmp_path / "r")])
