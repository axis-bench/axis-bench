"""Steps 2 and 3 without the renderer: the overlay and render calls they compose, and the resumable training view."""
from __future__ import annotations

import json

import pyarrow.parquet as pq

from axis_qvla.cli import main
from axis_qvla.realrobot.layout import ROBOTS, layout
from axis_qvla.realrobot.render import cli as render_cli
from axis_qvla.realrobot.render import droid8d
from axis_qvla.realrobot.steps import render_steps
from tests.unit.realrobot.realworld_fixture import write_realworld
from tests.unit.realrobot.render.test_writer import _episode, _frames, _writer
from tests.unit.realrobot.steps.hosts import qvla_toml


def _setup(tmp_path):
    toml = qvla_toml(tmp_path)
    lay = layout(str(toml))
    write_realworld(lay.data)
    return str(toml), lay


def test_check_camera_prepares_the_data_and_overlays_its_first_scene(tmp_path, monkeypatch):
    toml, lay = _setup(tmp_path)
    seen = []
    monkeypatch.setattr(render_cli, "main", lambda argv: seen.append(argv) or 0)
    assert main(["realrobot", "check-camera", "--image", "ext.png", "--joints", *["0.1"] * 7, "--host", toml]) == 0
    (argv,) = seen
    assert (lay.source / "manifest.json").is_file()
    assert argv[0] == "overlay" and argv[argv.index("--robots") + 1] == str(ROBOTS)
    assert argv[argv.index("--scene") + 1] == str(lay.source / "tasks" / "task_9001" / "scenes" / "variant_0.xml")
    assert argv[argv.index("--out") + 1] == "overlay.png" and argv[-7:] == ["0.1"] * 7


def _render_one_task(render):
    w = _writer(render)
    w.add_episode(_episode(6, 7001), _frames(6, 1))
    w.close()


def test_rerender_renders_into_the_layout_and_writes_its_training_view(tmp_path, monkeypatch, capsys):
    toml, lay = _setup(tmp_path)
    seen = []

    def fake_render(argv):
        seen.append(argv)
        if not (lay.render / "task_9001").is_dir():
            _render_one_task(lay.render)
        return 0

    monkeypatch.setattr(render_cli, "main", fake_render)
    assert main(["realrobot", "rerender", "--profile", "my_workspace.toml", "--host", toml]) == 0
    argv = seen[0]
    assert argv[:3] == ["render", "--data", str(lay.source)]
    assert argv[argv.index("--out") + 1] == str(lay.render) and argv[argv.index("--profile") + 1] == "my_workspace.toml"
    assert json.loads((lay.d8 / "roots.json").read_text())["roots"] == {"9001": "task_9001"}
    assert main(["realrobot", "rerender", "--host", toml]) == 0                  # resumes
    assert "training view: 0 tasks written, 1 kept" in capsys.readouterr().out


def test_the_resumable_training_view_is_droid8d_build_corpus(tmp_path):
    _render_one_task(tmp_path / "render")
    (tmp_path / "d8" / "task_9001.partial").mkdir(parents=True)       # an interrupted earlier write
    assert render_steps.training_view(tmp_path / "render", tmp_path / "d8") == (1, 0)
    droid8d.build_corpus(tmp_path / "render", tmp_path / "ref")
    got = sorted(p.relative_to(tmp_path / "d8") for p in (tmp_path / "d8").rglob("*"))
    ref = sorted(p.relative_to(tmp_path / "ref") for p in (tmp_path / "ref").rglob("*"))
    assert got == ref
    for rel in ref:
        a, b = tmp_path / "d8" / rel, tmp_path / "ref" / rel
        if rel.suffix == ".parquet":
            assert pq.read_table(a).equals(pq.read_table(b)), rel
        elif rel.suffix == ".json" and a.is_file():
            assert json.loads(a.read_text()) == json.loads(b.read_text()), rel
