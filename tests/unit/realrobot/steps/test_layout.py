"""The real-world layout is the one `fr3 local-runs` reads, and the raw tools default their paths to it."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from axis_qvla.cli import main
from axis_qvla.realrobot import layout as layout_mod
from axis_qvla.realrobot.fr3 import cli as fr3_cli
from axis_qvla.realrobot.fr3.line import load_line
from axis_qvla.realrobot.fr3.local_runs import write_local_runs
from axis_qvla.realrobot.layout import LINE, ROBOTS, layout, run_id
from axis_qvla.realrobot.render import cli as render_cli
from tests.unit.realrobot.steps.hosts import local_layout, qvla_toml

REPO = Path(__file__).resolve().parents[4]


def test_the_layout_is_the_one_local_runs_reads(tmp_path):
    _toml, lay = local_layout(tmp_path)
    cache, runs = lay.paths.cache, lay.paths.runs
    assert lay.d8 == cache / "derived" / "realworld" / "d8" and lay.bundles == cache / "bundles"
    assert lay.norm_file("pretrain") == cache / "derived/norm/norm_realalign_sim/qvla-v1-realworld/norm_stats.json"
    assert lay.norm_file("finetune") == cache / "derived/norm/norm_fr3_droid/fr3_demonstrations/norm_stats.json"
    assert lay.norm_file("pretrain").is_file() and lay.norm_file("finetune").is_file()
    assert lay.checkpoint("x") == runs / "x" / "train" / "final" and lay.serve_config("x") == runs / "x" / "serve.json"
    assert lay.data == lay.paths.corpus / "qvla-v1-realworld"
    assert ROBOTS == REPO / "third_party" / "axis-render" / "assets" / "robots"


def test_short_run_names():
    assert run_id("finetune", "cond") == "pi05.finetune.realrobot_cond"
    assert run_id("pretrain", "pi05.pretrain.realrobot_bc") == "pi05.pretrain.realrobot_bc"


def test_fr3_norm_writes_where_local_runs_reads_without_out(tmp_path):
    toml, lay = local_layout(tmp_path)
    lay.norm_file("finetune").unlink()
    assert fr3_cli.main(["norm", "--dataset", str(lay.demos), "--embodiment", "fr3_droid", "--host", toml]) == 0
    assert lay.norm_file("finetune").is_file()
    with pytest.raises(SystemExit) as exc:
        fr3_cli.main(["norm", "--dataset", str(lay.demos), "--embodiment", "fr3", "--host", toml])
    assert exc.value.code == 2


def test_serve_config_defaults_beside_the_run(tmp_path, capsys):
    toml, lay = local_layout(tmp_path)
    rid = "pi05.finetune.realrobot_only"
    spec = write_local_runs(lay.paths, load_line(LINE), runs=[rid])[rid]
    assert fr3_cli.main(["serve-config", "--run", str(spec), "--host", toml]) == 0
    assert lay.serve_config(rid).is_file()
    assert f"serve config of {rid} -> {lay.serve_config(rid)}" in capsys.readouterr().out


def test_render_and_overlay_default_robots(monkeypatch):
    seen = {}
    monkeypatch.setattr(render_cli, "_overlay", lambda a: seen.setdefault("overlay", a.robots) and 0)
    monkeypatch.setattr(render_cli, "_render", lambda a: seen.setdefault("render", a.robots) and 0)
    render_cli.main(["overlay", "--image", "x.png", "--scene", "s.xml", "--out", "o.png"])
    render_cli.main(["render", "--episodes", "e.json", "--out", "r"])
    assert seen == {"overlay": ROBOTS, "render": ROBOTS}
    seen.clear()
    render_cli.main(["render", "--episodes", "e.json", "--gripper-xml", "g.xml", "--out", "r"])
    assert seen == {"render": None}            # left to _render's own refusal of a gripper without --unlocked-gripper


def test_prepare_defaults_to_the_layout(tmp_path):
    from tests.unit.realrobot.realworld_fixture import write_realworld
    toml = qvla_toml(tmp_path)
    lay = layout(str(toml))
    write_realworld(lay.data)
    assert render_cli.main(["prepare", "--host", str(toml)]) == 0
    assert (lay.source / "manifest.json").is_file()


def test_the_step_parsers_are_stdlib_only_and_registered(capsys):
    tree = ast.parse((REPO / "axis_qvla" / "realrobot" / "steps" / "parsers.py").read_text())
    names = {a.name.split(".")[0] for n in tree.body if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module.split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom) and n.module}
    assert names <= set(sys.stdlib_module_names) | {"__future__"}
    with pytest.raises(SystemExit) as exc:
        main(["realrobot", "workspace", "--help"])
    assert exc.value.code == 0 and "--base-to-camera" in capsys.readouterr().out
