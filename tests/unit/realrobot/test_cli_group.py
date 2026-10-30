"""`qvla realrobot <tool> ...` forwards to the tool's own CLI through the core dispatcher."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from axis_qvla.realrobot import cli

REPO = Path(__file__).resolve().parents[3]


def test_the_group_module_is_stdlib_only():
    tree = ast.parse((REPO / "axis_qvla" / "realrobot" / "cli.py").read_text())
    names = {a.name.split(".")[0] for n in tree.body if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module.split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom) and n.module}
    assert names <= set(sys.stdlib_module_names) | {"__future__"}


def test_qvla_realrobot_dispatches_through_func(monkeypatch):
    from axis_qvla.cli import main
    seen = {}

    def fake(module, argv):
        seen[module] = list(argv)
        return 0

    monkeypatch.setattr(cli, "_forward", fake)
    for tool in cli.TOOLS:
        assert main(["realrobot", tool, "sub", "--x", "1"]) == 0     # every tool's CLI starts with a subcommand
    assert seen == {module: ["sub", "--x", "1"] for module in cli.TOOLS.values()}


def test_the_render_tool_answers_help_through_qvla(capsys):
    from axis_qvla.cli import main
    with pytest.raises(SystemExit) as exc:
        main(["realrobot", "render", "gates", "--help"])
    assert exc.value.code == 0 and "--attempt-curves" in capsys.readouterr().out


def test_a_bare_tool_help_reaches_the_tool(capsys):
    from axis_qvla.cli import main
    with pytest.raises(SystemExit) as exc:
        main(["realrobot", "render", "--help"])
    out = capsys.readouterr().out
    assert exc.value.code == 0 and "{source,prepare,render,droid8d,labels,gates,report,overlay}" in out


def test_the_fr3_tool_is_in_the_group_and_answers_help_through_qvla(capsys):
    from axis_qvla.cli import main
    assert set(cli.TOOLS) == {"render", "realign", "realdata", "fr3"}
    with pytest.raises(SystemExit) as exc:
        main(["realrobot", "fr3", "--help"])
    assert exc.value.code == 0 and "{check,norm,local-runs,serve-config}" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(["realrobot", "fr3", "check", "--help"])
    assert exc.value.code == 0 and "--handshake" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(["realrobot", "fr3", "serve-config", "--help"])
    out = capsys.readouterr().out
    assert exc.value.code == 0 and "--run RUN" in out and "[--out OUT]" in out    # --out now defaults from qvla.toml
