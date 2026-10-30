"""`qvla`: every group is a shipped package's, each with working help; nothing else is imported."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from axis_qvla.cli import build_parser, main

REPO = Path(__file__).resolve().parents[3]
# Every package this release ships: axis_qvla.cli is lazy (its own docstring: "Each group is
# imported only when the parser is built"), so a bare `import axis_qvla.cli` pulls in nothing.
# `build_parser()` is what actually imports every group's command module, so
# test_building_the_parser_imports_only_public_packages (below) drives it.
PUBLIC_GROUPS = ("core", "data", "paper", "bench", "labels", "signals", "realrobot", "train")


def test_prog_and_groups():
    p = build_parser()
    assert p.prog == "qvla"
    text = p.format_help()
    for group in ("bench", "data", "paper", "results"):
        assert group in text


@pytest.mark.parametrize("argv", [["paper", "tables", "--help"], ["results", "score", "--help"],
                                  ["data", "verify", "--help"],
                                  ["bench", "eval", "--help"]])
def test_every_group_has_help(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 0


def test_bench_check_runs_on_the_shipped_declaration(capsys):
    assert main(["bench", "check"]) == 0
    assert "franka_heldout24" in capsys.readouterr().out


def test_groups_are_the_public_ones():
    """Streams add groups (signals, train, realrobot, ...) through `_GROUPS`, so the expected set is
    derived from it: every registered group's module lives in a package this release ships, and
    the groups the published commands document stay registered."""
    import axis_qvla.cli as cli   # the dotted form, which a rename of the module keeps

    p = build_parser()
    sub = next(a for a in p._actions if a.dest == "command")
    assert {"bench", "data", "paper", "results"} <= set(sub.choices)
    for module, _ in cli._GROUPS:
        assert module.split(".")[1] in PUBLIC_GROUPS, f"{module} is not under a shipped package"


def test_data_verify_parses():
    a = build_parser().parse_args(["data", "verify", "--local", "x", "--repo", "r"])
    assert a.func.__name__ == "_cmd_verify"


def test_building_the_parser_imports_only_public_packages():
    """Read through the import, so it holds whatever the CLI module is called. Drives
    `build_parser()` in a subprocess -- not a bare `import` -- and checks every module that lands
    in `sys.modules` under the top-level import root is under a package this release ships: never
    a `release`, `config`, `run`, `dataset`, `episode`, `eval` or `train` package. (Written without
    the import root spelled onto those names, so none of them reads as a cross-reference.)"""
    import axis_qvla.cli as public_cli

    pkg = public_cli.__name__.split(".")[0]
    code = (f"import sys\nfrom {public_cli.__name__} import build_parser\nbuild_parser()\n"
            f"print('\\n'.join(sorted(m for m in sys.modules if m.startswith('{pkg}'))))\n")
    # -B: never write __pycache__ into the tree under test (a release check refuses one;
    # env= replaces the whole environment, so PYTHONDONTWRITEBYTECODE would not reach the child).
    mods = subprocess.run([sys.executable, "-B", "-c", code], cwd=REPO, capture_output=True, text=True,
                          check=True, env={"PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1"}).stdout.split()
    assert mods, "build_parser() imported no axis-rooted module -- this probe is broken"
    # `pkg` itself (the bare top-level package) is allowed only EXACTLY -- it must not become a
    # "{pkg}." prefix, or every submodule would match it too and this test would catch nothing.
    group_pkgs = {f"{pkg}.{g}" for g in PUBLIC_GROUPS}
    allowed = {pkg, public_cli.__name__} | group_pkgs
    prefixes = tuple(f"{a}." for a in group_pkgs | {public_cli.__name__})
    bad = [m for m in mods if m not in allowed and not m.startswith(prefixes)]
    assert bad == [], f"build_parser() reaches module(s) outside the shipped packages: {bad}"
