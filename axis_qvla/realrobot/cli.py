"""The `qvla realrobot` command group: render, realign, realdata and fr3, each forwarded to its own CLI.

Core-loadable: stdlib only at module scope (the `qvla` CLI builds every group's parser). Dispatch is
`args.func(args)`, as axis_qvla/cli.py calls it. Everything after the tool name is handed over untouched,
so it must start with the tool's subcommand: `qvla realrobot render gates --out DIR`. Help is the tool's
own: `qvla realrobot render --help` lists its commands, `qvla realrobot render gates --help` one command.

The tutorial's steps (`workspace`, `check-camera`, `rerender`, `labels`, `pretrain`, `demos`, `finetune`,
`serve`) are commands of this group too (axis_qvla.realrobot.steps.parsers); they take real arguments, not a
pass-through.
"""
from __future__ import annotations

import argparse
import importlib

TOOLS = {"render": "axis_qvla.realrobot.render.cli", "realign": "axis_qvla.realrobot.realign.cli",
         "realdata": "axis_qvla.realrobot.realdata.cli", "fr3": "axis_qvla.realrobot.fr3.cli"}


def _forward(module: str, argv: list[str]) -> int:
    return importlib.import_module(module).main(argv)


def add_realrobot_parsers(sub) -> None:
    rr = sub.add_parser("realrobot", help="real-robot steps (workspace … serve) and tools: kinematic renderer, "
                                           "REALALIGN labels, FR3 data and configs")
    tools = rr.add_subparsers(dest="realrobot_tool", required=True)
    for name, module in TOOLS.items():
        # add_help=False, so a leading --help is ours to forward: argparse.REMAINDER does not take an
        # option as its FIRST token, and an unknown one would exit 2 here instead of reaching the tool.
        p = tools.add_parser(name, help=f"python -m {module.rsplit('.', 1)[0]}", add_help=False)
        p.add_argument("-h", "--help", action="store_true", dest="tool_help", help="the tool's own help")
        p.add_argument("args", nargs=argparse.REMAINDER)
        p.set_defaults(func=lambda a, module=module: _forward(module, ["--help"] * a.tool_help + a.args))

    from axis_qvla.realrobot.steps.parsers import add_step_parsers   # stdlib only; imported here so this module's own imports stay stdlib
    add_step_parsers(tools)
