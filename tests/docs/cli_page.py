"""docs/reference/cli.md, generated from the parsers of `qvla` and of every tool a `qvla` group
forwards to.

    python -m tests.docs.cli_page > docs/reference/cli.md

`commands()` walks the real parsers:
- `qvla`'s groups (axis_qvla.cli.build_parser);
- the realrobot tools (axis_qvla.realrobot.cli.TOOLS);
- the words of `qvla train` (axis_qvla.train.cli._SUBCOMMANDS).

A hidden alias is left out: a parser already listed under another name (axis_qvla.paper.cli.hidden_alias)
and the train words of axis_qvla.train.cli.HIDDEN_SUBCOMMANDS.

The last two build their parsers inside their own `main`; `tool_parser` captures such a parser by
stopping that `main` when it starts parsing. A usage line is written with its whitespace collapsed,
so the page does not depend on the terminal's width. tests/docs/test_cli_page.py fails when the page
differs from `render()` or a command has no help line.

Core tier: stdlib only.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib

import axis_qvla.cli as qvla_cli
from axis_qvla.realrobot.cli import TOOLS
from axis_qvla.train.cli import _SUBCOMMANDS as TRAIN_WORDS, HIDDEN_SUBCOMMANDS

HEADER = """# CLI

The complete index of `qvla` commands, including the tools a `qvla` group forwards to, generated
from their argument parsers. Each entry gives the command's one-line help and its usage; run a
command with `--help` for what each option means. A forwarded tool's usage line names its module
(`python -m axis_qvla.realrobot.render source ...`); `qvla realrobot render source ...` runs the same
command. To choose a command for a task, start from the [tutorials](../index.md).
"""


@dataclasses.dataclass(frozen=True)
class Command:
    path: str            # "qvla bench eval"
    help: str            # the one line its parent lists (a train word: its parser's description)
    usage: str           # its usage line; "" for a group, which runs nothing itself
    level: int           # its heading level on the page


class _Built(Exception):
    def __init__(self, parser: argparse.ArgumentParser) -> None:
        super().__init__(parser.prog)
        self.parser = parser


def _stop(self, args=None, namespace=None):
    raise _Built(self)


def tool_parser(module: str) -> argparse.ArgumentParser:
    """The parser a forwarded tool's `main` builds, captured before it parses anything."""
    saved = argparse.ArgumentParser.parse_args
    argparse.ArgumentParser.parse_args = _stop
    try:
        importlib.import_module(module).main([])
    except _Built as built:
        return built.parser
    finally:
        argparse.ArgumentParser.parse_args = saved
    raise AssertionError(f"{module}.main returned without parsing its arguments")


def subparsers(p: argparse.ArgumentParser) -> argparse._SubParsersAction | None:
    return next((a for a in p._actions if isinstance(a, argparse._SubParsersAction)), None)


def _usage(p: argparse.ArgumentParser) -> str:
    return " ".join(p.format_usage().removeprefix("usage: ").split())


def _walk(p: argparse.ArgumentParser, path: str, help_: str, level: int, out: list[Command]) -> None:
    sub = subparsers(p)
    out.append(Command(path, help_, "" if sub else _usage(p), level))
    if sub is None:
        return
    helps = {a.dest: a.help or "" for a in sub._choices_actions}
    seen: set[int] = set()
    for name, child in sub.choices.items():
        if id(child) in seen:
            continue                  # a hidden alias of a command already listed
        seen.add(id(child))
        _walk(child, f"{path} {name}", helps.get(name, ""), level + 1, out)


def commands() -> list[Command]:
    out: list[Command] = []
    groups = subparsers(qvla_cli.build_parser())
    helps = {a.dest: a.help or "" for a in groups._choices_actions}
    for name, p in groups.choices.items():
        if name == "realrobot":
            out.append(Command("qvla realrobot", helps[name], "", 2))
            tool_helps = {a.dest: a.help or "" for a in subparsers(p)._choices_actions}
            for tool, module in TOOLS.items():
                _walk(tool_parser(module), f"qvla realrobot {tool}", tool_helps[tool], 3, out)
        elif name == "train":
            out.append(Command("qvla train", helps[name], _usage(p), 2))
            for word, module in TRAIN_WORDS.items():
                if word in HIDDEN_SUBCOMMANDS:
                    continue
                parser = tool_parser(module)
                _walk(parser, f"qvla train {word}", parser.description or "", 3, out)
        else:
            _walk(p, f"qvla {name}", helps[name], 2, out)
    return out


def render() -> str:
    parts = [HEADER]
    for c in commands():
        parts.append(f"{'#' * c.level} `{c.path}`\n\n{c.help}\n")
        if c.usage:
            parts.append(f"```text\n{c.usage}\n```\n")
    return "\n".join(parts)


if __name__ == "__main__":
    print(render(), end="")
