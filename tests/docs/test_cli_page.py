"""docs/reference/cli.md is generated from the parsers of `qvla` and of the tools its groups forward
to, so it lists every command; and every command its parent lists carries a one-line help."""
from __future__ import annotations

from tests.docs import cli_page
from tests.docs.runner import REPO

PAGE = REPO / "docs" / "reference" / "cli.md"


def test_every_command_has_a_help_line():
    assert [c.path for c in cli_page.commands() if not c.help.strip()] == []


def test_every_group_of_qvla_is_on_the_page():
    import axis_qvla.cli as qvla_cli

    groups = set(cli_page.subparsers(qvla_cli.build_parser()).choices)
    assert {c.path.split()[1] for c in cli_page.commands()} == groups


def test_a_forwarded_tool_parser_is_captured_without_running_the_tool():
    parser = cli_page.tool_parser("axis_qvla.train.plain_copy")
    assert parser.prog == "qvla train merge-check"


def test_the_page_is_the_generated_one():
    assert PAGE.read_text(encoding="utf-8") == cli_page.render(), \
        "regenerate: python -m tests.docs.cli_page > docs/reference/cli.md"
