"""The runner (tests/docs/runner.py) on made-up pages: order, output, isolation, the strict shell,
background blocks, timeouts and the process group."""
from __future__ import annotations

import pathlib
import subprocess
import sys
import time

import pytest

from tests.docs import runner


def _page(tmp_path: pathlib.Path, text: str) -> pathlib.Path:
    page = tmp_path / "page.md"
    page.write_text(text)
    return page


def _alive(pid: int) -> bool:
    try:
        state = pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:     # gone, or reaped between the open and the read (ProcessLookupError)
        return False
    return state not in ("Z", "X")


def test_the_runner_runs_blocks_in_order_in_a_public_view_and_checks_their_output(tmp_path):
    page = _page(tmp_path, "```bash ci\necho one > a.txt\n```\n```bash ci\ncat a.txt\nqvla --help\n"
                           "test -d tests/docs\ntest -f qvla.example.toml\n```\n```text output\none\nusage: qvla…\n```\n")
    assert runner.run_page(page, ("ci",), tmp_path / "run") == []
    assert (tmp_path / "run" / "work" / "a.txt").read_text() == "one\n"


def test_a_failing_block_ends_the_run_and_names_its_exit(tmp_path):
    page = _page(tmp_path, "```bash ci\nfalse\n```\n```bash ci\ntouch never\n```\n")
    (problem,) = runner.run_page(page, ("ci",), tmp_path / "run")
    assert problem.startswith("page.md:1: exit 1")
    assert not (tmp_path / "run" / "work" / "never").exists()


def test_a_missing_output_line_is_reported(tmp_path):
    page = _page(tmp_path, "```bash ci\necho two\n```\n```text output\nthree\n```\n")
    (problem,) = runner.run_page(page, ("ci",), tmp_path / "run")
    assert "not printed by the block at line 1: 'three'" in problem


def test_only_the_asked_tags_run(tmp_path):
    page = _page(tmp_path, "```bash gpu\ntouch gpu\n```\n```bash ci\ntouch ci\n```\n")
    assert runner.run_page(page, ("ci",), tmp_path / "run") == []
    assert sorted(p.name for p in (tmp_path / "run" / "work").iterdir() if not p.is_symlink()) == ["ci"]


def test_a_block_sees_none_of_this_shells_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("QVLA_DOCS_PROBE", "leaked")
    page = _page(tmp_path, '```bash ci\ntest -z "${QVLA_DOCS_PROBE:-}"\ntest "$HOME" != /\n'
                           f"python3 -c 'import sys; assert sys.prefix == \"{sys.prefix}\"'\n```\n")
    assert runner.run_page(page, ("ci",), tmp_path / "run") == []


def test_a_block_reads_nothing_from_the_runners_stdin(tmp_path):
    """The block's stdin is /dev/null, whatever the stdin of the process that runs it carries."""
    code = ("import pathlib, sys\nfrom tests.docs import runner\nroot = pathlib.Path(sys.argv[1])\n"
            "work, env = runner.scratch(root)\nprint(runner.run_block('cat > got.txt', work, env, root / 'b.log', 60))\n")
    r = subprocess.run([sys.executable, "-c", code, str(tmp_path / "run")], input="leaked\n", text=True,
                       capture_output=True, cwd=runner.REPO,
                       env={"PYTHONPATH": str(runner.REPO), "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)})
    assert r.returncode == 0 and r.stdout.startswith("(0, "), r.stdout + r.stderr
    assert (tmp_path / "run" / "work" / "got.txt").read_text() == ""


@pytest.mark.parametrize("body", ["false\ntrue\n", "false | true\n", 'echo "$QVLA_DOCS_UNSET"\n',
                                  'x=$(false; echo hi)\necho "$x"\n'], ids=["e", "pipefail", "u", "inherit_errexit"])
def test_every_failing_command_ends_its_block(tmp_path, body):
    """The strict options, each pinned: under plain `bash -c` every one of these blocks would pass.
    (The page rules refuse command substitution; run_block, below them, still runs it strictly.)"""
    work, env = runner.scratch(tmp_path / "run")
    code, out = runner.run_block(f"{body}touch ran\n", work, env, tmp_path / "b.log", 60)
    assert code == 1 and not (work / "ran").exists(), out


@pytest.mark.parametrize("body, line, why", [("! qvla --help\necho next\n", 2, "`!` stands only before"),
                                             ("false && true\necho after\n", 2, "an `&&` list")],
                         ids=["negated", "and-list"])
def test_a_form_the_rules_refuse_fails_its_block_before_it_runs(tmp_path, body, line, why):
    (problem,) = runner.run_page(_page(tmp_path, f"```bash ci\n{body}touch ran\n```\n"), ("ci",), tmp_path / "run")
    assert problem.startswith(f"page.md:{line}: `{body.splitlines()[0]}`: {why}"), problem
    assert not (tmp_path / "run" / "work" / "ran").exists()


def test_a_negated_last_command_is_checked_by_the_blocks_status(tmp_path):
    assert runner.run_page(_page(tmp_path, "```bash ci\n! false\n```\n"), ("ci",), tmp_path / "a") == []
    (problem,) = runner.run_page(_page(tmp_path, "```bash ci\n! qvla --help\n```\n"), ("ci",), tmp_path / "b")
    assert problem.startswith("page.md:1: exit 1"), problem


def test_a_background_process_does_not_outlive_its_block(tmp_path):
    work, env = runner.scratch(tmp_path / "run")
    assert runner.run_block("sleep 300 &\necho $! > pid\n", work, env, tmp_path / "b.log", 60)[0] == 0
    pid = int((work / "pid").read_text())
    deadline = time.monotonic() + 5
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(pid)


def test_a_block_is_killed_at_its_timeout(tmp_path):
    page = _page(tmp_path, "```bash ci timeout=1\nsleep 30\ntouch late\n```\n")
    (problem,) = runner.run_page(page, ("ci",), tmp_path / "run")
    assert "timed out after 1 s" in problem
    assert not (tmp_path / "run" / "work" / "late").exists()       # killed, not waited for


def test_a_hidden_block_runs_and_its_output_is_checked(tmp_path):
    page = _page(tmp_path, "<!-- ci\n```bash ci\necho hidden > h.txt\n```\n-->\n"
                           "```bash ci\ncat h.txt\n```\n```text output\nhidden\n```\n")
    assert runner.run_page(page, ("ci",), tmp_path / "run") == []
    bad = _page(tmp_path, "<!-- ci\n```bash ci\necho one\n```\n-->\n```text output\ntwo\n```\n")
    found = runner.run_page(bad, ("ci",), tmp_path / "bad")
    assert len(found) == 1 and "not printed by the block at line 2: 'two'" in found[0]
