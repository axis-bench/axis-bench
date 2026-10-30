"""Background blocks (tests/docs/servers.py): one server, checked ready by the next block, never
coverage, and failing the page the moment it exits."""
from __future__ import annotations

import pathlib
import re
import socket

import pytest

from tests.docs import blocks as bl
from tests.docs import coverage, runner, servers

SERVER = "python -m axis_qvla.bench.servers.lerobot_server --policy-type mock --ckpt mock --host 127.0.0.1 --port {port}"
READY = "qvla bench check-server --policy 127.0.0.1:{port} --timeout 60"


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


@pytest.mark.parametrize("page", [
    # each passed before: a failing command killed before it fails, a command that "covers" a gpu
    # block by being started, and a block that serves nothing
    "```bash ci background=1\nqvla train no_such.toml --no-bind\n```\n```bash ci\nls\n```\n",
    "```bash gpu\nqvla paper tables --results r --check\n```\n"
    "```bash ci background=1\nqvla paper tables --results no_such --check\n```\n```bash ci\nls\n```\n",
    "```bash ci background=1\nsleep 100\n```\n```bash ci\nls\n```\n",
], ids=["failing-train", "covering-tables", "sleep"])
def test_a_background_block_that_is_not_a_checked_server_fails(tmp_path, page):
    assert any("a background block runs exactly one command, a server" in p for p in bl.problems(page))
    problems = runner.run_page(_page(tmp_path, page), ("ci",), tmp_path / "run")
    assert problems and "a background block runs exactly one command, a server" in problems[0]
    assert not (tmp_path / "run" / "block-1.log").exists() and not (tmp_path / "run" / "block-4.log").exists()


@pytest.mark.parametrize("page, why", [
    (f"```bash ci background=1\n{SERVER.format(port=8001)}\n```\n```bash ci\n{READY.format(port=8002)}\n```\n",
     "line 4: the readiness check names port 8002, but the background block at line 1 serves port 8001"),
    (f"```bash ci background=1\n{SERVER.format(port=8001)}\n```\n```bash ci\nls\n{READY.format(port=8001)}\n```\n",
     "a background block runs exactly one command"),
    (f"```bash ci background=1\n{SERVER.format(port=8001)}\n```\n```bash ci-render\n{READY.format(port=8001)}\n```\n"
     "```bash ci\nls\n```\n", "a background block runs exactly one command"),
    (f"```bash ci background=1\n{SERVER.format(port=8001)}\nls\n```\n```bash ci\n{READY.format(port=8001)}\n```\n",
     "a background block runs exactly one command"),
    (f"```bash ci background=1\n! {SERVER.format(port=8001)}\n```\n```bash ci\n{READY.format(port=8001)}\n```\n",
     "a background block runs exactly one command"),
    (f"```bash gpu background=1\n{SERVER.format(port=8001)}\n```\n```bash ci\n{READY.format(port=8001)}\n```\n",
     "`background=1` marks a `ci`, `ci-render` or `data` block"),
    # a readiness command that does not connect: --help, -h, negated, or no --policy
    (f"```bash ci background=1\n{SERVER.format(port=8001)}\n```\n```bash ci\n{READY.format(port=8001)} --help\n```\n",
     "a background block runs exactly one command"),
    (f"```bash ci background=1\n{SERVER.format(port=8001)}\n```\n```bash ci\n{READY.format(port=8001)} -h\n```\n",
     "a background block runs exactly one command"),
    (f"```bash ci background=1\n{SERVER.format(port=8001)}\n```\n```bash ci\n{READY.format(port=8001)} --he\n```\n",
     "a background block runs exactly one command"),     # argparse reads a prefix of --help as --help
    (f"```bash ci background=1\n{SERVER.format(port=8001)}\n```\n```bash ci\n! {READY.format(port=8001)}\n```\n",
     "a background block runs exactly one command"),
    (f"```bash ci background=1\n{SERVER.format(port=8001)}\n```\n```bash ci\nqvla bench check-server --timeout 60\n```\n",
     "a background block runs exactly one command"),
    (f"```bash gpu\n{SERVER.format(port=9)}\n```\n```bash ci background=1\n{SERVER.format(port=8001)}\n```\n"
     f"```bash ci\n{READY.format(port=8001)}\n```\n",
     "`python -m axis_qvla.bench.servers.lerobot_server` runs in no `ci`, `ci-render` or `data` block"),   # never coverage
])
def test_the_server_rules_are_named(page, why):
    found = bl.problems(page)
    assert any(why in p for p in found), found


def test_a_checked_server_keeps_the_rules():
    page = f"```bash ci background=1\n{SERVER.format(port=8001)} > mock.log 2>&1\n```\n```bash ci\n{READY.format(port=8001)}\n```\n"
    assert bl.problems(page) == [] and servers.problems(bl.shell_blocks(bl.blocks(page))) == []


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """A server that serves a port until a `stop` file appears, then exits 3, and its readiness check;
    both declared for this test only."""
    srv, ready, gone = tmp_path / "srv.py", tmp_path / "ready.py", tmp_path / "gone.py"
    srv.write_text("import os, pathlib, socket, sys, time\nport = int(sys.argv[sys.argv.index('--port') + 1])\n"
                   "s = socket.socket()\ns.bind(('127.0.0.1', port))\ns.listen()\n"
                   "pathlib.Path('server.pgid').write_text(str(os.getpgid(0)))\n"
                   "while not pathlib.Path('stop').exists(): time.sleep(0.01)\nsys.exit(3)\n")
    ready.write_text("import socket, sys, time\nport = int(sys.argv[sys.argv.index('--port') + 1])\nwhile True:\n"
                     "    try:\n        socket.create_connection(('127.0.0.1', port)).close()\n        break\n"
                     "    except OSError:\n        time.sleep(0.01)\n")
    gone.write_text("import pathlib, time\nstat = pathlib.Path(f\"/proc/{pathlib.Path('server.pgid').read_text()}/stat\")\n"
                    "def zombie_or_gone():\n    try:\n        return stat.read_text().rsplit(')', 1)[1].split()[0] == 'Z'\n    except OSError:\n        return True\nwhile not zombie_or_gone(): time.sleep(0.01)\n")
    monkeypatch.setitem(servers.SERVERS, f"python {srv}", re.compile(r"--port (\d+)"))
    monkeypatch.setitem(servers.READINESS, f"python {ready}", re.compile(r"--port (\d+)"))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return (f"```bash ci background=1\npython {srv} --port {port}\n```\n```bash ci\npython {ready} --port {port}\n```\n",
            gone)


def test_a_server_serves_the_later_blocks_is_no_coverage_and_is_stopped(tmp_path, fake):
    head, _ = fake
    assert runner.run_page(_page(tmp_path, head + "```bash ci\ntouch after\n```\n"), ("ci",), tmp_path / "run") == []
    logged = (tmp_path / "run" / coverage.RAN).read_text()
    assert "ready.py" in logged and "srv.py" not in logged
    assert not _alive(int((tmp_path / "run" / "work" / "server.pgid").read_text()))


def test_a_server_that_exits_fails_the_page_at_the_next_block(tmp_path, fake):
    head, gone = fake
    page = _page(tmp_path, head + f"```bash ci\ntouch stop\npython {gone}\n```\n```bash ci\ntouch after\n```\n")
    (problem,) = runner.run_page(page, ("ci",), tmp_path / "run")
    assert problem.startswith("page.md:1: the background block exited (3) while the page ran"), problem
    assert not (tmp_path / "run" / "work" / "after").exists()
