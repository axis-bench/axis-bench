"""What a `background=1` block may be: one server, checked ready by the block right after it.

A tutorial that evaluates a policy needs a server running beside the command that talks to it, and a
block may not use `&` (tests/docs/shell.py). So a run block marked `background=1` is started and not
waited for (tests/docs/runner.py). To keep it from passing anything it does not prove, it:
- holds exactly one command, one of SERVERS, which names the port it serves;
- is followed by the next shell block of the page, of its tag, whose first command is one of
  READINESS against that same port (not negated, not `--help`), so a server that never serves
  fails there;
- never counts as coverage (its shims log nothing, and the static rule skips it);
- fails the page as soon as it has exited, checked after every later block and at the page's end.

Core tier: stdlib only.
"""
from __future__ import annotations

import re

from tests.docs import shell
from tests.docs.coverage import command_key

# The servers a background block may run, each with how its command names the port it serves.
SERVERS = {"python -m axis_qvla.bench.servers.lerobot_server": re.compile(r"--port[ =](\d+)\b"),
           "python experiments/examples/policy_server.py": re.compile(r"--port[ =](\d+)\b")}
# The readiness checks the next block starts with, each with how it names the server's port; its
# --timeout also waits for the port to open.
READINESS = {"qvla bench check-server": re.compile(r"--policy[ =](?:127\.0\.0\.1|localhost):(\d+)\b")}
RULE = ("a background block runs exactly one command, a server ({servers}), and the next shell block, "
        "of its tag, starts with a readiness check of that server's port ({checks})")


def _port(table: dict, cmd: str) -> str | None:
    pattern = table.get(command_key(cmd) or "")
    m = pattern.search(cmd) if pattern else None
    return m.group(1) if m else None


def problems(shells: list) -> list[str]:
    """Why the background blocks among `shells` (a page's non-excerpt shell blocks, in order) break
    the module docstring's rules; [] when they keep them."""
    out = []
    rule = RULE.format(servers=", ".join(sorted(SERVERS)), checks=", ".join(sorted(READINESS)))
    for k, b in enumerate(shells):
        if not b.background:
            continue
        at, cmds = f"line {b.line}", shell.commands(b.text)
        if b.attr("background") != "1" or b.tag not in ("ci", "ci-render", "data"):
            out.append(f"{at}: `background=1` marks a `ci`, `ci-render` or `data` block")
        port = _port(SERVERS, cmds[0]) if len(cmds) == 1 and not cmds[0].startswith("!") else None
        if port is None:
            out.append(f"{at}: {rule}")
            continue
        after = shells[k + 1] if k + 1 < len(shells) else None
        first = shell.commands(after.text)[:1] if after is not None and after.tag == b.tag else []
        words = first[0].split() if first else []
        # argparse takes any unambiguous prefix of --help (--he, --hel) as --help
        checks = bool(words) and words[0] != "!" and not any(w == "-h" or (len(w) >= 3 and "--help".startswith(w)) for w in words)
        checked = _port(READINESS, first[0]) if checks else None
        if checked is None:
            out.append(f"{at}: {rule}")
        elif checked != port:
            out.append(f"line {after.line}: the readiness check names port {checked}, but the background "
                       f"block at line {b.line} serves port {port}")
    return out
