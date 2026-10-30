"""Which commands of a page's at-scale blocks CI must run, and whether a run ran them.

A `gpu` or `setup` block is never run by CI, so every command of it (`qvla <words>`, `python <script>`,
`python -m <module>`, keyed by `command_key`) must run in a `ci`, `ci-render` or `data` block of the
same page: on a shipped fixture, or with `--dry-run`, `--no-bind` or `--help`. Two halves check it:
- statically (tests/docs/blocks.py), some run block of the page names each key;
- measured (tests/docs/runner.py), the `qvla` and `python` on a block's PATH log every command they
  start into the run's RAN log, and each key a run block of the run's tags names must be in that log.
  A command a block names but never reaches (after a failure, inside data) is therefore not coverage.

Fetches (`qvla data get`) are exempt: the `data` job runs them. A command CI cannot run at all is
listed in EXERCISED_BY with the command that checks its inputs, which must run instead. A
`background=1` block runs a server (tests/docs/servers.py) and is never coverage: its shims log
nothing.

Core tier: stdlib only.
"""
from __future__ import annotations

import re

from tests.docs import shell

FETCHES = ("qvla data get",)          # prefixes of a command key
# Commands CI cannot run at all, and the shipped command that checks the same inputs, which must run
# on the page instead.
EXERCISED_BY = {
    # The LeRobot merge needs the LeRobot environment (torch), which CI does not build; merge-check
    # compares its inputs the way the merge does, before it runs (axis_qvla.train.plain_copy).
    "python -m lerobot.scripts.axis_merge_datasets": "qvla train merge-check",
}
RAN = "ran.log"                        # under a run's root: one started command per line
_WORD = re.compile(r"^[a-z][a-z0-9-]*$")
_ENV = re.compile(r"^[A-Z_][A-Z0-9_]*=")
_PREFIX = ("!", "nohup", "env")        # words before the command
_PYTHON = re.compile(r"python(?:3(?:\.\d+)?)?")
MENTION = re.compile(r"(?<![\w/.-])(?:qvla|python(?:3(?:\.\d+)?)?)(?![\w.-])")


def command_key(line: str) -> str | None:
    """What a shell line runs, as far as CI coverage is concerned: `qvla` with its command words (up
    to the first argument), `python <script>`, or `python -m <module>` with its command words (any
    `python3.<n>` too); None for anything else (shell built-ins, `git`, `pip`, `hf`, `mkdir`, `cp`).
    Leading `!`, `nohup`, `env` and VAR=value words are not the command."""
    toks = line.split("#", 1)[0].split()
    while toks and (toks[0] in _PREFIX or _ENV.match(toks[0])):
        toks = toks[1:]
    if not toks:
        return None
    if toks[0] == "qvla":
        return " ".join(["qvla", *_words(toks[1:])])
    if _PYTHON.fullmatch(toks[0].rsplit("/", 1)[-1]) and len(toks) > 1:
        if toks[1] == "-m" and len(toks) > 2:
            return " ".join(["python", "-m", toks[2], *_words(toks[3:])])
        return f"python {toks[1]}"
    return None


def _words(toks: list[str]) -> list[str]:
    """The command words that open `toks`, up to the first argument."""
    out = []
    for t in toks:
        if not _WORD.match(t):
            break
        out.append(t)
    return out


def commands(text: str) -> list[str]:
    """The command keys of a shell block, one per command of each pipeline (heredoc bodies hold none)."""
    return [k for cmd in shell.commands(text) if (k := command_key(cmd)) is not None]


def needed(scale_texts: list[str]) -> set[str]:
    """The keys CI must run for these `gpu`/`setup` block texts, each mapped through EXERCISED_BY."""
    return {EXERCISED_BY.get(k, k) for t in scale_texts for k in commands(t) if not k.startswith(FETCHES)}


def ran(log: str) -> set[str]:
    """The keys of the commands a RAN log records."""
    return {k for line in log.splitlines() if (k := command_key(line)) is not None}
