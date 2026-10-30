"""The fenced blocks of Install and the tutorials, and the rules every such page obeys.

Every fenced block carries a tag in its info string, after the language; GitHub renders the first
word as the language and shows nothing of the rest:

    ```bash ci             run by the CI `test` job, as written, in a scratch directory
    ```bash ci-render      run by the CI `render` job (the renderer's environment, software GL)
    ```bash data           run by the CI `data` job; fetches published data through the release lock
    ```bash gpu            never run by CI: a GPU, a backbone's own environment, or data outside the lock
    ```bash setup          never run by CI: cloning and installing
    ```text output         what the shell block before it prints: each line appears in that output
    ```python file=<path>  a verbatim excerpt of a shipped file (any language; never run)

A `data`, `gpu` or `setup` block is what a reader runs at full size. Its twin is the next shell block after it, a `ci` or `ci-render`
block running the same step on a shipped fixture; the twin of a `data`
block runs at least one of its commands besides the fetches. A page hides a stand-in twin, with its `output` block, in
a CI comment: a line `<!-- ci`, the fences at the margin, and a line `-->` (tests/docs/markdown.py
`ci_regions`). GitHub and the site show nothing of it, and this parser opens its fences as any other
(`Block.hidden`). Only `ci`, `ci-render` and `output` blocks may hide, and a visible run block may hold no
stand-in (`STAND_IN`). A visible `output` block may follow a hidden run block: it is checked against that
block's output. Every command of a `gpu` or `setup` block must also run in a run block of the page
(tests/docs/coverage.py: named there, and measured when the page runs). A shell block may set `timeout=<seconds>` (default 600). A run block marked
`background=1` is a server that the next block checks ready (tests/docs/servers.py); no `output`
block follows it. In an `output` line, `…` stands for any text within that one line. The blocks of
each tag run in page order, apart from the other tags' blocks, so a block depends only on earlier
blocks of its own tag.

So that no block can pass without being checked:
- a block takes one tag, and a run block runs at least one command;
- every shell block holds only the forms tests/docs/shell.py allows (simple commands, pipelines,
  redirections, assignments, `!` before the last command);
- a `gpu` or `setup` command that runs `qvla` or `python` starts with it, so its key can be read;
- no code sits where this parser or a reader cannot see it (tests/docs/markdown.py): an indented
  fence or code block, a fence after a list marker or in a quote, a fence in an HTML comment.

Core tier: stdlib only.
"""
from __future__ import annotations

import dataclasses
import re

from tests.docs import markdown, servers, shell
from tests.docs.coverage import EXERCISED_BY, FETCHES, MENTION, command_key, commands  # noqa: F401 (the interface)

SHELL = ("bash", "sh", "console")
SHELL_TAGS = ("ci", "ci-render", "data", "gpu", "setup")
RUN_TAGS = ("ci", "ci-render", "data")
SCALE_TAGS = ("data", "gpu", "setup")
TWIN_TAGS = ("ci", "ci-render")
DEFAULT_TIMEOUT = 600
HIDDEN_TAGS = ("ci", "ci-render", "output")
# What only CI runs: the test suite's fixtures and the stand-in flags. A visible block holding one shows a
# reader a command that is not theirs.
STAND_IN = re.compile(r"(?<![\w./-])tests[/.][\w./-]*|--(?:toy|no-bind|bench-conf-dir|results-map)\b")
_OPEN = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})(?P<info>.*)$")


@dataclasses.dataclass(frozen=True)
class Block:
    line: int                                   # 1-based line of the opening fence
    lang: str
    tag: str                                    # the first word after the language without "=", or ""
    attrs: tuple[tuple[str, str], ...]
    text: str
    hidden: bool = False                        # inside a `<!-- ci` comment (tests/docs/markdown.py)

    @property
    def shell(self) -> bool:
        return self.lang in SHELL

    def attr(self, key: str, default: str | None = None) -> str | None:
        return dict(self.attrs).get(key, default)

    @property
    def timeout(self) -> int:
        return int(self.attr("timeout", str(DEFAULT_TIMEOUT)))

    @property
    def background(self) -> bool:
        return self.attr("background") is not None


def blocks(text: str) -> list[Block]:
    """Every fenced block of a Markdown text, in order; an unclosed fence runs to the end."""
    lines, out, i = text.splitlines(), [], 0
    ci = markdown.ci_lines(text)
    while i < len(lines):
        m = _OPEN.match(lines[i])
        if m is None or (m["fence"][0] == "`" and "`" in m["info"]):
            i += 1
            continue
        fence = m["fence"]
        close = re.compile(rf"^ {{0,3}}{re.escape(fence[0])}{{{len(fence)},}}\s*$")
        words = m["info"].split()
        body, j = [], i + 1
        while j < len(lines) and not close.match(lines[j]):
            body.append(lines[j])
            j += 1
        out.append(Block(line=i + 1, lang=words[0] if words else "",
                         tag=next((w for w in words[1:] if "=" not in w), ""),
                         attrs=tuple(tuple(w.split("=", 1)) for w in words[1:] if "=" in w),
                         text="".join(f"{ln}\n" for ln in body), hidden=i + 1 in ci))
        i = j + 1
    return out


def shell_blocks(bs: list[Block]) -> list[Block]:
    """The shell blocks of `bs` that run or are run at scale: every one but the excerpts."""
    return [b for b in bs if b.shell and not _excerpt(b)]


def spans(bs: list[Block]) -> set[int]:
    """The page lines the blocks `bs` cover, fences included."""
    return {n for b in bs for n in range(b.line, b.line + b.text.count("\n") + 2)}


def problems(text: str) -> list[str]:
    """Why a page breaks the rules of the module docstring; [] when it keeps them."""
    bs, out, lines = blocks(text), [], text.splitlines()
    shells = shell_blocks(bs)
    for k, b in enumerate(bs):
        tags = [w for w in _OPEN.match(lines[b.line - 1])["info"].split()[1:] if "=" not in w]
        if len(tags) > 1:
            out.append(f"line {b.line}: a block takes one tag, not {' '.join(tags)!r}")
        if b.shell and not _excerpt(b):
            out += _shell_problems(b, shells)
        elif b.tag == "output":
            before = bs[k - 1] if k else None
            if before is None or not before.shell or before.tag not in RUN_TAGS or before.background:
                out.append(f"line {b.line}: an `output` block follows a `ci`, `ci-render` or `data` block "
                           "that is not a background one")
        elif b.attr("file") is None:
            out.append(f"line {b.line}: a `{b.lang or 'plain'}` block needs `output` or `file=<path>`")
    out += markdown.problems(text, spans(bs), [b.line for b in bs])
    ci = markdown.ci_lines(text)
    for b in bs:
        if b.hidden:
            if b.tag not in HIDDEN_TAGS:
                out.append(f"line {b.line}: only `ci`, `ci-render` and `output` blocks may sit in a CI comment: "
                           "a reader must see every command they run")
            if not spans([b]) <= ci:
                out.append(f"line {b.line}: a block in a CI comment ends after the comment closes")
        elif b.shell and b.tag in RUN_TAGS and (m := STAND_IN.search(b.text)):
            out.append(f"line {b.line}: a visible block runs a CI stand-in ({m.group(0)!r}): "
                       "move it into a `<!-- ci` comment")
    out += servers.problems(shells)
    if not any(b.tag in TWIN_TAGS for b in shells):
        out.append("no `ci` or `ci-render` block: CI would run nothing on this page")
    named = {k for b in shells if b.tag in RUN_TAGS and not b.background for k in commands(b.text)}
    for b in shells:
        if b.tag in ("gpu", "setup"):
            out += [f"line {b.line}: `{k}` runs in no `ci`, `ci-render` or `data` block of this page"
                    for k in commands(b.text) if not k.startswith(FETCHES) and EXERCISED_BY.get(k, k) not in named]
    return out


def _shell_problems(b: Block, shells: list[Block]) -> list[str]:
    at, out = f"line {b.line}", []
    if b.tag not in SHELL_TAGS:
        out.append(f"{at}: a shell block needs one of the tags {', '.join(SHELL_TAGS)}, not {b.tag or 'none'!r}")
    if b.tag in SCALE_TAGS:
        twin = next((s for s in shells if s.line > b.line), None)
        own = {c for c in commands(b.text) if not c.startswith(FETCHES)}
        if twin is None or twin.tag not in TWIN_TAGS:
            out.append(f"{at}: a `{b.tag}` block needs a `ci` or `ci-render` twin as the next shell block")
        elif b.tag == "data" and own and not own & set(commands(twin.text)):
            out.append(f"{at}: the twin of a `data` block runs none of its commands ({', '.join(sorted(own))})")
    if not (b.attr("timeout") or "1").isdigit():
        out.append(f"{at}: timeout= takes whole seconds")
    lines = shell.logical_lines(b.text)
    if b.tag in RUN_TAGS and not lines:
        out.append(f"{at}: a `{b.tag}` block runs nothing")
    out += [f"line {b.line + n}: `{cur}`: {why}" for n, cur, why in shell.refused(b.text)]
    if b.tag in ("gpu", "setup"):
        out += [f"line {b.line + n}: `{' '.join(w.text for w in words)}` runs qvla or python in a form whose "
                "command key cannot be read: start the command with it"
                for n, _, toks, _ in lines for words in shell.stages(toks)
                if words and command_key(" ".join(w.text for w in words)) is None
                and any(not w.quoted and MENTION.fullmatch(w.text.rsplit("/", 1)[-1]) for w in words)]
    return out


def _excerpt(b: Block) -> bool:
    """A block with `file=<path>` and no tag quotes a file and never runs, whatever its language."""
    return b.attr("file") is not None and not b.tag


def excerpt_problems(text: str, resolve) -> list[str]:
    """`file=<path>` blocks that are not a verbatim excerpt of the file `resolve(path)` returns
    (None: no such file)."""
    out = []
    for b in blocks(text):
        rel = b.attr("file")
        if rel is None:
            continue
        path = resolve(rel)
        if path is None:
            out.append(f"line {b.line}: {rel} is not in the repository")
        elif b.text not in path.read_text(encoding="utf-8"):
            out.append(f"line {b.line}: not a verbatim excerpt of {rel}")
    return out


def output_problems(expected: str, got: str) -> list[str]:
    """Lines of an `output` block that appear in no line of `got` (`…` matches any text in a line)."""
    lines = got.splitlines()
    return [want for want in expected.splitlines()
            if want.strip() and not any(_in_order(want.split("…"), line) for line in lines)]


def _in_order(parts: list[str], line: str) -> bool:
    pos = 0
    for part in parts:
        k = line.find(part, pos)
        if k < 0:
            return False
        pos = k + len(part)
    return True
