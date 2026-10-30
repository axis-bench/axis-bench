"""The Markdown forms of a runnable page that would put code where the block parser cannot see it,
or where a reader cannot.

- A fence indented four or more spaces (a code block inside a list item, say), after a list marker
  or inside a quote is not opened by tests/docs/blocks.py, so its commands would run nowhere.
- A plain indented code block (four spaces more than its list item's content, after a blank line)
  has no info string, so it can carry no tag and never runs.
- A fence inside an HTML comment is refused, except inside a CI comment (`<!-- ci` ... `-->`,
  `ci_regions`), which holds the stand-ins CI runs and readers never see.

A fence has a line of its own, indented at most three spaces, outside any comment.

Core tier: stdlib only.
"""
from __future__ import annotations

import re

_HIDDEN = re.compile(r"^(?: *\t| {4,}|[ \t]*>|[ \t]*(?:[-*+]|\d{1,9}[.)])[ \t]+)[ \t>]*(?:`{3,}|~{3,})")
_ITEM = re.compile(r"^( *)([-*+]|\d{1,9}[.)])( +)\S")
_COMMENT = re.compile(r"<!--.*?-->", re.S)
CI_OPEN = re.compile(r"^ {0,3}<!-- ci[ \t]*$")
CI_CLOSE = re.compile(r"^ {0,3}-->[ \t]*$")
_DETAILS = re.compile(r"<details\b.*?</details>", re.S)


def _indent(line: str) -> int:
    return len(line.expandtabs(4)) - len(line.expandtabs(4).lstrip(" "))


def ci_regions(text: str) -> list[tuple[int, int | None]]:
    """(opening line, closing line or None when it never closes), 1-based, of each CI comment.

    A CI comment opens on a line `<!-- ci` and closes on the first later line holding `-->`: CommonMark's
    HTML block of type 2, so GitHub renders none of it, and the site hook blanks the same lines
    (tests/docs/site_hooks.py). The tester still opens the fences inside it, so a stand-in that only CI
    needs runs without a reader seeing it."""
    lines, out, i = text.splitlines(), [], 0
    while i < len(lines):
        if CI_OPEN.match(lines[i]):
            end = next((k for k in range(i + 1, len(lines)) if "-->" in lines[k]), None)
            out.append((i + 1, None if end is None else end + 1))
            i = len(lines) if end is None else end + 1
        else:
            i += 1
    return out


def ci_lines(text: str) -> set[int]:
    """Every line of every CI comment, both marker lines included; an unclosed one runs to the end."""
    last = text.count("\n") + 1
    return {n for a, b in ci_regions(text) for n in range(a, (b or last) + 1)}


def problems(text: str, seen: set[int], fences: list[int]) -> list[str]:
    """Why `text` holds code the parser or a reader cannot see. `seen` holds the lines of the blocks
    the parser opened, `fences` their opening lines."""
    lines, out = text.splitlines(), []
    hidden = {text.count("\n", 0, m.start()) + 1 + k for m in _COMMENT.finditer(text)
              for k in range(m.group(0).count("\n") + 1)}
    ci = ci_lines(text)
    for a, b in ci_regions(text):
        if b is None:
            out.append(f"line {a}: a `<!-- ci` comment is never closed: end it with a line `-->`")
        elif not CI_CLOSE.match(lines[b - 1]):
            out.append(f"line {b}: a `<!-- ci` comment closes with a line `-->` of its own")
    out += [f"line {n}: a fence inside an HTML comment runs in CI but no reader sees it: hide a stand-in in a "
            "`<!-- ci` comment, nothing else" for n in fences if n in hidden and n not in ci]
    boxed = {text.count("\n", 0, m.start()) + 1 + k for m in _DETAILS.finditer(text)
             for k in range(m.group(0).count("\n") + 1)}
    out += [f"line {n}: a fence inside a <details> box shows a CI stand-in to readers: put it in a `<!-- ci` comment"
            for n in fences if n in boxed]
    out += [f"line {n}: a fence this parser does not open (indented four or more spaces, after a list "
            "marker, or inside a quote) is neither run nor checked: give it a line of its own, indented "
            "at most three spaces" for n, line in enumerate(lines, 1) if n not in seen and _HIDDEN.match(line)]
    content, blank = 0, True                     # the open list item's content indent
    for n, line in enumerate(lines, 1):
        if n in seen or n in hidden:
            blank = False
            continue
        if not line.strip():
            blank = True
            continue
        indent, item = _indent(line), _ITEM.match(line.expandtabs(4))
        if blank and indent >= content + 4 and not _HIDDEN.match(line):
            out.append(f"line {n}: an indented code block has no tag, so it never runs: write it as a "
                       "fence with a tag")
        elif item:
            content = len(item.group(1)) + len(item.group(2)) + len(item.group(3))
        elif blank and indent < content:
            content = 0
        blank = False
    return out
