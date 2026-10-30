"""The documentation site's MkDocs hook: the tutorial harness's tagged fences render as plain code.

A fenced block on a runnable page carries a tag after its language in its info string: `bash ci`,
`text output`, `python file=<path>`, `bash data timeout=1800` (tests/docs/blocks.py). Python-Markdown
and pymdownx.superfences accept none of these info strings, and a block they do not open renders as
a paragraph that swallows the headings after it.
`site_markdown` rewrites each opening fence to its language, or to `<language> title="<path>"` for an
excerpt, and leaves every other line as it is, so the site shows the blocks the harness runs. It also
blanks each CI comment (`<!-- ci` ... `-->`), the stand-ins only CI runs, so the site carries none of
them, not even in its page source or search index.

Core tier: stdlib only (MkDocs imports this file by path, in the site's own environment).
"""
from __future__ import annotations

import re

_OPEN = re.compile(r"^(?P<indent> {0,3})(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
_CI_OPEN = re.compile(r"^ {0,3}<!-- ci[ \t]*$")


def ci_lines(lines: list[str]) -> set[int]:
    """0-based indexes of every line of each CI comment: from a line `<!-- ci` to the first later line
    holding `-->`, or to the end. The tester's own copy is tests/docs/markdown.py `ci_lines`, and
    tests/docs/test_site.py holds the two to the same lines (this file imports nothing of the repository)."""
    out, i = set(), 0
    while i < len(lines):
        if _CI_OPEN.match(lines[i]):
            j = next((k for k in range(i + 1, len(lines)) if "-->" in lines[k]), len(lines) - 1)
            out.update(range(i, j + 1))
            i = j + 1
        else:
            i += 1
    return out


def site_info(info: str) -> str:
    """The info string the site renders: the language, and an excerpt's file as its title."""
    words = info.split()
    if not words:
        return info
    file = next((w.split("=", 1)[1] for w in words[1:] if w.startswith("file=")), None)
    return f'{words[0]} title="{file}"' if file else words[0]


def site_markdown(text: str) -> str:
    """`text` with each CI comment blanked line for line (so line numbers hold) and every opening fence's
    info string replaced by `site_info`'s; bodies unchanged."""
    lines = text.split("\n")
    hide = ci_lines(lines)
    out, fence = [], None
    for n, line in enumerate(lines):
        if n in hide:
            out.append("")
            continue
        m = _OPEN.match(line)
        if fence is None:
            if m and not (m["fence"][0] == "`" and "`" in m["info"]):
                fence = m["fence"]
                line = f"{m['indent']}{fence}{site_info(m['info'])}"
        elif re.match(rf"^ {{0,3}}{re.escape(fence[0])}{{{len(fence)},}}\s*$", line):
            fence = None
        out.append(line)
    return "\n".join(out)


def on_page_markdown(markdown: str, page, config, files) -> str:
    return site_markdown(markdown)
