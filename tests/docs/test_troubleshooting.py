"""Every refusal message docs/reference/troubleshooting.md quotes is one the code can still print: each
quoted line (`…` for the parts that vary) appears, in order, in one string literal of the file the
quote names (adjacent literals and f-strings read as one)."""
from __future__ import annotations

import ast
import re

from tests.docs.runner import REPO

PAGE = REPO / "docs" / "reference" / "troubleshooting.md"
QUOTE = re.compile(r"<!-- quote: (?P<path>\S+) -->\n```text\n(?P<body>.*?)```", re.S)


def static_strings(source: str) -> list[str]:
    out = []
    for n in ast.walk(ast.parse(source)):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            out.append(n.value)
        elif isinstance(n, ast.JoinedStr):
            out.append("".join(v.value if isinstance(v, ast.Constant) else "\x00" for v in n.values))
    return out


def _in_order(parts: list[str], s: str) -> bool:
    pos = 0
    for part in parts:
        k = s.find(part, pos)
        if k < 0:
            return False
        pos = k + len(part)
    return True


def unquotable(path: str, body: str) -> list[str]:
    strings = static_strings((REPO / path).read_text(encoding="utf-8"))
    return [line for line in body.splitlines() if line.strip()
            and not any(_in_order([p.strip() for p in line.split("…") if p.strip()], s) for s in strings)]


def test_the_checker_reads_an_f_string_as_one_literal():
    src = 'def f(x):\n    raise ValueError(f"{x} already exists. Remove " f"{x} yourself")\n'
    strings = static_strings(src)
    assert any(_in_order(["already exists. Remove", "yourself"], s) for s in strings)
    assert not any(_in_order(["yourself", "already exists"], s) for s in strings)


def test_the_page_quotes_the_code():
    quotes = list(QUOTE.finditer(PAGE.read_text(encoding="utf-8")))
    assert len(quotes) >= 8
    found = {f"{m['path']}: {line}" for m in quotes for line in unquotable(m["path"], m["body"])}
    assert sorted(found) == []
