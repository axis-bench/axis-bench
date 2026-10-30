"""The documentation site (MkDocs Material): its hook renders the harness's tagged fences as the code
blocks the harness runs (tests/docs/site_hooks.py)."""
from __future__ import annotations

import re

from tests.docs import markdown, site_hooks
from tests.docs.blocks import blocks
from tests.docs.runner import public_file, shipped_pages

PAGE = ("# T\n\n```bash ci background=1\nserve\n```\n\n```bash data timeout=1800\nqvla x\n```\n\n"
        "```text output\nIDENTICAL\n```\n\n```python file=experiments/examples/keep_top.py\nx = 1\n```\n\n"
        "````markdown\n```bash ci\n````\n\n```\nplain\n```\n")


def test_a_tagged_fence_renders_as_its_language_and_an_excerpt_names_its_file():
    assert site_hooks.site_markdown(PAGE) == (
        "# T\n\n```bash\nserve\n```\n\n```bash\nqvla x\n```\n\n```text\nIDENTICAL\n```\n\n"
        '```python title="experiments/examples/keep_top.py"\nx = 1\n```\n\n'
        "````markdown\n```bash ci\n````\n\n```\nplain\n```\n")


def github_comment_lines(text: str) -> set[int]:
    """1-based lines GitHub renders as nothing: CommonMark HTML blocks of type 2, from a line that starts
    (after at most three spaces) with `<!--` to the first line, that one included, holding `-->` after it."""
    lines, out, i = text.split("\n"), set(), 0
    while i < len(lines):
        m = re.match(r"^ {0,3}<!--", lines[i])
        if m is None:
            i += 1
            continue
        j = next((k for k in range(i, len(lines)) if "-->" in lines[k][(m.end() if k == i else 0):]), len(lines) - 1)
        out.update(range(i + 1, j + 2))
        i = j + 1
    return out


def test_a_ci_comment_is_blanked_line_for_line():
    page = "# T\n\n```bash gpu\nqvla x\n```\n<!-- ci\n```bash ci\nqvla x --print\n```\n-->\nafter\n"
    assert site_hooks.site_markdown(page) == "# T\n\n```bash\nqvla x\n```\n\n\n\n\n\nafter\n"


def test_the_site_shows_exactly_the_visible_blocks():
    for rel, path in shipped_pages():
        text = path.read_text(encoding="utf-8")
        before, after = blocks(text), blocks(site_hooks.site_markdown(text))
        assert [(b.line, b.lang, b.text) for b in after] == \
               [(b.line, b.lang, b.text) for b in before if not b.hidden], rel
        assert all(b.tag == "" for b in after), rel


def test_the_hook_and_the_tester_hide_the_same_lines():
    for rel, path in shipped_pages():
        text = path.read_text(encoding="utf-8")
        assert {n + 1 for n in site_hooks.ci_lines(text.split("\n"))} == markdown.ci_lines(text), rel


def test_github_shows_no_hidden_block_and_hides_no_visible_one():
    for rel, path in shipped_pages():
        text = path.read_text(encoding="utf-8")
        gone = github_comment_lines(text)
        for b in blocks(text):
            lines = set(range(b.line, b.line + b.text.count("\n") + 2))
            assert (lines <= gone) if b.hidden else not (lines & gone), (rel, b.line)


def nav_pages(config: str) -> list[str]:
    """The pages of mkdocs.yml's `nav:` block, in order, relative to docs/."""
    nav = re.search(r"^nav:\n((?:[ \t]+.*\n?)*)", config, re.M).group(1)
    return re.findall(r"^\s*- (?:[^:\n]+: )?([\w./-]+\.md)\s*$", nav, re.M)


def test_nav_pages_reads_titled_and_untitled_entries_at_any_depth():
    config = ("site_name: x\nnav:\n  - Home: index.md\n  - Tutorials:\n      - tutorials/a.md\n"
              "      - B: tutorials/b.md\nmarkdown_extensions:\n  - tables\n")
    assert nav_pages(config) == ["index.md", "tutorials/a.md", "tutorials/b.md"]


def test_the_nav_lists_every_shipped_page_once():
    nav = nav_pages(public_file("mkdocs.yml").read_text(encoding="utf-8"))
    pages = sorted(rel.removeprefix("docs/") for rel, _ in shipped_pages() if rel.startswith("docs/"))
    assert len(nav) == len(set(nav)) and sorted(nav) == pages


SECTIONS = ["Home", "Install", "1 Reproduce the paper", "2 Train with your quality method", "3 Advanced", "Reference"]


def test_the_nav_sections_are_in_the_documented_order():
    config = public_file("mkdocs.yml").read_text(encoding="utf-8")
    nav = re.search(r"^nav:\n((?:[ \t]+.*\n?)*)", config, re.M).group(1)
    assert re.findall(r"^  - ([^:\n]+):", nav, re.M) == SECTIONS
