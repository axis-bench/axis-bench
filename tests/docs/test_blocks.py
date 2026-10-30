"""The block rules (tests/docs/blocks.py) on made-up pages, and every page against them (the runner's
tests are tests/docs/test_runner.py, the shell forms' tests/docs/test_shell.py)."""
from __future__ import annotations

import pytest

from tests.docs import blocks as bl
from tests.docs import runner

GOOD = """# A page

```bash data
qvla data get results --dest r
```

```text output
IDENTICAL…
```

```bash ci timeout=30
echo IDENTICAL 13
```

```python file=src/x.py
x = 1
```
"""


def test_blocks_reads_language_tag_and_attributes():
    b = bl.blocks(GOOD)
    assert [(x.line, x.lang, x.tag, x.attr("file"), x.timeout) for x in b] == [
        (3, "bash", "data", None, 600), (7, "text", "output", None, 600),
        (11, "bash", "ci", None, 30), (15, "python", "", "src/x.py", 600)]
    assert b[2].text == "echo IDENTICAL 13\n" and b[0].shell and not b[1].shell


def test_a_longer_fence_holds_a_shorter_one_and_tildes_work():
    (only,) = bl.blocks("````markdown\n```bash ci\nx\n```\n````\n")
    assert only.lang == "markdown" and only.text == "```bash ci\nx\n```\n"
    (tilde,) = bl.blocks("~~~bash ci\necho a\n~~~\n")
    assert (tilde.tag, tilde.text) == ("ci", "echo a\n")


def test_a_page_that_keeps_the_rules_has_no_problems():
    assert bl.problems(GOOD) == []


@pytest.mark.parametrize("page, why", [
    ("```bash\nls\n```\n```bash ci\nls\n```\n", "needs one of the tags"),
    ("```bash gpu\nnvidia-smi\n```\n", "needs a `ci` or `ci-render` twin"),
    ("```bash gpu\nx\n```\n```bash data\ny\n```\n```bash ci\nz\n```\n", "needs a `ci` or `ci-render` twin"),
    ("```bash setup\npip install .\n```\n", "needs a `ci` or `ci-render` twin"),
    ("```bash gpu\nx\n```\n```text output\nok\n```\n```bash ci\ny\n```\n", "follows a `ci`, `ci-render` or `data` block"),
    ("```python\nx = 1\n```\n```bash ci\ny\n```\n", "needs `output` or `file=<path>`"),
    ("```bash ci timeout=soon\ny\n```\n", "whole seconds"),
    ("```bash data\nx\n```\n```bash data\ny\n```\n", "no `ci` or `ci-render` block"),
    ("```bash gpu\nqvla train env plan openpi\n```\n```bash ci\nqvla --help\n```\n",
     "`qvla train env plan openpi` runs in no"),
    ("```bash gpu\npython -m lerobot.scripts.axis_merge_datasets --out m a b\n```\n```bash ci\nqvla --help\n```\n",
     "`python -m lerobot.scripts.axis_merge_datasets` runs in no"),
    # a fence the parser does not open: in a list item, tagged or not; behind a tab; after a list
    # marker; in a quote
    ("1. Run:\n\n    ```bash ci\n    echo a\n    ```\n\n```bash ci\nls\n```\n", "line 3: a fence this parser does not open"),
    ("1. Run:\n\n    ```bash\n    qvla train a.toml\n    ```\n\n```bash ci\nls\n```\n", "line 3: a fence this parser does not open"),
    ("\t```bash\n\tls\n\t```\n```bash ci\nls\n```\n", "line 1: a fence this parser does not open"),
    ("- ```bash ci\n  ls\n  ```\n\n```bash ci\nls\n```\n", "line 1: a fence this parser does not open"),
    ("> ```bash ci\n> ls\n> ```\n```bash ci\nls\n```\n", "line 1: a fence this parser does not open"),
    ("```bash ci gpu\nls\n```\n", "takes one tag, not 'ci gpu'"),
    ("```bash ci\n# nothing to run\n\n```\n", "a `ci` block runs nothing"),
    ("```bash data\nqvla data get results --dest r\nqvla paper tables --results r\n```\n```bash ci\nqvla --help\n```\n",
     "the twin of a `data` block runs none of its commands (qvla paper tables)"),
    ("```bash gpu\ntimeout 60 qvla train a.toml\n```\n```bash ci\nqvla train a.toml --no-bind\n```\n",
     "line 2: `timeout 60 qvla train a.toml` runs qvla or python in a form whose command key cannot be read"),
    # the shell forms (tests/docs/test_shell.py has them all), named with the line that holds them
    ("```bash ci\n! qvla --help\necho next\n```\n", "line 2: `! qvla --help`: `!` stands only before the block's last"),
    ("```bash ci\nfalse && true\necho after\n```\n", "line 2: `false && true`: an `&&` list"),
    ("```bash ci\nqvla x || true\n```\n", "line 2: `qvla x || true`: an `||` list"),
    ("```bash gpu\nset +e\n```\n```bash ci\nls\n```\n", "line 2: `set +e`: `set` changes the block's shell"),
    # Markdown that hides code: a plain indented code block, a fence in an HTML comment
    ("Run:\n\n    qvla train a.toml\n\n```bash ci\nls\n```\n", "line 3: an indented code block has no tag"),
    ("<!--\n```bash ci\ntouch x\n```\n-->\n```bash ci\nls\n```\n", "line 2: a fence inside an HTML comment"),
    # background blocks (tests/docs/test_servers.py has the rest)
    ("```bash ci background=1\nsleep 9\n```\n```bash ci\nls\n```\n", "a background block runs exactly one command"),
    ("```bash ci background=1\npython -m axis_qvla.bench.servers.lerobot_server --port 8001\n```\n"
     "```text output\nx\n```\n```bash ci\nqvla bench check-server --policy 127.0.0.1:8001\n```\n",
     "that is not a background one"),
    # CI comments: only stand-ins hide; a comment closes on a line of its own; nothing in <details>
    ("<!-- ci\n```bash gpu\nqvla train a.toml\n```\n-->\n```bash ci\nls\n```\n",
     "only `ci`, `ci-render` and `output` blocks may sit in a CI comment"),
    ("<!-- ci\n```bash ci\nls\n```\n", "a `<!-- ci` comment is never closed"),
    ("<!-- ci\n```bash ci\nls\n```\nok -->\n", "closes with a line `-->` of its own"),
    ("<!-- ci\n```bash ci\nls\n-->\n```\n", "a block in a CI comment ends after the comment closes"),
    ("```bash ci\npython -m tests.docs.fixtures toy-scores x.json\n```\n", "a visible block runs a CI stand-in"),
    ("```bash ci\nqvla train a.toml --no-bind\n```\n", "a visible block runs a CI stand-in"),
    ('<details class="note" markdown>\n\n```bash ci\nls\n```\n\n</details>\n', "a fence inside a <details> box"),
    # any other comment still hides nothing
    ("<!-- note\n```bash ci\ntouch x\n```\n-->\n```bash ci\nls\n```\n", "line 2: a fence inside an HTML comment"),
])
def test_each_broken_rule_is_named(page, why):
    found = bl.problems(page)
    assert any(why in p for p in found), found


def test_what_the_rules_allow_passes():
    page = ("1. Run:\n\n   ```bash ci\n   ls\n   ```\n\n"           # three spaces: still a fence
            "```bash file=x.sh\nfor x in a; do b; done\n```\n"         # a shell excerpt, never run
            "- a list item\n\n  its next paragraph\n\n  - a nested item\n\n"   # list content, not code
            "```bash ci background=1\npython -m axis_qvla.bench.servers.lerobot_server --port 8001\n```\n"
            "```bash ci\nqvla bench check-server --policy 127.0.0.1:8001 --timeout 60\n```\n"   # its readiness check
            "```bash ci\n"
            "# a comment\n"
            "export QVLA_X=1 Y=\"$PWD/a\"\n"
            "Z=2 qvla --help | grep -q usage\n"
            "qvla --help > out.txt 2>&1\n"
            "qvla --help \\\n  >> out.txt\n"
            "grep -c usage out.txt |\n  tee n.txt\n"
            "[ -s n.txt ]\n"
            "cat <<< 'here-string'\n"
            "! false\n"                                                  # the last command: its status is the block's
            "```\n")
    assert bl.problems(page) == []


def test_a_twin_in_a_ci_comment_is_opened_hidden_and_keeps_the_rules():
    """The pages hide each stand-in twin in a `<!-- ci` comment; its fences stay at the margin, so the
    parser opens them, marks them hidden, and the twin rule still pairs the at-scale block with it."""
    page = ("```bash gpu\nqvla train a.toml\n```\n\n"
            "<!-- ci\nThe same on the smoke fixture.\n\n"
            "```bash ci\nqvla train a.toml --print\n```\n\n```text output\nOK\n```\n-->\n")
    assert [(b.tag, b.hidden, b.text) for b in bl.blocks(page)] == [
        ("gpu", False, "qvla train a.toml\n"), ("ci", True, "qvla train a.toml --print\n"), ("output", True, "OK\n")]
    assert bl.problems(page) == []


def test_a_visible_output_may_follow_a_hidden_twin():
    page = ("```bash gpu\nqvla train a.toml\n```\n<!-- ci\n```bash ci\nqvla train a.toml --print\n```\n-->\n"
            "```text output\nOK\n```\n")
    assert [b.hidden for b in bl.blocks(page)] == [False, True, False]
    assert bl.problems(page) == []


def test_ci_regions_follow_the_html_block_rule():
    from tests.docs import markdown

    text = "a\n<!-- ci\nx\n-->\nb\n   <!-- ci\ny\n-->\n    <!-- ci\nz\n-->\n"
    assert markdown.ci_regions(text) == [(2, 4), (6, 8)]      # four spaces: a code block, not a comment
    assert markdown.ci_lines(text) == {2, 3, 4, 6, 7, 8}
    assert markdown.ci_regions("<!-- ci\nx\n") == [(1, None)]


def test_no_page_shows_a_details_box_or_a_visible_stand_in():
    for p in runner.pages():
        text = p.read_text(encoding="utf-8")
        assert "Check it on a small stand-in" not in text, p.name
        assert not [b.line for b in bl.blocks(text)
                    if not b.hidden and b.shell and b.tag in bl.RUN_TAGS and bl.STAND_IN.search(b.text)], p.name


def test_an_excerpt_must_be_verbatim_and_its_file_must_exist(tmp_path):
    (tmp_path / "x.py").write_text("a = 1\nb = 2\n")

    def resolve(rel):
        return tmp_path / rel if (tmp_path / rel).is_file() else None

    page = "```python file=x.py\nb = 2\n```\n```python file=x.py\nb = 3\n```\n```python file=y.py\nc\n```\n"
    assert bl.excerpt_problems(page, resolve) == ["line 4: not a verbatim excerpt of x.py",
                                                  "line 7: y.py is not in the repository"]


def test_output_lines_match_in_order_within_one_line():
    got = "OK    pi05: /scratch/a/roots/cache/derived/smoke/runs/pi05.toml\nIDENTICAL  13 outputs\n"
    assert bl.output_problems("OK    pi05: …/derived/smoke/runs/pi05.toml\nIDENTICAL  13 outputs\n", got) == []
    assert bl.output_problems("runs/pi05.toml…OK\n", got) == ["runs/pi05.toml…OK"]


def test_there_are_pages_to_run():
    assert runner.pages(), "docs/tutorials/ holds no page"


def test_every_page_keeps_the_block_rules():
    found = {p.name: bl.problems(p.read_text(encoding="utf-8")) for p in runner.pages()}
    assert {k: v for k, v in found.items() if v} == {}


def test_every_excerpt_on_a_shipped_page_is_verbatim():
    found = {rel: bl.excerpt_problems(p.read_text(encoding="utf-8"), runner.public_file)
             for rel, p in runner.shipped_pages()}
    assert {k: v for k, v in found.items() if v} == {}
