"""The README is the entry point (RoboTwin's README is the model): a centred title, the paper's title,
a row of badges, the overview video's poster linking to the website, then exactly the sections below,
in this order, in at most 140 lines. Its media are the overview figure, five GIFs and the real-world
workspace still; Usage is the three entries, and the one command it gives there is one line that a
tutorial runs. A `{{pending:<field>}}` marker is an input filled at publication."""
from __future__ import annotations

import pathlib
import re

from tests.docs.blocks import RUN_TAGS, blocks
from tests.docs.runner import REPO, pages, public_file

TEXT = public_file("README.md").read_text(encoding="utf-8")
LINES = TEXT.splitlines()
HEAD = TEXT.split("\n# ", 1)[0]
SECTIONS = ["Overview", "Updates", "Installation", "Task information", "Real-world pretraining data", "Usage",
            "Data and checkpoints", "Citation", "Acknowledgement", "License"]
BADGES = [("Paper", "https://arxiv.org/abs/2610.18869"), ("Website", "https://axis-bench.github.io/"),
          ("Docs", "https://axis-bench.github.io/axis-bench/"), ("Data", "https://huggingface.co/axisrobotics"),
          ("AXIS paper", "https://arxiv.org/abs/2607.21588"), ("License", "LICENSE")]
VIDEO = "https://github.com/user-attachments/assets/abdf2851-a6c9-43d9-9556-e4a4b285cb27"
MEDIA = ["docs/assets/overview.png", "docs/assets/pretraining-tasks.gif",
         "docs/assets/quality-levels.gif", "docs/assets/evaluation.gif", "docs/assets/realworld-demo.gif",
         "docs/assets/realworld-workspace.jpg", "docs/assets/realworld-pretraining.gif"]
CITED = ("@article{yan2026matters,", "@article{zhao2026axis,")
MAX_LINES = 140
# The page's weight as built (the overview video is a GitHub attachment, not a repository file): the overview figure (0.6 MB), GIFs of 7.0, 3.8, 3.9,
# 3.8 (the real-robot comparison) and 4.0 MB and the workspace still (0.02 MB): 23.19 MB.
MAX_MEDIA_BYTES = 23_200_000
MAX_FILE_BYTES = 8_000_000           # any one of them: a GIF over this is too slow on a page


def _section(title: str) -> str:
    return TEXT.split(f"\n# {title}\n", 1)[1].split("\n# ", 1)[0]


def test_the_readme_is_at_most_140_lines():
    assert len(LINES) <= MAX_LINES, len(LINES)


def test_title_badges_then_exactly_the_sections_in_order():
    assert LINES[0] == '<h1 align="center">AXIS-Bench</h1>'
    badges = re.findall(r'^  <a href="([^"]+)"><img src="https://img\.shields\.io/[^"]+" alt="([^"]+)"></a>$', HEAD, re.M)
    assert [(alt, href) for href, alt in badges] == BADGES
    assert re.findall(r"^# (.+)$", TEXT, re.M) == SECTIONS


def test_the_overview_video_plays_inline_with_the_website_link_and_no_author_list_ships():
    assert VIDEO in LINES                    # a bare attachment URL on its own line renders as a player
    assert '<a href="https://axis-bench.github.io/">website</a>' in HEAD
    assert "<sup>" not in TEXT and "Corresponding author" not in TEXT


def test_the_media_are_the_overview_figure_the_gifs_and_the_workspace_still_within_the_budget():
    images = [i for i in re.findall(r'<img src="([^"]+)"', TEXT) if not i.startswith("https://")]
    assert images == MEDIA
    assert sum(public_file(i).stat().st_size for i in images) <= MAX_MEDIA_BYTES
    assert {i: n for i in images if (n := public_file(i).stat().st_size) > MAX_FILE_BYTES} == {}


def test_updates_are_dated_and_newest_first():
    items = [line for line in _section("Updates").splitlines() if line.strip()]
    dates = [re.match(r"^- \*\*(\d{4}/\d{2}/\d{2}|\{\{pending:updates\}\})\*\* \S", line) for line in items]
    assert items and all(dates), items
    known = [d.group(1) for d in dates if d.group(1)[0].isdigit()]
    assert known == sorted(known, reverse=True) and all(d.group(1)[0].isdigit() for d in dates[1:]), items


def test_both_papers_are_cited():
    assert all(key in _section("Citation") for key in CITED)


ENTRIES = [("Reproduce the paper", "docs/tutorials/reproduce-from-results.md"),
           ("Train with your quality method", "docs/tutorials/write-a-quality-method.md"),
           ("Connect your own model or robot", "docs/tutorials/connect-your-own-model.md")]


def test_usage_is_the_three_entries_and_its_command_runs_on_their_pages():
    """Usage holds no code block; it links the documentation and the three entries in the documented order, and
    each `qvla` command it gives is one line that a run block of a tutorial runs as written."""
    usage = _section("Usage")
    assert blocks(usage) == [] and "(https://axis-bench.github.io/axis-bench/)" in usage
    links = re.findall(r"^- \[([^\]]+)\]\(([^)]+)\)", usage, re.M)
    assert links == ENTRIES
    ran = {line.strip() for p in pages() for b in blocks(p.read_text(encoding="utf-8"))
           if b.shell and b.tag in RUN_TAGS for line in b.text.splitlines()}
    commands = re.findall(r"`(qvla [^`]+)`", usage)
    assert [c for c in commands if c not in ran] == []


def test_no_section_holds_what_belongs_in_the_documentation():
    for word in ("Troubleshooting", "Architecture", "Protocol", "Recipe", "Results"):
        assert not re.search(rf"^#+ .*{word}", TEXT, re.M | re.I), word


# The README's install lines, one per line: a committed file, so this public test holds no command
# string that starts `git` (a public test runs no git, and holds no git command line).
INSTALL = (pathlib.Path(__file__).resolve().parent / "readme_install.txt").read_text(encoding="utf-8").splitlines()
GOAL_EXTRAS = ("reproduce", "eval", "method", "realrobot")


def test_the_install_lines_are_the_install_page_s():
    """README Installation shows the one way to install, and it is the Install page's: the same clone and
    pip line, then `qvla init`, which a run block of the Install page runs. The Install page names every
    goal's extra, and each is an extra the package declares."""
    import tomllib

    install = (REPO / "docs" / "install.md").read_text(encoding="utf-8")
    page = [line.strip() for b in blocks(install) if b.shell and not b.hidden for line in b.text.splitlines()
            if line.strip()]
    (block,) = blocks(_section("Installation"))
    readme = [c.strip() for line in block.text.splitlines() for c in line.split("&&") if c.strip()]
    assert readme == INSTALL
    assert [c for c in readme if c not in page] == []
    ran = {line.strip() for b in blocks(install) if b.shell and b.tag in RUN_TAGS for line in b.text.splitlines()}
    assert "qvla init" in ran
    pyproject = public_file("pyproject.toml.tmpl") or public_file("pyproject.toml")   # the template, else the rendered file
    declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["optional-dependencies"]
    assert [x for x in GOAL_EXTRAS if x not in declared or f"`{x}`" not in install] == []
