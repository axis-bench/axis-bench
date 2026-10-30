"""Every relative link, image, HTML link and reference definition in the prose of every shipped page
(fenced blocks and code spans are code, and a `[^n]:` footnote is not a link) resolves to a file (or directory) of a
public checkout, and every `#anchor` it names is a heading of that page; every docs page and anchor
the code names in a message exists too. A target that is exactly a `{{pending:<field>}}` marker is an
input the README has not received yet; anything else that holds one is checked like any link."""
from __future__ import annotations

import pathlib
import posixpath
import re

from tests.docs.blocks import blocks, spans
from tests.docs.runner import REPO, public_file, shipped_pages

LINK = re.compile(r'\]\(([^)\s]+)(?:\s+"[^"]*")?\)|<img[^>]*\ssrc="([^"]+)"|<a[^>]*\shref="([^"]+)"'
                  r'|^ {0,3}\[(?!\^)[^\]\n]+\]:[ \t]*<?([^\s>]+)>?', re.M)
EXTERNAL = re.compile(r"^(?:https?:|mailto:)")
POINTER = re.compile(r"(docs/[\w./-]+\.md)#([\w-]+)")
PENDING = re.compile(r"\{\{\s*pending:\s*[^}\s]+\s*\}\}")
PAGE = "/".join(("docs", "tutorials", "t.md"))
# A renamed page whose old name a code message still carries: the pointer is checked against the new
# page until the code names the new one, and a row no code names any more fails.
RENAMED: dict[str, str] = {}


def slug(heading: str) -> str:
    """GitHub's anchor for a heading: lower case, punctuation dropped, spaces to hyphens."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def anchors(text: str) -> set[str]:
    """The anchors of a Markdown text's headings, outside fenced blocks."""
    out, fence = set(), None
    for line in text.splitlines():
        m = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if m and (fence is None or m.group(1).startswith(fence)):
            fence = None if fence else m.group(1)
            continue
        h = re.match(r"^#{1,6} (.+?)\s*#*$", line) if fence is None else None
        if h:
            out.add(slug(h.group(1)))
    return out


def prose(text: str) -> str:
    """`text` with the lines of its fenced blocks blanked and its code spans dropped (the line count
    is kept): what GitHub renders as links."""
    code = spans(blocks(text))
    return "\n".join("" if n in code else re.sub(r"(`+).+?\1", "", line)
                     for n, line in enumerate(text.splitlines(), 1))


def broken(rel: str, text: str) -> list[str]:
    out = []
    for m in LINK.finditer(prose(text)):
        target = next(g for g in m.groups() if g is not None)
        if EXTERNAL.match(target) or PENDING.fullmatch(target):
            continue
        path, _, anchor = target.partition("#")
        path = posixpath.normpath(posixpath.join(posixpath.dirname(rel), path)) if path else rel
        found = public_file(path)
        if found is None and not (REPO / path).is_dir():
            out.append(f"{rel} -> {target}")
        elif anchor and (found is None or anchor not in anchors(found.read_text(encoding="utf-8"))):
            out.append(f"{rel} -> {target}: no such heading")
    return out


def test_slugs_and_anchors_follow_github():
    assert slug("From a local snapshot") == "from-a-local-snapshot"
    assert slug("`qvla train`") == "qvla-train"
    assert anchors("# Top\n```bash\n# not a heading\n```\n## Data on this machine\n") == {"top", "data-on-this-machine"}


def test_the_checker_finds_a_broken_link_or_anchor_and_skips_the_rest():
    text = ("[a](../reference/data.md#from-a-local-snapshot) [b](https://x.org) [d]({{pending:paper_url}}) "
            '[e](nowhere.md) <img src="../assets/none.gif"> [f](../reference/data.md#no-such-heading) '
            "[g]({{pendng:paper_url}}) [h](x{{pending:paper_url}})")
    assert broken(PAGE, text) == [f"{PAGE} -> nowhere.md", f"{PAGE} -> ../assets/none.gif",
                                  f"{PAGE} -> ../reference/data.md#no-such-heading: no such heading",
                                  f"{PAGE} -> {{{{pendng:paper_url}}}}", f"{PAGE} -> x{{{{pending:paper_url}}}}"]


def test_the_checker_reads_html_links_and_reference_definitions():
    text = ('<a href="nowhere.md">x</a> <a href="https://x.org">y</a>\n'
            "[r]: ../reference/data.md#no-such-heading\n[s]: <../reference/data.md>\n")
    assert broken(PAGE, text) == [f"{PAGE} -> nowhere.md",
                                  f"{PAGE} -> ../reference/data.md#no-such-heading: no such heading"]


def test_the_checker_skips_code_and_footnotes():
    text = ("[^1]: See the paper.\n```toml\n[x]: nowhere.md\nsee [a](nowhere.md)\n```\n"
            "Text[^1], `[b](code-span.md)` and [`code in a link`](nowhere.md).\n")
    assert broken(PAGE, text) == [f"{PAGE} -> nowhere.md"]


def test_every_relative_link_of_every_shipped_page_resolves():
    found = [b for rel, path in shipped_pages() for b in broken(rel, path.read_text(encoding="utf-8"))]
    assert found == []


def test_every_docs_anchor_the_code_names_exists():
    import axis_qvla as _pkg                      # the package, under whichever name this tree gives it

    root = pathlib.Path(_pkg.__file__).parent
    named = [(p, rel, a) for p in sorted(root.rglob("*.py")) for rel, a in POINTER.findall(p.read_text(encoding="utf-8"))]
    bad = [f"{p.relative_to(root)}: {rel}#{a}" for p, rel, a in named
           if (page := public_file(RENAMED.get(rel, rel))) is None or a not in anchors(page.read_text(encoding="utf-8"))]
    assert bad == []
    assert sorted(set(RENAMED) - {rel for _, rel, _ in named}) == [], "stale RENAMED rows: no code names the old page"
