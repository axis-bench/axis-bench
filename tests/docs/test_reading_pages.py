"""Reading pages show usage and link the rules. The Reference section is exactly the declared set,
every hash, digest, verification and exit-code rule lives on one page, "How results are verified", and
no reading page writes one out or shows a long hex id."""
from __future__ import annotations

import re

import pytest

from tests.docs.blocks import blocks
from tests.docs.runner import REPO, public_file, shipped_pages

REFERENCE = ("cli", "data", "results-format", "reporting-rules", "policy-server", "troubleshooting", "verification")
# The reading pages: everything a user follows step by step (README and the reference pages are not).
READING = ("docs/index.md", "docs/install.md", "docs/tutorials/")
# The site's pages in the documented order: Home, Install, the three goals, then Reference.
NAV = ["index.md", "install.md",
       "tutorials/reproduce-from-results.md", "tutorials/reeval-released-checkpoints.md",
       "tutorials/write-a-quality-method.md", "tutorials/pretrain.md", "tutorials/fine-tune-and-evaluate.md",
       "tutorials/relabel-with-a-new-reward.md",
       "tutorials/connect-your-own-model.md", "tutorials/real-world-your-workspace.md",
       "reference/cli.md", "reference/data.md", "reference/results-format.md", "reference/reporting-rules.md",
       "reference/policy-server.md", "reference/verification.md", "reference/troubleshooting.md"]
RUBRIC = re.compile(r"sha256|\bdigests?\b|byte for byte|byte-identical|\bfingerprint|\bexits? (?:code |status )?\d"
                    r"|integrity failure|\bchecksum", re.I)
# A long id holds a letter: a run of digits is a number (a quaternion component, a count), not an id.
HEX = re.compile(r"\b(?=[0-9]*[a-f])[0-9a-f]{12,}\b")


def _reading():
    return [(rel, p.read_text(encoding="utf-8")) for rel, p in shipped_pages() if rel.startswith(READING)]


def test_the_reference_section_is_the_declared_set():
    assert sorted(p.stem for p in (REPO / "docs" / "reference").glob("*.md")) == sorted(REFERENCE)


def test_no_reading_page_writes_out_a_verification_rule():
    from tests.docs.markdown import ci_lines

    found = []
    for rel, text in _reading():
        hidden = ci_lines(text)                  # CI comments: what only CI reads
        visible = "\n".join(line for n, line in enumerate(text.splitlines(), 1) if n not in hidden)
        found += [f"{rel}: {m.group(0)!r}" for m in RUBRIC.finditer(visible)]
    assert found == [], "link docs/reference/verification.md instead"


def test_no_reading_page_shows_a_long_hex_id():
    assert [(rel, m.group(0)) for rel, text in _reading() for m in HEX.finditer(text)] == []


def test_the_verification_page_holds_each_rule_family():
    text = public_file("docs/reference/verification.md").read_text(encoding="utf-8")
    for heading in ("## Downloads", "## The paper's tables", "## Quality labels and the human study",
                    "## Quality labels and compiled labels",
                    "## Training runs", "## The benchmark", "## Evaluation runs", "## Real robot", "## Exit codes"):
        assert f"\n{heading}\n" in text, heading


def test_the_nav_is_the_documented_one():
    from tests.docs.test_site import nav_pages

    assert nav_pages(public_file("mkdocs.yml").read_text(encoding="utf-8")) == NAV


def test_the_real_robot_page_is_eight_steps_then_the_serving_contract():
    """The page ends at serving: its last section states what the robot's client sends and executes,
    with no driving steps."""
    text = public_file("docs/tutorials/real-world-your-workspace.md").read_text(encoding="utf-8")
    assert re.findall(r"^## (\d)\. ", text, re.M) == [str(n) for n in range(1, 9)]
    assert re.findall(r"^## .*", text, re.M)[-1] == "## Connect your robot"
    tail = text.split("\n## Connect your robot\n", 1)[1]
    assert blocks(tail) == [] and "(../reference/policy-server.md)" in tail


def test_the_server_example_is_about_thirty_lines():
    text = public_file("docs/tutorials/connect-your-own-model.md").read_text(encoding="utf-8")
    (ex,) = [b for b in blocks(text) if b.attr("file") == "experiments/examples/policy_server.py"]
    assert 20 <= ex.text.count("\n") <= 40
