"""No published source or data file hardcodes a machine path.

The repository grandfathers nothing, so there is no allow-list at all (an empty constant tested
against itself would assert nothing). Every .py, .sh, .toml, .json and
.md file the repository ships is scanned: its own top-level directories and root files, walked (the
suite runs from a source archive as well as from a git checkout). What a user creates in a checkout
-- a venv, `qvla.toml`, the rollout logs `qvla bench eval` writes under `eval_results/`, caches --
is not the repository's and is never scanned. The roots are joined at run time so this file does not
match its own pattern.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_ROOTS = "|".join(("bigdata", "rhome", "mnt", "data", "home", "disk", "localdisk", "root", "opt"))
PATTERN = re.compile(rf"(?<![\w.}}$-])/(?:{_ROOTS})/[A-Za-z0-9_.-]+")
SUFFIXES = (".py", ".sh", ".toml", ".json", ".md")
# What the repository ships at its top level (third_party/ holds the submodules, scanned in their
# own repositories). A missing entry fails the scan.
SHIPPED_DIRS = ("axis_qvla", "conf", "docs", "experiments", "tests", ".github")
SHIPPED_ROOT_FILES = ("README.md", "pyproject.toml", "qvla.example.toml")
# Never shipped source, wherever they sit: version control, caches and build metadata.
_SKIP_ANYWHERE = {".git", "__pycache__", ".pytest_cache"}


def _raise(exc: OSError) -> None:
    raise exc   # an unreadable or missing directory must fail the scan, never read as empty


def _shipped() -> list[str]:
    """Every shipped text file: the root files, and every file under the shipped directories."""
    out = list(SHIPPED_ROOT_FILES)
    for top in SHIPPED_DIRS:
        for d, dirs, files in os.walk(REPO / top, onerror=_raise):
            rel = Path(d).relative_to(REPO)
            dirs[:] = sorted(x for x in dirs if x not in _SKIP_ANYWHERE)
            out += [(rel / f).as_posix() for f in sorted(files) if f.endswith(SUFFIXES)]
    return out


def test_there_are_files_to_scan():
    assert len(_shipped()) > 100


@pytest.mark.parametrize("rel", _shipped())
def test_no_machine_paths(rel):
    hits = PATTERN.findall((REPO / rel).read_text(encoding="utf-8", errors="ignore"))
    assert not hits, f"{rel} hardcodes {sorted(set(hits))[:3]}"


def test_the_pattern_catches_a_box_path_and_spares_urls():
    assert PATTERN.search("x = '" + "/" + "disk/demo/r1'")
    assert not PATTERN.search("https://example.com/data/x")
