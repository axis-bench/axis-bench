"""Every `data` block of Install and of each tutorial, run as written: each fetches what it
reads through the release lock (`qvla data get`), so it runs only where that lock and the network
are, the CI `data` job (`pytest tests/docs -m slow`). The job pins how many pages pass; a page that
gains or loses its data blocks re-pins that count here and in ci.yml. The job has no renderer
environment, so its step ignores the render tier's docs tests whenever the tree has them."""
from __future__ import annotations

import pathlib
import re

import pytest

from tests.docs.runner import REPO, ci_job, ci_workflow, pages_with, run_page

DATA_PAGES = pages_with("data")
RENDER = ("tests", "docs", "render")          # the render tier's docs tests (the CI `render` job runs them)
_STEP = re.compile(r"^[ \t]*(python -m pytest tests/docs -m slow [^\n]*)\| tee pytest-docs\.log"
                   r"\n\s*! grep -qE '\[0-9\]\+ skipped' pytest-docs\.log"
                   r"\n\s*grep -qE '\^(\d+) passed, \[0-9\]\+ deselected' pytest-docs\.log", re.M)


def data_step_problems(ci: str, tree: pathlib.Path, pages: int) -> list[str]:
    """Why the data job of the workflow `ci` does not run the data blocks of `tree`'s `pages` pages:
    one step runs `pytest tests/docs -m slow`, fails on a skip, pins `pages` passed tests, and ignores
    the render tier's docs tests exactly when `tree` has them."""
    step = _STEP.search(ci_job("data", ci))
    if step is None:
        return ["ci.yml's data job has no tutorial data-block step"]
    out = [] if int(step.group(2)) == pages else [f"the step pins {step.group(2)} passed; {pages} pages have data blocks"]
    render = "/".join(RENDER)
    if (f"--ignore={render}" in step.group(1).split()) != tree.joinpath(*RENDER).is_dir():
        out.append(f"the step passes --ignore={render} exactly when the tree has that directory")
    return out


@pytest.mark.slow
@pytest.mark.parametrize("page", DATA_PAGES, ids=lambda p: p.stem)
def test_the_data_blocks_run_as_written(page, tmp_path):
    assert run_page(page, ("data",), tmp_path) == []


def test_the_data_job_runs_every_page_with_data_blocks():
    assert DATA_PAGES
    assert data_step_problems(ci_workflow(), REPO, len(DATA_PAGES)) == []


_CI = """name: ci
jobs:
  test:
    steps:
      - run: |
          python -m pytest tests/docs -m slow{flag} -q | tee pytest-docs.log
          ! grep -qE '[0-9]+ skipped' pytest-docs.log
          grep -qE '^2 passed, [0-9]+ deselected' pytest-docs.log
  data:
    steps:
      - run: |
          set -o pipefail
          # python -m pytest tests/docs -m slow -q | tee pytest-docs.log
          python -m pytest tests/docs -m slow{flag} --basetemp "$RUNNER_TEMP/docs" -q | tee pytest-docs.log
          ! grep -qE '[0-9]+ skipped' pytest-docs.log
          grep -qE '^{count} passed, [0-9]+ deselected' pytest-docs.log
"""


@pytest.mark.parametrize("has_render", [False, True])
@pytest.mark.parametrize("flag", [False, True])
def test_the_step_ignores_the_render_tier_exactly_when_the_tree_has_it(tmp_path, has_render, flag):
    if has_render:
        tmp_path.joinpath(*RENDER).mkdir(parents=True)
    ci = _CI.format(flag=f" --ignore={'/'.join(RENDER)}" if flag else "", count=3)
    found = data_step_problems(ci, tmp_path, 3)
    assert found == ([] if flag == has_render else
                     [f"the step passes --ignore={'/'.join(RENDER)} exactly when the tree has that directory"])


def test_the_step_is_read_from_the_data_job_and_its_count_is_checked(tmp_path):
    assert data_step_problems(_CI.format(flag="", count=1), tmp_path, 3) == ["the step pins 1 passed; 3 pages have data blocks"]
    no_data_step = _CI.format(flag="", count=3).split("  data:")[0]
    assert data_step_problems(no_data_step, tmp_path, 3) == ["ci.yml's data job has no tutorial data-block step"]
