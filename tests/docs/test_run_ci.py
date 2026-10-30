"""Every `ci` block of Install and of each tutorial, run in page order as a reader would, in a
scratch directory showing the public layout (the CI `test` job runs this file), and every run tag a
page uses has a CI job that runs it."""
from __future__ import annotations

import pytest

from tests.docs.runner import ci_job, pages_with, run_page

HERE = "tests/docs/test_run_ci.py"


@pytest.mark.parametrize("page", pages_with("ci"), ids=lambda p: p.stem)
def test_the_ci_blocks_run_as_written(page, tmp_path):
    assert run_page(page, ("ci",), tmp_path) == []


def _pytest_lines(job: str) -> list[str]:
    return [s for line in ci_job(job).splitlines() if (s := line.strip()).startswith("python -m pytest ")]


def test_the_test_job_runs_this_file():
    (line,) = _pytest_lines("test")
    toks = line.split("|", 1)[0].split()
    ignored = [t.split("=", 1)[1].rstrip("/") for t in toks if t.startswith("--ignore=")]
    assert "tests" in toks and not any(HERE == d or HERE.startswith(d + "/") for d in ignored)


def test_a_page_uses_ci_render_only_once_the_render_job_runs_the_docs_tests():
    """`ci-render` blocks run in the render job (the renderer's environment). Until that job runs a
    docs test, a `ci-render` block would run nowhere, so no page may use the tag."""
    if pages_with("ci-render"):
        assert any(" tests/docs" in line for line in _pytest_lines("render")), [p.name for p in pages_with("ci-render")]
