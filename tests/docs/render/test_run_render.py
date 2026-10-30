"""Every `ci-render` block of the tutorials, run in page order in the renderer's environment with a
software GL (the CI `render` job). The blocks render with the 2F-85 model that the axis-render
submodule ships, so the job checks that submodule out."""
from __future__ import annotations

import pytest

from tests.docs.runner import REPO, pages_with, run_page

ROBOTS = REPO / "third_party" / "axis-render" / "assets" / "robots"


def test_the_2f85_the_blocks_render_with_is_checked_out():
    assert (ROBOTS / "robotiq_2f85").is_dir(), f"{ROBOTS} is not there: check out third_party/axis-render"


@pytest.mark.parametrize("page", pages_with("ci-render"), ids=lambda p: p.stem)
def test_the_render_blocks_run_as_written(page, tmp_path):
    assert run_page(page, ("ci-render",), tmp_path) == []
