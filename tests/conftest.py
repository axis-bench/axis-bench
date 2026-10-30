"""Every command-line option the test suite accepts, declared in one place.

Why one place: pytest refuses an option registered twice and ignores one registered in a conftest
it does not load first. Options, not environment variables: an option names the data it points at
on the command line.

The data options name local copies of published data for the slow golden tests. Those tests are
marked and deselected by default; a test that is SELECTED and finds its option missing must fail
(`require_option`), never skip, because a skipped test reads as a passing one.

This file is the one place the options are declared: a test that needs a new data option adds it here.
"""
from __future__ import annotations

import pathlib

import pytest

DATA_OPTIONS = {
    # labels and training weights
    "--pretraining-data": "a qvla-v1-corpus tree: index/, signals/v1/ and bundles/ once staged",
    "--episodes": "its episodes directory (task_*/meta, task_*/data, task_*/videos)",
    "--traces-sample": "a staged traces_sample/ tree",
    "--scene-cache": "a directory of _scene_task_<id>/scene.xml MJCF scenes",
    # training
    "--bench": "a qvla-v1-bench tree: ft_demos/splits/ and each finetune task's meta/",
    # real robot
    "--render-source": "source_episode.json of the renderer's reference attempt",
    "--render-oracle": "published render task directory (LeRobot v3) of that attempt",
    "--render-oracle-episode": "episode index of the reference attempt in that task",
    "--menagerie": "a MuJoCo Menagerie checkout matching menagerie.lock.json",
}


def pytest_addoption(parser):
    group = parser.getgroup("qvla-data", "local copies of published data, for the slow golden tests")
    for name, help_text in DATA_OPTIONS.items():
        group.addoption(name, default=None, help=help_text)


def require_option(config: pytest.Config, name: str) -> pathlib.Path | str:
    """The option's value, as a Path when it names an existing file or directory. A missing value
    fails the test that asked for it; it never skips."""
    if name not in DATA_OPTIONS:
        raise KeyError(f"{name} is not declared in tests/conftest.py DATA_OPTIONS")
    value = config.getoption(name)
    if value is None:
        pytest.fail(f"this test needs {name} ({DATA_OPTIONS[name]})", pytrace=False)
    path = pathlib.Path(value)
    return path if path.exists() else value
