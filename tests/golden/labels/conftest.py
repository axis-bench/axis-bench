"""Data fixtures for the slow label goldens. A missing option FAILS the test; it never skips.

The options are declared once, in tests/conftest.py (`DATA_OPTIONS`), and read through its
`require_option`.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from tests.conftest import require_option

REPO = pathlib.Path(__file__).resolve().parents[3]
LOCK = REPO / "conf" / "release" / "SIGNALS.lock.json"


def _dir(request, option: str) -> pathlib.Path:
    value = require_option(request.config, option)
    if not isinstance(value, pathlib.Path) or not value.is_dir():
        pytest.fail(f"{option} {value} is not a directory", pytrace=False)
    return value


@pytest.fixture(scope="session")
def traces_sample(request) -> pathlib.Path:
    return _dir(request, "--traces-sample")


@pytest.fixture(scope="session")
def scene_cache(request) -> pathlib.Path:
    return _dir(request, "--scene-cache")


@pytest.fixture(scope="session")
def signals_pack(request) -> pathlib.Path:
    pack = _dir(request, "--pretraining-data")
    if hashlib.sha256((pack / "SIGNALS.lock.json").read_bytes()).digest() != hashlib.sha256(LOCK.read_bytes()).digest():
        pytest.fail(f"{pack}/SIGNALS.lock.json is not this repository's conf/release/SIGNALS.lock.json")
    return pack


@pytest.fixture(scope="session")
def corpus(request) -> pathlib.Path:
    return _dir(request, "--episodes")


@pytest.fixture(scope="session")
def lock() -> dict:
    return json.loads(LOCK.read_text())
