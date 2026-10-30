"""Data fixtures for the slow signal goldens. A missing option FAILS the test; it never skips.

The options themselves are declared once, in tests/conftest.py (`DATA_OPTIONS`), and read through
its `require_option`.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from tests.conftest import require_option

REPO = pathlib.Path(__file__).resolve().parents[3]
LOCK = REPO / "conf" / "release" / "SIGNALS.lock.json"


def data_dir(request, option: str) -> pathlib.Path:
    value = require_option(request.config, option)
    if not isinstance(value, pathlib.Path) or not value.is_dir():
        pytest.fail(f"{option} {value} is not a directory", pytrace=False)
    return value


@pytest.fixture(scope="session")
def lock() -> dict:
    return json.loads(LOCK.read_text())


@pytest.fixture(scope="session")
def signals_pack(request) -> pathlib.Path:
    pack = data_dir(request, "--pretraining-data")
    if hashlib.sha256((pack / "SIGNALS.lock.json").read_bytes()).digest() != hashlib.sha256(LOCK.read_bytes()).digest():
        pytest.fail(f"{pack}/SIGNALS.lock.json is not this repository's conf/release/SIGNALS.lock.json")
    return pack


@pytest.fixture(scope="session")
def corpus(request) -> pathlib.Path:
    return data_dir(request, "--episodes")


@pytest.fixture(scope="session")
def paper_rowspaces(signals_pack, corpus) -> dict:
    from axis_qvla.signals.rowspace import paper_rowspaces as build
    return build(signals_pack, corpus)
