"""Autouse HF environment isolation for every test under tests/golden/ (as tests/unit/conftest.py).

The T0 golden tests drive axis_qvla/paper/cli.py, which resolves an hf:// results source's endpoint and
token from the environment (axis_qvla.data.hf); a machine that runs this suite may export a real
HF_TOKEN. A test's own `monkeypatch.setenv` (tests/golden/paper/test_t0_fixture.py's HTTP-fixture
tests) runs after this fixture and still wins.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_real_hub(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:9")
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)
