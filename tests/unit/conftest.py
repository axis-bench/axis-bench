"""Autouse HF environment isolation for every test under tests/unit/.

axis_qvla.data.hf.endpoint_from_env() defaults to the real Hub and token_from_env() reads
HF_TOKEN -- both may be set for real on a machine that runs this suite. axis_qvla/paper/cli.py resolves
an hf:// results source, and `qvla data verify` the Hub, through exactly those two helpers, so a
test that forgets to override either would send a real token to the real Hub -- a mistake this
repository has already made once, caught by luck rather than by a test. tests/golden/conftest.py
holds the same fixture for the golden tests.

`monkeypatch` is function-scoped and cached per test, so this fixture and a test's own
`monkeypatch.setenv`/`delenv` calls (e.g. tests/unit/data/test_hf.py's `test_env_helpers`)
share the same underlying object; this one runs first (autouse), and a test's own calls run
after it in the test body, so they still win and are undone in the right order at teardown.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_real_hub(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:9")
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)
