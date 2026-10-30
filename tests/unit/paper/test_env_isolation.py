"""Proves tests/unit/conftest.py's autouse HF-isolation fixture is actually wired in.

axis_qvla/paper/cli.py resolves an hf:// pack's token and endpoint through
axis_qvla.data.hf.{token_from_env,endpoint_from_env}; a T0 test must never see a real
HF_TOKEN just because the machine that runs the suite happens to export one.
Run with HF_TOKEN=dummy exported in the shell to prove the outer environment cannot leak
through -- this test takes no monkeypatch of its own, so it sees exactly what the autouse
fixture left behind.
"""
from __future__ import annotations

from axis_qvla.data.hf import endpoint_from_env, token_from_env


def test_autouse_fixture_hides_a_real_token_and_pins_a_loopback_endpoint():
    assert token_from_env() is None
    assert endpoint_from_env() == "http://127.0.0.1:9"
