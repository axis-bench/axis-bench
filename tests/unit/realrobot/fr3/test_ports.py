"""The port poll the serve step and the handshake probe share."""
from __future__ import annotations

import socket

import pytest

from axis_qvla.realrobot.fr3 import ports


@pytest.fixture
def listening():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen()
    yield s.getsockname()[1]
    s.close()


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_knock_tells_a_listening_port_from_a_free_one(listening):
    assert ports.knock("127.0.0.1", listening) and not ports.knock("127.0.0.1", _free_port())


def test_wait_for_port_returns_once_listening_and_refuses_at_the_timeout(listening):
    ports.wait_for_port("127.0.0.1", listening, 1.0)
    with pytest.raises(ports.NotListening, match="no server listening"):
        ports.wait_for_port("127.0.0.1", _free_port(), 0.2, poll_s=0.05)
