"""The handshake probe: waits for the port, reads the handshake with openpi's client, refuses cleanly."""
from __future__ import annotations

import json
import socket
import sys
import types

import numpy as np
import pytest

from axis_qvla.realrobot.fr3 import probe


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


def _client(monkeypatch, policy):
    pkg, sub = types.ModuleType("openpi_client"), types.ModuleType("openpi_client.websocket_client_policy")
    sub.WebsocketClientPolicy, pkg.websocket_client_policy = policy, sub
    monkeypatch.setitem(sys.modules, "openpi_client", pkg)
    monkeypatch.setitem(sys.modules, "openpi_client.websocket_client_policy", sub)


def test_nothing_listening_is_a_refusal_within_the_timeout(capsys):
    assert probe.main([f"127.0.0.1:{_free_port()}", "--timeout", "0.3"]) == 2
    assert "no server listening" in capsys.readouterr().err


def test_a_missing_client_names_its_install(listening, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "openpi_client", None)
    assert probe.main([f"127.0.0.1:{listening}", "--timeout", "1"]) == 2
    assert "openpi-client" in capsys.readouterr().err


def test_the_handshake_is_printed_as_json(listening, monkeypatch, capsys):
    class Policy:
        def __init__(self, host, port):
            self.at = (host, port)

        def get_server_metadata(self):
            return {"action_fps": np.int64(15), "image_geometry": {"base_0_rgb": "center_crop224"}}

    _client(monkeypatch, Policy)
    assert probe.main([f"127.0.0.1:{listening}", "--timeout", "1"]) == 0
    assert json.loads(capsys.readouterr().out) == {"action_fps": 15, "image_geometry": {"base_0_rgb": "center_crop224"}}


def test_a_server_that_sends_no_handshake_is_a_refusal_not_a_traceback(listening, monkeypatch, capsys):
    class Policy:
        def __init__(self, host, port):
            raise ConnectionResetError("closed")

    _client(monkeypatch, Policy)
    assert probe.main([f"127.0.0.1:{listening}", "--timeout", "1"]) == 2
    assert "sent no handshake (ConnectionResetError: closed)" in capsys.readouterr().err


def test_an_endpoint_without_a_port_is_a_usage_error():
    with pytest.raises(SystemExit) as exc:
        probe.main(["localhost", "--timeout", "1"])
    assert exc.value.code == 2

