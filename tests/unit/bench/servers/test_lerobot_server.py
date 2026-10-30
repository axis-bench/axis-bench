"""--config-name defaults to the published config name for --policy-type (the name cells.json's
eval_command passes as --config), and an explicit --config-name overrides it verbatim; --quality-tag
takes a quality bin and refuses the untagged sentinel 0 before anything is served."""
from __future__ import annotations

import pytest

from axis_qvla.bench.servers.lerobot_server import _resolved_config_name, build_parser


def test_config_name_defaults_to_the_published_name():
    args = build_parser().parse_args(["--policy-type", "smolvla", "--ckpt", "x"])
    assert _resolved_config_name(args) == "axis_smolvla"


def test_config_name_override_is_published_verbatim():
    args = build_parser().parse_args(
        ["--policy-type", "groot", "--ckpt", "x", "--config-name", "my_groot_config"])
    assert _resolved_config_name(args) == "my_groot_config"


@pytest.mark.parametrize("tag, match", [("0", "untagged sentinel"), ("-1", "not a quality bin"), ("x", "not a quality bin")])
def test_a_tag_that_is_not_a_bin_stops_the_server_at_parse_time(tag, match, capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--policy-type", "smolvla", "--ckpt", "x", "--quality-tag", tag])
    assert match in capsys.readouterr().err
    assert build_parser().parse_args(["--policy-type", "smolvla", "--ckpt", "x", "--quality-tag", "5"]).quality_tag == 5


class _ClosedBeforeHandshake:
    """A connection whose close lands before the server's metadata frame (a readiness probe)."""
    remote_address = ("127.0.0.1", 1)

    def __init__(self):
        self.recvs = 0

    async def send(self, data):
        import websockets

        raise websockets.ConnectionClosed(None, None)

    async def recv(self):
        self.recvs += 1
        raise AssertionError("no request is read on a closed connection")


@pytest.mark.parametrize("module", ["axis_qvla.bench.servers.lerobot_server", "axis_qvla.bench.servers.xvla"])
def test_a_connection_closed_before_the_handshake_ends_quietly(module):
    """The metadata send sits inside the ConnectionClosed handling, so a readiness
    probe that closes at once ends its handler instead of logging 'connection handler failed'."""
    import asyncio
    import importlib

    class _Predictor:
        resets = 0

        def reset(self):
            self.resets += 1

    server = importlib.import_module(module).PredictorPolicyServer(_Predictor(), host="127.0.0.1", port=0, metadata={})
    ws = _ClosedBeforeHandshake()
    assert asyncio.run(server._handler(ws)) is None
    assert ws.recvs == 0 and server._predictor.resets == 1
