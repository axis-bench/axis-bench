"""`python -m axis_qvla.realrobot.fr3.probe HOST:PORT [--timeout S]`: a running policy server's handshake, as JSON.

Why a module of its own: the handshake is read with openpi's client. The openpi environment has that
client, and the environment the real-robot steps run in need not. So `qvla realrobot serve` runs this
module with the openpi environment's interpreter (axis_qvla.realrobot.steps.serve). The module waits for the
port, because a server loads its checkpoint before it listens. Then it connects and prints the metadata
the server sends first.

Exit 0 printed; 2 when nothing listened within --timeout, the client is not installed, or the server
sent no handshake. Stdlib at module scope (the port poll is axis_qvla.realrobot.fr3.ports, which the serve
step imports instead of this module); openpi_client is imported inside `handshake`.
"""
from __future__ import annotations

import argparse
import json
import sys

from axis_qvla.realrobot.fr3.ports import NotListening, wait_for_port


class ProbeError(RuntimeError):
    """No handshake could be read."""


def handshake(host: str, port: int, timeout_s: float) -> dict:
    try:
        wait_for_port(host, port, timeout_s)
    except NotListening as exc:
        raise ProbeError(str(exc)) from None
    try:
        from openpi_client import websocket_client_policy
    except ImportError:
        raise ProbeError("openpi's client is not installed in this environment: pip install -e "
                         "<openpi checkout>/packages/openpi-client") from None
    try:
        return dict(websocket_client_policy.WebsocketClientPolicy(host, port).get_server_metadata())
    except Exception as exc:          # whatever its websocket or msgpack layer raises; reported, never a traceback
        raise ProbeError(f"{host}:{port} sent no handshake ({exc.__class__.__name__}: {exc})") from None


def _plain(value):
    """A metadata value JSON can hold: numpy scalars and arrays become Python numbers and lists."""
    return value.tolist() if hasattr(value, "tolist") else str(value)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m axis_qvla.realrobot.fr3.probe", description=__doc__.split("\n\n")[0])
    ap.add_argument("endpoint", help="HOST:PORT")
    ap.add_argument("--timeout", type=float, default=900.0, help="seconds to wait for the port (default 900)")
    a = ap.parse_args(argv)
    host, _, port = a.endpoint.rpartition(":")
    if not host or not port.isdigit():
        ap.error(f"{a.endpoint}: give HOST:PORT")
    try:
        md = handshake(host, int(port), a.timeout)
    except ProbeError as exc:
        print(f"FAIL  {exc}", file=sys.stderr)
        return 2
    print(json.dumps(md, default=_plain, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
