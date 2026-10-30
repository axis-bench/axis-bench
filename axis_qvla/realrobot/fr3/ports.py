"""Whether a policy server listens on a port, and waiting until one does.

Why a module of its own: `qvla realrobot serve` polls the port of the server it started in its own
environment (axis_qvla.realrobot.steps.serve), and the handshake probe polls it in openpi's
(axis_qvla.realrobot.fr3.probe). The probe is the one module of the line that imports openpi's client, and
only openpi's interpreter runs it (`python -m`); keeping the poll here means the serve step never imports
the probe. Stdlib only.
"""
from __future__ import annotations

import socket
import time


class NotListening(RuntimeError):
    """Nothing listened on the port within the timeout."""


def knock(host: str, port: int, *, timeout_s: float = 2.0) -> bool:
    """Whether something listens on host:port. It asks for openpi's health page (GET /healthz), which
    the server answers without logging; a bare TCP connect would log a failed WebSocket handshake with
    its traceback in the server's output on every poll. Any server counts once it accepts."""
    try:
        s = socket.create_connection((host, port), timeout=timeout_s)
    except OSError:
        return False
    with s:
        try:
            s.sendall(f"GET /healthz HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n".encode())
            s.settimeout(1.0)
            s.recv(256)
        except OSError:
            pass                        # accepted, but not an HTTP server, or a slow one: it listens
    return True


def wait_for_port(host: str, port: int, timeout_s: float, *, poll_s: float = 1.0) -> None:
    """Return once something listens on host:port; NotListening after `timeout_s` seconds."""
    deadline = time.monotonic() + timeout_s
    while not knock(host, port):
        if time.monotonic() >= deadline:
            raise NotListening(f"no server listening on {host}:{port} after {timeout_s:g} s")
        time.sleep(min(poll_s, max(deadline - time.monotonic(), 0.0)))
