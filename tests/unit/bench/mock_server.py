"""Start the reference LeRobot server's built-in mock predictor as a real websocket server.

The protocol tests talk to a real server over a real socket, so they catch what a unit test of
the validators cannot: framing, msgpack encoding, the handshake-first order and the noise key.
`axis_qvla/bench/servers/lerobot_server.py --policy-type mock` needs only websockets, msgpack, numpy and
openpi-client -- no torch, no weights, no GPU.

The broken servers below run in-process on websockets' threading implementation, through `serve`.
Since websockets 17.0, `Server.shutdown()` closes every open connection and then WAITS for every
connection handler to return (16.x closed only the listening socket and returned). A handler that
blocks on anything but its own connection -- these mocks once parked on a private
`threading.Event` nothing ever set -- therefore makes `shutdown()` poll forever, and the test with
it. So every handler here blocks on its connection (it receives until the connection closes), and
`InProcessServer.shutdown()` fails, naming the server, where websockets would wait forever.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

# The in-process servers' websocket close timeout, which bounds every wait below: a closing
# handshake gives up after it, and a handler blocked on its connection returns as soon as that
# connection is closed.
CLOSE_TIMEOUT_S = 5.0
# How long the silent mocks wait on a client that has neither spoken nor left before they close the
# connection themselves. The tests pass check-server a --timeout of a few seconds, so the client's
# own timeout fires first; a client whose timeout is missing or ignored then ends with "closed
# before sending a handshake" (or an invalid HTTP response) and fails its test, instead of hanging.
GIVE_UP_S = 60.0


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start(log: Path, *extra: str) -> tuple[subprocess.Popen, int]:
    from axis_qvla.bench.servers import lerobot_server

    port = free_port()
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    with log.open("w") as f:
        proc = subprocess.Popen([sys.executable, lerobot_server.__file__, "--policy-type", "mock", "--ckpt", "mock",
                                 "--host", "127.0.0.1", "--port", str(port), *extra],
                                stdout=f, stderr=subprocess.STDOUT, env=env)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"mock server exited {proc.returncode}: {log.read_text()[-2000:]}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return proc, port
        except OSError:
            time.sleep(0.2)
    proc.kill()
    raise RuntimeError("mock server did not listen within 60 s")


def stop(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


class InProcessServer:
    """A websockets threading-implementation server on 127.0.0.1, serving from a daemon thread,
    that counts the connection handlers it has started and how many are still running."""

    def __init__(self, name: str, handler: Callable) -> None:
        from websockets.sync.server import serve as ws_serve

        self.name = name
        self._cond = threading.Condition()
        self._started = 0
        self._running: set = set()

        def tracked(ws) -> None:
            with self._cond:
                self._started += 1
                self._running.add(ws)
                self._cond.notify_all()
            try:
                handler(ws)
            finally:
                with self._cond:
                    self._running.discard(ws)
                    self._cond.notify_all()

        self._srv = ws_serve(tracked, "127.0.0.1", 0, compression=None, max_size=None,
                             close_timeout=CLOSE_TIMEOUT_S)
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        self.port = self._srv.socket.getsockname()[1]

    def handlers_returned(self, started: int) -> bool:
        """True once `started` handlers have begun and every one has returned; False if that has
        not happened within CLOSE_TIMEOUT_S -- a handler that ignores its closed connection."""
        with self._cond:
            return self._cond.wait_for(lambda: self._started >= started and not self._running,
                                       timeout=CLOSE_TIMEOUT_S)

    def shutdown(self) -> None:
        """Close the server and every connection, then require every handler to have returned.

        The connections are closed here as well as by websockets, so 16.x (whose `shutdown()`
        leaves them open) and 17.x see the same thing. websockets' own `shutdown()` runs in a
        daemon thread: on 17.x it waits for the handlers without a bound, and a stuck one must fail
        this test, not hang the suite."""
        with self._cond:
            open_now = list(self._running)
        for ws in open_now:
            ws.close()
        closer = threading.Thread(target=self._srv.shutdown, daemon=True)
        closer.start()
        closer.join(CLOSE_TIMEOUT_S)
        with self._cond:
            self._cond.wait_for(lambda: not self._running, timeout=CLOSE_TIMEOUT_S)
            stuck = len(self._running)
        if closer.is_alive() or stuck:
            raise AssertionError(f"{self.name}: {stuck} connection handler(s) still running after "
                                 f"their connections were closed; a handler must block on its "
                                 f"connection, or websockets >= 17's shutdown() waits for it forever")


def serve(name: str, handler: Callable) -> tuple[InProcessServer, int]:
    """`(server, port)` for `handler(ws)`, which must return once `ws` closes."""
    srv = InProcessServer(name, handler)
    return srv, srv.port


def _until_closed(ws, *, give_up_s: float | None = None) -> None:
    """Receive and discard until the connection closes: how a handler that never answers again
    still notices its client has gone. With `give_up_s`, a client still there after that long is
    disconnected (see GIVE_UP_S)."""
    from websockets.exceptions import ConnectionClosed

    deadline = None if give_up_s is None else time.monotonic() + give_up_s
    while True:
        try:
            ws.recv(timeout=None if deadline is None else max(0.0, deadline - time.monotonic()))
        except ConnectionClosed:
            return
        except TimeoutError:
            ws.close()
            return


def bad_server(handshake: dict, actions) -> tuple[InProcessServer, int]:
    """An in-process server that sends `handshake`, then answers every request with `actions`."""
    from openpi_client import msgpack_numpy

    def handler(ws) -> None:
        ws.send(msgpack_numpy.packb(handshake))
        for message in ws:
            msgpack_numpy.unpackb(message)
            ws.send(msgpack_numpy.packb({"actions": actions}))

    return serve("bad_server", handler)


def sequenced_server(handshake: dict, replies: list) -> tuple[InProcessServer, int]:
    """Like `bad_server`, but answers the Nth request with `replies[N]` (the last entry repeats
    once the list is exhausted), so a server's SECOND reply can differ from its first. A `str`
    entry is sent as a TEXT frame (an error) instead of a packed `{"actions": ...}` reply."""
    from openpi_client import msgpack_numpy

    def handler(ws) -> None:
        ws.send(msgpack_numpy.packb(handshake))
        for i, message in enumerate(ws):
            msgpack_numpy.unpackb(message)
            reply = replies[min(i, len(replies) - 1)]
            ws.send(reply if isinstance(reply, str) else msgpack_numpy.packb({"actions": reply}))

    return serve("sequenced_server", handler)


def silent_server() -> tuple[InProcessServer, int]:
    """A real websocket server that completes the opening handshake and then never sends
    anything -- proves the client's handshake read is bounded by its timeout, not a hang. It
    disconnects a client that is still waiting after GIVE_UP_S."""
    return serve("silent_server", lambda ws: _until_closed(ws, give_up_s=GIVE_UP_S))


def text_first_server(text: str = "not a handshake") -> tuple[InProcessServer, int]:
    """A real websocket server whose first frame is TEXT, not the msgpack handshake."""

    def handler(ws) -> None:
        ws.send(text)
        _until_closed(ws)

    return serve("text_first_server", handler)


class SilentTcpListener:
    """A bare TCP listener that accepts every connection and never speaks a word of HTTP or
    websocket: not a policy server at all, the way a misconfigured `--policy` endpoint would look.
    Each accepted connection is held, silent, until GIVE_UP_S has passed or `close()` is called,
    then closed -- so a client whose opening-handshake timeout is missing sees an invalid HTTP
    response (end of stream), not a TimeoutError, and fails its test instead of hanging."""

    def __init__(self) -> None:
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port = self._sock.getsockname()[1]
        self._closed = threading.Event()
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return  # the listener was closed
            threading.Thread(target=self._hold, args=(conn,), daemon=True).start()

    def _hold(self, conn: socket.socket) -> None:
        self._closed.wait(GIVE_UP_S)
        conn.close()

    def close(self) -> None:
        self._closed.set()
        try:
            self._sock.shutdown(socket.SHUT_RDWR)  # wakes the blocked accept() on Linux
        except OSError:
            pass
        self._sock.close()


def tcp_listener() -> tuple[SilentTcpListener, int]:
    """`(listener, port)` of a SilentTcpListener; the caller closes the listener when done."""
    listener = SilentTcpListener()
    return listener, listener.port
