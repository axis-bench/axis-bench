"""`qvla bench check-server`: is a running policy server one `qvla bench eval` can score?

Connects once, reads the handshake, sends a few observations shaped exactly as the rollout
client sends them (synthetic images at the benchmark's declared render size -- its
execution.json's render.height x render.width, or RAW_HEIGHT x RAW_WIDTH for a benchmark that
declares none -- a fixed joint state, a real task prompt and the client's own per-call noise) and
checks every reply against axis_qvla.bench.protocol. It needs no simulator, so a policy author can run
it before spending an evaluation. `--benchmark` supplies the prompt (and the `image_geometry: raw`
assertion), the frame size and the replan.

Gated (exit 1 on a failure): the handshake, and every reply -- including the extra request the
repeatability check sends, which goes through the same guarded + validated path as the first
`calls` requests, never left to raise past this module. Reported only: whether an identical
request gets identical actions (GPU kernels need not be bitwise repeatable; a DIFFERENT shape
between two identical requests is still gated, since the handshake's action_horizon/action_dim
must not move between calls) and whether the gripper column stays in [0, 1].

Every check is printed as it is earned, and `--report` is written in a `finally`, so a failure
discovered partway through still leaves the earlier PASS lines on the screen and in the report
file instead of losing them.

A server that never completes the websocket handshake, never sends its first frame, sends text
instead of bytes, or sends bytes that are not valid msgpack all fail the gated "handshake" check
within --timeout seconds -- never a hang, never an uncaught exception. Only a listener that
refuses the TCP connection itself is reported as unreachable (exit 2).

Exit codes: 0 conformant; 1 a gated check failed; 2 the server could not be reached, or the
arguments are wrong.

The websocket connection is entered as a context manager, never used as the bare return value of
`websockets.sync.client.connect`: websockets 17.1 deprecated the bare form, and states that a later
release makes the non-legacy behaviour the default, in which a bare `connect(...)` returns a
reconnecting iterator instead of a connection. Entering it works on every websockets release
openpi-client accepts, and closing goes through the same exit.

Eval environment: numpy at module scope; openpi_client and websockets inside `connect`/`_Client`.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import socket
import time
from collections.abc import Callable

import numpy as np

from axis_qvla.bench import protocol
from axis_qvla.bench.sim.inference_noise import noise_for

# A reachable, mid-range Franka pose (rad) with the gripper open: a plausible state, not a test of
# the policy.
HOME_STATE = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0], dtype=np.float32)


def _check(name: str, ok: bool, detail: str, gated: bool) -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail, "gated": gated}


def open_tcp(host: str, port: int, timeout_s: float) -> socket.socket:
    """The TCP connection the websocket handshake then runs over, retried until `timeout_s` while
    nothing listens: a listener that refuses the connection outright must not also pay for the
    websocket opening handshake's own timeout on top of this one. The connected socket is handed to
    the websocket client, never closed bare: a server's websocket library logs a connection that
    closes before sending an HTTP request as a failed opening handshake, with a traceback, so a
    bare probe printed one into the user's server log on every check."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            sock = socket.create_connection((host, port), timeout=2.0)
        except OSError as exc:
            if time.monotonic() >= deadline:
                raise ConnectionError(f"{host}:{port} not reachable after {timeout_s:g} s "
                                      f"({type(exc).__name__})") from None
            time.sleep(0.5)
            continue
        # Blocking from here on: the 2 s bounds the connect only. Left on the socket, it would end the
        # websocket client's reads after 2 s of silence as a dropped connection, before --timeout.
        sock.settimeout(None)
        return sock


class _Client:
    """One open websocket connection, already past the handshake frame. `infer` packs and sends a
    request and reads the reply exactly as
    `openpi_client.websocket_client_policy.WebsocketClientPolicy` does, bounded by the same
    timeout used to open the connection and read the handshake -- that production client has no
    such bound (it is meant to wait out a slow real server); this one needs one to run unattended
    against an arbitrary, possibly broken, server."""

    def __init__(self, ws, timeout_s: float, *, exit_stack: contextlib.ExitStack) -> None:
        from openpi_client import msgpack_numpy

        self._ws = ws
        self._exit = exit_stack     # holds the entered connection; closing it closes the socket
        self._timeout = timeout_s
        self._packer = msgpack_numpy.Packer()
        self._unpackb = msgpack_numpy.unpackb

    def infer(self, obs: dict, *, noise=None) -> dict:
        if noise is not None:
            obs = {**obs, protocol.NOISE_KEY: noise}
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv(timeout=self._timeout)
        if isinstance(response, str):
            raise RuntimeError(f"error frame from the server:\n{response}")
        return self._unpackb(response)

    def close(self) -> None:
        self._exit.close()


def connect(host: str, port: int, timeout_s: float) -> tuple[_Client | None, dict | str]:
    """Open the socket, perform the websocket handshake and read the server's first frame, all
    bounded by `timeout_s`.

    Returns `(client, metadata)` on success. Short of genuine TCP-level unreachability (which
    still raises `ConnectionError`, as `open_tcp` does), every other way this can go
    wrong -- a listener that never completes the websocket upgrade, that opens but never speaks,
    that sends text instead of bytes, or sends bytes that are not valid msgpack -- returns
    `(None, "<problem text>")` instead of raising or hanging. The caller turns that string into a
    gated 'handshake' FAIL rather than a traceback.
    """
    sock = open_tcp(host, port, timeout_s)
    from openpi_client import msgpack_numpy
    from websockets.sync.client import connect as ws_connect

    uri = f"ws://{host}:{port}"
    stack = contextlib.ExitStack()
    try:
        # Entered, not called bare: see the module docstring. A failure here pushed nothing onto
        # `stack`, so only the socket is left to close.
        ws = stack.enter_context(ws_connect(uri, sock=sock, compression=None, max_size=None,
                                            open_timeout=timeout_s, ping_interval=None))
    except Exception as exc:  # noqa: BLE001 - not a websocket listener, or the upgrade timed out
        sock.close()
        return None, (f"{host}:{port} did not complete a websocket handshake within "
                      f"{timeout_s:g} s ({type(exc).__name__}: {exc})")
    try:
        frame = ws.recv(timeout=timeout_s)
    except TimeoutError:
        stack.close()
        return None, f"{host}:{port} sent no handshake within {timeout_s:g} s"
    except Exception as exc:  # noqa: BLE001 - the connection dropped mid-handshake
        stack.close()
        return None, f"{host}:{port} closed before sending a handshake ({type(exc).__name__}: {exc})"
    if isinstance(frame, str):
        stack.close()
        return None, f"{host}:{port} sent a text frame first, not a msgpack handshake: {frame[:200]!r}"
    try:
        md = msgpack_numpy.unpackb(frame)
    except Exception as exc:  # noqa: BLE001 - bytes that are not valid msgpack
        stack.close()
        return None, f"{host}:{port}'s first frame is not valid msgpack ({type(exc).__name__}: {exc})"
    return _Client(ws, timeout_s, exit_stack=stack), md


def observation(prompt: str, height: int, width: int, *, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {"base_0_rgb": rng.integers(0, 256, (height, width, 3), dtype=np.uint8),
            "left_wrist_0_rgb": rng.integers(0, 256, (height, width, 3), dtype=np.uint8),
            "state": HOME_STATE.copy(), "prompt": prompt}


def _infer(client: _Client, prompt: str, height: int, width: int, seed: int, *,
          noise: np.ndarray) -> tuple[dict | None, str | None]:
    """`(reply, None)` on success, `(None, "<problem>")` on anything the server does wrong -- an
    error frame, a closed socket, a request that times out -- so every caller validates the same
    way instead of each catching exceptions on its own."""
    try:
        reply = client.infer(observation(prompt, height, width, seed=seed), noise=noise)
    except Exception as exc:  # noqa: BLE001 - an error frame or a closed socket is the finding
        return None, f"{type(exc).__name__}: {exc}"[:300]
    return reply, None


def check(client: _Client, md, *, prompt: str, task_id: int, height: int, width: int,
          replan: int, calls: int, emit: Callable[[dict], None] | None = None) -> list[dict]:
    checks: list[dict] = []

    def _emit(c: dict) -> None:
        checks.append(c)
        if emit is not None:
            emit(c)

    hp = protocol.handshake_problems(md)
    _emit(_check("handshake", not hp, "; ".join(hp) or "required keys present", True))
    if hp:
        return checks
    stated = protocol.contract(md)
    if stated:
        _emit(_check("contract", True, "; ".join(f"{k} {v}" for k, v in stated.items()), False))
    h, d, seed = int(md["action_horizon"]), int(md["action_dim"]), task_id * 1000
    first = None
    for i in range(calls):
        t0 = time.monotonic()
        reply, err = _infer(client, prompt, height, width, i, noise=noise_for(task_id, seed, i, h, d))
        ms = (time.monotonic() - t0) * 1000
        if err is not None:
            _emit(_check(f"reply[{i}]", False, err, True))
            return checks
        problems = protocol.reply_problems(reply, replan=replan)
        actions = np.asarray(reply["actions"]) if not problems else None
        if actions is not None and not np.isfinite(actions).all():
            problems.append("actions holds a non-finite value")
        detail = "; ".join(problems) or f"actions {actions.shape} {actions.dtype}, {ms:.0f} ms"
        _emit(_check(f"reply[{i}]", not problems, detail, True))
        if problems:
            return checks
        first = actions if first is None else first

    # The repeatability request: the SAME guarded + validated path as every call above, so an
    # error frame or a malformed reply here gives a FAIL line too, never an uncaught exception.
    reply, err = _infer(client, prompt, height, width, 0, noise=noise_for(task_id, seed, 0, h, d))
    if err is not None:
        _emit(_check("repeatable", False, err, True))
        return checks
    problems = protocol.reply_problems(reply, replan=replan)
    again = np.asarray(reply["actions"]) if not problems else None
    if again is not None and not np.isfinite(again).all():
        problems.append("actions holds a non-finite value")
    if problems:
        _emit(_check("repeatable", False, "; ".join(problems), True))
        return checks
    if again.shape != first.shape:
        # The handshake's action_horizon/action_dim must not move between calls -- unlike a
        # same-shape VALUE difference (GPU nondeterminism), a shape difference is gated.
        _emit(_check("repeatable", False, f"an identical request got a different action shape "
                     f"({again.shape} vs {first.shape})", True))
        return checks
    same = bool(np.array_equal(again, first))
    detail = ("an identical request got identical actions" if same else
             f"an identical request differed by up to {float(np.abs(again - first).max()):.3g}"
             " (reported, not gated: GPU kernels need not be bitwise repeatable)")
    _emit(_check("repeatable", same, detail, False))
    g = first[:, protocol.GRIPPER_COLUMN]
    _emit(_check("gripper", bool(((g >= 0) & (g <= 1)).all()),
                 f"gripper column in [{float(g.min()):.3g}, {float(g.max()):.3g}]; "
                 f"the client closes at >= 0.5", False))
    return checks


def build_parser(prog: str = "python -m axis_qvla.bench.sim.check_server") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    protocol.add_check_server_arguments(ap)
    return ap


def run(args: argparse.Namespace) -> int:
    from axis_qvla.bench.spec import DEFAULT_NAME, BenchmarkError, load_benchmark

    try:
        host, port = protocol.parse_endpoint(args.policy)
        b = load_benchmark(args.benchmark or DEFAULT_NAME)
        tid = int(args.task) if args.task is not None else b.task_ids[0]
        prompt = b.task(tid).prompt
        if args.calls < 1:
            raise ValueError("--calls must be at least 1")
        if b.image_geometry != "raw":
            raise ValueError(f"check-server sends frames as rendered; benchmark {b.name} declares "
                             f"image_geometry {b.image_geometry!r}")
    except (ValueError, BenchmarkError) as exc:
        print(f"FAIL  {exc}")
        return 2
    try:
        client, result = connect(host, port, args.timeout)
    except ConnectionError as exc:
        print(f"FAIL  {exc}")
        return 2

    checks: list[dict] = []

    def emit(c: dict) -> None:
        checks.append(c)
        tag = ("PASS" if c["ok"] else "FAIL") if c["gated"] else ("note" if c["ok"] else "warn")
        print(f"{tag:5s} {c['name']}: {c['detail']}")

    md: dict = {}
    try:
        if client is None:
            # connect() already turned every non-TCP failure into a problem string; report it as
            # the gated handshake check it is, instead of a second, different, code path.
            emit(_check("handshake", False, result, True))
        else:
            md = result
            ex = b.execution or {"render": {"height": protocol.RAW_HEIGHT, "width": protocol.RAW_WIDTH},
                                 "replan": protocol.DEFAULT_REPLAN}
            check(client, md, prompt=prompt, task_id=tid, height=ex["render"]["height"],
                 width=ex["render"]["width"], replan=ex["replan"], calls=args.calls, emit=emit)
    finally:
        if client is not None:
            client.close()
        # In a `finally` so a bug discovered mid-check still writes whatever was already earned,
        # rather than losing every earlier PASS line to an exception raised after them.
        ok = bool(checks) and all(c["ok"] for c in checks if c["gated"])
        unwritten = None
        if args.report is not None:
            try:
                args.report.write_text(json.dumps({"endpoint": args.policy, "ok": ok, "handshake": md,
                                                   "checks": checks}, indent=1, default=str),
                                       encoding="utf-8")
            except OSError as exc:      # the verdict below still prints; the run exits 2
                unwritten = f"FAIL  --report {args.report} cannot be written ({type(exc).__name__}: {exc})"
    print(f"\n{args.policy}: {'conformant' if ok else 'NOT conformant'} ({protocol.PROTOCOL})")
    if unwritten:
        print(unwritten)
        return 2
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
