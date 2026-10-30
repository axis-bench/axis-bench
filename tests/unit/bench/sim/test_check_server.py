"""`qvla bench check-server` against real servers: the reference mock passes, broken ones fail
with the reason named, an unreachable one exits 2."""
from __future__ import annotations

import json

import numpy as np
import pytest

from axis_qvla.cli import main
from tests.unit.bench import mock_server


@pytest.fixture
def ws_timeouts(monkeypatch):
    """The timeouts check-server hands the websocket client, recorded as it opens the connection
    (`open_timeout`) and reads from it (`recv(timeout=...)`). The bounded-handshake tests pass
    --timeout 2 and require exactly 2.0 here: the deterministic half of those tests, which needs
    no clock."""
    import websockets.sync.client

    real = websockets.sync.client.connect
    seen: dict = {"open_timeout": [], "recv": []}

    class Recording:
        def __init__(self, ws) -> None:
            self._ws = ws

        def recv(self, *args, **kwargs):
            seen["recv"].append(kwargs.get("timeout", args[0] if args else None))
            return self._ws.recv(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._ws, name)

    class Opening:
        def __init__(self, cm) -> None:
            self._cm = cm

        def __enter__(self):
            return Recording(self._cm.__enter__())

        def __exit__(self, *exc):
            return self._cm.__exit__(*exc)

    def connect(*args, **kwargs):
        seen["open_timeout"].append(kwargs.get("open_timeout"))
        return Opening(real(*args, **kwargs))

    monkeypatch.setattr(websockets.sync.client, "connect", connect)
    return seen


@pytest.fixture
def mock(tmp_path):
    proc, port = mock_server.start(tmp_path / "server.log")
    yield port
    mock_server.stop(proc)


def test_the_reference_mock_server_conforms(mock, tmp_path, capsys):
    report = tmp_path / "report.json"
    assert main(["bench", "check-server", "--policy", f"127.0.0.1:{mock}", "--report", str(report)]) == 0
    out = capsys.readouterr().out
    assert "PASS  handshake" in out and "PASS  reply[2]" in out and "conformant" in out
    doc = json.loads(report.read_text())
    assert doc["ok"] is True and doc["handshake"]["action_horizon"] == 50
    # the mock stamps a call counter into actions[0, 0], so it is not repeatable: reported, not gated
    assert {c["name"]: c["gated"] for c in doc["checks"]}["repeatable"] is False


@pytest.mark.parametrize("handshake, actions, needle", [
    ({"action_horizon": 10, "action_dim": 32}, np.zeros((10, 8), np.float32), "supports_client_noise"),
    ({"supports_client_noise": True, "action_horizon": 10, "action_dim": 32},
     np.zeros((4, 8), np.float32), "at least 8"),
    ({"supports_client_noise": True, "action_horizon": 10, "action_dim": 32},
     np.full((10, 8), np.nan, np.float32), "non-finite"),
])
def test_a_broken_server_fails_with_the_reason(handshake, actions, needle, capsys):
    srv, port = mock_server.bad_server(handshake, actions)
    try:
        assert main(["bench", "check-server", "--policy", f"127.0.0.1:{port}", "--calls", "1"]) == 1
    finally:
        srv.shutdown()
    out = capsys.readouterr().out
    assert "FAIL" in out and needle in out


def test_a_stated_serving_contract_is_recorded(capsys, tmp_path):
    md = {"supports_client_noise": True, "action_horizon": 10, "action_dim": 32,
          "image_geometry": {"base_0_rgb": "center_crop224", "left_wrist_0_rgb": "center_crop224"},
          "action_units": "rad_s"}
    srv, port = mock_server.bad_server(md, np.zeros((10, 8), np.float32))
    report = tmp_path / "r.json"
    try:
        assert main(["bench", "check-server", "--policy", f"127.0.0.1:{port}", "--calls", "1", "--report", str(report)]) == 0
    finally:
        srv.shutdown()
    assert "note  contract: image_geometry" in capsys.readouterr().out
    assert json.loads(report.read_text())["handshake"]["image_geometry"]["base_0_rgb"] == "center_crop224"


def test_an_unreachable_server_exits_2(capsys):
    port = mock_server.free_port()
    assert main(["bench", "check-server", "--policy", f"127.0.0.1:{port}", "--timeout", "1"]) == 2
    assert "not reachable" in capsys.readouterr().out


def test_a_malformed_endpoint_exits_2(capsys):
    assert main(["bench", "check-server", "--policy", "localhost"]) == 2
    assert "HOST:PORT" in capsys.readouterr().out


def test_a_server_that_never_speaks_fails_within_timeout_not_hangs(ws_timeouts, capsys):
    """Completes the websocket upgrade, then never sends the handshake: must fail by --timeout,
    not hang (openpi_client's own client retries this forever). No clock is asserted on. The
    handshake read must carry exactly the --timeout given (recorded), and it must be what fires:
    the mock disconnects a client still waiting after GIVE_UP_S, so a read with no bound or a
    longer one ends as "closed before sending a handshake" and fails here instead of hanging."""
    srv, port = mock_server.silent_server()
    try:
        assert main(["bench", "check-server", "--policy", f"127.0.0.1:{port}", "--timeout", "2"]) == 1
    finally:
        srv.shutdown()
    out = capsys.readouterr().out
    assert f"FAIL  handshake: 127.0.0.1:{port} sent no handshake within 2 s" in out, out
    assert ws_timeouts == {"open_timeout": [2.0], "recv": [2.0]}


def test_a_server_whose_first_frame_is_text_fails_the_handshake(capsys):
    srv, port = mock_server.text_first_server()
    try:
        assert main(["bench", "check-server", "--policy", f"127.0.0.1:{port}", "--timeout", "5"]) == 1
    finally:
        srv.shutdown()
    out = capsys.readouterr().out
    assert "FAIL  handshake" in out and "text frame" in out


def test_a_non_websocket_tcp_listener_fails_cleanly_not_a_traceback(ws_timeouts, capsys):
    """A listener that accepts and never answers the HTTP upgrade fails the handshake check by
    --timeout, not a hang and not a traceback. No clock is asserted on. The opening must carry
    exactly the --timeout given (recorded), and it must be what fires: the listener drops each
    connection after GIVE_UP_S, so an unbounded opening ends as an invalid HTTP response, not the
    TimeoutError required here."""
    sock, port = mock_server.tcp_listener()
    try:
        assert main(["bench", "check-server", "--policy", f"127.0.0.1:{port}", "--timeout", "2"]) == 1
    finally:
        sock.close()
    out = capsys.readouterr().out
    assert (f"FAIL  handshake: 127.0.0.1:{port} did not complete a websocket handshake within 2 s "
            f"(TimeoutError") in out, out
    assert ws_timeouts == {"open_timeout": [2.0], "recv": []}


def test_the_repeatability_request_handles_an_error_frame(capsys, tmp_path):
    """The extra repeatability request goes through the same guarded path as the first `calls`
    requests -- an error frame on the SECOND reply gives a FAIL line, never an uncaught
    RuntimeError, and the report (with the earlier reply[0] PASS) still gets written."""
    md = {"supports_client_noise": True, "action_horizon": 10, "action_dim": 32}
    good = np.zeros((10, 8), np.float32)
    srv, port = mock_server.sequenced_server(md, [good, "boom"])
    report = tmp_path / "r.json"
    try:
        assert main(["bench", "check-server", "--policy", f"127.0.0.1:{port}", "--calls", "1",
                    "--report", str(report)]) == 1
    finally:
        srv.shutdown()
    out = capsys.readouterr().out
    assert "FAIL  repeatable" in out and "boom" in out
    doc = json.loads(report.read_text())
    assert doc["ok"] is False
    assert any(c["name"] == "reply[0]" and c["ok"] for c in doc["checks"])


def test_the_repeatability_request_is_gated_for_non_finite_actions(capsys, tmp_path):
    """The second reply gets the same finiteness check every reply[i] gets: a NaN in it is a
    gated FAIL (and exit 1), never reported merely as "an identical request differed"."""
    md = {"supports_client_noise": True, "action_horizon": 10, "action_dim": 32}
    good = np.zeros((10, 8), np.float32)
    bad = np.full((10, 8), np.nan, np.float32)
    srv, port = mock_server.sequenced_server(md, [good, bad])
    report = tmp_path / "r.json"
    try:
        assert main(["bench", "check-server", "--policy", f"127.0.0.1:{port}", "--calls", "1",
                    "--report", str(report)]) == 1
    finally:
        srv.shutdown()
    out = capsys.readouterr().out
    assert "FAIL  repeatable" in out and "non-finite" in out
    doc = json.loads(report.read_text())
    assert doc["ok"] is False
    assert any(c["name"] == "reply[0]" and c["ok"] for c in doc["checks"])


def test_check_server_sends_the_benchmarks_declared_frame_size_not_the_simulators(tmp_path, monkeypatch, capsys):
    """check_server.run must take height/width/replan from b.execution, not always
    protocol.RAW_HEIGHT/RAW_WIDTH/DEFAULT_REPLAN -- a benchmark whose execution.json declares a
    DIFFERENT frame size must see frames of exactly that size. A regression back to the
    hard-coded constants would send 360x640 and fail this."""
    import json as jsonlib
    import shutil

    from openpi_client import msgpack_numpy

    from axis_qvla.bench import spec

    conf = tmp_path / "conf"
    conf.mkdir()
    for name in ("franka_heldout24.json", "franka_heldout24.tasks.json", "franka_heldout24.layouts.json",
                "execution.json"):
        shutil.copy(spec.CONF_DIR / name, conf / name)
    ex = jsonlib.loads((conf / "execution.json").read_text())
    ex["render"] = {"height": 111, "width": 222}
    (conf / "execution.json").write_text(jsonlib.dumps(ex))
    monkeypatch.setattr(spec, "CONF_DIR", conf)

    shapes: list = []

    def handler(ws) -> None:
        ws.send(msgpack_numpy.packb({"supports_client_noise": True, "action_horizon": 10, "action_dim": 32}))
        for message in ws:
            obs = msgpack_numpy.unpackb(message)
            shapes.append(obs["base_0_rgb"].shape)
            ws.send(msgpack_numpy.packb({"actions": np.zeros((10, 8), np.float32)}))

    srv, port = mock_server.serve("frame_size_server", handler)
    try:
        assert main(["bench", "check-server", "--policy", f"127.0.0.1:{port}", "--calls", "1"]) == 0
    finally:
        srv.shutdown()
    assert shapes and shapes[0][:2] == (111, 222), shapes


def test_the_repeatability_request_handles_a_different_shape(capsys, tmp_path):
    """A second reply with a different action shape is a protocol violation (the handshake's
    action_horizon/action_dim must not move between calls), so it is gated -- unlike a same-shape
    value-only difference, which is reported but never fails the run."""
    md = {"supports_client_noise": True, "action_horizon": 10, "action_dim": 32}
    first = np.zeros((10, 8), np.float32)
    second = np.zeros((10, 9), np.float32)
    srv, port = mock_server.sequenced_server(md, [first, second])
    report = tmp_path / "r.json"
    try:
        assert main(["bench", "check-server", "--policy", f"127.0.0.1:{port}", "--calls", "1",
                    "--report", str(report)]) == 1
    finally:
        srv.shutdown()
    out = capsys.readouterr().out
    assert "FAIL  repeatable" in out and "different action shape" in out
    doc = json.loads(report.read_text())
    assert doc["ok"] is False
    assert any(c["name"] == "reply[0]" and c["ok"] for c in doc["checks"])


def test_an_unwritable_report_keeps_the_verdict_and_exits_2(mock, tmp_path, capsys):
    report = tmp_path / "no such directory" / "report.json"
    assert main(["bench", "check-server", "--policy", f"127.0.0.1:{mock}", "--calls", "1",
                 "--report", str(report)]) == 2
    out = capsys.readouterr().out
    assert f"127.0.0.1:{mock}: conformant" in out
    assert f"FAIL  --report {report} cannot be written (FileNotFoundError" in out


GOOD_MD = {"supports_client_noise": True, "action_horizon": 10, "action_dim": 32}


@pytest.mark.parametrize("make", [
    mock_server.silent_server,
    mock_server.text_first_server,
    lambda: mock_server.bad_server(GOOD_MD, np.zeros((10, 8), np.float32)),
    lambda: mock_server.sequenced_server(GOOD_MD, [np.zeros((10, 8), np.float32), "boom"]),
], ids=["silent", "text_first", "bad", "sequenced"])
def test_every_in_process_mock_server_lets_its_handler_go_when_check_server_disconnects(make, capsys):
    """websockets >= 17.0's Server.shutdown() waits for every connection handler to return, so a
    mock whose handler outlives its connection hangs the test that shuts it down (the silent and
    text-first mocks once parked on a threading.Event nothing set, and the CI test job stalled on
    websockets 17.1). After check-server has come and gone -- its own --timeout bounds the
    exchange -- every handler must already have returned, on any websockets version; the wait for
    that is bounded by the server's close timeout and fails rather than hangs."""
    srv, port = make()
    try:
        main(["bench", "check-server", "--policy", f"127.0.0.1:{port}", "--calls", "1", "--timeout", "2"])
        assert srv.handlers_returned(started=1), f"{srv.name}: a handler outlived check-server's connection"
    finally:
        srv.shutdown()
    capsys.readouterr()


def test_check_server_calls_no_deprecated_websockets_api(capsys):
    """websockets 17.1 deprecates using `sync.client.connect(...)`'s return value directly (it must
    be entered as a context manager); escalated to an error, a bare call would fail the handshake
    check of a conformant server. Bounded by check-server's own --timeout."""
    import re
    import warnings

    from axis_qvla.bench.sim import check_server

    srv, port = mock_server.bad_server(GOOD_MD, np.zeros((10, 8), np.float32))
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=DeprecationWarning,
                                    module=re.escape(check_server.__name__) + "$")
            rc = main(["bench", "check-server", "--policy", f"127.0.0.1:{port}", "--calls", "1",
                       "--timeout", "5"])
    finally:
        srv.shutdown()
    out = capsys.readouterr().out
    assert rc == 0 and "DeprecationWarning" not in out, out


def test_check_server_enters_the_connection_instead_of_using_connects_return_value(monkeypatch, capsys):
    """websockets states that `sync.client.connect(...)` will, by default, return something that is
    only a context manager (a reconnecting iterator), not a connection. check-server must enter
    it, on every websockets version: `connect` is wrapped here to return exactly that shape."""
    import websockets.sync.client

    real = websockets.sync.client.connect
    entered: list = []

    class EnterOnly:
        def __init__(self, cm) -> None:
            self._cm = cm

        def __enter__(self):
            entered.append(True)
            return self._cm.__enter__()

        def __exit__(self, *exc):
            return self._cm.__exit__(*exc)

    monkeypatch.setattr(websockets.sync.client, "connect", lambda *a, **k: EnterOnly(real(*a, **k)))
    srv, port = mock_server.bad_server(GOOD_MD, np.zeros((10, 8), np.float32))
    try:
        rc = main(["bench", "check-server", "--policy", f"127.0.0.1:{port}", "--calls", "1",
                   "--timeout", "5"])
    finally:
        srv.shutdown()
    out = capsys.readouterr().out
    assert rc == 0 and entered == [True], out
    assert srv.handlers_returned(started=1)


def test_neither_wait_makes_the_users_server_log_a_failed_handshake(tmp_path, capsys):
    """The example server of the connect page, run as a user runs it. check-server's TCP wait and the
    driver's readiness wait (what `bench eval --policy` polls) used to open and close a bare TCP
    connection, which websockets logs as "opening handshake failed" with a traceback (about 30
    lines) on every check; both now complete a websocket handshake instead."""
    import subprocess
    import sys
    import time
    from pathlib import Path

    from axis_qvla.bench.sim import campaign

    example = Path(__file__).resolve().parents[4] / "experiments" / "examples" / "policy_server.py"
    log, port_file = tmp_path / "server.log", tmp_path / "port.txt"
    with log.open("w") as f:
        proc = subprocess.Popen([sys.executable, str(example), "--port", "0", "--port-file", str(port_file)],
                                stdout=f, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 60
        while not (port_file.is_file() and port_file.read_text().strip()):
            assert proc.poll() is None and time.monotonic() < deadline, log.read_text()
            time.sleep(0.1)
        port = int(port_file.read_text())
        assert main(["bench", "check-server", "--policy", f"127.0.0.1:{port}", "--calls", "1"]) == 0
        md = campaign.wait_ready("127.0.0.1", port, timeout_s=30.0, require_noise=True, proc=proc)
        assert md["protocol"] == "qvla.policy_server/1"
    finally:
        mock_server.stop(proc)
    capsys.readouterr()
    text = log.read_text()
    assert "opening handshake failed" not in text and "Traceback" not in text, text


def _example_server(tmp_path):
    """experiments/examples/policy_server.py on a free port, as the connect page starts it -> (proc, port)."""
    import subprocess
    import sys
    import time
    from pathlib import Path

    example = Path(__file__).resolve().parents[4] / "experiments" / "examples" / "policy_server.py"
    port_file = tmp_path / "port.txt"
    with (tmp_path / "server.log").open("w") as f:
        proc = subprocess.Popen([sys.executable, str(example), "--port", "0", "--port-file", str(port_file)],
                                stdout=f, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 60
    while not (port_file.is_file() and port_file.read_text().strip()):
        if proc.poll() is not None or time.monotonic() > deadline:
            mock_server.stop(proc)
            raise RuntimeError((tmp_path / "server.log").read_text())
        time.sleep(0.1)
    return proc, int(port_file.read_text())


def _bare_check_server(tmp_path, eval_python, *argv):
    """`qvla bench check-server` from an interpreter with no site-packages (`python -S`: the standard
    library and this checkout only), which is what `pip install -e ".[eval]"` gives a user, with a
    qvla.toml whose [venv] eval holds `eval_python`."""
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[4]
    (tmp_path / "qvla.toml").write_text(
        "[roots]\n" + "".join(f'{k} = "{tmp_path / k}"\n' for k in ("corpus", "cache", "scratch", "runs"))
        + f'[venv]\neval = "{eval_python.parent.parent}"\n')
    return subprocess.run([sys.executable, "-S", "-m", "axis_qvla.cli", "bench", "check-server", *argv],
                          cwd=tmp_path, env={"PYTHONPATH": str(repo), "PATH": "/usr/bin:/bin"},
                          capture_output=True, text=True, timeout=300)


def test_without_numpy_check_server_names_the_command_that_builds_the_eval_environment(tmp_path):
    r = _bare_check_server(tmp_path, tmp_path / "venv-eval" / "bin" / "python", "--policy", "127.0.0.1:1")
    assert r.returncode == 2, r.stdout + r.stderr
    assert "Traceback" not in r.stderr and "ModuleNotFoundError" not in r.stderr, r.stderr
    assert r.stdout.startswith("FAIL  `qvla bench check-server` runs under the evaluation environment (")
    assert "has no numpy" in r.stdout and "build it with `qvla env build eval`" in r.stdout


def test_without_numpy_check_server_re_executes_under_the_eval_environment(tmp_path):
    """The eval environment here is a launcher of this test's interpreter (which has the three modules):
    the check runs there against the connect page's example server and passes."""
    import sys

    python = tmp_path / "venv-eval" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    python.chmod(0o755)
    proc, port = _example_server(tmp_path)
    try:
        r = _bare_check_server(tmp_path, python, "--policy", f"127.0.0.1:{port}", "--calls", "1")
    finally:
        mock_server.stop(proc)
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"127.0.0.1:{port}: conformant (qvla.policy_server/1)" in r.stdout


def test_without_numpy_check_server_reports_a_qvla_toml_that_does_not_load(tmp_path):
    """A ./qvla.toml that does not load is named, not passed over for the default
    evaluation environment (which would send the user to build an environment they already have)."""
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[4]
    (tmp_path / "qvla.toml").write_text('[roots\ncorpus = "x"\n')
    r = subprocess.run([sys.executable, "-S", "-m", "axis_qvla.cli", "bench", "check-server", "--policy", "127.0.0.1:1"],
                       cwd=tmp_path, env={"PYTHONPATH": str(repo), "PATH": "/usr/bin:/bin"},
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 2, r.stdout + r.stderr
    assert r.stdout.startswith("FAIL  ") and "qvla.toml is not valid TOML" in r.stdout
    assert ".venv-eval" not in r.stdout and "Traceback" not in r.stderr
