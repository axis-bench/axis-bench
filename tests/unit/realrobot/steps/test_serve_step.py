"""Step 8: the serve config, openpi's server command, and the handshake held to the run."""
from __future__ import annotations

import json
import os

from axis_qvla.cli import main
from axis_qvla.realrobot.fr3.line import load_line
from axis_qvla.realrobot.fr3.local_runs import write_local_runs
from axis_qvla.realrobot.layout import LINE
from axis_qvla.realrobot.steps import serve as step
from tests.unit.realrobot.fr3.test_checks import _handshake
from tests.unit.realrobot.steps.hosts import local_layout

ONLY, COND = "pi05.finetune.realrobot_only", "pi05.finetune.realrobot_cond"


class FakeServer:
    """A started server: never a real process. `exits` is what poll() reports (None while it runs)."""

    def __init__(self, exits=None):
        self.terminated = self.killed = False
        self.waits, self.exits = [], exits

    def poll(self):
        return self.exits

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        self.waits.append(timeout)
        self.exits = 0 if self.exits is None else self.exits        # it has exited once waited on
        return self.exits


def _served(tmp_path, monkeypatch, handshake, *, server=None, busy=False):
    toml, lay = local_layout(tmp_path)
    write_local_runs(lay.paths, load_line(LINE), runs=[ONLY])
    lay.checkpoint(ONLY).mkdir(parents=True)
    python = tmp_path / "openpi" / ".venv" / "bin" / "python"          # the openpi environment, as if built
    python.parent.mkdir(parents=True)
    python.touch()
    server, launched = server or FakeServer(), []
    monkeypatch.setattr(step, "_popen", lambda argv, env, cwd: launched.append((argv, env, cwd)) or server)
    monkeypatch.setattr(step, "_probe", lambda lay_, host, port, timeout: handshake)
    # Nothing listens before the server starts (unless `busy`); the started server listens at once.
    monkeypatch.setattr(step, "_listening", lambda host, port: busy or bool(launched))
    return toml, lay, server, launched


def test_print_writes_the_serve_config_and_prints_openpis_server(tmp_path, monkeypatch, capsys):
    toml, lay, _server, launched = _served(tmp_path, monkeypatch, _handshake(prompt_suffix=""))
    lay.checkpoint(ONLY).rmdir()                    # --print needs no checkpoint, no environment
    (tmp_path / "openpi" / ".venv" / "bin" / "python").unlink()
    assert main(["realrobot", "serve", "only", "--host", toml, "--print"]) == 0
    out = capsys.readouterr().out
    assert lay.serve_config(ONLY).is_file() and not launched
    assert f"serve config of {ONLY} -> {lay.serve_config(ONLY)}" in out
    assert (f"scripts/serve_policy.py --port=8000 policy:checkpoint --policy.config={lay.serve_config(ONLY)} "
            f"--policy.dir={lay.checkpoint(ONLY)}") in out
    assert "OPENPI_DATA_HOME=" in out and "--quality-tag" not in out


def test_the_conditioned_run_is_served_with_its_quality_tag(tmp_path):
    _toml, lay = local_layout(tmp_path)
    argv, env, cwd = step.server_command(lay, load_line(LINE), COND, 8000)
    assert "--quality-tag=5" in argv and "--guidance-scale=0" in argv
    assert argv[0] == str(tmp_path / "openpi" / ".venv" / "bin" / "python") and cwd == tmp_path / "openpi"
    assert env["OPENPI_DATA_HOME"].startswith(str(lay.paths.cache))


def test_serve_starts_the_server_and_passes_a_conforming_handshake(tmp_path, monkeypatch, capsys):
    toml, lay, server, launched = _served(tmp_path, monkeypatch, _handshake(prompt_suffix=""))
    assert main(["realrobot", "serve", "only", "--host", toml]) == 0
    assert len(launched) == 1 and server.waits == [None] and not server.terminated
    assert launched[0][2] == tmp_path / "openpi" and f"--policy.dir={lay.checkpoint(ONLY)}" in launched[0][0]
    out = capsys.readouterr().out
    assert "ALL PASS" in out and f"serving {ONLY} on port 8000; Ctrl+C stops it" in out


def test_a_mismatched_handshake_stops_the_server(tmp_path, monkeypatch, capsys):
    toml, _lay, server, _l = _served(tmp_path, monkeypatch, _handshake(prompt_suffix="", action_fps=10))
    assert main(["realrobot", "serve", "only", "--host", toml]) == 1
    assert server.terminated and server.waits and "fps" in capsys.readouterr().out


def test_a_server_that_exits_before_it_listens_is_reported(tmp_path, monkeypatch, capsys):
    toml, _lay, server, _l = _served(tmp_path, monkeypatch, _handshake(prompt_suffix=""), server=FakeServer(exits=1))
    monkeypatch.setattr(step, "_listening", lambda host, port: False)
    assert main(["realrobot", "serve", "only", "--host", toml]) == 1
    assert "exited 1 before it listened on port 8000" in capsys.readouterr().err and not server.terminated


def test_a_server_that_never_listens_is_stopped_at_the_timeout(tmp_path, monkeypatch, capsys):
    toml, _lay, server, _l = _served(tmp_path, monkeypatch, _handshake(prompt_suffix=""))
    monkeypatch.setattr(step, "_listening", lambda host, port: False)
    assert main(["realrobot", "serve", "only", "--host", toml, "--timeout", "0"]) == 2
    assert server.terminated and "not listening on port 8000 after 0 s" in capsys.readouterr().err


def test_a_port_already_in_use_starts_nothing(tmp_path, monkeypatch, capsys):
    toml, _lay, _server, launched = _served(tmp_path, monkeypatch, _handshake(prompt_suffix=""), busy=True)
    assert main(["realrobot", "serve", "only", "--host", toml]) == 2
    assert not launched and "--check 127.0.0.1:8000" in capsys.readouterr().err


def test_check_holds_a_running_server_to_the_run_and_starts_nothing(tmp_path, monkeypatch, capsys):
    toml, _lay, _server, launched = _served(tmp_path, monkeypatch, _handshake(prompt_suffix=""))
    assert main(["realrobot", "serve", "only", "--check", "10.0.0.5:8000", "--host", toml]) == 0
    assert not launched and "ALL PASS" in capsys.readouterr().out


def test_refusals(tmp_path, monkeypatch, capsys):
    toml, lay, _server, launched = _served(tmp_path, monkeypatch, _handshake(prompt_suffix=""))
    lay.checkpoint(ONLY).rmdir()
    assert main(["realrobot", "serve", "only", "--host", toml]) == 2
    assert "qvla realrobot finetune --runs only" in capsys.readouterr().err
    assert main(["realrobot", "serve", "bc", "--host", toml]) == 2          # no run spec written for bc
    assert "qvla realrobot finetune --runs bc" in capsys.readouterr().err
    assert main(["realrobot", "serve", "nope", "--host", toml]) == 2
    assert "not a fine-tuning run" in capsys.readouterr().err
    assert main(["realrobot", "serve", "only", "--check", "nohost", "--host", toml]) == 2
    assert "HOST:PORT" in capsys.readouterr().err
    assert not launched


def test_a_missing_openpi_environment_names_its_build(tmp_path, monkeypatch, capsys):
    toml, _lay, _server, launched = _served(tmp_path, monkeypatch, _handshake(prompt_suffix=""))
    (tmp_path / "openpi" / ".venv" / "bin" / "python").unlink()
    assert main(["realrobot", "serve", "only", "--host", toml]) == 2
    assert "qvla env build pi05" in capsys.readouterr().err
    assert main(["realrobot", "serve", "only", "--check", "10.0.0.5:8000", "--host", toml]) == 2
    assert "qvla env build pi05" in capsys.readouterr().err and not launched


def test_stale_demo_norm_statistics_are_refused(tmp_path, monkeypatch, capsys):
    toml, lay, _server, launched = _served(tmp_path, monkeypatch, _handshake(prompt_suffix=""))
    stand_in = {k: {"mean": [0.0] * 8, "std": [1.0] * 8, "q01": [-1.0] * 8, "q99": [1.0] * 8} for k in ("state", "actions")}
    lay.norm_file("finetune").write_text(json.dumps({"norm_stats": stand_in}))
    assert main(["realrobot", "serve", "only", "--host", toml]) == 1
    assert "qvla realrobot finetune" in capsys.readouterr().err and not launched


def test_stop_kills_only_a_server_that_ignores_terminate():
    class Stubborn(FakeServer):
        def wait(self, timeout=None):
            self.waits.append(timeout)
            if timeout is not None and not self.killed:
                raise step.subprocess.TimeoutExpired("serve_policy.py", timeout)
            return -9

    server = Stubborn()
    step._stop(server)
    assert server.terminated and server.killed and server.waits == [step.STOP_GRACE_S, None]


def test_the_probe_runs_in_openpis_environment(tmp_path, monkeypatch):
    _toml, lay = local_layout(tmp_path)
    seen = {}

    class Done:
        returncode, stdout, stderr = 0, json.dumps({"action_fps": 15}), ""

    monkeypatch.setattr(step.subprocess, "run", lambda argv, **kw: seen.update(argv=argv, env=kw["env"]) or Done())
    assert step._probe(lay, "127.0.0.1", 8000, 5.0) == {"action_fps": 15}
    assert seen["argv"][:3] == [str(tmp_path / "openpi" / ".venv" / "bin" / "python"), "-m", "axis_qvla.realrobot.fr3.probe"]
    assert str(step.REPO) in seen["env"]["PYTHONPATH"].split(os.pathsep)


def test_a_failed_probe_is_a_handshake_error_naming_its_last_line(tmp_path, monkeypatch):
    import pytest

    _toml, lay = local_layout(tmp_path)

    class Failed:
        returncode, stdout, stderr = 2, "", "noise\nFAIL  no server listening on 127.0.0.1:8000 after 5 s\n"

    monkeypatch.setattr(step.subprocess, "run", lambda argv, **kw: Failed())
    with pytest.raises(step.HandshakeError, match="no server listening"):
        step._probe(lay, "127.0.0.1", 8000, 5.0)


def test_a_server_that_exits_while_its_handshake_is_read_does_not_pass(tmp_path, monkeypatch, capsys):
    toml, _lay, server, _l = _served(tmp_path, monkeypatch, _handshake(prompt_suffix=""))

    def probe(lay_, host, port, timeout):
        server.exits = 1                # ours died; whatever answered on the port was not it
        return _handshake(prompt_suffix="")

    monkeypatch.setattr(step, "_probe", probe)
    assert main(["realrobot", "serve", "only", "--host", toml]) == 1
    cap = capsys.readouterr()
    assert "exited 1 while its handshake was read" in cap.err
    assert "ALL PASS" not in cap.out and "serving" not in cap.out and not server.terminated


def test_a_sigterm_stops_the_server_and_the_old_handler_comes_back(tmp_path, monkeypatch, capsys):
    """`kill <qvla pid>` while serving runs the `finally` that stops the server, as Ctrl+C does, and a
    second SIGTERM during the stop does not cut it short."""
    import signal

    class Killed(FakeServer):
        def wait(self, timeout=None):
            os.kill(os.getpid(), signal.SIGTERM)        # serving (timeout None), then again while stopping
            return super().wait(timeout)

    reached = []
    sentinel = lambda *a: reached.append(a)             # noqa: E731 -- receives the SIGTERM if serve installs nothing
    previous = signal.signal(signal.SIGTERM, sentinel)
    try:
        toml, _lay, server, _l = _served(tmp_path, monkeypatch, _handshake(prompt_suffix=""), server=Killed())
        assert main(["realrobot", "serve", "only", "--host", toml]) == 0
        assert server.terminated and server.waits == [step.STOP_GRACE_S] and not server.killed
        assert not reached and signal.getsignal(signal.SIGTERM) is sentinel
        assert "serving" in capsys.readouterr().out
    finally:
        signal.signal(signal.SIGTERM, previous)
