"""Step 8 of the real-world tutorial: `qvla realrobot serve RUN`, a fine-tuned run's policy server, checked against the run.

Why: one command for what serving needs, in two environments:
- `fr3 serve-config` with an --out path;
- openpi's serve_policy.py with eight flags typed from the page (the quality tag only for `cond`);
- `qvla bench check-server` in the evaluation environment;
- `fr3 check` in the offline one, because check-server judges a server against the simulator and fails
  an FR3 server's units.
This command does it all for a fine-tuning run (`bc`, `cond`, `only`). It also writes the run's drive
kit (axis_qvla.realrobot.steps.drive_kit), with `--print` as well; a kit file already there that was edited
is kept and printed as kept, not clobbered, unless `--replace-kit` asks for this checkout's copy
instead. It checks the demonstrations' norm statistics, and writes the serve config beside the run
(<runs>/<run id>/serve.json). It starts openpi's server in the openpi environment ([venv] online of
qvla.toml, from [roots] openpi) on the run's final checkpoint, with the run's serve condition from the
line record. Once the server listens, its handshake is read with openpi's client. That is
axis_qvla.realrobot.fr3.probe, run by the openpi environment's interpreter, because this environment has no
openpi client. The handshake is then held to the run with `fr3 check`'s rules. The command prints ALL
PASS and keeps serving until Ctrl+C; a failed check stops the server (exit 1). `--check HOST:PORT`
checks a server that is already running (started from `--print`'s command on a cluster, say) and
starts nothing -- and writes no kit.

The server is this command's own child, and only it is ever stopped: by its PID (Popen.terminate,
then kill after STOP_GRACE_S), never by a process pattern. That holds for Ctrl+C and for a SIGTERM
(`kill <pid>` of this command), which is taken as Ctrl+C while the server runs. A port something
already listens on is refused before anything starts, and the started server must still be running
once its handshake is read. So the handshake read is the started server's, unless another process
took the port between that refusal and the server's start.

Exit 0 served, printed or checked; 1 a check failed or the server exited before it listened; 2 a
refused input or an unreachable server. Module scope is stdlib only.
"""
from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from axis_qvla.realrobot.layout import LINE, REPO, layout, run_id
from axis_qvla.realrobot.steps.common import fail

PROBE = __name__.rsplit(".", 2)[0] + ".fr3.probe"      # axis_qvla.realrobot.fr3.probe, whatever the package is called
ENV = "pi05"                                           # `qvla env build pi05` builds the openpi environment
STOP_GRACE_S = 30.0                                    # SIGTERM, then this long before SIGKILL
POLL_S = 1.0


class HandshakeError(RuntimeError):
    """The server's handshake could not be read."""


def _python(lay) -> Path:
    venv = lay.paths.host.venvs.get("online")
    if venv is None:
        raise ValueError("qvla.toml declares no [venv] online: the openpi environment that serves the run "
                         f"(build it with `qvla env build {ENV}`)")
    return Path(venv) / "bin" / "python"


def _require_env(lay) -> None:
    python = _python(lay)
    if not python.is_file():
        raise ValueError(f"no openpi environment ({python} is missing): build it with `qvla env build {ENV}`")


def server_command(lay, line, rid: str, port: int) -> tuple[list[str], dict[str, str], Path]:
    """(argv, environment, working directory) of openpi's server for run `rid`'s final checkpoint."""
    from axis_qvla.realrobot.fr3.line import QUALITY_TAG
    from axis_qvla.train import ids

    openpi = lay.paths.host.roots.get("openpi")
    if openpi is None:
        raise ValueError("qvla.toml needs [roots] openpi (openpi's checkout, whose scripts/serve_policy.py serves the run)")
    name = line.runs[rid]["serve"]
    cond = line.serve[name]
    tag = []
    if cond["prompt_suffix"]:
        if cond["prompt_suffix"] != f"\nQuality: {QUALITY_TAG}":
            raise ValueError(f"{rid}: serve condition {name!r} is not the quality tag {QUALITY_TAG}")
        tag = [f"--quality-tag={QUALITY_TAG}", f"--guidance-scale={float(cond['guidance']):g}"]
    argv = [str(_python(lay)), "scripts/serve_policy.py", f"--port={port}", *tag, "policy:checkpoint",
            f"--policy.config={lay.serve_config(rid)}", f"--policy.dir={lay.checkpoint(rid)}"]
    return argv, {"OPENPI_DATA_HOME": str(ids.openpi_home(lay.paths))}, Path(openpi)


def _popen(argv, env, cwd):
    return subprocess.Popen(argv, env={**os.environ, **env}, cwd=cwd)


def _listening(host: str, port: int) -> bool:
    from axis_qvla.realrobot.fr3.ports import knock      # the readiness poll the probe makes too

    return knock(host, port)


def _probe(lay, host: str, port: int, timeout: float) -> dict:
    path = os.pathsep.join(p for p in (str(REPO), os.environ.get("PYTHONPATH", "")) if p)
    got = subprocess.run([str(_python(lay)), "-m", PROBE, f"{host}:{port}", "--timeout", f"{timeout:g}"],
                         env={**os.environ, "PYTHONPATH": path}, capture_output=True, text=True, check=False)
    if got.returncode != 0:
        raise HandshakeError((got.stderr.strip().splitlines() or [f"the probe exited {got.returncode}"])[-1]
                             .removeprefix("FAIL  "))
    try:
        return json.loads(got.stdout)
    except json.JSONDecodeError:
        raise HandshakeError(f"the probe printed no handshake: {got.stdout[:200]!r}") from None


def _check(lay, line, rid: str, host: str, port: int, timeout: float, server=None) -> int:
    """Hold the handshake at host:port to run `rid`. With `server` (the one this command started), it
    must still be running once the handshake is read: a handshake read after it exited is another's."""
    from axis_qvla.realrobot.fr3.checks import check_serve

    md = _probe(lay, host, port, timeout)
    rc = None if server is None else server.poll()
    if rc is not None:
        return fail(f"openpi's server exited {rc} while its handshake was read (its output is above)", 1)
    problems = check_serve(md, line, rid, lay.demos, "fr3_droid")
    for p in problems:
        print(f"FAIL  {p}")
    print("ALL PASS" if not problems else f"{len(problems)} problem{'s' * (len(problems) != 1)}")
    return 1 if problems else 0


def _endpoint(text: str) -> tuple[str, int]:
    host, _, port = text.rpartition(":")
    if not host or not port.isdigit():
        raise ValueError(f"--check {text}: give HOST:PORT")
    return host, int(port)


def _stop(server) -> None:
    """Stop the server this command started, by its PID: SIGTERM, and SIGKILL if it outlives the grace."""
    server.terminate()
    try:
        server.wait(timeout=STOP_GRACE_S)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()


@contextmanager
def _sigterm_as_interrupt():
    """While the started server runs, a SIGTERM to this command raises KeyboardInterrupt, as Ctrl+C
    does, so the `finally` that stops the server runs; without it SIGTERM ends this process and leaves
    the server running. Yields a list: once it is not empty (the first SIGTERM, or the caller appending
    to it before it stops the server) SIGTERMs are ignored, so stopping the server is not cut short.
    The old handler comes back on exit. Signals are the main thread's only: elsewhere nothing changes."""
    hit: list = []
    if threading.current_thread() is not threading.main_thread():
        yield hit
        return

    def interrupt(signum, frame):
        if not hit:
            hit.append(signum)
            raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        yield hit
    finally:
        signal.signal(signal.SIGTERM, previous)


def _wait_listening(server, port: int, timeout: float) -> int | None:
    """None once the started server listens on `port`; else the exit code to return: 1 when it exited
    first, 2 at the timeout (the caller stops it)."""
    deadline = time.monotonic() + timeout
    while not _listening("127.0.0.1", port):
        rc = server.poll()
        if rc is not None:
            return fail(f"openpi's server exited {rc} before it listened on port {port} (its output is above)", 1)
        if time.monotonic() >= deadline:
            return fail(f"openpi's server is not listening on port {port} after {timeout:g} s; stopping it "
                        f"(a larger --timeout gives the checkpoint longer to load)")
        time.sleep(POLL_S)
    return None


def serve(args) -> int:
    from axis_qvla.core.hosts import HostProfileError
    from axis_qvla.realrobot.fr3.line import load_line
    from axis_qvla.realrobot.fr3.local_runs import write_serve_config
    from axis_qvla.realrobot.fr3.norm_check import check_norm

    try:
        lay, line = layout(args.host), load_line(LINE)
        rid = run_id("finetune", args.run)
        if line.runs.get(rid, {}).get("stage") != "finetune":
            return fail(f"{args.run}: not a fine-tuning run of the line record (bc, cond, only)")
        spec = lay.run_spec(rid)
        if not spec.is_file():
            return fail(f"no run spec {spec}: run `qvla realrobot finetune --runs {args.run}` first")
        stale = check_norm(lay.norm_file("finetune"), lay.demos, "fr3_droid")
        if stale:
            for p in stale:
                print(f"FAIL  {p}")
            return fail(f"the fine-tuning norm statistics do not describe {lay.demos}: run `qvla realrobot finetune` again", 1)
        if args.check is not None:
            endpoint = _endpoint(args.check)
            _require_env(lay)
            return _check(lay, line, rid, *endpoint, args.timeout)
        _rid, config = write_serve_config(spec, None, lay.paths)
        print(f"serve config of {rid} -> {config}")
        from axis_qvla.realrobot.steps.drive_kit import write_drive_kit
        kit = write_drive_kit(lay, rid, replace=args.replace_kit)
        print(f"drive kit -> {kit} (an optional client for a DROID control stack: docs Real robot > Connect your robot)")
        argv, env, cwd = server_command(lay, line, rid, args.port)
        if args.print:
            print(f"cd {shlex.quote(str(cwd))} && " + shlex.join(["env", *(f"{k}={v}" for k, v in env.items()), *argv]))
            return 0
        if not lay.checkpoint(rid).exists():
            return fail(f"no checkpoint {lay.checkpoint(rid)}: train it with `qvla realrobot finetune --runs {args.run}`")
        _require_env(lay)
        if _listening("127.0.0.1", args.port):
            return fail(f"port {args.port} is in use: pass another --port, or check the server already there "
                        f"with `qvla realrobot serve {args.run} --check 127.0.0.1:{args.port}`")
    except (HandshakeError, ValueError, OSError, HostProfileError) as exc:
        return fail(str(exc))
    print(f"starting openpi's server for {rid} on port {args.port}", flush=True)
    with _sigterm_as_interrupt() as stopping:
        server, serving = _popen(argv, env, cwd), False
        try:
            rc = _wait_listening(server, args.port, args.timeout)
            if rc is None:
                rc = _check(lay, line, rid, "127.0.0.1", args.port, args.timeout, server)
            if rc != 0:
                return rc
            print(f"serving {rid} on port {args.port}; Ctrl+C stops it", flush=True)
            serving = True
            return server.wait()
        except HandshakeError as exc:
            return fail(str(exc))
        except KeyboardInterrupt:
            return 0 if serving else 130
        finally:
            if server.poll() is None:   # a failed check, a refusal, Ctrl+C, SIGTERM or any error: stop what we started
                stopping.append(True)
                _stop(server)
