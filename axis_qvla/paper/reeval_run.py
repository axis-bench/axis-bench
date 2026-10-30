"""Running a re-evaluation's plan: each step a `qvla` command, started and waited for by this process.

Why: one command re-evaluates a result, Table I or a figure, on this machine. The
fetches (`qvla data get`, stdlib) run in this interpreter. Each evaluation is `qvla bench eval` under
the host's evaluation environment: a child this process starts and waits for by its pid -- never a `ps | grep`
pattern its own argv could match, never a bare `wait`. The child starts its policy servers and
rollout workers and stops exactly those (axis_qvla.bench.sim.campaign). On Ctrl-C the terminal sends the
same SIGINT to the child; this process waits for it to stop them and, if it does not, terminates and
then kills only that child -- at once on a second Ctrl-C. A child ended by a signal exits 128 + the
signal number, as a shell reports it. A step whose results files are all there is skipped, so
running the same command again resumes: `bench eval` writes a results file only for a complete run
(axis_qvla.bench.finish builds and scores the whole run before an atomic write), so a file that is there
is a finished evaluation. Each evaluation's wall time is printed.

The host's side lives here too -- its roots and whether the environments the evaluations run under
are built -- so `paper tables` (axis_qvla.paper.cli) never imports the evaluation launcher: this module is
off the T0 path, the one module of axis_qvla/paper that reads a clock and the environment.

Core tier: stdlib only.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from axis_qvla.bench.finish import results_path
from axis_qvla.paper.reeval import Step

STOP_TIMEOUT_S = 120             # after Ctrl-C: for the child to stop the servers and workers it started
TERM_GRACE_S = 10                 # after SIGTERM, before SIGKILL
INTERRUPTED = 130                 # the shell's exit code for a SIGINT


class HostError(RuntimeError):
    """The host profile cannot be read, or lacks a root a re-evaluation writes under."""


def load_host(host_arg: str | None):
    """(the host profile `--host` names, its run root -- where each results file goes --, its runs
    root -- under which the tables go by default)."""
    from axis_qvla.bench.launch import LaunchError, run_root
    from axis_qvla.core.hosts import HostProfileError
    from axis_qvla.core.paths import Paths
    from axis_qvla.data.userconf import load_profile

    try:
        host = load_profile(host_arg)
        return host, run_root(host), Paths(host).runs
    except (HostProfileError, LaunchError, KeyError) as exc:
        raise HostError(str(exc)) from None


def env_gap(host, backbones: Sequence[str], prog: str) -> tuple[Path | None, str | None]:
    """(the evaluation-environment interpreter the evaluations run under, why an environment they need is not
    there -- naming the one `env build` command that builds it -- or None). Checked before any byte
    is fetched; nothing is built here (a build is several GB and needs a network)."""
    from axis_qvla.bench.launch import EvalRequest, interpreter_problem, missing_env
    from axis_qvla.bench.paper_checkpoint import BACKEND_OF_BACKBONE

    from axis_qvla.core.hosts import EVAL_VENV

    venv = host.venvs.get(EVAL_VENV)
    python = venv / "bin" / "python" if venv is not None else None
    gap = "is not declared ([venv] eval)" if python is None else interpreter_problem(python)
    if gap:
        return python, (f"the evaluations run under the evaluation environment ({python}), which {gap}: build it "
                        f"with `{prog} env build eval`. Nothing was fetched.")
    for bb in dict.fromkeys(backbones):
        gap = missing_env(EvalRequest(checkpoint=None, config=None, backend=BACKEND_OF_BACKBONE[bb]), host)
        if gap:
            return python, f"{gap} Nothing was fetched."
    return python, None


def _there(path: Path) -> bool:
    """An explicit stat, not Path.is_file, which can answer False for EACCES: an unreadable results
    directory raises (OSError) instead of reading as "not evaluated yet" and evaluating again."""
    try:
        return path.stat().st_mode & 0o170000 == 0o100000
    except (FileNotFoundError, NotADirectoryError):
        return False


def _done(step: Step, run_root: Path) -> bool:
    """Every results file of the step is there. Presence is completeness: axis_qvla.bench.finish writes a
    results file only for a whole scored run, and atomically (temp file, then rename), so no partial
    file is ever at that path."""
    return bool(step.runs) and all(_there(results_path(run_root, r)) for r in step.runs)


def exit_code(rc: int) -> int:
    """A child's return code as a shell reports it: a child ended by signal N (Popen's -N) is 128 + N,
    so it never reads as success or as a small exit code of the command's own."""
    return 128 - rc if rc < 0 else rc


def _stop(p) -> None:
    """After the first Ctrl-C: STOP_TIMEOUT_S for the child to stop what it started (the terminal sent
    it the same SIGINT), then SIGTERM and TERM_GRACE_S, then SIGKILL. A second Ctrl-C during either
    wait goes straight to SIGKILL, so the child never outlives this process's interrupt."""
    try:
        try:
            p.wait(timeout=STOP_TIMEOUT_S)
            return
        except subprocess.TimeoutExpired:
            p.terminate()
        try:
            p.wait(timeout=TERM_GRACE_S)
            return
        except subprocess.TimeoutExpired:
            pass
    except KeyboardInterrupt:
        pass
    p.kill()
    while True:                     # SIGKILL cannot be ignored: this wait is short, and is not cut short
        try:
            p.wait()
            return
        except KeyboardInterrupt:
            continue


def wait_child(argv: Sequence[str], env: dict, *, popen=subprocess.Popen) -> int:
    """Start `argv` and wait for that pid: its exit code (`exit_code`). On Ctrl-C stop it -- only it --
    (`_stop`) and re-raise."""
    p = popen(list(argv), env=env)
    try:
        return exit_code(p.wait())
    except KeyboardInterrupt:
        _stop(p)
        raise


def _in_process(fn: Callable[[list[str]], int], argv: list[str]) -> int:
    """A fetch run in this interpreter; an argparse refusal (SystemExit) is its exit code, not ours.
    SystemExit(None) is success, as it is for the interpreter (sys.exit())."""
    try:
        return int(fn(argv))
    except SystemExit as exc:
        if exc.code is None:
            return 0
        return exc.code if isinstance(exc.code, int) else 2


def run_steps(steps: Sequence[Step], *, run_root: Path, render_python: Path, prog: str = "qvla",
              in_process: Callable[[list[str]], int] | None = None, popen=subprocess.Popen,
              report: Callable[[str], None] = print, clock: Callable[[], float] = time.monotonic) -> int:
    """0 when every step ran (or was done already); else the first failing step's exit code, after
    which nothing else runs; INTERRUPTED on Ctrl-C, once the running evaluation has stopped."""
    from axis_qvla.cli import main as qvla_main
    from axis_qvla.core.hosts import REPO_ROOT

    in_process = in_process or qvla_main
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    resumes = "Running the same command again resumes: results already evaluated are kept"
    for step in steps:
        line = f"{prog} {shlex.join(step.argv)}"
        try:
            done = _done(step, run_root)
        except OSError as exc:
            report(f"FAIL  whether `{line}` is done already cannot be checked: {exc}")
            return 3
        if done:
            report(f"done already: {line}")
            continue
        report(f"run: {line}")
        try:
            if step.argv[:2] == ("bench", "eval"):
                t0 = clock()
                rc = wait_child([str(render_python), "-m", "axis_qvla.cli", *step.argv], env, popen=popen)
                report(f"evaluated {step.runs[0]} in {clock() - t0:.0f} s (exit {rc})")
            else:
                rc = _in_process(in_process, list(step.argv))
        except KeyboardInterrupt:
            report(f"INTERRUPTED  `{line}`; nothing after it ran. {resumes}")
            return INTERRUPTED
        if rc:
            report(f"FAIL  `{line}` exited {rc}; nothing after it ran. {resumes}")
            return rc
    return 0
