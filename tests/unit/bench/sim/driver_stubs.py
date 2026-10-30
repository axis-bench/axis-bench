"""Stubs for running the campaign driver, its queue workers and the real rollout CLI in one process.

The driver, the queue worker and the rollout CLI are three processes whose only contract is argv.
These stubs run all three for real except what needs a GPU: servers are stubs, each worker is
queue_worker.main on the argv the driver built (in-process), and each rollout is the rollout CLI's
own parser, task source and seed resolution, writing seed files through record.write_seed where the
CLI would write them. So a test sees what crosses each argv boundary: the log root the driver
tallies is the one every rollout writes under, and the benchmark and n_seeds reach every worker.

Shared by the driver tests (test_eval_queue_driver.py), the development-seed and results-build
tests, and the execution-constant tests, so none of them imports another test module's privates.
"""
from __future__ import annotations

from pathlib import Path

from axis_qvla.bench.sim import campaign as drv


class SubprocessShim:
    """The subprocess module with chosen attributes replaced, inside one module's namespace only."""

    def __init__(self, **over):
        self.__dict__.update(over)

    def __getattr__(self, name):
        import subprocess

        return getattr(subprocess, name)


class FakeProc:
    """A started child that is alive until waited on, then exits `rc`."""

    def __init__(self, rc: int = 0):
        self.returncode, self.pid = rc, 4242

    def poll(self):
        return None

    def wait(self, timeout=None):
        return self.returncode

    def send_signal(self, sig):
        pass

    def kill(self):
        pass


def make_checkpoint(tmp_path: Path) -> Path:
    """A one-file checkpoint directory (the driver digests it by content)."""
    ck = tmp_path / "ck"
    ck.mkdir(exist_ok=True)
    (ck / "w.bin").write_bytes(b"x")
    return ck


def stub_probes(monkeypatch, seen: dict | None = None) -> None:
    """No interpreter, GPU or git probes; records the interpreters probed in seen["probed"]."""
    probed = [] if seen is None else seen.setdefault("probed", [])
    monkeypatch.setattr(drv, "python_versions", lambda python: probed.append(python) or {})
    monkeypatch.setattr(drv, "gpu_info", lambda: {})
    monkeypatch.setattr(drv, "git_state", lambda d: {"path": str(d)})
    monkeypatch.setattr(drv, "log", lambda msg: None)


def driver_stack(monkeypatch, tmp_path: Path) -> dict:
    """Stub servers, run each worker in process and each rollout through the rollout CLI's parser;
    returns what was seen: every worker's argv, every rollout's parsed arguments, server starts."""
    import subprocess

    from axis_qvla.bench.sim import cli, runner
    from axis_qvla.bench.sim import queue_worker as qw
    from axis_qvla.bench.sim.arms import served_arm_name
    from axis_qvla.bench.sim.record import SeedResult, write_seed

    seen: dict = {"workers": [], "rollouts": [], "servers": 0}

    def rollout(cmd, **kw):
        assert cmd[1:3] == ["-m", qw.ROLLOUT_MODULE]
        ns = cli.build_parser().parse_args(cmd[3:])
        seen["rollouts"].append(ns)
        try:
            src = cli.resolve_task_source(ns)
            n = ns.n_seeds if ns.n_seeds is not None else src.n_seeds_default
            seeds = runner.resolve_seeds(src.seeds, n, only=cli._parse_seed_subset(ns.seeds))
        except (SystemExit, runner.RunnerError) as exc:
            return subprocess.CompletedProcess(cmd, 1, "", str(exc))
        arm = served_arm_name(ns.variant, cfg_conditioning=False, guidance_scale=0.0, domain="dualsim")
        for s in seeds:
            ok = s % 2 == 0
            write_seed(ns.log_root / arm / f"task_{ns.task_id}",
                       SeedResult(s, ok, 10, 1.0, "success" if ok else "timeout"))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def popen(cmd, **kw):
        assert cmd[1:3] == ["-m", qw.__name__]
        seen["workers"].append(list(cmd))
        try:
            rc = qw.main(cmd[3:])
        except RuntimeError:   # a worker whose batches keep failing gives up, non-zero
            rc = 1
        return FakeProc(rc)

    def start_server(*a, **k):
        seen["servers"] += 1
        return FakeProc()

    monkeypatch.setattr(qw, "subprocess", SubprocessShim(run=rollout))
    monkeypatch.setattr(drv, "subprocess", SubprocessShim(Popen=popen))
    monkeypatch.setattr(drv, "start_server", start_server)
    monkeypatch.setattr(drv, "wait_ready", lambda *a, **k: {"action_horizon": 10, "action_dim": 8})
    stub_probes(monkeypatch, seen)
    return seen
