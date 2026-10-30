"""`bench eval` under a tmp host profile with the campaign driver replaced: the argv and the stub.

Shared by the eval-roots and execution-constant tests (the host profile itself is the `host`
fixture of test_eval_roots.py).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def eval_argv(d: Path, *extra: str) -> list[str]:
    """`bench eval` on the `host` fixture's demo profile, checkpoint and asset directory."""
    return ["bench", "eval", "--host", "demo", "--checkpoint", str(d / "ck"), "--config", "cfg",
            "--run", "probe", "--asset-dir", str(d / "assets"), *extra]


def fake_driver(monkeypatch, d: Path) -> dict:
    """Replace the campaign driver; returns the argv and environment it was called with."""
    from axis_qvla.bench.sim import campaign

    seen: dict = {}

    def fake_main(argv):
        seen["argv"], seen["env"] = list(argv), dict(os.environ)
        return 0
    monkeypatch.setattr(campaign, "main", fake_main)
    monkeypatch.setattr(sys, "executable", str(d / "venvs" / "render" / "bin" / "python"))
    ready_host(monkeypatch, d)
    return seen


def make_envs(d: Path) -> None:
    """The render and pi0.5 server interpreters the `host` fixtures' profile names, as files: `bench
    eval` checks both are there before it starts anything."""
    for venv in ("render", "online"):
        py = d / "venvs" / venv / "bin" / "python"
        py.parent.mkdir(parents=True, exist_ok=True)
        py.touch()


def ready_host(monkeypatch, d: Path) -> None:
    """What every test that replaces the driver also needs: the interpreters on disk, and no results
    file built from a driver that wrote no run (axis_qvla.bench.finish is replaced by a no-op)."""
    from axis_qvla.bench import finish

    make_envs(d)
    monkeypatch.setattr(finish, "finish", lambda *a, **k: 0)


def ready_profile(monkeypatch) -> None:
    """`ready_host` for a checked-in host profile, whose interpreters live on that host rather than
    under tmp_path: every interpreter reads as built, and axis_qvla.bench.finish is a no-op."""
    from axis_qvla.bench import finish, launch

    monkeypatch.setattr(launch, "interpreter_problem", lambda python: None)
    monkeypatch.setattr(finish, "finish", lambda *a, **k: 0)


def bench_inputs(d: Path, monkeypatch) -> list[str]:
    """The on-disk inputs a real run checks before the driver (a locked Polyhaven bank, tasks,
    assets), as the `bench eval` flags that name them."""
    from tests.unit.bench.test_eval_roots import make_bank

    make_bank(d / "render", d / "lock.json", monkeypatch)
    for sub in ("tasks", "assets"):
        (d / sub).mkdir(exist_ok=True)
    return ["--asset-dir", str(d / "assets"), "--render-root", str(d / "render"), "--task-data-root", str(d / "tasks")]


def lerobot_host(d: Path) -> None:
    """The `host` fixture's demo profile (tests/unit/bench/test_openpi_serve.py) with the roots the
    LeRobot servers (SmolVLA, X-VLA) run from."""
    from tests.unit.bench.test_openpi_serve import PROFILE

    roots = "".join(f'{k} = "{d}/{k}"\n' for k in ("smolvla_venv", "smolvla_bench"))
    (d / "hosts" / "demo.toml").write_text(PROFILE.format(d=d).replace("[venv]", roots + "\n[venv]"))
