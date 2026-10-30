"""Worker batching and failure handling, with the rollout subprocess stubbed.

The GPU path is exercised by gates C and D on a box; what is testable here is the part that
decides WHICH seeds one invocation covers and what happens when one fails -- both of which can
silently corrupt a tally.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from axis_qvla.bench.sim import queue_worker as qw
from axis_qvla.bench.sim.queue_worker import WorkerConfig, group_batch, run_worker
from axis_qvla.bench.sim.workqueue import Unit, WorkQueue

UNITS = (
    [Unit(806, 806_000 + i) for i in range(12)]
    + [Unit(1889, 1_889_000 + i) for i in range(12)]
)


def _setup(tmp_path: Path, batch: int = 5) -> tuple[WorkQueue, WorkerConfig]:
    q = WorkQueue(tmp_path / "q")
    q.initialize(UNITS)
    cfg = WorkerConfig(
        queue_root=q.root,
        worker_id="w0",
        port=9000,
        variant="testarm",
        checkpoint_dir="/nowhere/13579",
        batch=batch,
    )
    return q, cfg


def test_batch_groups_only_one_task(tmp_path):
    """CATCHES a batch spanning two tasks -- `--seeds` is per task, so foreign seeds would be
    refused by resolve_seeds or, worse, silently attributed to the wrong task."""
    q, cfg = _setup(tmp_path, batch=20)
    first = q.claim(cfg.worker_id)
    batch = group_batch(q, cfg, first)
    assert {u.task_id for u in batch} == {first.task_id}


def test_batch_respects_its_size_limit(tmp_path):
    q, cfg = _setup(tmp_path, batch=4)
    first = q.claim(cfg.worker_id)
    assert len(group_batch(q, cfg, first)) == 4


def test_a_foreign_unit_peeked_at_is_released_not_held(tmp_path):
    """CATCHES a worker parking another task's unit while it works: that unit would sit idle for
    the whole staleness window, reintroducing exactly the straggler idling the queue removes."""
    q, cfg = _setup(tmp_path, batch=50)
    first = q.claim(cfg.worker_id)
    batch = group_batch(q, cfg, first)
    assert len(batch) == 12  # one task's worth
    other = q.claim("someone_else")
    assert other is not None and other.task_id != first.task_id


def test_successful_batches_drain_the_queue_exactly_once(tmp_path, monkeypatch):
    q, cfg = _setup(tmp_path, batch=5)
    seen: list[tuple[int, tuple[int, ...]]] = []

    def fake_run(cmd, **kw):
        tid = int(cmd[cmd.index("--task-id") + 1])
        seeds = tuple(int(s) for s in cmd[cmd.index("--seeds") + 1].split(","))
        seen.append((tid, seeds))
        return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(qw.subprocess, "run", fake_run)
    summary = run_worker(cfg, report=lambda *_: None)
    assert summary["seeds_done"] == len(UNITS)
    flat = [s for _, seeds in seen for s in seeds]
    assert sorted(flat) == sorted(u.seed for u in UNITS)
    assert len(flat) == len(set(flat)), "a seed was evaluated twice"


def test_a_failed_batch_is_released_not_completed(tmp_path, monkeypatch):
    """CATCHES the worst silent failure: marking a crashed batch done leaves missing seed files
    that scoring reads as a smaller denominator rather than as an error."""
    q, cfg = _setup(tmp_path, batch=5)
    calls = {"n": 0}

    def fake_run(cmd, **kw):
        calls["n"] += 1
        ok = calls["n"] != 1  # first batch fails, rest succeed
        return type("P", (), {"returncode": 0 if ok else 1, "stdout": "", "stderr": "boom"})()

    monkeypatch.setattr(qw.subprocess, "run", fake_run)
    summary = run_worker(cfg, report=lambda *_: None)
    # The released units come back and are retried, so everything still completes.
    assert summary["seeds_done"] == len(UNITS)
    assert summary["seeds_failed"] == 5


def test_repeated_failures_stop_the_worker_instead_of_spinning(tmp_path, monkeypatch):
    """A dead server must surface as an error, not as an infinite retry loop that burns the
    staleness window and leaves the queue looking merely slow."""
    q, cfg = _setup(tmp_path, batch=5)
    monkeypatch.setattr(
        qw.subprocess,
        "run",
        lambda cmd, **kw: type("P", (), {"returncode": 1, "stdout": "", "stderr": "dead"})(),
    )
    with pytest.raises(RuntimeError, match="consecutive batch failures"):
        run_worker(cfg, report=lambda *_: None)


def test_command_carries_the_explicit_seed_subset_and_dual_sim(tmp_path):
    _, cfg = _setup(tmp_path)
    cmd = qw._command(cfg, 806, [806_000, 806_003])
    assert "--seeds" in cmd and cmd[cmd.index("--seeds") + 1] == "806000,806003"
    assert "--dual-sim" in cmd
    assert cmd[cmd.index("--port") + 1] == "9000"
    assert cmd[cmd.index("--checkpoint-dir") + 1] == "/nowhere/13579"


def test_command_targets_the_one_rollout_entry_point(tmp_path):
    """A worker runs every batch through the one rollout entry point, `axis_qvla.bench.sim.cli`."""
    _, cfg = _setup(tmp_path)
    cmd = qw._command(cfg, 806, [806_000])
    assert cmd[:3] == [cfg.python, "-m", "axis_qvla.bench.sim.cli"]


def test_command_is_parseable_by_the_rollout_entry_point(tmp_path):
    """The worker's argv and the CLI's parser are two files; this is the only place they
    meet without a GPU. Every flag the worker emits must be one the CLI declares."""
    from axis_qvla.bench.sim.cli import build_parser

    _, cfg = _setup(tmp_path)
    cfg = WorkerConfig(**{**cfg.__dict__, "work_dir": tmp_path / "w", "log_root": tmp_path / "logs",
                          "layout_manifest": tmp_path / "m.json", "benchmark": "franka_heldout24",
                          "extra_args": ("--task-data-root", "/tasks", "--cfg-conditioning",
                                         "--guidance-scale", "0")})
    cmd = qw._command(cfg, 806, [806_000, 806_001])
    ns = build_parser().parse_args(cmd[3:])
    assert ns.task_id == 806 and ns.seeds == "806000,806001" and ns.n_seeds is None
    assert ns.dual_sim is True and ns.benchmark == "franka_heldout24"
    assert ns.cfg_conditioning is True and ns.guidance_scale == 0.0
    assert str(ns.work_dir).endswith("/w0/task_806")


def test_benchmark_is_forwarded_only_when_set(tmp_path):
    """Unset lets the CLI's own default apply, so the worker never restates it wrongly."""
    _, cfg = _setup(tmp_path)
    assert "--benchmark" not in qw._command(cfg, 806, [806_000])
    cmd = qw._command(WorkerConfig(**{**cfg.__dict__, "benchmark": "pilot7"}), 806, [806_000])
    assert cmd[cmd.index("--benchmark") + 1] == "pilot7"


def test_two_workers_split_the_queue_with_no_seed_evaluated_twice(tmp_path, monkeypatch):
    """The property gate D checks end to end, asserted here on the scheduling layer alone."""
    q, cfg = _setup(tmp_path, batch=3)
    seen: list[int] = []

    def fake_run(cmd, **kw):
        seen.extend(int(s) for s in cmd[cmd.index("--seeds") + 1].split(","))
        return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(qw.subprocess, "run", fake_run)
    run_worker(cfg, report=lambda *_: None)
    cfg_b = WorkerConfig(**{**cfg.__dict__, "worker_id": "w1"})
    run_worker(cfg_b, report=lambda *_: None)  # queue already drained
    assert sorted(seen) == sorted(u.seed for u in UNITS)
    assert len(seen) == len(set(seen))


def test_work_dir_is_per_worker_and_per_task(tmp_path):
    """CATCHES the scene-materialisation race: 32 workers sharing one work dir produce
    half-written derived MJCF includes ('ParseError: no element found'), and because a failed
    rollout still writes a seed file with reason='error:...', those land in the tally as ordinary
    failures. Measured at 2.1% of rollouts before this fix."""
    _, cfg = _setup(tmp_path)
    cfg = WorkerConfig(**{**cfg.__dict__, "work_dir": tmp_path / "work"})
    a = qw._command(cfg, 806, [806000])
    b = qw._command(WorkerConfig(**{**cfg.__dict__, "worker_id": "w9"}), 806, [806000])
    wa = a[a.index("--work-dir") + 1]
    wb = b[b.index("--work-dir") + 1]
    assert wa != wb, "two workers must not share a task's work dir"
    assert wa.endswith("/w0/task_806") and wb.endswith("/w9/task_806")
    c = qw._command(cfg, 1889, [1889000])
    assert c[c.index("--work-dir") + 1].endswith("/w0/task_1889")


def test_rollout_args_reach_the_rollout_command_in_order(tmp_path, monkeypatch):
    """CATCHES the pass-1 campaign failure: the driver handed the worker
    `--layout-manifest M --cfg-conditioning --guidance-scale 0` as separate argv tokens and
    argparse rejected the first option it did not declare, so every CONDITIONED arm exited 2
    in seconds while every bare arm ran. The flags must arrive as one string and be appended
    to the rollout command verbatim, in order (a reordered `--guidance-scale 0` is a different
    protocol)."""
    q = WorkQueue(tmp_path / "q")
    q.initialize([])  # empty queue: the worker drains immediately, we only inspect the config
    seen = {}

    def fake_run(cfg, *, report=print):
        seen["cfg"] = cfg
        return {"worker": cfg.worker_id, "seeds_done": 0, "seeds_failed": 0, "batches": 0,
                "elapsed_s": 0.0, "rollout_args": list(cfg.extra_args)}

    monkeypatch.setattr(qw, "run_worker", fake_run)
    rc = qw.main([
        "--queue-root", str(q.root), "--worker-id", "w0", "--port", "9000",
        "--variant", "arm", "--checkpoint-dir", "/nowhere/13579",
        "--rollout-args", "--layout-manifest /m.json --cfg-conditioning --guidance-scale 0",
    ])
    assert rc == 0
    cmd = qw._command(seen["cfg"], 806, [806_000])
    tail = cmd[cmd.index("--dual-sim") + 1:]
    assert tail == ["--layout-manifest", "/m.json", "--cfg-conditioning", "--guidance-scale", "0"]


def test_summary_records_the_serving_protocol(tmp_path, monkeypatch):
    """A bare and a conditioned read of one checkpoint must not have identical summaries."""
    q, cfg = _setup(tmp_path, batch=12)
    cfg = WorkerConfig(**{**cfg.__dict__, "extra_args": ("--cfg-conditioning", "--guidance-scale", "0")})
    monkeypatch.setattr(qw.subprocess, "run", lambda *a, **k: type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    summary = run_worker(cfg, report=lambda *_: None)
    assert summary["rollout_args"] == ["--cfg-conditioning", "--guidance-scale", "0"]


def test_worker_waits_for_foreign_in_flight_claims_instead_of_exiting(tmp_path, monkeypatch):
    """CATCHES a short drain: after a restart the dead workers' claims are still
    fresh, claim() returns None although units remain, and a worker that exits makes the driver
    declare the queue drained before it actually is. The worker must wait until those claims
    complete or go stale."""
    q, cfg = _setup(tmp_path, batch=5)
    cfg = WorkerConfig(**{**cfg.__dict__, "stale_after_s": 2.0, "idle_poll_s": 0.05})
    # a dead worker holds every unit but one
    held = [q.claim("dead") for _ in range(len(UNITS) - 1)]
    assert all(held)
    waits: list[str] = []

    def fake_run(cmd, **kw):
        return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(qw.subprocess, "run", fake_run)
    summary = run_worker(cfg, report=lambda m: waits.append(str(m)))
    # it did NOT stop at the one free unit: it waited, the claims went stale, it finished them all
    assert summary["seeds_done"] == len(UNITS)
    assert any("in flight elsewhere; waiting" in m for m in waits)


def test_a_hung_rollout_is_timed_out_and_released_not_waited_on_forever(tmp_path, monkeypatch):
    """CATCHES a deadlock where a stuck rollout idles a worker box indefinitely.

    A rollout that stops making progress (futex wait, many threads, 0% GPU) but never exits
    leaves the worker sitting in subprocess.run forever; the driver waits on worker PIDs, so the
    arm never finishes even after the rest of the queue has drained, idling every campaign queued
    behind it. Without a timeout this test hangs rather than fails.
    """
    q, cfg = _setup(tmp_path, batch=5)
    calls = {"n": 0}

    def fake_run(cmd, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            assert kw.get("timeout"), "no timeout passed: a hung rollout would block forever"
            raise qw.subprocess.TimeoutExpired(cmd, kw["timeout"])
        return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(qw.subprocess, "run", fake_run)
    summary = run_worker(cfg, report=lambda *_: None)
    # The timed-out units are released, retried by this same worker, and still complete.
    assert summary["seeds_done"] == len(UNITS)
    assert summary["seeds_failed"] == 5
    assert summary["batch_timeouts"] == 1


def test_the_timeout_budget_scales_with_the_batch_and_is_never_a_rollout_deadline(tmp_path, monkeypatch):
    """The ceiling must clear the slowest real rollout by a wide margin, or it becomes a
    scheduling knob that fails healthy SmolVLA batches (4-7 min/rollout) instead of a backstop."""
    q, cfg = _setup(tmp_path, batch=5)
    budgets: list[float] = []

    def fake_run(cmd, **kw):
        budgets.append(kw["timeout"])
        return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(qw.subprocess, "run", fake_run)
    run_worker(cfg, report=lambda *_: None)
    assert budgets, "no batch ran"
    per_seed = [b / 5 for b in budgets if b > cfg.timeout_overhead_s]
    assert min(per_seed) >= 420, "budget leaves no room for a 7-minute SmolVLA rollout"


def test_a_refused_asset_aborts_the_worker_and_the_queue(tmp_path, monkeypatch, capsys):
    """Exit 4 from the rollout CLI (a scene asset refused by assets.fetch=refuse or ASSETS.lock)
    is not a flaky batch: retrying cannot succeed and every other worker would hit it too. The
    batch is released (never completed), the queue is marked aborted with the reason, the worker
    stops at once, and a second worker on the same queue starts no further rollouts."""
    from axis_qvla.bench.sim.assets import EXIT_ASSET_REFUSED

    q, cfg = _setup(tmp_path, batch=5)
    calls: list[int] = []

    def fake_run(cmd, **kw):
        calls.append(int(cmd[cmd.index("--task-id") + 1]))
        return type("P", (), {"returncode": EXIT_ASSET_REFUSED, "stdout": "",
                              "stderr": "ABORT: scene asset refused: asset 'tabletopgen/a.obj' missing"})()

    monkeypatch.setattr(qw.subprocess, "run", fake_run)
    summary = run_worker(cfg, report=lambda *_: None)
    assert calls == [806] and summary["seeds_done"] == 0 and summary["seeds_failed"] == 5
    assert "tabletopgen/a.obj" in summary["aborted"] and "tabletopgen/a.obj" in q.aborted()
    assert q.progress()[0] == 0 and q.progress()[1] == 0, "units must be released, not completed"

    cfg_b = WorkerConfig(**{**cfg.__dict__, "worker_id": "w1"})
    assert run_worker(cfg_b, report=lambda *_: None)["aborted"] and calls == [806]

    monkeypatch.setattr(qw, "run_worker", lambda c, **k: {"aborted": "asset refused", "worker": c.worker_id})
    rc = qw.main(["--queue-root", str(q.root), "--worker-id", "w2", "--port", "9000",
                  "--variant", "arm", "--checkpoint-dir", "/nowhere/13579"])
    assert rc == EXIT_ASSET_REFUSED and "asset refused" in capsys.readouterr().err


# --- what the driver hands a worker reaches the rollout CLI --------------------

def _cfg_from_main(tmp_path, monkeypatch, *extra: str) -> WorkerConfig:
    """The WorkerConfig queue_worker.main builds from an argv, with the drain stubbed out."""
    q = WorkQueue(tmp_path / "q")
    q.initialize([])
    seen = {}

    def fake_run(cfg, *, report=print):
        seen["cfg"] = cfg
        return {"worker": cfg.worker_id, "seeds_done": 0, "seeds_failed": 0, "batches": 0,
                "elapsed_s": 0.0, "rollout_args": list(cfg.extra_args)}

    monkeypatch.setattr(qw, "run_worker", fake_run)
    assert qw.main(["--queue-root", str(q.root), "--worker-id", "w0", "--port", "9000",
                    "--variant", "arm", "--checkpoint-dir", "/nowhere/13579", *extra]) == 0
    return seen["cfg"]


# The roots the rollout CLI requires and these tests do not hand the worker, ahead of its argv so a
# flag the worker passes wins.
ROOTS = ["--log-root", "/logs", "--work-dir", "/work", "--task-data-root", "/tasks"]


def _cli_args(cfg: WorkerConfig):
    from axis_qvla.bench.sim.cli import build_parser

    return build_parser().parse_args([*ROOTS, *qw._command(cfg, 806, [806_000])[3:]])


def test_the_drivers_log_root_reaches_the_rollout_cli(tmp_path, monkeypatch):
    """The worker never passed a log root, so the CLI wrote under the checkout on its
    PYTHONPATH (+ $SLB_PILOT_SUFFIX) while the driver tallied the host's root."""
    cfg = _cfg_from_main(tmp_path, monkeypatch, "--log-root", str(tmp_path / "logs"))
    assert _cli_args(cfg).log_root == tmp_path / "logs"


def test_without_a_log_root_the_rollout_cli_refuses_the_command(tmp_path, monkeypatch, capsys):
    """The rollout CLI has no default log root: a worker started without --log-root hands it none,
    and the rollout refuses the command by the flag's name."""
    from axis_qvla.bench.sim.cli import build_parser

    cfg = _cfg_from_main(tmp_path, monkeypatch, "--work-dir", str(tmp_path / "w"),
                         "--rollout-args", "--task-data-root /tasks")
    cmd = qw._command(cfg, 806, [806_000])
    assert "--log-root" not in cmd
    capsys.readouterr()
    with pytest.raises(SystemExit):
        build_parser().parse_args(cmd[3:])
    assert "required: --log-root" in capsys.readouterr().err


def test_n_seeds_reaches_the_rollout_cli_only_when_given(tmp_path, monkeypatch):
    """A worker handed --n-seeds passes it on; one started without it passes none, so the rollout
    takes the benchmark's n_seeds."""
    assert _cli_args(_cfg_from_main(tmp_path / "a", monkeypatch, "--n-seeds", "60")).n_seeds == 60
    cfg = _cfg_from_main(tmp_path / "b", monkeypatch)
    assert "--n-seeds" not in qw._command(cfg, 806, [806_000]) and _cli_args(cfg).n_seeds is None


@pytest.mark.parametrize("rollout_args", ["--log-root {other}", "--log-root={other}", "--log-r {other}"])
def test_a_second_log_root_in_the_rollout_args_is_refused(tmp_path, monkeypatch, rollout_args):
    """The rollout args are appended after the worker's own --log-root, so argparse would take
    theirs and split writer from reader again."""
    other = tmp_path / "other"
    with pytest.raises(SystemExit, match="log-root"):
        _cfg_from_main(tmp_path, monkeypatch, "--log-root", str(tmp_path / "logs"),
                       f"--rollout-args={rollout_args.format(other=other)}")
    cfg = _cfg_from_main(tmp_path / "same", monkeypatch, "--log-root", str(tmp_path / "logs"),
                         "--rollout-args", f"--log-root {tmp_path / 'logs'}")
    assert _cli_args(cfg).log_root == tmp_path / "logs"


@pytest.mark.parametrize("handed,rollout_args,flag", [
    (("--benchmark", "franka_heldout24"), "--benchmark pilot", "--benchmark"),
    (("--benchmark", "franka_heldout24"), "--bench=pilot", "--benchmark"),
    (("--n-seeds", "50"), "--n-seeds 60", "--n-seeds"),
    (("--n-seeds", "50"), "--n-seeds=7", "--n-seeds"),
])
def test_a_second_benchmark_or_n_seeds_in_the_rollout_args_is_refused(tmp_path, monkeypatch, handed,
                                                                      rollout_args, flag):
    """The worker hands the rollout --benchmark and --n-seeds before the rollout args, so argparse
    would take a second copy there: the rollout would resolve prompts and seeds, or accept a seed
    count, the driver does not tally. The same value restated is one fact, not two."""
    with pytest.raises(SystemExit, match=flag):
        _cfg_from_main(tmp_path, monkeypatch, *handed, f"--rollout-args={rollout_args}")
    cfg = _cfg_from_main(tmp_path / "same", monkeypatch, *handed, "--rollout-args", " ".join(handed))
    assert str(getattr(_cli_args(cfg), handed[0][2:].replace("-", "_"))) == handed[1]


def test_a_flag_the_worker_was_not_handed_is_the_rollout_args_to_state(tmp_path, monkeypatch):
    """A worker started without --benchmark or --n-seeds passes neither, and its rollout args may
    choose (nothing was handed on, so there is nothing to split)."""
    cfg = _cfg_from_main(tmp_path, monkeypatch, "--rollout-args", "--benchmark pilot --n-seeds 7")
    ns = _cli_args(cfg)
    assert (ns.benchmark, ns.n_seeds) == ("pilot", 7)
