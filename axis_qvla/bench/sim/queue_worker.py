"""A long-lived worker that drains the (task, seed) queue against one persistent policy server.

DESIGN CHOICE THAT MATTERS. The worker does NOT reimplement the rollout. It claims units and
shells out to `python -m axis_qvla.bench.sim.cli --seeds ...`, the one rollout entry point, so the
scene build, checker, dual-sim stepping and result format (the run directory names,
seed_<n>.json, meta.json) are byte-for-byte the validated path. What changes is only WHICH seeds
one invocation covers and WHICH server it talks to.

The rollout entry point reads the server's conditioning off its handshake;
`--rollout-args '--cfg-conditioning --guidance-scale 0'`, when passed, is checked against the
server.

The campaign driver hands every worker the log root it will tally, the benchmark declaration,
its n_seeds and the seed set (test or dev), and the worker passes them to every rollout.

The unit is one (task, seed), but units are executed in BATCHES of consecutive same-task claims:
a subprocess pays ~15 s of robosuite/mujoco import, so running one seed per process would spend
more on imports than on rollouts. Sticky claims plus batching amortise both the import and the
per-task scene build, while the queue still hands out fine-grained units so no worker waits on a
peer at a wave barrier.

Correctness rests on `axis_qvla.bench.sim.inference_noise`: the sampler noise of every request is a
function of (task, seed, call index) alone, and the layout of (task, seed), so which worker claims
a unit, which server answers it, how many requests preceded it, or whether it is retried after a
crash changes neither, so the schedule does not change results. The rollout itself is not a pure
function of (task, seed): the renderer's multisample colour resolve differs by +-1 LSB between
renders, and the closed loop can amplify that into a different single-rollout outcome. A re-run
unit agrees with the first run statistically, in its success rate, not byte for byte
(docs/reference/reporting-rules.md).

Tier: online (eval). Stdlib only -- the heavy imports live in the subprocess.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from axis_qvla.bench.sim.assets import EXIT_ASSET_REFUSED
from axis_qvla.bench.sim.workqueue import Unit, WorkQueue
from axis_qvla.bench.spec import SEED_SETS

__all__ = ["HANDED_ON", "WorkerConfig", "group_batch", "named_rollout_flags", "refuse_second_copy",
           "run_worker"]


@dataclass(frozen=True)
class WorkerConfig:
    """Everything an invocation needs that is not the seed list."""

    queue_root: Path
    worker_id: str
    port: int
    variant: str
    checkpoint_dir: str
    host: str = "127.0.0.1"
    batch: int = 10
    stale_after_s: float = 1800.0
    idle_poll_s: float = 30.0
    extra_args: tuple[str, ...] = ()
    python: str = sys.executable
    work_dir: Path | None = None
    layout_manifest: Path | None = None
    benchmark: str | None = None
    # Where the rollout writes (the rollout CLI requires it). The driver always sets it.
    log_root: Path | None = None
    # First N declared seeds the rollout accepts; None leaves the rollout's own (the benchmark's
    # n_seeds). The driver always passes the declaration's.
    n_seeds: int | None = None
    # Which declared seeds the units are; handed to the rollout only when it is not the default.
    seed_set: str = "test"
    max_consecutive_failures: int = 3
    env: dict[str, str] = field(default_factory=dict)
    # Wall-clock ceiling per SEED in a batch, plus a fixed allowance for the subprocess's ~15 s of
    # robosuite/mujoco import and its first scene build. A rollout that deadlocks instead of
    # finishing would otherwise block this worker forever, and the driver waits on worker PIDs, so
    # ONE stuck rollout hangs the whole arm and every campaign queued behind it. A rollout stuck
    # in a kernel futex wait can hold over a hundred threads at 0% GPU for up to well over an hour
    # after the rest of the queue has already drained, leaving the driver unable to exit and
    # every campaign queued behind it idle. The default
    # is deliberately far above any real rollout (pi0.5 ~47 s, SmolVLA 4-7 min) -- this is a
    # deadlock backstop, not a scheduling knob.
    timeout_per_seed_s: float = 900.0
    timeout_overhead_s: float = 300.0


ROLLOUT_MODULE = "axis_qvla.bench.sim.cli"


# The rollout-CLI flags the driver hands every worker and the worker hands every rollout, by their
# argparse dest. The free-form --rollout-args FOLLOW them in the rollout's argv, so argparse would
# take a second copy there: the rollouts would write, resolve prompts and seeds, or accept a seed
# count that the driver's tally does not assume.
HANDED_ON = {"log_root": "--log-root", "benchmark": "--benchmark", "n_seeds": "--n-seeds",
             "seed_set": "--seed-set"}
_UNSET = object()


def named_rollout_flags(args: Sequence[str]) -> dict[str, object]:
    """The HANDED_ON flags `args` name, by dest, as the rollout CLI's own parser reads them.

    Read by that parser, so `--log-root=X` and an abbreviation such as `--log-r X` count exactly as
    they would in the rollout, and a value is typed as it would be there. Arguments the CLI does not
    declare are ignored; a flag `args` do not name is absent, not its default.
    """
    from axis_qvla.bench.sim.cli import build_parser  # noqa: PLC0415 - stdlib-only, kept off module scope

    ap = build_parser()
    ap.set_defaults(**dict.fromkeys(HANDED_ON, _UNSET))
    for action in ap._actions:   # only what `args` name is read: nothing is required of them
        action.required = False
    ns, _ = ap.parse_known_args(list(args))
    return {dest: getattr(ns, dest) for dest in HANDED_ON if getattr(ns, dest) is not _UNSET}


def refuse_second_copy(args: Sequence[str], where: str, **handed: object) -> None:
    """Refuse `args` that restate a handed-on flag with a different value. One fact, stated once:
    the same value restated is accepted (a log root compared as an absolute path).

    `handed` maps a HANDED_ON dest to the value handed on; None means none is handed on, so
    `args` are the only statement and there is nothing to split.
    """
    for dest, theirs in named_rollout_flags(args).items():
        mine = handed.get(dest)
        if mine is None:
            continue
        same = (Path(theirs).absolute() == Path(mine).absolute() if dest == "log_root"
                else theirs == mine)
        if not same:
            flag = HANDED_ON[dest]
            raise SystemExit(f"{where} name {flag} {theirs}, but {flag} {mine} is handed on to "
                             f"every rollout and is what the tally assumes; pass it once")


def group_batch(queue: WorkQueue, cfg: WorkerConfig, first: Unit) -> list[Unit]:
    """Claim up to `batch` further units of `first`'s task, so one subprocess covers them all.

    Stops at the first unit of a different task and RELEASES it rather than executing it out of
    band: mixing tasks inside one `--seeds` list is impossible (the flag is per task), and holding
    a foreign task's unit while we work would idle it.
    """
    batch = [first]
    while len(batch) < cfg.batch:
        nxt = queue.claim(cfg.worker_id, prefer_task=first.task_id)
        if nxt is None:
            break
        if nxt.task_id != first.task_id:
            queue.release(nxt, cfg.worker_id)
            break
        batch.append(nxt)
    return batch


def _command(cfg: WorkerConfig, task_id: int, seeds: list[int]) -> list[str]:
    cmd = [
        cfg.python,
        "-m",
        ROLLOUT_MODULE,
        "--task-id",
        str(task_id),
        "--variant",
        cfg.variant,
        "--host",
        cfg.host,
        "--port",
        str(cfg.port),
        "--checkpoint-dir",
        cfg.checkpoint_dir,
        "--seeds",
        ",".join(str(s) for s in seeds),
        "--dual-sim",
    ]
    if cfg.n_seeds is not None:
        cmd += ["--n-seeds", str(cfg.n_seeds)]
    if cfg.work_dir is not None:
        # PER-WORKER, PER-TASK work dir: scene materialisation writes derived MJCF includes there,
        # and two workers writing one file concurrently would let a reader see it half-written (a
        # ParseError on an empty XML), recorded as a policy failure.
        cmd += ["--work-dir", str(Path(cfg.work_dir) / cfg.worker_id / f"task_{task_id}")]
    if cfg.layout_manifest is not None:
        cmd += ["--layout-manifest", str(cfg.layout_manifest)]
    if cfg.benchmark is not None:
        cmd += ["--benchmark", cfg.benchmark]
    if cfg.log_root is not None:
        cmd += ["--log-root", str(cfg.log_root)]
    if cfg.seed_set != "test":
        cmd += ["--seed-set", cfg.seed_set]
    cmd += list(cfg.extra_args)
    return cmd


def run_worker(cfg: WorkerConfig, *, report=print) -> dict:
    """Drain the queue. Returns a summary; raises only on a repeated, systematic failure."""
    queue = WorkQueue(cfg.queue_root, stale_after_s=cfg.stale_after_s)
    env = {**os.environ, **cfg.env}
    done = failed = batches = timeouts = 0
    consecutive_failures = 0
    started = time.time()
    last_task: int | None = None
    aborted: str | None = None

    while True:
        aborted = queue.aborted()
        if aborted is not None:
            # Another worker (or this one) hit a refused scene asset: no rollout started from
            # here can be valid, so stop claiming. The driver reports the reason.
            report(f"[{cfg.worker_id}] queue aborted: {aborted}")
            break
        first = queue.claim(cfg.worker_id, prefer_task=last_task)
        if first is None:
            # Nothing claimable is not the same as nothing left. After a restart the previous
            # workers' claims are still fresh, so claim() returns None while units remain; a
            # worker that exits here makes the driver declare the queue drained short. Wait for
            # those claims to complete or go stale instead.
            _, in_flight, _ = queue.progress()
            if in_flight == 0:
                break
            report(f"[{cfg.worker_id}] nothing claimable, {in_flight} in flight elsewhere; waiting")
            time.sleep(cfg.idle_poll_s)
            continue
        batch = group_batch(queue, cfg, first)
        last_task = first.task_id
        seeds = [u.seed for u in batch]
        cmd = _command(cfg, first.task_id, seeds)
        budget = cfg.timeout_overhead_s + cfg.timeout_per_seed_s * len(batch)
        t0 = time.time()
        timed_out = False
        try:
            proc = subprocess.run(
                cmd, env=env, capture_output=True, text=True, timeout=budget
            )
        except subprocess.TimeoutExpired as exc:
            # run() has already killed the child by the time this lands. Treat it exactly like a
            # non-zero exit: release the units so another worker re-runs them (bit-identical), and
            # let max_consecutive_failures stop a worker whose every batch hangs.
            timed_out = True
            proc = subprocess.CompletedProcess(
                cmd, returncode=-1,
                stdout=(exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                stderr=f"TIMEOUT after {budget:.0f}s ({len(batch)} seeds); child killed",
            )
        dt = time.time() - t0
        batches += 1

        if proc.returncode == 0:
            for u in batch:
                queue.complete(u, cfg.worker_id, ok=True)
            done += len(batch)
            consecutive_failures = 0
            report(
                f"[{cfg.worker_id}] task {first.task_id} x{len(batch)} seeds ok "
                f"({dt:.0f}s, {dt / max(len(batch), 1):.0f}s/seed)"
            )
        else:
            # Hand the units back rather than marking them done. A re-run is bit-identical, so a
            # retry cannot corrupt the result; silently completing a failed batch would leave a
            # hole that scoring reads as a missing seed.
            for u in batch:
                queue.release(u, cfg.worker_id)
            failed += len(batch)
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
            if proc.returncode == EXIT_ASSET_REFUSED:
                # A scene asset missing from --asset-dir or off ASSETS.lock.json: a retry cannot
                # succeed and every worker would hit it, so the whole queue stops.
                why = next((ln for ln in reversed(tail) if "ABORT" in ln), tail[-1] if tail else "")
                queue.abort(f"{cfg.worker_id}: task {first.task_id}: {why}")
                report(f"[{cfg.worker_id}] task {first.task_id} ASSET REFUSED; queue aborted. {why}")
                continue
            timeouts += 1 if timed_out else 0
            consecutive_failures += 1
            report(
                f"[{cfg.worker_id}] task {first.task_id} x{len(batch)} "
                f"{'TIMED OUT' if timed_out else 'FAILED'} rc={proc.returncode} "
                f"({dt:.0f}s); released. " + " | ".join(tail)
            )
            if consecutive_failures >= cfg.max_consecutive_failures:
                # Do not spin: a worker that keeps failing is a dead server or a bad config, and
                # looping would burn the queue's staleness window on retries that cannot succeed.
                raise RuntimeError(
                    f"{cfg.worker_id}: {consecutive_failures} consecutive batch failures; "
                    f"last command: {' '.join(cmd)}"
                )

    return {
        "worker": cfg.worker_id,
        "seeds_done": done,
        "seeds_failed": failed,
        "batch_timeouts": timeouts,
        "batches": batches,
        "elapsed_s": round(time.time() - started, 1),
        # The serving protocol lives here (--cfg-conditioning, --guidance-scale). Without it a
        # bare read and a conditioned read of the same checkpoint have identical summaries.
        "rollout_args": list(cfg.extra_args),
        "aborted": aborted,   # the queue's abort reason (a refused scene asset), else None
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queue-root", type=Path, required=True)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--stale-after-s", type=float, default=1800.0,
                    help="a claim older than this is re-issued to another worker; must exceed "
                         "batch x the slowest rollout (SmolVLA: use --batch 3 or raise this)")
    ap.add_argument("--work-dir", type=Path, default=None)
    ap.add_argument("--layout-manifest", type=Path, default=None)
    ap.add_argument("--benchmark", default=None,
                    help=f"benchmark declaration passed to {ROLLOUT_MODULE}; its default "
                         "(franka_heldout24) applies when unset")
    ap.add_argument("--log-root", type=Path, default=None,
                    help=f"passed to {ROLLOUT_MODULE}, which requires it: where the seed files are "
                         "written. The campaign driver passes the root it tallies")
    ap.add_argument("--n-seeds", type=int, default=None,
                    help=f"passed to {ROLLOUT_MODULE}: first N declared seeds it accepts; unset leaves "
                         "the rollout's own (the benchmark's n_seeds). The campaign driver passes the "
                         "declaration's")
    ap.add_argument("--seed-set", choices=SEED_SETS, default="test",
                    help=f"passed to {ROLLOUT_MODULE}: which declared seeds the units are")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--summary", type=Path, default=None)
    ap.add_argument(
        "--rollout-args",
        default="",
        help="ONE string of flags appended verbatim to the rollout command, e.g. "
             "'--cfg-conditioning --guidance-scale 0'; one string, so argparse never reads its "
             "options as the worker's own",
    )
    ap.add_argument("extra", nargs="*", help=f"extra args passed through to {ROLLOUT_MODULE}")
    a = ap.parse_args(argv)

    cfg = WorkerConfig(
        queue_root=a.queue_root,
        worker_id=a.worker_id,
        port=a.port,
        variant=a.variant,
        checkpoint_dir=a.checkpoint_dir,
        host=a.host,
        batch=a.batch,
        stale_after_s=a.stale_after_s,
        work_dir=a.work_dir,
        layout_manifest=a.layout_manifest,
        benchmark=a.benchmark,
        log_root=a.log_root,
        n_seeds=a.n_seeds,
        seed_set=a.seed_set,
        python=a.python,
        extra_args=tuple(a.extra) + tuple(shlex.split(a.rollout_args)),
    )
    refuse_second_copy(cfg.extra_args, "--rollout-args", log_root=cfg.log_root,
                       benchmark=cfg.benchmark, n_seeds=a.n_seeds, seed_set=a.seed_set)
    summary = run_worker(cfg)
    print(json.dumps(summary))
    if a.summary is not None:
        a.summary.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if summary.get("aborted"):
        print(f"ABORT: {summary['aborted']}", file=sys.stderr)
        return EXIT_ASSET_REFUSED
    return 0


if __name__ == "__main__":
    sys.exit(main())
