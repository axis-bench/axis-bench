"""A crash-safe, file-backed work queue whose unit is one (task, seed) rollout.

WHY THIS EXISTS. Scheduling 24 units -- one per task -- across 8 GPUs in waves makes every wave wait
for its slowest task, and task durations differ several-fold. Making the unit of work a single
rollout turns 24 coarse units into 1,200 fine ones, so the load balances and a worker never waits
for a peer.

Correctness rests on one property established elsewhere (`axis_qvla.bench.sim.inference_noise`): a
rollout's inputs other than its rendered frames -- its layout and every request's sampler noise --
are a function of (task_id, seed) alone, so *which* worker claims a unit, in *what* order, and
whether it is retried after a crash add no variance of their own. The frames are not bitwise
reproducible (the renderer's multisample colour resolve differs by +-1 LSB between renders), so a
retried unit agrees with its first attempt statistically, not byte for byte
(docs/reference/reporting-rules.md). Claims are advisory-locked appends to a journal, so a killed
worker's units return to the pool and a restarted run resumes instead of recomputing.

Tier: online (eval). Stdlib only -- no numpy, no jax, no mujoco -- so it is importable anywhere and
testable without a GPU.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

__all__ = ["Unit", "WorkQueue"]


@dataclass(frozen=True)
class Unit:
    task_id: int
    seed: int

    @property
    def key(self) -> str:
        return f"{self.task_id}:{self.seed}"


class WorkQueue:
    """Units live in a plain text file; claims and completions live in a journal beside it.

    Both files are append-only and read whole. At 1,200 units that is trivially cheap, and it buys
    a property a database would not: after any crash, the queue's state is reconstructible by a
    human with `cat`.
    """

    def __init__(self, root: Path | str, *, stale_after_s: float = 1800.0) -> None:
        self.root = Path(root)
        self.units_path = self.root / "units.txt"
        self.journal_path = self.root / "journal.jsonl"
        self.lock_path = self.root / "queue.lock"
        self.abort_path = self.root / "ABORTED"
        # A claim older than this is treated as abandoned (worker died). Must exceed the time a
        # worker can hold a claim: a worker claims a BATCH and runs it serially, so that is
        # batch x the slowest rollout, not one rollout. pi0.5 takes ~47 s per 1500-step rollout
        # and a CPU-bound SmolVLA server 4-7 min, so size batch x rollout time under this window,
        # or raise it via the driver.
        self.stale_after_s = float(stale_after_s)

    # -- construction ----------------------------------------------------------
    def initialize(self, units: list[Unit]) -> None:
        """Write the unit list in a FIXED canonical order. Idempotent: refuses to clobber."""
        self.root.mkdir(parents=True, exist_ok=True)
        if self.units_path.exists():
            existing = self.read_units()
            if existing != units:
                raise RuntimeError(
                    f"{self.units_path} already exists with {len(existing)} different units; "
                    "refusing to overwrite a queue that may be mid-run"
                )
            return
        tmp = self.units_path.with_suffix(".tmp")
        tmp.write_text("".join(f"{u.task_id} {u.seed}\n" for u in units), encoding="utf-8")
        os.replace(tmp, self.units_path)  # atomic: a reader never sees a partial list
        self.journal_path.touch()

    def read_units(self) -> list[Unit]:
        out: list[Unit] = []
        for line in self.units_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            t, s = line.split()
            out.append(Unit(int(t), int(s)))
        return out

    # -- journal ---------------------------------------------------------------
    def _read_journal(self) -> tuple[dict[str, float], set[str]]:
        """Return (claim_key -> claim_time, done_keys). Tolerates a torn final line."""
        claims: dict[str, float] = {}
        done: set[str] = set()
        if not self.journal_path.exists():
            return claims, done
        for line in self.journal_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # A crash mid-append can leave a partial line. Ignoring it re-queues at most one
                # unit, which is safe precisely because re-running a unit is bit-identical.
                continue
            if rec.get("event") == "claim":
                claims[rec["key"]] = float(rec["t"])
            elif rec.get("event") == "done":
                done.add(rec["key"])
            elif rec.get("event") == "release":
                # Drop the claim outright rather than back-dating it. Back-dating "works" only
                # because wall-clock is a large number; it breaks under any injected clock.
                claims.pop(rec["key"], None)
        return claims, done

    def _append(self, rec: dict) -> None:
        with self.journal_path.open("a+", encoding="utf-8") as fh:
            # A crash mid-append leaves a line with no terminating newline. Appending straight on
            # would CONCATENATE onto it and corrupt the next record too -- one torn write silently
            # eating a second, valid claim, which reissues a unit that is already in flight.
            fh.seek(0, os.SEEK_END)
            if fh.tell():
                fh.seek(fh.tell() - 1)
                if fh.read(1) != "\n":
                    fh.write("\n")
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    # -- claiming --------------------------------------------------------------
    def claim(self, worker: str, *, prefer_task: int | None = None, now: float | None = None) -> Unit | None:
        """Atomically take the next unclaimed (or stale-claimed) unit. None when the queue drains.

        The whole read-decide-append is inside one exclusive flock, so two workers cannot take the
        same unit. flock is released when the fd closes, including if the process is killed.
        """
        t = time.time() if now is None else now
        self.root.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("w") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                claims, done = self._read_journal()

                def available(unit: Unit) -> bool:
                    if unit.key in done:
                        return False
                    at = claims.get(unit.key)
                    return at is None or (t - at) >= self.stale_after_s

                units = self.read_units()
                # Task locality: a worker that already has a task's scene compiled should keep
                # draining that task before paying setup for another. This is a preference, never
                # a reservation -- when the preferred task is exhausted the worker takes the next
                # available unit, so locality cannot reintroduce the straggler barrier the
                # seed-level queue exists to remove.
                ordered = units
                if prefer_task is not None:
                    ordered = [u for u in units if u.task_id == prefer_task] + [
                        u for u in units if u.task_id != prefer_task
                    ]
                for unit in ordered:
                    if not available(unit):
                        continue
                    self._append({"event": "claim", "key": unit.key, "worker": worker, "t": t})
                    return unit
                return None
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def complete(self, unit: Unit, worker: str, *, ok: bool = True) -> None:
        self._append(
            {"event": "done", "key": unit.key, "worker": worker, "ok": bool(ok), "t": time.time()}
        )

    def release(self, unit: Unit, worker: str) -> None:
        """Hand a claimed-but-unfinished unit back immediately, without waiting for staleness.

        Used when a driver knows a worker's server died, so its units need not sit out the full
        staleness window. Safe because re-running a unit is bit-identical (see inference_noise).
        """
        self._append({"event": "release", "key": unit.key, "worker": worker, "t": time.time()})

    def started(self) -> bool:
        """Whether any unit of this queue was ever claimed or finished: a run that wrote seed files
        claimed their units first. A queue that does not exist yet has started nothing."""
        try:
            lines = self.journal_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return False
        for line in lines:
            try:
                if json.loads(line).get("event") in ("claim", "done", "release"):
                    return True
            except (ValueError, AttributeError):
                continue      # a torn final line, as _read_journal tolerates
        return False

    def progress(self) -> tuple[int, int, int]:
        """(done, in_flight, total) -- for a driver's heartbeat log."""
        claims, done = self._read_journal()
        total = len(self.read_units())
        t = time.time()
        in_flight = sum(
            1
            for k, at in claims.items()
            if k not in done and (t - at) < self.stale_after_s
        )
        return len(done), in_flight, total

    # -- abort -----------------------------------------------------------------
    def abort(self, reason: str) -> None:
        """Stop the whole queue: a failure no retry can fix (a refused scene asset). The FIRST
        reason is kept; every worker checks `aborted()` before it claims again."""
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.abort_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            return
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(reason.rstrip("\n") + "\n")

    def aborted(self) -> str | None:
        try:
            return self.abort_path.read_text(encoding="utf-8").rstrip("\n")
        except FileNotFoundError:
            return None

    def clear_abort(self) -> str | None:
        """Remove a previous run's abort marker (the driver, before starting workers again)."""
        reason = self.aborted()
        if reason is not None:
            self.abort_path.unlink()
        return reason

    def duplicate_completions(self) -> int:
        """Completion events beyond the first per unit: work the staleness window re-issued.

        Zero is the healthy value. A large number means claims outlived `stale_after_s` (a batch
        of slow rollouts run serially) and other workers re-ran units still in progress. The
        aggregate stays correct -- each unit is counted once -- but throughput
        is lost, and this count is where the driver reports it.
        """
        events = 0
        if not self.journal_path.exists():
            return 0
        for line in self.journal_path.read_text(encoding="utf-8").splitlines():
            if '"event":"done"' in line:
                events += 1
        _, done = self._read_journal()
        return events - len(done)

    def iter_claims(self, worker: str, *, sticky: bool = False) -> Iterator[Unit]:
        """Convenience loop: claim until the queue drains. The worker marks completion itself.

        With `sticky`, each claim prefers the task the previous unit belonged to, so a worker
        reuses its compiled scene instead of rebuilding one per rollout.
        """
        last_task: int | None = None
        while True:
            unit = self.claim(worker, prefer_task=last_task if sticky else None)
            if unit is None:
                return
            last_task = unit.task_id
            yield unit
