"""The work queue must never hand one unit to two workers, and must re-queue a dead worker's unit.

Both properties are load-bearing for the 3.1x driver: the first protects correctness, the second is
what lets a crashed server's units be retried instead of lost.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import pytest

from axis_qvla.bench.sim.workqueue import Unit, WorkQueue

UNITS = [Unit(806, 806_000 + i) for i in range(20)] + [Unit(1889, 1_889_000 + i) for i in range(20)]


def _q(tmp_path: Path, **kw) -> WorkQueue:
    q = WorkQueue(tmp_path / "q", **kw)
    q.initialize(UNITS)
    return q


def test_every_unit_is_handed_out_exactly_once(tmp_path):
    q = _q(tmp_path)
    got = [u.key for u in q.iter_claims("w0")]
    assert len(got) == len(UNITS) == len(set(got))


def test_two_workers_partition_the_queue(tmp_path):
    """CATCHES a lost lock: interleaved claims must never overlap."""
    q = _q(tmp_path)
    a, b = [], []
    while True:
        ua = q.claim("A")
        ub = q.claim("B")
        if ua:
            a.append(ua.key)
            q.complete(ua, "A")
        if ub:
            b.append(ub.key)
            q.complete(ub, "B")
        if not ua and not ub:
            break
    assert set(a) & set(b) == set()
    assert sorted(a + b) == sorted(u.key for u in UNITS)


def _grab(root: str, worker: str, out):
    q = WorkQueue(root)
    got = []
    while True:
        u = q.claim(worker)
        if u is None:
            break
        got.append(u.key)
        q.complete(u, worker)
    out.put(got)


def test_concurrent_processes_never_double_claim(tmp_path):
    """The real concurrency case: 4 OS processes racing on one flock."""
    q = _q(tmp_path)
    out = mp.Queue()
    procs = [mp.Process(target=_grab, args=(str(q.root), f"w{i}", out)) for i in range(4)]
    for p in procs:
        p.start()
    collected = [out.get() for _ in procs]
    for p in procs:
        p.join(timeout=60)
    flat = [k for part in collected for k in part]
    assert len(flat) == len(UNITS), f"expected {len(UNITS)} claims, got {len(flat)}"
    assert len(set(flat)) == len(UNITS), "a unit was claimed twice"


def test_a_dead_workers_claim_is_reclaimed_after_staleness(tmp_path):
    """CATCHES units lost forever when a worker is killed mid-rollout."""
    q = _q(tmp_path, stale_after_s=10.0)
    lost = q.claim("dies", now=1000.0)
    assert lost is not None
    # Still fresh -> nobody else may take it.
    assert q.claim("other", now=1005.0).key != lost.key
    # Past the staleness horizon -> it comes back.
    assert q.claim("other", now=1100.0).key == lost.key


def test_completed_units_are_never_reissued(tmp_path):
    q = _q(tmp_path, stale_after_s=1.0)
    u = q.claim("w", now=0.0)
    q.complete(u, "w")
    later = [x.key for x in q.iter_claims("w2")]
    assert u.key not in later


def test_release_requeues_immediately(tmp_path):
    q = _q(tmp_path, stale_after_s=10_000.0)
    u = q.claim("w", now=0.0)
    q.release(u, "w")
    assert q.claim("w2", now=1.0).key == u.key


def test_resume_after_restart_skips_finished_work(tmp_path):
    """The queue is the resume mechanism: a rerun must not redo completed rollouts."""
    q = _q(tmp_path)
    for _ in range(7):
        u = q.claim("w")
        q.complete(u, "w")
    fresh = WorkQueue(q.root)
    assert len(list(fresh.iter_claims("w2"))) == len(UNITS) - 7


def test_torn_journal_line_does_not_crash_or_lose_the_queue(tmp_path):
    """A kill mid-append leaves a partial line; it must re-queue at most one unit, not explode."""
    q = _q(tmp_path)
    u = q.claim("w")
    q.complete(u, "w")
    with q.journal_path.open("a", encoding="utf-8") as fh:
        fh.write('{"event": "clai')  # torn
    remaining = [x.key for x in q.iter_claims("w2")]
    assert u.key not in remaining
    assert len(remaining) == len(UNITS) - 1


def test_initialize_refuses_to_clobber_a_different_queue(tmp_path):
    """CATCHES a relaunch with a changed task list silently discarding in-flight progress."""
    q = _q(tmp_path)
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        q.initialize(UNITS[:5])
    q.initialize(UNITS)  # identical list is idempotent


def test_progress_counts(tmp_path):
    q = _q(tmp_path)
    u = q.claim("w")
    done, in_flight, total = q.progress()
    assert (done, in_flight, total) == (0, 1, len(UNITS))
    q.complete(u, "w")
    done, in_flight, total = q.progress()
    assert (done, in_flight, total) == (1, 0, len(UNITS))


def test_unit_order_is_canonical_and_stable(tmp_path):
    """The eval fingerprint depends on a fixed unit order; a set/dict-order dependency would make
    the queue file differ run to run."""
    q = _q(tmp_path)
    assert q.read_units() == UNITS
    assert q.units_path.read_text().splitlines()[0] == "806 806000"


def test_journal_records_are_machine_readable(tmp_path):
    q = _q(tmp_path)
    u = q.claim("w")
    q.complete(u, "w")
    recs = [json.loads(l) for l in q.journal_path.read_text().splitlines() if l.strip()]
    assert {r["event"] for r in recs} == {"claim", "done"}
    assert all(r["key"] == u.key for r in recs)


def test_sticky_claims_prefer_the_current_task_but_never_reserve_it(tmp_path):
    """Locality must be a preference only. CATCHES a worker that waits for 'its' task and so
    reintroduces the straggler barrier the seed-level queue exists to remove."""
    q = _q(tmp_path)
    got = [u for u in q.iter_claims("w", sticky=True)]
    # One worker drains everything, and consecutive units cluster by task.
    assert len(got) == len(UNITS)
    switches = sum(1 for a, b in zip(got, got[1:]) if a.task_id != b.task_id)
    assert switches == 1, f"expected one task switch with sticky claims, got {switches}"


def test_sticky_worker_falls_through_when_its_task_is_exhausted(tmp_path):
    """Two workers, one task each; when a worker's task drains it must take the other's leftovers
    rather than exiting early and leaving work undone."""
    q = _q(tmp_path)
    a, b = [], []
    ua = q.claim("A", prefer_task=806)
    ub = q.claim("B", prefer_task=1889)
    a.append(ua); b.append(ub)
    q.complete(ua, "A"); q.complete(ub, "B")
    # A drains everything with a stale preference for a task that will run out.
    for u in q.iter_claims("A", sticky=True):
        a.append(u)
        q.complete(u, "A")
    assert len({u.key for u in a + b}) == len(UNITS)


def test_duplicate_completions_count_reissued_work(tmp_path):
    """CATCHES a throughput loss: a slow batch outlives the staleness window, another
    worker re-runs the unit, both complete it. The aggregate is unaffected, so the ONLY visible
    symptom is completion events beyond one per unit -- which must be reported, not hidden."""
    q = _q(tmp_path, stale_after_s=10.0)
    assert q.duplicate_completions() == 0
    u = q.claim("slow", now=0.0)
    again = q.claim("other", now=100.0)  # stale -> re-issued to a second worker
    assert again.key == u.key
    q.complete(again, "other")
    q.complete(u, "slow")  # the straggler finishes too
    done, _, _ = q.progress()
    assert done == 1  # distinct
    assert q.duplicate_completions() == 1
