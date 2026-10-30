"""The two content identities of a read: `eval_fingerprint` and `units_sha256`.

`eval_fingerprint` reproduces `axis_qvla.bench.sim.campaign.fingerprint` exactly: task ids ascending
as ints; within a task every seed file, sorted by name; for each, the basename bytes then the
raw bytes, no separator. Which names are seed files is decided by `axis_qvla.bench.results.seedfiles` here
and by `axis_qvla.bench.sim.record.seed_file_names` there (exactly `seed_<int>.json`); the two must
select the same files. It is computed from a `ServedRun`, so a directory and a tar fingerprint
identically. It differs in one place, on purpose: a task absent from the run raises instead of
contributing nothing, so a missing or unreadable task can never produce a plausible-looking
digest.

`units_sha256` is the work-unit list digest every run manifest records
(`axis_qvla.bench.sim.campaign.run_manifest`, over the units `build_units` makes). A development run's
seeds are tid*1000 + offset + i, so its digest takes the declared dev offset and can never equal
a test run's.

Core tier: stdlib only.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

from axis_qvla.bench.results.seedfiles import ServedRun


class FingerprintError(RuntimeError):
    """The run lacks a task the fingerprint must cover."""


def eval_fingerprint(run: ServedRun, task_ids: Sequence[int]) -> str:
    h = hashlib.sha256()
    for tid in sorted(int(t) for t in task_ids):
        files = run.tasks.get(tid)
        if files is None:
            raise FingerprintError(f"{run.served}: task {tid} is missing")
        for f in sorted(files, key=lambda s: s.name):
            h.update(f.name.encode())
            h.update(f.data)
    return h.hexdigest()


def units_sha256(task_ids: Sequence[int], n_seeds: int, *, offset: int = 0) -> str:
    """`offset` is the seed set's: 0 for the test seeds, the declared dev offset for development
    seeds."""
    h = hashlib.sha256()
    for tid in task_ids:
        for i in range(n_seeds):
            h.update(f"{int(tid)}:{int(tid) * 1000 + int(offset) + i}\n".encode())
    return h.hexdigest()
