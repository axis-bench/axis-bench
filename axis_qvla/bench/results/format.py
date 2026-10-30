"""`qvla.results/v1`: one JSON file per served arm, header plus one row per rollout.

Why: the per-seed files are the evidence but carry no identity; a results file states, in one
place, WHAT was measured (benchmark identity, checkpoint digests, how it was served, the
harness) and the per-rollout outcomes, and it can only be built from a run whose seed files
reproduce the header's `eval_fingerprint`. The ServeCondition is filled from the recorded
server metadata, never inferred from an arm name; it is the signal compiler's type
(axis_qvla.signals.schema), so a bundle's serve list and a results header describe serving alike.
Bytes are canonical JSON (axis_qvla.core.canon), so a file's sha256 is its identity.

Core tier: stdlib only.
"""
from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence

from axis_qvla.core.canon import dumps_canonical
from axis_qvla.bench.results.fingerprint import eval_fingerprint
from axis_qvla.bench.results.seedfiles import SEED_FILE, ServedRun
from axis_qvla.signals.schema import ServeCondition   # re-exported: one type for bundles and results

SCHEMA = "qvla.results/v1"


@dataclasses.dataclass(frozen=True)
class ResultsHeader:
    bench: Mapping[str, object]
    arm: str
    served: str
    training_seed: Mapping[str, int | None] | None   # {"pretrain": int | None, "ft": int}
    checkpoint: Mapping[str, object]
    serve: ServeCondition
    execution: Mapping[str, object]   # the executed constants; off_protocol and deviations when built
    image_geometry: str
    action_space: str
    harness: Mapping[str, object] | None
    eval_lock_sha256: str | None
    gpu: Mapping[str, object] | None
    xla_pin: str
    noise_salt: int | None
    eval_fingerprint: str


@dataclasses.dataclass(frozen=True)
class ResultsRow:
    task: int
    seed: int
    success: bool
    n_steps: int
    sim_time: float
    reason: str


@dataclasses.dataclass(frozen=True)
class Results:
    header: ResultsHeader
    rows: tuple[ResultsRow, ...]


def results_from_run(run: ServedRun, header: ResultsHeader, *, task_ids: Sequence[int]) -> Results:
    fp = eval_fingerprint(run, task_ids)
    if fp != header.eval_fingerprint:
        raise ValueError(f"{run.served}: seed files fingerprint {fp[:16]}, header says {header.eval_fingerprint[:16]}")
    if run.served != header.served:
        raise ValueError(f"run is {run.served}, header is for {header.served}")
    rows = []
    for tid in sorted(int(t) for t in task_ids):
        for f in run.tasks[tid]:
            rec = f.record()
            m = SEED_FILE.fullmatch(f.name)
            if m is None or int(m.group(1)) != int(rec["seed"]):
                raise ValueError(f"{run.served}/task_{tid}/{f.name}: record seed {rec.get('seed')} disagrees with its name")
            rows.append(ResultsRow(tid, int(rec["seed"]), rec["success"] is True, int(rec["n_steps"]),
                                   float(rec["sim_time"]), str(rec["reason"])))
    return Results(header, tuple(sorted(rows, key=lambda r: (r.task, r.seed))))


def dumps_results(r: Results) -> bytes:
    return dumps_canonical({"schema": SCHEMA, "header": dataclasses.asdict(r.header),
                            "rows": [dataclasses.asdict(x) for x in r.rows]})


def load_results(data: bytes) -> Results:
    doc = json.loads(data)
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"not a {SCHEMA} file (schema {doc.get('schema')!r})")
    h = dict(doc["header"])
    h["serve"] = ServeCondition(**h["serve"])
    header = ResultsHeader(**h)
    rows = tuple(ResultsRow(**x) for x in doc["rows"])
    keys = [(x.task, x.seed) for x in rows]
    if keys != sorted(set(keys)):
        raise ValueError("rows must be unique and sorted by (task, seed)")
    return Results(header, rows)
