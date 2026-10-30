"""The scorer's refusals, one function each: which benchmark version, which seeds, which execution,
which serving condition.

Why: a results file can be compared only with another measured on the same benchmark VERSION,
over the same seed set, under the same execution constants. Each rule raises ValueError naming
what differs; `score` and `contrast` (axis_qvla/bench/results/score.py) apply all of them, so a file one
command refuses no other command scores. A file without the version keys (every published paper
file) is accepted through the declaration lineage the benchmark itself declares
(`compatible_declarations`), and only with the same work-unit digest.

Off-protocol means the file's execution differs from the benchmark's: its header says so
(`execution.off_protocol`, or recorded `deviations`), or a constant or convention it records is
not the declared one. Such a file is scored only when the caller asks (`--allow-off-protocol`),
and every line that reports it says OFF-PROTOCOL; a contrast pairs only files executed alike.

Core tier: stdlib only.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from axis_qvla.bench import execution
from axis_qvla.bench.results.fingerprint import units_sha256
from axis_qvla.bench.results.format import Results
from axis_qvla.bench.results.recorded import check_quality_tag
from axis_qvla.bench.spec import SEED_SETS, Benchmark


def seed_set_of(r: Results) -> str:
    """The seed set a file was read on: `test` unless its header says otherwise (published files
    predate the key and are test reads)."""
    return str(r.header.bench.get("seed_set") or "test")


def units_for(bench: Benchmark, seed_set: str) -> str:
    """The work-unit digest of `seed_set`: tid*1000 + offset + i for the first n_seeds seeds."""
    if seed_set not in SEED_SETS:
        raise ValueError(f"seed set {seed_set!r} is not one of {SEED_SETS}")
    if seed_set == "test":
        return units_sha256(bench.task_ids, bench.n_seeds)
    if not bench.dev:
        raise ValueError(f"benchmark {bench.name} declares no development seeds")
    return units_sha256(bench.task_ids, bench.n_seeds, offset=int(bench.dev["offset"]))


def check_bench(r: Results, bench: Benchmark) -> None:
    """Name, work-unit digest (of the file's seed set) AND declaration content hash must match;
    a missing key reads as None and refuses like a wrong one. A recorded version hash must be the
    loaded version's, and a frozen version must still be intact."""
    want_units = units_for(bench, seed_set_of(r))
    got_name = r.header.bench.get("name")
    got_units = r.header.bench.get("units_sha256")
    got_decl = r.header.bench.get("declaration_sha256")
    if got_name != bench.name or got_units != want_units or got_decl not in bench.accepted_declarations():
        raise ValueError(f"{r.header.served}: results are for benchmark {got_name!r} "
                         f"units {str(got_units)[:16]} declaration {str(got_decl)[:16]}, not "
                         f"{bench.name} units {want_units[:16]} declaration {bench.sha256[:16]}")
    got_version = r.header.bench.get("version_sha256")
    if got_version is not None and got_version != bench.version_sha256:
        raise ValueError(f"{r.header.served}: measured on benchmark version {r.header.bench.get('version')!r} "
                         f"({str(got_version)[:16]}), not {bench.version!r} ({bench.version_sha256[:16]}); "
                         f"results of different versions are never pooled or compared")
    if bench.frozen_sha256 is not None and bench.frozen_sha256 != bench.version_sha256:
        raise ValueError(f"benchmark version {bench.version!r} was edited after it was frozen "
                         f"(`qvla bench check` names the files); nothing scores on it")


def stated(r: Results) -> dict[str, object]:
    """The execution values a file's header states, as `axis_qvla.bench.execution` keys them: the
    constants it recorded (a published file records no n_stable_frames) and its two conventions."""
    ex = r.header.execution or {}
    out: dict[str, object] = {k: ex[k] for k in execution.NUMERIC if ex.get(k) is not None}
    out["image_geometry"] = str(r.header.image_geometry).strip().lower()
    out["action_space"] = str(r.header.action_space).strip()
    return out


def deviations(r: Results, bench: Benchmark) -> dict[str, object]:
    """What the file executed that the benchmark does not declare: its recorded `deviations`, plus
    what `execution.deviations` finds between the values it states and execution.json's (when the
    benchmark has one; a value the header does not state is not a deviation)."""
    out: dict[str, object] = dict((r.header.execution or {}).get("deviations") or {})
    decl = execution.declared(bench)
    if decl is None:
        return out
    for k, v in execution.deviations({**decl, **stated(r)}, decl).items():
        out.setdefault(k, v)
    return out


def is_off_protocol(r: Results, bench: Benchmark) -> bool:
    return bool((r.header.execution or {}).get("off_protocol")) or bool(deviations(r, bench))


def check_execution(r: Results, bench: Benchmark, *, allow_off_protocol: bool) -> bool:
    """True when the file is off-protocol (and the caller allowed it); raises when not allowed."""
    off = is_off_protocol(r, bench)
    if off and not allow_off_protocol:
        dev = deviations(r, bench)

        def one(k: str, v: object) -> str:
            if isinstance(v, Mapping) and "executed" in v:
                return f"{k} {v['executed']!r}, declared {v.get('declared')!r}"
            return f"{k} {v!r}"
        what = "; ".join(one(k, v) for k, v in sorted(dev.items())) if dev else \
            "the run was started to allow deviations (--off-protocol)"
        raise ValueError(f"{r.header.served}: executed off-protocol ({what}); it is scored only with "
                         f"--allow-off-protocol, and then labelled OFF-PROTOCOL")
    return off


def check_serve(r: Results) -> None:
    """A quality-conditioned file must name a quality bin; tag 0 is the untagged sentinel
    (axis_qvla/bench/results/recorded.py, NO_TAG), never a serving condition."""
    fields = r.header.serve.fields or {}
    if "quality" in fields:
        try:
            check_quality_tag(fields["quality"])
        except ValueError as exc:
            raise ValueError(f"{r.header.served}: {exc}") from None


def check_rows(r: Results) -> None:
    """A rollout that ended in an `error:` reason is a broken harness, not a failure: a file with any
    is refused, never scored (`results build` refuses such a run before it is written)."""
    errored = [(x.task, x.seed) for x in r.rows if str(x.reason).startswith("error:")]
    if errored:
        raise ValueError(f"{r.header.served}: {len(errored)} rollouts ended with an `error:` reason (first "
                         f"{errored[0]}), a broken harness rather than a result")


def _executed(r: Results) -> dict[str, object]:
    out = stated(r)
    for k, d in ((r.header.execution or {}).get("deviations") or {}).items():
        out.setdefault(k, d.get("executed") if isinstance(d, Mapping) else d)
    return out


def check_alike(rs: Sequence[Results]) -> None:
    """Every file of one contrast executed the same constants: a constant both record must agree,
    and a deviation one records is one the other must have executed too."""
    first = rs[0]
    base, base_dev = _executed(first), set((first.header.execution or {}).get("deviations") or {})
    for r in rs[1:]:
        cur, cur_dev = _executed(r), set((r.header.execution or {}).get("deviations") or {})
        keys = (set(base) & set(cur)) | (base_dev ^ cur_dev)
        diff = sorted(k for k in keys if base.get(k) != cur.get(k))
        if diff:
            raise ValueError(f"{first.header.served} and {r.header.served} were executed differently ("
                             + ", ".join(f"{k} {base.get(k)!r} vs {cur.get(k)!r}" for k in diff)
                             + "); a contrast pairs reads executed alike")
