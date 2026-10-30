"""What `bench eval` does once the driver returns 0: the run's results file, and its score printed.

Why: a run is reported through its results file (qvla.results/v1, docs/reference/results-format.md).
It is built here, as `results build` would build it, and its number printed, as `bench score` would
print it. The run root, the run name and the benchmark are already known here, and the
training seeds are the ones `bench eval` was given (none for a policy whose seeds nobody stated). The
file lands beside the run's records, <r1>/out_<run>/results.json, and the table printed is the
reporting rules' all-24 aggregate and every task's rate with its Wilson interval.

A results file is a record, not a scratch file: `bench eval` run twice on the same `--run` (a tally,
a retry, a second `--ft-seed` by mistake) must never silently replace what an earlier run already
wrote there under a different build. `finish` builds and scores the new run FIRST, in memory, and
only then compares it against whatever is already on disk -- identical bytes are a no-op pass,
different bytes are refused (ResultsConflict) naming the first header field that disagrees, and a
score() that rejects the run never reaches disk at all.

Core tier: stdlib only.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path

from axis_qvla.bench.results.bootstrap import wilson
from axis_qvla.bench.results.build import BuildError, build_results
from axis_qvla.bench.results.format import Results, ResultsHeader, dumps_results, load_results
from axis_qvla.bench.results.score import ArmScore, score
from axis_qvla.bench.sim.arms import out_dir_name
from axis_qvla.bench.spec import BenchmarkError

RESULTS_FILE = "results.json"


class ResultsConflict(Exception):
    """A different results file already sits where this build would write; refused, not replaced."""


def results_path(run_root: Path, arm: str) -> Path:
    """Where `bench eval` writes the results file of the run filed as `arm` under `run_root`."""
    return Path(run_root) / out_dir_name(arm) / RESULTS_FILE


def _first_difference(old: Results, new: Results) -> tuple[str, object, object] | None:
    """(field, old value, new value) of the first header field (declared order) where `old` and
    `new` disagree; `("rows", <old count>, <first differing (task, seed)>)` when every header field
    agrees but the rollout outcomes differ; None when the two are the same record (two builds that
    disagree can never produce identical bytes, since dumps_results is canonical -- sorted keys, so
    this is the name a byte difference is always traceable to)."""
    for f in dataclasses.fields(ResultsHeader):
        a, b = getattr(old.header, f.name), getattr(new.header, f.name)
        if a != b:
            return f.name, a, b
    if old.rows == new.rows:
        return None
    old_keys = {(r.task, r.seed): r for r in old.rows}
    new_keys = {(r.task, r.seed): r for r in new.rows}
    first = next((k for k in sorted(old_keys.keys() | new_keys.keys())
                 if old_keys.get(k) != new_keys.get(k)), None)
    return "rows", f"{len(old.rows)} rows (first difference at task/seed {first})", f"{len(new.rows)} rows"


def _conflict_refusal(existing: bytes, new: Results, path: Path) -> str:
    """Why `new` cannot be written over the results file already at `path`."""
    try:
        old = load_results(existing)
    except ValueError as exc:
        return f"a different file already exists at {path} and is not a results file ({exc}); remove it to proceed"
    diff = _first_difference(old, new)
    if diff is None:
        return (f"an earlier results file at {path} has different bytes for a reason this check "
                f"cannot name (non-canonical JSON on disk?); remove it to rebuild")
    field, old_value, _new_value = diff
    advice = "the same --ft-seed/--pretrain-seed" if field == "training_seed" else "the same build inputs"
    # The rows summary is already prose; a header value is quoted so a string reads as one.
    shown = old_value if field == "rows" else repr(old_value)
    return f"an earlier results file records {field} {shown}; pass {advice}, or remove {path}"


def _write(run_root: Path, arm: str, r: Results) -> Path:
    """Write `r` atomically at its path; a different results file already there is refused
    (ResultsConflict), identical bytes are a no-op pass."""
    out = results_path(run_root, arm)
    data = dumps_results(r)
    if out.exists():
        existing = out.read_bytes()
        if existing == data:
            return out
        raise ResultsConflict(_conflict_refusal(existing, r, out))
    tmp = out.with_name(out.name + ".partial")
    tmp.write_bytes(data)
    tmp.replace(out)
    return out


def write_results(run_root: Path, arm: str, bench, *, ft_seed: int | None,
                  pretrain_seed: int | None) -> tuple[Path, Results]:
    """Build the run's results file (axis_qvla.bench.results.build) and write it atomically. `finish`
    does not call this -- it builds and scores before writing (see module docstring) -- this is the
    build-and-write-only path for a caller that does not need to score the run first."""
    r = build_results(Path(run_root), arm, bench, ft_seed=ft_seed, pretrain_seed=pretrain_seed)
    return _write(run_root, arm, r), r


def table(s: ArmScore) -> str:
    """The aggregate and the per-task rates as Markdown tables, tasks in numeric order."""
    lines = [f"| run | {s.view} SR (%) | SE | rollouts | missing |", "|---|---:|---:|---:|---:|",
             f"| {s.served} | {s.pct:.2f} | {s.se:.2f} | {s.rollouts} | {s.missing} |", "",
             "| task | successes | SR (%) | 95% Wilson interval |", "|---|---:|---:|---|"]
    for t, k, n in sorted(s.tallies):
        lo, hi = wilson(k, n)
        lines.append(f"| {t} | {k}/{n} | {100 * k / n:.1f} | [{lo:.1f}, {hi:.1f}] |")
    return "\n".join(lines)


def _existing_note(path: Path) -> str:
    """Whether a results file is still at `path` -- appended to a FAIL message so a refusal that
    wrote nothing new also says whether an earlier run's record survives there. `stat()` in a try,
    never `Path.exists()`: on 3.11-3.13 `exists()` lets EACCES on an unreadable parent escape as an
    uncaught traceback, and on 3.14 it swallows EACCES and would read that parent as absent --
    named here instead of guessed either way."""
    try:
        path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return f" No results file exists yet at {path}."
    except OSError as exc:
        return f" Whether an earlier results file is at {path} cannot be checked ({type(exc).__name__})."
    return f" An earlier results file is still at {path}."


def finish(run_root: Path, arm: str, bench, *, ft_seed: int | None, pretrain_seed: int | None,
           report: Callable[[str], None] = print) -> int:
    """0 with the file written and the table printed; 2 when the run cannot be built into a results
    file (errored rollouts, incomplete tasks, a rejected score, a results file already there under a
    different build, ...: the refusal is printed); 3 when a record or the output cannot be read or
    written. The run is built and scored before anything is written, so a rejection -- by the build
    or by score() -- never touches disk. An off-protocol run (`--off-protocol`) is scored and
    labelled, never dropped: the user asked for it."""
    path = results_path(run_root, arm)
    try:
        r = build_results(Path(run_root), arm, bench, ft_seed=ft_seed, pretrain_seed=pretrain_seed)
        s = score(r, bench, allow_off_protocol=True)
        out = _write(run_root, arm, r)
    except ResultsConflict as exc:
        report(f"FAIL  {exc}")
        return 2
    except OSError as exc:
        report(f"FAIL  the run finished, but its results file cannot be read or written: {exc}.{_existing_note(path)}")
        return 3
    except (BuildError, BenchmarkError, ValueError, KeyError) as exc:
        report(f"FAIL  the run finished, but it cannot be built into a results file: {exc}.{_existing_note(path)}")
        return 2
    report(f"results: {out}")
    report(table(s) + ("\n\nOFF-PROTOCOL: the execution deviates from the benchmark's" if s.off_protocol else ""))
    return 0
