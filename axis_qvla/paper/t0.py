"""T0: rebuild every simulation number of the paper from the published per-seed files, and check it.

Why: reproducibility acceptance for v1.0 is T0 only: anyone can recompute each figure value,
Table I entry and in-text number exactly from `qvla-v1-results` with stdlib Python on a CPU.
The pipeline:

  1. cells.json from the source must equal the committed cell map byte for byte;
  2. per cell: every file cells.json names besides the tar -- its `results` file and, when it
     names one, its run `manifest` -- must be in the source at the recorded sha256 (and size, where
     recorded); then the rollout tar's sha256, its eval fingerprint, the task set, the declared
     seeds and the work-unit digest (three ways: cells.json, benchmark, files);
  3. tallies -> per-box TSVs (cells order, tasks in string order) -> pooled cells -> the paper's
     outputs (axis_qvla.paper.render: figure data, Table I, claims.json, results page), laid out by
     experiments/paper/paper.json; a Table I cell whose recorded pretrain seed is not its replicate's
     is an integrity problem;
  4. `check`: compare every output with the source's expected/, reading expected/MANIFEST.json
     through the pin the caller holds for it (the CLI's, for the committed cell map).

Integrity failures are collected as `problems` (exit 2); an unobtainable byte raises
FetchError (exit 3). A referenced results or manifest file the source does not hold is an
integrity failure, not a fetch error: nothing is computed from it, so its absence is not
"T0 could not get a byte" but "the pack does not hold what cells.json says it publishes".

Core tier: stdlib only.
"""
from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping

from axis_qvla.bench.spec import Benchmark
from axis_qvla.paper.cells import CellEntry, bench_problems, load_cells, run_problems
from axis_qvla.paper.expected import compare_expected
from axis_qvla.paper.fetch import HfSource, MissingFile, ShaMismatch, Source
from axis_qvla.paper.pool import dump_pooled, pool_cells
from axis_qvla.paper.render import render_paper
from axis_qvla.paper.spec import PaperSpec
from axis_qvla.paper.tsv import TsvRow, rows_for_run, tallies_by_run, write_tsv
from axis_qvla.bench.results.fingerprint import FingerprintError, eval_fingerprint
from axis_qvla.bench.results.seedfiles import SeedFileError, read_served_tar


@dataclasses.dataclass(frozen=True)
class T0Result:
    outputs: Mapping[str, bytes]
    problems: tuple[str, ...]


def _fail(*problems: str) -> T0Result:
    return T0Result({}, tuple(problems))


def ref_problems(source: Source, c: CellEntry) -> list[str]:
    """The results map names a `results` file per result and a run `manifest` where the run published
    one; a corrupt or absent one fails the check. Each is fetched through the same Source as the
    tars, so over hf:// the release lock pins it too (a path the lock does not list is refused)."""
    out = []
    for kind, ref in (("results", c.results), ("manifest", c.manifest)):
        if ref is None:
            continue
        rel = str(ref["path"])
        try:
            p = source.path(rel, sha256=str(ref["sha256"]))
        except MissingFile:
            out.append(f"{c.cell}: {kind} {rel} is named in cells.json but the results source does not hold it")
            continue
        except ShaMismatch as exc:
            out.append(f"{c.cell}: {exc}")
            continue
        if "size" in ref and (size := p.stat().st_size) != ref["size"]:
            out.append(f"{c.cell}: {rel} is {size} bytes, cells.json says {ref['size']}")
    return out


def rebuild(source: Source, *, cells_ref: bytes, bench: Benchmark, spec: PaperSpec) -> T0Result:
    if source.read("cells.json", sha256=None) != cells_ref:
        return _fail("cells.json in the results source differs from the committed results map")
    try:
        cells = load_cells(cells_ref)
    except ValueError as exc:
        return _fail(str(exc))
    problems = bench_problems(cells, bench)
    if problems:
        return _fail(*problems)
    by_box: dict[str, list[TsvRow]] = {}
    for c in cells.cells:
        problems += ref_problems(source, c)
        try:
            tar = source.path(str(c.rollouts["path"]), sha256=str(c.rollouts["sha256"]))
            if tar.stat().st_size != c.rollouts["size"]:
                problems.append(f"{c.cell}: {c.rollouts['path']} is {tar.stat().st_size} bytes, cells.json says {c.rollouts['size']}")
                continue
            run = read_served_tar(tar, c.served)
            fp = eval_fingerprint(run, bench.task_ids)
        except (ShaMismatch, SeedFileError, FingerprintError) as exc:
            problems.append(f"{c.cell}: {exc}")
            continue
        if fp != c.eval_fingerprint:
            problems.append(f"{c.cell}: eval_fingerprint {fp[:16]} != cells.json {c.eval_fingerprint[:16]}")
            continue
        run_issues = run_problems(c, run, bench)
        problems += run_issues
        if not run_issues:
            by_box.setdefault(c.box, []).extend(rows_for_run(c.run, c.served, run.tallies(), fp))
    if problems:
        return _fail(*problems)
    out: dict[str, bytes] = {f"tsv/pt_r1_{box}.tsv": write_tsv(rows) for box, rows in by_box.items()}
    data = tallies_by_run(r for rows in by_box.values() for r in rows)
    out["pooled_r1.json"] = dump_pooled(pool_cells([(c.cell, c.run) for c in cells.cells], data,
                                                   n_tasks=len(bench.task_ids)))
    paper, problems = render_paper(json.loads(out["pooled_r1.json"]),
                                   {c.cell: c.training_seed for c in cells.cells}, spec)
    if problems:
        return _fail(*problems)
    out.update(paper)
    return T0Result(out, ())


def check(result: T0Result, source: Source, *, manifest_sha256: str | None = None) -> list[str]:
    """`manifest_sha256` pins expected/MANIFEST.json, so a LOCAL pack's expected/ is as fixed as
    hf://'s (where the checksum manifest pins it; given both, they must agree). None reads it
    unpinned: a pack whose results map has no committed pin, such as a test pack."""
    manifest = json.loads(source.read("expected/MANIFEST.json", sha256=manifest_sha256))
    for rel, sha in sorted(manifest["files"].items()):
        source.path(f"expected/{rel}", sha256=sha)
    return compare_expected(result.outputs, source.path("expected/MANIFEST.json", sha256=manifest_sha256).parent,
                            skip_part_files=isinstance(source, HfSource))
