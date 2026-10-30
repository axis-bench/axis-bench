"""`qvla.cells/1`: the frozen map from each paper cell to the exact bytes that produced it.

Why: T0 must be able to say, for every number in the paper, which run, which served
directory, which checkpoint and which rollout tar it came from, and refuse anything else.
`cells.json` is committed (experiments/paper/cells.json) and published byte-identically; T0
compares the two. `cells` is a JSON LIST so its order (the pooled order, which fixes the table
bytes) survives canonical key sorting.

This module also holds the per-run integrity checks T0 applies after reading a tar: the task
set and the declared seeds, which with bench_problems make the work-unit check three-way
(cells.json, benchmark, files). The `results` reference and the optional run `manifest` reference
are shape-checked here (a safe path, a sha256, and a size only if one is recorded); their BYTES are
verified by T0 itself (axis_qvla.paper.t0.ref_problems): reading them needs a Source, and
axis_qvla.paper.fetch imports this module. A cell names a run `manifest` when its run published one.

`bench.r1_declaration_sha256` documents which benchmark DECLARATION FILE the harvest was read
against (the sha256 of `conf/benchmarks/axis-heldout-v1/<name>.json` at harvest time) -- it is
checked here for shape only (64 hex), not compared against the live declaration, because a
declaration is allowed to gain fields after a harvest (new dev-seed subsets, for one)
without invalidating cells already harvested under an earlier, compatible declaration. The
digest that actually gates a rebuild is `bench.units_sha256` (`bench_problems`, below): it is
recomputed from the LIVE benchmark's task ids and n_seeds and compared byte for byte, so a
declaration change that would silently change what a "complete" run means is still caught even
though `r1_declaration_sha256` itself is not re-verified.

Each cell's `lineage` (how its checkpoint was trained) is shape-checked by axis_qvla.paper.lineage; its
`checkpoint` names the published checkpoint (`id`, `ref`, `pub_digest`).

Core tier: stdlib only.
"""
from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Mapping

from axis_qvla.bench.spec import Benchmark
from axis_qvla.bench.results.fingerprint import units_sha256
from axis_qvla.bench.results.seedfiles import SEED_FILE, ServedRun
from axis_qvla.paper.lineage import lineage_problems

CELLS_SCHEMA = "qvla.cells/1"
_HEX64 = re.compile(r"[0-9a-f]{64}")
_FIELDS = ("cell", "run", "box", "served", "backbone", "training_seed", "serve", "eval_fingerprint",
           "rollouts", "results", "checkpoint", "eval_command", "lineage")
_OPTIONAL = ("manifest",)   # the run's published run record, when it has one
_BENCH = ("name", "r1_declaration_sha256", "task_ids", "n_seeds", "units_sha256")
_BOXES = ("s2", "s3")   # the published result box ids; also name a rollouts/<box>/ path component (safe_relpath)


@dataclasses.dataclass(frozen=True)
class CellEntry:
    cell: str
    run: str
    box: str
    served: str
    backbone: str
    training_seed: Mapping[str, int | None] | None   # {"pretrain": int | None, "ft": int}
    serve: Mapping[str, object]
    eval_fingerprint: str
    rollouts: Mapping[str, object]
    results: Mapping[str, object] | None
    checkpoint: Mapping[str, object]
    eval_command: str
    lineage: Mapping[str, object]       # how the checkpoint was trained (axis_qvla.paper.lineage)
    manifest: Mapping[str, object] | None = None   # the run record, when the run published one


@dataclasses.dataclass(frozen=True)
class Cells:
    release: str
    bench: Mapping[str, object]
    cells: tuple[CellEntry, ...]


def _hex(x: object) -> bool:
    return isinstance(x, str) and _HEX64.fullmatch(x) is not None


def safe_relpath(p: object) -> bool:
    return (isinstance(p, str) and bool(p) and not p.startswith("/") and "\\" not in p
            and all(part not in ("", ".", "..") for part in p.split("/")))


def _file_ref(name: str, ref: object, need_size: bool) -> list[str]:
    if not isinstance(ref, dict) or not safe_relpath(ref.get("path")) or not _hex(ref.get("sha256")):
        return [f"{name}: needs a safe relative path and a sha256"]
    if need_size and not (isinstance(ref.get("size"), int) and ref["size"] >= 0):
        return [f"{name}: needs an integer size"]
    if "size" in ref and not (isinstance(ref["size"], int) and ref["size"] >= 0):
        return [f"{name}: a recorded size must be a non-negative integer"]
    return []


def _cell_problems(i: int, c: object) -> list[str]:
    if not isinstance(c, dict) or not set(_FIELDS) <= set(c) <= set(_FIELDS + _OPTIONAL):
        return [f"cells[{i}]: fields must be exactly {list(_FIELDS)}, plus {list(_OPTIONAL)} when published"]
    out = []
    for k in ("cell", "run", "box", "served", "backbone", "eval_command"):
        if not isinstance(c[k], str) or not c[k]:
            out.append(f"cells[{i}].{k}: must be a non-empty string")
    if isinstance(c["box"], str) and c["box"] not in _BOXES:
        out.append(f"cells[{i}].box: must be one of {_BOXES}")
    ts = c["training_seed"]
    if ts is not None and not (isinstance(ts, dict) and set(ts) == {"pretrain", "ft"} and isinstance(ts["ft"], int)
                               and (ts["pretrain"] is None or isinstance(ts["pretrain"], int))):
        out.append(f"cells[{i}].training_seed: {{pretrain: int or null, ft: int}} or null")
    s = c["serve"]
    if not isinstance(s, dict) or set(s) != {"fields", "uncond_fields", "beta"}:
        out.append(f"cells[{i}].serve: needs exactly fields, uncond_fields, beta")
    if not _hex(c["eval_fingerprint"]):
        out.append(f"cells[{i}].eval_fingerprint: must be 64 hex")
    out += _file_ref(f"cells[{i}].rollouts", c["rollouts"], True)
    if isinstance(c["rollouts"], dict) and not str(c["rollouts"].get("path", "")).startswith("rollouts/"):
        out.append(f"cells[{i}].rollouts.path: must be under rollouts/")
    if c["results"] is not None:
        out += _file_ref(f"cells[{i}].results", c["results"], False)
    if "manifest" in c:
        out += _file_ref(f"cells[{i}].manifest", c["manifest"], False)
    if not isinstance(c["checkpoint"], dict) or not isinstance(c["checkpoint"].get("ref"), str):
        out.append(f"cells[{i}].checkpoint: needs a ref")
    out += lineage_problems(i, c["lineage"])
    return out


def validate_cells(doc: Mapping[str, object]) -> list[str]:
    if doc.get("schema") != CELLS_SCHEMA:
        return [f"schema must be {CELLS_SCHEMA}"]
    out = [] if isinstance(doc.get("release"), str) and doc["release"] else ["release: non-empty string"]
    b = doc.get("bench")
    if not isinstance(b, dict) or set(b) != set(_BENCH):
        return out + [f"bench: needs exactly {list(_BENCH)}"]
    ids = b["task_ids"]
    if not (isinstance(ids, list) and ids and all(isinstance(t, int) for t in ids) and len(set(ids)) == len(ids)):
        out.append("bench.task_ids: a non-empty list of unique ints")
    if not (isinstance(b["n_seeds"], int) and b["n_seeds"] > 0):
        out.append("bench.n_seeds: a positive int")
    for k in ("r1_declaration_sha256", "units_sha256"):
        if not _hex(b[k]):
            out.append(f"bench.{k}: must be 64 hex")
    cells = doc.get("cells")
    if not isinstance(cells, list) or not cells:
        return out + ["cells: a non-empty list"]
    for i, c in enumerate(cells):
        out += _cell_problems(i, c)
    for k in ("cell", "run", "served"):
        vals = [c.get(k) for c in cells if isinstance(c, dict)]
        if len(set(vals)) != len(vals):
            out.append(f"cells: duplicate {k} values")
    return out


def load_cells(data: bytes) -> Cells:
    doc = json.loads(data)
    problems = validate_cells(doc)
    if problems:
        raise ValueError("invalid cells.json: " + "; ".join(problems[:5]))
    return Cells(doc["release"], doc["bench"], tuple(CellEntry(**c) for c in doc["cells"]))


def bench_problems(cells: Cells, bench: Benchmark) -> list[str]:
    want = units_sha256(bench.task_ids, bench.n_seeds)
    out = []
    if cells.bench["name"] != bench.name:
        out.append(f"cells.json is for benchmark {cells.bench['name']!r}, not {bench.name!r}")
    if list(cells.bench["task_ids"]) != list(bench.task_ids) or cells.bench["n_seeds"] != bench.n_seeds:
        out.append("cells.json task ids / n_seeds differ from the benchmark declaration")
    if cells.bench["units_sha256"] != want:
        out.append(f"units_sha256: cells.json {str(cells.bench['units_sha256'])[:16]} vs benchmark {want[:16]}")
    return out


def run_problems(cell: CellEntry, run: ServedRun, bench: Benchmark) -> list[str]:
    """The files leg of the three-way work-unit check: the task set, and per task exactly the declared
    seeds. A run that passes holds exactly the units units_sha256(bench) digests, so no digest is
    recomputed here: that comparison could never fail once the seed check passed."""
    if set(run.tasks) != set(bench.task_ids):
        return [f"{cell.cell}: task dirs {sorted(set(run.tasks) ^ set(bench.task_ids))} disagree with the benchmark"]
    out = []
    for tid in bench.task_ids:
        names = [SEED_FILE.fullmatch(f.name) for f in run.tasks[tid]]
        seeds = sorted(int(m.group(1)) for m in names if m is not None)
        if None in names or seeds != [tid * 1000 + i for i in range(bench.n_seeds)]:
            out.append(f"{cell.cell}: task {tid} does not hold exactly seeds {tid * 1000}..{tid * 1000 + bench.n_seeds - 1}")
    return out
