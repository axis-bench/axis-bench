"""Load and check a benchmark declaration.

A benchmark VERSION is a directory conf/benchmarks/<version>/ (axis_qvla.bench.versions) holding one
or more declarations and everything they need, identified by one hash over all of it:

  <name>.json          the declaration: task ids, n_seeds, subsets, exclusions, conventions
                       (action space, image geometry, dual-sim, XLA pin), and the names of
                       the two files below
  <name>.tasks.json    per task: canonical prompt, test seeds, gripper widths
  <name>.layouts.json  per task: the layout manifest entry the dual-sim scene is built from
  execution.json       the execution constants (fps, replan, step budget, stability window,
                       frame size, controller modes, noise salt)
  ASSETS.lock.json     every scene file, by sha256
  VERSION.json         the frozen record: the version hash and each file's sha256

`load_benchmark("<version>/<name>")` reads a declaration of a version; a bare name reads the
default version (CONF_DIR).

`load_benchmark` is the single reader of these facts and `check` the single test: every
consumer (the driver, the scorer, the paper tables) reads the declaration through them, so no
second copy of a task list, subset or seed list exists to disagree with it.

Core tier: stdlib only.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

# BenchmarkError is defined with the field validators and re-exported here, unchanged for callers.
from axis_qvla.bench import versions
from axis_qvla.bench.spec_checks import (BenchmarkError, dev_seed_tuple, parse_compatible, parse_dev,
                                    parse_execution)

__all__ = ["BenchmarkError", "Benchmark", "TaskEntry", "load_benchmark", "check", "CONF_DIR",
           "DEFAULT_NAME", "SEED_SETS"]
# The default version's directory, read at call time (tests point it elsewhere).
CONF_DIR = versions.BENCHMARKS_ROOT / versions.DEFAULT_VERSION
DEFAULT_NAME = "franka_heldout24"
# Which declared seeds a run evaluates: the test seeds every reported number is on, or the
# development seeds (dev_seeds), the only ones tuning and checkpoint selection may look at.
SEED_SETS = ("test", "dev")


@dataclasses.dataclass(frozen=True)
class TaskEntry:
    task_id: int
    prompt: str
    seeds: tuple[int, ...]
    open_w: float
    closed_w: float


@dataclasses.dataclass(frozen=True)
class Benchmark:
    name: str
    version: str
    task_ids: tuple[int, ...]
    n_seeds: int
    subsets: dict[str, tuple[int, ...]]
    excluded: dict[int, str]
    action_space: str
    image_geometry: str
    dual_sim: bool
    xla_pin: str
    tasks: dict[int, TaskEntry]
    declaration_path: Path
    tasks_path: Path
    layouts_path: Path
    sha256: str          # of the declaration file, byte-exact
    tasks_sha256: str
    layouts_sha256: str
    dev: dict[str, int] = dataclasses.field(default_factory=dict)
    # Older declarations whose results still score here, sha256 -> why (`compatible_declarations`).
    compatible: dict[str, str] = dataclasses.field(default_factory=dict)
    # execution.json, or {} for a directory that declares none (a test fixture); `bench eval`
    # refuses a benchmark without one.
    execution: dict = dataclasses.field(default_factory=dict)
    # sha256 over every file of the declaration's directory (axis_qvla.bench.versions.version_sha256).
    version_sha256: str = ""
    # VERSION.json's sha256 when the directory is frozen, else None.
    frozen_sha256: str | None = None

    @property
    def conf_dir(self) -> Path:
        return self.declaration_path.parent

    def task(self, task_id: int) -> TaskEntry:
        try:
            return self.tasks[int(task_id)]
        except KeyError:
            raise BenchmarkError(
                f"task {task_id} is not in benchmark {self.name} ({len(self.task_ids)} tasks)"
            ) from None

    def dev_seeds(self, task_id: int) -> tuple[int, ...]:
        """Development seeds, disjoint from the test seeds: tid*1000 + offset + i.
        Malformed or overlapping dev_seeds raise BenchmarkError (axis_qvla.bench.spec_checks)."""
        test = self.tasks.get(int(task_id))
        return dev_seed_tuple(self.name, self.dev, task_id, test.seeds if test else ())

    def seeds_for(self, task_id: int, seed_set: str = "test") -> tuple[int, ...]:
        """The task's seeds of one set: its declared test seeds, or its development seeds."""
        if seed_set == "test":
            return self.task(task_id).seeds
        if seed_set == "dev":
            self.task(task_id)        # the task must be one of this benchmark's
            return self.dev_seeds(task_id)
        raise BenchmarkError(f"seed set must be one of {SEED_SETS}, got {seed_set!r}")

    def accepted_declarations(self) -> frozenset[str]:
        """Declaration digests a results header may carry: this file's, plus the declared lineage."""
        return frozenset({self.sha256, *self.compatible})

    def identity(self) -> dict:
        """What a run record stores: enough to say WHICH benchmark, by content."""
        return {"name": self.name, "version": self.version, "sha256": self.sha256,
                "tasks_sha256": self.tasks_sha256, "layouts_sha256": self.layouts_sha256,
                "version_sha256": self.version_sha256,
                "n_tasks": len(self.task_ids), "n_seeds": self.n_seeds}


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _read(p: Path, what: str) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise BenchmarkError(f"no {what} at {p}") from None
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"{what} {p} is not valid JSON: {exc}") from None


def resolve_ref(ref: str, conf_dir: Path | None = None) -> tuple[Path, str]:
    """`<version>/<name>` -> (that version's directory, name); a bare `<name>` -> (CONF_DIR, name).
    An explicit conf_dir (a test fixture, a version under construction) takes a bare name only."""
    version, sep, name = str(ref).rpartition("/")
    if conf_dir is not None:
        if sep:
            raise BenchmarkError(f"{ref!r}: name a version or pass a directory, not both")
        return Path(conf_dir), name
    if not sep:
        return CONF_DIR, name
    try:
        return versions.version_dir(version), name
    except versions.VersionError as exc:
        raise BenchmarkError(str(exc)) from None


def load_benchmark(name: str = DEFAULT_NAME, *, conf_dir: Path | None = None) -> Benchmark:
    conf, name = resolve_ref(name, conf_dir)
    dpath = conf / f"{name}.json"
    d = _read(dpath, "benchmark declaration")
    for key in ("name", "version", "task_ids", "n_seeds", "tasks_file", "layouts_file",
                "action_space", "image_geometry"):
        if key not in d:
            raise BenchmarkError(f"{dpath} lacks required key {key!r}")
    tpath = conf / d["tasks_file"]
    lpath = conf / d["layouts_file"]
    tdoc = _read(tpath, "benchmark tasks file")
    tasks: dict[int, TaskEntry] = {}
    for tid_s, t in (tdoc.get("tasks") or {}).items():
        g = t.get("gripper") or {}
        tasks[int(tid_s)] = TaskEntry(
            task_id=int(tid_s), prompt=str(t.get("task_name_canonical", "")),
            seeds=tuple(int(s) for s in t.get("test_seeds") or ()),
            open_w=float(g.get("open_w", float("nan"))), closed_w=float(g.get("closed_w", float("nan"))),
        )
    if not lpath.exists():
        raise BenchmarkError(f"no benchmark layouts file at {lpath}")
    epath = conf / "execution.json"
    execution = parse_execution(_read(epath, "execution constants"), epath) if epath.exists() else {}
    try:
        frozen = versions.frozen_record(conf)
        version_sha = versions.version_sha256(versions.content_files(conf))
    except versions.VersionError as exc:
        raise BenchmarkError(str(exc)) from None
    return Benchmark(
        name=str(d["name"]), version=str(d["version"]),
        task_ids=tuple(int(t) for t in d["task_ids"]), n_seeds=int(d["n_seeds"]),
        subsets={k: tuple(int(t) for t in v) for k, v in (d.get("subsets") or {}).items()},
        excluded={int(k): str(v) for k, v in (d.get("excluded") or {}).items()},
        action_space=str(d["action_space"]), image_geometry=str(d["image_geometry"]),
        dual_sim=bool(d.get("dual_sim", True)), xla_pin=str(d.get("xla_pin", "")),
        tasks=tasks, declaration_path=dpath, tasks_path=tpath, layouts_path=lpath,
        sha256=_sha256(dpath), tasks_sha256=_sha256(tpath), layouts_sha256=_sha256(lpath),
        dev=parse_dev(d.get("dev_seeds"), dpath),
        compatible=parse_compatible(d.get("compatible_declarations"), dpath),
        execution=execution, version_sha256=version_sha,
        frozen_sha256=frozen["sha256"] if frozen else None,
    )



def check(b: Benchmark) -> list[str]:
    """Every way the three files can disagree, as human-readable problems. Empty = consistent.

    It trusts no one file: each is checked against the other two.
    """
    problems: list[str] = []
    ids = list(b.task_ids)
    if len(set(ids)) != len(ids):
        problems.append("duplicate task ids in declaration")
    for sub, members in b.subsets.items():
        extra = sorted(set(members) - set(ids))
        if extra:
            problems.append(f"subset {sub!r} names tasks outside the benchmark: {extra}")
    hit = sorted(set(b.excluded) & set(ids))
    if hit:
        problems.append(f"excluded tasks are also scored: {hit}")
    for tid in ids:
        t = b.tasks.get(tid)
        if t is None:
            problems.append(f"task {tid}: no entry in {b.tasks_path.name}")
            continue
        if not t.prompt:
            problems.append(f"task {tid}: empty prompt")
        if len(t.seeds) < b.n_seeds:
            problems.append(f"task {tid}: {len(t.seeds)} seeds, need {b.n_seeds}")
        if len(set(t.seeds)) != len(t.seeds):
            problems.append(f"task {tid}: duplicate seeds")
        if t.open_w != t.open_w or t.closed_w != t.closed_w or t.open_w <= t.closed_w:
            problems.append(f"task {tid}: gripper widths missing or inverted "
                            f"(open {t.open_w}, closed {t.closed_w})")
    lay = _read(b.layouts_path, "benchmark layouts file")
    entries = lay.get("tasks")
    if isinstance(entries, list):
        have = {int(e.get("task_id", -1)) for e in entries}
    elif isinstance(entries, dict):
        have = {int(k) for k in entries}
    else:
        have = set()
    if b.dev:
        try:
            for tid in ids:
                b.dev_seeds(tid)
        except BenchmarkError as exc:
            problems.append(str(exc))
    missing = sorted(set(ids) - have)
    if missing:
        problems.append(f"layouts file lacks tasks: {missing}")
    stray = sorted(t for t in b.tasks if t not in set(ids))
    if stray:
        problems.append(f"{b.tasks_path.name} carries tasks the declaration does not score: {stray}")
    # A directory under conf/benchmarks/ is a released version: frozen, intact, named as declared,
    # with its execution constants. (A fixture elsewhere is checked for consistency only.)
    if b.conf_dir.resolve().parent == versions.BENCHMARKS_ROOT.resolve():
        if b.version != b.conf_dir.name:
            problems.append(f"{b.declaration_path.name} declares version {b.version!r}, its directory is "
                            f"{b.conf_dir.name!r}")
        if not b.execution:
            problems.append(f"{b.conf_dir.name}: no execution.json")
        problems += versions.check_version(b.conf_dir)
    return problems
