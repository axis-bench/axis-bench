"""REALALIGN's four stage-1 runs, compiled by the shared signal compiler's built-ins.

Offline tier: numpy (and the compiler's modules). A `realign build` directory is a corpus (its roots
index and tail-10 ranges) plus the three weight files REALALIGN derives from phase curves. The row
space is built from the first two by the compiler's own builder; each run is the built-in method the
sim benchmark uses, fed the matching weight file as imported scores:

    bc            BC                                       the backbone sampler's permutation (seed 42)
    conditioning  CfgTags, episode quintiles, no dropout   rank weights      -> codes on bc's plan, pure conditioning
    awr           AwrEpisode, P-ratio episode weights      balanced weights  -> per-row weights on bc's plan, global ratio
    filtering     DropTop50, keep 0.5                      episode weights   -> the drop-top schedule (seed 0)

The build's gates run first, every time, read-only (`gates.run_gates(built, write=False)`): a build
whose gates fail is refused with the `GateError` naming them, so no compiled labels are ever compiled
from curves or weights the gates reject, and a stale `gates.json` is never trusted. Read-only because
`built` is a `realign build --out` directory this module only reads, never one it owns -- it may be a
shared or read-only-mounted copy, and compiling labels from it must not rewrite its `gates.json`.
Before a weight file is used, its header must also carry what the built-ins read (`version, reward_id,
tau, cap, horizon, n_rows`) and a provenance stamp naming THIS corpus: a file stamped against another
corpus would compile and train on the wrong rows without an error.

`runs` selects which runs are compiled; only their files are read and only their gates run.

The 'conditioning' run's built-in reports an entropy diagnostic over the tags it draws; when every
episode of the corpus carries the same quality (zero entropy) it refuses rather than return a number
that would read as the opposite finding (`axis_qvla.signals.blocks.tags.ConstantQualityError`). The refusal
is re-raised, with its type, naming the run it came from.
"""
from __future__ import annotations

import json
from pathlib import Path

from axis_qvla.core.canonical import canonical_name
from axis_qvla.realrobot.realign.runs import RUNS
from axis_qvla.realrobot.realign.gates import run_gates
from axis_qvla.signals.blocks.tags import ConstantQualityError
from axis_qvla.signals.bundle import write_bundle
from axis_qvla.signals.compile import compile_program
from axis_qvla.signals.imported import ImportedScores
from axis_qvla.signals.methods.awr import AwrEpisode
from axis_qvla.signals.methods.bc import BC
from axis_qvla.signals.methods.cfg import CfgTags
from axis_qvla.signals.methods.filtering import DropTop50
from axis_qvla.signals.rowspace import build_rowspace
from axis_qvla.signals.schema import Budget, RowSpace, SignalBundle

CONVENTION = "openpi-rowsampler"   # the π0.5 consumer's sampler
STAGE1 = Budget(20_000, 64)
PLAN_SEED = 42                     # openpi TrainConfig's default: the stage-1 runs passed none
DROP_SEED = 0                      # the drop-top schedule's own seed
REQUIRED_HEADER = ("version", "reward_id", "tau", "cap", "horizon", "n_rows")
MANIFEST = "qvla.realrobot.realign_compiled/1"
MANIFEST_FILE = "realign_compiled.json"   # beside the compiled labels: local name -> compiled-labels id
# Recorded in the row space, so part of its id: the module keeps the development root.
ROWSPACE_SOURCE = canonical_name("realrobot.realign") + " build"


def realign_rowspace(built: Path) -> RowSpace:
    """The row space of a `realign build --out` directory: its own roots index and ranges file."""
    built = Path(built)
    roots = built / "roots.json"
    base = roots.parent / json.loads(roots.read_text(encoding="utf-8")).get("base", ".")
    return build_rowspace(name="realworld_pretrain", corpus=base, roots_file=roots, ranges_file=built / "ranges.json",
                          roots_path="roots.json", ranges_path="ranges.json", source=ROWSPACE_SOURCE)


def check_weights_header(scores: ImportedScores, rows: RowSpace) -> None:
    """Refuse a weights file the built-ins cannot attribute to THIS corpus: missing header fields,
    no provenance stamp, a stamp naming another corpus, or a row count that disagrees with `rows`."""
    h = scores.header
    missing = [k for k in REQUIRED_HEADER if k not in h]
    if missing:
        raise ValueError(f"{scores.name}: header lacks {missing}")
    prov = h.get("provenance") or {}
    if "corpus_fingerprint" not in prov or "params" not in prov:
        raise ValueError(f"{scores.name}: no provenance stamp; cannot tell which corpus it was built on")
    if prov["corpus_fingerprint"] != rows.corpus_fingerprint:
        raise ValueError(f"{scores.name}: stamped against {prov['corpus_fingerprint']}, the row space is "
                         f"{rows.corpus_fingerprint}")
    if int(h["n_rows"]) != rows.n_rows:
        raise ValueError(f"{scores.name}: {h['n_rows']} rows, the row space has {rows.n_rows}")


def programs() -> dict:
    """`{run: (program, labels file name or None, seed)}` for every stage-1 run (spec table in the
    module docstring)."""
    return {
        "bc": (BC(CONVENTION), None, PLAN_SEED),
        "conditioning": (CfgTags("episode", CONVENTION, drop_whole=0.0, drop_component=0.0),
                          "awr_weights_phase_rank.json", PLAN_SEED),
        "awr": (AwrEpisode(CONVENTION, "global_ratio"), "awr_weights_phase.json", PLAN_SEED),
        "filtering": (DropTop50(keep=0.5), "awr_weights_phase_ep.json", DROP_SEED),
    }


def compile_realign(built: Path, *, runs: tuple[str, ...] = RUNS) -> dict[str, SignalBundle]:
    """Run the build's gates read-only, scoped to `runs` (raising `GateError` when one fails; never
    writes into `built`), then compile only `runs` of `programs()` over `realign_rowspace(built)`,
    checking each run's weights file against that row space before it reaches the built-in. A run
    `built` was not produced with is refused (`runs.UnbuiltRunError`, from `run_gates`)."""
    built = Path(built)
    run_gates(built, write=False, runs=runs)
    rows = realign_rowspace(built)
    out: dict[str, SignalBundle] = {}
    progs = programs()
    for run in runs:
        program, labels_file, seed = progs[run]
        labels = None
        if labels_file is not None:
            labels = ImportedScores.from_json(built / labels_file, name=labels_file)
            check_weights_header(labels, rows)
        try:
            out[run], _controls = compile_program(program, labels, rows, STAGE1, seed)
        except ConstantQualityError as exc:
            raise ConstantQualityError(
                f"the '{run}' run could not be compiled: quality is constant across this corpus "
                f"(H(q) = 0), so no conditioning tags can be built; {exc}") from exc
    return out


def write_bundles(built: Path, out: Path, *, runs: tuple[str, ...] = RUNS) -> dict[str, str]:
    """Write `runs`' compiled labels under `out` and `out/realign_compiled.json` (MANIFEST_FILE: local
    name -> compiled-labels id). Nothing is written when the gates or a weights check refuse the build. `runs` defaults to
    every run."""
    compiled = compile_realign(built, runs=runs)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ids = {f"realalign_{run}": write_bundle(b, out) for run, b in compiled.items()}
    (out / MANIFEST_FILE).write_text(json.dumps({"schema": MANIFEST, "runs": ids}, indent=1, sort_keys=True))
    # By name, as a method's compile records its labels: a run spec names them `name:realrobot.<run>`.
    from axis_qvla.train.names import record_labels_name

    for run in compiled:
        record_labels_name(out, f"realrobot.{run}", ids[f"realalign_{run}"])
    return ids
