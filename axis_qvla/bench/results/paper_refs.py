"""`paper:<result>` in `results contrast`: a published result of the paper, named as `qvla paper reeval
--list` prints it, in place of a results file.

Why: a method author's last number is the paper's Table I statistic -- their method against the
paper's plain BC, BC-normalized and paired per training replicate (the PRETRAIN seed). The BC side is
published, but naming its file (results/<box>/<run>_dualsim.json) means knowing the results pack's
layout. `paper:'bc|bare'` names it by its results-map id instead, and this module resolves it through
the committed results map (experiments/paper/cells.json) to that result's REPLICATE GROUP, the cells
Table I pools for the same VLA and role at each training seed (paper.json table1.replicates: for
`bc|bare` that is bc|bare, bc43|bare and bc44|bare at pretrain seeds 42, 43, 44), or to the cell's
own file alone when Table I does not replicate it. A member the map does not hold is left out, as
Table I leaves it out. Each file is read through the Source `qvla paper tables` reads (a local pack,
or hf:// through the release checksum manifest) with the sha256 the map records, so a substituted
file is refused, never contrasted.

Which replicates a reference contributes: when any side of the contrast holds files of the user's
own, the replicates at those files' replicate keys only, so one fine-tune of a pretrain at seed 43 is
contrasted with the paper's seed-43 replicate; a reference with no replicate at one of those keys is
refused, naming the seeds it has. With references only, every replicate. A reference to another
backbone's result than the user's files' (as their run id or checkpoint names it) is refused, naming
Table I's same role for theirs; so are references that between them span more than one backbone.

Core tier: stdlib only.
"""
from __future__ import annotations

import dataclasses
import difflib
from collections.abc import Callable, Iterable, Mapping

from axis_qvla.bench.results.format import Results, load_results
from axis_qvla.bench.results.score import _seed_key
from axis_qvla.paper.cells import CellEntry, Cells
from axis_qvla.paper.fetch import ShaMismatch

PREFIX = "paper:"


class PaperRefError(ValueError):
    """A `paper:` reference that names no published result, or none at the seeds asked for."""


@dataclasses.dataclass(frozen=True)
class Resolved:
    ref: str                              # as typed, e.g. "paper:bc|bare"
    members: tuple[CellEntry, ...]        # in replicate-key order
    results: tuple[Results, ...]          # one per member


def is_ref(arg: str) -> bool:
    return arg.startswith(PREFIX)


def _typed(cell_id: str) -> str:
    """A reference as a shell user types it: the id quoted, since result ids hold a `|`."""
    return f"{PREFIX}'{cell_id}'"


def _role(cell_id: str, table1: Mapping[str, object]) -> tuple[str, str] | None:
    """(VLA, role) of `cell_id` in Table I's replicates, or None when Table I does not replicate it."""
    for vla, reps in table1["replicates"].items():
        for r in reps:
            role = next((k for k, c in r["cells"].items() if c == cell_id), None)
            if role is not None:
                return vla, role
    return None


def _cell(cell_id: str, cells: Cells) -> CellEntry:
    by = {c.cell: c for c in cells.cells}
    if cell_id not in by:
        close = difflib.get_close_matches(cell_id, list(by), n=3)
        hint = f" (did you mean {', '.join(_typed(c) for c in close)}?)" if close else ""
        raise PaperRefError(f"{_typed(cell_id)} is not a published result{hint}; `qvla paper reeval --list` "
                            f"lists them (the first column), e.g. {PREFIX}'bc|bare'")
    return by[cell_id]


def replicate_group(cell_id: str, cells: Cells, table1: Mapping[str, object]) -> tuple[CellEntry, ...]:
    """The cells Table I pools with `cell_id` (same VLA, same role, every training seed) that the map
    holds; `cell_id` alone when Table I does not replicate it."""
    cell, by = _cell(cell_id, cells), {c.cell: c for c in cells.cells}
    where = _role(cell_id, table1)
    if where is None:
        return (cell,)
    vla, role = where
    return tuple(by[q["cells"][role]] for q in table1["replicates"][vla] if q["cells"][role] in by)


def backbone_of(r: Results, backbones: Iterable[str]) -> str | None:
    """The backbone a results file was trained on, when it says: a run `qvla train` named
    (`<backbone>.finetune.<name>...`, what `bench eval --from-run` files it under) or a published
    checkpoint (`<org>/qvla-v1-<backbone>@...`). None when neither tells."""
    known = set(backbones)
    head = r.header.arm.split(".", 2)
    if len(head) == 3 and head[0] in known and head[1] == "finetune":
        return head[0]
    ref = r.header.checkpoint.get("ref") if isinstance(r.header.checkpoint, Mapping) else None
    if isinstance(ref, str):
        repo = ref.split("@", 1)[0].split(":", 1)[0].rpartition("/")[2]
        if repo.startswith("qvla-v1-") and repo[len("qvla-v1-"):] in known:
            return repo[len("qvla-v1-"):]
    return None


def _same_backbone(cell: CellEntry, cells: Cells, table1: Mapping[str, object], backbones: set[str]) -> None:
    """Refuse a reference to another backbone's result than the user's files', naming the matching one."""
    if not backbones or cell.backbone in backbones:
        return
    held = {c.cell for c in cells.cells}
    where = _role(cell.cell, table1)
    match = None
    if where is not None:
        for b in sorted(backbones):
            match = next((r["cells"][where[1]] for r in table1["replicates"].get(b, ())
                          if r["cells"].get(where[1]) in held), None)
            if match is not None:
                break
    fix = (f"use {_typed(match)}, the main results' same role for that backbone" if match else
           "name a result of that backbone (`qvla paper reeval --list` prints each one's backbone)")
    raise PaperRefError(f"{_typed(cell.cell)} is a {cell.backbone} result and your results are "
                        f"{', '.join(sorted(backbones))}: {fix}")


def _key(c: CellEntry) -> int | None:
    return _seed_key(c.training_seed)


def select(ref: str, cells: Cells, table1: Mapping[str, object], keys: set[int] | None,
           backbones: set[str] | None = None) -> tuple[CellEntry, ...]:
    """The members of `ref`'s replicate group a contrast reads: those at `keys` (None: all). With
    `backbones` (the user's files'), a result of another backbone is refused."""
    _same_backbone(_cell(ref[len(PREFIX):], cells), cells, table1, backbones or set())
    group = replicate_group(ref[len(PREFIX):], cells, table1)
    by_key: dict[int, CellEntry] = {}
    for c in group:
        k = _key(c)
        if k is None or c.results is None:
            continue                       # no replicate key, or no published results file: never paired
        by_key.setdefault(k, c)
    if not by_key:
        raise PaperRefError(f"{ref}: the paper published no results file with a training seed for it")
    if keys is None:
        return tuple(by_key[k] for k in sorted(by_key))
    missing = sorted(set(keys) - set(by_key))
    if missing:
        have = ", ".join(str(k) for k in sorted(by_key))
        raise PaperRefError(f"{ref} has replicates at training seed(s) {have}, none at {', '.join(map(str, missing))} "
                            f"(the training seed of your results): pretrain with `qvla train pretrain ... --seed "
                            f"{sorted(by_key)[0]}` (or another of {have}) to contrast with it")
    return tuple(by_key[k] for k in sorted(keys))


def _read(source, c: CellEntry) -> Results:
    """One published results file, verified against the map; a mismatch is named by path, never by
    digest (the reader can act on the path: fetch the file again)."""
    rel = str(c.results["path"])
    try:
        return load_results(source.read(rel, sha256=str(c.results["sha256"])))
    except ShaMismatch:
        root = getattr(source, "root", None)
        shown = f"{root}/{rel}" if root is not None else rel
        raise PaperRefError(f"{shown} is not the file the paper published; re-fetch it with "
                            f"`qvla data get results`") from None


def _one_backbone(picked: Mapping[str, tuple[CellEntry, ...]]) -> None:
    """Refuse references whose members span more than one VLA: a contrast pairs one VLA's replicates
    on the training seed, and seed k of one VLA's run is unrelated to seed k of another's. Checked over
    the references themselves, so a contrast made only of references (or next to files that name no
    backbone) is held to it too, not only one beside the user's files."""
    first: dict[str, str] = {}
    for ref, members in picked.items():
        for c in members:
            first.setdefault(c.backbone, ref)
    if len(first) > 1:
        named = " and ".join(f"{_typed(ref[len(PREFIX):])} is {b}" for b, ref in first.items())
        raise PaperRefError(f"{named}: a contrast compares one VLA's replicates; name results of one "
                            f"backbone (`qvla paper reeval --list` prints each one's backbone)")


def resolve(refs: Iterable[str], opener: Callable[[], object], cells: Cells, table1: Mapping[str, object],
            keys: set[int] | None, backbones: set[str] | None = None) -> dict[str, Resolved]:
    """Every distinct reference in `refs`, read through the Source `opener()` returns (an
    axis_qvla.paper.fetch.Source) with the sha256 the map records. Every reference is resolved before the
    source is opened, so a mistyped one is refused before anything is fetched."""
    picked = {ref: select(ref, cells, table1, keys, backbones) for ref in dict.fromkeys(refs)}
    _one_backbone(picked)
    source = opener()
    return {ref: Resolved(ref, members, tuple(_read(source, c) for c in members)) for ref, members in picked.items()}


def describe(r: Resolved) -> str:
    """One line naming what a reference resolved to, by run and seed (no digests)."""
    return f"{r.ref} = " + ", ".join(f"{c.run} (seed {_key(c)})" for c in r.members)
