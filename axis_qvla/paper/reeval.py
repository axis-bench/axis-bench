"""`qvla paper reeval`: one published result, Table I or one figure, re-evaluated from the released
checkpoints -- which results it needs and the `qvla` commands that measure them.

Why: re-evaluating a paper number meant copying a cells.json `eval_command`, filling four
placeholders by hand and looking up its seeds. The unit a reader names is what the
paper shows -- a result, Table I or a figure, optionally one backbone's part of it -- and the paper
layout (experiments/paper/paper.json) already says which results each one reads. The commands are
the ones a user would type: `qvla data get` for the benchmark data, for what each backbone's
reference server reads besides the checkpoint, and for each checkpoint once; then one `qvla bench
eval --checkpoint-id` per result, under the run name the result was published under and with the
seeds the results map records. axis_qvla.paper.reeval_run runs them; axis_qvla.paper.reeval_tables lays the
user's results out as the paper does. Nothing here reads a published value.

Core tier: stdlib only.
"""
from __future__ import annotations

import dataclasses
import shlex
from collections.abc import Sequence

from axis_qvla.paper.cells import CellEntry, Cells
from axis_qvla.paper.spec import PaperSpec

BACKBONES = ("pi05", "smolvla", "xvla")
# The units a user names, by what each shows, and the paper layout's key for it (experiments/paper/
# paper.json; the published file names under figures/ and tables/ keep these keys). Figure and table
# numbers may change in the camera-ready, so no user-facing name carries one.
TABLE_UNITS = {"main-results": "table1"}          # each method relative to plain BC, per VLA
FIGURE_UNITS = {"headline": "fig1",               # the headline result
                "random-vs-filtered": "fig4c",    # random selection vs approach-aware filtering, half the data
                "methods": "fig5",                # weighting, top-50% filtering and conditioning
                "scaling": "fig6",                # scaling with pretraining data size
                "conditioning": "fig7"}           # conditioning stage and inference context
TABLES = tuple(TABLE_UNITS)
# The layout keys, still accepted (not listed) where a unit name is.
LEGACY_UNITS = {v: k for k, v in {**TABLE_UNITS, **FIGURE_UNITS}.items()}
FIG1 = "fig1"


class ReevalError(ValueError):
    """The unit names nothing in the paper layout or the results map."""


@dataclasses.dataclass(frozen=True)
class Unit:
    kind: str                     # "result" | "table" | "figure"
    name: str                     # a table or figure as the user names it (TABLE_UNITS, FIGURE_UNITS)
    backbone: str | None = None

    def __post_init__(self) -> None:
        if self.kind in ("table", "figure") and self.name in LEGACY_UNITS:
            object.__setattr__(self, "name", LEGACY_UNITS[self.name])

    @property
    def key(self) -> str:
        """The paper layout's key for a table or figure (`table1`, `fig4c`, ...): what the layout and
        the published file names use. A result's key is its name."""
        return {**TABLE_UNITS, **FIGURE_UNITS}.get(self.name, self.name)

    def flags(self) -> list[str]:
        """This unit as command-line flags, for the `--collect` line `--print` ends with."""
        return [f"--{self.kind}", self.name] + (["--backbone", self.backbone] if self.backbone else [])

    def slug(self) -> str:
        """A directory name for this unit's tables (`bc|bare` -> `bc-bare`)."""
        return self.name.replace("|", "-") + (f"-{self.backbone}" if self.backbone else "")


@dataclasses.dataclass(frozen=True)
class Step:
    argv: tuple[str, ...]         # a qvla command line, without the program name
    runs: tuple[str, ...] = ()    # the results' runs it serves; it is done once each has its results file


def figure_names(spec: PaperSpec) -> tuple[str, ...]:
    """The figures a user names, in layout order."""
    return tuple(LEGACY_UNITS.get(k, k) for k in (FIG1, *(f.key for f in spec.figures)))


def layout_cells(spec: PaperSpec, unit: Unit) -> list[str]:
    """The results-map cells a table or a figure reads, in layout order, for one backbone or all. A
    figure's bars are relative to their VLA's plain BC, so that cell comes with them."""
    if unit.kind not in ("table", "figure"):
        raise ReevalError(f"a {unit.kind!r} unit has no paper layout: only a table or a figure does")
    keep = [v for v in spec.vlas if unit.backbone in (None, v.key)]
    if unit.kind == "table":
        if unit.name not in TABLES:
            raise ReevalError(f"--table {unit.name!r}: the tables are {', '.join(TABLES)}")
        out = [c for v in keep for r in spec.table1["replicates"][v.key] for c in r["cells"].values()]
    elif unit.name not in figure_names(spec):
        raise ReevalError(f"--figure {unit.name!r}: the figures are {', '.join(figure_names(spec))}")
    elif unit.key == FIG1:
        row = next(r for r in spec.table1["rows"] if r["id"] == spec.fig1["row"])
        out = [r["cells"][role] for v in keep for r in spec.table1["replicates"][v.key]
               for role in (row["a"], row["b"], "bc")]
    else:
        fig = next(f for f in spec.figures if f.key == unit.key)
        keys = {v.key for v in keep}
        out = [c for b in fig.bars if b.vla in keys for c in (b.cell, spec.vla(b.vla).bc)]
    return list(dict.fromkeys(out))


def select(cells: Cells, spec: PaperSpec, unit: Unit) -> tuple[CellEntry, ...]:
    """The results-map entries `unit` re-evaluates, in results-map order."""
    if unit.backbone is not None and unit.backbone not in BACKBONES:
        raise ReevalError(f"--backbone {unit.backbone!r}: one of {', '.join(BACKBONES)}")
    if unit.kind == "result":
        hit = next((c for c in cells.cells if unit.name in (c.cell, c.run)), None)
        if hit is None:
            raise ReevalError(f"--result {unit.name!r} is no result or run of the results map; "
                              f"`qvla paper reeval --list` lists them")
        if unit.backbone not in (None, hit.backbone):
            raise ReevalError(f"--result {unit.name!r} is a {hit.backbone} result, not {unit.backbone}")
        return (hit,)
    want = layout_cells(spec, unit)
    sel = tuple(c for c in cells.cells if c.cell in set(want))
    missing = sorted(set(want) - {c.cell for c in sel})
    if missing:
        raise ReevalError(f"the paper layout reads {', '.join(missing)}, which the results map does not hold")
    if not sel:
        raise ReevalError(f"{unit.kind} {unit.name} reads no {unit.backbone} result")
    return sel


def _flag(tokens: Sequence[str], flag: str) -> str | None:
    return tokens[tokens.index(flag) + 1] if flag in tokens[:-1] else None


def eval_argv(c: CellEntry) -> tuple[str, ...]:
    """`bench eval` for one result: its checkpoint id, the serving (and any guidance) its recorded
    command uses, the run it was published under -- so its files land under the published names --
    and the training seeds the results map records. A recorded command without --serving is refused,
    never read as bare: a conditioned result evaluated bare would be filed under its name."""
    tokens = shlex.split(c.eval_command)
    serving = _flag(tokens, "--serving")
    if serving is None:
        raise ReevalError(f"{c.cell}: its eval_command in the results map names no --serving")
    argv = ["bench", "eval", "--checkpoint-id", str(c.checkpoint["id"]),
            "--serving", serving, "--run", c.run]
    guidance = _flag(tokens, "--guidance-scale")
    if guidance is not None:
        argv += ["--guidance-scale", guidance]
    if c.training_seed:
        argv += ["--ft-seed", str(c.training_seed["ft"])]
        if c.training_seed["pretrain"] is not None:
            argv += ["--pretrain-seed", str(c.training_seed["pretrain"])]
    return tuple(argv)


def plan(sel: Sequence[CellEntry], *, host: str | None = None) -> list[Step]:
    """Every command, in order: the benchmark data; what each backbone's reference server reads
    besides the checkpoint (`data get base-weights --serving`); each checkpoint once (one served
    bare and conditioned is two results); each evaluation."""
    h = ("--host", host) if host else ()
    runs = tuple(c.run for c in sel)
    steps = [Step(("data", "get", "bench", *h), runs)]
    for bb in dict.fromkeys(c.backbone for c in sel):
        steps.append(Step(("data", "get", "base-weights", "--serving", bb, *h),
                          tuple(c.run for c in sel if c.backbone == bb)))
    by_ckpt: dict[str, list[str]] = {}
    for c in sel:
        by_ckpt.setdefault(str(c.checkpoint["id"]), []).append(c.run)
    steps += [Step(("data", "get", f"checkpoints/{cid}", *h), tuple(rs)) for cid, rs in by_ckpt.items()]
    steps += [Step((*eval_argv(c), *h), (c.run,)) for c in sel]
    return steps


def uses(spec: PaperSpec) -> dict[str, list[str]]:
    """cell -> the tables and figures that read it, for `--list`."""
    out: dict[str, list[str]] = {}
    for unit in [Unit("table", t) for t in TABLES] + [Unit("figure", f) for f in figure_names(spec)]:
        for cell in layout_cells(spec, unit):
            out.setdefault(cell, []).append(unit.name)
    return out
