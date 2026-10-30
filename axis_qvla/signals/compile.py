"""Assemble a SignalBundle from a program's outputs, and assert the composition rules.

Each rule is a CompositionError. Every output has exactly one producing block (the plan always one,
weights and codes at most one). The plan has the budget's shape and draws only trainable rows, and
under a Selector only the rows it kept. A coverage-neutral program (no Selector) draws BC's plan
byte for byte: the canonical PermutationPlanner of its planner's convention over every trainable
row, same seed and budget. A program whose plan is meant to differ declares `coverage_neutral =
False`, and one whose planner has no canonical permutation must. Weights are finite on every
trainable row and come with a loss reducer (global_ratio or per_rank_ratio); no weights means
`mean`. Codes have one column per vocabulary field, never the forbidden code on a drawn row, and
only codes the vocabulary renders. Every serve condition names declared fields and drawable codes
(the schema's rule), and a guided one needs a drawn row coded exactly as its unconditional pass
renders -- each field at its `uncond_fields` code, every other field dropped -- because that branch
must have been trained. A program may name controls: Compose emits its Selector's own `control()`,
and a program may name one itself (the drop-top arm names its random-50 control) or none (a random
control compiles on its own). Every control present keeps as many rows as its program (size-matched),
is compiled the same way, recorded by id and returned with its own controls; controls nest at most
MAX_CONTROL_DEPTH levels. `requires` is derived from what the bundle carries.
Per-episode weights, selections and codes need a corpus row space; over a flat one they are refused.

Offline tier: numpy.
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from axis_qvla.core.canon import dumps_canonical, sha256_bytes
from axis_qvla.signals.bundle import bundle_id
from axis_qvla.signals.lerobot_files import lerobot_files
from axis_qvla.signals.planners import CONVENTIONS, PermutationPlanner, PlanError
from axis_qvla.signals.program import MAX_CONTROL_DEPTH, Compiled, EpisodeValues, Selection
from axis_qvla.signals.rowspace import RowSpaceError, trainable_blocks, trainable_rows
from axis_qvla.signals.schema import (BARE, FORBIDDEN, NO_CODE, REDUCERS, Budget, CondVocab, RowSpace, SchemaError,
                                 ServeCondition, SignalBundle, check_serve)


class CompositionError(ValueError):
    """A program's outputs break a composition rule; nothing is compiled from them."""


def flat_weights(rows: RowSpace, values: EpisodeValues) -> np.ndarray:
    """One float32 per corpus row: the episode's weight on a trainable frame, NaN elsewhere (the
    LeRobot flat-weights layout, axis_qvla/signals/views/lerobot.py `flat_weights`)."""
    if rows.kind != "corpus":
        raise CompositionError(f"{rows.name} is a flat frame index: per-episode weights need a pretraining-data frame index")
    out = np.full(rows.n_rows, np.nan, dtype=np.float32)
    for key, global_rows, frames, length in trainable_blocks(rows):
        w = values.values.get(key)
        if w is None:
            raise CompositionError(f"{key} has trainable rows but no weight")
        w = np.asarray(w, dtype=np.float64)
        if w.size != length:
            raise CompositionError(f"{key}: {w.size} weights for a {length}-frame episode")
        if not np.isfinite(w[frames]).all():
            raise CompositionError(f"{key}: a non-finite weight on a trainable frame")
        out[global_rows] = w[frames].astype(np.float32)
    return out


def _kept_rows(selection: Selection) -> np.ndarray:
    return np.concatenate([np.empty(0, dtype=np.int64)] + [np.asarray(b, dtype=np.int64) for b in selection.blocks])


def _check_plan(c: Compiled, rows: RowSpace, budget: Budget, seed: int, coverage_neutral: bool) -> None:
    plan = np.asarray(c.plan)
    if plan.dtype != np.int64 or plan.shape != (budget.steps, budget.global_batch):
        raise CompositionError(f"the plan is {plan.dtype} {plan.shape}, not int64 {(budget.steps, budget.global_batch)}")
    if not np.isin(plan, trainable_rows(rows)).all():
        raise CompositionError("the plan draws a row outside the trainable set")
    if c.selection is not None:
        if not np.isin(plan, _kept_rows(c.selection)).all():
            raise CompositionError("the plan draws a row outside the selection's kept rows")
        return
    convention = getattr(c.planner, "convention", None)
    if not coverage_neutral:
        return
    if convention in CONVENTIONS:
        try:
            bc = PermutationPlanner(convention).plan(rows, None, None, budget, seed)
        except PlanError as exc:
            raise CompositionError(f"BC's plan cannot be drawn to compare against: {exc}") from exc
        if not np.array_equal(bc, plan):
            raise CompositionError(f"a coverage-neutral program must draw exactly BC's plan: PermutationPlanner"
                                   f"({convention!r}) over every trainable row, same seed and budget")
    else:
        raise CompositionError(f"planner convention {convention!r} has no canonical BC permutation: a program "
                               f"without a Selector that plans this way declares coverage_neutral = False")


def _check_serve(vocab: CondVocab, serve: tuple[ServeCondition, ...], cond: np.ndarray, requires: set) -> None:
    try:
        check_serve(vocab, serve)
    except SchemaError as exc:
        raise CompositionError(f"serve condition refused: {exc}") from exc
    drawn = cond.reshape(-1, len(vocab.fields))
    for s in serve:
        if s.guided:
            requires.add("guidance")
            want = [int(s.uncond_fields.get(f, NO_CODE)) for f in vocab.fields]
            if not (drawn == np.asarray(want, dtype=np.int64)).all(axis=1).any():
                raise CompositionError(f"a guided serve condition needs its unconditional branch trained: no drawn "
                                       f"row carries the unconditional pass's codes {want} (fields {vocab.fields})")


def check_composition(c: Compiled, rows: RowSpace, budget: Budget, seed: int, *, weights, cond,
                      coverage_neutral: bool = True) -> frozenset[str]:
    """Every rule of the module docstring that one program's outputs must meet; returns `requires`.
    `coverage_neutral=False` is the program's declaration that its Selector-less plan differs from BC's."""
    for field in ("plan", "weights", "cond"):
        producers = tuple(c.produced.get(field, ()))
        if len(producers) > 1:
            raise CompositionError(f"{field} is produced by {producers}: each output has exactly one producing block")
    if len(c.produced.get("plan", ())) != 1:
        raise CompositionError("the plan has no producing block")
    if (weights is None) == bool(c.produced.get("weights")) or (cond is None) == bool(c.produced.get("cond")):
        raise CompositionError("`produced` does not match the outputs present")
    _check_plan(c, rows, budget, seed, coverage_neutral)
    if weights is None:
        if c.reducer != "mean":
            raise CompositionError(f"reducer {c.reducer!r} without weights")
    elif c.reducer not in REDUCERS or c.reducer == "mean":
        raise CompositionError(f"a weighted program declares global_ratio or per_rank_ratio, not {c.reducer!r}")
    requires = {"plan"} | ({"loss_weights"} if weights is not None else set())
    if cond is not None:
        vocab = c.conditioning.vocab
        if cond.shape != (budget.steps, budget.global_batch, len(vocab.fields)):
            raise CompositionError(f"codes {cond.shape} do not match the plan and {len(vocab.fields)} fields")
        if (cond == FORBIDDEN).any():
            raise CompositionError("a drawn row carries the forbidden code")
        for j, f in enumerate(vocab.fields):
            bad = set(np.unique(cond[..., j]).tolist()) - {NO_CODE} - set(vocab.text[f])
            if bad:
                raise CompositionError(f"field {f!r}: codes {sorted(bad)} have no text")
        _check_serve(vocab, tuple(c.conditioning.serve), cond, requires)
        requires.add("cond_prompt")
    for k, v in c.diagnostics.items():
        try:
            finite = math.isfinite(float(v))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            raise CompositionError(f"diagnostic {k!r} is not a finite number: {v!r}")
    return frozenset(requires)


def _gather_codes(c: Compiled, rows: RowSpace):
    if c.conditioning is None:
        return None
    dense = np.asarray(c.conditioning.dense)
    if dense.dtype != np.uint8 or dense.ndim != 2 or dense.shape[0] != rows.n_rows:
        raise CompositionError(f"codes must be uint8 [{rows.n_rows}, fields], got {dense.dtype} {dense.shape}")
    plan = np.asarray(c.plan)
    # Gather only through a plan that indexes the row space: numpy would wrap a negative row and
    # raise IndexError past the end, neither of which is a composition rule's refusal.
    if not np.issubdtype(plan.dtype, np.integer):
        raise CompositionError(f"the plan is {plan.dtype}, not int64")
    if plan.size and (int(plan.min()) < 0 or int(plan.max()) >= rows.n_rows):
        raise CompositionError("the plan draws a row outside the trainable set")
    return dense[plan]


def _kept(selection: Selection | None, rows: RowSpace) -> int:
    return rows.n_trainable if selection is None else selection.kept


def _compile(program, labels, rows: RowSpace, budget: Budget, seed: int, depth: int):
    if depth > MAX_CONTROL_DEPTH:
        raise CompositionError(f"controls nest deeper than {MAX_CONTROL_DEPTH} levels; a control's selector "
                               f"ends the chain by returning no control")
    try:
        c = program.run(labels, rows, budget, seed)
        weights = None if c.weights is None else flat_weights(rows, c.weights)
        cond = _gather_codes(c, rows)
        requires = check_composition(c, rows, budget, seed, weights=weights, cond=cond,
                                     coverage_neutral=bool(getattr(program, "coverage_neutral", True)))
    except RowSpaceError as exc:
        if rows.kind != "flat":
            raise
        raise CompositionError(f"{rows.name} is a flat frame index, which has no episodes: {exc}") from exc
    direct, every = [], {}
    for p, s, lab in c.controls:
        b, nested, sel = _compile(p, lab, rows, budget, s, depth + 1)
        if _kept(sel, rows) != _kept(c.selection, rows):
            raise CompositionError(f"control {b.program.get('method')!r} keeps {_kept(sel, rows)} rows and the "
                                   f"selection {_kept(c.selection, rows)}: a control is size-matched")
        bid = bundle_id(b)
        direct.append(bid)
        every.setdefault(bid, b)
        for x in nested:
            every.setdefault(bundle_id(x), x)
    spec = dict(program.spec())
    bundle = SignalBundle(
        program=spec, program_sha=sha256_bytes(dumps_canonical(spec)), rowspace=rows, budget=budget, seed=int(seed),
        plan=np.asarray(c.plan, dtype=np.int64), weights=weights, reducer=c.reducer, cond=cond,
        vocab=None if c.conditioning is None else c.conditioning.vocab,
        serve=(BARE,) if c.conditioning is None else tuple(c.conditioning.serve),
        controls=tuple(direct), requires=requires,
        diagnostics={k: float(v) for k, v in c.diagnostics.items()}, inputs=dict(c.inputs),
        legacy=tuple(c.legacy), legacy_arrays=dict(c.legacy_arrays))
    return bundle, tuple(every.values()), c.selection


def compile_program(program, labels, rows: RowSpace, budget: Budget, seed: int
                    ) -> tuple[SignalBundle, tuple[SignalBundle, ...]]:
    """Run `program`, check the composition rules, and return its bundle and every control's bundle
    (its controls first, each followed by its own; `bundle.controls` names the direct ones)."""
    bundle, controls, _ = _compile(program, labels, rows, budget, seed, 0)
    return bundle, controls


@dataclasses.dataclass(frozen=True)
class Compose:
    """A Program built from blocks, for method authors. Its compiled labels carry the LeRobot files the
    pinned LeRobot fork trains them from, where that fork can (axis_qvla.signals.lerobot_files); the
    built-ins orchestrate the same blocks themselves because they write the paper's locked files
    instead. `coverage_neutral=False` declares a Selector-less plan that differs from BC's (it is
    then part of `spec()`)."""

    method: str
    version: int
    params: Mapping[str, Any]
    planner: Any
    scorer: Any = None
    reducer: Any = None
    selector: Any = None
    weighter: Any = None
    conditioner: Any = None
    loss_reducer: str = "mean"
    coverage_neutral: bool = True

    def spec(self) -> dict:
        spec = {"method": self.method, "version": int(self.version), "params": dict(self.params)}
        if not self.coverage_neutral:
            spec["coverage_neutral"] = False
        return spec

    def run(self, labels, rows, budget, seed) -> Compiled:
        values = self.scorer.score(labels, rows) if self.scorer is not None else None
        if self.reducer is not None:
            values = self.reducer.reduce(values, rows)
        selection = self.selector.select(values, rows) if self.selector is not None else None
        plan = self.planner.plan(rows, selection, values, budget, seed)
        weights = self.weighter.weigh(values, rows) if self.weighter is not None else None
        conditioning = self.conditioner.condition(values, rows) if self.conditioner is not None else None
        produced = {"plan": (type(self.planner).__name__,)}
        if weights is not None:
            produced["weights"] = (type(self.weighter).__name__,)
        if conditioning is not None:
            produced["cond"] = (type(self.conditioner).__name__,)
        # One level per run: the compiler compiles the control program, which names its own
        # control in turn, and bounds the chain at MAX_CONTROL_DEPTH.
        control = self.selector.control() if self.selector is not None else None
        controls = (((dataclasses.replace(self, method=f"{self.method}.control", selector=control), int(seed), labels),)
                    if control is not None else ())
        drawn = selection.kept if selection is not None else rows.n_trainable
        legacy, arrays = lerobot_files(self.method, rows, budget, seed, plan=plan, planner=self.planner,
                                       selection=selection, weighted=weights is not None, conditioning=conditioning)
        return Compiled(plan=plan, planner=self.planner, produced=produced, selection=selection, weights=weights,
                        reducer=self.loss_reducer if weights is not None else "mean", conditioning=conditioning,
                        diagnostics={"epochs": budget.samples / max(drawn, 1)}, controls=controls,
                        legacy=legacy, legacy_arrays=arrays)

    def compile(self, labels, rows, budget, seed) -> SignalBundle:
        return compile_program(self, labels, rows, budget, seed)[0]
