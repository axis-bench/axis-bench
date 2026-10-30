"""The quality-method API: a Program compiles labels over a row space into a SignalBundle.

A method is a `Program` -- `compile(labels, rows, budget, seed) -> SignalBundle` -- usually built
from blocks (axis_qvla.signals.compile.Compose). A Scorer turns labels into per-frame scores and raises
`Unscorable` for an episode it cannot score: it never invents a neutral score, the program declares
what happens then. A Reducer turns scores into the values decisions are made on, by ranking rather
than thresholding (a reward curve has large atoms, and a threshold lands on one). A Selector keeps
rows and can name the size-matched random control its effect is measured against. A Planner
draws the rows of every step; a program without a Selector is coverage-neutral (its plan is BC's)
unless it declares `coverage_neutral = False`. A Weighter turns values into per-row loss weights; a Conditioner
turns values into prompt codes. `run` returns the blocks' outputs (`Compiled`);
axis_qvla.signals.compile.compile_program assembles the bundle and asserts the composition rules.

Offline tier: numpy.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np

from axis_qvla.signals.schema import Budget, CondVocab, LegacyFile, RowSpace, ServeCondition, SignalBundle

MAX_CONTROL_DEPTH = 2   # a control may name its own control; a third level is refused


class Unscorable(Exception):
    """A scorer cannot score this episode. Raised, never returned as a neutral score."""

    def __init__(self, key: str, reason: str):
        super().__init__(f"{key}: {reason}")
        self.key, self.reason = key, reason


class RefusedDtype(TypeError):
    """A scorer's own dtype check refused a corpus column: the column holds data of the wrong
    kind for what the scorer reads from it. Raised only at that check, so a caller that means to
    report refused inputs by name, not a bug's traceback, can catch this type without also
    catching an unrelated TypeError."""


@dataclasses.dataclass(frozen=True, eq=False)
class EpisodeValues:
    """One array per corpus episode (its full length, every frame), keyed `task_<id>--<episode>`."""

    values: Mapping[str, np.ndarray]
    header: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True, eq=False)
class Selection:
    """What a Selector kept: per trainable episode (row-space walk order) the kept rows and their
    values, 1:1; `n_trainable` counts the trainable rows before selection."""

    blocks: tuple
    row_w: tuple
    n_trainable: int

    @property
    def kept(self) -> int:
        return int(sum(len(b) for b in self.blocks))


@dataclasses.dataclass(frozen=True, eq=False)
class Conditioning:
    """A Conditioner's output: codes over every corpus row (uint8 [n_rows, fields], FORBIDDEN off
    the trainable set), the vocabulary that renders them, and the serve conditions."""

    dense: np.ndarray
    vocab: CondVocab
    serve: tuple[ServeCondition, ...]
    meta: Mapping[str, Any] = dataclasses.field(default_factory=dict)


class Scorer(Protocol):
    def score(self, labels: Any, rows: RowSpace) -> EpisodeValues: ...


class Reducer(Protocol):
    def reduce(self, scores: EpisodeValues, rows: RowSpace) -> EpisodeValues: ...


class Selector(Protocol):
    def select(self, values: EpisodeValues, rows: RowSpace) -> Selection: ...

    def control(self) -> "Selector | None":
        """The size-matched random control (it keeps as many rows as this selector), which Compose
        compiles with the program, or None when there is none (a random control is its own end).
        Controls nest at most MAX_CONTROL_DEPTH levels."""
        ...


class Planner(Protocol):
    convention: str

    def plan(self, rows: RowSpace, selection: Selection | None, values: EpisodeValues | None,
             budget: Budget, seed: int) -> np.ndarray: ...


class Weighter(Protocol):
    def weigh(self, values: EpisodeValues, rows: RowSpace) -> EpisodeValues: ...


class Conditioner(Protocol):
    def condition(self, values: EpisodeValues, rows: RowSpace) -> Conditioning: ...


@dataclasses.dataclass(frozen=True, eq=False)
class Compiled:
    """Everything a program's blocks produced, before the compiler checks and assembles it.
    `produced` names the producing block of each output ("plan", "weights", "cond");
    `controls` is ((program, seed, labels), ...)."""

    plan: np.ndarray
    planner: Any
    produced: Mapping[str, tuple[str, ...]]
    selection: Selection | None = None
    weights: EpisodeValues | None = None
    reducer: str = "mean"
    conditioning: Conditioning | None = None
    diagnostics: Mapping[str, float] = dataclasses.field(default_factory=dict)
    inputs: Mapping[str, str] = dataclasses.field(default_factory=dict)
    legacy: tuple[LegacyFile, ...] = ()
    legacy_arrays: Mapping[str, np.ndarray] = dataclasses.field(default_factory=dict)
    controls: tuple = ()


class Program(Protocol):
    """`spec()` is the declared identity (method, version, params) hashed into `program_sha`;
    `run()` returns the blocks' outputs; `compile()` returns the checked bundle. A program whose
    plan differs from BC's without a Selector says so with a `coverage_neutral = False` attribute;
    without it the compiler holds the plan to BC's."""

    def spec(self) -> Mapping[str, Any]: ...

    def run(self, labels: Any, rows: RowSpace, budget: Budget, seed: int) -> Compiled: ...

    def compile(self, labels: Any, rows: RowSpace, budget: Budget, seed: int) -> SignalBundle: ...
