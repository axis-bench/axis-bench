"""Filtering built-ins: drop-top-50 and its size-matched random-50 controls.

`DropTop50` keeps the top `keep` share of trainable rows ranked by the imported weight (per row for
chunk weights; episode-constant weights rank whole episodes), ties in corpus order
(axis_qvla.signals.blocks.schedule.top_fraction_mask), and draws the schedule over the kept rows.
`Random50` is its control: as many rows, chosen at random -- uniformly over rows for the chunk
control (the subset seeded `control_seed`; the paper's chunk control was also drawn with that seed,
so it compiles with seed 42), or whole episodes in a random order for the episode control
(synthetic per-episode weights U(1, 2) seeded `control_seed`, ranked like DropTop50) -- so the two
differ only in which rows. Either control's schedule is drawn with the bundle's seed. A DropTop50
that names its control also names the seed that control compiles with (`control_seed`, no default).

Offline tier: numpy.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np

from axis_qvla.signals.blocks.awr import DEFAULT_CAP, DEFAULT_HORIZON, DEFAULT_TAU
from axis_qvla.signals.blocks.schedule import DEFAULT_KEEP_TOP_FRAC, random_fraction_mask, top_fraction_mask
from axis_qvla.signals.compile import compile_program
from axis_qvla.signals.exports import legacy_exports
from axis_qvla.signals.imported import ImportedScores
from axis_qvla.signals.methods.random50 import SEED_CONTROL, synthetic_weights_doc
from axis_qvla.signals.methods.schedule import episode_stats_from_payload
from axis_qvla.signals.planners import SchedulePlanner
from axis_qvla.signals.program import Compiled, EpisodeValues, Selection
from axis_qvla.signals.rowspace import episode_rows


def stats_of(values: EpisodeValues, source: str) -> dict:
    """The schedule builder's per-episode statistics of AWR-shaped weights."""
    return episode_stats_from_payload({**values.header, "weights": values.values}, tau=DEFAULT_TAU,
                                      cap=DEFAULT_CAP, horizon=DEFAULT_HORIZON, source=source)


def check_rows(header, rows) -> None:
    declared = header.get("n_rows")
    if declared is not None and int(declared) != rows.n_rows:
        raise ValueError(f"the weights describe {int(declared)} pretraining frames but frame index {rows.name} has "
                         f"{rows.n_rows}; they were built against different pretraining data")


def _select(values, rows, masks_of, source) -> Selection:
    blocks, _progress, _unlab, n_trainable, _n_rows, row_w = episode_rows(rows, stats_of(values, source), drop=False)
    masks = masks_of(row_w)
    return Selection(tuple(b[m] for b, m in zip(blocks, masks)), tuple(w[m] for w, m in zip(row_w, masks)), n_trainable)


@dataclasses.dataclass(frozen=True)
class TopFraction:
    """Selector: the top `keep` of trainable rows by value, ties in corpus order."""

    keep: float = DEFAULT_KEEP_TOP_FRAC

    def select(self, values, rows) -> Selection:
        return _select(values, rows, lambda row_w: top_fraction_mask(row_w, self.keep), "the ranked values")

    def control(self):
        return RandomFraction(self.keep, SEED_CONTROL)


@dataclasses.dataclass(frozen=True)
class RandomFraction:
    """Selector: a uniformly random `keep` of trainable rows -- the count TopFraction keeps."""

    keep: float = DEFAULT_KEEP_TOP_FRAC
    seed: int = SEED_CONTROL

    def select(self, values, rows) -> Selection:
        return _select(values, rows, lambda row_w: random_fraction_mask(row_w, self.keep, self.seed), "the values")

    def control(self):
        return None


def _diagnostics(plan, selection, budget) -> dict:
    return {"keep_fraction": selection.kept / max(selection.n_trainable, 1),
            "unique_frames": float(np.unique(plan).size), "epochs": budget.samples / max(selection.kept, 1)}


@dataclasses.dataclass(frozen=True, eq=False)
class Random50:
    granularity: str                  # "chunk": random rows; "episode": whole episodes in random order
    template: ImportedScores           # the weights file whose keys and lengths the control copies
    keep: float = DEFAULT_KEEP_TOP_FRAC
    control_seed: int = SEED_CONTROL
    template_name: str | None = None
    exports: tuple = ()

    def __post_init__(self) -> None:
        if self.granularity not in ("chunk", "episode"):
            raise ValueError(f"granularity must be 'chunk' or 'episode', got {self.granularity!r}")

    def _name(self) -> str:
        return self.template_name or self.template.name

    def spec(self) -> dict:
        return {"method": "random50", "version": 1,
                "params": {"granularity": self.granularity, "keep": float(self.keep),
                           "control_seed": int(self.control_seed), "template_sha256": self.template.sha256,
                           "template_name": self._name(), "planner": "numpy-schedule"}}

    def run(self, labels, rows, budget, seed) -> Compiled:
        episode = self.granularity == "episode"
        reward_id = "rand50_ep" if episode else "rand50_row"
        doc = synthetic_weights_doc(self.template.payload, template_sha256=self.template.sha256,
                                    template_name=self._name(), reward_id=reward_id,
                                    per_episode_random=episode, seed=self.control_seed)
        synth = ImportedScores(doc, hashlib.sha256(json.dumps(doc).encode()).hexdigest(), f"{reward_id}_weights.json")
        check_rows(synth.header, rows)
        values = synth.values()
        selector = TopFraction(self.keep) if episode else RandomFraction(self.keep, self.control_seed)
        selection = selector.select(values, rows)
        mode = "drop_top" if episode else "drop_random"
        planner = SchedulePlanner(mode)
        plan = planner.plan(rows, selection, values, budget, seed)
        legacy, arrays = legacy_exports(self.exports, rows, schedule=dict(
            plan=plan, selection=selection, mode=mode, keep=self.keep, budget=budget, seed=seed, header=synth.header,
            weights_sha256=synth.sha256))
        return Compiled(plan=plan, planner=planner, produced={"plan": ("SchedulePlanner",)}, selection=selection,
                        diagnostics=_diagnostics(plan, selection, budget), inputs={self._name(): self.template.sha256},
                        legacy=legacy, legacy_arrays=arrays)

    def compile(self, labels, rows, budget, seed):
        return compile_program(self, labels, rows, budget, seed)[0]


@dataclasses.dataclass(frozen=True, eq=False)
class DropTop50:
    keep: float = DEFAULT_KEEP_TOP_FRAC
    control: Random50 | None = None
    control_seed: int | None = None    # the seed the control bundle is compiled with; required with a control
    exports: tuple = ()

    def __post_init__(self) -> None:
        if self.control is not None and self.control_seed is None:
            raise ValueError("a control needs control_seed, the seed its compiled labels are compiled with (the paper's "
                             "chunk control: 42); a default would build a different control under the same name")
        if self.control is None and self.control_seed is not None:
            raise ValueError("control_seed seeds the control's compiled labels, and there is no control to seed")

    def spec(self) -> dict:
        control = None if self.control is None else {**self.control.spec(), "seed": int(self.control_seed)}
        return {"method": "drop_top50", "version": 1,
                "params": {"keep": float(self.keep), "planner": "numpy-schedule", "control": control}}

    def run(self, labels: ImportedScores, rows, budget, seed) -> Compiled:
        check_rows(labels.header, rows)
        if labels.header.get("reward_id") is None:
            raise ValueError(f"{labels.name} names no reward_id; a schedule could not say which run it was drawn from")
        values = labels.values()
        selection = TopFraction(self.keep).select(values, rows)
        planner = SchedulePlanner("drop_top")
        plan = planner.plan(rows, selection, values, budget, seed)
        legacy, arrays = legacy_exports(self.exports, rows, schedule=dict(
            plan=plan, selection=selection, mode="drop_top", keep=self.keep, budget=budget, seed=seed,
            header=labels.header, weights_sha256=labels.sha256))
        controls = ((self.control, int(self.control_seed), None),) if self.control is not None else ()
        return Compiled(plan=plan, planner=planner, produced={"plan": ("SchedulePlanner",)}, selection=selection,
                        diagnostics=_diagnostics(plan, selection, budget), inputs={labels.name: labels.sha256},
                        legacy=legacy, legacy_arrays=arrays, controls=controls)

    def compile(self, labels, rows, budget, seed):
        return compile_program(self, labels, rows, budget, seed)[0]
