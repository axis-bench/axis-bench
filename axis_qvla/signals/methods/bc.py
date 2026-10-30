"""BC: every trainable row, drawn by the backbone's own sampler.

Its plan is the one every coverage-neutral method (AWR, CFG) must reproduce byte for byte. Its
legacy exports are the row space's index files (openpi) or row plan (LeRobot).

Offline tier: numpy.
"""
from __future__ import annotations

import dataclasses

from axis_qvla.signals.compile import compile_program
from axis_qvla.signals.exports import legacy_exports
from axis_qvla.signals.planners import PermutationPlanner
from axis_qvla.signals.program import Compiled


@dataclasses.dataclass(frozen=True, eq=False)
class BC:
    convention: str
    exports: tuple = ()

    def spec(self) -> dict:
        return {"method": "bc", "version": 1, "params": {"planner": self.convention}}

    def run(self, labels, rows, budget, seed) -> Compiled:
        planner = PermutationPlanner(self.convention)
        legacy, arrays = legacy_exports(self.exports, rows)
        return Compiled(plan=planner.plan(rows, None, None, budget, seed), planner=planner,
                        produced={"plan": ("PermutationPlanner",)}, legacy=legacy, legacy_arrays=arrays,
                        diagnostics={"epochs": budget.samples / rows.n_trainable})

    def compile(self, labels, rows, budget, seed):
        return compile_program(self, labels, rows, budget, seed)[0]
