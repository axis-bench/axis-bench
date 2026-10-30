"""Planners: the rows every optimizer step draws, as an int64 [steps, global_batch] array.

`PermutationPlanner` replays the backbones' own samplers in numpy. Epoch e is
`rows[torch.randperm(len(rows), generator=manual_seed(s_e))]` with `s_e = seed + e` (openpi's
RowSampler) or `seed * 100_003 + e` (the LeRobot fork's AxisRowSampler), and epochs follow one
another until the budget is drawn (`stream`). A plan is that stream cut into global batches the
way each loader batches it. openpi's loader drops an epoch's last partial batch (drop_last=True),
so each epoch contributes its first whole batches; LeRobot's keeps it as a short batch
(drop_last=False), which a [steps, batch] plan cannot hold, so a LeRobot plan that crosses an epoch
boundary needs an epoch length that is a multiple of the batch and is refused otherwise. Within
those rules the plan is the row sequence the loader yields. torch's CPU randperm, for n below
2**32 // 20, is a Fisher-Yates
shuffle on MT19937 32-bit outputs, `j = i + random() % (n - i)`; `torch_randperm` replays it from
numpy's MT19937 seeded the way torch seeds it (init_genrand), and matches torch 2.7.1 and 2.11.0
bit for bit (tests/unit/signals/test_planners.py). `SchedulePlanner` is the filtering schedules'
draw (axis_qvla.signals.blocks.schedule.draw_schedule): permutation passes over the kept rows under
numpy's default_rng(seed).

Offline tier: numpy.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from axis_qvla.signals.blocks.schedule import MODES, draw_schedule
from axis_qvla.signals.rowspace import trainable_rows

CONVENTIONS = {
    "openpi-rowsampler": lambda seed, epoch: seed + epoch,
    "lerobot-axisrowsampler": lambda seed, epoch: seed * 100_003 + epoch,
}
DROP_LAST = {"openpi-rowsampler": True, "lerobot-axisrowsampler": False}   # each loader's drop_last
_TORCH_FISHER_YATES_MAX = (2**32 - 1) // 20     # torch uses another algorithm from this size up


class PlanError(ValueError):
    """A plan cannot be drawn as asked."""


def mt19937_key(seed: int) -> np.ndarray:
    """MT19937 state for a 32-bit seed, exactly as std::mt19937 / torch's CPU generator set it."""
    if not 0 <= int(seed) < 2**32:
        raise PlanError(f"seed {seed} is not a 32-bit unsigned integer")
    key = np.empty(624, dtype=np.uint32)
    x = int(seed)
    key[0] = x
    for i in range(1, 624):
        x = (1812433253 * (x ^ (x >> 30)) + i) & 0xFFFFFFFF
        key[i] = x
    return key


def torch_randperm(n: int, seed: int) -> np.ndarray:
    """`torch.randperm(n, generator=torch.Generator().manual_seed(seed))`, as int64."""
    if not 0 <= int(n) < _TORCH_FISHER_YATES_MAX:
        raise PlanError(f"n={n} is outside the range torch shuffles with Fisher-Yates")
    key = mt19937_key(seed)
    r = list(range(int(n)))
    if n > 1:
        bg = np.random.MT19937(0)
        bg.state = {"bit_generator": "MT19937", "state": {"key": key, "pos": 624}}
        bound = int(n) - np.arange(int(n) - 1, dtype=np.uint64)
        z = (bg.random_raw(int(n) - 1).astype(np.uint64) % bound).tolist()
        for i in range(int(n) - 1):
            j = i + z[i]
            r[i], r[j] = r[j], r[i]
    return np.asarray(r, dtype=np.int64)


@dataclasses.dataclass(frozen=True)
class PermutationPlanner:
    convention: str

    def __post_init__(self) -> None:
        if self.convention not in CONVENTIONS:
            raise PlanError(f"convention must be one of {sorted(CONVENTIONS)}, got {self.convention!r}")

    def _draw(self, rows: np.ndarray, total: int, seed: int, per_epoch: int) -> np.ndarray:
        parts, need, epoch = [np.empty(0, dtype=np.int64)], int(total), 0
        while need > 0:
            perm = torch_randperm(rows.size, CONVENTIONS[self.convention](int(seed), epoch))
            take = perm[:min(per_epoch, need)]
            parts.append(rows[take])
            need -= take.size
            epoch += 1
        return np.concatenate(parts)

    def stream(self, rows, total: int, seed: int) -> np.ndarray:
        """The first `total` rows of the successive epoch permutations, unbatched."""
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            raise PlanError("no rows to draw from")
        return self._draw(rows, total, seed, rows.size)

    def plan(self, rows, selection, values, budget, seed) -> np.ndarray:
        flat = trainable_rows(rows) if selection is None else np.concatenate(
            [np.empty(0, dtype=np.int64)] + [np.asarray(b, dtype=np.int64) for b in selection.blocks])
        n, batch = int(flat.size), budget.global_batch
        if n == 0:
            raise PlanError("no rows to draw from")
        per_epoch = n
        if budget.samples > n and n % batch:
            if not DROP_LAST[self.convention]:
                raise PlanError(f"{self.convention}: {n} drawable rows are not a multiple of the global batch "
                                f"{batch} and the plan spans more than one epoch; the loader yields the short last "
                                f"batch of every epoch (drop_last=False), which a [steps, batch] plan cannot hold")
            per_epoch = (n // batch) * batch
            if per_epoch == 0:
                raise PlanError(f"{self.convention}: {n} drawable rows are fewer than one global batch of {batch}; "
                                f"the loader drops every batch (drop_last=True)")
        return self._draw(flat, budget.samples, seed, per_epoch).reshape(budget.steps, budget.global_batch)


@dataclasses.dataclass(frozen=True)
class SchedulePlanner:
    mode: str
    convention: str = "numpy-schedule"

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise PlanError(f"mode must be one of {MODES}, got {self.mode!r}")

    def plan(self, rows, selection, values, budget, seed) -> np.ndarray:
        if selection is None:
            raise PlanError("a schedule is drawn over a selection; pass the Selector's output")
        return draw_schedule([np.asarray(b, dtype=np.int64) for b in selection.blocks], mode=self.mode,
                             total_steps=budget.steps, batch=budget.global_batch, seed=int(seed))
