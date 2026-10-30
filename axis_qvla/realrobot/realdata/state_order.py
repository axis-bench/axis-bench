"""The FR3 recording's state and action layouts, declared, and the adapter to the model's order.

WHY: the FR3 collection stack writes `state[8] = [gripper, q0..q6]` (its observation dict
concatenated in key order: gripper_position before joint_position), while a DROID-8D model reads
`[q0..q6, gripper]`. Both are 8 floats, so the wrong order trains and serves without an error and
only fails on the robot. The order is written down here once and every consumer permutes through
it; nothing infers it from the data.

Actions are `[v0..v6, gripper_cmd]`: `v` is DROID's normalised joint-velocity COMMAND in [-1, 1] at
15 Hz -- the controller's target is q + 0.2 * v per step -- not a velocity in rad/s, and `gripper_cmd`
is an absolute gripper position command in [0, 1] (0 = open). ACTION_UNITS declares this once; the FR3
line's unit and serve checks (axis_qvla.realrobot.fr3.checks) compare against it.

Layouts as the dataset's producers wrote and repacked them (credited in NOTICE).
Offline tier, numpy only.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np

JOINTS: tuple[str, ...] = tuple(f"q{i}" for i in range(7))
STORED_STATE: tuple[str, ...] = ("gripper", *JOINTS)
MODEL_STATE: tuple[str, ...] = (*JOINTS, "gripper")
STORED_ACTIONS: tuple[str, ...] = (*(f"v{i}" for i in range(7)), "gripper_cmd")
ACTION_UNITS = {"arm": "droid_normalized_joint_velocity_command", "arm_range": (-1.0, 1.0), "fps": 15,
                "controller": "joint target q + 0.2 * v per step", "gripper": "absolute_position_command_0_1"}


def select(values: object, layout: Sequence[str], names: Sequence[str]) -> np.ndarray:
    """The `names` columns of `values`, whose last axis is laid out as `layout`."""
    x = np.asarray(values)
    if x.ndim == 0 or x.shape[-1] != len(layout):
        raise ValueError(f"expected (..., {len(layout)}) values laid out as {tuple(layout)}, got shape {x.shape}")
    missing = [n for n in names if n not in layout]
    if missing:
        raise KeyError(f"{missing} not in {tuple(layout)}")
    return x[..., [list(layout).index(n) for n in names]]


@dataclasses.dataclass(frozen=True)
class StateOrder:
    """A declared reordering of the same named quantities: `source` order in, `target` order out."""

    source: tuple[str, ...]
    target: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(set(self.source)) != len(self.source) or sorted(self.source) != sorted(self.target):
            raise ValueError(f"{self.target} is not a permutation of {self.source}")

    def __call__(self, values: object) -> np.ndarray:
        return select(values, self.source, self.target)

    def inverse(self) -> StateOrder:
        return StateOrder(self.target, self.source)


STORED_TO_MODEL = StateOrder(STORED_STATE, MODEL_STATE)
MODEL_TO_STORED = STORED_TO_MODEL.inverse()
