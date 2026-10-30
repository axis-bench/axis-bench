"""How many actions of a policy's chunk to execute OPEN-LOOP before re-observing.

CORE-ADJACENT: stdlib only, deliberately. It lives apart from `rollout_dualsim` because that module
imports `axis_replay_native`, which exists only in the render venv — a policy knob that cannot be
unit-tested without a renderer is a knob nobody checks.

WHY THIS IS A KNOB AND NOT A CONSTANT. The rollout consumes `replan` steps of a 16-step chunk before
looking at the world again. What that costs depends on the ACTION SPACE: `eef7` commands per-step
EEF position deltas, so a longer open-loop window accumulates directional error unobserved, while
`droid8` commands joint velocities, which tolerate the same window well.

Lowering it multiplies server round-trips (replan=1 is 8x the inference calls). The default is 8,
the value every reported evaluation used.
"""

from __future__ import annotations

import os

ENV_VAR = "AXIS_EVAL_REPLAN"


def effective_replan(default: int) -> int:
    """`default`, unless AXIS_EVAL_REPLAN overrides it.

    Refuses a malformed or non-positive value rather than falling back: a silently-ignored typo
    would make an A/B on this knob report a null result that never ran.
    """
    raw = os.environ.get(ENV_VAR)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        raise ValueError(f"{ENV_VAR}={raw!r} is not an integer") from None
    if value < 1:
        raise ValueError(
            f"{ENV_VAR}={value} must be >= 1; 0 would consume no actions and re-infer forever "
            f"on the same observation"
        )
    return value
