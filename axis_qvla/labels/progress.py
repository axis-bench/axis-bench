"""Which goal record of an attempt the reward of Eq. (1) measures progress toward.

The task checkers' record of one attempt (axis_qvla.labels.goals) lists one leaf per sub-checker. The
reward targets the leaf with a resolved goal pose, skipping stability monitors that hold from frame
0, preferring a reward goal family (pnp, position_delta, directed_rotation), and, among those, the
leaf whose first pass is closest to the whole task's first pass.

Stdlib only.
"""
from __future__ import annotations

AUX_STABILITY_TYPES = frozenset({
    "PositionDeltaChecker",
    "RelativePositionBoundsChecker",
})

REWARD_GOAL_TYPES = frozenset({"pnp", "position_delta", "directed_rotation"})


def _is_aux_stability_leaf(leaf: dict) -> bool:
    """A PositionDelta / RelativePositionBounds leaf that fires from frame 0
    is a stability-monitor sub-checker (e.g. "plate must not move"). Treat
    as auxiliary; the primary manipulation goal is elsewhere in the tree.
    """
    if leaf.get("checker_type") not in AUX_STABILITY_TYPES:
        return False
    pass_frame = leaf.get("pass_frame")
    n_pass = leaf.get("n_pass_frames", 0) or 0
    return pass_frame == 0 and n_pass > 0


def pick_reward_leaf(info: dict) -> dict | None:
    """Choose the leaf to use as the reward target for one attempt's goals.json.

    Selection strategy:
      1. Collect leaves with goal_xyz set (a resolved goal pose).
      2. Drop auxiliary stability checkers (see _is_aux_stability_leaf).
      3. Prefer leaves whose goal_type is a reward-mappable family
         (pnp / position_delta / directed_rotation).
      4. Among those, if `top_first_pass_frame` is set, pick the leaf whose
         pass_frame is closest to it (ties broken by preferring larger
         n_pass_frames for stability).
      5. Fallback: first eligible leaf in tree order.
      6. Final fallback: any leaf with goal_xyz.

    Returns the chosen leaf dict or None if no candidate qualifies.
    """
    leaves = info.get("leaves") or []
    if not leaves:
        return None

    candidates = [L for L in leaves if L.get("goal_xyz")]
    if not candidates:
        return None

    non_aux = [L for L in candidates if not _is_aux_stability_leaf(L)]
    typed = [L for L in non_aux if L.get("goal_type") in REWARD_GOAL_TYPES]
    pool = typed or non_aux or candidates

    top_pf = info.get("top_first_pass_frame")
    if top_pf is not None:
        with_pf = [L for L in pool if L.get("pass_frame") is not None]
        if with_pf:
            def _key(L: dict):
                pf = int(L["pass_frame"])
                return (abs(pf - int(top_pf)), -int(L.get("n_pass_frames") or 0))
            return min(with_pf, key=_key)

    # No alignment info → first candidate in tree order.
    return pool[0]
