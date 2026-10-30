"""The reward of Eq. (1) per trace step: task progress toward the checker's goal plus hand-object contact.

r = W_PLACE_V2 * progress + W_CONTACT_V2 * (1 - tanh(CONTACT_TANH_K * |hand - object|)). Progress is
specific to the goal family the task checkers recorded for the attempt (axis_qvla.labels.goals, chosen by
axis_qvla.labels.progress.pick_reward_leaf): `pnp` covers the initial object-to-goal distance,
`position_delta` the per-axis displacement toward the checker's bounds, `directed_rotation` the
rotation toward the target angle within the tilt limit. The curve is computed on the 5 Hz trace
clipped at the demonstration's success frame and resampled onto render rows by axis_qvla.labels.pipeline.

This is the arithmetic the published labels were computed with. Both trace-tail controls are
constants, at the values those labels were built with: the curve stops at the success frame
(CLIP_TAIL_FRAMES = 0) and carries no hold bonus (HOLD_STABILITY_BONUS = 0); no environment variable
changes them. A position-delta leaf without per-axis bounds takes them from the task's checker
configuration (`data/task_<id>/checker_config.json` beside the attempt's `traces/` directory); when
that file cannot supply them, AxisBoundsError names it and why.

Offline tier: numpy.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np


HAND_CANDIDATES = (
    "franka/panda_hand",
    "left_hand_link",
    "right_hand_link",
    "panda_hand",
    "tool0",
    "ee_link",
)

# The reward of Eq. (1), the same for every skill:
#   r = W_PLACE_V2 · r_progress + W_CONTACT_V2 · r_contact_shape
# Placement is progress-normalised task-state (goal-independent of initial
# distance). Contact is a bounded auxiliary that smoothly ramps from
# "hand approaching object" to a saturated "in contact" value; no cliff
# at any threshold and no explicit grasp state.
W_PLACE_V2 = 0.8
W_CONTACT_V2 = 0.2
CONTACT_TANH_K = 5.0            # tanh(k·d) — most gradient in the last ~20cm
PROGRESS_EPS_M = 0.005          # min initial distance to avoid /0 on start-at-goal
PLACE_SUCCESS_DIST = 0.05      # object within this dist of goal ==> placed
# The published labels clip the trace at the success frame and add no hold bonus.
CLIP_TAIL_FRAMES = 0
HOLD_STABILITY_BONUS = 0.0


OBJECT_KEYS = ("objName", "bodyName", "sampleBodyName")
TARGET_KEYS = ("refName",)


def _collect_key(node, keys: tuple[str, ...]) -> list[str]:
    """Return every value associated with any of `keys`, in traversal order."""
    out: list[str] = []
    if isinstance(node, dict):
        for k in keys:
            if k in node and isinstance(node[k], str):
                out.append(node[k])
        for v in node.values():
            out.extend(_collect_key(v, keys))
    elif isinstance(node, list):
        for v in node:
            out.extend(_collect_key(v, keys))
    return out


def _collect_obj_target_pairs(node) -> list[tuple[str, str]]:
    """Return (objName, refName) pairs from RelativeCylinderChecker-like nodes."""
    pairs: list[tuple[str, str]] = []
    if isinstance(node, dict):
        obj = node.get("objName")
        ref = node.get("refName")
        if isinstance(obj, str) and isinstance(ref, str):
            pairs.append((obj, ref))
        for v in node.values():
            pairs.extend(_collect_obj_target_pairs(v))
    elif isinstance(node, list):
        for v in node:
            pairs.extend(_collect_obj_target_pairs(v))
    return pairs


def _strip_index(body: str) -> str:
    return re.sub(r"_\d+$", "", body)


def _infer_object_body(checker: dict, all_bodies: list[str]) -> str | None:
    for b in _collect_key(checker, OBJECT_KEYS):
        if b in all_bodies:
            return b
    return None


def _infer_target_body(
    checker: dict,
    object_body: str,
    all_bodies: list[str],
    task_name: str,
) -> str | None:
    # Preferred: checker directly names (obj, ref) pairs.
    obj_stem = _strip_index(object_body)
    for obj, ref in _collect_obj_target_pairs(checker):
        if _strip_index(obj) == obj_stem and ref in all_bodies:
            return ref
    # Fallback: parse "on/beside/into/to/onto/in/near the X" from the task name.
    m = re.search(
        r"(?:on|onto|beside|into|to|in|near|next\s+to)\s+the\s+([A-Za-z][A-Za-z\s]+?)(?:\s|$|[,.])",
        task_name.lower(),
    )
    if not m:
        return None
    token = m.group(1).strip().replace(" ", "_")
    for b in all_bodies:
        stem = _strip_index(b).lower()
        if stem == _strip_index(object_body).lower():
            continue
        if token in stem or stem in token:
            return b
    return None


def _load_trace(path: Path) -> tuple[list[dict], list[str]]:
    steps: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("event") == "state":
            steps.append(entry)
    all_bodies = list(steps[0]["body_pose"].keys())
    return steps, all_bodies


# ---------------------------------------------------------------------------
# Per-axis bounds of a position-delta goal.
#
# A goal record's position-delta leaf (`goal_type: position_delta`) may carry
# `delta_axes` but no `axis_bounds`. Progress toward such a goal is per-axis,
# toward the checker's own bounds. The task's checker configuration
# (`data/task_<id>/checker_config.json`, beside the attempt's `traces/`
# directory) records the per-axis minDelta / maxDelta the PositionDeltaChecker
# was constructed with, so the bounds of a leaf without them are read from it;
# when it cannot supply them, AxisBoundsError names the file and why.
# ---------------------------------------------------------------------------

FN_RE_TRACE = re.compile(r"task_(\d+)_attempt_(\d+)\.jsonl$")


def _task_id_from_trace(trace_path: Path) -> int | None:
    m = FN_RE_TRACE.search(trace_path.name)
    return int(m.group(1)) if m else None


class AxisBoundsError(Exception):
    """A position-delta leaf lacks its per-axis bounds and the task's checker configuration cannot
    supply them. Deliberately not a ValueError: the pipeline labels an unformable reward "no reward"
    (axis_qvla.labels.pipeline), and an attempt whose checker states a goal has a reward to label."""


def _checker_config_path(trace_path: Path) -> Path:
    """`{TRACE_DIR}/task_{tid}_attempt_{aid}.jsonl` -> `{DATA_DIR}/task_{tid}/checker_config.json`.

    The layout is `<root>/{traces,data,goals_per_attempt}/`; given a trace path we walk up one dir
    to `<root>/`, then down into `data/task_{tid}/checker_config.json`. Whether that file exists is
    the reader's question (`_maybe_recover_axis_bounds`); a trace whose name does not give its task
    raises AxisBoundsError.
    """
    tid = _task_id_from_trace(trace_path)
    if tid is None:
        raise AxisBoundsError(f"{trace_path}: not named task_<id>_attempt_<id>.jsonl, so the task whose checker "
                              f"configuration holds this position-delta leaf's per-axis bounds is unknown")
    survey_root = trace_path.parent.parent
    return survey_root / "data" / f"task_{tid}" / "checker_config.json"


_POSDELTA_TYPES = frozenset({"PositionDeltaChecker", "SamplePositionDeltaChecker"})


def _iter_leaf_dicts(node):
    """Yield every dict leaf that carries a `type` key (checker configs)."""
    if isinstance(node, dict):
        if isinstance(node.get("type"), str) and "checkers" not in node:
            yield node
        for v in node.values():
            yield from _iter_leaf_dicts(v)
    elif isinstance(node, list):
        for v in node:
            yield from _iter_leaf_dicts(v)


# Bound keys the reward does NOT read: when any matching position-delta checker spells its bounds this
# way, the recovery is refused (`_maybe_recover_axis_bounds`), never read as unbounded.
_SNAKE_BOUND_KEYS = frozenset(f"{side}_delta_{ax}" for side in ("min", "max") for ax in "xyz")


def _posdelta_leaves_for(checker_cfg: dict, tracked_body: str):
    """Every PositionDeltaChecker leaf of the configuration whose body is `tracked_body`, in tree order."""
    for leaf_cfg in _iter_leaf_dicts(checker_cfg.get("checker", checker_cfg)):
        if leaf_cfg.get("type") not in _POSDELTA_TYPES:
            continue
        body = (leaf_cfg.get("sampleBodyName")
                or leaf_cfg.get("sample_body_name")
                or leaf_cfg.get("bodyName")
                or leaf_cfg.get("body_name"))
        # Match by tail so `soap_0` matches `sample/soap_0` (runtime name).
        if body != tracked_body and not (
            isinstance(body, str)
            and (body.endswith(f"/{tracked_body}") or tracked_body.endswith(f"/{body}"))
        ):
            continue
        yield leaf_cfg


def _bound_value(leaf_cfg: dict, key: str) -> float | None:
    """`float(leaf_cfg[key])`, None when absent. A value float() refuses, or one that is not finite
    (NaN or infinite, which would score that axis's progress as complete on every frame), is an
    AxisBoundsError naming it."""
    v = leaf_cfg.get(key)
    if v is None:
        return None
    try:
        value = float(v)
    except (TypeError, ValueError):
        raise AxisBoundsError(f"its {key} = {v!r} is not a number") from None
    if not math.isfinite(value):
        raise AxisBoundsError(f"its {key} = {v!r} is not a finite number")
    return value


def _axis_bounds_from_checker_cfg(
    checker_cfg: dict, tracked_body: str, axes: list[str] | None
) -> list[dict] | None:
    """Recover per-axis min/max for the PositionDeltaChecker leaf whose
    `sampleBodyName` matches `tracked_body`. Returns None if no matching
    leaf is found. `axes` restricts which axes are recovered; if None or
    empty the runtime default `("x", "z")` is used (matches AXIS-Replay's
    PositionDeltaChecker constructor default). A bound that is not a
    finite number raises AxisBoundsError.
    """
    for leaf_cfg in _posdelta_leaves_for(checker_cfg, tracked_body):
        cfg_axes = leaf_cfg.get("axes") or axes or ["x", "z"]
        if isinstance(cfg_axes, str):
            cfg_axes = [cfg_axes]
        min_keys = {"x": "minDeltaX", "y": "minDeltaY", "z": "minDeltaZ"}
        max_keys = {"x": "maxDeltaX", "y": "maxDeltaY", "z": "maxDeltaZ"}
        recovered: list[dict] = []
        for ax in cfg_axes:
            if ax not in min_keys:
                continue
            recovered.append({
                "axis": ax,
                "min": _bound_value(leaf_cfg, min_keys[ax]),
                "max": _bound_value(leaf_cfg, max_keys[ax]),
            })
        if recovered:
            return recovered
    return None


def _maybe_recover_axis_bounds(leaf: dict, trace_path: Path) -> list[dict] | None:
    """If the leaf lacks axis_bounds, derive them from the task's
    checker_config.json. Returns the recovered list (which the caller
    should splice into a shallow leaf copy), or None for a leaf that has them.

    Raises AxisBoundsError, naming the file, the leaf and why, when they
    cannot be recovered -- the file is missing, unreadable, not JSON or not
    a JSON object; it holds no position-delta checker for the leaf's body;
    any matching checker spells its bounds in snake_case, which the reward
    does not read; a bound is not a finite number; or no recovered axis has
    a numeric min or max. Without usable bounds the reward would be another
    label than the checker's.
    """
    if leaf.get("axis_bounds"):
        return None
    cfg_path = _checker_config_path(trace_path)
    why = "the per-axis bounds of a position-delta leaf without axis_bounds are read from it"
    try:
        cfg = json.loads(cfg_path.read_text())
    except FileNotFoundError:
        raise AxisBoundsError(f"{cfg_path}: does not exist; {why}") from None
    except OSError as exc:
        raise AxisBoundsError(f"{cfg_path}: cannot be read ({exc.strerror}); {why}") from exc
    except ValueError as exc:
        raise AxisBoundsError(f"{cfg_path}: is not a JSON checker configuration ({exc}); {why}") from exc
    if not isinstance(cfg, dict):
        raise AxisBoundsError(f"{cfg_path}: is a JSON {type(cfg).__name__}, not a checker configuration object; {why}")
    tracked = leaf.get("tracked_body_name") or leaf.get("resolved_body_name")
    if not tracked:
        raise AxisBoundsError(f"the position-delta leaf {leaf.get('path')!r} names no tracked body, so "
                              f"{cfg_path} cannot supply its per-axis bounds")
    # `tracked_body_name` from goals.json may include a namespace prefix
    # (`sample/soap_0`); strip it for the tail match in the checker config.
    tracked_tail = tracked.split("/")[-1]
    checker = f"the position-delta checker for {tracked_tail!r} (goal leaf {leaf.get('path')!r})"
    for leaf_cfg in _posdelta_leaves_for(cfg, tracked_tail):
        snake = sorted(_SNAKE_BOUND_KEYS & set(leaf_cfg))
        if snake:
            raise AxisBoundsError(f"{cfg_path}: {checker} spells its bounds {snake}, which the reward does not "
                                  f"read (it reads minDeltaX ... maxDeltaZ); {why}")
    try:
        bounds = _axis_bounds_from_checker_cfg(
            cfg, tracked_tail, leaf.get("delta_axes"),
        )
    except AxisBoundsError as exc:
        raise AxisBoundsError(f"{cfg_path}: {checker}: {exc}; {why}") from None
    if bounds is None:
        raise AxisBoundsError(f"{cfg_path}: holds no position-delta checker with an x/y/z bound for "
                              f"{tracked_tail!r}; {why}")
    if all(b["min"] is None and b["max"] is None for b in bounds):
        raise AxisBoundsError(f"{cfg_path}: {checker} has no numeric min or max bound on axes "
                              f"{[b['axis'] for b in bounds]}; {why}")
    return bounds


def compute_attempt_reward_curve(
    leaf: dict, trace_path: Path
) -> tuple[dict, str, int]:
    """Load a trace, clip to the demo's success frame + CLIP_TAIL_FRAMES,
    and return (rewards_dict, hand_body, n_full_frames).

    Dispatches to the reward family named by `leaf["goal_type"]`
    (`pnp`, `position_delta`, `directed_rotation`). A leaf with no
    `goal_type` is treated as `pnp`: such a leaf carries `goal_xyz` instead.
    """
    obj = leaf.get("resolved_body_name") or leaf["tracked_body_name"]
    steps_full, _ = _load_trace(trace_path)
    src = leaf.get("goal_source")
    sf = (int(leaf["stable_window"][0]) if src == "stable_window"
          else int(leaf["pass_frame"]))
    end = min(sf + 1 + CLIP_TAIL_FRAMES, len(steps_full))
    steps = steps_full[:end]

    goal_type = leaf.get("goal_type", "pnp")
    if goal_type == "pnp":
        rewards, hand = compute_rewards(
            steps, obj, None, goal_xyz=np.asarray(leaf["goal_xyz"]))
    elif goal_type == "position_delta":
        axis_bounds = leaf.get("axis_bounds")
        if not axis_bounds:
            # A leaf without per-axis bounds takes them from the task's
            # checker configuration, or raises AxisBoundsError.
            axis_bounds = _maybe_recover_axis_bounds(leaf, trace_path)
        rewards, hand = compute_rewards_position_delta(
            steps, obj,
            initial_position=np.asarray(leaf["initial_object_position"]),
            axis_bounds=axis_bounds,
        )
    elif goal_type == "directed_rotation":
        rewards, hand = compute_rewards_directed_rotation(
            steps, obj,
            initial_quat_wxyz=list(leaf["initial_object_quat_wxyz"]),
            target_angle_deg=float(leaf["target_angle_deg"]),
            angle_tol_deg=float(leaf["angle_tol_deg"]),
            max_tilt_deg=float(leaf["max_tilt_deg"]),
        )
    else:
        raise ValueError(f"unknown goal_type: {goal_type!r}")
    return rewards, hand, len(steps_full)


def _pos(entry: dict, body: str) -> np.ndarray:
    p = entry["body_pose"][body]["pos"]
    return np.asarray(p, dtype=np.float64)


def _pick_hand(entry0: dict, object_body: str) -> str:
    """Choose the hand body closest to the object at t=0 from the candidate list."""
    bp = entry0["body_pose"]
    obj = _pos(entry0, object_body)
    best, best_d = None, float("inf")
    for name in HAND_CANDIDATES:
        if name in bp:
            d = float(np.linalg.norm(_pos(entry0, name) - obj))
            if d < best_d:
                best, best_d = name, d
    if best is None:
        raise RuntimeError(f"no known hand body found; candidates: {HAND_CANDIDATES}")
    return best


def compute_rewards(
    steps: list[dict],
    object_body: str,
    target_body: str | None,
    goal_xyz: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], str]:
    """Compute per-frame rewards of a pick-and-place goal (Eq. 1, the same for every skill).

    r_total = W_PLACE_V2 · r_progress + W_CONTACT_V2 · r_contact_shape

    - r_progress: fraction of the initial-to-goal distance already covered.
      `r_progress = clip((d_initial − d_current) / d_initial, 0, 1)`. Starts
      at 0 regardless of how close the object begins to the goal; goes to
      1 at success. Falls (object leaves the goal region) roll it back
      naturally without any grasp/release state machine.
    - r_contact_shape: `1 − tanh(k · d_hand_object)`. Smoothly saturates
      near 1 as the hand approaches the object; no cliff at any threshold.
      Acts as a small auxiliary shaping term (W_CONTACT_V2 = 0.2) so a
      policy that hasn't made any placement progress still gets a gradient
      to reach the object.

    `placed` (the object within PLACE_SUCCESS_DIST of its goal) rides along;
    it never enters r_total.

    Priority for the place-term goal:
      1. `goal_xyz` (fixed point, e.g. demo-derived from the pass-frame)
      2. `target_body` (live pose of a reference body in the trace)
      3. neither -> progress = 0 for every frame
    """
    T = len(steps)
    r_contact = np.zeros(T)   # smooth "hand near object" shaping (bounded)
    r_progress = np.zeros(T)  # progress toward goal state (task-state signal)

    hand_body = _pick_hand(steps[0], object_body)
    o0 = _pos(steps[0], object_body)

    # Initial object-to-goal distance for progress normalisation.
    def _goal_of(s: dict) -> np.ndarray | None:
        if goal_xyz is not None:
            return goal_xyz
        if target_body is not None:
            return _pos(s, target_body)
        return None

    g0 = _goal_of(steps[0])
    if g0 is not None:
        d_initial = max(float(np.linalg.norm(o0 - g0)), PROGRESS_EPS_M)
    else:
        d_initial = None

    for t, s in enumerate(steps):
        h = _pos(s, hand_body)
        o = _pos(s, object_body)
        d_ho = float(np.linalg.norm(h - o))
        # Contact shaping: 1 - tanh(k * d) smoothly saturates as d -> 0.
        # No cliff at any threshold; small aux term with weight W_CONTACT_V2.
        r_contact[t] = 1.0 - float(np.tanh(CONTACT_TANH_K * d_ho))

        # Progress: fraction of initial goal distance already covered.
        g = _goal_of(s)
        if g is not None and d_initial is not None:
            d_ot = float(np.linalg.norm(o - g))
            r_progress[t] = max(0.0, min(1.0, (d_initial - d_ot) / d_initial))

    r_total = W_PLACE_V2 * r_progress + W_CONTACT_V2 * r_contact

    # Success channel: the object within PLACE_SUCCESS_DIST of its goal.
    if g0 is not None:
        d_ot_series = np.array([
            float(np.linalg.norm(
                _pos(s, object_body) - (goal_xyz if goal_xyz is not None
                                        else _pos(s, target_body))
            )) for s in steps
        ])
        placed = d_ot_series < PLACE_SUCCESS_DIST
    else:
        placed = np.zeros(T, dtype=bool)

    # Hold-stability bonus: if the trace was clipped to include a
    # CLIP_TAIL_FRAMES-long hold window and the object stayed within
    # PLACE_SUCCESS_DIST for the entire tail, reward the terminal frame
    # with an additive HOLD_STABILITY_BONUS. Drifts during hold naturally
    # forfeit the bonus. With CLIP_TAIL_FRAMES = 0 and HOLD_STABILITY_BONUS
    # = 0, the published labels' values, it never applies.
    if CLIP_TAIL_FRAMES > 0 and HOLD_STABILITY_BONUS > 0.0 and T > CLIP_TAIL_FRAMES:
        hold_start = T - CLIP_TAIL_FRAMES
        if bool(placed[hold_start:].all()):
            r_total[-1] += HOLD_STABILITY_BONUS

    return {"contact": r_contact, "progress": r_progress, "placed": placed.astype(float), "total": r_total}, hand_body


_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _per_axis_place_reward(
    delta_xyz: np.ndarray, axis_bounds: list[dict]
) -> float:
    """Fraction of the per-axis bound region covered by delta.

    For each axis with a (min, max) bound, compute how far into the
    admissible half-open interval delta_axis lies:

      unmet_range = max_i - min_i                       (bound width)
      progress_i  = clip(delta_i - min_i, 0, unmet_range) / unmet_range

    A missing min is treated as -inf (no lower bound), a missing max as
    +inf. When either endpoint is infinite, progress is 1.0 iff delta_i
    is on the "inside" side, else 0.0 — this gives sensible behavior for
    the common one-sided lift case (min z = 0.05, max z = None):
    progress ramps linearly from 0 -> 1 as delta_z climbs from 0 to
    min_z, then saturates.

    Aggregate across axes with min (all bounds must be met to hit r=1).
    """
    per_axis: list[float] = []
    for b in axis_bounds:
        axis = b.get("axis")
        if axis not in _AXIS_INDEX:
            continue
        lo = b.get("min")
        hi = b.get("max")
        if lo is None and hi is None:
            # Both endpoints missing on this axis — skip; still allow other
            # axes to constrain the reward. If nothing survives, fail loud
            # below (see `raise ValueError`) rather than silently returning
            # 0.0 and looking like the runtime is progressing at all times.
            continue
        d = float(delta_xyz[_AXIS_INDEX[axis]])
        lo_val = float(lo) if lo is not None else None
        hi_val = float(hi) if hi is not None else None
        # For a bounded [lo, hi] interval, progress = normalized location
        # within the interval; below lo -> 0, above hi -> 0 (out of bounds)
        # ramp linearly toward center of range.
        if lo_val is not None and hi_val is not None:
            width = hi_val - lo_val
            if width <= 1e-9:
                per_axis.append(1.0 if abs(d - lo_val) < 1e-6 else 0.0)
                continue
            # Ramp: reward peaks where the bound is comfortably met (any
            # point in [lo, hi]), zero outside.
            if lo_val <= d <= hi_val:
                per_axis.append(1.0)
            elif d < lo_val:
                # approaching from below: linear ramp on the last width.
                per_axis.append(max(0.0, 1.0 - (lo_val - d) / max(width, 1e-6)))
            else:  # d > hi_val
                per_axis.append(max(0.0, 1.0 - (d - hi_val) / max(width, 1e-6)))
        elif lo_val is not None:
            # One-sided lower bound (e.g. lift z >= 0.05). Linear 0..1 as
            # delta rises from 0 to lo, saturates at 1 above.
            denom = max(abs(lo_val), 1e-6)
            if lo_val >= 0:
                per_axis.append(max(0.0, min(1.0, d / denom)))
            else:
                # bound of the form d >= -0.05 (allow up to that much
                # displacement in the negative direction). Progress = 1 if
                # d >= lo, else linear ramp starting from d=lo down to
                # d=2*lo (bound-width heuristic).
                per_axis.append(max(0.0, min(1.0, 1.0 - (lo_val - d) / denom)) if d < lo_val else 1.0)
        elif hi_val is not None:
            # One-sided upper bound (e.g. don't move more than X). Reward
            # is 1 while inside, ramps down once past hi.
            denom = max(abs(hi_val), 1e-6)
            if d <= hi_val:
                per_axis.append(1.0)
            else:
                per_axis.append(max(0.0, 1.0 - (d - hi_val) / denom))
    if not per_axis:
        # Caller passed axis_bounds explicitly but none yielded usable
        # bounds — a silent zero-reward here is worse than a crash because
        # every downstream metric would look "healthy" while the reward
        # signal is dead. Fail loud so the caller notices and either
        # supplies valid bounds or drops the leaf.
        raise ValueError(
            f"axis_bounds provided but all entries missing usable min/max: "
            f"{axis_bounds!r}"
        )
    return float(min(per_axis))


def compute_rewards_position_delta(
    steps: list[dict],
    object_body: str,
    initial_position: np.ndarray,
    axis_bounds: list[dict],
) -> tuple[dict[str, np.ndarray], str]:
    """Position-delta reward (Eq. 1, the same for every skill):

        r_total = W_PLACE_V2 · r_progress + W_CONTACT_V2 · r_contact

    Progress is per-axis (`_per_axis_place_reward`) toward `axis_bounds`: the leaf's own, or those
    recovered from the task's checker configuration (compute_attempt_reward_curve). Contact is the
    smooth `1 - tanh(k · ||h - o||)` term of the pnp reward.
    """
    T = len(steps)
    hand_body = _pick_hand(steps[0], object_body)
    r_contact = np.zeros(T)
    r_progress = np.zeros(T)
    for t, s in enumerate(steps):
        o = _pos(s, object_body)
        h = _pos(s, hand_body)
        r_contact[t] = 1.0 - float(np.tanh(CONTACT_TANH_K * float(np.linalg.norm(h - o))))
        r_progress[t] = _per_axis_place_reward(o - initial_position, axis_bounds)
    r_total = W_PLACE_V2 * r_progress + W_CONTACT_V2 * r_contact
    placed = r_progress >= 1.0
    # Hold-stability bonus (as in compute_rewards; never applies at the published values).
    if CLIP_TAIL_FRAMES > 0 and HOLD_STABILITY_BONUS > 0.0 and T > CLIP_TAIL_FRAMES:
        hold_start = T - CLIP_TAIL_FRAMES
        if bool(placed[hold_start:].all()):
            r_total[-1] += HOLD_STABILITY_BONUS
    return {"contact": r_contact, "progress": r_progress, "placed": placed.astype(float), "total": r_total}, hand_body


def _quat_wxyz_to_xyzw(q):
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]


def _quat_inverse(q):
    """Inverse of a unit quaternion [x, y, z, w]."""
    return [-q[0], -q[1], -q[2], q[3]]


def _quat_multiply(a, b):
    """Hamilton product of two quaternions [x, y, z, w]."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _quat_angle_deg(q_init_xyzw, q_curr_xyzw) -> float:
    """Total rotation angle between two unit quaternions (any axis), deg."""
    q_rel = _quat_multiply(q_curr_xyzw, _quat_inverse(q_init_xyzw))
    w = min(1.0, max(-1.0, abs(q_rel[3])))
    return float(np.degrees(2.0 * np.arccos(w)))


def _quat_rotate_vec(q, v):
    """Rotate a 3-vector v by quaternion q [x, y, z, w]."""
    qx, qy, qz, qw = q
    vx, vy, vz = v
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return [
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    ]


def _tilt_deg_from_init(q_init_xyzw, q_curr_xyzw) -> float:
    """Angle between the body's up axis (local +Z) as rotated by q_init vs
    q_curr, measured in world frame. Mirrors DirectedRotationChecker."""
    up_i = _quat_rotate_vec(q_init_xyzw, [0.0, 0.0, 1.0])
    up_c = _quat_rotate_vec(q_curr_xyzw, [0.0, 0.0, 1.0])
    n_i = float(np.linalg.norm(up_i))
    n_c = float(np.linalg.norm(up_c))
    if n_i < 1e-9 or n_c < 1e-9:
        return 0.0
    cos = float(np.dot(up_i, up_c)) / (n_i * n_c)
    cos = max(-1.0, min(1.0, cos))
    return float(np.degrees(np.arccos(cos)))


def compute_rewards_directed_rotation(
    steps: list[dict],
    object_body: str,
    initial_quat_wxyz: list[float],
    target_angle_deg: float,
    angle_tol_deg: float,
    max_tilt_deg: float,
) -> tuple[dict[str, np.ndarray], str]:
    """Directed-rotation reward (Eq. 1, the same for every skill), for "Rotate X to angle A" tasks
    whose success checker is DirectedRotationChecker:

        r_total = W_PLACE_V2 · r_progress + W_CONTACT_V2 · r_contact

    Progress is the rotation from the initial pose toward `target_angle_deg`, discounted by the
    tilt beyond `max_tilt_deg`: `min(angle / target, 1) · (1 - tanh(3 · tilt_pen))` with
    `tilt_pen = max(0, tilt - max_tilt) / max_tilt`. Contact is the smooth `1 - tanh(k · ||h - o||)`
    term of the pnp reward. `placed`: the angle within `angle_tol_deg` of the target and the tilt
    within `max_tilt_deg`.
    """
    T = len(steps)
    hand_body = _pick_hand(steps[0], object_body)
    q_init = _quat_wxyz_to_xyzw(initial_quat_wxyz)
    target = max(target_angle_deg, 1e-6)
    max_tilt = max(max_tilt_deg, 1e-6)

    r_contact = np.zeros(T)
    r_progress = np.zeros(T)
    placed = np.zeros(T, dtype=bool)
    for t, s in enumerate(steps):
        h = _pos(s, hand_body)
        o = _pos(s, object_body)
        d_ho = float(np.linalg.norm(h - o))
        r_contact[t] = 1.0 - float(np.tanh(CONTACT_TANH_K * d_ho))
        q_curr = _quat_wxyz_to_xyzw(s["body_pose"][object_body]["quat"])
        angle = _quat_angle_deg(q_init, q_curr)
        tilt = _tilt_deg_from_init(q_init, q_curr)
        rot_prog = min(1.0, angle / target)
        tilt_pen = max(0.0, tilt - max_tilt) / max_tilt
        r_progress[t] = rot_prog * (1.0 - float(np.tanh(3.0 * tilt_pen)))
        placed[t] = (abs(angle - target_angle_deg) <= angle_tol_deg
                     and tilt <= max_tilt_deg)

    r_total = W_PLACE_V2 * r_progress + W_CONTACT_V2 * r_contact
    if CLIP_TAIL_FRAMES > 0 and HOLD_STABILITY_BONUS > 0.0 and T > CLIP_TAIL_FRAMES:
        hold_start = T - CLIP_TAIL_FRAMES
        if bool(placed[hold_start:].all()):
            r_total[-1] += HOLD_STABILITY_BONUS
    return {"contact": r_contact, "progress": r_progress, "placed": placed.astype(float), "total": r_total}, hand_body
