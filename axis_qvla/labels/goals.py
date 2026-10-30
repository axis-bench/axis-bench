"""The task checkers' goal record of one attempt: where each sub-checker's tracked object was when the
checker first held (or, for placing checkers, its stable end pose).

For an attempt's trace and its task's checker configuration, every leaf checker of the task's checker
tree is instantiated through the replay runtime's own registry (`axis_replay_native.checkers`, from
the axis-render submodule), so a record means what the live replay meant; each leaf is run over every
frame. This is the input of the reward's progress term (axis_qvla.labels.progress, axis_qvla.labels.reward). The
checker classes are imported only when a record is extracted.

Offline tier: numpy; the checker package of the axis-render submodule when extracting.
"""
from __future__ import annotations

import functools
import json
import types
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:       # an annotation only; the class itself is imported by _checkers()
    from axis_replay_native.checkers.runtime_types import CheckerRuntimePayload


@functools.lru_cache(maxsize=1)
def _checkers() -> types.SimpleNamespace:
    from axis_replay_native.checkers.base import AlwaysFalseChecker
    from axis_replay_native.checkers.CheckerManager import CheckerManager
    from axis_replay_native.checkers.CompositeChecker import CompositeChecker
    from axis_replay_native.checkers.runtime_types import CheckerRuntimePayload

    return types.SimpleNamespace(AlwaysFalseChecker=AlwaysFalseChecker, CheckerManager=CheckerManager,
                                 CompositeChecker=CompositeChecker, CheckerRuntimePayload=CheckerRuntimePayload)


# Placing checkers: goal is a *stable end pose* where the object comes to
# rest inside the passing region. Picking / moving / rotating checkers
# just need to fire once (delta / rotation reached); first-pass is fine.
PLACING_CHECKER_TYPES = frozenset({
    "RelativeCylinderChecker",
    "RelativePositionBoundsChecker",
    "RelativeAxisRangeChecker",
    "BowlPositionChecker",
    "BowlInDrawerChecker",
    "FrameBBoxChecker",
    "DrawerBBoxChecker",
})
STABLE_MAX_DIAMETER = 0.005       # max pairwise distance in the window (m)
STABLE_MIN_SECONDS = 0.5          # window must span >= this much sim_time
TRIVIAL_PLACEMENT_M = 0.03        # reject pnp leaves whose object was already
                                  # within this distance of the recorded goal
                                  # at t=0 (OR-composite trivial matches)

# --------------------------------------------------------------------------
# Naming: pull the tracked-object body name out of any leaf-checker config.
# --------------------------------------------------------------------------

OBJECT_KEYS = ("objName", "obj_name", "sampleBodyName", "sample_body_name", "bodyName", "body_name")


def _tracked_body(cfg_dict: dict) -> str | None:
    for k in OBJECT_KEYS:
        v = cfg_dict.get(k)
        if isinstance(v, str) and v:
            return v
    return None


# --------------------------------------------------------------------------
# Flatten a checker tree to a list of (path, config, checker_instance).
# --------------------------------------------------------------------------

def _flatten(node, path: str = ""):
    if isinstance(node, _checkers().CompositeChecker):
        for i, child in enumerate(node.checkers):
            yield from _flatten(child, f"{path}/{i}" if path else str(i))
    else:
        yield path or "0", node


# --------------------------------------------------------------------------
# Build a CheckerRuntimePayload from one trace frame.
# --------------------------------------------------------------------------

def _wxyz_to_xyzw(q):
    return [q[1], q[2], q[3], q[0]]


def _payload_from_step(
    step: dict,
    initial_positions: dict[str, list[float]],
    initial_orientations: dict[str, list[float]],
) -> CheckerRuntimePayload:
    bp = step["body_pose"]
    object_positions = {name: list(v["pos"]) for name, v in bp.items()}
    object_orientations = {name: _wxyz_to_xyzw(v["quat"]) for name, v in bp.items()}
    # joint_qpos contains scalar hinge/slide values *and* 7-vector free joints;
    # CheckerRuntimePayload.joint_positions expects scalars only.
    joint_positions = {
        k: float(v) for k, v in (step.get("joint_qpos") or {}).items()
        if isinstance(v, (int, float))
    }
    return _checkers().CheckerRuntimePayload(
        object_positions=object_positions,
        object_orientations=object_orientations,
        joint_positions=joint_positions,
        initial_object_positions=initial_positions,
        initial_object_orientations=initial_orientations,
    )


def _load_trace(path: Path) -> list[dict]:
    steps = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("event") == "state":
            steps.append(e)
    if not steps:
        raise RuntimeError(f"empty trace: {path}")
    return steps


def read_trace_states(path) -> list[dict]:
    """The `state` events of a trace (JSON lines), in order: the one reader the label pipeline and the
    goal extraction share. A missing or unreadable trace raises OSError, one with no state event
    RuntimeError."""
    return _load_trace(Path(path))


def _resolve_key(name: str, mapping: dict) -> str | None:
    if name in mapping:
        return name
    suffix_slash = "/" + name
    for key in mapping:
        if key.endswith(suffix_slash):
            return key
    for key in mapping:
        if key.endswith(name):
            return key
    return None


# --------------------------------------------------------------------------
# Main extraction.
# --------------------------------------------------------------------------

def _find_stable_window(
    steps: list[dict],
    body_key: str,
    pass_mask: list[bool],
    max_diameter: float,
    min_seconds: float,
) -> tuple[int, int] | None:
    """Find the longest tail window in which
      - the checker holds every frame,
      - max pairwise object displacement over the window < max_diameter,
      - the window spans >= min_seconds of sim_time.
    Returns (i0, i1) inclusive, or None.

    Scans backward from the last passing frame; extends the window one frame
    at a time as long as both invariants still hold, then verifies duration.
    """
    pass_indices = [i for i, p in enumerate(pass_mask) if p]
    if not pass_indices:
        return None
    last_pass = pass_indices[-1]

    def _pos(i: int) -> tuple[float, float, float]:
        p = steps[i]["body_pose"][body_key]["pos"]
        return (float(p[0]), float(p[1]), float(p[2]))

    def _t(i: int) -> float:
        return float(steps[i].get("sim_time", i))

    window_positions = [_pos(last_pass)]
    i0 = last_pass
    while i0 > 0 and pass_mask[i0 - 1]:
        cand = _pos(i0 - 1)
        # Only need to check the new candidate vs. every existing point;
        # existing pairwise distances already satisfy the bound.
        worst = 0.0
        for p in window_positions:
            dx = cand[0] - p[0]
            dy = cand[1] - p[1]
            dz = cand[2] - p[2]
            d = (dx * dx + dy * dy + dz * dz) ** 0.5
            if d > worst:
                worst = d
                if worst > max_diameter:
                    break
        if worst > max_diameter:
            break
        window_positions.append(cand)
        i0 -= 1

    if _t(last_pass) - _t(i0) < min_seconds:
        return None
    return (i0, last_pass)


def extract_for_attempt(task_dir: Path, trace_path: Path) -> dict:
    checker_config = json.loads((task_dir / "checker_config.json").read_text())
    steps = _load_trace(trace_path)
    step0 = steps[0]
    initial_positions = {n: list(v["pos"]) for n, v in step0["body_pose"].items()}
    initial_orientations = {n: _wxyz_to_xyzw(v["quat"]) for n, v in step0["body_pose"].items()}

    # Build the checker tree, then flatten to leaf checkers.
    manager = _checkers().CheckerManager(checker_config)
    if manager.checker is None:
        return {"error": "checker tree failed to build", "n_steps": len(steps)}

    leaves: list[tuple[str, object]] = list(_flatten(manager.checker))

    # For each leaf: figure out the tracked body name, and scan the trace.
    leaf_records = []
    for path, leaf in leaves:
        cfg_dict = leaf.config or {}
        checker_type = type(leaf).__name__
        if isinstance(leaf, _checkers().AlwaysFalseChecker):
            leaf_records.append({
                "path": path,
                "checker_type": "AlwaysFalseChecker",
                "config_type": cfg_dict.get("type"),
                "reason": "unimplemented checker",
                "pass_frame": None,
                "goal_xyz": None,
            })
            continue

        body_name = _tracked_body(cfg_dict)

        # Evaluate the leaf on every frame -> pass_mask; catch first exception.
        pass_mask = [False] * len(steps)
        err = None
        for i, s in enumerate(steps):
            payload = _payload_from_step(s, initial_positions, initial_orientations)
            try:
                pass_mask[i] = bool(leaf.check(payload))
            except Exception as ex:
                err = f"{type(ex).__name__}: {ex}"
                break
        if err is not None:
            leaf_records.append({
                "path": path, "checker_type": checker_type,
                "error": err, "pass_frame": None, "goal_xyz": None,
            })
            continue

        pass_indices = [i for i, p in enumerate(pass_mask) if p]
        first_pass = pass_indices[0] if pass_indices else None

        rec = {
            "path": path,
            "checker_type": checker_type,
            "tracked_body_name": body_name,
            "pass_frame": first_pass,
            "n_pass_frames": len(pass_indices),
            "goal_source": None,
            "stable_window": None,
            "goal_xyz": None,
            "goal_quat_xyzw": None,
        }

        if first_pass is None or body_name is None:
            leaf_records.append(rec)
            continue

        bp_at_first = steps[first_pass]["body_pose"]
        key = _resolve_key(body_name, bp_at_first)
        if key is None:
            leaf_records.append(rec)
            continue
        rec["resolved_body_name"] = key

        is_placing = checker_type in PLACING_CHECKER_TYPES
        if is_placing:
            # Preferred: stable settled window over the pass region.
            # Fallback: the object's position at first_pass (single-frame
            # goal), for a demo whose object keeps rolling after it is placed.
            window = _find_stable_window(steps, key, pass_mask,
                                         max_diameter=STABLE_MAX_DIAMETER,
                                         min_seconds=STABLE_MIN_SECONDS)
            if window is not None:
                i0, i1 = window
                positions = np.array([steps[j]["body_pose"][key]["pos"]
                                      for j in range(i0, i1 + 1)])
                quats_wxyz = np.array([steps[j]["body_pose"][key]["quat"]
                                       for j in range(i0, i1 + 1)])
                goal_xyz = positions.mean(axis=0).tolist()
                # Quats: pick last frame's orientation (averaging quats is
                # not generally well-defined without care).
                goal_quat = _wxyz_to_xyzw(quats_wxyz[-1].tolist())
                goal_source = "stable_window"
                stable_win = [int(i0), int(i1)]
            else:
                goal_xyz = list(bp_at_first[key]["pos"])
                goal_quat = _wxyz_to_xyzw(bp_at_first[key]["quat"])
                goal_source = "first_pass_fallback"
                stable_win = None
            # Trivial-placement guard: reject leaves where the object was
            # already within TRIVIAL_PLACEMENT_M of its recorded goal at
            # t=0. OR-composite tasks (e.g. "place one of these dice")
            # register every candidate object; leaves whose object was
            # already in the target zone from the start satisfy the checker
            # trivially and carry no manipulation signal.
            initial_obj_pos = list(initial_positions[key])
            d_init_goal = float(np.linalg.norm(
                np.asarray(initial_obj_pos) - np.asarray(goal_xyz)))
            if d_init_goal < TRIVIAL_PLACEMENT_M:
                rec["goal_source"] = "trivial_placement"
                rec["initial_object_position"] = initial_obj_pos
                rec["d_init_goal"] = d_init_goal
                leaf_records.append(rec)
                continue
            rec.update(
                goal_type="pnp",
                goal_source=goal_source,
                stable_window=stable_win,
                goal_xyz=goal_xyz,
                goal_quat_xyzw=goal_quat,
                initial_object_position=initial_obj_pos,
                d_init_goal=d_init_goal,
            )
        elif checker_type in ("PositionDeltaChecker", "SamplePositionDeltaChecker"):
            # Default axes match the AXIS-Replay runtime PositionDeltaChecker
            # constructor (which uses ("x", "z")). SamplePositionDeltaChecker
            # is a subclass alias exposed by the runtime registry — its
            # config type may still be "PositionDeltaChecker" but its
            # runtime class name will be "SamplePositionDeltaChecker".
            axes = cfg_dict.get("axes") or ["x", "z"]
            if isinstance(axes, str):
                axes = [axes]
            # Store the FULL per-axis min/max bounds. The AXIS runtime
            # checker requires the delta to satisfy every axis's [min,max]
            # simultaneously, so one axis's min alone would reward sideways
            # motion for lift tasks and moving OUT of a stay-within-bounds box.
            axis_min_keys = {"x": "minDeltaX", "y": "minDeltaY", "z": "minDeltaZ"}
            axis_max_keys = {"x": "maxDeltaX", "y": "maxDeltaY", "z": "maxDeltaZ"}
            axis_bounds: list[dict] = []
            for ax in axes:
                if ax not in axis_min_keys:
                    continue
                lo_raw = cfg_dict.get(axis_min_keys[ax])
                hi_raw = cfg_dict.get(axis_max_keys[ax])
                lo = float(lo_raw) if lo_raw is not None else None
                hi = float(hi_raw) if hi_raw is not None else None
                axis_bounds.append({"axis": ax, "min": lo, "max": hi})

            # The strictest of the recorded mins, a field of the published goal records.
            primary_axis = axes[0]
            legacy_min_key = axis_min_keys.get(primary_axis, "minDeltaZ")
            delta_thresh = abs(float(cfg_dict.get(legacy_min_key) or 0.05))
            initial_pos = list(initial_positions[key])
            rec.update(
                goal_type="position_delta",
                goal_source="first_pass",
                goal_xyz=list(bp_at_first[key]["pos"]),
                goal_quat_xyzw=_wxyz_to_xyzw(bp_at_first[key]["quat"]),
                delta_thresh=delta_thresh,
                delta_axes=list(axes),
                axis_bounds=axis_bounds,
                initial_object_position=initial_pos,
            )
        elif checker_type == "DirectedRotationChecker":
            target_angle = float(cfg_dict.get("targetAngleDeg", 90.0))
            angle_tol = float(cfg_dict.get("angleToleranceDeg", 10.0))
            max_tilt = float(cfg_dict.get("maxTiltDeg", 15.0))
            # initial_orientations is stored xyzw (converted at load); the
            # reward function expects wxyz.
            q_xyzw = initial_orientations[key]
            initial_quat_wxyz = [float(q_xyzw[3]), float(q_xyzw[0]),
                                 float(q_xyzw[1]), float(q_xyzw[2])]
            rec.update(
                goal_type="directed_rotation",
                goal_source="first_pass",
                goal_xyz=list(bp_at_first[key]["pos"]),
                goal_quat_xyzw=_wxyz_to_xyzw(bp_at_first[key]["quat"]),
                target_angle_deg=target_angle,
                angle_tol_deg=angle_tol,
                max_tilt_deg=max_tilt,
                initial_object_quat_wxyz=initial_quat_wxyz,
            )
        else:
            # Other implemented families (rotation-threshold, joint, keypress)
            # don't have a reward mapping yet; record first-pass pose only so
            # the leaf isn't lost.
            rec.update(
                goal_source="first_pass",
                goal_xyz=list(bp_at_first[key]["pos"]),
                goal_quat_xyzw=_wxyz_to_xyzw(bp_at_first[key]["quat"]),
            )
        leaf_records.append(rec)

    # Top-level: also record first frame at which the WHOLE tree fires.
    top_first_pass = None
    for i, s in enumerate(steps):
        payload = _payload_from_step(s, initial_positions, initial_orientations)
        if manager.check(payload):
            top_first_pass = i
            break

    return {
        "n_steps": len(steps),
        "top_first_pass_frame": top_first_pass,
        "leaves": leaf_records,
    }
