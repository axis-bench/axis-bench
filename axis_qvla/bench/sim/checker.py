"""Evaluate a task's *real* success checker during a policy rollout.

The real checker is *relative to a reference body* (a basket, a
sandwich, sometimes an OR over several), requires the gripper to be
open, and is resolved per scene variant. This module wires the
``axis_replay_native`` checker suite into the rollout, so success is
measured exactly as the task defines it.

Two entry points:

* :func:`make_stable_checker` — a ``checker_fn(model, data) -> bool`` for
  :func:`axis_replay_native.rollout.rollout_policy`, requiring the real
  checker to hold for ``n_stable`` consecutive calls (keeps the v4
  stability semantics).
* :func:`satisfied_at_init` — a single evaluation used by the seed
  screener to drop eval seeds whose scene *starts* already solved.
"""

from __future__ import annotations

import json
from typing import Any, Callable
from pathlib import Path

import mujoco

from axis_replay_native.checkers.CheckerManager import CheckerManager
from axis_replay_native.checkers.runtime_types import CheckerRuntimePayload
from axis_replay_native.checker_payload_extractor import (
    extract_object_positions,
    extract_object_orientations,
    extract_joint_positions,
    extract_site_positions,
    extract_site_rot_mats,
    extract_body_rot_mats,
    extract_initial_positions,
    extract_initial_orientations,
)

CheckerFn = Callable[[mujoco.MjModel, mujoco.MjData], bool]


def load_checker_config(task_id: int | str, *, task_data_root: Path) -> dict:
    """Load a task's ``checker_config.json`` (the real success spec).

    `task_data_root` is the directory holding ``task_<id>/``. The eval CLI passes
    ``--task-data-root`` here, so a checkout whose task files come from qvla-v1-bench scores
    against exactly those files.
    """
    root = Path(task_data_root)
    p = root / f"task_{task_id}" / "checker_config.json"
    if not p.exists():
        raise FileNotFoundError(f"no checker_config.json for task {task_id}: {p}")
    return json.loads(p.read_text())


def _build_payload(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    initial_positions: dict[str, list[float]] | None,
    initial_orientations: dict[str, list[float]] | None,
) -> CheckerRuntimePayload:
    """Assemble a :class:`CheckerRuntimePayload` from live MuJoCo state.

    Built directly (not via ``build_checker_payload``) so we avoid the
    ``DecodedTrajectory`` dependency — a policy rollout has no decoded
    trajectory. Initial pose fields are precomputed by the caller.
    """
    return CheckerRuntimePayload(
        object_positions=extract_object_positions(model, data),
        object_orientations=extract_object_orientations(model, data),
        joint_positions=extract_joint_positions(model, data),
        site_positions=extract_site_positions(model, data),
        site_rot_mats=extract_site_rot_mats(model, data),
        body_rot_mats=extract_body_rot_mats(model, data),
        pressed_keys=set(),
        initial_object_positions=initial_positions,
        initial_object_orientations=initial_orientations,
    )


def make_stable_checker(
    task_id: int,
    initial_state: dict[str, Any] | None,
    n_stable: int = 1,
    *,
    task_data_root: Path,
) -> CheckerFn:
    """Return a rollout ``checker_fn`` running the task's real checker.

    ``initial_state`` is the *selected scene-variant's* initial state
    (``{objects, robots}``); it fixes the reference pose used by delta /
    rotation checkers. Success requires the real checker to hold for
    ``n_stable`` consecutive calls.
    """
    cm = CheckerManager(load_checker_config(task_id, task_data_root=task_data_root))
    state: dict[str, Any] = {
        "stable": 0,
        "init_pos": None,
        "init_ori": None,
        "captured": False,
    }

    def checker_fn(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        if not state["captured"]:
            # Reference pose from the variant's initial_state (authoritative
            # t=0), resolving body names against this compiled model.
            state["init_pos"] = extract_initial_positions(
                model, data, None, initial_state,
            )
            state["init_ori"] = extract_initial_orientations(
                model, data, None, initial_state,
            )
            state["captured"] = True
        payload = _build_payload(model, data, state["init_pos"], state["init_ori"])
        ok = bool(cm.check(payload))
        state["stable"] = state["stable"] + 1 if ok else 0
        return state["stable"] >= n_stable

    return checker_fn


def make_detail_probe(
    task_id: int,
    initial_state: dict[str, Any] | None,
    *,
    task_data_root: Path,
) -> Callable[[mujoco.MjModel, mujoco.MjData], "tuple[bool, list[dict]]"]:
    """Return a READ-ONLY probe exposing the real checker's tracked quantities.

    ``probe(model, data) -> (passed, sub_results)`` runs the same
    ``CheckerManager`` :func:`make_stable_checker` does, but via
    ``check_with_details`` so a diagnostic can read the obj/ref relative
    delta and the gripper joint values -- not just a pass/fail bit. It is a
    separate manager instance and holds no stability state, so it never
    affects the success checker used for the actual rollout verdict.
    """
    cm = CheckerManager(load_checker_config(task_id, task_data_root=task_data_root))
    state: dict[str, Any] = {"init_pos": None, "init_ori": None, "captured": False}

    def probe(model: mujoco.MjModel, data: mujoco.MjData):
        if not state["captured"]:
            state["init_pos"] = extract_initial_positions(model, data, None, initial_state)
            state["init_ori"] = extract_initial_orientations(model, data, None, initial_state)
            state["captured"] = True
        payload = _build_payload(model, data, state["init_pos"], state["init_ori"])
        return cm.check_with_details(payload)

    return probe


def satisfied_at_init(
    task_id: int,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    initial_state: dict[str, Any] | None,
    *,
    task_data_root: Path,
) -> bool:
    """True if the task's real checker already passes at t=0.

    Used to drop eval seeds whose scene starts already solved (any no-op
    policy would "succeed" on them).
    """
    cm = CheckerManager(load_checker_config(task_id, task_data_root=task_data_root))
    init_pos = extract_initial_positions(model, data, None, initial_state)
    init_ori = extract_initial_orientations(model, data, None, initial_state)
    payload = _build_payload(model, data, init_pos, init_ori)
    return bool(cm.check(payload))
