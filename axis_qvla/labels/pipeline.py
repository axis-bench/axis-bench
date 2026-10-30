"""One episode's per-frame labels from its recording: the columns the corpus publishes per frame.

`label_episode` takes an attempt's trace (5 Hz state events), the task checkers' goal record for it
and the episode's render length, and returns: every body's pose resampled onto the 15 fps render
rows, the joint positions, the end-effector pose, the reward of Eq. (1) (axis_qvla.labels.reward, on the
trace clipped at the success frame) with `reward_valid` marking the rows the clipped curve covers,
`success` from the checkers' first pass, `is_anchor` on rows that carry a measured pose, and -- given
the task's MuJoCo scene (axis_qvla.labels.contacts.ContactModel) -- the four contact flags. An attempt
whose goal record has no reward leaf, or whose reward cannot be formed (UNFORMABLE_REWARD), gets
reward 0 and reward_valid False everywhere, as the corpus labels it. Nothing else reads as "no
reward": a trace that is missing, unreadable, not JSON or empty raises, as do per-axis bounds the
checker configuration cannot supply (axis_qvla.labels.reward.AxisBoundsError).

Offline tier: numpy, scipy; MuJoCo only for contacts.
"""
from __future__ import annotations

import pathlib

import numpy as np

from axis_qvla.labels.contacts import CONTACT_COLUMNS, HAND_BODY, ROBOT_PREFIX, ContactError, resolve_target_bodies, stem
from axis_qvla.labels.goals import read_trace_states
from axis_qvla.labels.progress import pick_reward_leaf
from axis_qvla.labels.resample import (anchor_mask, resample_poses_to_rows, reward_to_rows, reward_valid_mask,
                                  success_from_pass_frame, trace_anchor_rows)
from axis_qvla.labels.reward import compute_attempt_reward_curve

RENDER_FPS = 15.0
# How the reward of an attempt fails to form from its goal record and its (readable) trace: a leaf
# lacking a field its goal family needs, or a trace lacking the tracked body (KeyError); an unknown
# goal family, or per-axis bounds stated on the goal leaf with no usable min or max (ValueError); no
# known hand body in the trace (RuntimeError). Only these mean "no reward"; any other exception
# propagates -- bounds recovered from the checker configuration fail as AxisBoundsError, never here.
UNFORMABLE_REWARD = (KeyError, ValueError, RuntimeError)


# The state events of a trace, read by the one reader the goal extraction also uses
# (axis_qvla.labels.goals.read_trace_states): missing or unreadable raises OSError, no state event RuntimeError.
trace_states = read_trace_states


def reward_curve(goals: dict, trace_path) -> tuple[np.ndarray | None, str | None]:
    """The 5 Hz reward curve of Eq. (1) and the reward leaf's target body (either may be None).

    The curve is None when the goal record has no reward leaf, when the reward is unformable
    (UNFORMABLE_REWARD), or when it has fewer than two steps -- the attempts the corpus labels "no
    reward". The trace is read first, so its I/O and parse errors propagate instead."""
    leaf = pick_reward_leaf(goals)
    if leaf is None:
        return None, None
    target = leaf.get("resolved_body_name") or leaf.get("tracked_body_name")
    trace_path = pathlib.Path(trace_path)
    trace_states(trace_path)
    try:
        rewards, _hand, _n = compute_attempt_reward_curve(leaf, trace_path)
    except UNFORMABLE_REWARD:
        return None, target
    total = np.asarray(rewards["total"], dtype=np.float64)
    return (total if total.size >= 2 else None), target


def _contact_columns(model, jnames, jrows, obj_pose, obj_names, n_rows, target_body) -> dict:
    """The four per-frame contact booleans, with the target set resolved by stem."""
    f2b = {fj: b for fj in model.free_joints for b in obj_names if stem(fj[:-5]) == stem(b)}
    tset: set[str] = set()
    if target_body is not None and not target_body.startswith(ROBOT_PREFIX):
        tset = resolve_target_bodies(target_body, model.object_bodies)
        if not tset:
            raise ContactError(f"target body {target_body!r} matches no scene body by stem; contact_target_* "
                               f"would be silently all-False")
    out = {c: np.zeros(n_rows, bool) for c in CONTACT_COLUMNS}
    for r in range(n_rows):
        eef, fing, eef_t, fing_t = model.contacts({k: jrows[k][r] for k in jnames},
                                                  {fj: obj_pose[b][r] for fj, b in f2b.items()}, tset)
        out["contact_eef"][r], out["contact_finger"][r] = eef, fing
        out["contact_target_eef"][r], out["contact_target_finger"][r] = eef_t, fing_t
    return out


def label_episode(states: list, goals: dict, *, n_rows: int, trace_path, contact_model=None,
                  fps: float = RENDER_FPS) -> dict:
    """The per-frame columns of one episode of `n_rows` render rows (see the module docstring).
    `states` are the state events of `trace_path` (`trace_states`): the poses come from them, the
    reward from the trace."""
    curve, target = reward_curve(goals, trace_path)
    sim = np.array([s["sim_time"] for s in states], dtype=float)
    anchors = trace_anchor_rows(sim, fps)
    # The trace span can be shorter than the rendered episode; the resamplers clamp outside it, so
    # tail rows repeat the last trace state.
    gen = max(int(anchors[-1]) + 1, n_rows)
    all_names = [b for b in states[0]["body_pose"] if b != "world"]
    obj_names = [b for b in all_names if not b.startswith(ROBOT_PREFIX)]
    jnames = [k for k in states[0]["joint_qpos"] if np.ndim(states[0]["joint_qpos"][k]) == 0]

    def pose_rows(body):
        p = np.array([[*s["body_pose"][body]["pos"], *s["body_pose"][body]["quat"]] for s in states])
        return resample_poses_to_rows(anchors, p, gen)[:n_rows]

    obj_pose = {b: pose_rows(b) for b in obj_names}
    stored_pose = np.stack([pose_rows(b) for b in all_names], axis=1)
    jrows = {k: np.interp(np.clip(np.arange(gen), anchors[0], anchors[-1]), anchors,
                          np.array([s["joint_qpos"][k] for s in states], float))[:n_rows] for k in jnames}
    hand_i = all_names.index(HAND_BODY) if HAND_BODY in all_names else None
    columns = {
        "observation.body_pose": stored_pose,
        "body_names": all_names,
        "observation.joint_qpos": np.stack([jrows[k] for k in jnames], axis=1),
        "joint_names": jnames,
        "observation.eef_pose": stored_pose[:, hand_i, :] if hand_i is not None else pose_rows(HAND_BODY),
        "success": success_from_pass_frame(goals.get("top_first_pass_frame"), anchors, n_rows),
        "is_anchor": anchor_mask(anchors, n_rows),
        "target_body": target,
    }
    if contact_model is not None:
        columns.update(_contact_columns(contact_model, jnames, jrows, obj_pose, obj_names, n_rows, target))
    if curve is None:
        columns["reward"], columns["reward_valid"] = np.zeros(n_rows), np.zeros(n_rows, dtype=bool)
    else:
        columns["reward"] = reward_to_rows(curve, anchors, gen)[:n_rows]
        columns["reward_valid"] = reward_valid_mask(anchors, curve.size, n_rows)
    return columns
