"""Approach-aware phase curves of rendered episodes, from the published recordings.

OFFLINE TIER: numpy, scipy, pandas; MuJoCo for contacts.

For each episode of a rendered pretraining-data tree (its roots index; `meta/episodes` gives each
episode's attempt id and length) the label pipeline (axis_qvla.labels.pipeline.label_episode) recomputes the
per-frame labels from the attempt's trace, goal record and recorded scene, on exactly the episode's
frames: the renderer puts recorded row k on frame rint((t_k - t_0) * 15) + 2, and so does the pipeline,
and the render ends on its last row's frame, so an episode whose length is not that is refused by task
and attempt, as is one whose task or attempt the data does not hold.
The approach-aware curve (axis_qvla.labels.phase.phase_reward) binds the task object through the reward
leaf's body and reads float32 poses, as the pretraining data stores them. An episode without a reward,
or whose curve cannot be formed, is neutral: no valid row and no contact onset. The result is an
attempt store, which `render labels` keys by the render's episodes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from axis_qvla.labels.contacts import ContactModel
from axis_qvla.labels.phase import phase_reward
from axis_qvla.labels.pipeline import RENDER_FPS, label_episode, trace_states
from axis_qvla.labels.resample import trace_anchor_rows
from axis_qvla.realrobot.realign.curves import PhaseCurve


def _neutral(n_rows: int) -> PhaseCurve:
    return PhaseCurve(np.zeros(n_rows, np.float32), np.zeros(n_rows, bool), None)


def episode_curve(trace: Path, goals: dict, n_rows: int, contact_model, *, episode: str = "the episode") -> PhaseCurve:
    """ValueError, naming `episode`, when `n_rows` is not the render of this recording: the render ends on
    its last recorded row's frame (axis_qvla.realrobot.render.resample.frame_grid), which the label pipeline
    computes from the trace's sim times (trace_anchor_rows)."""
    states = trace_states(trace)
    last = int(trace_anchor_rows([s["sim_time"] for s in states], RENDER_FPS)[-1])
    if n_rows != last + 1:
        raise ValueError(f"{episode}: rendered {n_rows} frames; its recording's last row is frame {last}, so its "
                         f"render has {last + 1}: the render is not of this recording")
    cols = label_episode(states, goals, n_rows=n_rows, trace_path=trace, contact_model=contact_model)
    valid = np.asarray(cols["reward_valid"], bool)
    if cols["target_body"] is None or not valid.any():
        return _neutral(n_rows)
    frames = pd.DataFrame({
        "reward": np.asarray(cols["reward"], np.float32),
        "contact_target_finger": np.asarray(cols["contact_target_finger"], bool),
        "contact_target_eef": np.asarray(cols["contact_target_eef"], bool),
        "observation.body_pose": list(np.asarray(cols["observation.body_pose"], np.float32).reshape(n_rows, -1)),
        "observation.eef_pose": list(np.asarray(cols["observation.eef_pose"], np.float32)),
    })
    res = phase_reward(frames, {"body_names": list(cols["body_names"]), "object_body": cols["target_body"]})
    if res is None:
        return _neutral(n_rows)
    curve, t0 = res
    return PhaseCurve(np.asarray(curve, np.float32), valid, int(t0))


def _episodes(task_dir: Path) -> list[tuple[int, int]]:
    """(attempt id, length) of every episode of one rendered task, in its meta/episodes order."""
    meta = pd.concat([pd.read_parquet(p, columns=["episode_index", "length", "attempt_id"])
                      for p in sorted((task_dir / "meta" / "episodes").rglob("*.parquet"))], ignore_index=True)
    return [(int(a), int(n)) for a, n in zip(meta["attempt_id"], meta["length"])]


def phase_curves(roots_path: Path, data: Path, tasks=None) -> dict[tuple[int, int], PhaseCurve]:
    from axis_qvla.realrobot.render import data as published

    roots_path, data = Path(roots_path), Path(data)
    doc = json.loads(roots_path.read_text())
    base = roots_path.parent / doc.get("base", ".")
    roots = {int(t): rel for t, rel in doc["roots"].items()}
    chosen = sorted(roots) if tasks is None else sorted({int(t) for t in tasks})
    unrendered = [t for t in chosen if t not in roots]
    if unrendered:
        raise ValueError(f"tasks {unrendered} have no rendered episodes under {roots_path.parent}")
    manifest = published.load_manifest(data)
    absent = [t for t in chosen if str(t) not in manifest["tasks"]]
    if absent:
        raise ValueError(f"task {', '.join(map(str, absent))} is not in the qvla-v1-realworld data at {data}: render and "
                         f"label from the same prepared tree")
    out: dict[tuple[int, int], PhaseCurve] = {}
    for tid in chosen:
        rel = roots[tid]
        variant = {int(e["attempt_id"]): int(e["variant"]) for e in manifest["tasks"][str(tid)]["episodes"]}
        models: dict[int, ContactModel] = {}
        for aid, n_rows in _episodes(base / rel):
            if aid not in variant:
                raise ValueError(f"task {tid} attempt {aid} is not in the qvla-v1-realworld data at {data}: render and "
                                 f"label from the same prepared tree")
            v = variant[aid]
            if v not in models:
                models[v] = ContactModel(str(data / published.scene_rel(tid, v)))
            goals = json.loads((data / published.goals_rel(tid, aid)).read_text())
            out[(tid, aid)] = episode_curve(data / published.trace_rel(tid, aid), goals, n_rows, models[v],
                                            episode=f"task {tid} attempt {aid}")
    return out
