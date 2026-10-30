"""Relabel recorded episodes with a reward of your own, and turn it into AWR weights.

docs/tutorials/relabel-with-a-new-reward.md walks through this file. For every episode of a traces
sample (qvla-v1-corpus/traces_sample/, or a synthetic one), the label pipeline
(axis_qvla.labels.pipeline.label_episode) recomputes the published per-frame columns from the raw
recording. `reach_reward` below is the new reward, computed from those columns. The AWR block
(axis_qvla.signals.blocks.awr) turns it into per-frame weights, written as a scores file that any method
reads with axis_qvla.signals.imported.ImportedScores.

    python experiments/examples/relabel.py --traces <traces_sample dir> --out reach_weights.json

Prints one line per episode (its key, its rows, the published reward's mean and the new reward's
mean), then the scores file's header.

Offline tier: numpy, scipy.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from axis_qvla.labels.contacts import HAND_BODY
from axis_qvla.labels.pipeline import label_episode, reward_curve, trace_states
from axis_qvla.signals.blocks.awr import build_episode_weights, neutral_episode, write_manifest
from axis_qvla.signals.imported import ImportedScores

REWARD_ID = "example.reach"


def reach_reward(cols: dict, target: str) -> np.ndarray:
    """The new reward: 1 - tanh(5 d), d the hand-to-object distance in m, on every render row."""
    names = list(cols["body_names"])
    pose = cols["observation.body_pose"]                      # (rows, bodies, 7): xyz, then a quaternion
    d = np.linalg.norm(pose[:, names.index(HAND_BODY), :3] - pose[:, names.index(target), :3], axis=1)
    return (1.0 - np.tanh(5.0 * d)).astype(np.float64)


def relabel(sample: pathlib.Path):
    """(task id, episode index, published reward, new reward or None, validity) per sampled episode."""
    manifest = json.loads((sample / "manifest.json").read_text())
    for task in manifest["tasks"]:
        for ep in task["episodes"]:
            name = f"task_{task['task_id']}_attempt_{ep['attempt_id']}"
            trace = sample / "traces" / f"{name}.jsonl"
            goals = json.loads((sample / "goals_per_attempt" / f"{name}.json").read_text())
            cols = label_episode(trace_states(trace), goals, n_rows=ep["length"], trace_path=trace)
            _, target = reward_curve(goals, trace)
            new = None if target is None else reach_reward(cols, target)
            yield task["task_id"], ep["episode_index"], cols["reward"], new, cols["reward_valid"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--traces", type=pathlib.Path, required=True,
                    help="a traces sample directory (manifest.json, traces/, goals_per_attempt/)")
    ap.add_argument("--out", type=pathlib.Path, required=True, help="the scores file to write (JSON)")
    args = ap.parse_args(argv)
    weights = []
    for tid, idx, published, new, valid in relabel(args.traces):
        if new is None:                                       # no reward leaf: every row at the BC weight
            weights.append(neutral_episode(tid, idx, len(published)))
            print(f"task_{tid}--{idx}  {len(published)} rows  no reward: neutral")
            continue
        weights.append(build_episode_weights(tid, idx, new, valid))
        print(f"task_{tid}--{idx}  {len(new)} rows  published {published.mean():.4f}  reach {new.mean():.4f}")
    header = write_manifest(args.out, iter(weights), reward_id=REWARD_ID, tau=10.0, cap=2.0, horizon=10)
    ImportedScores.from_json(args.out)                        # what a method reads; refuses a malformed file
    print(json.dumps(header, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
