"""Trainable ranges for the REALALIGN corpus: every frame but the last `tail` of each episode.

OFFLINE TIER: pandas only.

The sim benchmark's ranges come from an idle filter over the stored state (`axis_qvla.labels.idle_filter`).
The real-world line trims a uniform tail instead: `[[0, T - 10]]` per episode, which also keeps every
horizon-10 window inside the episode. Format is the ranges JSON every signal builder reads:
`{"task_<id>--<ep>": [[start, end)]}`.
"""
from __future__ import annotations

import json
from pathlib import Path

from axis_qvla.signals.methods import schedule as bis
from axis_qvla.signals.blocks.awr import episode_key

TAIL = 10


def tail_ranges(roots_path: Path, tail: int = TAIL) -> dict[str, list[list[int]]]:
    """`[[0, max(T - tail, 0)]]` for every episode the roots index enumerates."""
    out: dict[str, list[list[int]]] = {}
    for tid, root in bis._ordered_roots(roots_path):
        episodes, _total = bis._episode_offsets(root)
        for ep, (_start, length) in sorted(episodes.items()):
            out[episode_key(int(tid), int(ep))] = [[0, max(int(length) - int(tail), 0)]]
    return out


def write_ranges(path: Path, ranges: dict[str, list[list[int]]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(ranges))
