"""A metadata-only corpus view: just enough of a LeRobot corpus for the signal tools to index it.

OFFLINE TIER: numpy, pandas, pyarrow.

The signal builders never read frames. They place every episode in the flat row space from
`meta/episodes` (`episode_index`, `dataset_from_index`, `length`), bind artifacts to the corpus through
its fingerprint (`episode_index`, `length`, `attempt_id`), and read prompts from `meta/tasks.parquet`.
A user who rendered the corpus points the tools at it directly. Labelling export-keyed curves without the
frames needs only this view, built from the alignment map. It uses the same layout as a rendered corpus,
so the rows it enumerates are the rows a loader over the real corpus enumerates.
"""
from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

_MAP_KEY = re.compile(r"task_(\d+)--(\d+)")


@dataclasses.dataclass(frozen=True)
class ViewEpisode:
    task_id: int
    episode_index: int
    length: int
    attempt_id: int
    prompt: str


def write_meta_view(out_dir: Path, episodes: list[ViewEpisode]) -> Path:
    """Write `out_dir/view/task_<tid>/meta/...` and `out_dir/roots.json` (v2, relative base)."""
    out = Path(out_dir)
    by_task: dict[int, list[ViewEpisode]] = {}
    for e in episodes:
        by_task.setdefault(int(e.task_id), []).append(e)
    roots = {}
    for tid, eps in sorted(by_task.items()):
        eps = sorted(eps, key=lambda e: e.episode_index)
        lengths = np.array([e.length for e in eps], dtype=np.int64)
        meta = out / "view" / f"task_{tid}" / "meta"
        (meta / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "episode_index": [int(e.episode_index) for e in eps],
            "dataset_from_index": np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64),
            "length": lengths,
            "attempt_id": [int(e.attempt_id) for e in eps],
        }).to_parquet(meta / "episodes" / "chunk-000" / "file-000.parquet")
        pd.DataFrame({"task": sorted({e.prompt for e in eps})}).to_parquet(meta / "tasks.parquet")
        roots[str(tid)] = f"task_{tid}"
    roots_path = out / "roots.json"
    roots_path.write_text(json.dumps({"version": 2, "base": "view", "roots": roots}))
    return roots_path


def view_episodes_from_map(alignment_map: Path, prompts: dict[int, str]) -> list[ViewEpisode]:
    """One `ViewEpisode` per training key of the alignment map, sorted by (task, episode)."""
    doc = json.loads(Path(alignment_map).read_text())
    out = []
    for key, v in doc["episodes"].items():
        m = _MAP_KEY.fullmatch(key)
        if m is None:
            raise ValueError(f"alignment-map key {key!r} is not task_<id>--<episode>")
        tid = int(m.group(1))
        if tid not in prompts:
            raise KeyError(f"no prompt for task {tid}")
        out.append(ViewEpisode(tid, int(m.group(2)), int(v["train_frame_count"]), int(v["attempt_id"]),
                               prompts[tid]))
    return sorted(out, key=lambda e: (e.task_id, e.episode_index))
