"""The toy corpus of axis_qvla.signals.testing written to disk, for tests that run the legacy builders.

task_800 has episodes of 6, 4 and 5 frames, task_1000 one of 8; the ranges keep 9 rows. The weights
file carries the header the schedule and tag builders read.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib

import pandas as pd

from axis_qvla.signals import rowspace as rsp
from axis_qvla.signals.schema import RowSpace

RANGES = {"task_1000--0": [[0, 3], [6, 100]], "task_800--0": [[1, 3]], "task_800--1": [[0, 2]]}
WEIGHTS = {"task_800--0": [2.0, 1.0, 0.5, 2.0, 2.0, 2.0], "task_800--1": [1.5, 1.5, 0.5, 0.5],
           "task_800--2": [1.0] * 5, "task_1000--0": [1.0, 1.25, 0.75, 1.0, 1.0, 1.0, 1.5, 2.0]}
EPISODES = ((800, [(0, 0, 6, 11), (1, 6, 4, 12), (2, 10, 5, 13)], "pick up the bowl"),
            (1000, [(0, 0, 8, 21)], "open the drawer"))


@dataclasses.dataclass(frozen=True)
class ToyCorpus:
    root: pathlib.Path
    roots: pathlib.Path
    ranges: pathlib.Path
    weights: pathlib.Path
    rs: RowSpace


def write_toy_corpus(root: pathlib.Path, *, weights: dict | None = None) -> ToyCorpus:
    for tid, eps, prompt in EPISODES:
        meta = root / "corpus" / f"task_{tid}" / "meta"
        (meta / "episodes" / "chunk-000").mkdir(parents=True)
        pd.DataFrame([{"episode_index": e, "dataset_from_index": f, "length": n, "attempt_id": a}
                      for e, f, n, a in eps]).to_parquet(meta / "episodes" / "chunk-000" / "file-000.parquet")
        pd.DataFrame({"task": [prompt]}).to_parquet(meta / "tasks.parquet")
    (root / "index").mkdir()
    roots = root / "index" / "roots_toy.json"
    roots.write_text(json.dumps({"version": 2, "base": "../corpus",
                                 "roots": {"1000": "task_1000", "800": "task_800"}}, indent=1))
    ranges = root / "index" / "ranges_toy.json"
    ranges.write_text(json.dumps(RANGES))
    path = root / "w.json"
    path.write_text(json.dumps(weights or {"version": 1, "reward_id": "v2", "tau": 10.0, "cap": 2.0, "horizon": 10,
                                           "n_episodes": 4, "n_rows": 23, "weights": WEIGHTS}))
    rs = rsp.build_rowspace(name="toy", corpus=root / "corpus", roots_file=roots, ranges_file=ranges,
                            roots_path="index/roots_toy.json", ranges_path="index/ranges_toy.json", source="test")
    return ToyCorpus(root, roots, ranges, path, rs)
