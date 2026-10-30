"""Scores someone else computed, as a Scorer's output: per-frame values for every corpus episode.

`from_json` reads an AWR-weights-shaped file ({..., "weights": {"task_<id>--<ep>": [one value per
frame]}}), the format every paper signal derives from; `from_parquet` reads the columns
`task_id, episode_index, frame_index, <column>` and requires every frame of every episode of the
row space. Either way the sha256 of the bytes read is recorded: a bundle names what it read.

Offline tier: numpy, pandas.
"""
from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import pathlib
from collections.abc import Mapping
from typing import Any

import numpy as np

from axis_qvla.signals.blocks.awr import episode_key
from axis_qvla.signals.program import EpisodeValues
from axis_qvla.signals.rowspace import episode_keys
from axis_qvla.signals.schema import RowSpace


@dataclasses.dataclass(frozen=True, eq=False)
class ImportedScores:
    payload: Mapping[str, Any]
    sha256: str
    name: str

    @classmethod
    def from_json(cls, path, *, name: str | None = None, expect_sha256: str | None = None) -> "ImportedScores":
        data = pathlib.Path(path).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if expect_sha256 is not None and digest != expect_sha256:
            raise ValueError(f"{path} has sha256 {digest}, not the expected {expect_sha256}")
        payload = json.loads(data)
        if not (isinstance(payload, dict) and isinstance(payload.get("weights"), dict)):
            raise ValueError(f"{path} is not a scores file: no 'weights' mapping of episode -> per-frame values")
        return cls(payload, digest, name or pathlib.Path(path).name)

    @classmethod
    def from_parquet(cls, path, rows: RowSpace, *, reward_id: str, column: str = "score") -> "ImportedScores":
        import pandas as pd

        data = pathlib.Path(path).read_bytes()
        df = pd.read_parquet(io.BytesIO(data), columns=["task_id", "episode_index", "frame_index", column])
        df = df.sort_values(["task_id", "episode_index", "frame_index"], kind="stable")
        lengths = dict(zip(episode_keys(rows), np.asarray(rows.arrays["episode_length"]).tolist()))
        found: dict[str, list[float]] = {}
        for (t, e), g in df.groupby(["task_id", "episode_index"], sort=True):
            key = episode_key(t, e)
            if key not in lengths:
                raise ValueError(f"{path}: {key} is not an episode of frame index {rows.name}")
            if g["frame_index"].tolist() != list(range(lengths[key])):
                raise ValueError(f"{path}: {key} does not give one score for each of its {lengths[key]} frames")
            found[key] = [float(x) for x in g[column]]
        missing = [k for k in lengths if k not in found]
        if missing:
            raise ValueError(f"{path}: {len(missing)} episodes have no frames scored (first: {missing[0]})")
        weights = {k: found[k] for k in lengths}
        payload = {"version": 1, "reward_id": reward_id, "n_episodes": len(weights),
                   "n_rows": int(sum(lengths.values())), "weights": weights}
        return cls(payload, hashlib.sha256(data).hexdigest(), pathlib.Path(path).name)

    @property
    def header(self) -> dict:
        return {k: v for k, v in self.payload.items() if k != "weights"}

    def values(self) -> EpisodeValues:
        return EpisodeValues({k: np.asarray(v, dtype=np.float64) for k, v in self.payload["weights"].items()},
                             header=self.header)
