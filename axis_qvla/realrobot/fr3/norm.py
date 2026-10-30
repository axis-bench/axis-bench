"""Norm statistics computed on a run's own training set, in the model's state order.

Three readers, one per embodiment, each over every frame of the dataset:
- `fr3`: a LeRobot v2.1 dataset (the idle-filtered FR3 export), opened and held to the fr3 embodiment
  first (axis_qvla.realrobot.fr3.dataset.open_set: fps, widths, camera columns, frame shape). `state` is
  stored as [gripper, q0..q6] and permuted to the model's [q0..q6, gripper] through the one declared
  adapter (axis_qvla.realrobot.realdata.state_order.STORED_TO_MODEL); `actions` keep their stored order;
- `realalign_sim`: the rendered corpus's DROID-8D view, `observation.state` / `action` of every task
  under its roots.json, already in model order;
- `fr3_droid`: your demonstrations in the training layout (`realdata training-set`), read as
  `realalign_sim` is: every task under its roots.json, already in model order.
Either way the data read must be as wide as the embodiment's state and action layouts, whatever the
metadata says.
The file has openpi's layout, {"norm_stats": {"state": ..., "actions": ...}}, with mean, std, q01 and
q99 per dimension. The quantiles are numpy's exact ones, not openpi's histogram estimate, so a
consumer that recomputes them agrees within its histogram's resolution.

Offline tier (numpy, pyarrow).
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

READERS = ("fr3", "realalign_sim", "fr3_droid")


def _fsl(table, name: str, width: int, emb_id: str) -> np.ndarray:
    col = table.column(name).combine_chunks()
    x = np.asarray(col.flatten(), np.float64).reshape(len(col), -1)
    if x.shape[1] != width:
        raise ValueError(f"its {name} are {x.shape[1]} wide, the {emb_id} embodiment's are {width}")
    return x


def columns(dataset_dir: Path, embodiment: str) -> tuple[np.ndarray, np.ndarray]:
    """(state, actions) over every frame of the training set, state in the model's order. ValueError,
    naming the directory, when it is not a training set of that embodiment."""
    from axis_qvla.realrobot.fr3.dataset import DatasetError
    from axis_qvla.realrobot.fr3.embodiment import load_embodiment

    if embodiment not in READERS:
        raise ValueError(f"no norm reader for embodiment {embodiment!r} (readers: {', '.join(READERS)})")
    emb = load_embodiment(embodiment)
    try:
        return _read(Path(dataset_dir), emb)
    except DatasetError:
        raise
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f"{dataset_dir}: cannot read the {embodiment} training set "
                         f"({exc.__class__.__name__}: {exc})") from None


def _read(dataset_dir: Path, emb) -> tuple[np.ndarray, np.ndarray]:
    ns, na = len(emb.state_model), len(emb.action_stored)
    if emb.id == "fr3":
        from axis_qvla.realrobot.fr3.dataset import open_set
        from axis_qvla.realrobot.realdata.state_order import STORED_TO_MODEL
        ds = open_set(dataset_dir, emb)
        tables = [ds.read_episode(int(e["episode_index"]), ["state", "actions"]) for e in ds.episodes]
        return (STORED_TO_MODEL(np.concatenate([_fsl(t, "state", ns, emb.id) for t in tables])),
                np.concatenate([_fsl(t, "actions", na, emb.id) for t in tables]))
    if emb.id in ("realalign_sim", "fr3_droid"):
        roots = json.loads((dataset_dir / "roots.json").read_text(encoding="utf-8"))
        if not (isinstance(roots, Mapping) and isinstance(roots.get("roots"), Mapping)
                and all(isinstance(v, str) for v in roots["roots"].values())
                and isinstance(roots.get("base", "."), str)):
            raise ValueError(f"{dataset_dir / 'roots.json'} is not a roots index: a table whose `roots` maps each "
                             f"task id to its directory (a string)")
        base = dataset_dir / roots.get("base", ".")
        files = [f for rel in roots["roots"].values() for f in _parquet_files(base / rel / "data")]
        tables = [pq.read_table(f, columns=["observation.state", "action"]) for f in files]
        return (np.concatenate([_fsl(t, "observation.state", ns, emb.id) for t in tables]),
                np.concatenate([_fsl(t, "action", na, emb.id) for t in tables]))
    raise AssertionError(emb.id)


def _parquet_files(data_dir: Path) -> list[Path]:
    """Every parquet file under one task's data/, sorted. os.walk with an onerror that raises: an
    unreadable directory is an error, never a task with fewer files (Path.rglob skips it silently)."""
    def fail(exc: OSError) -> None:
        raise exc

    files = sorted(Path(root) / f for root, _, names in os.walk(data_dir, onerror=fail)
                   for f in names if f.endswith(".parquet"))
    if not files:
        raise ValueError(f"{data_dir} holds no parquet files")
    return files


def _stats(x: np.ndarray) -> dict:
    return {"mean": x.mean(0).tolist(), "std": x.std(0).tolist(),
            "q01": np.quantile(x, 0.01, axis=0).tolist(), "q99": np.quantile(x, 0.99, axis=0).tolist()}


def compute_norm(dataset_dir: Path, embodiment: str) -> dict:
    state, actions = columns(dataset_dir, embodiment)
    return {"norm_stats": {"state": _stats(state), "actions": _stats(actions)}}


def write_norm(path: Path, stats: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
