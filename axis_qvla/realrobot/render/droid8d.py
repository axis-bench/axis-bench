"""The DROID-8D training view of a rendered corpus.

Render tier (numpy, pyarrow). state = 7 joints + closedness, action = 7 joint velocities (forward
difference x 15, zero on each episode's last frame, clipped to +-1 rad/s) + closedness, closedness =
gripper ctrl / 255; the layouts and the clip are rig's (DROID8D_STATE, DROID8D_ACTION, VELOCITY_CLIP),
which the FR3 checks read. Videos are a RELATIVE symlink to the render's. meta is copied with the two
8-wide shapes patched and the state and action statistics recomputed from the converted arrays, in
meta/stats.json and in each episode's `stats/` columns: a LeRobot consumer builds its normalisers from them.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from axis_qvla.data.convert_droid_actions import joint_velocity
from axis_qvla.realrobot.render import rig, writer


def to_droid8d(state7: np.ndarray, action8: np.ndarray, episode_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    st, ac = np.asarray(state7, np.float64), np.asarray(action8, np.float64)
    closed = ac[:, 7] / 255.0
    vel = np.clip(joint_velocity(st, np.asarray(episode_index), float(rig.FPS)).astype(np.float64),
                  -rig.VELOCITY_CLIP, rig.VELOCITY_CLIP)
    return (np.concatenate([st, closed[:, None]], 1).astype(np.float32),
            np.concatenate([vel, closed[:, None]], 1).astype(np.float32))


def convert_task(src: Path, dst: Path) -> dict:
    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise FileExistsError(f"{dst} exists; write the training view into a new directory")
    rows = clipped = 0
    converted: dict[str, list[np.ndarray]] = {"observation.state": [], "action": []}
    episodes: list[np.ndarray] = []
    for f in sorted((src / "data").rglob("*.parquet")):
        t = pq.read_table(f)
        st = np.asarray(t.column("observation.state").combine_chunks().flatten()).reshape(-1, 7)
        ac = np.asarray(t.column("action").combine_chunks().flatten()).reshape(-1, 8)
        s8, a8 = to_droid8d(st, ac, t.column("episode_index").to_numpy())
        converted["observation.state"].append(s8)
        converted["action"].append(a8)
        episodes.append(t.column("episode_index").to_numpy())
        clipped += int((np.abs(a8[:, :7]) >= 1.0).sum())
        t = t.set_column(t.schema.get_field_index("observation.state"), "observation.state", writer._fsl(s8))
        t = t.set_column(t.schema.get_field_index("action"), "action", writer._fsl(a8))
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, out)
        rows += len(t)
    shutil.copytree(src / "meta", dst / "meta")
    (dst / "videos").symlink_to(os.path.relpath(src / "videos", dst), target_is_directory=True)
    info = json.loads((dst / "meta" / "info.json").read_text())
    for k in ("observation.state", "action"):
        info["features"][k]["shape"] = [8]
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    arrays = {k: np.concatenate(v) for k, v in converted.items()}
    _restat(dst / "meta", arrays, np.concatenate(episodes))
    return {"rows": rows, "velocity_at_bound": clipped}


def _restat(meta: Path, arrays: dict[str, np.ndarray], episode_index: np.ndarray) -> None:
    """Replace the statistics of each feature in `arrays` with the writer's own statistics of the
    converted values: meta/stats.json for the whole task, and every meta/episodes row's
    `stats/<feature>/<stat>` columns for its episode. Every other feature's statistics are left as
    copied."""
    path = meta / "stats.json"
    stats = json.loads(path.read_text(encoding="utf-8"))
    stats.update({k: writer._stats(a) for k, a in arrays.items()})
    path.write_text(json.dumps(stats, indent=1), encoding="utf-8")
    for f in sorted((meta / "episodes").rglob("*.parquet")):
        df = pd.read_parquet(f)
        per = {int(e): {k: writer._stats(a[episode_index == e]) for k, a in arrays.items()}
               for e in df["episode_index"]}
        for k in arrays:
            for stat in writer._stats(arrays[k][:1]):
                # A whole column from a list: a per-cell assignment squeezes the 1-long count.
                df[f"stats/{k}/{stat}"] = [per[int(e)][k][stat] for e in df["episode_index"]]
        df.to_parquet(f, index=False)


def build_corpus(src_root: Path, dst_root: Path) -> Path:
    tasks = writer.task_dirs(src_root)
    if not tasks:
        raise FileNotFoundError(f"no task_* directories under {src_root}")
    Path(dst_root).mkdir(parents=True, exist_ok=True)
    for tid, src in tasks.items():
        convert_task(src, Path(dst_root) / f"task_{tid}")
    return writer.roots_index(dst_root)
