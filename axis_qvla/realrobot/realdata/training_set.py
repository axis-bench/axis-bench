"""DROID-layout demonstrations in the layout the π0.5 loader reads: one LeRobot v3 dataset per instruction.

The input is a LeRobot v2.1 recording in the DROID layout (openpi's examples/droid converter writes it):
`exterior_image_1_left` and `wrist_image_left` frames in the parquet, `joint_position` (7, rad),
`gripper_position` (1), and `actions` (8): DROID's normalised joint-velocity command in [-1, 1] at 15 Hz
(the controller's target is q + 0.2 * v per step) and the absolute gripper position command. Each
instruction becomes `task_<k>/` (k = 1, 2, ... in sorted instruction order) with
`observation.images.third_person` and `observation.images.wrist` as h264 video at the recorded size,
`observation.state` = `[q0..q6, gripper]` (the model's order, conf/embodiments/fr3_droid.toml) and
`action` as recorded; beside them `roots.json` and `ranges.json`, the tail-10 rule of the stage-2
fine-tuning (the last 10 frames of an episode start no action chunk). The frames keep their 16:9
geometry; training and serving crop the centre square. Episodes are read and encoded one at a time.

Offline tier: numpy, pyarrow, pandas; PyAV and Pillow inside the functions that use them.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from axis_qvla.realrobot.realdata.lerobot_v21 import Dataset, RealdataError, decode_png
from axis_qvla.realrobot.realdata.tail import TAIL, tail_ranges

DROID_COLUMNS = ("exterior_image_1_left", "wrist_image_left", "joint_position", "gripper_position", "actions")
CAMERAS = {"exterior_image_1_left": "observation.images.third_person", "wrist_image_left": "observation.images.wrist"}
FPS = 15
STORED = (320, 180)     # width x height: conf/embodiments/fr3_droid.toml images.stored


def _vec(n: int, name: str) -> dict:
    return {"dtype": "float32", "shape": [n], "names": [name]}


def _fsl(a: np.ndarray) -> pa.FixedSizeListArray:
    a = np.ascontiguousarray(a, np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(a.reshape(-1), pa.float32()), a.shape[1])


def _read_episode(ds: Dataset, episode_index: int) -> tuple[np.ndarray, np.ndarray, pa.Table]:
    """(state in the model's order, action as recorded, the episode's table with its PNG frames)."""
    t = ds.read_episode(episode_index, list(DROID_COLUMNS))
    q = np.stack(t.column("joint_position").to_pylist()).astype(np.float32)
    g = np.stack(t.column("gripper_position").to_pylist()).astype(np.float32).reshape(-1, 1)
    return np.concatenate([q, g], axis=1), np.stack(t.column("actions").to_pylist()).astype(np.float32), t


def _check_size(pixels: np.ndarray, episode_index: int, column: str) -> None:
    if pixels.shape[:2] != (STORED[1], STORED[0]):
        raise RealdataError(f"episode {episode_index}: {column} frames are {pixels.shape[1]} x {pixels.shape[0]}; "
                            f"the fr3_droid runs train on {STORED[0]} x {STORED[1]}, the frames openpi's DROID "
                            f"converter writes")


def _write_task(task_dir: Path, prompt: str, ds: Dataset, episode_indices: list[int]) -> list[int]:
    """One instruction's episodes as `task_dir`, read, encoded and released one episode at a time, so
    memory holds one episode's PNG frames whatever the recording's size. Returns their lengths."""
    import av
    w, h = STORED
    videos = {}
    for key in CAMERAS.values():
        p = task_dir / "videos" / key / "chunk-000" / "file-000.mp4"
        p.parent.mkdir(parents=True, exist_ok=True)
        c = av.open(str(p), "w")
        s = c.add_stream("libx264", rate=FPS)
        s.width, s.height, s.pix_fmt, s.options = w, h, "yuv420p", {"crf": "23", "g": "2"}
        videos[key] = (c, s)
    tables, rows, lengths, total = [], [], [], 0
    for e, episode_index in enumerate(episode_indices):
        state, action, t = _read_episode(ds, episode_index)
        T = len(state)
        for column, key in CAMERAS.items():
            c, s = videos[key]
            for cell in t.column(column).to_pylist():
                pixels = decode_png(cell["bytes"])
                _check_size(pixels, episode_index, column)
                frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(pixels), format="rgb24")
                for packet in s.encode(frame):
                    c.mux(packet)
        tables.append(pa.table({"observation.state": _fsl(state), "action": _fsl(action),
                                "timestamp": pa.array((np.arange(T) / FPS).astype(np.float32)),
                                "frame_index": pa.array(np.arange(T, dtype=np.int64)),
                                "episode_index": pa.array(np.full(T, e, np.int64)),
                                "index": pa.array(np.arange(total, total + T, dtype=np.int64)),
                                "task_index": pa.array(np.zeros(T, np.int64))}))
        row = {"episode_index": e, "tasks": [prompt], "length": T, "data/chunk_index": 0, "data/file_index": 0,
               "dataset_from_index": total, "dataset_to_index": total + T, "meta/episodes/chunk_index": 0,
               "meta/episodes/file_index": 0}
        for key in CAMERAS.values():
            row |= {f"videos/{key}/chunk_index": 0, f"videos/{key}/file_index": 0,
                    f"videos/{key}/from_timestamp": total / FPS, f"videos/{key}/to_timestamp": (total + T) / FPS}
        rows.append(row)
        lengths.append(T)
        total += T
    for c, s in videos.values():
        for packet in s.encode():
            c.mux(packet)
        c.close()
    (task_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.concat_tables(tables), task_dir / "data" / "chunk-000" / "file-000.parquet")
    meta = task_dir / "meta"
    (meta / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(meta / "episodes" / "chunk-000" / "file-000.parquet", index=False)
    pd.DataFrame({"task_index": [0]}, index=pd.Index([prompt], name="task")).to_parquet(meta / "tasks.parquet")
    video = {"dtype": "video", "shape": [int(h), int(w), 3], "names": ["height", "width", "channel"],
             "info": {"video.height": int(h), "video.width": int(w), "video.codec": "h264", "video.pix_fmt": "yuv420p",
                      "video.is_depth_map": False, "video.fps": FPS, "video.channels": 3, "has_audio": False}}
    features = {"observation.state": _vec(8, "state"), "action": _vec(8, "action"),
                "observation.images.third_person": video, "observation.images.wrist": dict(video),
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                **{k: {"dtype": "int64", "shape": [1], "names": None} for k in ("frame_index", "episode_index", "index", "task_index")}}
    (meta / "info.json").write_text(json.dumps({
        "codebase_version": "v3.0", "robot_type": "fr3_droid", "total_episodes": len(lengths), "total_frames": total,
        "total_tasks": 1, "chunks_size": 1000, "data_files_size_in_mb": 100, "video_files_size_in_mb": 200, "fps": FPS,
        "splits": {"train": f"0:{len(lengths)}"}, "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4", "features": features}, indent=2))
    return lengths


def write_training_set(dataset: Path, out: Path, *, tail: int = TAIL) -> dict:
    """Write `dataset` (DROID layout, LeRobot v2.1) into the new directory `out` in the training layout;
    returns `{"tasks", "episodes", "frames"}`. A recording that is not in the DROID layout, at 15 fps and
    at 320 x 180 is refused before anything is written; a later frame of another size is refused by
    episode and the half-written `out` removed."""
    if tail < 0:
        raise RealdataError(f"--tail {tail}: the frames that start no chunk number 0 or more")
    ds = Dataset.open(dataset)
    missing = [c for c in DROID_COLUMNS if c not in ds.features]
    if missing:
        raise RealdataError(f"{Path(dataset).name}: no {', '.join(missing)}; give a recording in the DROID layout "
                            f"({', '.join(DROID_COLUMNS)})")
    if ds.fps != FPS:
        raise RealdataError(f"{Path(dataset).name}: {ds.fps:g} fps; the FR3 runs train at {FPS}")
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists; write the training set into a new directory")
    first = int(ds.episodes[0]["episode_index"])
    for column in CAMERAS:
        _check_size(decode_png(ds.read_episode(first, [column]).column(0)[0].as_py()["bytes"]), first, column)
    by_prompt: dict[str, list[int]] = {}
    for e in ds.episodes:
        by_prompt.setdefault(str(e["tasks"][0]), []).append(int(e["episode_index"]))
    roots, ranges, episodes, frames = {}, {}, 0, 0
    try:
        for k, (prompt, indices) in enumerate(sorted(by_prompt.items()), start=1):
            lengths = _write_task(out / f"task_{k}", prompt, ds, indices)
            roots[str(k)] = f"task_{k}"
            spans = tail_ranges(dict(enumerate(lengths)), tail)
            ranges |= {f"task_{k}--{e}": spans[str(e)] for e in range(len(lengths))}
            episodes, frames = episodes + len(lengths), frames + sum(lengths)
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)     # `out` is new: nothing of a half-written set is kept
        raise
    (out / "roots.json").write_text(json.dumps({"version": 2, "base": ".", "roots": roots}))
    (out / "ranges.json").write_text(json.dumps(ranges))
    return {"tasks": len(roots), "episodes": episodes, "frames": frames}
