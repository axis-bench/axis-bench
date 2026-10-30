"""One task of the render as a LeRobot v3 dataset, with the published REALALIGN render's feature schema.

Render tier (numpy, pyarrow, pandas, PyAV). Layout `<root>/task_<tid>/` (written as `task_<tid>.partial/`
and renamed when the task is closed): data/chunk-000/file-000.parquet,
videos/observation.images.<key>/chunk-000/file-000.mp4 (h264, yuv420p, crf 30, gop 2, episodes back to
back), meta/info.json, meta/stats.json, meta/tasks.parquet, meta/episodes/chunk-000/file-000.parquet
(with `attempt_id`: every downstream key is (task_id, attempt_id)), and meta/axis_episodes/episode_<i>.json
(the per-episode sidecar the gates read). FEATURES is the published info.json `features` block verbatim.

Two info.json fields name this renderer: `robot_type` is ROBOT_TYPE, `render_profile` is rig.PROFILE_ID,
and `workspace_profile` names the workspace (profile.py).
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from axis_qvla.realrobot.render import rig

CODEBASE_VERSION = "v3.0"
ROBOT_TYPE = "qvla_kinematic_franka_robotiq"
_VIDEO_INFO = {"video.height": rig.HEIGHT, "video.width": rig.WIDTH, "video.codec": "h264", "video.pix_fmt": "yuv420p",
               "video.is_depth_map": False, "video.fps": rig.FPS, "video.channels": 3, "has_audio": False}


def _vec(n: int, name: str) -> dict:
    return {"dtype": "float32", "shape": [n], "names": [name]}


FEATURES: dict = {
    "observation.state": _vec(7, "state"),
    "action": _vec(8, "action"),
    "observation.ee_state": _vec(7, "ee_state"),
    "observation.images.third_person": {"dtype": "video", "shape": [rig.HEIGHT, rig.WIDTH, 3],
                                        "names": ["height", "width", "channel"], "info": dict(_VIDEO_INFO)},
    "observation.cameras.third_person.intrinsics": {"dtype": "float32", "shape": [9], "names": ["row_major"]},
    "observation.cameras.third_person.camera_to_world": {"dtype": "float32", "shape": [16], "names": ["row_major"]},
    "observation.images.wrist": {"dtype": "video", "shape": [rig.HEIGHT, rig.WIDTH, 3],
                                 "names": ["height", "width", "channel"], "info": dict(_VIDEO_INFO)},
    "observation.cameras.wrist.intrinsics": {"dtype": "float32", "shape": [9], "names": ["row_major"]},
    "observation.cameras.wrist.camera_to_world": {"dtype": "float32", "shape": [16], "names": ["row_major"]},
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}
VIDEO_KEYS = tuple(k for k, f in FEATURES.items() if f["dtype"] == "video")
NUMERIC = ("observation.state", "action", "observation.ee_state")


@dataclasses.dataclass(frozen=True)
class EpisodeData:
    attempt_id: int
    state: np.ndarray                         # (T, 7) float32
    action: np.ndarray                        # (T, 8) float32
    ee_state: np.ndarray                      # (T, 7) float32
    camera_to_world: Mapping[str, np.ndarray]  # key -> (T, 16) float32
    sidecar: Mapping


def _fsl(a: np.ndarray) -> pa.FixedSizeListArray:
    a = np.ascontiguousarray(a, np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(a.reshape(-1), pa.float32()), a.shape[1])


def _stats(a: np.ndarray) -> dict:
    a = np.asarray(a, np.float64)
    return {"min": a.min(0).tolist(), "max": a.max(0).tolist(), "mean": a.mean(0).tolist(),
            "std": a.std(0).tolist(), "count": [int(len(a))]}


_TASK_DIR = re.compile(r"task_(\d+)")


def task_dirs(root: Path) -> dict[int, Path]:
    """{task id: directory} of every rendered task in `root`: the `task_<tid>/` directories. A task being
    written is `task_<tid>.partial/` and is not one of them."""
    out = {}
    for p in Path(root).iterdir():
        m = _TASK_DIR.fullmatch(p.name)
        if m is not None and p.is_dir():
            out[int(m.group(1))] = p
    return dict(sorted(out.items()))


class TaskWriter:
    """One task, written into `task_<tid>.partial/` and renamed `task_<tid>/` by `close`, so a task
    directory is always a whole task; `abort` removes the partial directory. A partial directory left by
    an interrupted process is replaced."""

    def __init__(self, root: Path, task_id: int, prompt: str, *, cameras: Iterable[rig.Camera], workspace: str):
        self.dir = Path(root) / f"task_{int(task_id)}"
        if self.dir.exists():
            raise FileExistsError(f"{self.dir} exists; a rendered task is never overwritten")
        self.partial = self.dir.with_name(self.dir.name + ".partial")
        shutil.rmtree(self.partial, ignore_errors=True)
        self.task_id, self.prompt = int(task_id), str(prompt)
        self.cameras, self.workspace = tuple(cameras), str(workspace)
        self.tables: list[pa.Table] = []
        self.episodes: list[dict] = []
        self.numeric: dict[str, list[np.ndarray]] = {k: [] for k in NUMERIC}
        self.total = 0
        self.containers, self.streams = {}, {}
        for key in VIDEO_KEYS:
            path = self.partial / "videos" / key / "chunk-000" / "file-000.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            c = av.open(str(path), "w")
            s = c.add_stream("libx264", rate=rig.FPS)
            s.width, s.height, s.pix_fmt = rig.WIDTH, rig.HEIGHT, "yuv420p"
            s.options = {"crf": "30", "g": "2"}
            self.containers[key], self.streams[key] = c, s

    def add_episode(self, ep: EpisodeData, frames: Iterable[Mapping[str, np.ndarray]]) -> None:
        T, e = len(ep.state), len(self.episodes)
        n = 0
        for frame in frames:
            for key in VIDEO_KEYS:
                vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame[key.rsplit(".", 1)[-1]]), format="rgb24")
                for packet in self.streams[key].encode(vf):
                    self.containers[key].mux(packet)
            n += 1
        if n != T:
            raise ValueError(f"episode {e}: {n} frames for {T} states")
        cols = {"observation.state": _fsl(ep.state), "action": _fsl(ep.action), "observation.ee_state": _fsl(ep.ee_state)}
        for cam in self.cameras:
            cols[f"observation.cameras.{cam.key}.intrinsics"] = _fsl(np.tile(np.asarray(rig.intrinsics(cam), np.float32), (T, 1)))
            cols[f"observation.cameras.{cam.key}.camera_to_world"] = _fsl(ep.camera_to_world[cam.key])
        cols["timestamp"] = pa.array((np.arange(T) / rig.FPS).astype(np.float32))
        cols["frame_index"] = pa.array(np.arange(T, dtype=np.int64))
        cols["episode_index"] = pa.array(np.full(T, e, np.int64))
        cols["index"] = pa.array(np.arange(self.total, self.total + T, dtype=np.int64))
        cols["task_index"] = pa.array(np.zeros(T, np.int64))
        names = [k for k in FEATURES if k not in VIDEO_KEYS]
        self.tables.append(pa.table([cols[k] for k in names], names=names))
        row = {"episode_index": e, "tasks": [self.prompt], "length": T, "data/chunk_index": 0, "data/file_index": 0,
               "dataset_from_index": self.total, "dataset_to_index": self.total + T}
        for key in VIDEO_KEYS:
            row |= {f"videos/{key}/chunk_index": 0, f"videos/{key}/file_index": 0,
                    f"videos/{key}/from_timestamp": self.total / rig.FPS,
                    f"videos/{key}/to_timestamp": (self.total + T) / rig.FPS}
        row |= {"meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0, "attempt_id": int(ep.attempt_id)}
        for k, a in (("observation.state", ep.state), ("action", ep.action), ("observation.ee_state", ep.ee_state)):
            self.numeric[k].append(np.asarray(a))
            row |= {f"stats/{k}/{s}": v for s, v in _stats(a).items()}
        self.episodes.append(row)
        side = self.partial / "meta" / "axis_episodes" / f"episode_{e:06d}.json"
        side.parent.mkdir(parents=True, exist_ok=True)
        side.write_text(json.dumps(dict(ep.sidecar), sort_keys=True), encoding="utf-8")
        self.total += T

    def close(self) -> Path:
        for key in VIDEO_KEYS:
            for packet in self.streams[key].encode():
                self.containers[key].mux(packet)
            self.containers[key].close()
        data = self.partial / "data" / "chunk-000" / "file-000.parquet"
        data.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.concat_tables(self.tables), data)
        meta = self.partial / "meta"
        (meta / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.episodes).to_parquet(meta / "episodes" / "chunk-000" / "file-000.parquet", index=False)
        pd.DataFrame({"task_index": [0]}, index=pd.Index([self.prompt], name="task")).to_parquet(meta / "tasks.parquet")
        stats = {k: _stats(np.concatenate(v)) for k, v in self.numeric.items()}
        (meta / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
        info = {"codebase_version": CODEBASE_VERSION, "robot_type": ROBOT_TYPE, "total_episodes": len(self.episodes),
                "total_frames": self.total, "total_tasks": 1, "chunks_size": 1000, "data_files_size_in_mb": 100,
                "video_files_size_in_mb": 200, "fps": rig.FPS, "splits": {"train": f"0:{len(self.episodes)}"},
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "features": FEATURES, "render_profile": rig.PROFILE_ID,
                "workspace_profile": self.workspace}
        (meta / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        os.rename(self.partial, self.dir)
        return self.dir

    def abort(self) -> None:
        """Close the videos and remove everything written of the task."""
        for c in self.containers.values():
            try:
                c.close()
            except Exception:       # a container that failed mid-write: its file is removed below anyway
                pass
        shutil.rmtree(self.partial, ignore_errors=True)


def video_frame_count(path: Path) -> int:
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        if (s.codec_context.name, s.width, s.height, s.codec_context.pix_fmt) != ("h264", rig.WIDTH, rig.HEIGHT, "yuv420p"):
            raise ValueError(f"{path}: not h264 {rig.WIDTH}x{rig.HEIGHT} yuv420p")
        return sum(1 for _ in c.decode(video=0))


def roots_index(root: Path) -> Path:
    """roots.json over every task directory in `root`, so renders of disjoint selections into one
    directory index together."""
    tids = sorted(task_dirs(root))
    path = Path(root) / "roots.json"
    path.write_text(json.dumps({"version": 2, "base": ".", "roots": {str(t): f"task_{t}" for t in tids}}))
    return path
