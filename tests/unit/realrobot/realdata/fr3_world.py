"""A synthetic FR3-shaped LeRobot v2.1 dataset, written under a test's tmp_path.

Same layout as the published FR3 data: state[8] = [gripper, q0..q6], actions[8] =
[v0..v6, gripper_cmd], PNG frames inside the parquet under `image` / `extra_view_image`, and the
`huggingface` schema metadata LeRobot requires.
"""
from __future__ import annotations

import io
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from axis_qvla.realrobot.realdata.lerobot_v21 import episode_stats

FPS = 15
TASKS = ("pick up the cup", "press the button")
IMAGE_COLUMNS = ("image", "extra_view_image")
_IMG = pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])

Episode = tuple[np.ndarray, np.ndarray, int]   # state (T, 8), actions (T, 8), task_index


def png(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buf, format="PNG")
    return buf.getvalue()


def noise_image(episode: int, row: int, camera: int, shape: tuple[int, int, int] = (16, 16, 3)) -> np.ndarray:
    """A distinct, deterministic frame per (episode, row, camera)."""
    return np.random.default_rng([episode, row, camera]).integers(0, 256, size=shape, dtype=np.uint8)


def _features(image_shape: tuple[int, int, int], images: bool) -> dict:
    flag = {"dtype": "bool", "shape": [1]}
    out = {"state": {"dtype": "float32", "shape": [8], "names": ["state"]},
           "actions": {"dtype": "float32", "shape": [8], "names": ["actions"]},
           **{k: {**flag, "names": [k]} for k in ("done", "is_success", "intervene_flag")}}
    if images:
        out.update({c: {"dtype": "image", "shape": list(image_shape), "names": ["height", "width", "channel"]}
                    for c in IMAGE_COLUMNS})
    out["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
    out.update({k: {"dtype": "int64", "shape": [1], "names": None}
                for k in ("frame_index", "episode_index", "index", "task_index")})
    return out


def _schema(images: bool) -> pa.Schema:
    vec = pa.list_(pa.float32(), 8)
    fields = [pa.field("state", vec), pa.field("actions", vec), pa.field("done", pa.bool_()),
              pa.field("is_success", pa.bool_()), pa.field("intervene_flag", pa.bool_())]
    if images:
        fields += [pa.field(c, _IMG) for c in IMAGE_COLUMNS]
    fields += [pa.field("timestamp", pa.float32())]
    fields += [pa.field(k, pa.int64()) for k in ("frame_index", "episode_index", "index", "task_index")]
    hf = {"info": {"features": {f.name: {"_type": "Image" if f.type == _IMG else "Value"} for f in fields}}}
    return pa.schema(fields, metadata={b"huggingface": json.dumps(hf).encode()})


def _vec(x: np.ndarray) -> pa.Array:
    return pa.FixedSizeListArray.from_arrays(pa.array(np.asarray(x, dtype=np.float32).reshape(-1)), 8)


def write_fr3_dataset(root: Path, episodes: Sequence[Episode], *, images: bool = True,
                      image_shape: tuple[int, int, int] = (16, 16, 3),
                      image_of: Callable[[int, int, int], np.ndarray] | None = None) -> Path:
    """Write `episodes` as a LeRobot v2.1 dataset at `root` (which must not exist); return `root`."""
    root = Path(root)
    (root / "meta").mkdir(parents=True)
    feats, schema = _features(image_shape, images), _schema(images)
    image_of = image_of or (lambda e, r, c: noise_image(e, r, c, image_shape))
    index, ep_lines, stats_lines = 0, [], []
    for ep, (state, actions, task) in enumerate(episodes):
        n = len(state)
        done = np.zeros(n, dtype=bool)
        done[-1:] = True
        cols = {"state": _vec(state), "actions": _vec(actions), "done": pa.array(done),
                "is_success": pa.array(np.ones(n, dtype=bool)), "intervene_flag": pa.array(np.ones(n, dtype=bool)),
                "timestamp": pa.array((np.arange(n) / FPS).astype(np.float32)), "frame_index": pa.array(np.arange(n)),
                "episode_index": pa.array(np.full(n, ep)), "index": pa.array(np.arange(index, index + n)),
                "task_index": pa.array(np.full(n, task))}
        if images:
            for cam, col in enumerate(IMAGE_COLUMNS):
                cols[col] = pa.array([{"bytes": png(image_of(ep, r, cam)), "path": f"frame_{r:06d}.png"}
                                      for r in range(n)], type=_IMG)
        table = pa.Table.from_arrays([cols[f.name] for f in schema], schema=schema)
        path = root / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        ep_lines.append(json.dumps({"episode_index": ep, "tasks": [TASKS[task]], "length": n}) + "\n")
        stats_lines.append(json.dumps({"episode_index": ep, "stats": episode_stats(table, feats)}) + "\n")
        index += n
    info = {"codebase_version": "v2.1", "robot_type": "panda", "total_episodes": len(episodes), "total_frames": index,
            "total_tasks": len(TASKS), "total_videos": 0, "total_chunks": 1, "chunks_size": 1000, "fps": FPS,
            "splits": {"train": f"0:{len(episodes)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": feats}
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    (root / "meta" / "tasks.jsonl").write_text(
        "".join(json.dumps({"task_index": i, "task": t}) + "\n" for i, t in enumerate(TASKS)))
    (root / "meta" / "episodes.jsonl").write_text("".join(ep_lines))
    (root / "meta" / "episodes_stats.jsonl").write_text("".join(stats_lines))
    return root
