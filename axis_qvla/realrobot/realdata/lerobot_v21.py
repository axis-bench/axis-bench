"""Just enough of the LeRobot v2.1 on-disk format to rewrite an FR3 dataset without lerobot.

WITHOUT lerobot: these tools run in the offline venv, which never imports lerobot (venv tiers). The
FR3 datasets are LeRobot v2.1 with PNG frames inside the parquet, and rewriting one needs little:
read meta/, locate an episode's parquet, write a table with its `huggingface` schema metadata intact
(LeRobot refuses image columns without it), and recompute an episode's statistics the way lerobot's
compute_episode_stats does, so a rewritten dataset stays self-consistent.

The statistics are vendored from lerobot v2.1's compute_stats (Apache-2.0, Hugging Face). Columns
become arrays the way the original FR3 export built them -- python values through np.asarray, so
float32 vectors are summed as float64, while the float32 timestamp stays float32 and booleans become
float32 -- because that is what makes the published meta/episodes_stats.jsonl reproduce line for
line (axis_qvla/realrobot/realdata/validate_fr3.py).

Offline tier: numpy, pyarrow; Pillow inside the PNG helpers.
"""
from __future__ import annotations

import dataclasses
import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

IMAGE = "image"
VIDEO = "video"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class RealdataError(ValueError):
    """A named refusal of the FR3 data tools: an input that is not what the tool needs, or an output
    that exists. The realdata CLI prints it as one FAIL line and exits 2."""


@dataclasses.dataclass(frozen=True)
class Dataset:
    """A LeRobot v2.1 dataset on disk: its root and the three meta files that index it."""

    root: Path
    info: dict[str, Any]
    tasks: dict[int, str]
    episodes: tuple[dict[str, Any], ...]     # meta/episodes.jsonl rows, ascending episode_index

    @classmethod
    def open(cls, root: Path) -> Dataset:
        root = Path(root)
        meta = root / "meta"
        missing = [n for n in ("info.json", "tasks.jsonl", "episodes.jsonl") if not (meta / n).is_file()]
        if missing:
            raise RealdataError(f"{root}: no meta/{', meta/'.join(missing)}; give a LeRobot v2.1 dataset directory")
        info = json.loads((meta / "info.json").read_text(encoding="utf-8"))
        if info.get("codebase_version") != "v2.1":
            raise RealdataError(f"{root.name}: codebase_version {info.get('codebase_version')!r}, expected 'v2.1'")
        tasks = {int(r["task_index"]): r["task"] for r in read_jsonl(meta / "tasks.jsonl")}
        episodes = tuple(sorted(read_jsonl(meta / "episodes.jsonl"), key=lambda r: int(r["episode_index"])))
        return cls(root, info, tasks, episodes)

    @property
    def fps(self) -> float:
        return float(self.info["fps"])

    @property
    def features(self) -> dict[str, dict[str, Any]]:
        return self.info["features"]

    def columns_of(self, dtype: str) -> tuple[str, ...]:
        return tuple(k for k, f in self.features.items() if f["dtype"] == dtype)

    def episode_path(self, episode_index: int) -> Path:
        rel = self.info["data_path"].format(episode_chunk=episode_index // int(self.info["chunks_size"]),
                                            episode_index=episode_index)
        return self.root / rel

    def read_episode(self, episode_index: int, columns: list[str] | None = None) -> pa.Table:
        return pq.read_table(self.episode_path(episode_index), columns=columns)


def require_hf_metadata(schema: pa.Schema, where: str) -> None:
    if not schema.metadata or b"huggingface" not in schema.metadata:
        raise ValueError(f"{where}: the parquet schema has no `huggingface` metadata; LeRobot cannot read "
                         f"its image columns without it")


def write_table(table: pa.Table, path: Path) -> None:
    """One parquet file, schema (and its metadata) as given, snappy as LeRobot writes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(path, table.schema) as writer:
        writer.write_table(table)


def _pil():
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("decoding and writing stored PNG frames needs Pillow: pip install 'pillow>=12.2.0' "
                          "(the offline extra declares it)") from exc
    return Image


def decode_png(data: bytes) -> np.ndarray:
    """(H, W, 3) RGB uint8 from encoded image bytes."""
    Image = _pil()
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def encode_png(rgb: np.ndarray) -> bytes:
    Image = _pil()
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buf, format="PNG")
    return buf.getvalue()


# --- vendored from lerobot v2.1, lerobot/common/datasets/compute_stats.py (Apache-2.0) ----------------
def estimate_num_samples(dataset_len: int, min_num_samples: int = 100, max_num_samples: int = 10_000,
                         power: float = 0.75) -> int:
    if dataset_len < min_num_samples:
        min_num_samples = dataset_len
    return max(min_num_samples, min(int(dataset_len**power), max_num_samples))


def sample_indices(data_len: int) -> list[int]:
    return np.round(np.linspace(0, data_len - 1, estimate_num_samples(data_len))).astype(int).tolist()


def auto_downsample_height_width(img: np.ndarray, target_size: int = 150, max_size_threshold: int = 300) -> np.ndarray:
    _, height, width = img.shape
    if max(width, height) < max_size_threshold:
        return img
    factor = int(width / target_size) if width > height else int(height / target_size)
    return img[:, ::factor, ::factor]


def get_feature_stats(array: np.ndarray, axis: int | tuple[int, ...], keepdims: bool) -> dict[str, np.ndarray]:
    return {"min": np.min(array, axis=axis, keepdims=keepdims), "max": np.max(array, axis=axis, keepdims=keepdims),
            "mean": np.mean(array, axis=axis, keepdims=keepdims), "std": np.std(array, axis=axis, keepdims=keepdims),
            "count": np.array([len(array)])}
# -------------------------------------------------------------------------------------------------------


def image_stats(n_frames: int, frame: Callable[[int], np.ndarray]) -> dict[str, np.ndarray]:
    """lerobot's per-episode image statistics over sampled frames: (3, 1, 1) per channel, in [0, 1]."""
    if n_frames < 1:
        raise ValueError("an episode without frames has no image statistics")
    imgs = [auto_downsample_height_width(np.asarray(frame(i)).transpose(2, 0, 1)) for i in sample_indices(n_frames)]
    st = get_feature_stats(np.stack(imgs), axis=(0, 2, 3), keepdims=True)
    return {k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in st.items()}


def _column_array(table: pa.Table, name: str) -> np.ndarray:
    col = table.column(name)
    arr = col.to_numpy() if pa.types.is_floating(col.type) else np.asarray(col.to_pylist())
    return arr.astype(np.float32) if arr.dtype == bool else arr


def episode_stats(table: pa.Table, features: dict[str, dict[str, Any]]) -> dict[str, dict[str, list]]:
    """One meta/episodes_stats.jsonl `stats` entry, every feature in `features` order, JSON-ready."""
    out: dict[str, dict[str, list]] = {}
    for name, feature in features.items():
        if feature["dtype"] == VIDEO:
            raise ValueError(f"{name}: video features keep their frames outside the parquet; not supported")
        if feature["dtype"] == IMAGE:
            col = table.column(name)
            st = image_stats(len(col), lambda i, col=col: decode_png(col[i].as_py()["bytes"]))
        else:
            arr = _column_array(table, name)
            st = get_feature_stats(arr, axis=0, keepdims=arr.ndim == 1)
        out[name] = {k: v.tolist() for k, v in st.items()}
    return out
