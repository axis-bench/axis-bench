"""Rebuild a letterboxed FR3 LeRobot v2.1 dataset as a centre-crop-224 one from its recordings.

WHY: the crop geometry (axis_qvla.realrobot.realdata.crop224) needs the full-resolution frames, which a
letterboxed dataset does not hold. Only the pixel bytes of the image columns change: rows, every
other column, the image structs' `path` strings, the parquet schema with its `huggingface` metadata,
meta/info.json, tasks.jsonl and episodes.jsonl are carried over; episodes_stats.jsonl keeps every
entry but the image ones, which are recomputed on the new frames. meta/crop224_report.json logs the
match per episode and camera.

Recordings are named per episode and image column in a JSON manifest, with paths RELATIVE to the
manifest's directory (the portability contract):
    {"0": {"image": "ep0/exterior.npy", "extra_view_image": "ep0/wrist"}, ...}
A recording is a .npy (N, H, W, 3) RGB uint8 array, a directory of PNG/JPEG frames sorted by name,
or a video file read through PyAV. ZED .svo/.svo2 files need the proprietary ZED SDK to decode:
export them losslessly to PNG frames first, then point the manifest at the frames.

`episodes` is a smoke test: it writes those episodes and the report, not the meta files.

Rewritten from the dataset producers' FR3 data tools (credited in NOTICE).
Offline tier: numpy, pyarrow; OpenCV (crop224), Pillow (PNG frames) and PyAV (video) where used.
"""
from __future__ import annotations

import json
import shutil
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa

from axis_qvla.realrobot.realdata.crop224 import SIZE, WINDOW, crop224, match_report, match_rows, pad224
from axis_qvla.realrobot.realdata.lerobot_v21 import (IMAGE, Dataset, RealdataError, decode_png, encode_png, image_stats,
                                                  read_jsonl, require_hf_metadata, write_table)

_FRAME_SUFFIXES = {".png", ".jpg", ".jpeg"}


def iter_recording(path: Path) -> Iterator[np.ndarray]:
    """The frames of one recording, (H, W, 3) RGB uint8, in recorded order."""
    path = Path(path)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in _FRAME_SUFFIXES)
        if not files:
            raise FileNotFoundError(f"{path.name}: no PNG/JPEG frames")
        for p in files:
            yield decode_png(p.read_bytes())
    elif path.suffix == ".npy":
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 4 or arr.shape[-1] != 3 or arr.dtype != np.uint8:
            raise ValueError(f"{path.name}: expected (N, H, W, 3) uint8, got {arr.shape} {arr.dtype}")
        for f in arr:
            yield np.asarray(f)
    else:
        try:
            import av
        except ImportError as exc:
            raise ImportError(f"{path.name}: reading a video recording needs PyAV (pip install av)") from exc
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                yield frame.to_ndarray(format="rgb24")


def recording_derivatives(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """pad224 and crop224 of every recorded frame, (m, 224, 224, 3) each."""
    pads, crops = [], []
    for frame in iter_recording(path):
        pads.append(pad224(frame))
        crops.append(crop224(frame))
    if not pads:
        raise ValueError(f"{Path(path).name}: empty recording")
    return np.stack(pads), np.stack(crops)


def load_manifest(path: Path) -> dict[int, dict[str, str]]:
    """{episode_index: {image column: recording path relative to the manifest}}."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[int, dict[str, str]] = {}
    for ep, cols in raw.items():
        for col, rel in cols.items():
            if Path(rel).is_absolute():
                raise RealdataError(f"{Path(path).name}: episode {ep} {col}: store recording paths relative to "
                                    f"the manifest, not {rel!r}")
        out[int(ep)] = dict(cols)
    return out


def rebuild_episode(ds: Dataset, episode_index: int, recordings: Mapping[str, Path], out: Path,
                    window: int = WINDOW) -> tuple[dict, dict[str, dict[str, list]]]:
    """Write one episode with crop224 pixels; return its match report and its new image statistics."""
    table = ds.read_episode(episode_index)
    metadata = table.schema.metadata
    require_hf_metadata(table.schema, ds.episode_path(episode_index).name)
    columns = ds.columns_of(IMAGE)
    if set(recordings) != set(columns):
        raise ValueError(f"episode {episode_index}: recordings for {sorted(recordings)}, image columns {sorted(columns)}")
    report: dict = {"episode_index": episode_index, "rows": table.num_rows, "cols": {}}
    stats: dict[str, dict[str, list]] = {}
    for col in columns:
        pads, crops = recording_derivatives(recordings[col])
        stored = table.column(col).to_pylist()
        match = match_rows(np.stack([decode_png(r["bytes"]) for r in stored]), pads, window)
        frames = crops[match.frame]
        field = table.schema.field(col)
        new = pa.StructArray.from_arrays([pa.array([encode_png(f) for f in frames], pa.binary()),
                                          pa.array([r["path"] for r in stored], pa.string())], names=["bytes", "path"])
        if new.type != field.type:
            raise ValueError(f"episode {episode_index} {col}: rebuilt type {new.type}, stored {field.type}")
        table = table.set_column(table.schema.get_field_index(col), field, new)
        report["cols"][col] = match_report(match, len(pads))
        stats[col] = {k: v.tolist() for k, v in image_stats(len(frames), lambda i, f=frames: f[i]).items()}
    write_table(table.replace_schema_metadata(metadata), out / ds.episode_path(episode_index).relative_to(ds.root))
    return report, stats


def rebuild(ds: Dataset, manifest: Path, out: Path, window: int = WINDOW,
            episodes: Sequence[int] | None = None) -> dict:
    """Rebuild every episode of `ds` (or the smoke subset `episodes`) at `out`, which must not exist.
    The whole manifest is checked first -- every wanted episode, each naming exactly the image columns,
    each recording present -- so a gap is refused before `out` is created, never half-way through."""
    recordings = load_manifest(manifest)
    base, name = Path(manifest).parent, Path(manifest).name
    wanted = [int(e["episode_index"]) for e in ds.episodes] if episodes is None else [int(e) for e in episodes]
    missing = [e for e in wanted if e not in recordings]
    if missing:
        raise RealdataError(f"{name} has no recordings for episodes {missing[:10]}")
    columns = sorted(ds.columns_of(IMAGE))
    gaps = [f"episode {e}: {sorted(recordings[e])}" for e in wanted if sorted(recordings[e]) != columns]
    if gaps:
        raise RealdataError(f"{name}: each episode must name a recording for exactly the image columns "
                            f"{columns}; {'; '.join(gaps[:5])}")
    absent = [rel for e in wanted for rel in recordings[e].values() if not (base / rel).exists()]
    if absent:
        raise RealdataError(f"{name}: {len(absent)} recordings not found beside it, e.g. {absent[:5]}")
    out = Path(out)
    if out.exists():
        raise RealdataError(f"{out} exists; refusing to overwrite")
    (out / "meta").mkdir(parents=True)
    reports, new_stats = [], {}
    for ep in wanted:
        rep, new_stats[ep] = rebuild_episode(ds, ep, {c: base / rel for c, rel in recordings[ep].items()}, out, window)
        rep["recordings"] = recordings[ep]
        reports.append(rep)
    if episodes is None:
        for name in ("info.json", "tasks.jsonl", "episodes.jsonl"):
            shutil.copyfile(ds.root / "meta" / name, out / "meta" / name)
        lines = []
        for rec in read_jsonl(ds.root / "meta" / "episodes_stats.jsonl"):
            rec["stats"].update(new_stats[int(rec["episode_index"])])
            lines.append(json.dumps(rec) + "\n")
        (out / "meta" / "episodes_stats.jsonl").write_text("".join(lines), encoding="utf-8")
    report = {"window": window, "size": SIZE, "episodes": reports}
    (out / "meta" / "crop224_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report
