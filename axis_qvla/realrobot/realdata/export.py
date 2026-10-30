"""Export the idle-filtered copy of an FR3 LeRobot v2.1 dataset: one kept segment, one episode.

WHY PHYSICAL: the filter's output is a dataset, not a ranges file. Every kept segment of the
recording becomes its own episode (250 recorded -> 392 exported on FR3), so episode boundaries,
lengths and the later tail-10 rule all refer to the exported episodes. Rows are copied unchanged
(images, state, actions, task, flags); only episode_index, frame_index, timestamp, index and done
are renumbered. meta/source_segments.json maps each new episode back to (source episode, start,
end); it names the source by its directory name, never a path, and words its criterion as the
original export did, so the two files compare field for field.

Image-in-parquet datasets only: a video-backed dataset would need its videos cut too, which this
does not do, so it refuses one.

Rewritten from the dataset producers' FR3 data pipeline (credited in NOTICE). Offline tier: numpy, pyarrow.
"""
from __future__ import annotations

import dataclasses
import json
import math
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa

from axis_qvla.realrobot.realdata.idle_filter import FR3_RULE, IdleRule, segments_from_recording
from axis_qvla.realrobot.realdata.lerobot_v21 import (VIDEO, Dataset, RealdataError, episode_stats, require_hf_metadata,
                                                  write_table)
from axis_qvla.realrobot.realdata.state_order import JOINTS, STORED_ACTIONS, STORED_STATE, select


@dataclasses.dataclass(frozen=True)
class Segment:
    """Rows [start, end) of source episode `src_episode`: one exported episode."""

    src_episode: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def _vectors(table: pa.Table, name: str, width: int) -> np.ndarray:
    return np.asarray(table.column(name).to_pylist(), dtype=np.float64).reshape(table.num_rows, width)


def dataset_segments(ds: Dataset, rule: IdleRule = FR3_RULE) -> list[Segment]:
    """The kept segments of every episode in episode order: the exported episodes, in order."""
    out: list[Segment] = []
    for rec in ds.episodes:
        ep = int(rec["episode_index"])
        table = ds.read_episode(ep, ["state", "actions"])
        q = select(_vectors(table, "state", len(STORED_STATE)), STORED_STATE, JOINTS)
        grip = select(_vectors(table, "actions", len(STORED_ACTIONS)), STORED_ACTIONS, ("gripper_cmd",))[:, 0]
        out.extend(Segment(ep, s, e) for s, e in segments_from_recording(q, grip, ds.fps, rule))
    return out


def criterion(rule: IdleRule = FR3_RULE) -> str:
    return ("compute_droid_nonidle_ranges: vel_source=state, grip_guard=%d, grip_delta=%g, "
            "filter_last_n_in_ranges=0" % (rule.grip_guard, rule.grip_delta))


def provenance(source: str, segments: Sequence[Segment], rule: IdleRule = FR3_RULE) -> dict:
    return {"source": source, "criterion": criterion(rule),
            "constants": {"min_idle_len": rule.min_idle_len, "min_non_idle_len": rule.min_segment_len,
                          "idle_threshold": rule.idle_threshold},
            "episodes": {str(i): {"src_episode": s.src_episode, "start": s.start, "end": s.end}
                         for i, s in enumerate(segments)}}


def write_provenance(path: Path, prov: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prov, indent=1), encoding="utf-8")


def _column(field: pa.Field, rows: pa.Table, episode: int, index0: int, fps: float) -> pa.Array | pa.ChunkedArray:
    n = rows.num_rows
    if field.name == "episode_index":
        return pa.array(np.full(n, episode), type=field.type)
    if field.name == "frame_index":
        return pa.array(np.arange(n), type=field.type)
    if field.name == "timestamp":
        return pa.array((np.arange(n) / fps).astype(np.float32), type=field.type)
    if field.name == "index":
        return pa.array(np.arange(index0, index0 + n), type=field.type)
    if field.name == "done":
        done = np.zeros(n, dtype=bool)
        done[-1] = True
        return pa.array(done, type=field.type)
    return rows.column(field.name)


def check_exportable(ds: Dataset, out: Path) -> None:
    """Refuse, before any work, an export that cannot be made: a video-backed dataset (the export cuts
    parquet rows, not videos) or an output that exists."""
    if ds.columns_of(VIDEO):
        raise RealdataError(f"{ds.root.name}: video features {list(ds.columns_of(VIDEO))}; the export cuts parquet "
                            f"rows, not videos -- use an image-in-parquet copy of the dataset")
    if Path(out).exists():
        raise RealdataError(f"{out} exists; refusing to overwrite")


def export(ds: Dataset, segments: Sequence[Segment], out: Path, rule: IdleRule = FR3_RULE) -> dict[str, int]:
    """Write `segments` of `ds` as a new LeRobot v2.1 dataset at `out`, which must not exist."""
    check_exportable(ds, out)
    out = Path(out)
    (out / "meta").mkdir(parents=True)
    chunks = int(ds.info["chunks_size"])
    episodes, stats, index, cached = [], [], 0, None
    for new_ep, seg in enumerate(segments):
        if cached is None or cached[0] != seg.src_episode:
            cached = (seg.src_episode, ds.read_episode(seg.src_episode))
            require_hf_metadata(cached[1].schema, ds.episode_path(seg.src_episode).name)
        src = cached[1]
        rows = src.slice(seg.start, seg.length)
        table = pa.Table.from_arrays([_column(f, rows, new_ep, index, ds.fps) for f in src.schema], schema=src.schema)
        write_table(table, out / ds.info["data_path"].format(episode_chunk=new_ep // chunks, episode_index=new_ep))
        task = ds.tasks[int(src.column("task_index")[0].as_py())]
        episodes.append(json.dumps({"episode_index": new_ep, "tasks": [task], "length": seg.length},
                                   ensure_ascii=False) + "\n")
        stats.append(json.dumps({"episode_index": new_ep, "stats": episode_stats(table, ds.features)}) + "\n")
        index += seg.length
    meta = out / "meta"
    shutil.copyfile(ds.root / "meta" / "tasks.jsonl", meta / "tasks.jsonl")
    (meta / "episodes.jsonl").write_text("".join(episodes), encoding="utf-8")
    (meta / "episodes_stats.jsonl").write_text("".join(stats), encoding="utf-8")
    write_provenance(meta / "source_segments.json", provenance(ds.root.name, segments, rule))
    info = dict(ds.info)
    info.update(total_episodes=len(segments), total_frames=index,
                total_chunks=max(1, math.ceil(len(segments) / chunks)), splits={"train": f"0:{len(segments)}"})
    (meta / "info.json").write_text(json.dumps(info, indent=4), encoding="utf-8")
    return {"source_episodes": len(ds.episodes), "source_frames": sum(int(e["length"]) for e in ds.episodes),
            "episodes": len(segments), "frames": index}
