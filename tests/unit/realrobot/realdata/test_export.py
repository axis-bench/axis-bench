"""The idle-filtered export: one kept segment per episode, rows unchanged, five columns renumbered."""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from axis_qvla.realrobot.realdata.export import Segment, criterion, dataset_segments, export, provenance
from axis_qvla.realrobot.realdata.idle_filter import FR3_RULE
from axis_qvla.realrobot.realdata.lerobot_v21 import Dataset, RealdataError
from tests.unit.realrobot.realdata.fr3_world import write_fr3_dataset
from tests.unit.realrobot.realdata.test_idle_filter import _recording


def _episode(q: np.ndarray, grip_cmd: np.ndarray, grip_state: np.ndarray, task: int):
    n = len(q)
    state = np.concatenate([grip_state[:, None], q], axis=1)
    actions = np.concatenate([np.full((n, 7), 0.1), grip_cmd[:, None]], axis=1)
    return state, actions, task


def _world(tmp_path):
    step = np.zeros(60)
    step[25:] = 1.0
    eps = [_episode(_recording(60, hold=(21, 30)), np.zeros(60), step, 0),     # measured gripper moves: ignored
           _episode(np.outer(np.arange(40) * 0.01, np.ones(7)), np.zeros(40), np.zeros(40), 1),  # all idle
           _episode(_recording(60, hold=(21, 30)), step, np.zeros(60), 1)]     # commanded gripper protects
    return Dataset.open(write_fr3_dataset(tmp_path / "rec_250", eps))


def test_segments_read_the_gripper_command_not_the_measurement(tmp_path):
    assert dataset_segments(_world(tmp_path)) == [Segment(0, 0, 21), Segment(0, 29, 60), Segment(2, 0, 60)]


def test_provenance_names_the_source_by_directory_name_only(tmp_path):
    prov = provenance("rec_250", [Segment(0, 0, 21), Segment(2, 0, 60)])
    assert prov == {"source": "rec_250", "criterion": criterion(FR3_RULE),
                    "constants": {"min_idle_len": 7, "min_non_idle_len": 16, "idle_threshold": 0.001},
                    "episodes": {"0": {"src_episode": 0, "start": 0, "end": 21},
                                 "1": {"src_episode": 2, "start": 0, "end": 60}}}
    assert criterion(FR3_RULE) == ("compute_droid_nonidle_ranges: vel_source=state, grip_guard=5, grip_delta=0.02, "
                             "filter_last_n_in_ranges=0")


def test_export_copies_rows_and_renumbers_five_columns(tmp_path):
    ds = _world(tmp_path)
    segs = dataset_segments(ds)
    summary = export(ds, segs, tmp_path / "out")
    assert summary == {"source_episodes": 3, "source_frames": 160, "episodes": 3, "frames": 112}
    src = ds.read_episode(0)
    new = pq.read_table(tmp_path / "out" / "data" / "chunk-000" / "episode_000001.parquet")   # src ep 0, rows 29..59
    assert new.schema.metadata[b"huggingface"] == src.schema.metadata[b"huggingface"]
    for col in ("state", "actions", "image", "extra_view_image", "task_index", "is_success", "intervene_flag"):
        assert new.column(col).to_pylist() == src.column(col).to_pylist()[29:60], col
    assert new.column("episode_index").to_pylist() == [1] * 31
    assert new.column("frame_index").to_pylist() == list(range(31))
    assert new.column("index").to_pylist() == list(range(21, 52))
    np.testing.assert_array_equal(new.column("timestamp").to_numpy(), (np.arange(31) / 15).astype(np.float32))
    assert new.column("done").to_pylist() == [False] * 30 + [True]


def test_export_writes_consistent_meta(tmp_path):
    ds = _world(tmp_path)
    export(ds, dataset_segments(ds), tmp_path / "out")
    meta = tmp_path / "out" / "meta"
    info = json.loads((meta / "info.json").read_text())
    assert (info["total_episodes"], info["total_frames"], info["splits"]) == (3, 112, {"train": "0:3"})
    assert (meta / "tasks.jsonl").read_bytes() == (ds.root / "meta" / "tasks.jsonl").read_bytes()
    assert [json.loads(x) for x in (meta / "episodes.jsonl").read_text().splitlines()] == [
        {"episode_index": 0, "tasks": ["pick up the cup"], "length": 21},
        {"episode_index": 1, "tasks": ["pick up the cup"], "length": 31},
        {"episode_index": 2, "tasks": ["press the button"], "length": 60}]
    prov = json.loads((meta / "source_segments.json").read_text())
    assert prov["source"] == "rec_250" and prov["episodes"]["1"] == {"src_episode": 0, "start": 29, "end": 60}
    stats = [json.loads(x) for x in (meta / "episodes_stats.jsonl").read_text().splitlines()]
    assert [s["episode_index"] for s in stats] == [0, 1, 2]
    src_state = np.asarray(ds.read_episode(0).column("state").to_pylist())[29:60]
    np.testing.assert_array_equal(stats[1]["stats"]["state"]["mean"], np.mean(src_state, axis=0))
    assert stats[1]["stats"]["image"]["count"] == [31]


def test_export_refuses_an_existing_output_and_a_video_dataset(tmp_path):
    ds = _world(tmp_path)
    (tmp_path / "out").mkdir()
    with pytest.raises(RealdataError, match="exists"):
        export(ds, [], tmp_path / "out")
    video = Dataset(ds.root, {**ds.info, "features": {**ds.features, "cam": {"dtype": "video"}}}, ds.tasks, ds.episodes)
    with pytest.raises(ValueError, match="video"):
        export(video, [], tmp_path / "out2")
