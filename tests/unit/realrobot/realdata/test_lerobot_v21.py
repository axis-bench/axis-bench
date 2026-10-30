"""The LeRobot v2.1 layer: meta, parquet with its schema metadata, and lerobot's episode statistics."""
from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from axis_qvla.realrobot.realdata.lerobot_v21 import (Dataset, auto_downsample_height_width, decode_png, encode_png, episode_stats,
                                       image_stats, require_hf_metadata, sample_indices, write_table)
from tests.unit.realrobot.realdata.fr3_world import write_fr3_dataset


def _episodes(lengths=(20, 5)):
    rng = np.random.default_rng(0)
    return [(rng.normal(size=(n, 8)), rng.normal(size=(n, 8)), i % 2) for i, n in enumerate(lengths)]


def test_open_reads_the_three_meta_files(tmp_path):
    ds = Dataset.open(write_fr3_dataset(tmp_path / "ds", _episodes()))
    assert ds.fps == 15.0 and ds.tasks == {0: "pick up the cup", 1: "press the button"}
    assert [e["length"] for e in ds.episodes] == [20, 5]
    assert ds.columns_of("image") == ("image", "extra_view_image") and ds.columns_of("video") == ()
    assert ds.episode_path(1) == tmp_path / "ds" / "data" / "chunk-000" / "episode_000001.parquet"
    assert ds.read_episode(1, ["state"]).num_rows == 5


def test_open_refuses_another_codebase_version(tmp_path):
    root = write_fr3_dataset(tmp_path / "ds", _episodes())
    info = json.loads((root / "meta" / "info.json").read_text())
    (root / "meta" / "info.json").write_text(json.dumps({**info, "codebase_version": "v3.0"}))
    with pytest.raises(ValueError, match="v2.1"):
        Dataset.open(root)


def test_sample_indices_follow_lerobot():
    assert sample_indices(50) == list(range(50))
    idx = sample_indices(1000)
    assert len(idx) == 177 and idx[0] == 0 and idx[-1] == 999
    assert len(sample_indices(200)) == 100


def test_auto_downsample_only_large_frames():
    assert auto_downsample_height_width(np.zeros((3, 224, 224))).shape == (3, 224, 224)
    assert auto_downsample_height_width(np.zeros((3, 720, 1280))).shape == (3, 90, 160)


def test_png_round_trip_is_lossless():
    rgb = np.random.default_rng(1).integers(0, 256, (7, 9, 3), dtype=np.uint8)
    np.testing.assert_array_equal(decode_png(encode_png(rgb)), rgb)


def test_image_stats_are_per_channel_in_unit_range():
    frames = [np.full((4, 4, 3), (0, 51, 255), dtype=np.uint8), np.full((4, 4, 3), (255, 51, 0), dtype=np.uint8)]
    st = image_stats(2, lambda i: frames[i])
    assert st["mean"].shape == (3, 1, 1) and st["count"].tolist() == [2]
    np.testing.assert_allclose(st["mean"].ravel(), [0.5, 0.2, 0.5])
    np.testing.assert_allclose(st["max"].ravel(), [1.0, 0.2, 1.0])


def test_episode_stats_use_the_fr3_export_conversions(tmp_path):
    ds = Dataset.open(write_fr3_dataset(tmp_path / "ds", _episodes()))
    table = ds.read_episode(0)
    st = episode_stats(table, ds.features)
    assert list(st) == list(ds.features)
    state = np.asarray(table.column("state").to_pylist())                    # float32 values as float64
    np.testing.assert_array_equal(st["state"]["mean"], np.mean(state, axis=0))
    ts = table.column("timestamp").to_numpy()                                # float32 stays float32
    assert st["timestamp"]["mean"] == [float(np.mean(ts, keepdims=True)[0])]
    assert st["done"]["min"] == [0.0] and st["done"]["max"] == [1.0]
    assert st["image"]["count"] == [20] and np.asarray(st["image"]["mean"]).shape == (3, 1, 1)
    json.dumps(st)


def test_write_table_keeps_the_huggingface_metadata(tmp_path):
    ds = Dataset.open(write_fr3_dataset(tmp_path / "ds", _episodes()))
    table = ds.read_episode(0)
    write_table(table, tmp_path / "out" / "x.parquet")
    back = pq.read_table(tmp_path / "out" / "x.parquet")
    assert back.schema.metadata[b"huggingface"] == table.schema.metadata[b"huggingface"]
    require_hf_metadata(back.schema, "x")
    with pytest.raises(ValueError, match="huggingface"):
        require_hf_metadata(pa.schema([pa.field("a", pa.int64())]), "bare.parquet")
