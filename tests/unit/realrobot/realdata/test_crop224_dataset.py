"""The crop224 rebuild's plumbing, without OpenCV: the resize is swapped for a nearest-neighbour one.

The OpenCV byte pin lives in test_crop224_cv2.py, which needs OpenCV; this file reaches crop224_dataset
in the public suite."""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from axis_qvla.realrobot.realdata import crop224 as crop_mod
from axis_qvla.realrobot.realdata.crop224 import SIZE, crop224, pad224
from axis_qvla.realrobot.realdata.crop224_dataset import iter_recording, rebuild
from axis_qvla.realrobot.realdata.lerobot_v21 import Dataset, decode_png, read_jsonl
from tests.unit.realrobot.realdata.fr3_world import write_fr3_dataset

M = 9


def _nearest224(img):
    h, w = img.shape[:2]
    return np.ascontiguousarray(img[(np.arange(SIZE) * h) // SIZE][:, (np.arange(SIZE) * w) // SIZE])


@pytest.fixture(autouse=True)
def no_opencv(monkeypatch):
    monkeypatch.setattr(crop_mod, "resize224", _nearest224)


def _rec(ep, cam):
    return np.random.default_rng([ep, cam, 7]).integers(0, 256, (M, 72, 128, 3), dtype=np.uint8)


def _world(tmp_path):
    eps = [(np.zeros((M, 8)), np.zeros((M, 8)), 0), (np.zeros((M, 8)), np.zeros((M, 8)), 1)]
    ds = Dataset.open(write_fr3_dataset(tmp_path / "lb", eps, image_shape=(SIZE, SIZE, 3),
                                        image_of=lambda e, r, c: pad224(_rec(e, c)[max(r - 2, 0)])))
    rec = tmp_path / "recordings"
    manifest = {}
    for ep in range(2):
        manifest[str(ep)] = {}
        for cam, col in enumerate(("image", "extra_view_image")):
            (rec / f"ep{ep}").mkdir(parents=True, exist_ok=True)
            np.save(rec / f"ep{ep}" / f"{col}.npy", _rec(ep, cam))
            manifest[str(ep)][col] = f"ep{ep}/{col}.npy"
    (rec / "manifest.json").write_text(json.dumps(manifest))
    return ds, rec / "manifest.json"


def test_rebuild_swaps_only_the_pixels_and_matches_the_two_frame_lag(tmp_path):
    ds, manifest = _world(tmp_path)
    report = rebuild(ds, manifest, tmp_path / "cc")
    new = pq.read_table(tmp_path / "cc" / "data" / "chunk-000" / "episode_000000.parquet")
    for row, r in enumerate(new.column("image").to_pylist()):
        np.testing.assert_array_equal(decode_png(r["bytes"]), crop224(_rec(0, 0)[max(row - 2, 0)]))
    assert report["episodes"][0]["cols"]["image"]["offset_hist"] == {"0": 1, "-1": 1, "-2": M - 2}


def test_rebuild_carries_meta_and_a_smoke_subset_writes_none(tmp_path):
    ds, manifest = _world(tmp_path)
    rebuild(ds, manifest, tmp_path / "cc")
    for name in ("info.json", "tasks.jsonl", "episodes.jsonl"):
        assert (tmp_path / "cc" / "meta" / name).read_bytes() == (ds.root / "meta" / name).read_bytes()
    assert len(read_jsonl(tmp_path / "cc" / "meta" / "episodes_stats.jsonl")) == 2
    rebuild(ds, manifest, tmp_path / "smoke", episodes=[1])
    assert not (tmp_path / "smoke" / "meta" / "info.json").exists()


def test_an_npy_recording_reads_back_in_order(tmp_path):
    np.save(tmp_path / "r.npy", _rec(3, 0))
    assert len(list(iter_recording(tmp_path / "r.npy"))) == M


def _crop224_cli(ds, manifest, out):
    from axis_qvla.realrobot.realdata.cli import main
    return main(["crop224", "--dataset", str(ds.root), "--recordings", str(manifest), "--out", str(out)])


def test_the_crop224_command_rebuilds_through_the_cli(tmp_path, capsys):
    ds, manifest = _world(tmp_path)
    assert _crop224_cli(ds, manifest, tmp_path / "cc") == 0
    assert "2 episodes rebuilt" in capsys.readouterr().out and (tmp_path / "cc" / "meta" / "info.json").exists()


def test_a_manifest_without_some_episodes_is_refused_and_nothing_is_written(tmp_path, capsys):
    ds, manifest = _world(tmp_path)
    doc = json.loads(manifest.read_text())
    del doc["1"]
    manifest.write_text(json.dumps(doc))
    assert _crop224_cli(ds, manifest, tmp_path / "cc") == 2
    err = capsys.readouterr().err
    assert err.startswith("FAIL  ") and "no recordings for episodes [1]" in err and not (tmp_path / "cc").exists()


def test_a_manifest_episode_without_a_camera_is_refused_before_any_episode_is_written(tmp_path, capsys):
    ds, manifest = _world(tmp_path)
    doc = json.loads(manifest.read_text())
    del doc["1"]["extra_view_image"]
    manifest.write_text(json.dumps(doc))
    assert _crop224_cli(ds, manifest, tmp_path / "cc") == 2
    err = capsys.readouterr().err
    assert "episode 1" in err and "extra_view_image" in err and not (tmp_path / "cc").exists()


def test_a_manifest_naming_a_missing_recording_is_refused_and_nothing_is_written(tmp_path, capsys):
    ds, manifest = _world(tmp_path)
    (manifest.parent / "ep1" / "image.npy").unlink()
    assert _crop224_cli(ds, manifest, tmp_path / "cc") == 2
    assert "ep1/image.npy" in capsys.readouterr().err and not (tmp_path / "cc").exists()
