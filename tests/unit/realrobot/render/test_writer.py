from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from axis_qvla.realrobot.render import profile, rig, writer


def _episode(T, attempt):
    rng = np.random.default_rng(attempt)
    c2w = {k: np.tile(np.eye(4, dtype=np.float32).reshape(-1), (T, 1)) for k in ("third_person", "wrist")}
    return writer.EpisodeData(attempt_id=attempt, state=rng.random((T, 7), dtype=np.float32),
                              action=rng.random((T, 8), dtype=np.float32), ee_state=rng.random((T, 7), dtype=np.float32),
                              camera_to_world=c2w, sidecar={"attempt_id": attempt, "n_knots": T // 3})


def _writer(root, prompt="x"):
    ws = profile.load_profile()
    return writer.TaskWriter(root, 9001, prompt, cameras=ws.cameras, workspace=ws.id)


def _frames(T, seed):
    rng = np.random.default_rng(seed)
    for _ in range(T):
        yield {k: rng.integers(0, 256, (rig.HEIGHT, rig.WIDTH, 3), dtype=np.uint8) for k in ("third_person", "wrist")}


@pytest.fixture
def task(tmp_path):
    w = _writer(tmp_path / "out", "push the red box")
    w.add_episode(_episode(6, 7001), _frames(6, 1))
    w.add_episode(_episode(9, 7002), _frames(9, 2))
    return w.close()


def test_the_data_parquet_has_the_published_schema(task):
    s = pq.read_schema(task / "data" / "chunk-000" / "file-000.parquet")
    for name, f in writer.FEATURES.items():
        if f["dtype"] == "video":
            continue
        assert name in s.names
    assert str(s.field("observation.state").type) == "fixed_size_list<element: float>[7]"
    t = pq.read_table(task / "data" / "chunk-000" / "file-000.parquet").to_pandas()
    assert list(t["index"]) == list(range(15)) and list(t["episode_index"]) == [0] * 6 + [1] * 9
    np.testing.assert_array_equal(t["timestamp"].to_numpy(), (np.r_[np.arange(6), np.arange(9)] / 15).astype(np.float32))
    np.testing.assert_array_equal(np.stack(t["observation.cameras.third_person.intrinsics"])[0],
                                  np.asarray(rig.intrinsics(profile.load_profile().camera), np.float32))


def test_meta_carries_attempt_ids_lengths_and_video_spans(task):
    ep = pd.read_parquet(task / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    assert list(ep["attempt_id"]) == [7001, 7002] and list(ep["length"]) == [6, 9]
    assert list(ep["dataset_from_index"]) == [0, 6]
    assert list(ep["videos/observation.images.wrist/from_timestamp"]) == [0.0, 6 / 15]
    info = json.loads((task / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0" and info["fps"] == 15 and info["total_frames"] == 15
    assert info["features"] == writer.FEATURES
    assert info["render_profile"] == rig.PROFILE_ID and info["workspace_profile"] == "fr3_reference"
    assert json.loads((task / "meta" / "axis_episodes" / "episode_000001.json").read_text())["attempt_id"] == 7002
    assert pd.read_parquet(task / "meta" / "tasks.parquet").index.tolist() == ["push the red box"]


def test_videos_hold_every_frame_as_h264_yuv420p(task):
    for key in ("third_person", "wrist"):
        p = task / "videos" / f"observation.images.{key}" / "chunk-000" / "file-000.mp4"
        assert writer.video_frame_count(p) == 15


def test_a_task_directory_is_never_overwritten(tmp_path, task):
    with pytest.raises(FileExistsError):
        _writer(tmp_path / "out")


def test_the_roots_index_lists_every_task_rendered_into_the_directory(tmp_path):
    ws = profile.load_profile()
    for tid in (9002, 9001):                       # two renders of disjoint selections into one directory
        w = writer.TaskWriter(tmp_path / "out", tid, "x", cameras=ws.cameras, workspace=ws.id)
        w.add_episode(_episode(3, 7001), _frames(3, 1))
        w.close()
        roots = json.loads(writer.roots_index(tmp_path / "out").read_text())
    assert roots == {"version": 2, "base": ".", "roots": {"9001": "task_9001", "9002": "task_9002"}}


def test_a_task_is_written_under_a_partial_name_and_appears_when_closed(tmp_path):
    w = _writer(tmp_path / "out")
    w.add_episode(_episode(3, 7001), _frames(3, 1))
    assert (tmp_path / "out" / "task_9001.partial").is_dir() and not (tmp_path / "out" / "task_9001").exists()
    (tmp_path / "out" / "task_9002.partial").mkdir()                       # another process's task, in progress
    assert json.loads(writer.roots_index(tmp_path / "out").read_text())["roots"] == {}
    assert w.close() == tmp_path / "out" / "task_9001"
    assert not (tmp_path / "out" / "task_9001.partial").exists()
    assert json.loads(writer.roots_index(tmp_path / "out").read_text())["roots"] == {"9001": "task_9001"}
    assert writer.task_dirs(tmp_path / "out") == {9001: tmp_path / "out" / "task_9001"}


def test_an_aborted_task_leaves_nothing(tmp_path):
    w = _writer(tmp_path / "out")
    w.add_episode(_episode(3, 7001), _frames(3, 1))
    w.abort()
    assert list((tmp_path / "out").iterdir()) == []
