from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from axis_qvla.realrobot.render import droid8d
from tests.unit.realrobot.render.test_writer import _episode, _frames, _writer
from axis_qvla.realrobot.render import writer


def test_velocity_is_the_forward_difference_times_15_clipped_and_zero_at_episode_ends():
    st = np.zeros((4, 7), np.float32)
    st[1:, 0] = [0.01, 0.2, 0.21]
    ac = np.zeros((4, 8), np.float32)
    ac[:, 7] = [0, 51, 255, 255]
    s8, a8 = droid8d.to_droid8d(st, ac, np.array([0, 0, 0, 0]))
    np.testing.assert_allclose(a8[:, 0], [0.15, 1.0, 0.15, 0.0], atol=1e-6)
    np.testing.assert_allclose(s8[:, 7], [0.0, 0.2, 1.0, 1.0])
    np.testing.assert_array_equal(a8[:, 7], s8[:, 7])


def test_a_task_converts_with_relative_video_links_and_8_wide_shapes(tmp_path):
    w = _writer(tmp_path / "render")
    w.add_episode(_episode(6, 7001), _frames(6, 1))
    w.close()
    roots = droid8d.build_corpus(tmp_path / "render", tmp_path / "d8")
    doc = json.loads(roots.read_text())
    assert doc == {"version": 2, "base": ".", "roots": {"9001": "task_9001"}}
    t = pq.read_table(tmp_path / "d8" / "task_9001" / "data" / "chunk-000" / "file-000.parquet")
    assert str(t.schema.field("action").type) == "fixed_size_list<element: float>[8]"
    assert str(t.schema.field("observation.state").type) == "fixed_size_list<element: float>[8]"
    link = tmp_path / "d8" / "task_9001" / "videos"
    assert link.is_symlink() and not str(link.readlink()).startswith("/")
    info = json.loads((tmp_path / "d8" / "task_9001" / "meta" / "info.json").read_text())
    assert info["features"]["observation.state"]["shape"] == [8]


def test_the_views_statistics_are_the_converted_datas_at_both_levels(tmp_path):
    """A LeRobot consumer builds its normalisers from meta/stats.json and merges from the per-episode
    `stats/` columns. Both must describe the 8-wide view (joint velocities and closedness in [0, 1]),
    never the render's 7-wide state or its joint targets and 0..255 gripper ctrl."""
    w = _writer(tmp_path / "render")
    w.add_episode(_episode(6, 7001), _frames(6, 1))
    w.add_episode(_episode(9, 7002), _frames(9, 2))
    w.close()
    rendered = json.loads((tmp_path / "render" / "task_9001" / "meta" / "stats.json").read_text())
    droid8d.build_corpus(tmp_path / "render", tmp_path / "d8")
    task = tmp_path / "d8" / "task_9001"
    t = pq.read_table(task / "data" / "chunk-000" / "file-000.parquet")
    data = {k: np.asarray(t.column(k).combine_chunks().flatten()).reshape(-1, 8) for k in ("observation.state", "action")}
    ep = t.column("episode_index").to_numpy()
    stats = json.loads((task / "meta" / "stats.json").read_text())
    episodes = pd.read_parquet(task / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    for k, arr in data.items():
        assert stats[k] == writer._stats(arr), k
        assert len(stats[k]["mean"]) == 8 and stats[k]["count"] == [15], k
        for _, row in episodes.iterrows():
            want = writer._stats(arr[ep == row["episode_index"]])
            assert {s: list(row[f"stats/{k}/{s}"]) for s in want} == want, (k, row["episode_index"])
    assert max(stats["action"]["max"]) <= 1.0 and stats["action"]["max"][7] < 0.01   # closedness, not ctrl
    assert stats["observation.ee_state"] == rendered["observation.ee_state"]          # untouched features stay
