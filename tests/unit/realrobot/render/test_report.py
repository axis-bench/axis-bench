"""The reported (never gated) metrics: known image distances, a render against itself, and the
DROID-8D view's velocities at the clip bound."""
from __future__ import annotations

import dataclasses
import math
import shutil

import av
import numpy as np
import pandas as pd
import pytest

from axis_qvla.realrobot.render import droid8d, report, rig
from tests.unit.realrobot.render.test_writer import _episode, _frames, _writer


def _rendered(T, attempt, knot_rows=None):
    """A written episode whose sidecar names its knot frames, as the renderer's does."""
    ep = _episode(T, attempt)
    rows = (3 * np.arange(T // 3) + 2).tolist() if knot_rows is None else list(knot_rows)
    return dataclasses.replace(ep, sidecar={**ep.sidecar, "knot_rows": rows})


def test_psnr_and_ssim_on_known_images():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (36, 64, 3), dtype=np.uint8)
    assert report.psnr(a, a) == math.inf and report.ssim(a, a) == pytest.approx(1.0)
    assert report.psnr(a, a + 1) == pytest.approx(10 * math.log10(255.0 ** 2))   # MSE 1
    assert report.ssim(a, 255 - a) < 0.0                                         # anti-correlated


@pytest.fixture
def task(tmp_path):
    w = _writer(tmp_path / "out")
    w.add_episode(_rendered(6, 7001), _frames(6, 1))
    w.add_episode(_rendered(9, 7002, knot_rows=[2, 4, 8]), _frames(9, 2))
    return w.close()


def test_episode_frames_decode_the_episode_span(task):
    frames = report.episode_frames(task, "observation.images.wrist", 1)
    assert frames.shape == (9, rig.HEIGHT, rig.WIDTH, 3) and frames.dtype == np.uint8


def _solid_video(path, levels):
    """An h264 file as the writer encodes one (15 fps, yuv420p), one solid grey frame per level."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=rig.FPS)
        s.width, s.height, s.pix_fmt = rig.WIDTH, rig.HEIGHT, "yuv420p"
        s.options = {"crf": "30", "g": "2"}
        for v in levels:
            frame = av.VideoFrame.from_ndarray(np.full((rig.HEIGHT, rig.WIDTH, 3), v, np.uint8), format="rgb24")
            for packet in s.encode(frame):
                c.mux(packet)
        for packet in s.encode():
            c.mux(packet)


def _move_episode_1(task, start, levels):
    """Episode 1 now lives in videos/<key>/chunk-000/file-001.mp4 from frame `start` on; file-000 still
    holds the writer's copy, so a reader that ignores file_index or the timestamps reads other frames."""
    for key in ("observation.images.third_person", "observation.images.wrist"):
        _solid_video(task / "videos" / key / "chunk-000" / "file-001.mp4", levels)
    p = task / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    meta = pd.read_parquet(p)
    for key in ("observation.images.third_person", "observation.images.wrist"):
        meta.loc[meta["episode_index"] == 1, f"videos/{key}/file_index"] = 1
        meta.loc[meta["episode_index"] == 1, f"videos/{key}/from_timestamp"] = start / rig.FPS
        meta.loc[meta["episode_index"] == 1, f"videos/{key}/to_timestamp"] = (start + 9) / rig.FPS
    meta.to_parquet(p, index=False)


def test_an_episode_is_read_from_its_own_file_at_its_own_timestamps(task):
    _move_episode_1(task, 4, [40] * 4 + [200] * 9 + [90] * 3)
    for key in ("observation.images.third_person", "observation.images.wrist"):
        frames = report.episode_frames(task, key, 1)
        assert frames.shape[0] == 9 and np.abs(frames.reshape(9, -1).mean(1) - 200).max() < 3, key
        assert report.episode_frames(task, key, 0).shape[0] == 6           # episode 0 still in file-000


def test_an_episode_span_the_file_does_not_hold_is_refused(task):
    _move_episode_1(task, 4, [40] * 4 + [200] * 5)                         # 5 of the episode's 9 frames
    with pytest.raises(ValueError, match="5 frames"):
        report.episode_frames(task, "observation.images.wrist", 1)


def test_a_render_compared_with_itself_reports_zero_distance(task):
    rep = report.report_episode(task, 1, task, 1)
    assert rep["frames"] == [9, 9]
    assert rep["joint_parity"] == {"all_rows_max_rad": 0.0, "all_rows_median_rad": 0.0, "knot_rows_max_rad": 0.0,
                                   "knot_rows_median_rad": 0.0}
    assert rep["gripper_max_abs_ctrl"] == 0.0
    for cam in ("third_person", "wrist"):
        assert rep["cameras"][cam]["psnr_db_mean"] == math.inf and rep["cameras"][cam]["ssim_mean"] == pytest.approx(1.0)
        assert rep["cameras"][cam]["palette_abs_mean_diff"] == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("frame, knot", [(4, True), (5, False)])
def test_the_knot_frames_are_the_ones_our_sidecar_names(task, tmp_path, frame, knot):
    ref = tmp_path / "ref" / task.name
    shutil.copytree(task, ref)
    p = ref / "data" / "chunk-000" / "file-000.parquet"
    df = pd.read_parquet(p)
    row = int(np.flatnonzero(df["episode_index"].to_numpy() == 1)[frame])
    df.at[row, "observation.state"] = df.at[row, "observation.state"] + np.float32(0.25)
    df.to_parquet(p, index=False)
    parity = report.report_episode(task, 1, ref, 1)["joint_parity"]
    assert parity["all_rows_max_rad"] == pytest.approx(0.25)
    assert parity["knot_rows_max_rad"] == (pytest.approx(0.25) if knot else 0.0)


def test_episodes_of_different_lengths_are_not_compared(task):
    with pytest.raises(ValueError, match="6 frames"):
        report.report_episode(task, 0, task, 1)


def test_the_droid8d_velocity_report_counts_what_the_view_clipped(task, tmp_path):
    droid8d.build_corpus(task.parent, tmp_path / "d8")
    rep = report.droid8d_velocity(tmp_path / "d8")
    want = droid8d.convert_task(task, tmp_path / "again" / "task_9001")
    assert rep["tasks"]["9001"]["rows"] == want["rows"] == 15
    assert rep["tasks"]["9001"]["velocity_at_bound"] == want["velocity_at_bound"] > 0
    assert rep["velocity_at_bound"] == want["velocity_at_bound"]
    assert 0 < rep["tasks"]["9001"]["rows_at_bound"] <= 15
