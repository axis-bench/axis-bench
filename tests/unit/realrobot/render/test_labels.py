from __future__ import annotations

import json

import numpy as np
import pytest

from axis_qvla.realrobot.realign.curves import PhaseCurve, read_curve_store
from axis_qvla.realrobot.render import labels, writer
from tests.unit.realrobot.render.test_writer import _episode, _frames, _writer


def test_attempt_stores_round_trip_and_neutral_curves_have_no_t0(tmp_path):
    curves = {(9001, 7001): PhaseCurve(np.linspace(0, 1, 6, dtype=np.float32), np.ones(6, bool), 3),
              (9001, 7002): PhaseCurve(np.zeros(9, np.float32), np.zeros(9, bool), None)}
    labels.write_attempt_store(tmp_path / "a", curves)
    back = labels.read_attempt_store(tmp_path / "a")
    assert back[(9001, 7001)].t0 == 3 and back[(9001, 7002)].t0 is None
    np.testing.assert_array_equal(back[(9001, 7001)].curve, curves[(9001, 7001)].curve)


def test_the_render_store_is_keyed_by_training_episode_through_attempt_ids(tmp_path):
    w = _writer(tmp_path / "r")
    w.add_episode(_episode(9, 7002), _frames(9, 2))
    w.add_episode(_episode(6, 7001), _frames(6, 1))
    w.close()
    writer.roots_index(tmp_path / "r")
    curves = {(9001, 7001): PhaseCurve(np.full(6, 0.5, np.float32), np.ones(6, bool), 2),
              (9001, 7002): PhaseCurve(np.full(9, 0.25, np.float32), np.ones(9, bool), 4)}
    assert labels.render_curve_store(tmp_path / "r", curves, tmp_path / "store") == 2
    store = read_curve_store(tmp_path / "store")
    assert store[(9001, 0)].t0 == 4 and store[(9001, 1)].t0 == 2


def test_a_curve_of_the_wrong_length_or_a_missing_attempt_is_refused(tmp_path):
    w = _writer(tmp_path / "r")
    w.add_episode(_episode(6, 7001), _frames(6, 1))
    w.close()
    writer.roots_index(tmp_path / "r")
    with pytest.raises(ValueError, match="6 frames"):
        labels.render_curve_store(tmp_path / "r", {(9001, 7001): PhaseCurve(np.zeros(5, np.float32), np.zeros(5, bool), None)},
                                  tmp_path / "short")
    with pytest.raises(KeyError, match="7001"):
        labels.render_curve_store(tmp_path / "r", {}, tmp_path / "missing")


def test_export_keyed_curves_convert_to_attempt_keys(tmp_path):
    export_dir = tmp_path / "export_dir"
    export_dir.mkdir()
    np.savez(export_dir / "task_9001.npz", phase_ep_0=np.stack([np.linspace(0, 1, 6), np.ones(6)]).astype(np.float32),
             t0_ep_0=np.int64(2))
    amap = tmp_path / "map.json"
    amap.write_text(json.dumps({"episodes": {"task_9001--3": {"attempt_id": 7001, "export_episode_index": 0,
                                                              "train_frame_count": 6, "export_frame_count": 6}}}))
    assert labels.attempt_store_from_export(export_dir, amap, tmp_path / "a") == 1
    assert labels.read_attempt_store(tmp_path / "a")[(9001, 7001)].t0 == 2


def test_export_curves_whose_map_gives_one_attempt_two_episodes_is_refused(tmp_path):
    export_dir = tmp_path / "export_dir"
    export_dir.mkdir()
    row = np.stack([np.linspace(0, 1, 6), np.ones(6)]).astype(np.float32)
    np.savez(export_dir / "task_9001.npz", phase_ep_0=row, phase_ep_1=row)
    amap = tmp_path / "map.json"
    amap.write_text(json.dumps({"episodes": {
        f"task_9001--{ep}": {"attempt_id": 7001, "export_episode_index": ep, "train_frame_count": 6,
                             "export_frame_count": 6} for ep in (0, 1)}}))
    with pytest.raises(ValueError, match="7001"):
        labels.attempt_store_from_export(export_dir, amap, tmp_path / "a")
