"""crop224's geometry and its row-to-frame match, without OpenCV."""
from __future__ import annotations

import numpy as np
import pytest

from axis_qvla.realrobot.realdata.crop224 import RowMatch, center_square, letterbox_square, match_report, match_rows


def _frame(h: int, w: int) -> np.ndarray:
    return np.random.default_rng(h * w).integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_the_centre_square_is_full_height():
    f = _frame(720, 1280)
    np.testing.assert_array_equal(center_square(f), f[:, 280:1000])
    np.testing.assert_array_equal(center_square(_frame(721, 1280)), _frame(721, 1280)[:, 279:1000])
    p = _frame(1280, 720)
    np.testing.assert_array_equal(center_square(p), p[280:1000])
    sq = _frame(224, 224)
    assert np.shares_memory(center_square(sq), sq) and center_square(sq).shape == (224, 224, 3)


def test_the_letterbox_centres_the_frame_on_black():
    f = _frame(720, 1280)
    box = letterbox_square(f)
    assert box.shape == (1280, 1280, 3)
    np.testing.assert_array_equal(box[280:1000], f)
    assert not box[:280].any() and not box[1000:].any()


def test_frames_must_be_rgb_uint8():
    with pytest.raises(ValueError):
        center_square(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        letterbox_square(np.zeros((4, 4, 3), dtype=np.float32))


def _flat(values) -> np.ndarray:
    return np.stack([np.full((4, 4, 3), v, dtype=np.uint8) for v in values])


def test_each_row_finds_its_lagged_frame():
    recorded = _flat(range(0, 200, 10))                      # 20 distinct frames
    stored = recorded[[max(i - 2, 0) for i in range(20)]]    # rows lag the recording by 2
    m = match_rows(stored, recorded)
    assert m.frame.tolist() == [max(i - 2, 0) for i in range(20)]
    assert not m.mse.any()
    assert m.offsets.tolist() == [0, -1] + [-2] * 18


def test_the_window_is_four_frames():
    recorded = _flat(range(0, 200, 10))
    stored = recorded[[5]]                                   # row 0 shows frame 5: out of reach
    m = match_rows(stored, recorded)
    assert m.frame.tolist() == [4] and m.mse.tolist() == [100.0]


def test_ties_go_to_the_earlier_frame():
    m = match_rows(_flat([7] * 10), _flat([7] * 12))
    assert m.frame.tolist() == [max(0, i - 4) for i in range(10)]


def test_a_row_past_the_recording_takes_its_last_frame():
    m = match_rows(_flat([30] * 10), _flat([0, 10, 20]))
    assert m.frame.tolist() == [2] * 10
    with pytest.raises(ValueError, match="empty"):
        match_rows(_flat([0]), np.zeros((0, 4, 4, 3), dtype=np.uint8))


def test_the_report_counts_offsets_and_rows_outside():
    m = RowMatch(np.array([0, 0, 0, 1, 2]), np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    rep = match_report(m, n_recorded=3)
    assert rep["offset_hist"] == {"0": 1, "-1": 1, "-2": 3} and rep["median_offset"] == -2
    assert rep["rows_outside_recording"] == 2 and rep["mse_max"] == 4.0 and rep["recorded_frames"] == 3


def test_a_manifest_holds_relative_paths_only(tmp_path):
    import json

    from axis_qvla.realrobot.realdata.crop224_dataset import load_manifest

    p = tmp_path / "recordings.json"
    p.write_text(json.dumps({"1": {"image": "ep1/ext.npy"}, "0": {"image": "ep0/ext.npy"}}))
    assert load_manifest(p) == {1: {"image": "ep1/ext.npy"}, 0: {"image": "ep0/ext.npy"}}
    p.write_text(json.dumps({"0": {"image": str(tmp_path / "ext.npy")}}))
    with pytest.raises(ValueError, match="relative"):
        load_manifest(p)
