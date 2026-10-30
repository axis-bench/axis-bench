"""build_lerobot_signals: the flat LeRobot views index the same frames the schedule walk does."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from axis_qvla.signals.methods import schedule as bis
from axis_qvla.signals.views import lerobot as bls


@pytest.fixture
def toy(tmp_path, monkeypatch):
    """task_800: 3 episodes (6/4/5 rows, ep2 idle-only), task_1000: 1 episode of 8 -> 23 rows."""
    monkeypatch.delenv(bis.CORPUS_ROOT_ENV, raising=False)
    for tid, eps in ((800, [(0, 0, 6), (1, 6, 4), (2, 10, 5)]), (1000, [(0, 0, 8)])):
        d = tmp_path / f"task_{tid}" / "meta" / "episodes" / "chunk-000"
        d.mkdir(parents=True)
        pd.DataFrame([{"episode_index": e, "dataset_from_index": f, "length": n} for e, f, n in eps]
                     ).to_parquet(d / "file-000.parquet")
    roots = tmp_path / "roots.json"
    roots.write_text(json.dumps({"version": 2, "base": ".", "roots": {"1000": "task_1000", "800": "task_800"}}))
    ranges = tmp_path / "ranges.json"
    ranges.write_text(json.dumps({"task_800--0": [[1, 3]], "task_800--1": [[0, 2]], "task_1000--0": [[0, 3], [6, 100]]}))
    w = {"task_800--0": [2.0, 1.0, 0.5, 2.0, 2.0, 2.0], "task_800--1": [1.5, 1.5, 0.5, 0.5],
         "task_800--2": [1.0] * 5, "task_1000--0": [1.0, 1.25, 0.75, 1.0, 1.0, 1.0, 1.5, 2.0]}
    weights = tmp_path / "w.json"
    weights.write_text(json.dumps({"reward_id": "v2", "n_rows": 23, "weights": w}))
    return roots, ranges, weights, w


def test_rows_are_the_schedule_walk_in_order(toy):
    roots, ranges, weights, _ = toy
    rows = bls.plan_rows(roots, ranges)
    blocks, *_ = bis.episode_rows(roots, ranges, bis.episode_stats_from_weights(weights), drop=False)
    np.testing.assert_array_equal(rows, np.concatenate(blocks))
    np.testing.assert_array_equal(rows, [1, 2, 6, 7, 15, 16, 17, 21, 22])


def test_flat_weights_fill_trainable_frames_and_nan_elsewhere(toy):
    roots, ranges, weights, w = toy
    flat = bls.flat_weights(roots, ranges, weights)
    assert flat.dtype == np.float32 and flat.size == 23
    rows = bls.plan_rows(roots, ranges)
    assert np.isnan(np.delete(flat, rows)).all()
    np.testing.assert_array_equal(flat[[1, 2, 6, 7, 15, 21, 22]], np.float32([1.0, 0.5, 1.5, 1.5, 1.0, 1.5, 2.0]))


def test_flat_refuses_a_weights_file_for_another_corpus(toy, tmp_path):
    roots, ranges, weights, w = toy
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"n_rows": 24, "weights": w}))
    with pytest.raises(ValueError, match="different pretraining data"):
        bls.flat_weights(roots, ranges, bad)
    short = dict(w); short["task_800--1"] = [1.0]
    bad.write_text(json.dumps({"weights": short}))
    with pytest.raises(ValueError, match="1 weights for a 4-frame episode"):
        bls.flat_weights(roots, ranges, bad)
    missing = {k: v for k, v in w.items() if k != "task_1000--0"}
    bad.write_text(json.dumps({"weights": missing}))
    with pytest.raises(ValueError, match="no weight"):
        bls.flat_weights(roots, ranges, bad)


def test_cli_writes_deterministic_uncompressed_npz(toy, tmp_path):
    roots, ranges, weights, _ = toy
    for i in (1, 2):
        assert bls.main(["rows", "--roots", str(roots), "--ranges", str(ranges), "--out", str(tmp_path / f"r{i}.npz")]) == 0
        assert bls.main(["flat", "--roots", str(roots), "--ranges", str(ranges), "--weights", str(weights),
                         "--out", str(tmp_path / f"f{i}.npz")]) == 0
        assert bls.main(["ft-rows", "--n-rows", "7", "--out", str(tmp_path / f"a{i}.npz")]) == 0
    for stem in ("r", "f", "a"):
        assert (tmp_path / f"{stem}1.npz").read_bytes() == (tmp_path / f"{stem}2.npz").read_bytes()
    np.testing.assert_array_equal(np.load(tmp_path / "a1.npz")["rows"], np.arange(7))


def test_rows_refuse_overlapping_spans(toy, tmp_path):
    roots, _, _, _ = toy
    r = tmp_path / "overlap.json"
    r.write_text(json.dumps({"task_800--0": [[0, 4], [2, 5]]}))
    with pytest.raises(ValueError, match="strictly increasing"):
        bls.plan_rows(roots, r)
