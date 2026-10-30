from __future__ import annotations

import json

import numpy as np
import pytest

from axis_qvla.realrobot.realign.curves import (
    CurveError, PhaseCurve, load_export_curves, read_curve_store, write_curve_store,
)


def _export_curves(tmp_path, episodes, *, extra_export=None):
    """Export-keyed curves plus their alignment map.

    `episodes`: list of (task_id, training_ep, export_ep, T, t0 or None).
    """
    rng = np.random.default_rng(0)
    curves_dir = tmp_path / "curves"
    curves_dir.mkdir()
    by_task: dict[int, dict] = {}
    amap = {}
    for tid, ep, x, T, t0 in episodes:
        valid = np.ones(T, np.float32) if t0 is not None else np.zeros(T, np.float32)
        arrs = by_task.setdefault(tid, {"_provenance": np.array("{}")})
        arrs[f"phase_ep_{x}"] = np.stack([rng.random(T).astype(np.float32), valid])
        arrs[f"v2_ep_{x}"] = np.stack([rng.random(T).astype(np.float32), valid])
        if t0 is not None:
            arrs[f"t0_ep_{x}"] = np.int64(t0)
        amap[f"task_{tid}--{ep}"] = {"attempt_id": 1000 * tid + ep, "export_episode_index": x,
                                     "train_frame_count": T, "export_frame_count": T}
    for tid, x, T in extra_export or []:
        by_task[tid][f"phase_ep_{x}"] = np.zeros((2, T), np.float32)
    for tid, arrs in by_task.items():
        np.savez(curves_dir / f"task_{tid}.npz", **arrs)
    map_path = tmp_path / "map.json"
    map_path.write_text(json.dumps({"schema": "axis_training_episode_alignment_v1", "episodes": amap}))
    return curves_dir, map_path


def test_export_curves_are_rekeyed_from_export_to_training_episodes(tmp_path):
    curves_dir, amap = _export_curves(tmp_path, [(800, 0, 2, 20, 5), (800, 1, 0, 30, 9), (801, 0, 0, 25, 3)])
    raw = np.load(curves_dir / "task_800.npz")
    got = load_export_curves(curves_dir, amap)
    assert set(got) == {(800, 0), (800, 1), (801, 0)}
    np.testing.assert_array_equal(got[(800, 0)].curve, raw["phase_ep_2"][0])
    np.testing.assert_array_equal(got[(800, 1)].curve, raw["phase_ep_0"][0])
    assert got[(800, 0)].t0 == 5 and got[(800, 1)].t0 == 9
    assert got[(800, 0)].curve.dtype == np.float32 and got[(800, 0)].valid.dtype == bool


def test_an_episode_without_t0_is_neutral(tmp_path):
    curves_dir, amap = _export_curves(tmp_path, [(800, 0, 0, 20, None)])
    c = load_export_curves(curves_dir, amap)[(800, 0)]
    assert c.t0 is None and not c.valid.any()


def test_a_length_that_disagrees_with_the_map_is_refused(tmp_path):
    curves_dir, amap = _export_curves(tmp_path, [(800, 0, 0, 20, 5)])
    doc = json.loads(amap.read_text())
    doc["episodes"]["task_800--0"]["train_frame_count"] = 21
    amap.write_text(json.dumps(doc))
    with pytest.raises(CurveError, match="task 800 export episode 0"):
        load_export_curves(curves_dir, amap)


def test_a_map_episode_without_a_curve_is_refused(tmp_path):
    curves_dir, amap = _export_curves(tmp_path, [(800, 0, 0, 20, 5)])
    doc = json.loads(amap.read_text())
    doc["episodes"]["task_800--1"] = {"attempt_id": 7, "export_episode_index": 5,
                                      "train_frame_count": 20, "export_frame_count": 20}
    amap.write_text(json.dumps(doc))
    with pytest.raises(CurveError, match="no curve"):
        load_export_curves(curves_dir, amap)


def test_a_curve_the_map_does_not_know_is_refused(tmp_path):
    curves_dir, amap = _export_curves(tmp_path, [(800, 0, 0, 20, 5)], extra_export=[(800, 9, 20)])
    with pytest.raises(CurveError, match="not in the alignment map"):
        load_export_curves(curves_dir, amap)


def test_t0_outside_the_episode_is_refused():
    with pytest.raises(CurveError, match="t0"):
        PhaseCurve(np.zeros(5, np.float32), np.ones(5, bool), 6)


def test_store_round_trip_keeps_every_field(tmp_path):
    curves = {(800, 0): PhaseCurve(np.linspace(0, 1, 12, dtype=np.float32), np.ones(12, bool), 4),
              (800, 3): PhaseCurve(np.zeros(7, np.float32), np.zeros(7, bool), None),
              (912, 1): PhaseCurve(np.full(9, 0.5, np.float32), np.r_[np.ones(5), np.zeros(4)].astype(bool), 0)}
    write_curve_store(tmp_path / "store", curves)
    back = read_curve_store(tmp_path / "store")
    assert set(back) == set(curves)
    for k, c in curves.items():
        np.testing.assert_array_equal(back[k].curve, c.curve)
        np.testing.assert_array_equal(back[k].valid, c.valid)
        assert back[k].t0 == c.t0


def test_a_store_without_validity_is_refused(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    np.savez(store / "task_800_phase_curves.npz", ep_0=np.zeros(5, np.float32), t0_0=np.int64(2))
    with pytest.raises(CurveError, match="valid_0"):
        read_curve_store(store)


def _dumped_corpus(tmp_path):
    """A corpus with reward_valid in its data parquet, and a dump of two of its three episodes."""
    import pandas as pd
    base = tmp_path / "corpus"
    (base / "task_5" / "data" / "chunk-000").mkdir(parents=True)
    (base / "task_5" / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    lens = [6, 5, 4]
    rows = [(e, f, f < L - 1) for e, L in enumerate(lens) for f in range(L)]
    pd.DataFrame(rows, columns=["episode_index", "frame_index", "reward_valid"]).to_parquet(
        base / "task_5" / "data" / "chunk-000" / "file-000.parquet")
    pd.DataFrame({"episode_index": [0, 1, 2], "dataset_from_index": [0, 6, 11], "length": lens,
                  "attempt_id": [50, 51, 52]}).to_parquet(base / "task_5" / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    roots = tmp_path / "roots.json"
    roots.write_text(json.dumps({"version": 2, "base": "corpus", "roots": {"5": "task_5"}}))
    store = tmp_path / "dump"
    store.mkdir()
    np.savez_compressed(store / "task_5_phase_curves.npz", ep_0=np.linspace(0, 1, 6).astype(np.float32),
                        t0_0=np.int64(2), ep_2=np.linspace(0, 1, 4).astype(np.float32), t0_2=np.int64(1))
    return store, roots


def test_a_dumped_store_takes_its_masks_from_the_corpus_and_restores_neutral_episodes(tmp_path):
    from axis_qvla.realrobot.realign.curves import read_dumped_curve_store
    store, roots = _dumped_corpus(tmp_path)
    got = read_dumped_curve_store(store, roots)
    assert set(got) == {(5, 0), (5, 1), (5, 2)}
    np.testing.assert_array_equal(got[(5, 0)].valid, [True] * 5 + [False])
    assert got[(5, 1)].t0 is None and not got[(5, 1)].valid.any() and len(got[(5, 1)].curve) == 5


def test_a_store_with_masks_is_not_a_dump(tmp_path):
    from axis_qvla.realrobot.realign.curves import read_dumped_curve_store
    store, roots = _dumped_corpus(tmp_path)
    np.savez_compressed(store / "task_5_phase_curves.npz", ep_0=np.zeros(6, np.float32), valid_0=np.ones(6, bool))
    with pytest.raises(CurveError, match="not a dump"):
        read_dumped_curve_store(store, roots)


def test_a_missing_or_empty_curve_directory_is_refused_naming_it(tmp_path):
    """A mistyped directory must not read as a store with no curves, which then fails for another
    reason (no contact onset to balance, map episodes without curves)."""
    from axis_qvla.realrobot.realign.curves import CurveError, load_export_curves, read_curve_store
    with pytest.raises(CurveError, match=f"{tmp_path / 'nope'}: no such directory"):
        read_curve_store(tmp_path / "nope")
    (tmp_path / "empty").mkdir()
    with pytest.raises(CurveError, match="holds no task_\\*_phase_curves.npz"):
        read_curve_store(tmp_path / "empty")
    (tmp_path / "map.json").write_text(json.dumps({"episodes": {}}))
    with pytest.raises(CurveError, match="holds no task_\\*.npz"):
        load_export_curves(tmp_path / "empty", tmp_path / "map.json")


def test_an_attempt_store_that_is_absent_or_unreadable_is_not_an_empty_store(tmp_path):
    """An absent or unreadable directory must not read as a store without curves; an empty one is one."""
    from axis_qvla.realrobot.realign.curves import read_attempt_store
    with pytest.raises(CurveError, match=f"{tmp_path / 'nope'}: no such directory"):
        read_attempt_store(tmp_path / "nope")
    (tmp_path / "empty").mkdir()
    assert read_attempt_store(tmp_path / "empty") == {}
    (tmp_path / "locked").mkdir()
    (tmp_path / "locked").chmod(0)
    try:
        with pytest.raises(PermissionError):
            read_attempt_store(tmp_path / "locked")
    finally:
        (tmp_path / "locked").chmod(0o755)    # tmp_path cleanup needs to list and delete it
