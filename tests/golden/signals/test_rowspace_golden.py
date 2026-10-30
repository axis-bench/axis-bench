"""The paper's row spaces, built from metadata, are the locked index files and row plans.

Needs --pretraining-data and --episodes (metadata only is read).
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
import pytest

from axis_qvla.signals import rowspace as rsp
from axis_qvla.signals.methods import schedule as bis

REPO = pathlib.Path(__file__).resolve().parents[3]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.mark.slow
def test_the_paper_row_space_is_the_locked_one(paper_rowspaces, signals_pack, lock):
    full = paper_rowspaces["libero5k_v2_d8"]
    rows = {r["id"]: r for r in lock["files"]}
    space = lock["rowspaces"]["libero5k_v2_d8"]
    assert (full.n_rows, full.n_trainable, len(full.arrays["episode_task"])) == (
        space["n_rows"], space["n_trainable"], space["n_episodes"])
    assert full.corpus_fingerprint == space["corpus_fingerprint"]
    assert _sha(rsp.roots_bytes(full)) == rows["index.roots_d8"]["sha256"]
    assert _sha(rsp.ranges_bytes(full)) == rows["index.ranges"]["sha256"]
    spec = json.loads((REPO / "conf" / "signals" / "generators.json").read_text())
    assert _sha(rsp.roots_v2_bytes(full)) == spec["inputs"]["roots_v2"]["sha256"]
    with np.load(signals_pack / rows["approach_aware.episode.tags"]["path"]) as z:
        assert list(full.prompts) == z["prompts"].tolist() and len(full.prompts) == 566


@pytest.mark.slow
@pytest.mark.parametrize("name,rows_id,ranges_id", [
    ("libero5k_v2_d8", "lerobot.rows", "index.ranges"),
    ("libero5k_v2_d8.sub1000", "lerobot.rows_s1000", "index.scaling_1000"),
    ("libero5k_v2_d8.sub2500", "lerobot.rows_s2500", "index.scaling_2500"),
])
def test_trainable_rows_and_ranges_files_are_the_locked_ones(paper_rowspaces, signals_pack, lock, name, rows_id, ranges_id):
    rs = paper_rowspaces[name]
    rows = {r["id"]: r for r in lock["files"]}
    with np.load(signals_pack / rows[rows_id]["path"]) as z:
        np.testing.assert_array_equal(rsp.trainable_rows(rs), z["rows"])
    assert rs.ranges_path == rows[ranges_id]["path"]
    assert _sha(rsp.ranges_bytes(rs)) == rows[ranges_id]["sha256"]


@pytest.mark.slow
def test_the_row_space_walk_is_the_schedule_builders_corpus_walk(paper_rowspaces, signals_pack, lock, corpus,
                                                                  tmp_path, monkeypatch):
    full = paper_rowspaces["libero5k_v2_d8"]
    rows = {r["id"]: r for r in lock["files"]}
    roots = tmp_path / "roots_libero_5k_v2.json"
    roots.write_bytes(rsp.roots_v2_bytes(full))
    monkeypatch.setenv(bis.CORPUS_ROOT_ENV, str(corpus))
    stats = bis.episode_stats_from_weights(signals_pack / rows["approach_aware.chunk.awr_weights"]["path"])
    want = bis.episode_rows(roots, signals_pack / rows["index.ranges"]["path"], stats, drop=False)
    got = rsp.episode_rows(full, stats, drop=False)
    assert got[3:5] == want[3:5] and len(got[0]) == len(want[0])
    assert all(np.array_equal(a, b) for a, b in zip(got[0], want[0]))
    assert all(np.array_equal(a, b) for a, b in zip(got[5], want[5]))
    np.testing.assert_array_equal(got[1], want[1])
    np.testing.assert_array_equal(got[2], want[2])
