"""A row space is the schedule builder's corpus walk, read from metadata, and writes its index files back."""
from __future__ import annotations

import json

import numpy as np
import pytest

from axis_qvla.signals import rowspace as rsp
from axis_qvla.signals.methods import schedule as bis
from axis_qvla.signals.methods.scaling_subsets import filter_ranges, scaling_subsets
from axis_qvla.signals.testing import TOY_TRAINABLE, toy_rowspace
from axis_qvla.signals.views import lerobot as bls
from tests.unit.signals.toy_corpus import RANGES, write_toy_corpus


@pytest.fixture
def toy(tmp_path, monkeypatch):
    monkeypatch.delenv(bis.CORPUS_ROOT_ENV, raising=False)
    t = write_toy_corpus(tmp_path)
    return t.rs, t.roots, t.ranges, t.weights


def test_the_built_row_space_is_the_toy_one(toy):
    rs, *_ = toy
    want = toy_rowspace()
    assert (rs.n_rows, rs.n_trainable, rs.corpus_fingerprint, rs.prompts) == \
           (want.n_rows, want.n_trainable, want.corpus_fingerprint, want.prompts)
    assert (rs.roots_codec, rs.ranges_codec) == ("json-indent1", "json")
    for k, v in want.arrays.items():
        np.testing.assert_array_equal(rs.arrays[k], v, err_msg=k)


def test_index_files_are_written_back_byte_for_byte(toy):
    rs, roots, ranges, _ = toy
    assert rsp.roots_bytes(rs) == roots.read_bytes()
    assert rsp.ranges_bytes(rs) == ranges.read_bytes()
    assert rsp.ranges_mapping(rs) == RANGES and rsp.ranges_keys(rs) == list(RANGES)


def test_trainable_rows_are_the_lerobot_row_plan(toy):
    rs, roots, ranges, _ = toy
    np.testing.assert_array_equal(rsp.trainable_rows(rs), bls.plan_rows(roots, ranges))
    assert rsp.trainable_rows(rs).tolist() == list(TOY_TRAINABLE)


@pytest.mark.parametrize("drop", [False, True])
def test_the_walk_is_the_schedule_builders_walk(toy, drop):
    rs, roots, ranges, weights = toy
    stats = bis.episode_stats_from_weights(weights)
    want = bis.episode_rows(roots, ranges, stats, drop=drop)
    got = rsp.episode_rows(rs, stats, drop=drop)
    assert len(got[0]) == len(want[0])
    for a, b in zip(got[0], want[0]):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(got[5], want[5]):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(got[1], want[1])
    np.testing.assert_array_equal(got[2], want[2])
    assert got[3:5] == want[3:5]
    assert rsp.block_task_ids(rsp.trainable_blocks(rs)).tolist() == [800, 800, 1000]


def test_a_subset_keeps_the_source_key_order_and_writes_the_scaling_format(toy):
    rs, *_ = toy
    keys = scaling_subsets(list(RANGES), [2], 0)[2]
    sub = rsp.scaling_subset(rs, 2, seed=0)
    assert sub.ranges_path == "index/subsets/scaling_2.ranges.json" and sub.n_rows == rs.n_rows
    assert rsp.ranges_bytes(sub) == json.dumps(filter_ranges(RANGES, set(keys)), indent=1).encode()
    one = rsp.subset(rs, ["task_800--0"], name="toy.one", ranges_path="index/one.json",
                     ranges_codec="json-indent1", source="test")
    assert rsp.trainable_rows(one).tolist() == [1, 2] and one.n_trainable == 2


def test_a_flat_row_space_has_every_row_trainable_and_no_episodes():
    rs = rsp.flat_rowspace(name="ft", n_rows=5, source="test")
    assert rsp.trainable_rows(rs).tolist() == [0, 1, 2, 3, 4] and rs.n_trainable == 5
    with pytest.raises(rsp.RowSpaceError):
        rsp.trainable_blocks(rs)


def test_files_it_cannot_reproduce_or_keys_it_cannot_place_are_refused(toy, tmp_path):
    rs, roots, ranges, _ = toy
    bad = tmp_path / "index" / "roots_bad.json"
    bad.write_text(json.dumps(json.loads(roots.read_text()), indent=4))
    with pytest.raises(rsp.RowSpaceError, match="writes these bytes back"):
        rsp.build_rowspace(name="x", corpus=tmp_path / "corpus", roots_file=bad, ranges_file=ranges,
                           roots_path="index/roots_bad.json", ranges_path="index/ranges_toy.json", source="t")
    stray = tmp_path / "index" / "ranges_stray.json"
    stray.write_text(json.dumps({**RANGES, "task_999--0": [[0, 1]]}))
    with pytest.raises(rsp.RowSpaceError, match="name no pretraining episode"):
        rsp.build_rowspace(name="x", corpus=tmp_path / "corpus", roots_file=roots, ranges_file=stray,
                           roots_path="index/roots_toy.json", ranges_path="index/ranges_stray.json", source="t")
    with pytest.raises(rsp.RowSpaceError, match="d8"):
        rsp.roots_v2_bytes(rs)
