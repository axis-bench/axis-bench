"""Pure-conditioning tags are the pinned builder's tags with the dropout stream removed, nothing else."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from axis_qvla.signals.methods import awr_chunk as baw
from axis_qvla.signals.methods import cfg_tags as bql
from axis_qvla.signals.blocks.tags import NO_TAG, NOT_TRAINABLE
from axis_qvla.realrobot.realign.ranges import tail_ranges, write_ranges
from axis_qvla.realrobot.realign.tags import build_pure_tags
from tests.golden.signals.test_paper_signal_rules import _phase_corpus


@pytest.fixture
def built(tmp_path, monkeypatch):
    idx, _rows = _phase_corpus(tmp_path)
    task = tmp_path / "corpus" / "task_900" / "meta"
    pd.DataFrame({"task": ["put the cup on the plate"]}).to_parquet(task / "tasks.parquet")
    # The shared fixture omits the row offsets a real corpus's meta carries; the flat row walk needs them.
    ep = task / "episodes" / "chunk-000" / "file-000.parquet"
    meta = pd.read_parquet(ep)
    meta["dataset_from_index"] = meta["length"].cumsum() - meta["length"]
    meta.to_parquet(ep)
    weights = tmp_path / "w_rank.json"
    monkeypatch.setattr("sys.argv", ["x", "--reward", "phase", "--binding", "names", "--index", str(idx),
                                     "--out", str(weights)])
    assert baw.main() == 0
    ranges = tmp_path / "ranges.json"
    write_ranges(ranges, tail_ranges(idx))
    pinned = tmp_path / "pinned" / "quality_phase.npz"
    assert bql.main(["--weights", str(weights), "--roots", str(idx), "--ranges", str(ranges),
                     "--out", str(pinned), "--seed", "0"]) == 0
    mine = tmp_path / "mine" / "quality_phase.npz"
    meta = build_pure_tags(weights, idx, ranges, mine)
    return np.load(pinned), np.load(mine), meta


def test_every_row_the_pinned_builder_tags_carries_the_same_tag(built):
    pinned, mine, _ = built
    p, m = pinned["tag"], mine["tag"]
    tagged = (p != NO_TAG) & (p != NOT_TRAINABLE)
    assert tagged.any() and (p == NO_TAG).any()
    np.testing.assert_array_equal(m[tagged], p[tagged])


def test_every_trainable_row_is_tagged_and_nothing_else_is(built):
    pinned, mine, meta = built
    p, m = pinned["tag"], mine["tag"]
    np.testing.assert_array_equal(m == NOT_TRAINABLE, p == NOT_TRAINABLE)
    assert not (m == NO_TAG).any()
    assert meta["realized_tagged_fraction"] == 1.0 and meta["realized_drop_whole"] == 0.0
    assert sum(meta["bin_row_counts"].values()) == meta["n_trainable"] == int((m != NOT_TRAINABLE).sum())


def test_the_artifact_declares_itself_pure_and_not_the_pinned_builders(built):
    pinned, mine, meta = built
    stored = json.loads(str(mine["meta"]))
    assert stored["pure_conditioning"] is True and stored["builder"] == "qvla.realrobot.realign.tags/1"
    assert stored["generator_sha256"] != json.loads(str(pinned["meta"]))["generator_sha256"]
    assert list(mine["prompts"]) == list(pinned["prompts"])


def test_a_weights_file_for_another_corpus_is_refused(built, tmp_path):
    _, _, _ = built
    w = json.loads((tmp_path / "w_rank.json").read_text())
    w["n_rows"] += 1
    bad = tmp_path / "w_bad.json"
    bad.write_text(json.dumps(w))
    with pytest.raises(ValueError, match="different corpora"):
        build_pure_tags(bad, tmp_path / "artifacts" / "roots.json", tmp_path / "ranges.json",
                        tmp_path / "x" / "quality_phase.npz")
