from __future__ import annotations

import json

import pytest

from axis_qvla.signals.methods import schedule as bis
from axis_qvla.signals.provenance import corpus_fingerprint
from axis_qvla.realrobot.realign.ranges import tail_ranges, write_ranges
from axis_qvla.realrobot.realign.view import ViewEpisode, view_episodes_from_map, write_meta_view


def _eps():
    return [ViewEpisode(912, 1, 30, 5, "stack the cups"), ViewEpisode(800, 0, 12, 9, "open the box"),
            ViewEpisode(800, 1, 8, 11, "open the box"), ViewEpisode(912, 0, 25, 4, "stack the cups")]


def test_the_view_is_read_by_the_schedule_walk_in_task_then_episode_order(tmp_path):
    roots = write_meta_view(tmp_path, _eps())
    ordered = bis._ordered_roots(roots)
    assert [t for t, _ in ordered] == [800, 912]
    offsets, total = bis._episode_offsets(ordered[0][1])
    assert offsets == {0: (0, 12), 1: (12, 8)} and total == 20
    offsets, total = bis._episode_offsets(ordered[1][1])
    assert offsets == {0: (0, 25), 1: (25, 30)} and total == 55


def test_the_roots_index_is_relative(tmp_path):
    roots = write_meta_view(tmp_path, _eps())
    doc = json.loads(roots.read_text())
    assert doc["version"] == 2 and not doc["base"].startswith("/")
    assert all(not v.startswith("/") for v in doc["roots"].values())


def test_the_corpus_fingerprint_is_stable_and_sees_attempt_ids(tmp_path):
    write_meta_view(tmp_path / "a", _eps())
    write_meta_view(tmp_path / "b", list(reversed(_eps())))
    assert corpus_fingerprint(tmp_path / "a" / "view") == corpus_fingerprint(tmp_path / "b" / "view")
    changed = [ViewEpisode(e.task_id, e.episode_index, e.length, e.attempt_id + 1, e.prompt) for e in _eps()]
    write_meta_view(tmp_path / "c", changed)
    assert corpus_fingerprint(tmp_path / "c" / "view") != corpus_fingerprint(tmp_path / "a" / "view")


def test_tail_ranges_drop_the_last_ten_frames_and_empty_short_episodes(tmp_path):
    roots = write_meta_view(tmp_path, _eps())
    r = tail_ranges(roots)
    assert r == {"task_800--0": [[0, 2]], "task_800--1": [[0, 0]],
                 "task_912--0": [[0, 15]], "task_912--1": [[0, 20]]}
    write_ranges(tmp_path / "ranges.json", r)
    assert json.loads((tmp_path / "ranges.json").read_text()) == r


def test_two_prompts_for_one_task_are_kept(tmp_path):
    eps = _eps() + [ViewEpisode(800, 2, 9, 12, "open the lid")]
    write_meta_view(tmp_path, eps)
    import pandas as pd
    assert sorted(pd.read_parquet(tmp_path / "view" / "task_800" / "meta" / "tasks.parquet")["task"]) == \
        ["open the box", "open the lid"]


def test_episodes_from_the_alignment_map(tmp_path):
    amap = tmp_path / "map.json"
    amap.write_text(json.dumps({"episodes": {
        "task_800--1": {"attempt_id": 11, "export_episode_index": 0, "train_frame_count": 8, "export_frame_count": 8},
        "task_800--0": {"attempt_id": 9, "export_episode_index": 1, "train_frame_count": 12, "export_frame_count": 12}}}))
    eps = view_episodes_from_map(amap, {800: "open the box"})
    assert eps == [ViewEpisode(800, 0, 12, 9, "open the box"), ViewEpisode(800, 1, 8, 11, "open the box")]
    with pytest.raises(KeyError, match="800"):
        view_episodes_from_map(amap, {801: "x"})
