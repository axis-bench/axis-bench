"""Tail-10: one [0, L - 10] range per episode, written exactly as openpi's filter file."""
from __future__ import annotations

import json

import pytest

from axis_qvla.realrobot.realdata.tail import TAIL, chunk_starts, dumps, episode_lengths, tail_ranges


def test_each_episode_gets_one_range_ending_ten_frames_early():
    assert TAIL == 10
    assert tail_ranges({0: 130, 1: 138}) == {"0": [[0, 120]], "1": [[0, 128]]}


def test_an_episode_of_ten_frames_or_fewer_has_no_start():
    assert tail_ranges({0: 11, 1: 10, 2: 3, 3: 0}) == {"0": [[0, 1]], "1": [], "2": [], "3": []}


def test_keys_are_in_episode_order_whatever_the_input_order():
    assert list(tail_ranges({10: 50, 2: 50, 1: 50})) == ["1", "2", "10"]


def test_the_bytes_are_openpis_json_dump():
    assert dumps(tail_ranges({1: 20, 0: 130, 2: 10})) == '{"0": [[0, 120]], "1": [[0, 10]], "2": []}'


def test_chunk_starts_counts_every_start():
    assert chunk_starts(tail_ranges({0: 130, 1: 138, 2: 5})) == 248


def test_other_tail_lengths_and_bad_input():
    assert tail_ranges({0: 20}, n=0) == {"0": [[0, 20]]}
    with pytest.raises(ValueError):
        tail_ranges({0: 20}, n=-1)
    with pytest.raises(ValueError):
        tail_ranges({0: -1})


def test_episode_lengths_reads_episodes_jsonl(tmp_path):
    p = tmp_path / "episodes.jsonl"
    p.write_text("".join(json.dumps({"episode_index": i, "tasks": ["t"], "length": n}) + "\n"
                         for i, n in ((1, 40), (0, 30))))
    assert episode_lengths(p) == {1: 40, 0: 30}
    p.write_text(p.read_text() + json.dumps({"episode_index": 1, "tasks": ["t"], "length": 9}) + "\n")
    with pytest.raises(ValueError, match="twice"):
        episode_lengths(p)
