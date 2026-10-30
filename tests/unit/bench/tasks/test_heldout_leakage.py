"""A scored task that stage-1 trained on measures memorisation, not transfer.

On the real indices: 10 of the 20 held-out corpus tasks ARE in stage-1's 731,
and the 10 scored happen to be exactly the 10 that are not. These tests exist so that stays a
property instead of a coincidence between two hand-maintained lists.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from axis_qvla.bench.tasks.leakage import (
    LeakageError,
    assert_no_leakage,
    benchmark_task_ids,
    leaked_tasks,
    read_roots,
)

# The real numbers, so a regression in either list is visible here.
STAGE1_LEAKED = ["810", "815", "929", "970", "1046", "1252", "1427", "1458", "1746", "1891"]
SCORED = ["809", "811", "866", "868", "945", "953", "966", "1426", "1459", "1889"]


def _v2(tmp_path, ids, name="roots_v2.json"):
    p = tmp_path / name
    p.write_text(json.dumps({
        "version": 2, "base": "../corpus_d8", "derived_from": "roots.json",
        "action_space": "droid8: [7 joint velocities rad/s, gripper closedness 0..1] @ 15 fps",
        "roots": {str(i): f"task_{i}" for i in ids},
    }))
    return p


def _v1(tmp_path, ids, name="roots_v1.json"):
    p = tmp_path / name
    p.write_text(json.dumps({str(i): f"/abs/task_{i}" for i in ids}))
    return p


class TestReadRoots:
    def test_reads_a_v2_index_from_its_nested_roots(self, tmp_path):
        assert set(read_roots(_v2(tmp_path, [1, 2, 3]))) == {"1", "2", "3"}

    def test_reads_a_v1_bare_mapping(self, tmp_path):
        assert set(read_roots(_v1(tmp_path, [4, 5]))) == {"4", "5"}

    def test_a_versioned_index_without_roots_is_refused_not_read_as_task_ids(self, tmp_path):
        """The bug this prevents: reading a v2 file as v1 yields metadata keys, and every
        disjointness check then passes vacuously."""
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"version": 2, "base": "../x", "derived_from": "y"}))
        with pytest.raises(ValueError, match="no 'roots' key"):
            read_roots(p)

    def test_a_non_object_index_is_refused(self, tmp_path):
        p = tmp_path / "list.json"
        p.write_text(json.dumps(["task_1", "task_2"]))
        with pytest.raises(ValueError):
            read_roots(p)

    def test_a_missing_index_raises_rather_than_reading_as_empty(self, tmp_path):
        with pytest.raises(OSError):
            read_roots(tmp_path / "nope.json")


class TestLeakage:
    def test_the_real_scored_set_is_clean_against_the_real_leaked_set(self, tmp_path):
        idx = _v2(tmp_path, STAGE1_LEAKED)
        assert leaked_tasks(SCORED, idx) == []
        assert_no_leakage(SCORED, idx)

    def test_a_leaked_task_is_caught_and_named(self, tmp_path):
        idx = _v2(tmp_path, STAGE1_LEAKED)
        with pytest.raises(LeakageError, match="1891"):
            assert_no_leakage(SCORED + ["1891"], idx)

    def test_it_reports_every_leak_not_just_the_first(self, tmp_path):
        idx = _v2(tmp_path, STAGE1_LEAKED)
        assert leaked_tasks(["810", "1891", "811"], idx) == ["810", "1891"]

    def test_ids_compare_as_strings_so_int_keys_still_match(self, tmp_path):
        idx = _v2(tmp_path, [810, 815])
        assert leaked_tasks([810, 999], idx) == ["810"]

    def test_an_empty_pretrain_index_cannot_silently_pass(self, tmp_path):
        """A v2 file whose roots are genuinely empty is legal, but then nothing is leaked --
        the danger is an index that FAILS to parse, which read_roots raises on."""
        assert leaked_tasks(SCORED, _v2(tmp_path, [])) == []


GOLDEN_ROOTS = Path(__file__).resolve().parents[3] / "golden" / "bench" / "roots_libero_5k_v2_d8.json"
GOLDEN_SHA = "5a24a938405193689146184171b08f184c5eecdcc65295109a184d427d52daeb"


class TestTheBenchmarkIsNotInThePaperCorpus:
    """The guard reads every task the benchmark scores (24), from the declaration."""

    def test_the_fixture_is_the_published_index(self):
        assert hashlib.sha256(GOLDEN_ROOTS.read_bytes()).hexdigest() == GOLDEN_SHA

    def test_the_guard_covers_all_24_benchmark_tasks(self):
        ids = benchmark_task_ids()
        assert len(ids) == 24 and {"806", "1889", "1459"} <= set(ids)

    def test_no_benchmark_task_is_in_the_corpus_roots_index(self):
        assert len(read_roots(GOLDEN_ROOTS)) == 731
        assert leaked_tasks(benchmark_task_ids(), GOLDEN_ROOTS) == []
        assert_no_leakage(benchmark_task_ids(), GOLDEN_ROOTS)
