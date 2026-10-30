"""The projection the paper-config lock and the parity check share."""
from __future__ import annotations

import pytest

from axis_qvla.train.parity import compare, flatten, lock_sha256, normalized, not_compared


def test_flatten_keeps_lists_and_empty_tables_as_leaves():
    assert flatten({"a": {"b": 1, "c": {"d": [1, 2]}}, "e": {}, "f": None}) == {
        "a.b": 1, "a.c.d": [1, 2], "e": {}, "f": None}


def test_numbers_compare_by_value():
    assert lock_sha256({"clip": 10}) == lock_sha256({"clip": 10.0})
    assert lock_sha256({"flag": True}) != lock_sha256({"flag": 1})
    assert normalized({"x": [1, {"y": 2}]}) == {"x": [1.0, {"y": 2.0}]}
    with pytest.raises(TypeError):
        normalized({"x": object()})


def test_machine_and_logging_keys_are_not_compared():
    for key in ("output_dir", "dataset.root", "wandb.enable", "eval.n_episodes", "dataset.video_backend"):
        assert not_compared(key), key
    for key in ("dataset.repo_id", "policy.chunk_size", "optimizer.lr", "seed", "parallelism.dp_replicate"):
        assert not not_compared(key), key


# --- compare(): field-by-field, and every way a declared [paper.differences] entry is itself
# checked rather than trusted ---------------------------------------------------------------

ROW = {"fields": {"a": 1, "b": None}}


def test_a_field_the_projection_lacks_is_reported():
    assert compare({}, ROW, {}) == ["a: arm '<absent>' != lock 1.0", "b: arm '<absent>' != lock None"]


def test_a_field_the_lock_lacks_is_reported():
    assert compare({"a": 1, "b": None, "c": 1}, ROW, {}) == ["c: arm 1.0 != lock '<absent>'"]


def test_the_absent_sentinel_stored_as_real_lock_data_does_not_self_match():
    """A lock row that happens to store the literal string "<absent>" as a value is not confused
    with a field the projection or the lock simply does not carry."""
    assert compare({}, {"fields": {"z": "<absent>"}}, {}) == []


def test_a_declared_difference_that_actually_agrees_is_reported():
    assert compare({"a": 1, "b": None}, ROW, {"a": {"arm": "1", "lock": "1", "reason": "r"}}) == [
        "a: declared different, but the arm and the lock agree (1.0)"]


def test_a_declared_difference_that_matches_is_accepted():
    assert compare({"a": 2, "b": None}, ROW, {"a": {"arm": "2", "lock": "1", "reason": "r"}}) == []


def test_a_declared_difference_with_the_wrong_values_is_reported():
    assert compare({"a": 2, "b": None}, ROW, {"a": {"arm": "3", "lock": "1", "reason": "r"}}) == [
        "a: declared difference does not match (arm 2.0, lock 1.0)"]


def test_a_declared_difference_without_a_reason_is_refused():
    assert compare({"a": 2, "b": None}, ROW, {"a": {"arm": "2", "lock": "1"}}) == [
        "a: declared difference has no reason"]
    assert compare({"a": 2, "b": None}, ROW, {"a": {"arm": "2", "lock": "1", "reason": "  "}}) == [
        "a: declared difference has no reason"]


def test_a_declared_difference_missing_arm_or_lock_is_reported_not_raised():
    assert compare({"a": 2, "b": None}, ROW, {"a": {"reason": "r"}}) == [
        "a: declared difference is malformed (needs arm, lock and reason)"]
    assert compare({"a": 2, "b": None}, ROW, {"a": {"arm": "2", "reason": "r"}}) == [
        "a: declared difference is malformed (needs arm, lock and reason)"]
    assert compare({"a": 2, "b": None}, ROW, {"a": "not even a table"}) == [
        "a: declared difference is malformed (needs arm, lock and reason)"]


def test_a_declared_difference_with_invalid_json_is_reported_not_raised():
    assert compare({"a": 2, "b": None}, ROW, {"a": {"arm": "not json!", "lock": "1", "reason": "r"}}) == [
        "a: declared difference's arm/lock is not valid JSON (JSONDecodeError)"]


def test_a_declared_difference_for_a_field_neither_side_has_is_reported():
    assert compare({"a": 1, "b": None}, ROW, {"q": {"arm": "2", "lock": "1", "reason": "r"}}) == [
        "q: declared a [paper.differences] entry for a field neither the arm nor the lock has"]
