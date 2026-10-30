"""Legacy codecs write the paper's file formats: npz members in order, json with the writer's indent."""
from __future__ import annotations

import io
import json

import numpy as np
import pytest

from axis_qvla.signals import legacy, otree
from axis_qvla.signals.schema import LegacyFile


def _savez(**members) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **members)
    return buf.getvalue()


def test_an_npz_record_is_the_members_np_savez_would_write():
    rows = np.arange(6, dtype=np.int64).reshape(3, 2)
    meta = {"mode": "drop_top", "seed": 0, "keep_fraction": 0.5, "realized": {"1": 3}}
    want = _savez(rows=rows, meta=np.array(json.dumps(meta)))
    lf = LegacyFile("openpi-legacy", None, "signals/s.npz", "npz",
                    {"$o": [["rows", {"$array": "plan"}], ["meta", {"$json": otree.encode(meta)}]]})
    assert legacy.render(lf, {"plan": rows}) == want


def test_string_members_keep_numpys_own_unicode_width():
    prompts = ["open the drawer", "pick up the bowl"]
    want = _savez(tag=np.zeros(3, np.uint8), prompts=np.array(prompts), meta=np.array("{}"))
    lf = LegacyFile("openpi-legacy", None, "t.npz", "npz", {"$o": [
        ["tag", {"$array": "legacy.t_tag"}], ["prompts", {"$strings": prompts}], ["meta", {"$json": {"$o": []}}]]})
    assert legacy.render(lf, {"legacy.t_tag": np.zeros(3, np.uint8)}) == want


@pytest.mark.parametrize("codec,indent,newline", [("json", None, ""), ("json-indent1", 1, ""), ("json-indent2-nl", 2, "\n")])
def test_json_codecs_use_the_original_writers_indent(codec, indent, newline):
    obj = {"version": 2, "base": "../corpus", "roots": {"1000": "task_1000", "800": "task_800"}}
    lf = LegacyFile("openpi-legacy", None, "index/r.json", codec, otree.encode(obj))
    assert legacy.render(lf, {}) == (json.dumps(obj, indent=indent) + newline).encode()


def test_episode_floats_rebuild_per_episode_lists_from_one_float32_array():
    w = {"task_800--0": [2.0, 1.5], "task_800--1": [], "task_1000--0": [0.75]}
    doc = {"$o": [["reward_id", "v2"], ["weights", {"$episode_floats": {
        "keys": list(w), "lengths": "legacy.w_lengths", "values": "legacy.w_values"}}], ["n", 3]]}
    arrays = {"legacy.w_lengths": np.array([2, 0, 1]), "legacy.w_values": np.array([2.0, 1.5, 0.75], np.float32)}
    lf = LegacyFile("openpi-legacy", None, "a.json", "json", doc)
    assert legacy.render(lf, arrays) == json.dumps({"reward_id": "v2", "weights": w, "n": 3}).encode()


def test_spans_rebuild_a_ranges_file():
    r = {"task_1000--0": [[0, 3], [6, 100]], "task_800--0": [[1, 3]], "task_800--2": []}
    doc = {"$spans": {"keys": list(r), "counts": "c", "spans": "s"}}
    arrays = {"c": np.array([2, 1, 0]), "s": np.array([[0, 3], [6, 100], [1, 3]])}
    assert legacy.render(LegacyFile("openpi-legacy", None, "r.json", "json", doc), arrays) == json.dumps(r).encode()


def test_a_table_whose_counts_do_not_add_up_is_refused():
    doc = {"$spans": {"keys": ["a"], "counts": "c", "spans": "s"}}
    with pytest.raises(legacy.LegacyError):
        legacy.render(LegacyFile("openpi-legacy", None, "r.json", "json", doc),
                      {"c": np.array([2]), "s": np.array([[0, 1]])})


def test_codecs_follow_the_published_file_names():
    assert legacy.codec_for_path("signals/v1/approach_aware/chunk/tags.npz") == "npz"
    assert legacy.codec_for_path("index/roots_libero_5k_v2_d8.json") == "json-indent2-nl"
    assert legacy.codec_for_path("index/subsets/scaling_1000.ranges.json") == "json-indent1"
    assert legacy.codec_for_path("index/ranges_libero_5k_v2.json") == "json"


def test_decompose_then_render_is_the_identity():
    rows = np.arange(8, dtype=np.int64).reshape(4, 2)
    data = _savez(rows=rows, meta=np.array(json.dumps({"b": 1, "a": [0.1, 0.2]})))
    lf, arrays = legacy.decompose(data, path="signals/x.npz", lock_id="x", fmt="openpi-legacy", name="x")
    assert legacy.render(lf, {f"legacy.{k}": v for k, v in arrays.items()}) == data
    w = json.dumps({"version": 1, "weights": {"task_1--0": [1.0, 2.0], "task_1--1": [1.25]},
                    "provenance": {"z": 1, "a": 2}}).encode()
    lf, arrays = legacy.decompose(w, path="signals/w.json", lock_id=None, fmt="openpi-legacy", name="w")
    assert legacy.render(lf, {f"legacy.{k}": v for k, v in arrays.items()}) == w
    r = json.dumps({"task_1--0": [[0, 4]], "task_1--1": []}, indent=1).encode()
    lf, arrays = legacy.decompose(r, path="index/subsets/s.ranges.json", lock_id=None, fmt="openpi-legacy", name="r")
    assert legacy.render(lf, {f"legacy.{k}": v for k, v in arrays.items()}) == r


def test_decompose_refuses_weights_that_float32_cannot_hold():
    w = json.dumps({"weights": {"task_1--0": [0.1]}}).encode()   # 0.1 is not a float32 value
    with pytest.raises(legacy.LegacyError):
        legacy.decompose(w, path="w.json", lock_id=None, fmt="openpi-legacy", name="w")


def test_decompose_refuses_a_file_it_cannot_render_back_byte_for_byte():
    # a duplicate JSON key collapses on json.loads, so decompose cannot reproduce these exact bytes.
    data = b'{"a": 1, "a": 2}'
    with pytest.raises(legacy.LegacyError):
        legacy.decompose(data, path="w.json", lock_id=None, fmt="openpi-legacy", name="w")
