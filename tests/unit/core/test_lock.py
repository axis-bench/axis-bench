"""The v1.0 release lock: deterministic bytes, and a validator that catches every malformed row.

This is the read half (axis_qvla.core.lock): load_lock, validate_lock, lock_bytes, default_lock. Writing
a lock is the publisher's tool, tested with it; this file imports only the read half.
"""

from __future__ import annotations

import json

import pytest

from axis_qvla.core import lock as lk

C1, C2 = "a" * 40, "b" * 40
S = "d" * 64


def _valid_doc() -> dict:
    """A hand-built valid lock doc, shaped like what build_lock would produce from the
    test_lock_build fixtures -- built by hand so this file never imports the private build_lock."""
    return {
        "schema": lk.LOCK_SCHEMA,
        "release": "v1.0",
        "repos": {
            "axisrobotics/qvla-v1-pi05": {"type": "model", "tag": "v1.0", "commit": C2},
            "axisrobotics/qvla-v1-results": {"type": "dataset", "tag": "v1.0", "commit": C1},
        },
        "files": [
            {"repo": "axisrobotics/qvla-v1-pi05", "path": "ft/x/13579/params/a", "size": 3,
             "sha256": S, "git_oid": "3" * 40},
            {"repo": "axisrobotics/qvla-v1-results", "path": "cells.json", "size": 5,
             "sha256": "e" * 64, "git_oid": "1" * 40},
            {"repo": "axisrobotics/qvla-v1-results", "path": "rollouts/s2/x.tar", "size": 9,
             "sha256": S, "git_oid": "2" * 40},
        ],
    }


def test_lock_bytes_is_deterministic_and_newline_terminated():
    a = lk.lock_bytes(_valid_doc())
    b = lk.lock_bytes(_valid_doc())
    assert a == b and a.endswith(b"\n")
    doc = json.loads(a)
    assert [(f["repo"], f["path"]) for f in doc["files"]] == [
        ("axisrobotics/qvla-v1-pi05", "ft/x/13579/params/a"),
        ("axisrobotics/qvla-v1-results", "cells.json"),
        ("axisrobotics/qvla-v1-results", "rollouts/s2/x.tar")]
    assert doc["repos"]["axisrobotics/qvla-v1-pi05"] == {"type": "model", "tag": "v1.0", "commit": C2}


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d["repos"]["axisrobotics/qvla-v1-pi05"].update(commit="short"), "commit"),
    (lambda d: d["files"].reverse(), "sorted"),
    (lambda d: d["files"].append(dict(d["files"][0])), "duplicate"),
    (lambda d: d["files"][0].update(repo="axisrobotics/unknown"), "unknown repo"),
    (lambda d: d["files"][0].update(sha256="xyz"), "sha256"),
    (lambda d: d.update(schema="other"), "schema"),
    (lambda d: d.update(files={}), "files"),
    (lambda d: d["files"][0].update(repo=1), "repo"),
    (lambda d: d["files"][0].update(path="/abs/path"), "path"),
    (lambda d: d["files"][0].update(path="a/../../etc/passwd"), "path"),
    (lambda d: d["files"][0].update(path="a\\b"), "path"),
    (lambda d: d["files"][0].update(size=-1), "size"),
    (lambda d: d["files"][0].update(size="3"), "size"),
])
def test_validate_lock_reports_each_defect(mutate, message):
    doc = _valid_doc()
    mutate(doc)
    assert any(message in p for p in lk.validate_lock(doc)), lk.validate_lock(doc)


def test_validate_lock_never_raises_on_malformed_row_types():
    """A row with non-string repo/path or an unhashable field must not crash
    validate_lock -- it must return problems, never a TypeError that escapes _cmd_verify."""
    doc = _valid_doc()
    doc["files"][0]["repo"] = ["not", "a", "string"]
    doc["files"].append({"repo": {"nested": 1}, "path": 5, "size": "x", "sha256": None, "git_oid": None})
    problems = lk.validate_lock(doc)  # must not raise
    assert problems


def test_a_repo_listed_with_zero_files_is_a_problem():
    doc = _valid_doc()
    doc["files"] = [f for f in doc["files"] if f["repo"] != "axisrobotics/qvla-v1-pi05"]
    assert any("no files listed" in p for p in lk.validate_lock(doc))


def test_files_as_a_dict_is_refused_not_silently_treated_as_empty(tmp_path):
    """A `files` that is not a list is refused, so a checksum manifest with no rows never verifies
    clean."""
    doc = {"schema": lk.LOCK_SCHEMA, "release": "v1.0",
           "repos": {"o/r": {"type": "dataset", "tag": "main", "commit": "a" * 40}}, "files": {}}
    p = tmp_path / "lock.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="invalid release checksum manifest"):
        lk.load_lock(p)


def test_load_lock_round_trips_and_refuses_invalid(tmp_path):
    p = tmp_path / "v1.0.lock.json"
    p.write_bytes(lk.lock_bytes(_valid_doc()))
    assert lk.load_lock(p)["release"] == "v1.0"
    p.write_text(json.dumps({"schema": "nope"}))
    with pytest.raises(ValueError, match="invalid release checksum manifest"):
        lk.load_lock(p)


def test_the_read_half_imports_no_hf_client():
    import ast, pathlib
    import axis_qvla.core.lock as cl
    src = pathlib.Path(cl.__file__).read_text()
    mods = {n.module for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom)}
    assert not any(m and "hf_http" in m for m in mods)
