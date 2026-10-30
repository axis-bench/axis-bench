"""Canonical bytes are the identity of every lock and record the v1.0 release publishes.

A lock is compared by the sha256 of its bytes, so two writers (or two runs of one writer) that
serialise the same object differently would make an unchanged release look changed -- or, worse,
make a changed one look pinned. These tests pin the byte format itself.
"""
from __future__ import annotations

import hashlib
import re

import pytest

from axis_qvla.core.canon import dumps_canonical, sha256_bytes, sha256_file


def test_keys_sorted_at_every_depth_indent_one_trailing_newline():
    got = dumps_canonical({"b": 1, "a": {"d": 2, "c": [3, {"z": 0, "y": 1}]}})
    assert got == (
        b'{\n "a": {\n  "c": [\n   3,\n   {\n    "y": 1,\n    "z": 0\n   }\n  ],\n'
        b'  "d": 2\n },\n "b": 1\n}\n'
    )


def test_non_ascii_is_utf8_bytes_not_an_escape():
    got = dumps_canonical({"dash": "—"})
    assert got == b'{\n "dash": "\xe2\x80\x94"\n}\n'
    assert b"\\u" not in got


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_floats_raise(bad):
    with pytest.raises(ValueError):
        dumps_canonical({"x": bad})


def test_two_calls_are_byte_identical():
    obj = {"k": [1.5, "s", None, True], "j": {"x": 1}}
    assert dumps_canonical(obj) == dumps_canonical(dict(reversed(list(obj.items()))))


def test_sha256_bytes_is_lowercase_hex():
    assert sha256_bytes(b"") == hashlib.sha256(b"").hexdigest()
    assert re.fullmatch(r"[0-9a-f]{64}", sha256_bytes(b"abc"))


def test_sha256_file_agrees_with_hashlib_across_chunk_boundaries(tmp_path):
    data = bytes(range(256)) * (3 * 4096) + b"tail!!!"   # 3 MiB + 7 bytes: several 1 MiB chunks
    p = tmp_path / "blob.bin"
    p.write_bytes(data)
    assert sha256_file(p) == hashlib.sha256(data).hexdigest()
