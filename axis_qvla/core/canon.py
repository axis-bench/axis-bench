"""Canonical JSON bytes and digest helpers for everything the v1.0 release pins.

Why this module exists: a lock file, a cell map or a harvest report is identified by the sha256
of its bytes, so one object must always serialise to one byte string -- sorted keys, one-space
indent, UTF-8 with no unicode escapes, no NaN, one trailing newline. Every writer of a release
record calls these helpers instead of json.dumps with its own options. pooled_r1.json, a T0
output with its own frozen byte format, is written by axis_qvla.paper.pool.

Core tier: stdlib only; Python >= 3.11.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

_CHUNK = 1 << 20


def dumps_canonical(obj: object) -> bytes:
    """Serialise `obj` canonically; NaN and Inf raise ValueError (allow_nan=False)."""
    text = json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False)
    return (text + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()
