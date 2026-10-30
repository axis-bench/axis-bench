"""Name the first field in which two quality-signal files differ.

Stdlib only. A rebuild that differs from its locked file is reported by WHERE it differs -- the
npz member, the `meta` key inside a schedule or tag file, or the JSON key path of a weights file --
because "the sha256 differs" says nothing about whether an array moved or one diagnostic float in
the metadata did. The npz `meta` member is a 0-d unicode array; its text is read straight off the
.npy bytes so this needs no numpy.
"""
from __future__ import annotations

import ast
import io
import json
import pathlib
import zipfile


def _npy_text(blob: bytes) -> str | None:
    """The text of a 0-d unicode .npy member (the npz `meta`), else None."""
    if not blob.startswith(b"\x93NUMPY"):
        return None
    n = 2 if blob[6] == 1 else 4
    hlen = int.from_bytes(blob[8:8 + n], "little")
    header = ast.literal_eval(blob[8 + n:8 + n + hlen].decode("latin1"))
    if not str(header.get("descr", "")).startswith("<U") or header.get("shape") != ():
        return None
    return blob[8 + n + hlen:].decode("utf-32-le").rstrip("\x00")


def _first_key(x, y, prefix: str) -> str:
    if isinstance(x, dict) and isinstance(y, dict):
        for k in list(x) + [k for k in y if k not in x]:
            if x.get(k) != y.get(k):
                return _first_key(x.get(k), y.get(k), f"{prefix}.{k}" if prefix else k)
        return prefix or "(key order)"
    return prefix


def first_difference(got, want) -> str:
    """The first field in which two signal files differ: npz member / meta key, or JSON key."""
    a, b = pathlib.Path(got).read_bytes(), pathlib.Path(want).read_bytes()
    if a == b:
        return ""
    if a[:2] == b"PK" and b[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(a)) as za, zipfile.ZipFile(io.BytesIO(b)) as zb:
            if za.namelist() != zb.namelist():
                return f"members {za.namelist()} != {zb.namelist()}"
            for name in za.namelist():
                x, y = za.read(name), zb.read(name)
                if x != y:
                    tx, ty = _npy_text(x), _npy_text(y)
                    if tx is not None and ty is not None:
                        return name[:-4] + "." + _first_key(json.loads(tx), json.loads(ty), "")
                    return name[:-4]
    try:
        return _first_key(json.loads(a), json.loads(b), "")
    except ValueError:
        return "bytes"


def within_tolerance(got, want, tolerance: dict) -> bool:
    """True when two npz files differ ONLY in `meta` keys `tolerance` names ("meta.<key>": bound),
    each numeric value by at most its bound; every other member, meta key, value and the key order
    must be equal. This is how a spec declares a documented platform difference (a libm's last
    bits) so a replay can name it instead of reporting a regression."""
    a, b = pathlib.Path(got).read_bytes(), pathlib.Path(want).read_bytes()
    if not (a[:2] == b"PK" and b[:2] == b"PK"):
        return False
    with zipfile.ZipFile(io.BytesIO(a)) as za, zipfile.ZipFile(io.BytesIO(b)) as zb:
        if za.namelist() != zb.namelist():
            return False
        for name in za.namelist():
            x, y = za.read(name), zb.read(name)
            if x == y:
                continue
            tx, ty = _npy_text(x), _npy_text(y)
            if name != "meta.npy" or tx is None or ty is None:
                return False
            mx, my = json.loads(tx), json.loads(ty)
            if list(mx) != list(my):
                return False
            for k in mx:
                if mx[k] == my[k]:
                    continue
                bound = tolerance.get(f"meta.{k}")
                vals = (mx[k], my[k])
                if bound is None or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
                    return False
                if abs(vals[0] - vals[1]) > bound:
                    return False
    return True
