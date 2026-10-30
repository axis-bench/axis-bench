"""Ordered JSON trees: JSON values whose object key order survives canonical serialisation.

A bundle manifest is canonical JSON (sorted keys), but a legacy file's metadata must be written
back in the order it was written in, or its bytes change. So an object is stored as
{"$o": [[key, value], ...]}; lists, strings, numbers, booleans and null are stored as themselves.
Any other single-key object whose key starts with "$" is a node a codec resolves
(axis_qvla.signals.legacy). Stdlib only.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any


class OTreeError(ValueError):
    """A value cannot be stored as, or read back from, an ordered tree."""


def encode(obj: Any) -> Any:
    if type(obj).__module__ == "numpy" and hasattr(obj, "item"):
        obj = obj.item()
    if isinstance(obj, Mapping):
        pairs = []
        for k, v in obj.items():
            if not isinstance(k, str):
                raise OTreeError(f"object key {k!r} is not a string")
            pairs.append([k, encode(v)])
        return {"$o": pairs}
    if isinstance(obj, (list, tuple)):
        return [encode(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        raise OTreeError(f"{obj!r} is not representable in JSON")
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    raise OTreeError(f"{type(obj).__name__} is not a JSON value")


def decode(tree: Any, resolve: Callable[[str, Any], Any] | None = None) -> Any:
    if isinstance(tree, Mapping):
        if set(tree) == {"$o"}:
            return {k: decode(v, resolve) for k, v in tree["$o"]}
        if len(tree) == 1:
            (key, arg), = tree.items()
            if isinstance(key, str) and key.startswith("$"):
                if resolve is None:
                    raise OTreeError(f"node {key} needs a codec to resolve it")
                return resolve(key, arg)
        raise OTreeError(f"a plain object inside an ordered tree (keys {sorted(map(str, tree))})")
    if isinstance(tree, list):
        return [decode(v, resolve) for v in tree]
    return tree
