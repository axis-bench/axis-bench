"""Ordered trees keep a legacy file's key order through canonical (sorted-key) JSON."""
from __future__ import annotations

import json

import numpy as np
import pytest

from axis_qvla.core.canon import dumps_canonical
from axis_qvla.signals import otree


def test_key_order_survives_a_canonical_round_trip():
    obj = {"z": 1, "a": [{"y": 2.5, "b": None}], "m": {"k": True, "c": "s"}}
    tree = json.loads(dumps_canonical(otree.encode(obj)))
    back = otree.decode(tree)
    assert json.dumps(back) == json.dumps(obj)
    assert list(back) == ["z", "a", "m"] and list(back["m"]) == ["k", "c"]


def test_numpy_scalars_become_python_scalars():
    tree = otree.encode({"n": np.int64(3), "x": np.float64(0.25), "b": np.bool_(True)})
    assert otree.decode(tree) == {"n": 3, "x": 0.25, "b": True}


@pytest.mark.parametrize("bad", [{1: "a"}, {"x": float("nan")}, {"x": object()}])
def test_what_json_cannot_hold_is_refused(bad):
    with pytest.raises(otree.OTreeError):
        otree.encode(bad)


def test_codec_nodes_go_to_the_resolver_and_plain_objects_are_refused():
    tree = {"$o": [["w", {"$thing": 7}]]}
    assert otree.decode(tree, lambda kind, arg: (kind, arg)) == {"w": ("$thing", 7)}
    with pytest.raises(otree.OTreeError):
        otree.decode(tree)
    with pytest.raises(otree.OTreeError):
        otree.decode({"a": 1, "b": 2})
