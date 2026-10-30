"""Every parameter the paper leaves undefined is a named constant, equal to what the code uses."""
from __future__ import annotations

import importlib

import pytest

from axis_qvla.labels.constants import CONSTANTS

@pytest.mark.parametrize("c", CONSTANTS, ids=lambda c: c.name)
def test_every_named_constant_is_the_value_the_code_uses(c):
    assert getattr(importlib.import_module(c.module), c.attribute) == c.value


def test_names_are_unique():
    names = [c.name for c in CONSTANTS]
    assert len(names) == len(set(names))
