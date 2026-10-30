"""Every registered constant is the value its source holds."""
from __future__ import annotations

import re

import pytest

from tests.docs import constants_page

ALL = [c for _, cs in constants_page.constants() for c in cs]


@pytest.mark.parametrize("c", ALL, ids=lambda c: c.name)
def test_every_constant_is_the_value_its_source_holds(c):
    assert constants_page.resolve(c) == c.value


LOCATION = re.compile(r"^(Eq\. \(?\d+\)?|Sec\. [IVX]+(\.[A-Z])?|Table [IVX]+|Fig\. \d+|App\. [A-Z]"
                      r"|implementation \(not in the paper\))$")


@pytest.mark.parametrize("c", ALL, ids=lambda c: c.name)
def test_every_constant_names_its_place_in_the_paper_or_says_it_is_not_there(c):
    assert LOCATION.match(c.equation), c.equation


def test_names_are_unique_across_the_registries():
    names = [c.name for c in ALL]
    assert len(names) == len(set(names))


def test_a_file_source_resolves_a_key_path_and_a_count():
    from axis_qvla.labels.constants import Constant

    tasks = Constant("X", 0, "", "", "conf/benchmarks/axis-heldout-v1/franka_heldout24.json", "len:task_ids")
    seeds = Constant("Y", 0, "", "", "conf/benchmarks/axis-heldout-v1/franka_heldout24.json", "n_seeds")
    assert (constants_page.resolve(tasks), constants_page.resolve(seeds)) == (24, 50)


def test_the_data_page_tabulates_the_label_constants_a_method_needs():
    text = constants_page.DATA_PAGE.read_text(encoding="utf-8")
    assert "<!-- label-constants:begin -->\n" in text and "\n<!-- label-constants:end -->" in text
    block = text.split("<!-- label-constants:begin -->\n", 1)[1].split("\n<!-- label-constants:end -->", 1)[0]
    assert block == constants_page.label_table(), \
        "regenerate the block: python -m tests.docs.constants_page --label-table"
