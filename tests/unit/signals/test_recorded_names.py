"""Recorded identities spell module names under the development root, in every tree.

A row space records the idle filter it was cut with by function name, and its id (and every bundle
id over it) is a sha256 over that record, so the published ids hold only while every tree records
the same spelling: the development one, `<root>.labels.idle_filter.nonidle_ranges` with the root
of axis_qvla.core.canonical. These ids are pinned: the toy corpus's row space, built from disk with the
real idle filter, and the docs' example method compiled over it, give the same ids whatever the
import root is called. Each recorded name must still name live code.
"""
from __future__ import annotations

import dataclasses
import importlib

from axis_qvla.core import canonical
from axis_qvla.data import lerobot_local
from axis_qvla.labels import idle_filter
from axis_qvla.signals.bundle import bundle_id, rowspace_id
from axis_qvla.signals.compile import Compose, compile_program
from axis_qvla.signals.imported import ImportedScores
from axis_qvla.signals.methods import schedule as bis
from axis_qvla.signals.methods.filtering import TopFraction
from axis_qvla.signals.planners import SchedulePlanner
from axis_qvla.signals.schema import Budget
from tests.unit.signals.toy_corpus import write_toy_corpus

# Independent of the import root's name, which is the point.
TOY_ROWSPACE = "76e7857cc4b7615f130686e9a9d58ba37c63e2540f61173327f379100359b325"
EXAMPLE_BUNDLE = "6cc5e4e5dfc8126a79d116ee7acf64fc324a406c3a4f8c46193b0196705198ed"
EXAMPLE_CONTROL = "d756db9affc6439c518a6aad227c1ae0b0aee49334fba4dbe0489af5ea475698"


def _resolve(recorded: str):
    module, _, attr = canonical.live_name(recorded).rpartition(".")
    return getattr(importlib.import_module(module), attr)


def test_the_idle_filter_names_its_code_under_the_canonical_root():
    f = idle_filter.IDLE_FILTER
    assert f["function"] == canonical.canonical_name("labels.idle_filter.nonidle_ranges")
    assert _resolve(f["function"]) is idle_filter.nonidle_ranges
    name = canonical.canonical_name("data.lerobot_local.state_to_idle_signal")
    assert f"({name})" in f["signal"] and _resolve(name) is lerobot_local.state_to_idle_signal


@dataclasses.dataclass(frozen=True)
class ImportedValues:
    def score(self, labels, rows):
        return labels.values()


def test_the_toy_row_space_and_the_example_bundle_keep_their_ids(tmp_path, monkeypatch):
    monkeypatch.delenv(bis.CORPUS_ROOT_ENV, raising=False)
    toy = write_toy_corpus(tmp_path)
    assert rowspace_id(toy.rs) == TOY_ROWSPACE
    keep_top = Compose(method="example.keep_top", version=1, params={"keep": 0.3}, scorer=ImportedValues(),
                       selector=TopFraction(0.3), planner=SchedulePlanner("drop_top"))
    bundle, controls = compile_program(keep_top, ImportedScores.from_json(toy.weights), toy.rs, Budget(6, 2), 0)
    assert (bundle_id(bundle), [bundle_id(c) for c in controls]) == (EXAMPLE_BUNDLE, [EXAMPLE_CONTROL])
