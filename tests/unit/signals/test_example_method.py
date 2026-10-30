"""A new method composed from the shipped blocks, end to end: experiments/examples/keep_top.py, which
docs/tutorials/write-a-quality-method.md walks through."""
from __future__ import annotations

import numpy as np

from axis_qvla.signals.bundle import load_bundle, write_bundle
from axis_qvla.signals.compile import compile_program
from axis_qvla.signals.imported import ImportedScores
from axis_qvla.signals.methods import schedule as bis
from axis_qvla.signals.methods.filtering import TopFraction
from axis_qvla.signals.schema import Budget
from experiments.examples.keep_top import keep_top
from tests.unit.signals.toy_corpus import write_toy_corpus


def test_a_composed_method_compiles_with_its_control_and_round_trips(tmp_path, monkeypatch):
    monkeypatch.delenv(bis.CORPUS_ROOT_ENV, raising=False)
    toy = write_toy_corpus(tmp_path)
    scores = ImportedScores.from_json(toy.weights)
    bundle, controls = compile_program(keep_top(0.3), scores, toy.rs, Budget(6, 2), 0)
    kept = np.concatenate(TopFraction(0.3).select(scores.values(), toy.rs).blocks)
    assert set(bundle.plan.ravel().tolist()) <= set(kept.tolist())
    assert bundle.requires == frozenset({"plan"}) and len(controls) == 1
    assert controls[0].program["method"] == "example.keep_top.control"
    assert load_bundle(tmp_path / "b" / write_bundle(bundle, tmp_path / "b")).plan.tolist() == bundle.plan.tolist()
