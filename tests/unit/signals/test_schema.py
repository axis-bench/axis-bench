"""The SignalBundle schema refuses what a consumer could not use, and is stdlib-only."""
from __future__ import annotations

import ast
import dataclasses
import sys
from pathlib import Path

import pytest

from axis_qvla.signals.schema import (BARE, Budget, CondVocab, LegacyFile, RowSpace, SchemaError, ServeCondition,
                                 check_serve)
from axis_qvla.signals.testing import toy_bundle

REPO = Path(__file__).resolve().parents[3]


def test_budget_counts_draws_and_refuses_nonpositive_sizes():
    assert Budget(20605, 64).samples == 1_318_720
    for bad in ((0, 64), (10, -1), (True, 64), (1.5, 64)):
        with pytest.raises(SchemaError):
            Budget(*bad)


def test_the_vocabulary_renders_the_exact_training_prompt():
    v = CondVocab(("quality",), {"quality": {q: f"\nQuality: {q}" for q in range(1, 6)}},
                  ("pick up the bowl", "open the drawer"))
    assert v.render("pick up the bowl", [5]) == "pick up the bowl\nQuality: 5"
    assert v.render("pick up the bowl", [0]) == "pick up the bowl"
    assert v.worst_case_prompt() == "pick up the bowl\nQuality: 5"
    with pytest.raises(SchemaError):
        v.render("x", [255])
    with pytest.raises(SchemaError):
        v.render("x", [5, 5])


@pytest.mark.parametrize("fields,text", [
    (("q", "q"), {"q": {1: "a"}}),            # a field declared twice
    (("q",), {"q": {}}),                       # no code text
    (("q",), {"q": {0: "a"}}),                 # 0 is the dropped code
    (("q",), {"q": {255: "a"}}),               # 255 is forbidden
    (("q",), {"q": {1: "a"}, "r": {1: "b"}}),  # text for an undeclared field
])
def test_a_vocabulary_that_could_tag_a_field_twice_or_emit_a_reserved_code_is_refused(fields, text):
    with pytest.raises(SchemaError):
        CondVocab(fields, text)


def test_weights_and_reducer_and_cond_and_vocab_must_agree():
    b = toy_bundle(weights=True, cond=True)
    with pytest.raises(SchemaError):
        dataclasses.replace(b, reducer="mean")
    with pytest.raises(SchemaError):
        dataclasses.replace(toy_bundle(), reducer="global_ratio")
    with pytest.raises(SchemaError):
        dataclasses.replace(b, vocab=None)
    with pytest.raises(SchemaError):
        dataclasses.replace(b, requires=frozenset({"plan", "telepathy"}))
    with pytest.raises(SchemaError):
        dataclasses.replace(b, serve=())


def test_weights_or_cond_without_the_matching_capability_is_refused():
    b = toy_bundle(weights=True, cond=True)
    with pytest.raises(SchemaError):
        dataclasses.replace(b, requires=frozenset({"plan", "cond_prompt"}))          # drops loss_weights
    with pytest.raises(SchemaError):
        dataclasses.replace(b, requires=frozenset({"plan", "loss_weights"}))         # drops cond_prompt


def test_a_serve_condition_naming_a_field_or_code_the_vocab_does_not_have_is_refused():
    b = toy_bundle(cond=True)
    with pytest.raises(SchemaError):
        dataclasses.replace(b, serve=(ServeCondition({"granularity": "1"}, None, 1.0),))   # not a vocab field
    with pytest.raises(SchemaError):
        dataclasses.replace(b, serve=(ServeCondition({"quality": "9"}, None, 1.0),))       # not a drawable code
    with pytest.raises(SchemaError):
        dataclasses.replace(toy_bundle(), serve=(ServeCondition({"quality": "5"}, None, 1.0),))  # no vocab at all


def test_check_serve_is_the_one_serve_rule():
    """The bundle and the compiler apply the same rule: declared fields, drawable codes, 0 allowed."""
    vocab = CondVocab(("quality",), {"quality": {q: f"\nQuality: {q}" for q in range(1, 6)}})
    check_serve(vocab, (BARE, ServeCondition({"quality": "0"}), ServeCondition({"quality": "5"}, {"quality": "0"}, 2.0)))
    check_serve(None, (BARE,))
    for bad in (ServeCondition({"speed": "1"}), ServeCondition({"quality": "9"}), ServeCondition({"quality": "high"}),
                ServeCondition({"quality": "5"}, {"speed": "0"}, 2.0), ServeCondition({"quality": "5"}, {"quality": "9"}, 2.0)):
        with pytest.raises(SchemaError):
            check_serve(vocab, (BARE, bad))
    with pytest.raises(SchemaError, match="no vocab"):
        check_serve(None, (ServeCondition({"quality": "5"}),))


def test_program_sha_must_be_the_canonical_sha256_of_program():
    b = toy_bundle()
    with pytest.raises(SchemaError):
        dataclasses.replace(b, program_sha="0" * 64)


def test_a_rowspace_with_an_unknown_roots_or_ranges_codec_is_refused():
    kwargs = dict(name="x", kind="corpus", n_rows=1, n_trainable=1, corpus_fingerprint=None, source="t")
    with pytest.raises(SchemaError):
        RowSpace(**kwargs, roots_codec="bogus")
    with pytest.raises(SchemaError):
        RowSpace(**kwargs, ranges_codec="bogus")


def test_a_legacy_file_path_stays_inside_the_export_root():
    LegacyFile("openpi-legacy", None, "index/x.json", "json", {"$o": []})
    for bad in ("/abs/x.json", "../x.json", "a/../../x.json"):
        with pytest.raises(SchemaError):
            LegacyFile("openpi-legacy", None, bad, "json", {"$o": []})
    with pytest.raises(SchemaError):
        LegacyFile("tf-legacy", None, "x.json", "json", {"$o": []})


def test_serve_conditions_say_whether_they_guide():
    assert not BARE.guided
    assert ServeCondition({"quality": "5"}, {}, 2.0).guided


@pytest.mark.parametrize("name", ["schema.py", "otree.py"])
def test_the_schema_modules_import_the_standard_library_only(name):
    """The core tier (the results format, the scorer) may import these types."""
    tree = ast.parse((REPO / "axis_qvla" / "signals" / name).read_text())
    mods = {a.name.split(".")[0] for n in tree.body if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module.split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom) and n.module and n.level == 0}
    # The package's own top-level name, read at runtime, so the check holds whatever it is called.
    assert mods <= set(sys.stdlib_module_names) | {Budget.__module__.split(".")[0]}, mods
