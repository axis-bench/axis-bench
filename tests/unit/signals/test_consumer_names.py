"""The names other streams import from the signals package keep their names and signatures.

"Kept" means: the name exists; every recorded parameter is still there, in order, with the same kind
and default; and any parameter added since has a default. An exception type a consumer catches
keeps its module, name and base class. Bytes may change (the methods are refactored); a call that
worked keeps working.
"""
from __future__ import annotations

import importlib
import inspect

import pytest

P, K = "POSITIONAL_OR_KEYWORD", "KEYWORD_ONLY"
CALLABLES = {
    ("axis_qvla.signals.methods.schedule", "main"): [("argv", P, "None")],
    ("axis_qvla.signals.methods.schedule", "_ordered_roots"): [("roots_path", P, None)],
    ("axis_qvla.signals.methods.schedule", "_episode_offsets"): [("root", P, None)],
    ("axis_qvla.signals.methods.schedule", "episode_stats_from_weights"): [
        ("weights_path", P, None), ("tau", K, "10.0"), ("cap", K, "2.0"), ("horizon", K, "10")],
    ("axis_qvla.signals.methods.schedule", "_weights_header"): [("weights_path", P, None)],
    ("axis_qvla.signals.methods.schedule", "episode_rows"): [
        ("roots_path", P, None), ("ranges_path", P, None), ("stats", P, None), ("drop", K, None)],
    ("axis_qvla.signals.methods.schedule", "_unique_tmp_path"): [("p", P, None)],
    ("axis_qvla.signals.methods.schedule", "_frames_in_spans"): [("spans", P, None), ("length", P, None)],
    ("axis_qvla.signals.methods.schedule", "_file_digest"): [("path", P, None)],
    ("axis_qvla.signals.methods.awr_episode", "main"): [("argv", P, "None")],
    ("axis_qvla.signals.methods.cfg_tags", "_generator_digest"): [],
    ("axis_qvla.signals.methods.cfg_tags", "corpus_prompts"): [("roots_path", P, None)],
    ("axis_qvla.signals.methods.cfg_tags", "episode_task_ids"): [
        ("roots_path", P, None), ("ranges_path", P, None), ("stats", P, None), ("block_lengths", K, None)],
    ("axis_qvla.signals.methods.cfg_tags", "check_artifact_filename"): [
        ("path", P, None), ("reward_id", P, None), ("granularity", P, "'episode'")],
    ("axis_qvla.signals.methods.cfg_tags", "quality_meta"): [
        ("tag", P, None), ("prompts", P, None), ("reward_id", K, None), ("seed", K, None), ("horizon", K, None),
        ("n_rows", K, None), ("config_hash", K, None), ("episode_rows", K, None), ("q_ep", K, None),
        ("task_ids", K, None), ("dropout_meta", K, None), ("granularity", K, "'episode'"), ("row_bins", K, "None"),
        ("plateau", K, "'rank'")],
    ("axis_qvla.signals.methods.awr_chunk", "_neutralised_mask_clamped"): [("valid", P, None), ("t0", P, None),
                                                                      ("horizon", P, None)],
    ("axis_qvla.signals.methods.awr_chunk", "phase_delta_clamped"): [("curve", P, None), ("t0", P, None),
                                                                ("horizon", P, None)],
    ("axis_qvla.signals.blocks.awr", "episode_key"): [("task_id", P, None), ("episode_index", P, None)],
    ("axis_qvla.signals.provenance", "corpus_fingerprint"): [("corpus", P, None)],
}
VALUES = {("axis_qvla.signals.methods.schedule", "CORPUS_ROOT_ENV"): "AXIS_PRETRAIN_CORPUS_ROOT",
          ("axis_qvla.signals.methods.awr_chunk", "CORPUS_ROOT_ENV"): "AXIS_PRETRAIN_CORPUS_ROOT"}
# Exception types a consumer catches by type: the class keeps its module, name and base.
EXCEPTIONS = {("axis_qvla.signals.blocks.tags", "ConstantQualityError"): ValueError}


def _params(fn):
    return [(p.name, p.kind.name, None if p.default is p.empty else repr(p.default))
            for p in inspect.signature(fn).parameters.values()]


@pytest.mark.parametrize("module,name", sorted(CALLABLES))
def test_a_consumed_function_keeps_its_signature(module, name):
    now = _params(getattr(importlib.import_module(module), name))
    recorded = CALLABLES[(module, name)]
    assert now[:len(recorded)] == recorded
    assert all(default is not None or kind == "VAR_KEYWORD" for _n, kind, default in now[len(recorded):])


@pytest.mark.parametrize("module,name", sorted(VALUES))
def test_a_consumed_constant_keeps_its_value(module, name):
    assert getattr(importlib.import_module(module), name) == VALUES[(module, name)]


@pytest.mark.parametrize("module,name", sorted(EXCEPTIONS))
def test_a_consumed_exception_keeps_its_module_and_base(module, name):
    cls = getattr(importlib.import_module(module), name)
    assert isinstance(cls, type) and issubclass(cls, EXCEPTIONS[(module, name)])
    assert (cls.__module__, cls.__qualname__) == (module, name)


def test_the_bundle_module_carries_the_types_a_bundle_is_built_from():
    from axis_qvla.signals import bundle, schema

    for name in ("SignalBundle", "Budget", "RowSpace", "CondVocab", "ServeCondition", "LegacyFile"):
        assert getattr(bundle, name) is getattr(schema, name), name
