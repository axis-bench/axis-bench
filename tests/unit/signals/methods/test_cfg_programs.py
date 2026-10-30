"""CFG tags compile to the live tag builder's arrays and metadata, as prompt codes on BC's plan."""
from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np
import pytest

from axis_qvla.signals.blocks.tags import ConstantQualityError, q_given_task_entropy_ratio
from axis_qvla.signals.bundle import export_legacy
from axis_qvla.signals.imported import ImportedScores
from axis_qvla.signals.legacy import LegacyTarget, tags_config_hash
from axis_qvla.signals.methods import cfg_tags
from axis_qvla.signals.methods import schedule as bis
from axis_qvla.signals.methods.bc import BC
from axis_qvla.signals.methods.cfg import CfgTags
from axis_qvla.signals.methods.cfg_tags import entropy_units
from axis_qvla.signals.numerics import EXTENDED, log2_extended
from axis_qvla.signals.rowspace import subset
from axis_qvla.signals.schema import NO_CODE, Budget
from tests.unit.signals.toy_corpus import WEIGHTS, write_toy_corpus

ZERO = "0" * 64


def _target(path="signals/toy/quality_v2.npz", fmt="openpi-legacy") -> LegacyTarget:
    return LegacyTarget(lock_id="toy.tags", path=path, kind="quality_tags", sha256="", format=fmt,
                        generator="tags@888d0d2", generator_sha256=ZERO, roots_form="d8", numerics=None,
                        tolerance=None, template_name=None)


def _sha(p) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load(p):
    with np.load(p, allow_pickle=False) as z:
        return z["tag"], z["prompts"].tolist(), json.loads(str(z["meta"]))


@pytest.fixture
def toy(tmp_path, monkeypatch):
    monkeypatch.delenv(bis.CORPUS_ROOT_ENV, raising=False)
    return write_toy_corpus(tmp_path)


@pytest.mark.parametrize("granularity,name", [("episode", "quality_v2.npz"), ("row", "quality_v2_seg.npz")])
def test_cfg_tags_are_the_live_builders_tags(toy, tmp_path, granularity, name):
    cli = tmp_path / name
    assert cfg_tags.main(["--weights", str(toy.weights), "--roots", str(toy.roots), "--ranges", str(toy.ranges),
                          "--out", str(cli), "--granularity", granularity]) == 0
    b = CfgTags(granularity, "openpi-rowsampler", exports=(_target(),)).compile(
        ImportedScores.from_json(toy.weights), toy.rs, Budget(4, 2), 42)
    [ours] = export_legacy(b, "openpi-legacy", tmp_path / "o")
    tag1, prompts1, meta1 = _load(ours)
    tag2, prompts2, meta2 = _load(cli)
    np.testing.assert_array_equal(tag1, tag2)
    assert prompts1 == prompts2 and list(meta1) == list(meta2)
    assert {k: v for k, v in meta1.items() if k not in ("generator_sha256", "config_hash")} == \
           {k: v for k, v in meta2.items() if k not in ("generator_sha256", "config_hash")}
    digests = dict(weights_sha256=_sha(toy.weights), roots_sha256=_sha(toy.roots), ranges_sha256=_sha(toy.ranges),
                   dialect="row-v2", seed=0, granularity=granularity, plateau="rank")
    assert meta1["config_hash"] == tags_config_hash(generator_sha256=ZERO, **digests)
    assert meta2["config_hash"] == tags_config_hash(generator_sha256=cfg_tags._generator_digest(), **digests)


@pytest.mark.parametrize("granularity,name", [("episode", "quality_v2.npz"), ("row", "quality_v2_seg.npz")])
def test_codes_are_the_tags_of_the_planned_rows_and_the_plan_is_bcs(toy, tmp_path, granularity, name):
    cli = tmp_path / name
    assert cfg_tags.main(["--weights", str(toy.weights), "--roots", str(toy.roots), "--ranges", str(toy.ranges),
                          "--out", str(cli), "--granularity", granularity]) == 0
    b = CfgTags(granularity, "openpi-rowsampler").compile(ImportedScores.from_json(toy.weights), toy.rs, Budget(4, 2), 42)
    bc = BC("openpi-rowsampler").compile(None, toy.rs, Budget(4, 2), 42)
    np.testing.assert_array_equal(b.plan, bc.plan)
    np.testing.assert_array_equal(b.cond[..., 0], _load(cli)[0][b.plan])
    assert b.cond.shape == (4, 2, 1) and b.vocab.render("pick up the bowl", [5]) == "pick up the bowl\nQuality: 5"
    assert [s.fields for s in b.serve] == [{}, {"quality": "5"}] and not any(s.guided for s in b.serve)


def test_pure_conditioning_tags_every_row(toy):
    b = CfgTags("episode", "openpi-rowsampler", drop_whole=0.0, drop_component=0.0).compile(
        ImportedScores.from_json(toy.weights), toy.rs, Budget(4, 2), 42)
    assert not (b.cond == NO_CODE).any() and "guidance" not in b.requires


def test_subset_arms_reuse_the_full_corpus_tags(toy):
    scores = ImportedScores.from_json(toy.weights)
    full = CfgTags("episode", "openpi-rowsampler", exports=(_target(),)).compile(scores, toy.rs, Budget(4, 2), 42)
    sub = subset(toy.rs, ["task_800--0", "task_1000--0"], name="toy.sub", ranges_path="index/sub.json",
                 ranges_codec="json-indent1", source="test")
    b = CfgTags("episode", "openpi-rowsampler", tag_rows=toy.rs).compile(scores, sub, Budget(3, 2), 42)
    assert set(b.plan.ravel().tolist()) <= {1, 2, 15, 16, 17, 21, 22}
    np.testing.assert_array_equal(b.cond[..., 0], full.legacy_arrays["tags"][b.plan])


def test_the_dispatcher_writes_tags_only_from_a_program_that_tags(toy):
    """A tag target reaches its builder through the one dispatcher: a program that tags no rows
    refuses it, and the tag program refuses a kind it does not produce."""
    from axis_qvla.signals.legacy import LegacyError

    scores = ImportedScores.from_json(toy.weights)
    with pytest.raises(LegacyError, match="tags no rows"):
        BC("openpi-rowsampler", exports=(_target(),)).compile(None, toy.rs, Budget(4, 2), 42)
    schedule = dataclasses.replace(_target(), kind="schedule")
    with pytest.raises(LegacyError, match="draws no schedule"):
        CfgTags("episode", "openpi-rowsampler", exports=(schedule,)).compile(scores, toy.rs, Budget(4, 2), 42)


def test_tags_over_another_corpus_are_refused(toy):
    other = dataclasses.replace(toy.rs, name="other", corpus_fingerprint="sha256:" + "1" * 32)
    with pytest.raises(ValueError, match="different pretraining data"):
        CfgTags("episode", "openpi-rowsampler", tag_rows=other).compile(
            ImportedScores.from_json(toy.weights), toy.rs, Budget(4, 2), 42)


def test_the_entropy_units_are_the_tagged_units_and_no_episodes_is_refused_by_name():
    blocks = [np.array([0, 1, 2]), np.array([5, 6])]
    q_ep, task_ids, row_bins = np.array([2, 4]), np.array([800, 1000]), [np.array([1, 1, 3]), np.array([5, 2])]
    q, t = entropy_units(blocks, q_ep, task_ids, granularity="row", row_bins=row_bins)
    assert q.tolist() == [1, 1, 3, 5, 2] and t.tolist() == [800, 800, 800, 1000, 1000]
    q, t = entropy_units(blocks, q_ep, task_ids)
    assert q.tolist() == [2, 4] and t.tolist() == [800, 1000]
    with pytest.raises(ValueError, match="2 episode row blocks, 1 bin blocks and 2 task ids"):
        entropy_units(blocks, q_ep, task_ids, granularity="row", row_bins=row_bins[:1])
    for granularity, bins in (("row", []), ("episode", None)):
        q, t = entropy_units([], np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), granularity=granularity,
                             row_bins=bins)
        assert q.size == t.size == 0
        with pytest.raises(ValueError, match="no episodes"):
            q_given_task_entropy_ratio(q, t, log2=log2_extended)


# A platform-capability skip: where long double is plain double (macOS arm64, Windows) the export of
# a log2-extended tag file is refused by design (test_a_tag_file_is_refused_where_its_declared_log2_
# cannot_be_evaluated), so there is no ratio to compare. log2-libm is evaluated the same way everywhere.
NUMERICS = [
    pytest.param("log2-extended", marks=pytest.mark.skipif(
        not EXTENDED, reason="long double is plain double here, so a log2-extended tag export is refused")),
    "log2-libm",
]


@pytest.mark.parametrize("numerics", NUMERICS)
@pytest.mark.parametrize("granularity,name", [("episode", "quality_v2.npz"), ("row", "quality_v2_seg.npz")])
def test_the_declared_log2_ratio_is_taken_over_the_live_builders_units(toy, tmp_path, granularity, name, numerics):
    """The export recomputes the ratio with the declared log2 over the units `quality_meta` uses:
    on the toy corpus every log2 agrees, so the ratio is the live builder's, and the episode and
    row ratios differ, so units of the other granularity would show."""
    cli = tmp_path / name
    assert cfg_tags.main(["--weights", str(toy.weights), "--roots", str(toy.roots), "--ranges", str(toy.ranges),
                          "--out", str(cli), "--granularity", granularity]) == 0
    target = dataclasses.replace(_target(), numerics=numerics)
    b = CfgTags(granularity, "openpi-rowsampler", exports=(target,)).compile(
        ImportedScores.from_json(toy.weights), toy.rs, Budget(4, 2), 42)
    [ours] = export_legacy(b, "openpi-legacy", tmp_path / "o")
    ratio = _load(ours)[2]["q_given_task_entropy_ratio"]
    assert ratio == _load(cli)[2]["q_given_task_entropy_ratio"] == b.diagnostics["q_given_task_entropy_ratio"]
    other = CfgTags("row" if granularity == "episode" else "episode", "openpi-rowsampler").compile(
        ImportedScores.from_json(toy.weights), toy.rs, Budget(4, 2), 42)
    assert other.diagnostics["q_given_task_entropy_ratio"] != ratio


@pytest.mark.parametrize("numerics,capability,match", [("log2-extended", "EXTENDED", "extended mantissa"),
                                                       ("tanh-f32-fma", None, "not a log2")])
def test_a_tag_file_is_refused_where_its_declared_log2_cannot_be_evaluated(toy, monkeypatch, numerics, capability,
                                                                          match):
    from axis_qvla.signals import exports
    from axis_qvla.signals.legacy import LegacyError

    if capability:
        monkeypatch.setattr(exports, capability, False)
    target = dataclasses.replace(_target(), numerics=numerics)
    with pytest.raises(LegacyError, match=match):
        CfgTags("row", "openpi-rowsampler", exports=(target,)).compile(
            ImportedScores.from_json(toy.weights), toy.rs, Budget(4, 2), 42)


def test_a_constant_quality_set_is_refused_by_type(tmp_path, monkeypatch):
    """Every episode's weights are 1.0, so every episode tags at the neutral bin: zero quality
    entropy, no conditioning signal. CfgTags refuses it with the typed error a consumer can catch."""
    monkeypatch.delenv(bis.CORPUS_ROOT_ENV, raising=False)
    flat = {"version": 1, "reward_id": "v2", "tau": 10.0, "cap": 2.0, "horizon": 10, "n_episodes": 4, "n_rows": 23,
            "weights": {k: [1.0] * len(v) for k, v in WEIGHTS.items()}}
    toy = write_toy_corpus(tmp_path, weights=flat)
    with pytest.raises(ConstantQualityError, match="H\\(q\\) is zero"):
        CfgTags("episode", "openpi-rowsampler").compile(ImportedScores.from_json(toy.weights), toy.rs, Budget(4, 2), 42)
