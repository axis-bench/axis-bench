"""REALALIGN's four stage-1 runs, compiled by the shared compiler's built-ins over its own row space, carry
exactly the arrays the REALALIGN tools built: one plan for bc/conditioning/awr, the pure-conditioning tags
of the drawn rows, the episode AWR weights, and the drop-top schedule's rows."""
from __future__ import annotations

import json
import os
import shutil
import stat

import numpy as np
import pytest

from axis_qvla.realrobot.realign import bundles
from axis_qvla.realrobot.realign.cli import build
from axis_qvla.signals.blocks.tags import ConstantQualityError
from axis_qvla.signals.bundle import load_bundle
from axis_qvla.signals.imported import ImportedScores
from tests.unit.realrobot.realign.test_driver import _export_curves


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("realign_bundles")
    export_dir, amap, prompts = _export_curves(tmp)
    build(tmp / "out", export_curves=export_dir, alignment_map=amap, prompts=prompts)
    return tmp / "out"


@pytest.fixture(scope="module")
def compiled(built):
    return bundles.compile_realign(built)


def test_the_row_space_is_the_built_corpus(built):
    rows = bundles.realign_rowspace(built)
    gates = json.loads((built / "gates.json").read_text())
    assert rows.n_rows == gates["G3"]["n_rows"] and rows.n_trainable == gates["G3"]["trainable_rows"]
    stamp = json.loads((built / "awr_weights_phase.json").read_text())["provenance"]["corpus_fingerprint"]
    assert rows.corpus_fingerprint == stamp


def test_bc_conditioning_and_awr_draw_one_plan_and_filtering_draws_the_drop_top_rows(compiled, built):
    assert set(compiled) == set(bundles.RUNS)
    plan = np.asarray(compiled["bc"].plan)
    assert plan.shape == (20000, 64) and compiled["bc"].seed == bundles.PLAN_SEED
    for run in ("conditioning", "awr"):
        np.testing.assert_array_equal(np.asarray(compiled[run].plan), plan)
    np.testing.assert_array_equal(np.asarray(compiled["filtering"].plan), np.load(built / "schedule_drop_top.npz")["rows"])


def test_conditioning_codes_are_the_pure_conditioning_tags_of_the_drawn_rows(compiled, built):
    b = compiled["conditioning"]
    tag = np.load(built / "quality_phase.npz")["tag"]
    np.testing.assert_array_equal(np.asarray(b.cond)[..., 0], tag[np.asarray(b.plan)])
    assert set(np.unique(b.cond).tolist()) <= {1, 2, 3, 4, 5}                 # no dropped row: pure conditioning
    assert b.vocab.text["quality"][5] == "\nQuality: 5"
    assert [dict(s.fields) for s in b.serve] == [{}, {"quality": "5"}] and not any(s.guided for s in b.serve)


def test_awr_weights_are_the_episode_weights_on_every_trainable_row(compiled, built):
    b = compiled["awr"]
    assert b.reducer == "global_ratio" and b.weights.dtype == np.float32
    ep = json.loads((built / "awr_weights_phase_ep.json").read_text())["weights"]
    rows = bundles.realign_rowspace(built)
    from axis_qvla.signals.rowspace import trainable_blocks
    for key, global_rows, frames, _n in trainable_blocks(rows):
        np.testing.assert_array_equal(np.asarray(b.weights)[global_rows], np.asarray(ep[key], np.float32)[frames])
    trainable = np.zeros(rows.n_rows, bool)
    trainable[[r for _k, g, _f, _n in trainable_blocks(rows) for r in g]] = True
    assert np.isnan(np.asarray(b.weights)[~trainable]).all()


def test_a_weights_file_without_its_corpus_stamp_is_refused(built, tmp_path):
    doc = json.loads((built / "awr_weights_phase.json").read_text())
    doc.pop("provenance")
    (tmp_path / "w.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="provenance"):
        bundles.check_weights_header(ImportedScores.from_json(tmp_path / "w.json"), bundles.realign_rowspace(built))


def test_bundles_round_trip_and_do_not_depend_on_where_the_build_sits(built, tmp_path):
    ids = bundles.write_bundles(built, tmp_path / "b1")
    assert set(ids) == {f"realalign_{r}" for r in bundles.RUNS}
    assert json.loads((tmp_path / "b1" / bundles.MANIFEST_FILE).read_text())["runs"] == ids
    for local, bid in ids.items():
        np.testing.assert_array_equal(np.asarray(load_bundle(tmp_path / "b1" / bid).plan),
                                      np.asarray(bundles.compile_realign(built)[local.removeprefix("realalign_")].plan))
    moved = tmp_path / "elsewhere" / "out"
    shutil.copytree(built, moved, symlinks=True)
    assert bundles.write_bundles(moved, tmp_path / "b2") == ids


def _clear_write_bits(root):
    for p in [root, *root.rglob("*")]:
        os.chmod(p, stat.S_IMODE(os.stat(p).st_mode) & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _restore_write_bits(root):
    # chmod needs ownership, not the write bit, so this works even though the files are read-only.
    for p in [root, *root.rglob("*")]:
        os.chmod(p, os.stat(p).st_mode | stat.S_IWUSR)


def test_compile_realign_does_not_write_into_a_readonly_build_directory(built, tmp_path):
    """`compile_realign` (and so `write_bundles`) must gate a `build --out` directory read-only: it
    only reads `built`, it does not own it, and a build whose gates all pass must still compile when
    the directory is read-only (a shared or read-only-mounted copy).

    chmod does not stop a root-euid process from writing, so the test does not rely on a PermissionError
    actually being raised; it chmods anyway (the realistic failure mode for a non-root user) and then
    checks the one fact that is true regardless of euid: `gates.json` is byte-identical afterwards. The
    old code rewrote it every call, so this assertion alone would have caught the regression under root
    too.
    """
    ro = tmp_path / "readonly_build"
    shutil.copytree(built, ro)
    before = (ro / "gates.json").read_bytes()
    before_mtime = os.stat(ro / "gates.json").st_mtime_ns
    _clear_write_bits(ro)
    try:
        compiled = bundles.compile_realign(ro)
    finally:
        _restore_write_bits(ro)
    assert set(compiled) == set(bundles.RUNS)
    after = ro / "gates.json"
    assert after.read_bytes() == before, "compile_realign rewrote gates.json in a build directory it only reads"
    assert os.stat(after).st_mtime_ns == before_mtime, "gates.json was reopened for writing (mtime moved)"


def test_a_constant_quality_corpus_is_refused_by_type_with_a_clear_message(built, tmp_path):
    """Every episode tagged the same (the rank weights all equal) drives CfgTags' H(q) diagnostic to
    0/0, which the built-in refuses as `ConstantQualityError`, never silently. The compiler names the
    'conditioning' run in the message (the raw diagnostic text alone does not), and the refusal is still of
    the type a caller can recognise -- never downgraded to a generic ValueError or swallowed."""
    copy = tmp_path / "constant"
    shutil.copytree(built, copy)
    p = copy / "awr_weights_phase_rank.json"
    doc = json.loads(p.read_text())
    doc["weights"] = {k: [1.0] * len(v) for k, v in doc["weights"].items()}
    p.write_text(json.dumps(doc))
    # Every trainable row at the neutral bin, as for an all-unlabelled corpus, so the gates pass and the
    # refusal is the compiler's (the tag builder itself refuses such weights by the same type).
    from axis_qvla.signals.blocks.schedule import NEUTRAL_Q
    from axis_qvla.signals.blocks.tags import NOT_TRAINABLE
    q = dict(np.load(copy / "quality_phase.npz"))
    q["tag"] = np.where(q["tag"] == NOT_TRAINABLE, NOT_TRAINABLE, NEUTRAL_Q).astype(q["tag"].dtype)
    np.savez(copy / "quality_phase.npz", **q)
    with pytest.raises(ConstantQualityError, match="conditioning") as err:
        bundles.compile_realign(copy)
    assert "H(q)" in str(err.value)


def test_an_unrelated_valueerror_during_compile_is_not_mistaken_for_constant_quality(built, tmp_path, monkeypatch):
    """A ValueError that is not ConstantQualityError (even sharing no relation to it) propagates
    unchanged: the compiler's catch is typed, not a blanket `except ValueError`."""
    import axis_qvla.realrobot.realign.bundles as bundles_mod

    def _boom(*_a, **_k):
        raise ValueError("unrelated failure, nothing to do with quality entropy")

    monkeypatch.setattr(bundles_mod, "compile_program", _boom)
    with pytest.raises(ValueError, match="unrelated failure") as err:
        bundles.compile_realign(built)
    assert not isinstance(err.value, ConstantQualityError)
