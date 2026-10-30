"""BC, drop-top-50 and random-50 compile to the rows and metadata the legacy builders write."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from axis_qvla.signals.bundle import bundle_id, export_legacy
from axis_qvla.signals.compile import compile_program
from axis_qvla.signals.imported import ImportedScores
from axis_qvla.signals.exports import SCHEDULE_KINDS, legacy_exports
from axis_qvla.signals.legacy import KIND_FORMATS, LegacyError, LegacyTarget, schedule_config_hash
from axis_qvla.signals.methods import random50
from axis_qvla.signals.methods import schedule as bis
from axis_qvla.signals.methods.bc import BC
from axis_qvla.signals.methods.filtering import DropTop50, Random50, TopFraction
from axis_qvla.signals.schema import Budget, LegacyFile
from axis_qvla.signals.testing import TOY_TRAINABLE
from tests.unit.signals.toy_corpus import write_toy_corpus

ZERO = "0" * 64


def _target(kind="schedule", path="signals/toy/schedule.npz", fmt="openpi-legacy", **kw) -> LegacyTarget:
    base = dict(lock_id=f"toy.{kind}", path=path, kind=kind, sha256="", format=fmt, generator="toy",
                generator_sha256=ZERO, roots_form="d8", numerics=None, tolerance=None, template_name=None)
    return LegacyTarget(**{**base, **kw})


def _sha(p) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _npz(p):
    with np.load(p, allow_pickle=False) as z:
        return z["rows"], json.loads(str(z["meta"]))


@pytest.fixture
def toy(tmp_path, monkeypatch):
    monkeypatch.delenv(bis.CORPUS_ROOT_ENV, raising=False)
    return write_toy_corpus(tmp_path)


def test_bc_writes_the_rowspace_index_files_and_row_plan(toy, tmp_path):
    exports = (_target("rowspace", toy.rs.roots_path), _target("rowspace", toy.rs.ranges_path),
               _target("lerobot_rows", "signals/toy/lerobot_rows.npz", fmt="lerobot-legacy"))
    b = BC("lerobot-axisrowsampler", exports=exports).compile(None, toy.rs, Budget(4, 2), 0)
    [roots, ranges] = export_legacy(b, "openpi-legacy", tmp_path / "o")
    assert roots.read_bytes() == toy.roots.read_bytes() and ranges.read_bytes() == toy.ranges.read_bytes()
    [rows] = export_legacy(b, "lerobot-legacy", tmp_path / "o")
    with np.load(rows) as z:
        assert z["rows"].tolist() == list(TOY_TRAINABLE)


def _assert_same_schedule(ours, theirs, *, config_hash):
    r1, m1 = _npz(ours)
    r2, m2 = _npz(theirs)
    np.testing.assert_array_equal(r1, r2)
    assert m1["generator_sha256"] == ZERO and m1["config_hash"] == config_hash
    assert list(m1) == list(m2)
    assert {k: v for k, v in m1.items() if k not in ("generator_sha256", "config_hash")} == \
           {k: v for k, v in m2.items() if k not in ("generator_sha256", "config_hash")}


def test_drop_top50_draws_the_schedule_builders_rows_and_meta(toy, tmp_path):
    cli = tmp_path / "cli.npz"
    bis.main(["--mode", "drop_top", "--weights", str(toy.weights), "--roots", str(toy.roots), "--ranges",
              str(toy.ranges), "--out", str(cli), "--total-steps", "6", "--batch", "2", "--seed", "0",
              "--keep-top-frac", "0.5"])
    scores = ImportedScores.from_json(toy.weights)
    b = DropTop50(exports=(_target(),)).compile(scores, toy.rs, Budget(6, 2), 0)
    [ours] = export_legacy(b, "openpi-legacy", tmp_path / "o")
    want_hash = schedule_config_hash(mode="drop_top", keep=0.5, steps=6, batch=2, seed=0, generator_sha256=ZERO,
                                     weights_sha256=_sha(toy.weights), roots_sha256=_sha(toy.roots),
                                     ranges_sha256=_sha(toy.ranges))
    _assert_same_schedule(ours, cli, config_hash=want_hash)


@pytest.mark.parametrize("granularity,mode,reward,seed", [("episode", "drop_top", "rand50_ep", 0),
                                                          ("chunk", "drop_random", "rand50_row", 42)])
def test_random50_draws_the_control_builders_rows_and_meta(toy, tmp_path, granularity, mode, reward, seed):
    out = tmp_path / "cli"
    random50.main(["--template", str(toy.weights), "--template-name", "awr_weights_v2.json", "--roots", str(toy.roots),
                   "--ranges", str(toy.ranges), "--out-dir", str(out), "--total-steps", "6", "--batch", "2"])
    tpl = ImportedScores.from_json(toy.weights)
    b = Random50(granularity, tpl, template_name="awr_weights_v2.json", exports=(_target(),)).compile(
        None, toy.rs, Budget(6, 2), seed)
    [ours] = export_legacy(b, "openpi-legacy", tmp_path / "o")
    synth = out / f"{reward}_weights.json"
    want_hash = schedule_config_hash(mode=mode, keep=0.5, steps=6, batch=2, seed=seed, generator_sha256=ZERO,
                                     weights_sha256=_sha(synth), roots_sha256=_sha(toy.roots),
                                     ranges_sha256=_sha(toy.ranges))
    _assert_same_schedule(ours, out / f"lerobot_schedule_{mode}_{reward}_smolvla.npz", config_hash=want_hash)


def test_drop_top50_compiles_its_declared_control(toy):
    tpl = ImportedScores.from_json(toy.weights)
    ctl = Random50("chunk", tpl, template_name="awr_weights_v2.json")
    b, controls = compile_program(DropTop50(control=ctl, control_seed=42), tpl, toy.rs, Budget(6, 2), 0)
    direct = compile_program(ctl, None, toy.rs, Budget(6, 2), 42)[0]
    assert b.controls == (bundle_id(direct),) == (bundle_id(controls[0]),)
    assert len(controls) == 1 and controls[0].program == ctl.spec()
    assert controls[0].diagnostics["keep_fraction"] == b.diagnostics["keep_fraction"]   # size-matched


def test_a_weights_file_for_another_corpus_is_refused(toy, tmp_path):
    p = tmp_path / "other.json"
    p.write_text(json.dumps({**json.loads(toy.weights.read_text()), "n_rows": 24}))
    with pytest.raises(ValueError, match="24 pretraining frames"):
        DropTop50().compile(ImportedScores.from_json(p), toy.rs, Budget(6, 2), 0)


def test_the_contract_call_compiles_a_drop_top50_with_no_control_and_no_exports(toy):
    """The call the real-robot line makes, as written: `DropTop50().compile(ImportedScores.from_json(w), rs,
    Budget(20000, 64), 0)` -- no control, no legacy export, the plan drawn over the kept rows only."""
    scores = ImportedScores.from_json(toy.weights)
    b = DropTop50().compile(scores, toy.rs, Budget(20000, 64), 0)
    kept = TopFraction().select(scores.values(), toy.rs)
    assert b.controls == () and b.legacy == () and b.plan.shape == (20000, 64)
    assert set(np.unique(b.plan).tolist()) == set(np.concatenate(kept.blocks).tolist())
    assert b.diagnostics["keep_fraction"] == kept.kept / toy.rs.n_trainable == 4 / 9
    assert b.inputs == {"w.json": scores.sha256}


def test_the_filtering_programs_refuse_what_they_cannot_draw_from(toy, tmp_path):
    with pytest.raises(ValueError, match="granularity"):
        Random50("row", ImportedScores.from_json(toy.weights))
    doc = json.loads(toy.weights.read_text())
    for name, bad, match in (("anonymous", {k: v for k, v in doc.items() if k != "reward_id"}, "names no reward_id"),
                             ("tau5", {**doc, "tau": 5.0}, "was built with tau=5.0")):
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match=match):
            DropTop50().compile(ImportedScores.from_json(p), toy.rs, Budget(6, 2), 0)


def test_one_dispatcher_picks_every_targets_builder(toy):
    """`legacy_exports` is the one place a target's kind picks its builder: the schedule kinds are the
    lock's, and a program that draws no schedule refuses a schedule target instead of mis-writing it."""
    assert set(SCHEDULE_KINDS) == {k for k in KIND_FORMATS if k.endswith("schedule")}
    roots = _target("rowspace", toy.rs.roots_path)
    assert legacy_exports((roots,), toy.rs) == (
        (LegacyFile("openpi-legacy", "toy.rowspace", toy.rs.roots_path, toy.rs.roots_codec, toy.rs.roots),), {})
    with pytest.raises(LegacyError, match="draws no schedule"):
        BC("openpi-rowsampler", exports=(_target(),)).compile(None, toy.rs, Budget(4, 2), 0)


def test_a_drop_top50_control_names_the_seed_it_compiles_with(toy):
    """The paper's chunk control compiles with seed 42; a default seed would build a different control
    under the same name, so a control without its seed is refused, and so is a seed without a control."""
    ctl = Random50("chunk", ImportedScores.from_json(toy.weights))
    with pytest.raises(ValueError, match="control_seed"):
        DropTop50(control=ctl)
    with pytest.raises(ValueError, match="no control"):
        DropTop50(control_seed=42)
    assert DropTop50().control_seed is None
    assert DropTop50(control=ctl, control_seed=0).spec()["params"]["control"]["seed"] == 0
