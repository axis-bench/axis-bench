"""The refusals made before any GPU time, on in-memory bundles (no GPU; files only where inputs are)."""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest

from axis_qvla.data.layout import BENCH, CORPUS, checkpoint_dest, default_dest, norm_stats_dest
from axis_qvla.signals.schema import LegacyFile, ServeCondition
from axis_qvla.train import ids
from axis_qvla.train.armspec import loads_armspec
from axis_qvla.train.bind import CUT_RECORD, BindError, check_bundle, check_inputs, legacy_field, run_shape
from axis_qvla.train.parity import SIGNALS_LOCK_PATH
from axis_qvla.train.recipes import load_recipe
from tests.unit.train._arms import H, arm_toml, host
from tests.unit.train._bundles import (FT_RANGES, FT_ROOTS, finetune_digests, ft_rowspace, make_bundle, pin_bench,
                                       rowspace_ref)


def _plan(steps=13580, batch=32):
    return (np.arange(steps * batch, dtype=np.int64) % 100).reshape(steps, batch)


def _spec(**kw):
    s = loads_armspec(arm_toml(**kw))
    return s, load_recipe(s.preset)


def _check(bundle, smoke=False, rowspace=None, **kw):
    """check_bundle for an arm that trains on the bundle's own row space unless `rowspace` says otherwise."""
    spec, recipe = _spec(rowspace=rowspace or rowspace_ref(bundle.rowspace), **kw)
    steps, gb, pr, w = run_shape(spec, recipe, smoke=smoke)
    return check_bundle(spec, bundle, steps=steps, global_batch=gb, per_rank=pr, world=w, smoke=smoke)


def test_a_fitting_bundle_passes():
    assert _check(make_bundle(plan=_plan())) == []


def test_a_smoke_run_trains_a_prefix_only_when_asked():
    assert _check(make_bundle(plan=_plan()), smoke=True) == []
    assert _check(make_bundle(plan=_plan(steps=13600)), smoke=False) == [
        "plan: 13600 steps but the run trains 13580; only a smoke run trains a prefix"]


def test_a_bundle_over_another_row_space_is_refused():
    other = rowspace_ref(ft_rowspace(name="other"))
    problems = _check(make_bundle(plan=_plan()), rowspace=other.replace("other#", "heldout24_ft#"))
    assert len(problems) == 1 and problems[0].startswith("rowspace: the labels were compiled over frame index heldout24_ft#")


TAG5 = np.full((13580, 32, 1), 5, np.uint8)


@pytest.mark.parametrize("kw, match", [
    (dict(requires={"plan", "flow_time"}), "capability"),
    (dict(weights=np.ones(100, np.float32), reducer="per_rank_ratio"), "global_ratio"),
    (dict(plan=_plan(batch=64)), "batch 64 != the run's global batch 32"),
    (dict(plan=_plan(steps=100)), "100 steps, the run trains 13580"),
    (dict(rowspace=ft_rowspace(n_rows=10)), "rows outside"),
    (dict(weights=np.full(100, np.nan, np.float32)), "finite weight"),
    (dict(cond=np.full((13580, 32, 1), 255, np.uint8)), "255"),
    (dict(cond=np.full((13580, 32, 1), 4, np.uint8)), "no vocabulary"),
    (dict(cond=np.full((13580, 32, 2), 5, np.uint8), fields=("quality", "mistake")), "conditions on one"),
    (dict(cond=np.full((13580, 32, 2), 5, np.uint8)), "expected uint8 [13580, 32, 1]"),
    (dict(cond=TAG5, serve=(ServeCondition({"quality": "5"}, {}, 1.5),)), "unconditional"),
])
def test_refusals(kw, match):
    kw.setdefault("plan", _plan())
    problems = _check(make_bundle(**kw))
    assert len(problems) == 1 and match in problems[0], problems


def test_pure_conditioning_without_guidance_is_accepted():
    """REALALIGN-style tags: every sample tagged, no dropout, served without guidance."""
    b = make_bundle(plan=_plan(), cond=np.full((13580, 32, 1), 3, np.uint8), codes=(3,),
                    serve=(ServeCondition({"quality": "3"}),))
    assert _check(b) == []


def test_guidance_with_an_unconditional_branch_is_accepted():
    cond = TAG5.copy()
    cond[::7, 3] = 0
    b = make_bundle(plan=_plan(), cond=cond, serve=(ServeCondition({"quality": "5"}, {}, 1.5),))
    assert _check(b) == []


def test_a_weighted_lerobot_bundle_reduces_per_rank():
    kw = dict(id="smolvla.pretrain.example", backbone="smolvla-lerobot", stage="pretrain",
              preset="smolvla.pretrain.paper", init="base:smolvlm2_500m", norm=False)
    w = np.ones(100, np.float32)
    assert _check(make_bundle(plan=_plan(20621, 64), weights=w, reducer="per_rank_ratio"), **kw) == []
    problems = _check(make_bundle(plan=_plan(20621, 64), weights=w, reducer="global_ratio"), **kw)
    assert len(problems) == 1 and "'per_rank_ratio'" in problems[0]
    problems = _check(make_bundle(plan=_plan(20621, 64), requires={"plan", "guidance"}), **kw)
    assert len(problems) == 1 and problems[0].startswith("capability: the compiled labels require ['guidance']")


def test_per_rank_times_world_must_be_the_global_batch():
    kw = dict(id="smolvla.pretrain.example", backbone="smolvla-lerobot", stage="pretrain",
              preset="smolvla.pretrain.paper", init="base:smolvlm2_500m", norm=False)
    b = make_bundle(plan=_plan(20621, 64))
    assert _check(b, **kw) == []
    assert _check(b, overrides="batch_size = 64\n\"parallelism.dp_replicate\" = 1", **kw) == []
    assert _check(b, overrides="batch_size = 32", **kw) == [
        "batch: per-rank batch 32 x world 4 != global batch 64"]


def test_run_shape_reads_the_overrides_the_generator_applies():
    spec, recipe = _spec(id="smolvla.pretrain.example", backbone="smolvla-lerobot", stage="pretrain",
                         preset="smolvla.pretrain.paper", init="base:smolvlm2_500m", norm=False,
                         overrides="steps = 500\nparallelism = { dp_replicate = 2 }\nbatch_size = 32")
    assert run_shape(spec, recipe, smoke=False) == (500, 64, 32, 2)
    assert run_shape(spec, recipe, smoke=True) == (20, 64, 32, 2)
    spec, recipe = _spec()
    assert run_shape(spec, recipe, smoke=False) == (13580, 32, 32, 1)


def test_legacy_records_map_to_their_fields():
    lock = json.loads(SIGNALS_LOCK_PATH.read_text())

    def rec(path, lock_id=None):
        return LegacyFile("lerobot-legacy", lock_id, path, "npz", {"$o": []})

    assert legacy_field(rec("signals/v1/approach_aware/chunk/tags.npz", "approach_aware.chunk.tags"), lock) == \
        "axis_quality_path"
    assert legacy_field(rec("signals/v1/lerobot/lerobot_rows_s1000.npz", "lerobot.rows_s1000"), lock) == "axis_rows_path"
    assert legacy_field(rec("x/lerobot_awr_flat_mine.npz"), lock) == "sample_weighting.extra_params.weights_path"
    assert legacy_field(rec("x/lerobot_schedule_mine.npz"), lock) == "axis_schedule_path"
    with pytest.raises(BindError):
        legacy_field(rec("signals/v1/approach_aware/chunk/awr_weights.json", "approach_aware.chunk.awr_weights"), lock)
    with pytest.raises(BindError):
        legacy_field(rec("x/weights.npz"), lock)
    with pytest.raises(BindError):
        legacy_field(rec("signals/v1/lerobot/lerobot_rows.npz", "lerobot.no_such_file"), lock)


def test_inputs_are_checked_by_content(tmp_path):
    _, paths = host(tmp_path)
    spec, _ = _spec()
    problems = check_inputs(spec, paths, make_bundle(plan=_plan()))
    assert any(p.startswith("init:") for p in problems) and any(p.startswith("norm:") for p in problems)
    params = checkpoint_dest("axisrobotics/qvla-v1-pi05:pretrain/example/20604", paths) / "params"
    params.mkdir(parents=True)
    norm = norm_stats_dest("heldout24_qual_v2", paths) / "Devon018" / "Franka-Datasets-v2" / "norm_stats.json"
    norm.parent.mkdir(parents=True)
    norm.write_text("{}")
    roots = default_dest(BENCH, paths, FT_ROOTS)
    roots.parent.mkdir(parents=True)
    roots.write_text("{}")
    default_dest(BENCH, paths, FT_RANGES).write_text("{}")
    flat = make_bundle(plan=_plan(), inputs=finetune_digests(paths))
    problems = check_inputs(spec, paths, flat)
    assert problems == [f"norm: {norm} has sha256 {hashlib.sha256(b'{}').hexdigest()[:12]}, the run names {H[:12]}"]
    run = paths.runs / "pi05.pretrain.base" / "train"
    run.mkdir(parents=True)
    (run / "train_summary.json").write_text(json.dumps({"declared_steps": 20605, "trained_steps": 20, "smoke": True}))
    spec_run, _ = _spec(init="run:pi05.pretrain.base", norm=False)
    assert check_inputs(spec_run, paths, flat) == ["init: run:pi05.pretrain.base trained 20 of 20605 steps"]
    (run / "train_summary.json").write_text(json.dumps({"declared_steps": 20605, "trained_steps": 20605,
                                                        "smoke": False}))
    assert check_inputs(spec_run, paths, flat) == [
        f"init: run:pi05.pretrain.base: {run / 'final' / 'params'} is not there"]
    (run / "final" / "params").mkdir(parents=True)
    assert check_inputs(spec_run, paths, flat) == [                   # the norm stats the fine-tune reads
        f"init: run:pi05.pretrain.base: {run / 'final' / 'assets'} is not there"]
    (run / "final" / "assets").mkdir()
    assert check_inputs(spec_run, paths, flat) == []


@pytest.mark.parametrize("summary", [{}, {"smoke": False}, {"declared_steps": 20605, "smoke": False},
                                     {"declared_steps": 20605, "trained_steps": "20605", "smoke": False},
                                     {"declared_steps": True, "trained_steps": True, "smoke": False}])
def test_a_run_init_summary_without_integer_steps_is_refused(tmp_path, summary):
    """Missing fields are not a complete run: None == None proves nothing."""
    _, paths = host(tmp_path)
    run = paths.runs / "pi05.pretrain.base" / "train"
    (run / "final" / "params").mkdir(parents=True)
    (run / "train_summary.json").write_text(json.dumps(summary))
    spec, _ = _spec(init="run:pi05.pretrain.base", norm=False)
    problems = [p for p in check_inputs(spec, paths, None) if p.startswith("init:")]
    assert problems == [f"init: run:pi05.pretrain.base: {run / 'train_summary.json'} records no integer "
                        f"trained_steps and declared_steps"]


def test_a_lerobot_run_init_must_have_trained_its_recorded_steps(tmp_path):
    _, paths = host(tmp_path)
    default_dest(CORPUS, paths, "lerobot_merged").mkdir(parents=True)
    spec, _ = _spec(id="smolvla.finetune.example", backbone="smolvla-lerobot", stage="finetune",
                    preset="smolvla.finetune.paper", rowspace="libero5k_v2_d8", init="run:smolvla.pretrain.x",
                    norm=False)
    last = paths.runs / "smolvla.pretrain.x" / "train" / "checkpoints" / "last"
    step, cfg = last / "training_state" / "training_step.json", last / "pretrained_model" / "train_config.json"
    assert check_inputs(spec, paths, None) == [f"init: run:smolvla.pretrain.x: {step} is not there"]
    step.parent.mkdir(parents=True)
    cfg.parent.mkdir(parents=True)
    step.write_text(json.dumps({}))
    cfg.write_text(json.dumps({}))
    assert check_inputs(spec, paths, None) == [f"init: run:smolvla.pretrain.x: {step} records no integer step"]
    step.write_text(json.dumps({"step": 20}))
    assert check_inputs(spec, paths, None) == [f"init: run:smolvla.pretrain.x: {cfg} records no integer steps"]
    cfg.write_text("not json")
    assert check_inputs(spec, paths, None) == [f"init: run:smolvla.pretrain.x: {cfg} is not JSON"]
    cfg.write_text(json.dumps({"steps": 20621}))
    assert check_inputs(spec, paths, None) == ["init: run:smolvla.pretrain.x trained 20 of 20621 steps"]
    (last / "training_state" / "training_step.json").write_text(json.dumps({"step": 20621}))
    model = last / "pretrained_model" / "model.safetensors"
    assert check_inputs(spec, paths, None) == [f"init: run:smolvla.pretrain.x: {model} is not there"]
    model.mkdir()
    assert check_inputs(spec, paths, None) == [f"init: run:smolvla.pretrain.x: {model} is not a file"]
    model.rmdir()
    model.write_bytes(b"")
    assert check_inputs(spec, paths, None) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads through a 000 directory")
def test_an_unreadable_input_is_not_read_as_missing(tmp_path):
    """A permission error is reported as one, never as "is not there"."""
    _, paths = host(tmp_path)
    spec, _ = _spec(norm=False)
    ckpt = checkpoint_dest("axisrobotics/qvla-v1-pi05:pretrain/example/20604", paths)
    (ckpt / "params").mkdir(parents=True)
    ckpt.chmod(0)
    try:
        problems = check_inputs(spec, paths, None)
    finally:
        ckpt.chmod(0o755)
    assert f"init: {ckpt / 'params'} cannot be read (PermissionError) (fetch {spec.init} with `qvla data get`)" \
        in problems


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a mode-000 file")
def test_an_unreadable_run_summary_is_not_read_as_missing(tmp_path):
    _, paths = host(tmp_path)
    summary = paths.runs / "pi05.pretrain.base" / "train" / "train_summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({"declared_steps": 1, "trained_steps": 1, "smoke": False}))
    summary.chmod(0)
    try:
        problems = check_inputs(_spec(init="run:pi05.pretrain.base", norm=False)[0], paths, None)
    finally:
        summary.chmod(0o644)
    assert f"init: run:pi05.pretrain.base: {summary} cannot be read (PermissionError)" in problems


def _flat_layout(paths, ranges=b'{"task_900--0": [[0, 10]]}'):
    roots = default_dest(BENCH, paths, FT_ROOTS)
    roots.parent.mkdir(parents=True, exist_ok=True)
    roots.write_text('{"900": "ft_task_900"}')
    default_dest(BENCH, paths, FT_RANGES).write_bytes(ranges)
    return roots, default_dest(BENCH, paths, FT_RANGES)


def test_a_flat_bundle_cut_by_other_ranges_is_refused(tmp_path):
    """A flat pi0.5 row r is a frame of the ranges walk over the roots index, so a bundle compiled
    against another ranges file means other frames by the same row numbers."""
    _, paths = host(tmp_path)
    roots, ranges = _flat_layout(paths)
    spec, _ = _spec(norm=False)
    b = make_bundle(plan=_plan(), inputs=finetune_digests(paths))
    assert [p for p in check_inputs(spec, paths, b) if p.startswith("rowspace")] == []
    ranges.write_bytes(b'{"task_900--0": [[1, 11]]}')     # same row count, other frames
    got = hashlib.sha256(ranges.read_bytes()).hexdigest()
    problems = [p for p in check_inputs(spec, paths, b) if p.startswith("rowspace")]
    assert problems == [f"rowspace: {ranges} has sha256 {got[:12]}, the labels' flat rows were cut by the "
                        f"ranges {b.inputs['ranges'][:12]}"]
    roots.write_text('{"901": "ft_task_901"}')
    assert len([p for p in check_inputs(spec, paths, b) if p.startswith("rowspace")]) == 2


def test_a_flat_bundle_that_names_no_input_digests_is_refused(tmp_path):
    _, paths = host(tmp_path)
    _flat_layout(paths)
    spec, _ = _spec(norm=False)
    problems = [p for p in check_inputs(spec, paths, make_bundle(plan=_plan())) if p.startswith("rowspace")]
    assert len(problems) == 1 and "name no roots_index or ranges digest" in problems[0]
    partial = make_bundle(plan=_plan(), inputs={"roots_index": finetune_digests(paths)["roots_index"]})
    problems = [p for p in check_inputs(spec, paths, partial) if p.startswith("rowspace")]
    assert len(problems) == 1 and "name no ranges digest" in problems[0]


@pytest.mark.parametrize("params", [{"row_decode": "flat row r = concat row r"}, {}])
def test_a_flat_bundle_must_record_the_decode_rule_the_openpi_line_reads(tmp_path, params):
    from axis_qvla.train.ft_bundle import FLAT_ROW_DECODE

    _, paths = host(tmp_path)
    _flat_layout(paths)
    spec, _ = _spec(norm=False)
    b = make_bundle(plan=_plan(), inputs=finetune_digests(paths), params=params)
    assert [p for p in check_inputs(spec, paths, b) if p.startswith("rowspace")] == [
        f"rowspace: the compiled labels decode their flat rows as {params.get('row_decode')!r}; the pinned openpi "
        f"fork reads flat row r as {FLAT_ROW_DECODE!r}"]


def test_a_pretrain_roots_index_must_be_the_row_spaces_bytes(tmp_path):
    import dataclasses

    from axis_qvla.signals.rowspace import roots_bytes
    from axis_qvla.signals.testing import toy_rowspace

    _, paths = host(tmp_path)
    rs = dataclasses.replace(toy_rowspace(), name="libero5k_v2_d8")
    spec, _ = _spec(id="pi05.pretrain.example", stage="pretrain", preset="pi05.pretrain.full.paper",
                    rowspace=rowspace_ref(rs), init="base:pi05_droid", norm=False)
    roots = default_dest(CORPUS, paths, "index/roots_libero_5k_v2_d8.json")
    roots.parent.mkdir(parents=True)
    roots.write_bytes(roots_bytes(rs))
    b = make_bundle(plan=np.full((20605, 64), 1, np.int64), rowspace=rs)
    assert [p for p in check_inputs(spec, paths, b) if p.startswith("rowspace")] == [
        f"rowspace: the episodes directory {default_dest(CORPUS, paths, 'episodes')} is not there"]
    default_dest(CORPUS, paths, "episodes").mkdir()
    assert [p for p in check_inputs(spec, paths, b) if p.startswith("rowspace")] == []
    roots.write_bytes(roots_bytes(rs) + b"\n")
    problems = [p for p in check_inputs(spec, paths, b) if p.startswith("rowspace")]
    assert len(problems) == 1 and "the labels' frame index names" in problems[0]


def test_a_real_robot_arm_is_refused_before_anything_is_read(tmp_path):
    from axis_qvla.train.bind import bind

    _, paths = host(tmp_path)
    spec, recipe = _spec(id="realrobot.finetune.bc", track="realrobot", preset="pi05.finetune.lora.realrobot",
                         init="run:realrobot.pretrain.bc", rowspace="local:realalign_droid8d",
                         bundle="local:realalign.bc", norm=False)
    with pytest.raises(BindError, match="by a local: name"):
        bind(spec, recipe, paths)
    assert not paths.runs.exists()


def _ft_lerobot(paths):
    """A SmolVLA finetune arm over heldout24_ft (a "cut" copy), the bench's roots index, ranges file and
    task directory as published, and the copy's directory; returns (spec, copy dir, its full record)."""
    spec, _ = _spec(id="smolvla.finetune.example", backbone="smolvla-lerobot", stage="finetune",
                    preset="smolvla.finetune.paper", rowspace="heldout24_ft", init="base:smolvla_base", norm=False)
    rs = ids.rowspace_dirs(paths, spec.rowspace)
    rs["ranges"].parent.mkdir(parents=True, exist_ok=True)
    rs["ranges"].write_text('{"task_900--0": [[0, 10]]}')
    rs["roots"].write_text('{"900": "ft_demos/out_eef/task_900"}')
    info = rs["corpus_base"] / "ft_demos/out_eef/task_900/meta/info.json"
    info.parent.mkdir(parents=True)
    info.write_text('{"total_frames": 10}')
    rs["lerobot_dataset"].mkdir(parents=True)
    pin_bench(paths)
    record = {"ranges_sha256": hashlib.sha256(rs["ranges"].read_bytes()).hexdigest(), "action_source": "action",
              "inputs": [{"root": "task_900", "info_sha256": hashlib.sha256(info.read_bytes()).hexdigest()}],
              "episodes": []}
    return spec, rs["lerobot_dataset"], record


def _rowspace_problems(spec, paths):
    return [p for p in check_inputs(spec, paths, None) if p.startswith("rowspace:")]


def test_a_built_finetune_copy_binds_only_through_its_cut_record(tmp_path):
    """The frame count alone cannot tell a copy cut from other spans, inputs or action column:
    binding refuses a copy without its cut record and one whose record differs from the row space
    (each field: tests/unit/train/test_cut_copy.py), and binds the record the command writes."""
    _, paths = host(tmp_path)
    spec, copy, record = _ft_lerobot(paths)
    path = copy / CUT_RECORD
    assert _rowspace_problems(spec, paths) == [
        f"rowspace: {path} is not there: {copy} is not a copy cut by `axis_merge_datasets --ranges`; "
        "remove it and run `qvla train` again, which rebuilds it "
        "(docs/reference/data.md#the-merged-lerobot-copy)"]
    path.write_text(json.dumps({**record, "action_source": "action_eef"}))
    assert _rowspace_problems(spec, paths) == [
        f"rowspace: {copy} reads its action from 'action_eef', but frame index heldout24_ft trains on 'action'; "
        "rebuild the copy with --action-source action"]
    path.write_text(json.dumps(record))
    assert _rowspace_problems(spec, paths) == []


def test_a_published_lerobot_copy_needs_no_cut_record(tmp_path):
    """The pretraining copies are published, lock-verified repositories: no axis_spans.json is read."""
    _, paths = host(tmp_path)
    default_dest(CORPUS, paths, "lerobot_merged").mkdir(parents=True)
    spec, _ = _spec(id="smolvla.pretrain.example", backbone="smolvla-lerobot", stage="pretrain",
                    preset="smolvla.pretrain.paper", rowspace="libero5k_v2_d8", init="base:smolvlm2_500m", norm=False)
    assert _rowspace_problems(spec, paths) == []
