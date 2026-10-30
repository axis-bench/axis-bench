"""bind() end to end on bundles written by the signal stream's writer."""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest

from axis_qvla.data.layout import BENCH, CORPUS, default_dest, local_dest
from axis_qvla.train import ft_bundle
from axis_qvla.train.armspec import loads_armspec
from axis_qvla.train.bind import BindError, bind
from axis_qvla.train.recipes import load_recipe
from tests.unit.train._arms import H, arm_toml, host
from tests.unit.train._bundles import (FT_EPISODE, FT_FRAME, FT_RANGES, FT_ROOTS, FT_SPANS, finetune_corpus,
                                       finetune_digests, finetune_inputs, ft_rowspace, lerobot_rows_legacy,
                                       pin_bench, rowspace_ref, write_fixture_bundle)


def _bundles(paths):
    """The published bundles (what the signal compiler stages in the corpus release)."""
    return default_dest(CORPUS, paths, "bundles")


def _ft_arm(paths, **kw):
    """The default finetune arm, naming the norm statistics `finetune_inputs` wrote."""
    norm_sha = finetune_inputs(paths)
    return loads_armspec(arm_toml(**kw).replace(f"heldout24_qual_v2#sha256:{H}", f"heldout24_qual_v2#sha256:{norm_sha}"))


def _named_ft(paths, root=None, **kw):
    """The default finetune arm and a fitting flat bundle it names, written under `root` (the
    published bundles by default), naming this layout's roots index and ranges file."""
    rs = ft_rowspace()
    norm_sha = finetune_inputs(paths)
    bid = write_fixture_bundle(_bundles(paths) if root is None else root,
                               plan=(np.arange(13580 * 32) % 100).reshape(13580, 32), rowspace=rs,
                               inputs=finetune_digests(paths))
    text = arm_toml(bundle=bid, rowspace=rowspace_ref(rs), **kw)
    return loads_armspec(text.replace(f"heldout24_qual_v2#sha256:{H}", f"heldout24_qual_v2#sha256:{norm_sha}")), bid


def test_bind_writes_the_binding_of_a_pi05_arm(tmp_path):
    _, paths = host(tmp_path)
    spec, bid = _named_ft(paths)
    b = bind(spec, load_recipe(spec.preset), paths)
    assert (b.bundle, b.steps, b.global_batch, b.reducer, dict(b.signal_files)) == (bid, 13580, 32, "mean", {})
    assert b.bundle_dir == str(_bundles(paths) / bid[len("sha256:"):])
    doc = json.loads((paths.runs / spec.id / "binding.json").read_text())
    assert (doc["schema"], doc["bundle"], doc["smoke"]) == ("qvla.binding/1", bid, False)


def test_a_smoke_binding_never_touches_the_real_runs(tmp_path):
    _, paths = host(tmp_path)
    spec, bid = _named_ft(paths)
    bind(spec, load_recipe(spec.preset), paths)
    real = (paths.runs / spec.id / "binding.json").read_bytes()
    b = bind(spec, load_recipe(spec.preset), paths, smoke=True)
    assert b.steps == 20 and (paths.runs / spec.id / "binding.json").read_bytes() == real
    smoke = json.loads((paths.runs / spec.id / "rendered-smoke" / "binding.json").read_text())
    assert (smoke["steps"], smoke["smoke"], smoke["bundle"]) == (20, True, bid)


def test_bind_lists_every_problem_at_once(tmp_path):
    _, paths = host(tmp_path)
    bid = write_fixture_bundle(_bundles(paths), plan=(np.arange(100 * 64) % 100).reshape(100, 64),
                               rowspace=ft_rowspace(name="other"))
    spec = loads_armspec(arm_toml(bundle=bid))
    with pytest.raises(BindError) as e:
        bind(spec, load_recipe(spec.preset), paths)
    msg = str(e.value)
    for part in ("refused before launch", "frame index", "batch 64", "100 steps", "init:", "norm:", "rowspace:"):
        assert part in msg
    assert not (paths.runs / spec.id).exists()


def test_a_quality_tag_beside_a_named_bundle_is_refused(tmp_path):
    _, paths = host(tmp_path)
    spec, _ = _named_ft(paths, overrides='"data.quality_tag" = 5')
    with pytest.raises(BindError) as e:
        bind(spec, load_recipe(spec.preset), paths)
    assert str(e.value).splitlines()[1:] == [
        "  recipe: data.quality_tag is set but the run names compiled labels, whose codes are their own; drop the "
        "override or the labels"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads through a 000 directory")
def test_an_unreadable_published_bundles_tree_is_refused_not_passed_over(tmp_path):
    _, paths = host(tmp_path)
    spec, bid = _named_ft(paths, root=local_dest("bundles", ".", paths))     # a usable local copy
    _bundles(paths).mkdir(parents=True)
    _bundles(paths).chmod(0)
    try:
        with pytest.raises(BindError) as e:
            bind(spec, load_recipe(spec.preset), paths)
    finally:
        _bundles(paths).chmod(0o755)
    assert f"bundle: cannot read {_bundles(paths) / bid[len('sha256:'):]} (PermissionError)" in str(e.value)


def test_a_bundle_that_is_nowhere_is_listed_with_the_rest(tmp_path):
    _, paths = host(tmp_path)
    spec = loads_armspec(arm_toml())
    with pytest.raises(BindError) as e:
        bind(spec, load_recipe(spec.preset), paths)
    msg = str(e.value)
    assert f"bundle: cannot load sha256:{H} from {local_dest('bundles', H, paths)}" in msg
    assert "init:" in msg and "norm:" in msg


def test_a_new_methods_lerobot_export_carries_its_bundle_id(tmp_path):
    _, paths = host(tmp_path)
    default_dest(CORPUS, paths, "lerobot_merged").mkdir(parents=True)   # the published merged copy
    rs = ft_rowspace(name="libero5k_v2_d8")
    bid = write_fixture_bundle(_bundles(paths), plan=(np.arange(2000 * 64) % 100).reshape(2000, 64), rowspace=rs,
                               **lerobot_rows_legacy(np.arange(100)))
    spec = loads_armspec(arm_toml(id="smolvla.pretrain.mine", backbone="smolvla-lerobot", stage="pretrain",
                                  track="small", preset="smolvla.pretrain.small", rowspace=rowspace_ref(rs),
                                  init="base:smolvlm2_500m", norm=False, bundle=bid, seed=0))
    b = bind(spec, load_recipe(spec.preset), paths)
    assert list(b.signal_files) == ["axis_rows_path"]
    exported = paths.runs / spec.id / "signals/test" / f"{bid[7:19]}-lerobot_rows.npz"
    assert b.signal_files["axis_rows_path"] == str(exported)
    assert np.load(exported)["rows"].tolist() == list(range(100))
    assert not (paths.runs / spec.id / "signals/test/lerobot_rows.npz").exists()
    assert (b.per_rank_batch, b.world) == (16, 4)


def test_a_paper_arms_locked_file_keeps_its_name_and_a_smoke_exports_beside_its_render(tmp_path):
    _, paths = host(tmp_path)
    default_dest(CORPUS, paths, "lerobot_merged").mkdir(parents=True)
    rs = ft_rowspace(name="libero5k_v2_d8")
    bid = write_fixture_bundle(_bundles(paths), plan=(np.arange(20 * 64) % 100).reshape(20, 64), rowspace=rs,
                               **lerobot_rows_legacy(np.arange(100), path="signals/v1/lerobot/lerobot_rows.npz",
                                                     lock_id="lerobot.rows"))
    spec = loads_armspec(arm_toml(id="smolvla.pretrain.example", backbone="smolvla-lerobot", stage="pretrain",
                                  preset="smolvla.pretrain.paper", rowspace=rowspace_ref(rs),
                                  init="base:smolvlm2_500m", norm=False, bundle=bid))
    b = bind(spec, load_recipe(spec.preset), paths, smoke=True)
    smoke = paths.runs / spec.id / "rendered-smoke"
    assert dict(b.signal_files) == {"axis_rows_path": str(smoke / "signals/v1/lerobot/lerobot_rows.npz")}
    assert (smoke / "binding.json").is_file() and not (paths.runs / spec.id / "binding.json").exists()
    assert not (paths.runs / spec.id / "signals").exists()
    with pytest.raises(BindError, match="20 steps, the run trains 20621"):   # a real run needs the budget
        bind(spec, load_recipe(spec.preset), paths)


def _lerobot_arm(paths, plan, **legacy):
    default_dest(CORPUS, paths, "lerobot_merged").mkdir(parents=True, exist_ok=True)
    rs = ft_rowspace(name="libero5k_v2_d8")
    bid = write_fixture_bundle(_bundles(paths), plan=plan, rowspace=rs, **legacy)
    return loads_armspec(arm_toml(id="smolvla.pretrain.mine", backbone="smolvla-lerobot", stage="pretrain",
                                  track="small", preset="smolvla.pretrain.small", rowspace=rowspace_ref(rs),
                                  init="base:smolvlm2_500m", norm=False, bundle=bid, seed=0))


def test_lerobot_export_refusals_are_listed_with_the_rest(tmp_path):
    from axis_qvla.signals.schema import LegacyFile

    _, paths = host(tmp_path)
    bad_batch = (np.arange(2000 * 32) % 100).reshape(2000, 32)
    spec = _lerobot_arm(paths, bad_batch)
    with pytest.raises(BindError) as e:
        bind(spec, load_recipe(spec.preset), paths)
    assert str(e.value).splitlines()[1:] == [
        "  plan: batch 32 != the run's global batch 64",
        "  export: the compiled labels carry no lerobot-legacy files, the only label files the pinned LeRobot "
        "fork reads; a composed method carries them where that fork can train it (axis_qvla.signals.lerobot_files)"]
    doc = {"$o": [["rows", {"$array": "legacy.rows"}]]}
    legacy = (LegacyFile("lerobot-legacy", None, "signals/test/lerobot_rows.npz", "npz", doc),
              LegacyFile("lerobot-legacy", None, "signals/test/ft_rows_all.npz", "npz", doc),
              LegacyFile("lerobot-legacy", None, "signals/test/mystery.npz", "npz", doc))
    spec = _lerobot_arm(paths, bad_batch, legacy=legacy, legacy_arrays={"rows": np.arange(100)})
    with pytest.raises(BindError) as e:
        bind(spec, load_recipe(spec.preset), paths)
    assert str(e.value).splitlines()[1:] == [
        "  plan: batch 32 != the run's global batch 64",
        "  export: signals/test/lerobot_rows.npz and signals/test/ft_rows_all.npz both go to axis_rows_path",
        "  export: 'mystery.npz' is not a legacy LeRobot label file the pinned LeRobot fork reads"]
    assert not (paths.runs / spec.id).exists()


def test_a_lerobot_bundle_without_lerobot_files_is_refused(tmp_path):
    _, paths = host(tmp_path)
    default_dest(CORPUS, paths, "lerobot_merged").mkdir(parents=True)
    rs = ft_rowspace(name="libero5k_v2_d8")
    bid = write_fixture_bundle(_bundles(paths), plan=(np.arange(2000 * 64) % 100).reshape(2000, 64), rowspace=rs)
    spec = loads_armspec(arm_toml(id="smolvla.pretrain.mine", backbone="smolvla-lerobot", stage="pretrain",
                                  track="small", preset="smolvla.pretrain.small", rowspace=rowspace_ref(rs),
                                  init="base:smolvlm2_500m", norm=False, bundle=bid))
    with pytest.raises(BindError, match="no lerobot-legacy files"):
        bind(spec, load_recipe(spec.preset), paths)
    assert not (paths.runs / spec.id / "binding.json").exists()


def test_a_pi05_finetune_without_a_bundle_compiles_its_plan(tmp_path):
    from axis_qvla.signals.bundle import load_bundle

    _, paths = host(tmp_path)
    spec = _ft_arm(paths, bundle=False, track="small", preset="pi05.finetune.lora.small",
                   rowspace=rowspace_ref(ft_rowspace(n_rows=53)), overrides='"data.quality_tag" = 5')
    finetune_corpus(paths)
    b = bind(spec, load_recipe(spec.preset), paths)
    local = local_dest("bundles", b.bundle[len("sha256:"):], paths)
    assert b.bundle_dir == str(local) and not (_bundles(paths) / b.bundle[len("sha256:"):]).exists()
    bundle = load_bundle(local)
    assert bundle.plan.shape == (1000, 32) and bundle.program["params"]["quality_tag"] == 5
    assert bundle.vocab.prompts == ("put block 0 away", "put block 1 away")
    # Every sample's code is the dropout rule on the row's (episode, frame) from the ranges-cut
    # table and the pass it was drawn in: 53 rows make one whole batch of 32 per pass.
    flat, cond = bundle.plan.reshape(-1)[:3200], bundle.cond.reshape(-1)[:3200]
    want = [0 if ft_bundle.dropped(FT_EPISODE[r], FT_FRAME[r], k // 32) else 5 for k, r in enumerate(flat.tolist())]
    assert cond.tolist() == want and set(want) == {0, 5}
    assert dict(bundle.inputs) == finetune_digests(paths)
    assert bundle.inputs["ranges"] == hashlib.sha256(json.dumps(FT_SPANS).encode()).hexdigest()
    assert bundle.program["params"]["row_decode"] == ft_bundle.FLAT_ROW_DECODE
    assert json.loads((paths.runs / spec.id / "binding.json").read_text())["bundle"] == b.bundle


def test_a_compiled_bundle_bound_again_after_its_ranges_change_is_refused(tmp_path):
    _, paths = host(tmp_path)
    spec = _ft_arm(paths, bundle=False, track="small", preset="pi05.finetune.lora.small",
                   rowspace=rowspace_ref(ft_rowspace(n_rows=53)), overrides='"data.quality_tag" = 5')
    finetune_corpus(paths)
    b = bind(spec, load_recipe(spec.preset), paths)
    named = loads_armspec(arm_toml(bundle=b.bundle, track="small", preset="pi05.finetune.lora.small",
                                   rowspace=spec.rowspace).replace(f"heldout24_qual_v2#sha256:{H}", spec.norm))
    assert bind(named, load_recipe(named.preset), paths).bundle == b.bundle
    ranges = default_dest(BENCH, paths, FT_RANGES)
    ranges.write_text(json.dumps({"task_900--0": [[3, 31]], "task_901--0": [[0, 10], [15, 30]]}))   # still 53 rows
    with pytest.raises(BindError) as e:
        bind(named, load_recipe(named.preset), paths)
    assert f"rowspace: {ranges} has sha256 {hashlib.sha256(ranges.read_bytes()).hexdigest()[:12]}" in str(e.value)


def test_a_compiled_finetune_is_not_written_when_another_check_refuses(tmp_path):
    _, paths = host(tmp_path)
    finetune_inputs(paths)
    rs = finetune_corpus(paths)
    spec = loads_armspec(arm_toml(bundle=False, track="small", preset="pi05.finetune.lora.small",
                                  rowspace=rowspace_ref(rs)))          # names a norm digest the file is not
    with pytest.raises(BindError) as e:
        bind(spec, load_recipe(spec.preset), paths)
    assert str(e.value).splitlines()[1].startswith("  norm:") and len(str(e.value).splitlines()) == 2
    assert not local_dest("bundles", ".", paths).exists() and not (paths.runs / spec.id).exists()


def test_a_finetune_over_a_corpus_row_space_is_not_compiled(tmp_path):
    import dataclasses

    from axis_qvla.signals.bundle import write_rowspace
    from axis_qvla.signals.testing import toy_rowspace

    _, paths = host(tmp_path)
    rs = dataclasses.replace(toy_rowspace(), name="heldout24_ft")
    write_rowspace(rs, _bundles(paths) / "rowspaces")
    spec = loads_armspec(arm_toml(bundle=False, rowspace=rowspace_ref(rs)))
    with pytest.raises(BindError, match="binding compiles a finetune's labels over a flat finetune frame index only"):
        bind(spec, load_recipe(spec.preset), paths)


def test_a_roots_index_whose_base_is_another_corpus_is_refused(tmp_path):
    _, paths = host(tmp_path)
    rs = finetune_corpus(paths)
    index = default_dest(BENCH, paths, FT_ROOTS)
    index.write_text(json.dumps({**json.loads(index.read_text()), "base": "../../elsewhere"}))
    pin_bench(paths)                                     # as published: the index pinned as it is
    spec = loads_armspec(arm_toml(bundle=False, rowspace=rowspace_ref(rs)))
    with pytest.raises(BindError, match="records base '../../elsewhere'"):
        bind(spec, load_recipe(spec.preset), paths)


def test_a_finetune_corpus_that_is_not_its_row_space_is_listed_with_the_rest(tmp_path):
    _, paths = host(tmp_path)
    rs = finetune_corpus(paths, n_rows=60)
    spec = loads_armspec(arm_toml(bundle=False, rowspace=rowspace_ref(rs)))
    with pytest.raises(BindError) as e:
        bind(spec, load_recipe(spec.preset), paths)
    msg = str(e.value)
    assert "cannot compile the finetune's labels" in msg and "has 53 rows" in msg and "init:" in msg
    assert not local_dest("bundles", ".", paths).exists()


def test_only_a_pi05_finetune_may_name_no_bundle(tmp_path):
    _, paths = host(tmp_path)
    spec = loads_armspec(arm_toml(id="pi05.pretrain.x", stage="pretrain", preset="pi05.pretrain.full.paper",
                                  rowspace="libero5k_v2_d8", init="base:pi05_droid", bundle=False))
    with pytest.raises(BindError, match="bundle: the run names no compiled labels"):
        bind(spec, load_recipe(spec.preset), paths)


def test_a_locally_compiled_bundle_is_found_outside_the_published_tree(tmp_path):
    _, paths = host(tmp_path)
    spec, bid = _named_ft(paths, root=local_dest("bundles", ".", paths))
    b = bind(spec, load_recipe(spec.preset), paths)
    assert b.bundle_dir == str(local_dest("bundles", bid[len("sha256:"):], paths))


def _compiling_arm(paths):
    spec = _ft_arm(paths, bundle=False, track="small", preset="pi05.finetune.lora.small",
                   rowspace=rowspace_ref(ft_rowspace(n_rows=53)), overrides='"data.quality_tag" = 5')
    finetune_corpus(paths)
    return spec


def _episodes(paths, task=900):
    return default_dest(BENCH, paths, f"ft_task_{task}/meta/episodes/file-000.parquet")


@pytest.mark.parametrize("edit", ["ranges", "roots", "episodes", "extra episodes file"])
def test_a_finetune_is_compiled_only_from_the_files_the_repository_pins(tmp_path, edit):
    """An edit in the local mirror that keeps the row count (spans shifted, episodes reordered or
    added) would compile the paper's plan over other frames; each file the compile reads is
    compared with its BENCH_FILES.json pin first, and nothing is compiled or written."""
    import pandas as pd

    _, paths = host(tmp_path)
    spec = _compiling_arm(paths)
    pins = pin_bench(paths)
    assert bind(spec, load_recipe(spec.preset), paths).bundle.startswith("sha256:")      # pinned files bind
    if edit == "ranges":
        changed = default_dest(BENCH, paths, FT_RANGES)
        changed.write_text(json.dumps({"task_900--0": [[3, 31]], "task_901--0": [[0, 10], [15, 30]]}))
    elif edit == "roots":
        changed = default_dest(BENCH, paths, FT_ROOTS)
        changed.write_text(changed.read_text() + "\n")
    elif edit == "episodes":
        changed = _episodes(paths)
        pd.DataFrame({"episode_index": [0], "dataset_from_index": [0], "length": [41]}).to_parquet(changed)
    else:
        changed = _episodes(paths).with_name("file-001.parquet")
        pd.DataFrame({"episode_index": [1], "dataset_from_index": [40], "length": [5]}).to_parquet(changed)
    for d in (local_dest("bundles", ".", paths), paths.runs / spec.id):
        if d.exists():
            import shutil

            shutil.rmtree(d)
    with pytest.raises(BindError) as e:
        bind(spec, load_recipe(spec.preset), paths)
    msg = str(e.value)
    assert "cannot compile the finetune's labels from files that are not the repository's" in msg
    if edit == "extra episodes file":
        assert f"holds 1 file(s) {pins} does not list (first: file-001.parquet)" in msg
    else:
        got = hashlib.sha256(changed.read_bytes()).hexdigest()
        assert f"{changed} has sha256 {got[:12]}, but {pins} pins" in msg
    assert not local_dest("bundles", ".", paths).exists() and not (paths.runs / spec.id).exists()


def test_a_finetune_over_a_bench_mirror_without_its_file_list_is_not_compiled(tmp_path):
    """A partial fetch that left BENCH_FILES.json out is refused, naming the list and how to fetch
    it; nothing is compiled or written."""
    _, paths = host(tmp_path)
    spec = _compiling_arm(paths)
    pins = default_dest(BENCH, paths, "BENCH_FILES.json")
    pins.unlink()
    with pytest.raises(BindError) as e:
        bind(spec, load_recipe(spec.preset), paths)
    assert (f"cannot compile the finetune's labels from files that are not the repository's: rowspace: {pins} is "
            f"not there" in str(e.value)) and "`qvla data get bench --include BENCH_FILES.json`" in str(e.value)
    assert not local_dest("bundles", ".", paths).exists() and not (paths.runs / spec.id).exists()


def test_a_resume_is_bound_without_writing_its_record(tmp_path):
    _, paths = host(tmp_path)
    spec, bid = _named_ft(paths)
    b = bind(spec, load_recipe(spec.preset), paths, record=False)
    assert b.bundle == bid and not (paths.runs / spec.id).exists()


def test_a_lerobot_arms_override_problems_are_listed_with_the_rest(tmp_path):
    """Binding knows which signal files the bundle exports, so it refuses there (before writing
    anything) an override of a field qvla train writes, and per-frame weights without the
    axis_flat weighter."""
    _, paths = host(tmp_path)
    plan = (np.arange(2000 * 64) % 100).reshape(2000, 64)
    legacy = lerobot_rows_legacy(np.arange(100))
    spec = _lerobot_arm(paths, plan, **legacy)
    over = loads_armspec(arm_toml(id=spec.id, backbone=spec.backbone, stage=spec.stage, track="small",
                                  preset=spec.preset, rowspace=spec.rowspace, init=spec.init, norm=False,
                                  bundle=spec.bundle, seed=spec.seed,
                                  overrides='axis_rows_path = "my/own_rows.npy"\nseed = 9'))
    with pytest.raises(BindError) as e:
        bind(over, load_recipe(over.preset), paths)
    assert str(e.value).splitlines()[1:] == [
        "  recipe: the override 'axis_rows_path' sets axis_rows_path, which binding writes from the compiled "
        "labels' exported label files; drop the override",
        "  recipe: the override 'seed' sets seed, which qvla train sets from the run spec and this machine; "
        "drop the override"]
    assert not (paths.runs / spec.id).exists()
    from axis_qvla.signals.schema import LegacyFile

    weights = (LegacyFile("lerobot-legacy", None, "signals/test/lerobot_awr_flat_w.npz", "npz",
                          {"$o": [["weights", {"$array": "legacy.weights"}]]}),)
    spec = _lerobot_arm(paths, plan, legacy=legacy["legacy"] + weights,
                        legacy_arrays={**legacy["legacy_arrays"], "weights": np.ones(100, np.float32)})
    with pytest.raises(BindError, match="applies only with sample_weighting.type = 'axis_flat'"):
        bind(spec, load_recipe(spec.preset), paths)
    assert not (paths.runs / spec.id).exists()


def test_an_upstream_weighter_over_a_rows_only_bundle_is_refused(tmp_path):
    """`rabc` builds per-frame weights from a progress file of its own, which binding never loads:
    over a bundle that exports only rows the run would train weighted while binding.json names the
    bundle as its signal."""
    _, paths = host(tmp_path)
    plan = (np.arange(2000 * 64) % 100).reshape(2000, 64)
    spec = _lerobot_arm(paths, plan, **lerobot_rows_legacy(np.arange(100)))
    over = loads_armspec(arm_toml(id=spec.id, backbone=spec.backbone, stage=spec.stage, track="small",
                                  preset=spec.preset, rowspace=spec.rowspace, init=spec.init, norm=False,
                                  bundle=spec.bundle, seed=spec.seed,
                                  overrides='"sample_weighting.type" = "rabc"\n'
                                            '"sample_weighting.progress_path" = "/anywhere/progress.parquet"'))
    with pytest.raises(BindError) as e:
        bind(over, load_recipe(over.preset), paths)
    assert str(e.value).splitlines()[1:] == [
        "  recipe: the config sets sample_weighting (type 'rabc') but the compiled labels export no per-frame "
        "weights file; a run's loss weights come only from its compiled labels, so drop the sample_weighting "
        "overrides"]
    assert not (paths.runs / spec.id).exists()
    assert bind(spec, load_recipe(spec.preset), paths).signal_files.keys() == {"axis_rows_path"}
