"""The smoke fixture: its signals, its three arms on one GPU each, and its build.

The build tests write a two-task corpus of short episodes (tasks 2013 and 2232, in the raw 9-D
layout) under tmp_path, so they need no published data; the fixture the release checks is the same
build over the real corpus tasks.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib

import numpy as np
import pytest

from axis_qvla.cli import main
from axis_qvla.signals.bundle import export_legacy, load_bundle, rowspace_id
from axis_qvla.signals.rowspace import trainable_rows
from axis_qvla.train import ids, plain_copy, smoke
from axis_qvla.train.armspec import dumps_armspec, load_armspec, loads_armspec
from axis_qvla.train.generate import openpi_config, render
from axis_qvla.train.lerobot_cfg import lerobot_config
from axis_qvla.train.recipes import load_recipe, load_rowspaces
from tests.unit.train._arms import H, host
from tests.unit.train._bundles import smoke_shaped_rowspace

LENGTHS = {2013: (40, 36, 44), 2232: (38, 42, 30)}


def _task(root: pathlib.Path, tid: int, lengths, *, width: int = 9) -> None:
    """One LeRobot v3 task directory: meta (info, tasks, episodes), one data file, one video file.
    The joints random-walk (never idle); the two finger widths open and close; the body poses' names
    are the task's own scene objects, as the corpus records them."""
    import pandas as pd

    rng = np.random.default_rng(tid)
    d, n = root / f"task_{tid}", sum(lengths)
    ep = np.repeat(np.arange(len(lengths)), lengths)
    frame = np.concatenate([np.arange(k) for k in lengths])
    joints = np.cumsum(rng.normal(0.0, 0.05, (n, 7)), axis=0)
    fingers = np.repeat(0.02 + 0.02 * np.sin(np.arange(n) / 5.0)[:, None], 2, axis=1)
    state = np.concatenate([joints, fingers], axis=1)[:, :width].astype(np.float32)
    (d / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (d / "data" / "chunk-000").mkdir(parents=True)
    (d / "videos" / "observation.images.wrist" / "chunk-000").mkdir(parents=True)
    (d / "videos" / "observation.images.wrist" / "chunk-000" / "file-000.mp4").write_bytes(b"\0")
    features = {k: {"dtype": "float32", "shape": [width], "names": [k]} for k in ("observation.state", "action")}
    features["observation.body_pose"] = {"dtype": "float32", "shape": [14], "names": ["franka/", f"plate_{tid}"]}
    (d / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v3.0", "fps": 15, "total_frames": n,
                                                      "total_episodes": len(lengths), "features": features}))
    pd.DataFrame({"task_index": [0]}, index=pd.Index([f"pick up block {tid}"], name="task")).to_parquet(
        d / "meta" / "tasks.parquet")
    pd.DataFrame({"episode_index": np.arange(len(lengths)), "length": list(lengths),
                  "dataset_from_index": np.concatenate([[0], np.cumsum(lengths)[:-1]])}).to_parquet(
        d / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    pd.DataFrame({"observation.state": list(state), "action": list(state), "episode_index": ep,
                  "frame_index": frame, "index": np.arange(n), "task_index": np.zeros(n, dtype=np.int64)}).to_parquet(
        d / "data" / "chunk-000" / "file-000.parquet")


def _corpus(root: pathlib.Path, *, width: dict | None = None) -> pathlib.Path:
    for tid, lengths in LENGTHS.items():
        _task(root, tid, lengths, width=(width or {}).get(tid, 9))
    return root


def _build(tmp_path, capsys, episodes) -> tuple[pathlib.Path, object, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    hosts, paths = host(tmp_path)
    assert main(["train", "smoke-fixture", "--host", str(hosts), "--episodes", str(episodes)]) == 0
    return hosts, paths, capsys.readouterr().out


def test_weights_cover_exactly_the_trainable_rows():
    rows = np.arange(0, 609, 2)
    w = smoke.fixture_weights(609, rows)
    assert np.isfinite(w[rows]).all() and np.isnan(np.delete(w, rows)).all()
    assert 0.5 <= np.nanmin(w) and np.nanmax(w) <= 2.0


def test_the_tag_fraction_is_the_paper_marginal():
    c = smoke.fixture_cond()
    sd = math.sqrt(smoke.TAG_FRACTION * (1 - smoke.TAG_FRACTION) / c.size)
    assert abs(float((c > 0).mean()) - smoke.TAG_FRACTION) < 4 * sd and set(np.unique(c)) <= {0, 1, 2, 3, 4, 5}
    assert c.shape == (smoke.STEPS, smoke.BATCH, 1) and c.dtype == np.uint8


def test_the_bundles_plan_the_trainable_rows_and_export_for_lerobot(tmp_path):
    rs = smoke_shaped_rowspace()
    rows = trainable_rows(rs)
    b = smoke.fixture_bundles(rs)
    pi05, lerobot = b["pi05"], b["lerobot"]
    assert pi05.plan.shape == (20, 8) and set(pi05.plan.ravel()) <= set(rows.tolist())
    # openpi's row sampler: an epoch is its first whole batches of one permutation, so no row twice.
    first_epoch = pi05.plan[: rows.size // smoke.BATCH].ravel()
    assert len(set(first_epoch.tolist())) == first_epoch.size
    np.testing.assert_array_equal(pi05.plan, lerobot.plan)
    assert np.isfinite(pi05.weights[pi05.plan]).all()
    assert (pi05.reducer, lerobot.reducer) == ("global_ratio", "per_rank_ratio")
    assert pi05.requires == {"plan", "loss_weights", "cond_prompt"} and lerobot.cond is None
    paths = export_legacy(lerobot, "lerobot-legacy", tmp_path)
    assert sorted(p.name for p in paths) == ["lerobot_awr_flat_smoke.npz", "lerobot_rows.npz"]
    with np.load(tmp_path / "signals/smoke/lerobot_rows.npz") as z:
        assert z["rows"].tolist() == rows.tolist()
    with np.load(tmp_path / "signals/smoke/lerobot_awr_flat_smoke.npz") as z:
        np.testing.assert_array_equal(z["weights"], lerobot.weights)


def test_norm_stats_have_openpis_layout():
    s = smoke.norm_stats(np.random.default_rng(0).random((50, 8)), np.zeros((50, 8)))
    assert set(s["norm_stats"]) == {"state", "actions"}
    assert all(len(v) == 8 for v in s["norm_stats"]["state"].values())
    assert set(s["norm_stats"]["actions"]) == {"mean", "std", "q01", "q99"}
    json.dumps(s)


def test_the_fixture_row_space_is_this_machines_own_data(tmp_path):
    lay = load_rowspaces()[smoke.SMOKE_ROWSPACE]
    assert (lay.n_rows, lay.repo, lay.lerobot_copy, lay.action_source) == (609, "local", "plain", None)
    _, paths = host(tmp_path)
    where = ids.rowspace_dirs(paths, f"{smoke.SMOKE_ROWSPACE}#sha256:{H}")
    assert all(p.is_relative_to(ids.local_derived(paths, smoke.SMOKE_DIR)) for p in where.values())


def test_the_smoke_arms_render_to_20_steps_on_one_gpu_each(tmp_path):
    _, paths = host(tmp_path)
    arms = smoke.fixture_arms(f"{smoke.SMOKE_ROWSPACE}#sha256:{H}", {"pi05": f"sha256:{H}", "lerobot": f"sha256:{H}"},
                              f"{smoke.SMOKE_ROWSPACE}#sha256:{H}", 609)
    assert sorted(arms) == sorted(smoke.ARMS)
    assert all(loads_armspec(dumps_armspec(spec)) == spec for spec in arms.values())
    # The batch is the smoke track's, declared by the presets: no arm overrides a batch or a world.
    assert {(s.track, s.preset.rsplit(".", 1)[1]) for s in arms.values()} == {("smoke", "smoke")}
    assert not any(k == "batch_size" or k.startswith("parallelism") for s in arms.values() for k in s.overrides)
    pi = arms["pi05"]
    for fsdp in (None, 1):                                   # one device is the preset's, and the flag's
        c = openpi_config(pi, load_recipe(pi.preset), paths, smoke=True, fsdp_devices=fsdp)
        assert (c["num_train_steps"], c["batch_size"], c["fsdp_devices"], c["data"]["smoke"]) == (20, 8, 1, True)
    assert c["data"]["norm_stats_dir"] == str(ids.local_derived(paths, f"norm/{smoke.SMOKE_ROWSPACE}"))
    signal = {"axis_rows_path": "r.npz", "sample_weighting.extra_params.weights_path": "w.npz"}
    for name in ("smolvla", "xvla"):
        spec = arms[name]
        lc = lerobot_config(spec, load_recipe(spec.preset), paths, smoke=True, signal_files=signal)
        assert (lc["steps"], lc["save_freq"], lc["batch_size"], lc["parallelism"]["dp_replicate"]) == (20, 20, 8, 1)
        assert lc["axis_expected_frames"] == 609 and lc["sample_weighting"]["type"] == "axis_flat"
        assert lc["sample_weighting"]["extra_params"]["expected_frames"] == 609
        assert lc["dataset"]["root"] == str(ids.local_derived(paths, "smoke/lerobot_merged"))
        _, argv = render(spec, load_recipe(spec.preset), paths, smoke=True, signal_files=signal)
        assert "--nproc-per-node=1" in argv


def test_norm_stats_are_written_once(tmp_path):
    p = tmp_path / "norm" / "norm_stats.json"
    smoke.write_once(p, b"a")
    smoke.write_once(p, b"a")
    with pytest.raises(ValueError, match="never replaced"):
        smoke.write_once(p, b"b")
    assert p.read_bytes() == b"a" and sorted(x.name for x in p.parent.iterdir()) == ["norm_stats.json"]


def test_the_fixture_builds_and_its_three_arms_bind_on_one_gpu_each(tmp_path, capsys):
    hosts, paths, out = _build(tmp_path, capsys, _corpus(tmp_path / "episodes"))
    arms_dir = ids.local_derived(paths, "smoke/runs")
    assert out.splitlines() == [f"OK    {a}: {arms_dir / f'{a}.toml'}" for a in smoke.ARMS]
    assert sorted(p.name for p in paths.cache.iterdir()) == ["bundles", "derived"]
    assert sorted(p.name for p in ids.local_derived(paths, "").iterdir()) == ["norm", "smoke"]
    d8 = ids.local_derived(paths, "smoke/corpus_d8")
    for tid in smoke.SMOKE_TASKS:
        assert smoke.state_width(d8 / f"task_{tid}") == 8 and (d8 / f"task_{tid}" / "videos").is_symlink()
        features = json.loads((d8 / f"task_{tid}" / "meta" / "info.json").read_text())["features"]
        assert features["observation.body_pose"]["names"] is None                 # differed: cleared
        assert features["observation.state"]["names"] == ["observation.state"]   # equal: kept
    assert plain_copy.merge_problems([d8 / f"task_{tid}" for tid in smoke.SMOKE_TASKS]) == []
    specs = {a: load_armspec(arms_dir / f"{a}.toml") for a in smoke.ARMS}
    rs = load_bundle(ids.bundle_dirs(paths, specs["pi05"].bundle)[1]).rowspace
    assert rs.name == smoke.SMOKE_ROWSPACE and rs.n_rows == sum(map(sum, LENGTHS.values()))
    assert {s.rowspace for s in specs.values()} == {f"{rs.name}#sha256:{rowspace_id(rs)}"}
    assert ids.rowspace_record_dirs(paths, specs["pi05"].rowspace)[1].is_dir()
    norm = ids.norm_dir(paths, specs["pi05"].norm) / smoke.ASSET_ID / "norm_stats.json"
    assert specs["pi05"].norm == f"{smoke.SMOKE_ROWSPACE}#sha256:{hashlib.sha256(norm.read_bytes()).hexdigest()}"
    assert specs["smolvla"].bundle == specs["xvla"].bundle != specs["pi05"].bundle
    # What a run opens besides the fixture: the pi0.5 base weights and the merged LeRobot copy.
    pathlib.Path(ids.openpi_init_params(paths, "base:pi05_droid")).mkdir(parents=True)
    ids.local_derived(paths, "smoke/lerobot_merged").mkdir()
    for a, extra, world in (("pi05", ["--fsdp-devices", "1"], None), ("smolvla", [], 1), ("xvla", [], 1)):
        assert main(["train", "--render", str(arms_dir / f"{a}.toml"), "--host", str(hosts), "--smoke", *extra]) == 0
        printed = capsys.readouterr().out
        assert printed.startswith(f"OK    {specs[a].id} -> ") and "--config" in printed
        if world is not None:
            assert f"--nproc-per-node={world}" in printed
    cfg = json.loads((ids.rendered_dir(paths, "pi05.pretrain.smoke", smoke=True) / "openpi_config.json").read_text())
    assert cfg["fsdp_devices"] == 1 and cfg["num_train_steps"] == 20


def test_a_fixture_is_never_replaced_and_a_failed_build_leaves_nothing(tmp_path, capsys):
    hosts, paths, _ = _build(tmp_path, capsys, _corpus(tmp_path / "episodes"))
    arms = ids.local_derived(paths, "smoke/runs")
    before = {p.name: p.read_bytes() for p in arms.iterdir()}
    assert main(["train", "smoke-fixture", "--host", str(hosts), "--episodes", str(tmp_path / "episodes")]) == 2
    assert "already exists" in capsys.readouterr().out
    assert {p.name: p.read_bytes() for p in arms.iterdir()} == before
    other = tmp_path / "other"
    other.mkdir()
    other_hosts, other_paths = host(other)
    _corpus(other / "episodes", width={2232: 7})
    assert main(["train", "smoke-fixture", "--host", str(other_hosts), "--episodes", str(other / "episodes")]) == 2
    assert "7-D" in capsys.readouterr().out
    assert list(ids.local_derived(other_paths, "").iterdir()) == []      # no fixture, no build directory


def test_only_the_corpus_layout_is_converted(tmp_path, capsys):
    _, paths, _ = _build(tmp_path / "raw", capsys, _corpus(tmp_path / "episodes"))
    hosts, _ = host(tmp_path)
    converted = ids.local_derived(paths, "smoke/corpus_d8")     # a second conversion is refused
    assert main(["train", "smoke-fixture", "--host", str(hosts), "--episodes", str(converted)]) == 2
    assert "observation.state is 8-D" in capsys.readouterr().out
