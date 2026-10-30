"""A refused scene asset ABORTS the evaluation; an ordinary simulation error is still a seed record.

Before this, rollout_dualsim/rollout_render caught every scene-build exception and returned
reason="error:build_render:..." -- so a mesh missing from --asset-dir (or one whose sha256 differs
from ASSETS.lock.json) became one error seed file per rollout, tallied as a task failure. A missing
eval input is not a property of the policy: the run must stop and name the file.

The dual-sim and render modules import robosuite through render_scene, which the offline venv does
not have, so render_scene is stubbed in sys.modules and the rollout modules are imported fresh
against the stub; everything else they import (MuJoCo, openpi_client) is real.
"""
from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest

import axis_qvla.bench.sim as sim_pkg
from axis_qvla.bench.sim.assets import EXIT_ASSET_REFUSED, AssetFetchRefused

_FRESH = ("axis_qvla.bench.sim.rollout_dualsim", "axis_qvla.bench.sim.rollout_render")


@pytest.fixture
def backends(monkeypatch):
    """(rollout_dualsim, rollout_render, stub) imported against a stub render_scene."""
    stub = types.ModuleType("axis_qvla.bench.sim.render_scene")
    stub.build_render_scene = lambda *a, **k: (_ for _ in ()).throw(AssertionError("set per test"))
    monkeypatch.setitem(sys.modules, "axis_qvla.bench.sim.render_scene", stub)
    saved = {n: sys.modules.pop(n, None) for n in _FRESH}
    try:
        dual = importlib.import_module("axis_qvla.bench.sim.rollout_dualsim")
        render = importlib.import_module("axis_qvla.bench.sim.rollout_render")
        yield dual, render, stub
    finally:
        for n in _FRESH:
            sys.modules.pop(n, None)
            if hasattr(sim_pkg, n.rsplit(".", 1)[1]):
                delattr(sim_pkg, n.rsplit(".", 1)[1])
            if saved[n] is not None:
                sys.modules[n] = saved[n]


def _kw(tmp_path: Path) -> dict:
    task_dir = tmp_path / "task_806"
    task_dir.mkdir(exist_ok=True)
    (task_dir / "task.json").write_text(json.dumps({"scene_variants": []}))
    return dict(init_state_seed=806000, task_id=806, task_name="t", prompt="p", open_w=0.08,
                closed_w=0.0, checker_factory=lambda s: (lambda m, d: False),
                render_root=tmp_path / "render", work_dir=tmp_path / "work",
                asset_dir=str(tmp_path / "assets"), asset_fetch="refuse")


def _raise(exc):
    def build(*a, **k):
        raise exc
    return build


class _Scene:
    model = data = initial_state = None
    variant_index = 0
    qpos_offset = None

    def close(self):
        pass


def test_dualsim_reraises_a_refused_asset_from_the_render_build(backends, tmp_path, monkeypatch):
    dual, _, stub = backends
    monkeypatch.setattr(dual, "build_render_scene", _raise(AssetFetchRefused("asset 'a/b.obj' missing")))
    with pytest.raises(AssetFetchRefused, match="a/b.obj"):
        dual.rollout_pi05_dualsim(tmp_path / "task_806", None, **_kw(tmp_path))


def test_dualsim_reraises_a_refused_asset_from_the_physics_build(backends, tmp_path, monkeypatch):
    dual, _, _ = backends
    monkeypatch.setattr(dual, "build_render_scene", lambda *a, **k: _Scene())
    monkeypatch.setattr(dual, "set_camera_near_plane", lambda model: None)
    monkeypatch.setattr(dual, "_build_physics_scene",
                        _raise(AssetFetchRefused("asset 'x.stl': sha256 differs from ASSETS.lock.json")))
    with pytest.raises(AssetFetchRefused, match="sha256"):
        dual.rollout_pi05_dualsim(tmp_path / "task_806", None, **_kw(tmp_path))


def test_dualsim_still_records_an_ordinary_build_error(backends, tmp_path, monkeypatch):
    dual, _, _ = backends
    monkeypatch.setattr(dual, "build_render_scene", _raise(RuntimeError("bad xml")))
    res = dual.rollout_pi05_dualsim(tmp_path / "task_806", None, **_kw(tmp_path))
    assert res.success is False and res.reason == "error:build_render:RuntimeError: bad xml"
    monkeypatch.setattr(dual, "build_render_scene", lambda *a, **k: _Scene())
    monkeypatch.setattr(dual, "set_camera_near_plane", lambda model: None)
    monkeypatch.setattr(dual, "_build_physics_scene", _raise(ValueError("no mjcf_xml")))
    res = dual.rollout_pi05_dualsim(tmp_path / "task_806", None, **_kw(tmp_path))
    assert res.reason == "error:build_physics:no mjcf_xml"


def test_render_reraises_a_refused_asset_and_records_other_errors(backends, tmp_path, monkeypatch):
    _, render, _ = backends
    monkeypatch.setattr(render, "build_render_scene", _raise(AssetFetchRefused("asset 'c.png' missing")))
    with pytest.raises(AssetFetchRefused, match="c.png"):
        render.rollout_pi05_render(tmp_path / "task_806", None, **_kw(tmp_path))
    monkeypatch.setattr(render, "build_render_scene", _raise(RuntimeError("boom")))
    res = render.rollout_pi05_render(tmp_path / "task_806", None, **_kw(tmp_path))
    assert res.success is False and res.reason == "error:build_scene:boom"


def test_the_abort_exit_code_is_distinct():
    from axis_qvla.bench.sim.layout_plan import EXIT_UNEVALUABLE

    assert EXIT_ASSET_REFUSED == 4 and EXIT_ASSET_REFUSED != EXIT_UNEVALUABLE


def test_the_campaign_driver_aborts_on_a_refused_asset(tmp_path):
    """A worker that saw exit 4 marks the queue aborted; the driver then returns 4 with the reason
    and writes no arm_summary.json (a partial arm must never read as a scored one)."""
    from axis_qvla.bench.sim import campaign
    from axis_qvla.bench.sim.workqueue import Unit, WorkQueue

    q = WorkQueue(tmp_path / "q")
    q.initialize([Unit(806, 806000)])
    lines: list[str] = []
    assert campaign.asset_abort(q, lines.append) is None and lines == []
    q.abort("gpu0s0w0: asset 'tabletopgen/a.obj' is not under /assets")
    assert campaign.asset_abort(q, lines.append) == EXIT_ASSET_REFUSED
    assert any("ABORTED" in ln and "tabletopgen/a.obj" in ln for ln in lines)


def test_a_previous_abort_is_cleared_when_the_driver_starts_again(tmp_path):
    from axis_qvla.bench.sim.workqueue import Unit, WorkQueue

    q = WorkQueue(tmp_path / "q")
    q.initialize([Unit(806, 806000)])
    q.abort("first reason")
    q.abort("second reason")             # the first abort wins
    assert q.aborted() == "first reason"
    assert q.clear_abort() == "first reason" and q.aborted() is None


def test_tally_only_refuses_an_aborted_queue(tmp_path, monkeypatch):
    """--tally-only never looked at the queue, so it wrote arm_summary.json --
    the sequencer's skip marker -- for an arm a refused asset had aborted part-way. It now returns 4,
    names the reason, writes no summary and leaves the marker for the next real run to clear."""
    from axis_qvla.bench.sim import campaign
    from axis_qvla.bench.sim.workqueue import Unit, WorkQueue

    monkeypatch.setattr(campaign, "python_versions", lambda python: {})
    monkeypatch.setattr(campaign, "gpu_info", lambda: {})
    lines: list[str] = []
    monkeypatch.setattr(campaign, "log", lines.append)
    ck = tmp_path / "ck"
    ck.mkdir()
    (ck / "w.bin").write_bytes(b"x")
    run_root = tmp_path / "r1"
    q = WorkQueue(run_root / "q_probe")
    q.initialize([Unit(806, 806000)])
    q.abort("gpu0s0w0: task 806: ABORT: scene asset refused for task 806 (probe): tabletopgen/a.obj")
    argv = ["--run", "probe", "--config", "cfg", "--checkpoint", str(ck), "--tasks", "806",
            "--run-root", str(run_root), "--log-root", str(tmp_path / "logs"),
            "--openpi-dir", str(tmp_path), "--n-seeds", "1", "--tally-only"]
    out = run_root / "out_probe"
    # One seed finished before the abort: --tally-only must not summarise it as a complete run.
    seed_dir = tmp_path / "logs" / campaign.result_dir_name("probe", "") / "task_806"
    seed_dir.mkdir(parents=True)
    (seed_dir / "seed_806000.json").write_text(json.dumps(
        {"seed": 806000, "success": True, "n_steps": 1, "sim_time": 0.1, "reason": "success"}))

    assert campaign.main(argv) == EXIT_ASSET_REFUSED
    assert not (out / "arm_summary.json").exists() and not (out / "arm_summary.FAILED.json").exists()
    assert any("ABORTED" in ln and "tabletopgen/a.obj" in ln for ln in lines)
    assert q.aborted() is not None                     # tally-only does not clear the marker

    q.clear_abort()
    assert campaign.main(argv) == 0                    # a clean queue tallies as before
    assert json.loads((out / "arm_summary.json").read_text())["successes"] == 1
