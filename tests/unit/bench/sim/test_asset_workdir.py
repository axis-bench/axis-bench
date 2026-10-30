"""A scene compiles only from bytes the asset store returned in THIS build.

axis_replay_native.materialize_scene (the pinned AXIS-Render cut, which this repo does not edit)
reads any XML or mesh already present in its work dir and asks the store only for what is absent
(asset_materializer.py:130-135 and :156-162). The eval work dir outlives a run
(<run_root>/work_<arm>/<worker>/task_<id>), so a stale or wrong mesh there would be compiled without
ever meeting --asset-dir or ASSETS.lock.json. axis-render's resolver also swallows every loader
exception (xml_dependency_resolver.py, `except Exception: continue`), so a refused XML include would
surface later as a MuJoCo compile error -- an error seed -- instead of an abort.

Both builders run here with the REAL materialiser, the real lock-checking store and a real MuJoCo
compile. render_scene imports robosuite at module scope, so its two robosuite modules are stubbed;
the stub env compiles the entry XML it is handed, which is exactly what the render shell reads.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
import types
from pathlib import Path

import mujoco
import numpy as np
import pytest

import axis_qvla.bench.sim as sim_pkg
from axis_qvla.bench.sim import asset_guard
from axis_qvla.bench.sim.assets import AssetFetchRefused

TETRA = b"v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nv 0 0 0.1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"   # 4 vertices
CUBE = (b"v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 0 0 1\nv 1 0 1\nv 1 1 1\nv 0 1 1\n"
        b"f 1 3 2\nf 1 4 3\nf 5 6 7\nf 5 7 8\nf 1 2 6\nf 1 6 5\nf 2 3 7\nf 2 7 6\n"
        b"f 3 4 8\nf 3 8 7\nf 4 1 5\nf 4 5 8\n")                                              # 8 vertices
ROBOT = b'<mujoco><worldbody><body name="robot"><geom type="sphere" size="0.05"/></body></worldbody></mujoco>'
MJCF = ('<mujoco model="t"><include file="../robots/r.xml"/>'
        '<asset><mesh name="m" file="../meshes/box.obj"/></asset>'
        '<worldbody><body name="obj"><freejoint/><geom type="mesh" mesh="m"/></body></worldbody></mujoco>')
MESH, INCLUDE = "meshes/box.obj", "robots/r.xml"


def _lock(tmp_path: Path, **files: bytes) -> Path:
    rows = [{"path": f"assets/{rel}", "size": len(b), "sha256": hashlib.sha256(b).hexdigest()}
            for rel, b in {MESH: TETRA, INCLUDE: ROBOT, **files}.items()]
    p = tmp_path / "ASSETS.lock.json"
    p.write_text(json.dumps({"schema": "qvla.assets_lock/1", "files": rows}))
    return p


def _asset_dir(tmp_path: Path, mesh: bytes | None = TETRA, include: bytes | None = ROBOT) -> Path:
    d = tmp_path / "assets"
    for rel, data in ((MESH, mesh), (INCLUDE, include)):
        if data is not None:
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / rel).write_bytes(data)
    d.mkdir(exist_ok=True)
    return d


def _seed_work_dir(work: Path, mesh: bytes) -> None:
    """What a previous run left: the mesh directly under the work dir AND in its scene dir --
    neither may reach the compile."""
    for root in (work, work / asset_guard.SCENE_DIR):
        (root / MESH).parent.mkdir(parents=True, exist_ok=True)
        (root / MESH).write_bytes(mesh)
        (root / INCLUDE).parent.mkdir(parents=True, exist_ok=True)
        (root / INCLUDE).write_bytes(ROBOT)


# --- the dual-sim physics twin ----------------------------------------------------------------

@pytest.fixture
def dual(monkeypatch):
    """rollout_dualsim imported against a stub render_scene (robosuite-free); compile is plain MuJoCo."""
    stub = types.ModuleType("axis_qvla.bench.sim.render_scene")
    stub.build_render_scene = None
    monkeypatch.setitem(sys.modules, "axis_qvla.bench.sim.render_scene", stub)
    saved = sys.modules.pop("axis_qvla.bench.sim.rollout_dualsim", None)
    try:
        mod = importlib.import_module("axis_qvla.bench.sim.rollout_dualsim")

        def compile_plain(entry):
            # Resolved against the cwd the builder chdir'd into (the scene dir, as in the eval).
            # Absolute, because MuJoCo's process-wide asset cache keys a RELATIVE mesh path by its
            # string, and every test here spells it 'scenes/../meshes/box.obj'.
            model = mujoco.MjModel.from_xml_path(str(Path(entry).resolve()))
            return model, mujoco.MjData(model)
        monkeypatch.setattr(mod, "compile_with_cameras", compile_plain)
        yield mod
    finally:
        sys.modules.pop("axis_qvla.bench.sim.rollout_dualsim", None)
        if hasattr(sim_pkg, "rollout_dualsim"):
            delattr(sim_pkg, "rollout_dualsim")
        if saved is not None:
            sys.modules["axis_qvla.bench.sim.rollout_dualsim"] = saved


def _physics(dual, tmp_path, asset_dir, lock):
    model, _ = dual._build_physics_scene(
        {"mjcf_xml": MJCF}, -1, None, tmp_path / "work" / "physics",
        asset_dir=str(asset_dir), asset_fetch="refuse", asset_base_url=None,
        asset_lock=str(lock) if lock else None)
    return model


def test_a_clean_build_compiles_the_locked_bytes(dual, tmp_path):
    assert int(_physics(dual, tmp_path, _asset_dir(tmp_path), _lock(tmp_path)).mesh_vertnum[0]) == 4


def test_a_wrong_mesh_left_in_the_work_dir_is_never_compiled(dual, tmp_path):
    _seed_work_dir(tmp_path / "work" / "physics", CUBE)
    model = _physics(dual, tmp_path, _asset_dir(tmp_path), _lock(tmp_path))
    assert int(model.mesh_vertnum[0]) == 4, "the work dir's stale mesh was compiled, not the locked one"


def test_a_stale_work_dir_cannot_mask_a_refused_asset(dual, tmp_path):
    """The asset dir's mesh is off the lock; a previous run left the locked bytes in the work dir.
    A cached copy must not satisfy the build without asking the store."""
    _seed_work_dir(tmp_path / "work" / "physics", TETRA)
    with pytest.raises(AssetFetchRefused, match="box.obj"):
        _physics(dual, tmp_path, _asset_dir(tmp_path, mesh=CUBE), _lock(tmp_path))
    with pytest.raises(AssetFetchRefused, match="box.obj"):        # and without a lock: missing
        _physics(dual, tmp_path, _asset_dir(tmp_path / "x", mesh=None), None)


def test_a_refused_include_aborts_instead_of_failing_the_compile(dual, tmp_path):
    """axis-render's resolver swallows the loader's exception; the refusal must still abort."""
    with pytest.raises(AssetFetchRefused, match="r.xml"):
        _physics(dual, tmp_path, _asset_dir(tmp_path, include=None), _lock(tmp_path))
    with pytest.raises(AssetFetchRefused, match="r.xml"):
        _physics(dual, tmp_path, _asset_dir(tmp_path / "y", include=ROBOT + b" "), _lock(tmp_path / "y"))


# --- the render shell -------------------------------------------------------------------------

@pytest.fixture
def render(monkeypatch):
    """render_scene imported against stub robosuite modules; the stub env compiles its entry XML."""
    seen: dict = {}

    class Env:
        def __init__(self, *, entry_xml, layout_cache_dir, **kw):
            seen["entry"], seen["layout_cache_dir"] = Path(entry_xml), Path(layout_cache_dir)
            model = mujoco.MjModel.from_xml_path(str(entry_xml))
            self.sim = types.SimpleNamespace(model=types.SimpleNamespace(_model=model),
                                             data=types.SimpleNamespace(_data=mujoco.MjData(model)))
            self.xml_metadata = {}

    rr = types.ModuleType("axis_replay_native.robosuite_render")
    rr.AxisReplayRobosuiteEnv = Env
    rr._build_scene_state_transform = lambda model, meta: (np.zeros(model.nq), None)
    rr._load_embodiment_camera_config = lambda embodiment, cams: {}
    vs = types.ModuleType("axis_replay_native.robosuite_visual_scene")
    vs.resolve_visual_scene_profile = lambda cfg, **kw: cfg
    monkeypatch.setitem(sys.modules, "axis_replay_native.robosuite_render", rr)
    monkeypatch.setitem(sys.modules, "axis_replay_native.robosuite_visual_scene", vs)
    saved = sys.modules.pop("axis_qvla.bench.sim.render_scene", None)
    try:
        yield importlib.import_module("axis_qvla.bench.sim.render_scene"), seen
    finally:
        sys.modules.pop("axis_qvla.bench.sim.render_scene", None)
        if hasattr(sim_pkg, "render_scene"):
            delattr(sim_pkg, "render_scene")
        if saved is not None:
            sys.modules["axis_qvla.bench.sim.render_scene"] = saved


def _render(render, tmp_path, asset_dir, lock):
    mod, _ = render
    task_dir = tmp_path / "tasks" / "task_806"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.json").write_text(json.dumps({"mjcf_xml": MJCF}))
    return mod.build_render_scene(task_dir, init_state_seed=0, task_id=806, task_name="t",
                                  render_root=tmp_path / "render_root", work_dir=tmp_path / "work",
                                  asset_dir=str(asset_dir), asset_fetch="refuse",
                                  asset_lock=str(lock) if lock else None)


def test_the_render_shell_never_compiles_a_wrong_mesh_left_in_the_work_dir(render, tmp_path):
    _seed_work_dir(tmp_path / "work", CUBE)
    scene = _render(render, tmp_path, _asset_dir(tmp_path), _lock(tmp_path))
    assert int(scene.model.mesh_vertnum[0]) == 4
    _, seen = render
    assert seen["layout_cache_dir"] == tmp_path / "work" / "layouts"   # the shell's cache is not a scene input


def test_the_render_shell_cannot_mask_a_refused_asset_or_include(render, tmp_path):
    _seed_work_dir(tmp_path / "work", TETRA)
    with pytest.raises(AssetFetchRefused, match="box.obj"):
        _render(render, tmp_path, _asset_dir(tmp_path, mesh=CUBE), _lock(tmp_path))
    with pytest.raises(AssetFetchRefused, match="r.xml"):
        _render(render, tmp_path, _asset_dir(tmp_path / "z", include=None), _lock(tmp_path))


def test_the_scene_dir_is_emptied_but_a_symlink_is_never_followed(tmp_path):
    """fresh_scene_dir deletes only what it owns: a symlinked scene dir is unlinked, its target kept."""
    work = tmp_path / "work"
    (work / asset_guard.SCENE_DIR / "old").mkdir(parents=True)
    (work / "keep.txt").write_text("x")
    d = asset_guard.fresh_scene_dir(work)
    assert d == work / asset_guard.SCENE_DIR and list(d.iterdir()) == [] and (work / "keep.txt").exists()
    target = tmp_path / "elsewhere"
    (target / "precious").mkdir(parents=True)
    d.rmdir()
    d.symlink_to(target, target_is_directory=True)
    d = asset_guard.fresh_scene_dir(work)
    assert not d.is_symlink() and list(d.iterdir()) == [] and (target / "precious").is_dir()


@pytest.fixture
def physics(monkeypatch):
    """The real rollout_physics, removed again afterwards: other tests stub it in sys.modules and
    reach it as an attribute of the package, which a real import would otherwise leave behind.
    Its one name from axis_replay_native.rollout (RolloutResult) is stubbed: an earlier test may
    leave AXIS-Render's axis_replay_native imported, and that package has no rollout module."""
    rollout = types.ModuleType("axis_replay_native.rollout")
    rollout.RolloutResult = lambda success, n_steps, sim_time, reason="": types.SimpleNamespace(
        success=success, n_steps=n_steps, sim_time=sim_time, reason=reason)
    monkeypatch.setitem(sys.modules, "axis_replay_native.rollout", rollout)
    name = "axis_qvla.bench.sim.rollout_physics"
    saved_mod, saved_attr = sys.modules.pop(name, None), sim_pkg.__dict__.get("rollout_physics")
    try:
        yield importlib.import_module(name)
    finally:
        sys.modules.pop(name, None)
        if saved_mod is not None:
            sys.modules[name] = saved_mod
        if saved_attr is not None:
            sim_pkg.rollout_physics = saved_attr
        elif hasattr(sim_pkg, "rollout_physics"):
            delattr(sim_pkg, "rollout_physics")


def test_the_physics_backend_aborts_on_a_refused_include(physics, tmp_path):
    """rollout_physics already builds in a fresh TemporaryDirectory, but the swallowed include refusal
    reached it too: it returned `error:compile:` -- a seed record -- instead of aborting."""
    rollout_pi05_policy = physics.rollout_pi05_policy
    task_dir = tmp_path / "tasks" / "task_806"
    task_dir.mkdir(parents=True)
    (task_dir / "task.json").write_text(json.dumps({"mjcf_xml": MJCF}))
    with pytest.raises(AssetFetchRefused, match="r.xml"):
        rollout_pi05_policy(task_dir, None, init_state_seed=0, prompt="p", open_w=0.08, closed_w=0.0,
                            checker_factory=lambda s: (lambda m, d: False),
                            asset_dir=str(_asset_dir(tmp_path, include=None)), asset_fetch="refuse",
                            asset_lock=str(_lock(tmp_path)))
