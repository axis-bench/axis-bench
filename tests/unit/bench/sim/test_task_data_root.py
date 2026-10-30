"""--task-data-root reaches the readers that decide a result, not only the rollout backends.

The checker, the layout generators and the planner read the task files under --task-data-root,
so a checkout whose task files come from qvla-v1-bench scores against exactly those files.

Covered here: the success checker (all three factories), both layout generators, the planner,
the CLI wiring into the planner and the checker, the run record, and rollout_render's
diagnostics-only readers `_tracked_body_names` and `make_detail_probe`, which read the
rollout's own task_dir.parent.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from axis_qvla.bench.sim import layouts
from axis_qvla.bench.sim.layout_plan import plan_layouts

TASK = 806
CHECKER = {"checker": {"type": "CompositeChecker", "children": []}}
VARIANTS = {"scene_variants": [{"initial_state": {"objects": {
    "apple_1": {"pos": [0.40, -0.20, 0.80]}, "plate_2": {"pos": [0.65, 0.15, 0.80]}}}}]}


def _task_dir(root: Path) -> Path:
    d = root / f"task_{TASK}"
    d.mkdir(parents=True)
    (d / "checker_config.json").write_text(json.dumps(CHECKER))
    (d / "task.json").write_text(json.dumps(VARIANTS))
    return d


def _checker():
    """axis_qvla.bench.sim.checker, imported when a test runs, never at collection: it imports
    AXIS-Replay's axis_replay_native, and a collection-time import makes that the package every
    later test sees, so a later importorskip of AXIS-Render's
    axis_replay_native.robosuite_visual_scene skips in a suite that has both on the path."""
    from axis_qvla.bench.sim import checker
    return checker


def _manifest(tmp_path: Path) -> Path:
    p = tmp_path / "layouts.json"
    p.write_text(json.dumps({"tasks": [{"task_id": TASK, "train": [{"scene_variant_index": 0}]}]}))
    return p


def test_checker_config_is_read_from_the_given_root(tmp_path):
    _task_dir(tmp_path / "bench")
    assert _checker().load_checker_config(TASK, task_data_root=tmp_path / "bench") == CHECKER


def test_a_root_without_the_task_fails_by_name(tmp_path):
    with pytest.raises(FileNotFoundError, match="task 806"):
        _checker().load_checker_config(TASK, task_data_root=tmp_path)


def test_faithful_layout_reads_task_json_from_the_given_root(tmp_path):
    _task_dir(tmp_path / "bench")
    spec = layouts.generate_layout_faithful(
        TASK, 806000, homogeneous_manifest=_manifest(tmp_path), task_data_root=tmp_path / "bench")
    assert spec.base_variant_index == 0
    assert set(spec.object_xy) == {"apple_1", "plate_2"}


def test_plan_layouts_threads_the_root_through(tmp_path):
    _task_dir(tmp_path / "bench")
    plan = plan_layouts(TASK, [806000, 806001], manifest=_manifest(tmp_path),
                        task_data_root=tmp_path / "bench")
    assert sorted(plan) == [806000, 806001]


def _multi_variant_manifest(tmp_path: Path) -> Path:
    p = tmp_path / "multi.json"
    p.write_text(json.dumps({"tasks": [{
        "task_id": TASK, "train": [{"scene_variant_index": 0}, {"scene_variant_index": 1}],
        "layout_center_xy": [0.5, 0.0], "layout_box_m": 0.6, "target_body": "apple_1"}]}))
    return p


TWO_VARIANTS = {"scene_variants": [
    {"initial_state": {"objects": {"apple_1": {"pos": [0.40, -0.20, 0.80]},
                                   "plate_2": {"pos": [0.65, 0.15, 0.80]}}}},
    {"initial_state": {"objects": {"apple_1": {"pos": [0.45, -0.10, 0.80]},
                                   "plate_2": {"pos": [0.60, 0.20, 0.80]}}}}]}


def test_novelty_layout_reads_task_json_from_the_given_root(tmp_path):
    """generate_layout (the multi-variant sampler) honours the root too."""
    d = _task_dir(tmp_path / "bench")
    (d / "task.json").write_text(json.dumps(TWO_VARIANTS))
    spec = layouts.generate_layout(TASK, 806000, homogeneous_manifest=_multi_variant_manifest(tmp_path),
                                   task_data_root=tmp_path / "bench")
    assert spec.base_variant_index in (0, 1)
    assert set(spec.object_xy) == {"apple_1", "plate_2"}


class _Stop(Exception):
    pass


@pytest.mark.parametrize("factory", ["make_stable_checker", "make_detail_probe", "satisfied_at_init"])
def test_each_checker_factory_reads_the_given_root(tmp_path, monkeypatch, factory):
    """All three build the CheckerManager from `task_data_root`."""
    checker = _checker()
    _task_dir(tmp_path / "bench")
    seen = []

    def _manager(cfg):
        seen.append(cfg)
        raise _Stop

    monkeypatch.setattr(checker, "CheckerManager", _manager)
    args = (TASK, None, None, None) if factory == "satisfied_at_init" else (TASK, None)
    with pytest.raises(_Stop):
        getattr(checker, factory)(*args, task_data_root=tmp_path / "bench")
    assert seen == [CHECKER]


# --- the run record names the root and hashes what was read ---------------

def test_the_record_hashes_each_task_file_read(tmp_path, monkeypatch):
    import hashlib

    from axis_qvla.bench.sim.task_data import task_data_record

    monkeypatch.delenv("AXIS_HOST", raising=False)
    d = _task_dir(tmp_path / "bench")
    rec = task_data_record(tmp_path / "bench", TASK)
    assert rec == {"root": None, "files": {
        "task.json": hashlib.sha256((d / "task.json").read_bytes()).hexdigest(),
        "checker_config.json": hashlib.sha256((d / "checker_config.json").read_bytes()).hexdigest()}}
    assert task_data_record(tmp_path / "empty", TASK)["files"] == {
        "task.json": None, "checker_config.json": None}


def _host(tmp_path: Path, monkeypatch, name: str = "fx_eval") -> Path:
    hosts = tmp_path / "hosts"
    hosts.mkdir()
    roots = {k: str(tmp_path / k) for k in ("corpus", "cache", "scratch", "runs")}
    hosts.joinpath(f"{name}.toml").write_text(
        'launcher = "local"\n\n[roots]\n' + "".join(f'"{k}" = "{v}"\n' for k, v in roots.items()))
    monkeypatch.setenv("AXIS_HOSTS_DIR", str(hosts))
    monkeypatch.setenv("AXIS_HOST", name)
    return tmp_path / "scratch"


def test_the_root_is_a_local_logical_id_on_a_configured_host(tmp_path, monkeypatch):
    from axis_qvla.bench.sim.task_data import logical_root

    scratch = _host(tmp_path, monkeypatch)
    assert logical_root(scratch / "bench" / "tasks") == "local:scratch/bench/tasks"
    assert logical_root(scratch) == "local:scratch"
    assert logical_root(tmp_path / "elsewhere") is None


def test_every_host_records_the_local_form_and_no_box_map_is_read(tmp_path, monkeypatch):
    """The record never names a box: any configured host, whatever a development box map would call
    it, records the `local:` id, and the module imports no release module (no box map is read)."""
    import ast

    from axis_qvla.bench.sim import task_data
    from axis_qvla.bench.sim.task_data import logical_root, task_data_record

    scratch = _host(tmp_path, monkeypatch, name="fx_box")
    _task_dir(scratch / "bench")
    assert logical_root(scratch / "bench") == "local:scratch/bench"
    assert task_data_record(scratch / "bench", TASK)["root"] == "local:scratch/bench"
    pkg = task_data.__name__.split(".")[0]
    tree = ast.parse(Path(task_data.__file__).read_text())
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m.startswith(f"{pkg}.release") for m in imported)


def _diag_readers_without_the_root(source: str) -> list[str]:
    import ast

    out = []
    for n in ast.walk(ast.parse(source)):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id in ("load_checker_config", "make_detail_probe", "_tracked_body_names")):
            passed = {k.arg for k in n.keywords} | {a.id for a in n.args if isinstance(a, ast.Name)}
            if "task_data_root" not in passed:
                out.append(f"{n.func.id}:{n.lineno}")
    return out


def test_render_diagnostics_read_the_rollouts_own_task_root():
    """Every checker-config reader in rollout_render gets the task-data root
    (task_dir.parent), so a qvla-v1-bench checkout reads only its own task files. Static: the
    module imports openpi_client and MuJoCo."""
    import axis_qvla.bench.sim as pkg

    src = (Path(pkg.__file__).parent / "rollout_render.py").read_text()
    assert _diag_readers_without_the_root(src) == []
    assert "task_data_root = Path(task_dir).parent" in src
