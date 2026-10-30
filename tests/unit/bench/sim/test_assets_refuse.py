"""Scene assets fail closed: a local directory, or an explicit URL -- never a default bucket.

Before v1 every rollout backend fell back to HttpAssetStore(DEFAULT_ASSET_BASE_URL), the
company's S3 bucket, and the harness never passed an asset directory, so a public checkout
would have fetched from a private bucket without saying so.
"""
from __future__ import annotations

import json
import socket
import sys
import types
from pathlib import Path

import pytest

from axis_qvla.bench.sim import assets
from axis_qvla.bench.sim.assets import (
    FETCH_MODES,
    AssetFetchRefused,
    RefusingAssetStore,
    bench_asset_fetch,
    make_asset_store,
)
from axis_qvla.bench.sim.domain import served_rollout_fn

SIM = Path(assets.__file__).parent
REPO = SIM.parents[2]


@pytest.fixture
def asset_dir(tmp_path):
    d = tmp_path / "assets"
    (d / "tabletopgen/scene_a/apple_1/visual").mkdir(parents=True)
    (d / "tabletopgen/scene_a/apple_1/visual/apple.obj").write_bytes(b"v 0 0 0\n")
    (d / "robots/franka_emika_panda/assets").mkdir(parents=True)
    (d / "robots/franka_emika_panda/assets/link0.stl").write_bytes(b"solid")
    return d


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def test_the_two_modes():
    assert FETCH_MODES == ("refuse", "http")


def test_refuse_without_a_directory_is_refused_by_name():
    with pytest.raises(AssetFetchRefused, match="--asset-dir"):
        make_asset_store(None, fetch="refuse")


def test_a_missing_directory_is_refused(tmp_path):
    with pytest.raises(AssetFetchRefused, match="not found"):
        make_asset_store(tmp_path / "nope", fetch="refuse")


def test_present_files_follow_the_local_store_lookup(asset_dir, no_network):
    store = make_asset_store(asset_dir, fetch="refuse")
    assert isinstance(store, RefusingAssetStore)
    assert store.fetch_file("tabletopgen/scene_a/apple_1/visual/apple.obj") == b"v 0 0 0\n"
    assert store.fetch_file("./scenes/../tabletopgen/scene_a/apple_1/visual/apple.obj") == b"v 0 0 0\n"
    assert store.fetch_file("mujoco-assets/robots/franka_emika_panda/assets/link0.stl") == b"solid"
    assert store.resolved == ["tabletopgen/scene_a/apple_1/visual/apple.obj",
                              "tabletopgen/scene_a/apple_1/visual/apple.obj",
                              "robots/franka_emika_panda/assets/link0.stl"]


def test_a_missing_file_raises_without_touching_the_network(asset_dir, no_network):
    store = make_asset_store(asset_dir, fetch="refuse")
    with pytest.raises(AssetFetchRefused, match="tabletopgen/scene_a/pear_3/visual/pear.obj"):
        store.fetch_file("tabletopgen/scene_a/pear_3/visual/pear.obj")


def test_fetch_files_has_the_materializer_shape(asset_dir):
    found, missing = make_asset_store(asset_dir, fetch="refuse").fetch_files(
        ["robots/franka_emika_panda/assets/link0.stl"])
    assert found == {"robots/franka_emika_panda/assets/link0.stl": b"solid"} and missing == []


def test_a_lock_verifies_every_file_on_first_open(asset_dir, no_network):
    """Loaders resolve through the lock and refuse a mismatch."""
    import hashlib
    rel = "tabletopgen/scene_a/apple_1/visual/apple.obj"
    good = {"files": [{"path": f"assets/{rel}", "size": 8, "sha256": hashlib.sha256(b"v 0 0 0\n").hexdigest()}]}
    assert make_asset_store(asset_dir, fetch="refuse", lock=good).fetch_file(rel) == b"v 0 0 0\n"
    bad = {"files": [dict(good["files"][0], sha256="0" * 64)]}
    with pytest.raises(AssetFetchRefused, match="sha256"):
        make_asset_store(asset_dir, fetch="refuse", lock=bad).fetch_file(rel)
    with pytest.raises(AssetFetchRefused, match="not in ASSETS.lock"):
        make_asset_store(asset_dir, fetch="refuse", lock=good).fetch_file(
            "robots/franka_emika_panda/assets/link0.stl")


def test_http_has_no_default_url():
    with pytest.raises(ValueError, match="base_url"):
        make_asset_store(None, fetch="http")


def test_an_unknown_mode_is_refused(asset_dir):
    with pytest.raises(ValueError, match="fetch"):
        make_asset_store(asset_dir, fetch="gcs")


def test_the_benchmark_declares_refuse():
    assert bench_asset_fetch(REPO / "conf/benchmarks/axis-heldout-v1/franka_heldout24.json") == "refuse"
    decl = json.loads((REPO / "conf/benchmarks/axis-heldout-v1/franka_heldout24.json").read_text())
    assert decl["assets"] == {"fetch": "refuse", "root": "axisrobotics/qvla-v1-bench@v1.0:assets/"}


def test_a_declaration_without_assets_defaults_to_refuse(tmp_path):
    p = tmp_path / "b.json"
    p.write_text(json.dumps({"name": "x"}))
    assert bench_asset_fetch(p) == "refuse"
    p.write_text(json.dumps({"assets": {"fetch": "gcs"}}))
    with pytest.raises(ValueError, match="assets.fetch"):
        bench_asset_fetch(p)


def test_no_backend_names_a_default_bucket():
    for name in ("rollout_dualsim.py", "render_scene.py", "rollout_physics.py"):
        src = (SIM / name).read_text()
        assert "DEFAULT_ASSET_BASE_URL" not in src, name
        assert "HttpAssetStore(" not in src, name
        assert "make_asset_store(" in src, name


def test_the_domain_forwards_the_asset_policy_to_every_backend(monkeypatch):
    calls = {}
    for modname, fn in (("axis_qvla.bench.sim.rollout_dualsim", "rollout_pi05_dualsim"),
                        ("axis_qvla.bench.sim.rollout_render", "rollout_pi05_render"),
                        ("axis_qvla.bench.sim.rollout_physics", "rollout_pi05_policy")):
        mod = types.ModuleType(modname)

        def run(*a, _fn=fn, **kw):
            calls[_fn] = kw
            return types.SimpleNamespace(success=True, n_steps=1, sim_time=0.1, reason="success")
        setattr(mod, fn, run)
        monkeypatch.setitem(sys.modules, modname, mod)
    policy = {"asset_dir": "/assets", "asset_fetch": "refuse", "asset_base_url": None, "asset_lock": None}
    for dom in ("dualsim", "render", "physics"):
        fn = served_rollout_fn(dom, client=object(), task_dir=Path("t"), task_id=806, prompt="p",
                               common={"fps": 15.0}, layouts={1: "L"}, render_root=Path("r"),
                               work_dir=Path("w"), assets=policy)
        fn(1)
    for name in ("rollout_pi05_dualsim", "rollout_pi05_render", "rollout_pi05_policy"):
        assert {k: calls[name][k] for k in policy} == policy, name


def test_the_asset_lock_reaches_every_store_in_every_backend():
    """With --asset-checksums every opened file is checked, in dual-sim and render too.
    Every function taking asset_fetch also takes asset_lock, every make_asset_store call passes
    lock=asset_lock, and every build_render_scene/_build_physics_scene call passes asset_lock=asset_lock.
    Static, like test_rollout_dualsim_names.py: these modules import MuJoCo and openpi_client."""
    import ast
    for name in ("rollout_dualsim.py", "render_scene.py", "rollout_physics.py", "rollout_render.py"):
        tree = ast.parse((SIM / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                params = {a.arg for a in node.args.args + node.args.kwonlyargs}
                assert "asset_fetch" not in params or "asset_lock" in params, (name, node.name)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                kws = {k.arg: k.value for k in node.keywords}
                want = {"make_asset_store": "lock", "build_render_scene": "asset_lock",
                        "_build_physics_scene": "asset_lock"}.get(node.func.id)
                if want:
                    got = kws.get(want)
                    assert isinstance(got, ast.Name) and got.id == "asset_lock", (name, node.func.id, node.lineno)


def test_the_campaign_driver_declares_and_forwards_the_asset_flags(tmp_path):
    """campaign.main parses strictly, so it must declare the asset flags `bench eval`
    passes, and it prepends them to every worker's rollout args."""
    from axis_qvla.bench.sim import campaign

    src = (SIM / "campaign.py").read_text()
    assert 'ap.add_argument("--asset-dir"' in src and 'ap.add_argument("--asset-checksums"' in src
    got = campaign.with_asset_flags("--layout-manifest L", tmp_path / "a", tmp_path / "lock.json")
    assert got == f"--asset-dir {tmp_path / 'a'} --asset-checksums {tmp_path / 'lock.json'} --layout-manifest L"
    assert campaign.with_asset_flags("", None, None) == ""


# --- a broken lock, an unreadable asset and the HTTP store all REFUSE ----------
# The rollout modules re-raise only AssetFetchRefused past their per-seed `except Exception`; any
# other exception from the store (FileNotFoundError, JSONDecodeError, KeyError, PermissionError)
# becomes an `error:build_render:` seed that scoring tallies as a policy failure.

def _row(rel: str, data: bytes) -> dict:
    import hashlib
    return {"path": f"assets/{rel}", "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


_APPLE = "tabletopgen/scene_a/apple_1/visual/apple.obj"
_GOOD = _row(_APPLE, b"v 0 0 0\n")

_BROKEN_LOCKS = {
    "not an object": [_GOOD],
    "no files key": {"schema": "qvla.assets_lock/1"},
    "files not a list": {"files": {"a": _GOOD}},
    "empty files": {"files": []},
    "row not an object": {"files": ["assets/x"]},
    "row without sha256": {"files": [{k: v for k, v in _GOOD.items() if k != "sha256"}]},
    "row without size": {"files": [{k: v for k, v in _GOOD.items() if k != "size"}]},
    "row without path": {"files": [{k: v for k, v in _GOOD.items() if k != "path"}]},
    "uppercase sha256": {"files": [dict(_GOOD, sha256=_GOOD["sha256"].upper())]},
    "short sha256": {"files": [dict(_GOOD, sha256=_GOOD["sha256"][:63])]},
    "sha256 not a string": {"files": [dict(_GOOD, sha256=1)]},
    "negative size": {"files": [dict(_GOOD, size=-1)]},
    "bool size": {"files": [dict(_GOOD, size=True)]},
    "string size": {"files": [dict(_GOOD, size="8")]},
    "float size": {"files": [dict(_GOOD, size=8.0)]},
    "absolute path": {"files": [dict(_GOOD, path="/" + _GOOD["path"])]},
    "dotdot path": {"files": [dict(_GOOD, path="assets/../" + _APPLE)]},
    "path outside assets/": {"files": [dict(_GOOD, path=_APPLE)]},
    "duplicate path": {"files": [_GOOD, _GOOD]},
}


@pytest.mark.parametrize("name", sorted(_BROKEN_LOCKS))
def test_a_malformed_lock_is_refused_as_a_mapping_and_as_a_file(name, asset_dir, tmp_path):
    doc = _BROKEN_LOCKS[name]
    p = tmp_path / "ASSETS.lock.json"
    p.write_text(json.dumps(doc))
    for lock in ([doc] if isinstance(doc, dict) else []) + [p]:   # a non-mapping only via a file
        with pytest.raises(AssetFetchRefused, match="lock"):
            assets.load_assets_lock(lock)
        with pytest.raises(AssetFetchRefused, match="lock"):
            make_asset_store(asset_dir, fetch="refuse", lock=lock)


@pytest.mark.parametrize("make", ["missing", "directory", "not json", "not utf-8"])
def test_an_unreadable_lock_file_is_refused_not_raised_raw(make, asset_dir, tmp_path):
    p = tmp_path / "ASSETS.lock.json"
    if make == "directory":
        p.mkdir()
    elif make == "not json":
        p.write_text("{not json")
    elif make == "not utf-8":
        p.write_bytes(b"\xff\xfe{}")
    with pytest.raises(AssetFetchRefused, match="lock"):
        assets.load_assets_lock(p)
    with pytest.raises(AssetFetchRefused, match="lock"):
        make_asset_store(asset_dir, fetch="refuse", lock=p)


def test_a_size_mismatch_is_refused(asset_dir, no_network):
    lock = {"files": [dict(_GOOD, size=9)]}
    with pytest.raises(AssetFetchRefused, match="size"):
        make_asset_store(asset_dir, fetch="refuse", lock=lock).fetch_file(_APPLE)


def test_an_unreadable_asset_is_refused_not_raised_raw(asset_dir, no_network):
    import os
    if os.geteuid() == 0:
        pytest.skip("root reads a mode-000 file")
    p = asset_dir / _APPLE
    p.chmod(0)
    try:
        with pytest.raises(AssetFetchRefused, match="unreadable"):
            make_asset_store(asset_dir, fetch="refuse").fetch_file(_APPLE)
    finally:
        p.chmod(0o644)


@pytest.fixture
def fake_http(monkeypatch):
    """axis_replay_native.asset_store.HttpAssetStore, served from a dict: no socket is opened."""
    served: dict[str, bytes] = {}
    calls: list[str] = []

    class HttpAssetStore:
        def __init__(self, base_url):
            self.base_url, self.closed = base_url, False

        def fetch_file(self, path):
            calls.append(path)
            return served.get(path)

        def fetch_files(self, paths):
            found = {p: served[p] for p in paths if p in served}
            return found, [p for p in paths if p not in served]

        def close(self):
            self.closed = True

    mod = types.ModuleType("axis_replay_native.asset_store")
    mod.HttpAssetStore = HttpAssetStore
    monkeypatch.setitem(sys.modules, "axis_replay_native.asset_store", mod)
    return types.SimpleNamespace(served=served, calls=calls, cls=HttpAssetStore)


def test_http_without_a_lock_is_the_plain_store(fake_http):
    assert type(make_asset_store(None, fetch="http", base_url="https://mirror.invalid/")) is fake_http.cls


def test_http_with_a_lock_checks_every_file(fake_http, no_network):
    """--asset-checksums was silently dropped under http while meta.json
    still recorded it. Every file the HTTP store serves is now checked exactly like a local one."""
    rel_robot = "robots/franka_emika_panda/assets/link0.stl"
    lock = {"files": [_GOOD, _row(rel_robot, b"solid")]}
    fake_http.served[_APPLE] = b"v 0 0 0\n"
    fake_http.served[f"mujoco-assets/{rel_robot}"] = b"solid"   # the LocalAssetStore candidate rule
    store = make_asset_store(None, fetch="http", base_url="https://mirror.invalid/", lock=lock)
    assert store.fetch_file(_APPLE) == b"v 0 0 0\n"
    assert store.fetch_file(f"mujoco-assets/{rel_robot}") == b"solid"
    assert store.fetch_files([_APPLE]) == ({_APPLE: b"v 0 0 0\n"}, [])

    fake_http.served[_APPLE] = b"v 9 9 9\n"                     # the mirror drifted
    with pytest.raises(AssetFetchRefused, match="sha256"):
        make_asset_store(None, fetch="http", base_url="https://mirror.invalid/", lock=lock).fetch_file(_APPLE)

    n = len(fake_http.calls)
    with pytest.raises(AssetFetchRefused, match="not in ASSETS.lock"):
        store.fetch_file("tabletopgen/scene_a/pear_3/visual/pear.obj")
    assert len(fake_http.calls) == n, "an unlocked file must be refused before it is fetched"

    with pytest.raises(AssetFetchRefused, match="did not serve"):
        make_asset_store(None, fetch="http", base_url="https://mirror.invalid/",
                         lock={"files": [_row("robots/x.stl", b"x")]}).fetch_file("robots/x.stl")


def test_http_with_a_lock_closes_the_wrapped_store(fake_http):
    store = make_asset_store(None, fetch="http", base_url="https://mirror.invalid/", lock={"files": [_GOOD]})
    store.close()
    assert store.inner.closed


def test_http_with_a_broken_lock_is_refused(fake_http, tmp_path):
    with pytest.raises(AssetFetchRefused, match="lock"):
        make_asset_store(None, fetch="http", base_url="https://mirror.invalid/", lock=tmp_path / "nope.json")


def test_the_driver_prepends_the_render_and_task_data_roots_only_when_given(tmp_path):
    """`bench eval --render-root/--task-data-root` reach every worker."""
    from axis_qvla.bench.sim import campaign

    got = campaign.with_root_flags("--layout-manifest L", tmp_path / "r", tmp_path / "t")
    assert got == f"--render-root {tmp_path / 'r'} --task-data-root {tmp_path / 't'} --layout-manifest L"
    assert campaign.with_root_flags("--x 1", None, None) == "--x 1"


def test_the_driver_hands_every_worker_the_roots_it_was_given(tmp_path, monkeypatch):
    """campaign.main itself applies with_root_flags -- the recorded
    worker flags carry both roots, and a run without them records none."""
    from axis_qvla.bench.sim import campaign

    monkeypatch.setattr(campaign, "python_versions", lambda python: {})
    monkeypatch.setattr(campaign, "gpu_info", lambda: {})
    monkeypatch.setattr(campaign, "log", lambda line: None)
    ck = tmp_path / "ck"
    ck.mkdir()
    (ck / "w.bin").write_bytes(b"x")
    base = ["--run", "probe", "--config", "cfg", "--checkpoint", str(ck), "--tasks", "806",
            "--log-root", str(tmp_path / "logs"), "--openpi-dir", str(tmp_path), "--n-seeds", "1",
            "--gpus", "0", "--dry-run"]
    flags = []
    for i, extra in enumerate((["--render-root", str(tmp_path / "r"), "--task-data-root", str(tmp_path / "t")], [])):
        root = tmp_path / f"run{i}"
        assert campaign.main([*base, "--run-root", str(root), *extra]) == 0
        flags.append(json.loads((root / "out_probe" / "dry_run_manifest.json").read_text())["serving"]["worker_flags"])
    assert f"--render-root {tmp_path / 'r'}" in flags[0] and f"--task-data-root {tmp_path / 't'}" in flags[0]
    assert "--render-root" not in flags[1] and "--task-data-root" not in flags[1]
