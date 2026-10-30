"""`qvla bench eval` checks its on-disk inputs before any server starts.

The asset store checks every scene file it opens against conf/benchmarks/axis-heldout-v1/ASSETS.lock.json, but the
AXIS-Render shell reads the Polyhaven bank straight from
<render-root>/assets/table_textures/polyhaven/downloaded_256/. So `bench eval` compares that
directory with the checksum manifest's bank rows -- the same set both ways, every sha256 -- and refuses
otherwise; and it refuses a checkout whose ASSETS.lock.json is missing instead of silently
dropping --asset-checksums. Runs against a tmp host profile and a tmp lock, so it holds in the public
tree as written.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from axis_qvla.bench import inputs, versions
from axis_qvla.cli import main
from tests.unit.bench.eval_stubs import ready_host
from tests.unit.bench.test_eval_roots import PROFILE, make_bank


def _lock(tmp_path) -> dict:
    return json.loads((tmp_path / "lock.json").read_text())


def test_an_identical_bank_passes(tmp_path, monkeypatch):
    make_bank(tmp_path / "render", tmp_path / "lock.json", monkeypatch)
    assert inputs.bank_problems(tmp_path / "render", _lock(tmp_path)) == []


@pytest.mark.parametrize("change,why", [
    (lambda d: (d / "polyhaven_0001_b.jpg").unlink(), "polyhaven_0001_b.jpg: in the checksum manifest, not under the render root"),
    (lambda d: (d / "extra.jpg").write_bytes(b"x"), "extra.jpg: under the render root, not in the checksum manifest"),
    (lambda d: (d / "polyhaven_0000_a.jpg").write_bytes(b"aab"), "polyhaven_0000_a.jpg: sha256 differs from the checksum manifest"),
    (lambda d: (d / "sub").mkdir(), "sub: not a regular file"),
])
def test_any_difference_from_the_lock_is_a_problem(tmp_path, monkeypatch, change, why):
    change(make_bank(tmp_path / "render", tmp_path / "lock.json", monkeypatch))
    assert inputs.bank_problems(tmp_path / "render", _lock(tmp_path)) == [why]


def test_a_missing_or_unreadable_bank_is_refused_never_skipped(tmp_path, monkeypatch):
    d = make_bank(tmp_path / "render", tmp_path / "lock.json", monkeypatch)
    probs = inputs.bank_problems(tmp_path / "other", _lock(tmp_path))
    assert len(probs) == 1 and "does not exist" in probs[0] and "data.md#the-benchmark-inputs" in probs[0]
    d.chmod(0)
    try:
        probs = inputs.bank_problems(tmp_path / "render", _lock(tmp_path))
    finally:
        d.chmod(0o755)
    assert len(probs) == 1 and "cannot be read (PermissionError)" in probs[0]
    assert inputs.bank_problems(tmp_path / "render", {"files": []}) == [
        f"ASSETS.lock.json has no rows under {inputs.BANK_LOCK_PREFIX}"]


@pytest.fixture
def host(tmp_path, monkeypatch):
    hosts = tmp_path / "hosts"
    hosts.mkdir()
    (hosts / "demo.toml").write_text(PROFILE.format(d=tmp_path))
    monkeypatch.setenv("AXIS_HOSTS_DIR", str(hosts))
    monkeypatch.setattr(os, "environ", dict(os.environ))  # restored after the test
    for d in ("render", "tasks", "assets", "ck"):
        (tmp_path / d).mkdir()
    from axis_qvla.bench.sim import campaign

    seen: list = []
    monkeypatch.setattr(campaign, "main", lambda argv: seen.append(list(argv)) or 0)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "venvs" / "render" / "bin" / "python"))
    ready_host(monkeypatch, tmp_path)
    return tmp_path, seen


def _run(d: Path, *extra: str) -> int:
    return main(["bench", "eval", "--host", "demo", "--checkpoint", str(d / "ck"), "--config", "cfg", "--run", "p",
                 "--asset-dir", str(d / "assets"), "--render-root", str(d / "render"),
                 "--task-data-root", str(d / "tasks"), *extra])


def test_bench_eval_runs_on_a_locked_bank_and_refuses_any_other_before_the_driver(host, monkeypatch, capsys):
    d, seen = host
    bank = make_bank(d / "render", d / "lock.json", monkeypatch)
    assert _run(d) == 0 and f"--asset-checksums={d / 'lock.json'}" in seen.pop()
    (bank / "polyhaven_0000_a.jpg").write_bytes(b"tampered")
    assert _run(d) == 2 and seen == []
    out = capsys.readouterr().out
    assert "Polyhaven bank" in out and "polyhaven_0000_a.jpg: sha256 differs" in out and "before any server" in out


def test_bench_eval_refuses_a_missing_asset_lock_and_always_passes_the_flag(host, monkeypatch, capsys):
    d, seen = host
    make_bank(d / "render", d / "lock.json", monkeypatch)
    (d / "lock.json").unlink()
    assert _run(d) == 2 and seen == []
    out = capsys.readouterr().out
    assert "ASSETS.lock.json" in out and "missing" in out and "before any server" in out
    assert _run(d, "--dry-run") == 0 and f"--asset-checksums={d / 'lock.json'}" in capsys.readouterr().out


def test_without_render_root_the_bank_check_verifies_the_tree_the_workers_import(host, monkeypatch, capsys):
    """ONE resolver (launch.render_tree) for the workers' PYTHONPATH and the bank check: without
    --render-root the workers import <checkout>/third_party/axis-render/src, so that tree's bank is
    the one checked."""
    from axis_qvla.bench import launch
    from axis_qvla.bench.spec import load_benchmark
    from axis_qvla.core.hosts import load_host

    d, seen = host
    monkeypatch.setattr(launch, "REPO_ROOT", d / "checkout")
    h, b = load_host("demo"), load_benchmark()
    tree = launch.render_tree(h, None)
    assert tree == d / "checkout" / "third_party" / "axis-render"
    for given, want in ((None, tree), (str(d / "render"), d / "render")):
        env = launch.compose(launch.EvalRequest(checkpoint="x", config="c", render_root=given), h, b).env
        assert env["PYTHONPATH"].split(":")[-1] == str(launch.render_tree(h, given) / "src") == str(want / "src")
    bank = make_bank(tree, d / "lock.json", monkeypatch)
    run = ["bench", "eval", "--host", "demo", "--checkpoint", str(d / "ck"), "--config", "cfg", "--run", "p",
           "--asset-dir", str(d / "assets"), "--task-data-root", str(d / "tasks")]
    assert main(run) == 0 and seen
    (bank / "polyhaven_0000_a.jpg").write_bytes(b"tampered")
    assert main(run) == 2 and "polyhaven_0000_a.jpg: sha256 differs" in capsys.readouterr().out


@pytest.mark.parametrize("text,why", [("[1, 2]", "is not a JSON object"), ("{\"files\": 3}", "no files list"),
                                      ("{\"files\": [3]}", "row 0 is not an object")])
def test_a_malformed_lock_is_a_refusal_line_never_a_traceback(host, monkeypatch, capsys, text, why):
    d, seen = host
    make_bank(d / "render", d / "lock.json", monkeypatch)
    (d / "lock.json").write_text(text)
    assert _run(d) == 2 and seen == []
    out = capsys.readouterr().out
    assert why in out and "Refused before any server starts" in out and "Traceback" not in out


def test_a_link_to_the_locked_bytes_is_accepted_a_dangling_one_is_not(tmp_path, monkeypatch):
    """The evaluation tutorial says copy or link the bank: a symlink is followed and its bytes are checked."""
    d = make_bank(tmp_path / "render", tmp_path / "lock.json", monkeypatch)
    real = tmp_path / "real.jpg"
    real.write_bytes((d / "polyhaven_0000_a.jpg").read_bytes())
    (d / "polyhaven_0000_a.jpg").unlink()
    (d / "polyhaven_0000_a.jpg").symlink_to(real)
    assert inputs.bank_problems(tmp_path / "render", _lock(tmp_path)) == []
    real.unlink()
    assert inputs.bank_problems(tmp_path / "render", _lock(tmp_path)) == ["polyhaven_0000_a.jpg: not a regular file"]


# --- the pre-launch gate must check the SAME lock the workers get -----------------------------

def test_asset_lock_for_the_default_version_is_assets_lock(monkeypatch, tmp_path):
    from axis_qvla.bench import spec

    lock = tmp_path / "default.json"
    monkeypatch.setattr(inputs, "ASSETS_LOCK", lock)

    class _B:
        conf_dir = spec.CONF_DIR

    assert inputs.asset_lock_for(_B()) == lock


def test_asset_lock_for_a_non_default_version_is_its_own_lock(tmp_path):
    class _B:
        conf_dir = tmp_path / "benchmarks" / "axis-heldout-v2"

    assert inputs.asset_lock_for(_B()) == tmp_path / "benchmarks" / "axis-heldout-v2" / "ASSETS.lock.json"


def _second_version(root: Path, bank: dict[str, bytes]) -> Path:
    """A frozen, self-consistent second version directory (never touches axis-heldout-v1/):
    the shipped files, renamed, with its OWN ASSETS.lock.json describing `bank`."""
    v2 = root / "axis-heldout-v2"
    shutil.copytree(versions.version_dir(versions.DEFAULT_VERSION), v2)
    (v2 / versions.VERSION_FILE).unlink()
    decl = json.loads((v2 / "franka_heldout24.json").read_text())
    decl["version"] = "axis-heldout-v2"
    (v2 / "franka_heldout24.json").write_text(json.dumps(decl))
    rows = [{"path": inputs.BANK_LOCK_PREFIX + n, "sha256": hashlib.sha256(b).hexdigest(), "size": len(b)}
            for n, b in sorted(bank.items())]
    (v2 / "ASSETS.lock.json").write_text(json.dumps({"schema": "qvla.assets_lock/1", "files": rows}))
    versions.freeze(v2)
    return v2


def test_bench_eval_checks_the_requested_versions_own_lock_not_the_default(tmp_path, monkeypatch, capsys):
    """`inputs.refusal` must check the SAME lock `path_argv` hands the workers (asset_lock_for(b)),
    not always the hard-coded ASSETS_LOCK (axis-heldout-v1's). A version whose own lock matches
    the render root exactly, but whose bank the real production lock cannot match, proves the
    distinction: refused against the wrong lock, accepted against its own."""
    hosts = tmp_path / "hosts"
    hosts.mkdir()
    (hosts / "demo.toml").write_text(PROFILE.format(d=tmp_path))
    monkeypatch.setenv("AXIS_HOSTS_DIR", str(hosts))
    monkeypatch.setattr(os, "environ", dict(os.environ))  # restored after the test
    for sub in ("render", "tasks", "assets", "ck"):
        (tmp_path / sub).mkdir()

    root = tmp_path / "benchmarks"
    bank = {"v2_only.jpg": b"v2-bytes"}
    v2 = _second_version(root, bank)
    bank_dir = tmp_path / "render" / inputs.BANK_RENDER_REL
    bank_dir.mkdir(parents=True)
    for n, b in bank.items():
        (bank_dir / n).write_bytes(b)
    monkeypatch.setattr(versions, "BENCHMARKS_ROOT", root)

    from axis_qvla.bench.sim import campaign

    seen: dict = {}

    def fake_main(argv):
        seen["argv"] = list(argv)
        return 0

    monkeypatch.setattr(campaign, "main", fake_main)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "venvs" / "render" / "bin" / "python"))
    ready_host(monkeypatch, tmp_path)

    ref = "axis-heldout-v2/franka_heldout24"
    argv = ["bench", "eval", "--host", "demo", "--checkpoint", str(tmp_path / "ck"), "--config", "cfg",
            "--run", "p2", "--benchmark", ref, "--asset-dir", str(tmp_path / "assets"),
            "--render-root", str(tmp_path / "render"), "--task-data-root", str(tmp_path / "tasks")]

    assert main([*argv, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert f"--benchmark={ref}" in out and f"--asset-checksums={v2 / 'ASSETS.lock.json'}" in out

    assert main(argv) == 0, capsys.readouterr().out
    assert f"--benchmark={ref}" in seen["argv"] and f"--asset-checksums={v2 / 'ASSETS.lock.json'}" in seen["argv"]
