"""SIGNALS.lock guard: locked bytes pass, deprecated bytes and other generations are refused.

Two generations of a signal file can agree on every content property a loader reads (dtype,
n_rows, reward_id, the AWR corpus fingerprint), so the guard identifies a file by its bytes. These
tests pin the four verify_signal branches plus the kind check, the published lock's form, and the
property that lets the module be vendored byte-identically into both forks: no paths, no axis
imports, Python 3.10 syntax.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import pathlib
import re
import sys

import pytest

from axis_qvla.core import signals_lock as sl

MODULE = pathlib.Path(sl.__file__)


def _put(path: pathlib.Path, data: bytes) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _row(fid: str, path: pathlib.Path, kind: str, legacy: str, rowspace: str = "rs") -> dict:
    sha, md5, size = sl.file_digests(path)
    return {"id": fid, "path": f"signals/v1/{fid.replace('.', '/')}.npz", "legacy_names": [legacy], "kind": kind,
            "reward_id": "phase", "granularity": "chunk", "rowspace": rowspace, "size": size,
            "sha256": sha, "md5": md5, "generator": {}, "derived_from": [], "consumers": {},
            "trained": ["pi05.pt.fixture"], "reproduces": None}


def _dep(path: pathlib.Path, legacy: str, superseded_by: str | None, *, md5_only: bool = False) -> dict:
    sha, md5, _ = sl.file_digests(path)
    d = {"md5": md5, "legacy_name": legacy, "superseded_by": superseded_by,
         "reason": "fixture stale copy"}
    if not md5_only:
        d["sha256"] = sha
    return d


@pytest.fixture()
def world(tmp_path):
    study = _put(tmp_path / "study" / "quality_phase.npz", b"study generation")
    stale = _put(tmp_path / "stale" / "quality_phase.npz", b"stale generation")
    plateau = _put(tmp_path / "plateau" / "quality_phase_seg.npz", b"plateau rebuild")
    lock = {
        "schema": sl.SCHEMA, "release": "test",
        "rowspaces": {"rs": {"n_rows": 1906469, "corpus_fingerprint": "sha256:fixture"}},
        "files": [_row("approach_aware.episode.tags", study, "quality_tags", "quality_phase.npz")],
        "deprecated": [_dep(stale, "quality_phase.npz", "approach_aware.episode.tags"),
                       _dep(plateau, "quality_phase_seg.npz", None, md5_only=True)],
    }
    assert sl.validate_lock(lock) == []
    return tmp_path, lock, study, stale, plateau


def test_file_digests_agree_with_hashlib(tmp_path):
    p = _put(tmp_path / "x.bin", b"abc" * 1000)
    data = p.read_bytes()
    assert sl.file_digests(p) == (hashlib.sha256(data).hexdigest(), hashlib.md5(data).hexdigest(), 3000)


def test_locked_bytes_pass_and_return_their_row(world):
    _, lock, study, _, _ = world
    row = sl.verify_signal(study, kind="quality_tags", rowspace_key=1906469, lock=lock)
    assert row is not None and row["id"] == "approach_aware.episode.tags"


def test_deprecated_sha256_is_refused_and_names_the_replacement(world):
    _, lock, _, stale, _ = world
    with pytest.raises(sl.SignalLockError, match="approach_aware.episode.tags"):
        sl.verify_signal(stale, kind="quality_tags", rowspace_key=1906469, lock=lock)


def test_deprecated_md5_only_row_is_refused(world):
    _, lock, _, _, plateau = world
    with pytest.raises(sl.SignalLockError, match="deprecated"):
        sl.verify_signal(plateau, kind="quality_tags", rowspace_key=1906469, lock=lock)


def test_other_generation_of_a_locked_name_is_refused(world):
    tmp_path, lock, _, _, _ = world
    other = _put(tmp_path / "rebuilt" / "quality_phase.npz", b"a third generation")
    with pytest.raises(sl.SignalLockError, match="different generation of approach_aware.episode.tags"):
        sl.verify_signal(other, kind="quality_tags", rowspace_key=1906469, lock=lock)
    # The same name over a DIFFERENT row space is another corpus's file: unlocked, not refused.
    assert sl.verify_signal(other, kind="quality_tags", rowspace_key=12345, lock=lock) is None


def test_unknown_file_passes_as_unlocked(world):
    tmp_path, lock, _, _, _ = world
    unknown = _put(tmp_path / "x" / "quality_randcam.npz", b"another corpus")
    assert sl.verify_signal(unknown, kind="quality_tags", rowspace_key=1906469, lock=lock) is None


def test_locked_bytes_under_the_wrong_kind_are_refused(world):
    _, lock, study, _, _ = world
    with pytest.raises(sl.SignalLockError, match="kind"):
        sl.verify_signal(study, kind="schedule", rowspace_key=1906469, lock=lock)


def test_verify_file_refuses_a_mismatch(world):
    _, _, study, stale, _ = world
    sl.verify_file(study, sl.file_digests(study)[0])
    with pytest.raises(sl.SignalLockError, match="sha256"):
        sl.verify_file(stale, sl.file_digests(study)[0])


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d["files"][0].update(sha256="xyz"), "sha256"),
    (lambda d: d["files"][0].update(rowspace="nope"), "rowspace"),
    (lambda d: d["files"].append(dict(d["files"][0])), "duplicate"),
    (lambda d: d["deprecated"][0].update(superseded_by="ghost"), "superseded_by"),
    (lambda d: d["deprecated"][0].update(sha256=d["files"][0]["sha256"]), "also a locked"),
    (lambda d: d["files"][0].update(kind="weird"), "kind"),
    (lambda d: d.update(schema="other"), "schema"),
])
def test_validate_lock_reports_each_defect(world, mutate, message):
    _, lock, _, _, _ = world
    doc = json.loads(json.dumps(lock))
    mutate(doc)
    assert any(message in p for p in sl.validate_lock(doc)), sl.validate_lock(doc)


PUBLISHED = "signals/v1/approach_aware/episode/tags.npz"


def test_other_generation_at_the_published_path_is_refused(world):
    """The 14 pi0.5 files are republished as tags.npz / awr_weights.json / ...: a legacy-name
    match alone would let tampered bytes at the v1.0 path through with only a warning."""
    tmp_path, lock, _, _, _ = world
    tampered = _put(tmp_path / "corpus" / PUBLISHED, b"tampered at the published path")
    with pytest.raises(sl.SignalLockError, match="different generation of approach_aware.episode.tags"):
        sl.verify_signal(tampered, kind="quality_tags", rowspace_key=1906469, lock=lock)
    assert sl.verify_signal(tampered, kind="quality_tags", rowspace_key=12345, lock=lock) is None


def test_a_published_path_matches_whole_components_only(world):
    tmp_path, lock, _, _, _ = world
    near = _put(tmp_path / ("my" + PUBLISHED), b"not under the published path")
    assert sl.verify_signal(near, kind="quality_tags", rowspace_key=1906469, lock=lock) is None


@pytest.mark.parametrize("rel", [PUBLISHED, "elsewhere/quality_phase.npz"])
def test_a_known_path_or_name_without_a_rowspace_key_is_refused(world, rel):
    tmp_path, lock, _, _, _ = world
    f = _put(tmp_path / "corpus" / rel, b"tampered, no row space given")
    with pytest.raises(sl.SignalLockError, match="rowspace_key"):
        sl.verify_signal(f, kind="quality_tags", rowspace_key=None, lock=lock)


def test_locked_bytes_are_refused_under_another_or_no_rowspace_key(world):
    _, lock, study, _, _ = world
    row = sl.verify_signal(study, kind="quality_tags", rowspace_key="sha256:fixture", lock=lock)
    assert row["id"] == "approach_aware.episode.tags"
    for key in (None, 12345):
        with pytest.raises(sl.SignalLockError, match="rowspace"):
            sl.verify_signal(study, kind="quality_tags", rowspace_key=key, lock=lock)


def test_an_unknown_file_without_a_rowspace_key_passes_as_unlocked(world):
    tmp_path, lock, _, _, _ = world
    unknown = _put(tmp_path / "x" / "quality_randcam.npz", b"another corpus")
    assert sl.verify_signal(unknown, kind="quality_tags", rowspace_key=None, lock=lock) is None


def test_a_file_row_with_no_trained_arm_and_extra_keys_is_valid(world):
    """A row needs FILE_KEYS; keys beyond them (a full lock's evidence) are not validated."""
    _, lock, _, _, _ = world
    doc = json.loads(json.dumps(lock))
    doc["files"][0].update(trained=[], evidence=["x"])
    assert sl.validate_lock(doc) == []


def test_a_missing_or_corrupt_lock_is_a_signal_lock_error(world):
    tmp_path, _, study, _, _ = world
    bad = _put(tmp_path / "corrupt.json", b"{not json")
    for p in (tmp_path / "absent.json", bad):
        with pytest.raises(sl.SignalLockError, match="cannot read signal lock"):
            sl.load_lock(p)
        sl._CACHE.clear()
        with pytest.raises(sl.SignalLockError, match="cannot read signal lock"):
            sl.guard_signal(study, kind="quality_tags", rowspace_key=1906469, lock_path=p)


def test_load_lock_refuses_an_invalid_file(tmp_path):
    p = _put(tmp_path / "SIGNALS.lock.json", json.dumps({"schema": "other"}).encode())
    with pytest.raises(sl.SignalLockError, match="invalid"):
        sl.load_lock(p)


def test_guard_signal_loads_each_lock_once_and_logs(world, monkeypatch, caplog):
    tmp_path, lock, study, _, _ = world
    lock_path = _put(tmp_path / "SIGNALS.lock.json", json.dumps(lock).encode())
    sl._CACHE.clear()
    calls = []
    real = sl.load_lock
    monkeypatch.setattr(sl, "load_lock", lambda p: calls.append(p) or real(p))
    caplog.set_level(logging.INFO, logger="qvla_signals")
    sl.guard_signal(study, kind="quality_tags", rowspace_key=1906469, lock_path=lock_path)
    sl.guard_signal(study, kind="quality_tags", rowspace_key=1906469, lock_path=lock_path)
    assert len(calls) == 1
    assert "signal approach_aware.episode.tags sha256" in caplog.text
    sl._CACHE.clear()


def test_module_is_vendorable():
    """Byte-identical copies live in openpi (3.11) and LeRobot (3.13): no axis imports, no
    paths, and nothing newer than 3.10 syntax."""
    src = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(src, feature_version=(3, 10))
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
        else:
            continue
        assert all(n == "__future__" or n in sys.stdlib_module_names for n in names), names
    consts = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert not [c for c in consts if c.startswith("/")], "the module must not know any path"


def test_guard_source_has_no_renameable_axis_token():
    """The release rename pass rewrites the import root inside strings, dotted or as a path. One
    such token here (a logger named "axis_qvla.signals") would make the public axis_qvla copy differ from
    the fork copies, and `release check` could never pass. The module must be rename-invariant."""
    src = (pathlib.Path(__file__).resolve().parents[3] / "axis_qvla/core/signals_lock.py").read_text()
    assert re.search(r"(?<![\w./-])axis(?=\.[a-z_]|/)", src) is None


REPO = pathlib.Path(__file__).resolve().parents[3]


def test_the_real_lock_is_valid_and_lists_the_published_files():
    lock = sl.load_lock(REPO / "conf" / "release" / "SIGNALS.lock.json")
    by_id = {f["id"]: f for f in lock["files"]}
    assert by_id["approach_aware.chunk.tags"]["sha256"] == (
        "849987860b9e370c5ce1b753be10b4d71b6d14d7afa3ecabb93fc2385747a4f7")
    assert len([f for f in lock["files"] if f["kind"] in ("quality_tags", "awr_weights", "schedule")]) == 12
    assert len(lock["files"]) == 30
    assert all(f["path"].startswith(("signals/v1/", "index/")) for f in lock["files"])
    assert lock["rowspaces"]["libero5k_v2_d8"]["n_rows"] == 1906469


def test_the_real_lock_carries_what_the_guard_and_the_loaders_read():
    """Each file row is FILE_KEYS exactly; a trained arm is its run and its basis; a generator is
    its digests; the lock has no deprecation list and no row space records where it was read."""
    lock = sl.load_lock(REPO / "conf" / "release" / "SIGNALS.lock.json")
    assert set(lock) == {"schema", "release", "rules", "rowspaces", "files"}
    for f in lock["files"]:
        assert set(f) == set(sl.FILE_KEYS), f["id"]
        assert all(set(t) == {"arm", "basis"} for t in f["trained"]), f["id"]
        assert set(f["generator"]) == {"generator_sha256", "config_hash"}, f["id"]
    assert not [n for n, sp in lock["rowspaces"].items() if "source" in sp]


PUBLISHED_SHA256 = {  # the 12 pi0.5 label and weight files v1.0 locks
    "approach_aware.episode.tags": "ed532d06f83721b69ad4d14f8a82cb42b6d7eaf6c27cea534e559393c273b9ba",
    "approach_aware.chunk.tags": "849987860b9e370c5ce1b753be10b4d71b6d14d7afa3ecabb93fc2385747a4f7",
    "object_centric.episode.tags": "621a430addd79e0c27086e809923bceda762b022d8b6b193dc0b53a3d88f7acb",
    "object_centric.chunk.tags": "88f692f9cad3fc221733721411d5be7178fc29d1519ec47267a3681046206959",
    "approach_aware.chunk.awr_weights": "0def8a0808dec5c7ad2b7dabc29a1201686d23aae85fcc963cb0a00daa12692e",
    "approach_aware.episode.awr_weights": "0fe760afa8ec1063dae2a0eed5277f9a59604b0e9b6f95a156004bd5c80e0989",
    "object_centric.chunk.awr_weights": "84f3fb4fea2b180833617106399e0220e75488e9029762e4b3874674d207f9a1",
    "object_centric.episode.awr_weights": "023300e1463b59091ea2dc8eebff946714fe6f71b83efd7af2ee6f6cdb2c943d",
    "approach_aware.chunk.schedule_drop_top50": "fcd920b6e8a479a20db6c257e72affd79393d29f975195f1a88b1587abd49718",
    "approach_aware.episode.schedule_drop_top50": "0662d940f7d51e0d98845b51bac564beb1232eab15cc851ca5b07798c2c78242",
    "object_centric.chunk.schedule_drop_top50": "900782d354ad0e8a4b9f50e85b704cf940e44ed8b08a3a8d4cf4d63052198edb",
    "object_centric.episode.schedule_drop_top50": "20953f84c53cd11720fc954adafa3178d6f1889b15c7380799e2f885b4efb505",
}


def test_the_real_lock_pins_all_12_pi05_files():
    lock = sl.load_lock(REPO / "conf" / "release" / "SIGNALS.lock.json")
    pinned = {f["id"]: f["sha256"] for f in lock["files"] if f["kind"] in ("quality_tags", "awr_weights", "schedule")}
    assert pinned == PUBLISHED_SHA256


@pytest.fixture()
def real_layout(tmp_path):
    """The real lock's 30 rows (ids, published paths, kinds, row spaces) with each
    row's digests swapped for those of small stand-in bytes, so path rules run on the real layout
    without the real (private) bytes."""
    lock = json.loads((REPO / "conf" / "release" / "SIGNALS.lock.json").read_text())
    blobs = {}
    for f in lock["files"]:
        blobs[f["id"]] = data = f"stand-in bytes of {f['id']}".encode()
        f["sha256"], f["md5"] = hashlib.sha256(data).hexdigest(), hashlib.md5(data).hexdigest()
        f["size"] = len(data)
    assert sl.validate_lock(lock) == []
    return tmp_path, lock, blobs


def _key(lock, row):
    return lock["rowspaces"][row["rowspace"]]["n_rows"]


def test_another_rows_locked_bytes_at_a_locked_path_are_refused(real_layout):
    """The silent arm swap: v2 chunk tags (same kind, same row space) presented as the
    approach-aware chunk tags must not pass as 'locked'."""
    tmp_path, lock, blobs = real_layout
    rows = {f["id"]: f for f in lock["files"]}
    phase, v2 = rows["approach_aware.chunk.tags"], rows["object_centric.chunk.tags"]
    assert (phase["kind"], phase["rowspace"]) == (v2["kind"], v2["rowspace"])
    for rel in (phase["path"], "mirror/" + phase["path"]):
        swapped = _put(tmp_path / "corpus" / rel, blobs[v2["id"]])
        with pytest.raises(sl.SignalLockError, match="object_centric.chunk.tags.*approach_aware.chunk.tags"):
            sl.verify_signal(swapped, kind="quality_tags", rowspace_key=_key(lock, phase), lock=lock)


def test_every_locked_row_passes_at_its_own_published_path(real_layout):
    tmp_path, lock, blobs = real_layout
    for f in lock["files"]:
        for rel in (f["path"], f"mirror/{f['id']}/{f['path']}"):
            p = _put(tmp_path / "own" / rel, blobs[f["id"]])
            assert sl.verify_signal(p, kind=f["kind"], rowspace_key=_key(lock, f), lock=lock)["id"] == f["id"]


def test_no_locked_row_passes_at_another_rows_path(real_layout):
    tmp_path, lock, blobs = real_layout
    for at in lock["files"]:
        for f in lock["files"]:
            if f["id"] == at["id"]:
                continue
            for rel in (at["path"], f"mirror/{at['path']}"):
                p = _put(tmp_path / "swap" / f["id"] / rel, blobs[f["id"]])
                with pytest.raises(sl.SignalLockError, match="locked bytes of"):
                    sl.verify_signal(p, kind=f["kind"], rowspace_key=_key(lock, f), lock=lock)
