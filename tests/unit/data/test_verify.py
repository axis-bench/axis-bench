"""`qvla data verify`: the Hub against the lock without downloading, and a local tree against it.

Driven through the public CLI (the private `release verify` runs the same command), so it holds in
both trees."""

from __future__ import annotations

import dataclasses
import hashlib
import http.server
import threading

import pytest

import axis_qvla.cli as cli
from axis_qvla.core import lock as cl
from axis_qvla.data import verify as vf
from axis_qvla.data.hf import HfHttpError, TreeEntry

C = "a" * 40
RID = "axisrobotics/qvla-v1-results"
DATA = {"cells.json": b"{}\n", "rollouts/s2/x.tar": b"tar-bytes"}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _blob(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def _entries(data=DATA):
    return [TreeEntry("cells.json", len(data["cells.json"]), _blob(data["cells.json"]), None),
            TreeEntry("rollouts/s2/x.tar", len(data["rollouts/s2/x.tar"]), _blob(b"pointer"),
                      _sha(data["rollouts/s2/x.tar"]))]


def _lock():
    """Hand-built -- what build_lock("v1.0", {RID: (HfRepo(RID, "dataset"), "v1.0", C,
    _entries())}, {(RID, "cells.json"): _sha(DATA["cells.json"])}) would produce. Built by hand,
    not with build_lock, because this file ships and build_lock does not."""
    cells_entry, tar_entry = _entries()
    return {
        "schema": cl.LOCK_SCHEMA,
        "release": "v1.0",
        "repos": {RID: {"type": "dataset", "tag": "v1.0", "commit": C}},
        "files": [
            {"repo": RID, "path": cells_entry.path, "size": cells_entry.size,
             "sha256": _sha(DATA["cells.json"]), "git_oid": cells_entry.git_oid},
            {"repo": RID, "path": tar_entry.path, "size": tar_entry.size,
             "sha256": tar_entry.lfs_sha256, "git_oid": tar_entry.git_oid},
        ],
    }


def _remote(monkeypatch, entries, tag_commit=C):
    monkeypatch.setattr(vf, "revision_commit", lambda repo, rev, **k: tag_commit)
    monkeypatch.setattr(vf, "list_tree", lambda repo, rev, **k: list(entries))


def test_a_matching_hub_verifies_clean(monkeypatch):
    _remote(monkeypatch, _entries())
    assert vf.verify_remote(_lock(), token=None, endpoint="x") == []


def test_changed_missing_extra_and_moved_tag_are_each_reported(monkeypatch):
    changed = dict(DATA, **{"rollouts/s2/x.tar": b"other-bytes"})
    entries = _entries(changed)[1:] + [TreeEntry("stray.txt", 1, "9" * 40, None)]
    _remote(monkeypatch, entries, tag_commit="b" * 40)
    problems = vf.verify_remote(_lock(), token=None, endpoint="x")
    text = "\n".join(problems)
    assert "cells.json: missing" in text
    assert "rollouts/s2/x.tar: size" in text
    assert "stray.txt: on the Hub but not in the checksum manifest" in text
    assert "tag v1.0 is at bbbbbbbbbbbb" in text


def test_equal_size_sha256_mismatch_is_reported(monkeypatch):
    """The sha256-differs branch (size matches but the LFS digest doesn't) needs its
    own case."""
    changed = dict(DATA, **{"rollouts/s2/x.tar": b"OTHERBYTE"})
    assert len(changed["rollouts/s2/x.tar"]) == len(DATA["rollouts/s2/x.tar"])
    _remote(monkeypatch, _entries(changed))
    problems = vf.verify_remote(_lock(), token=None, endpoint="x")
    assert any("rollouts/s2/x.tar: sha256 differs from the checksum manifest" in p for p in problems)


def test_equal_size_git_blob_mismatch_is_reported(monkeypatch):
    """The git-blob-differs branch (a git-stored file, not LFS) needs its own case."""
    entries = _entries()
    bad = [dataclasses.replace(entries[0], git_oid="9" * 40), entries[1]]
    _remote(monkeypatch, bad)
    problems = vf.verify_remote(_lock(), token=None, endpoint="x")
    assert any("cells.json: git blob differs from the checksum manifest" in p for p in problems)


def test_verify_remote_raises_instead_of_reporting_an_unreachable_repo_as_a_diff(monkeypatch):
    """'Could not check' must not be folded into the diff list -- a caller needs to
    tell an unreachable/401 Hub (exit 2) apart from a real mismatch (exit 1)."""
    monkeypatch.setattr(vf, "revision_commit",
                        lambda repo, rev, **k: (_ for _ in ()).throw(HfHttpError("HTTP 401")))
    with pytest.raises(HfHttpError, match="401"):
        vf.verify_remote(_lock(), token=None, endpoint="x")


def test_verify_remote_exits_2_on_an_unreachable_hub_through_the_real_fake_server(tmp_path, monkeypatch):
    """Exercised through a real (fake) HTTP server, NOT monkeypatched away --
    a 401 from the Hub itself must surface as exit 2, not get swallowed into the diff list."""
    class _Unauthorized(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = b"unauthorized"
            self.send_response(401)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Unauthorized)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        lock_path = tmp_path / "lock.json"
        lock_path.write_bytes(cl.lock_bytes(_lock()))
        monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{srv.server_address[1]}")
        monkeypatch.delenv("HF_TOKEN", raising=False)
        assert cli.main(["data", "verify", "--checksums", str(lock_path)]) == 2
    finally:
        srv.shutdown()
        srv.server_close()


def test_verify_local_raises_on_a_repo_not_in_the_lock(tmp_path):
    """'Not in the lock' is a configuration mistake, not a diff -- exit 2, not exit 1."""
    with pytest.raises(ValueError, match="not in the checksum manifest"):
        vf.verify_local(_lock(), "axisrobotics/unknown", tmp_path)


def test_cli_verify_local_exits_2_for_a_repo_not_in_the_lock(tmp_path):
    lock_path = tmp_path / "lock.json"
    lock_path.write_bytes(cl.lock_bytes(_lock()))
    argv = ["data", "verify", "--checksums", str(lock_path), "--local", str(tmp_path), "--repo", "axisrobotics/unknown"]
    assert cli.main(argv) == 2


def test_verify_local(tmp_path):
    for rel, data in DATA.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(data)
    assert vf.verify_local(_lock(), RID, tmp_path) == []
    (tmp_path / "cells.json").write_bytes(b"[]\n")
    (tmp_path / "extra.bin").write_bytes(b"x")
    problems = vf.verify_local(_lock(), RID, tmp_path)
    assert any("cells.json: differs" in p for p in problems)
    assert any("extra.bin: present locally but not in the checksum manifest" in p for p in problems)


def test_cli_verify_local_exit_codes(tmp_path):
    for rel, data in DATA.items():
        (tmp_path / "tree" / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "tree" / rel).write_bytes(data)
    lock_path = tmp_path / "lock.json"
    lock_path.write_bytes(cl.lock_bytes(_lock()))
    argv = ["data", "verify", "--checksums", str(lock_path), "--local", str(tmp_path / "tree"), "--repo", RID]
    assert cli.main(argv) == 0
    (tmp_path / "tree" / "cells.json").write_bytes(b"[]\n")
    assert cli.main(argv) == 1
    assert cli.main(["data", "verify", "--checksums", str(tmp_path / "missing.json")]) == 2


def test_cli_verify_exits_2_not_0_when_the_lock_has_no_real_files_row(tmp_path):
    """End to end: a checksum manifest whose 'files' is {} (not a list) is refused, exit 2."""
    import json

    doc = {"schema": cl.LOCK_SCHEMA, "release": "v1.0",
           "repos": {RID: {"type": "dataset", "tag": "v1.0", "commit": C}}, "files": {}}
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(doc))
    argv = ["data", "verify", "--checksums", str(lock_path), "--local", str(tmp_path), "--repo", RID]
    assert cli.main(argv) == 2


def test_default_lock_path_is_the_source_trees_whatever_the_working_directory(tmp_path, monkeypatch):
    """The default lock is lock.default_lock(), the checkout's conf/release/v1.0.lock.json -- the
    one rule every T0 default follows (cells, layout, benchmark), so `data verify` and `paper
    tables` read the same lock.

    SOURCE_TREE and default_lock() live in axis_qvla.core.lock (the private release lock module
    re-exports them), so the monkeypatch must target the defining module."""
    tree = tmp_path / "checkout"
    (tree / "conf" / "release").mkdir(parents=True)
    (tree / "conf" / "release" / "v1.0.lock.json").write_bytes(cl.lock_bytes(_lock()))
    data_dir = tmp_path / "data"
    for rel, data in DATA.items():
        (data_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (data_dir / rel).write_bytes(data)
    monkeypatch.setattr(cl, "SOURCE_TREE", tree)
    monkeypatch.chdir(data_dir)
    assert cli.main(["data", "verify", "--local", str(data_dir), "--repo", RID]) == 0


def test_verify_local_fails_on_an_unreadable_directory(tmp_path):
    (tmp_path / "rollouts").mkdir()
    (tmp_path / "rollouts").chmod(0o000)
    try:
        with pytest.raises(OSError):
            vf.verify_local(_lock(), RID, tmp_path)
    finally:
        (tmp_path / "rollouts").chmod(0o755)
