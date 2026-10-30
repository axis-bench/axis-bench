"""The v1.0 release checksum manifest: every published file of every qvla-v1-* repo, pinned by sha256.

Core tier (stdlib only). This is the read half: loading and validating a checksum manifest. Writing
one is the publisher's tool. conf/release/v1.0.lock.json is the one input that says what v1.0 IS:
for each repo its type, its tag and the commit that tag points at; for each file its size,
sha256 and git blob id. `qvla data verify` compares it with the Hub (no download) or with a
local copy; publication additionally requires every row to be under a confirmed licence.
"""

from __future__ import annotations

import json
import pathlib
import re
from collections.abc import Mapping

from axis_qvla.core.canon import dumps_canonical

LOCK_SCHEMA = "qvla.release_lock/1"
# The checkout this code runs from. T0's default inputs (cells, layout, benchmark, lock) all resolve
# against it, so T0 runs from a checkout -- an editable install or `python -m` from the repository
# root -- never from a non-editable install, which carries no conf/.
SOURCE_TREE = pathlib.Path(__file__).resolve().parents[2]
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")


def _valid_rel_path(s: object) -> bool:
    """A file path stored in the lock: relative, no `..` segment, no leading `/`, no backslash."""
    if not isinstance(s, str) or not s or s.startswith("/") or "\\" in s:
        return False
    return all(part not in ("", "..") for part in s.split("/"))


def validate_lock(doc: Mapping[str, object]) -> list[str]:
    """Every defect this reports is a str; this function itself never raises -- a malformed
    field (wrong type, unhashable) is a problem to report, not a crash to propagate out of a
    CLI command that is supposed to exit cleanly with a message."""
    if not isinstance(doc, Mapping):
        return ["the checksum manifest is not a JSON object"]
    p: list[str] = []
    if doc.get("schema") != LOCK_SCHEMA:
        p.append(f"schema is {doc.get('schema')!r}, expected {LOCK_SCHEMA!r}")
    if not isinstance(doc.get("release"), str):
        p.append("release must be a string")
    repos_raw = doc.get("repos")
    repos = repos_raw if isinstance(repos_raw, Mapping) else {}
    if not isinstance(repos_raw, Mapping):
        p.append("repos must be an object")
    if not repos:
        p.append("repos must be a non-empty object")
    for rid, meta in repos.items():
        if not isinstance(meta, Mapping) or meta.get("type") not in ("dataset", "model"):
            p.append(f"repo {rid}: type must be dataset or model")
        elif not isinstance(meta.get("tag"), str) or not _HEX40.fullmatch(str(meta.get("commit", ""))):
            p.append(f"repo {rid}: tag must be a string and commit 40 hex")

    files_raw = doc.get("files")
    if not isinstance(files_raw, list):
        p.append(f"files must be a list, got {type(files_raw).__name__}")
        files_raw = []

    keys: list[tuple[str, str]] = []
    repos_with_rows: set[str] = set()
    for i, f in enumerate(files_raw):
        where = f"files[{i}]"
        if not isinstance(f, Mapping):
            p.append(f"{where}: must be an object")
            continue
        where = f"files[{i}] {f.get('repo')!r}:{f.get('path')!r}"
        if not {"repo", "path", "size", "sha256", "git_oid"} <= set(f):
            p.append(f"{where}: needs repo, path, size, sha256, git_oid")
            continue
        repo_ok = isinstance(f["repo"], str)
        if not repo_ok:
            p.append(f"{where}: repo must be a string")
        path_ok = _valid_rel_path(f["path"])
        if not path_ok:
            p.append(f"{where}: path must be a relative path with no leading '/', no '..', no '\\\\'")
        if not _HEX64.fullmatch(str(f["sha256"])):
            p.append(f"{where}: sha256 must be 64 hex")
        if not _HEX40.fullmatch(str(f["git_oid"])):
            p.append(f"{where}: git_oid must be 40 hex")
        if not isinstance(f["size"], int) or isinstance(f["size"], bool) or f["size"] < 0:
            p.append(f"{where}: size must be a non-negative int")
        if repo_ok and f["repo"] not in repos:
            p.append(f"{where}: unknown repo")
        if repo_ok and path_ok:
            keys.append((f["repo"], f["path"]))
            repos_with_rows.add(f["repo"])
    if len(set(keys)) != len(keys):
        p.append("duplicate (repo, path) rows")
    if keys != sorted(keys):
        p.append("files must be sorted by (repo, path)")
    for rid in repos:
        if rid not in repos_with_rows:
            p.append(f"repo {rid}: no files listed")
    return p


def default_lock() -> pathlib.Path:
    """The one default --checksums of `paper tables` and `release verify`: the checkout's
    conf/release/v1.0.lock.json (read at call time, so a test can point SOURCE_TREE elsewhere)."""
    return SOURCE_TREE / "conf" / "release" / "v1.0.lock.json"


def load_lock(path: pathlib.Path) -> dict:
    doc = json.loads(pathlib.Path(path).read_bytes())
    problems = validate_lock(doc)
    if problems:
        raise ValueError(f"invalid release checksum manifest {path}: {problems[:3]}")
    return doc


def lock_bytes(doc: Mapping[str, object]) -> bytes:
    return dumps_canonical(doc)
