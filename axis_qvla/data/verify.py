"""Compare the release checksum manifest with the Hub (listing only, no download) or with a local tree.

CORE TIER (stdlib only). Directories are read with os.scandir and any OSError propagates:
an unreadable directory must fail the check, never read as "nothing there".
"""

from __future__ import annotations

import os
import pathlib
from collections.abc import Mapping

from axis_qvla.core.canon import sha256_file
from axis_qvla.data.hf import HfHttpError, HfRepo, list_tree, revision_commit


def _rows_by_repo(lock: Mapping[str, object]) -> dict[str, dict[str, Mapping]]:
    out: dict[str, dict[str, Mapping]] = {}
    for f in lock["files"]:
        out.setdefault(f["repo"], {})[f["path"]] = f
    return out


def verify_remote(lock: Mapping[str, object], *, token: str | None, endpoint: str) -> list[str]:
    """Diffs vs. "could not check" are kept apart: a repo the Hub refused or could not reach is
    raised as HfHttpError (the caller's "configuration or I/O error", exit 2), never folded into
    the returned diff list (the caller's "mismatches", exit 1) -- those are different failures
    with different remedies."""
    problems: list[str] = []
    unreachable: list[str] = []
    rows = _rows_by_repo(lock)
    for rid, meta in sorted(lock["repos"].items()):
        repo = HfRepo(rid, meta["type"])
        try:
            at_tag = revision_commit(repo, meta["tag"], token=token, endpoint=endpoint)
            remote = {e.path: e for e in list_tree(repo, meta["commit"], token=token, endpoint=endpoint)}
        except HfHttpError as exc:
            unreachable.append(f"{rid}: {exc}")
            continue
        if at_tag != meta["commit"]:
            problems.append(f"{rid}: tag {meta['tag']} is at {at_tag[:12]}, the checksum manifest pins {meta['commit'][:12]}")
        want = rows.get(rid, {})
        for path, row in sorted(want.items()):
            e = remote.get(path)
            if e is None:
                problems.append(f"{rid}:{path}: missing on the Hub")
            elif e.size != row["size"]:
                problems.append(f"{rid}:{path}: size {e.size} != listed {row['size']}")
            elif e.lfs_sha256 is not None and e.lfs_sha256 != row["sha256"]:
                problems.append(f"{rid}:{path}: sha256 differs from the checksum manifest")
            elif e.lfs_sha256 is None and e.git_oid != row["git_oid"]:
                problems.append(f"{rid}:{path}: git blob differs from the checksum manifest")
        problems += [f"{rid}:{p}: on the Hub but not in the checksum manifest" for p in sorted(set(remote) - set(want))]
    if unreachable:
        raise HfHttpError("; ".join(unreachable))
    return problems


def _walk(root: pathlib.Path) -> dict[str, pathlib.Path]:
    out: dict[str, pathlib.Path] = {}
    stack = [(root, "")]
    while stack:
        d, rel = stack.pop()
        with os.scandir(d) as it:
            for e in it:
                r = f"{rel}/{e.name}" if rel else e.name
                if e.is_dir(follow_symlinks=False):
                    if r != ".cache":
                        stack.append((pathlib.Path(e.path), r))
                elif e.is_file():
                    out[r] = pathlib.Path(e.path)
    return out


def verify_local(lock: Mapping[str, object], repo_id: str, root: pathlib.Path) -> list[str]:
    if repo_id not in lock["repos"]:
        raise ValueError(f"{repo_id}: not in the checksum manifest")
    want = _rows_by_repo(lock).get(repo_id, {})
    have = _walk(pathlib.Path(root))
    problems = []
    for path, row in sorted(want.items()):
        p = have.get(path)
        if p is None:
            problems.append(f"{repo_id}:{path}: missing locally")
        elif p.stat().st_size != row["size"] or sha256_file(p) != row["sha256"]:
            problems.append(f"{repo_id}:{path}: differs from the checksum manifest")
    problems += [f"{repo_id}:{p}: present locally but not in the checksum manifest" for p in sorted(set(have) - set(want))]
    return problems
