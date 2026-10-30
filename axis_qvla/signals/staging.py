"""Where a build may write inside a published corpus tree.

A `qvla-v1-corpus` tree (the directory holding SIGNALS.lock.json) is an upload source and a
lock-verified download: its index/ and signals/ files are the locked inputs every bundle is
compiled from. Staging adds exactly one subtree to it (bundles/ or traces_sample/); a local or
partial build belongs outside it. `check_out` refuses any other destination before anything is
written.

One other entry exists transiently: while `qvla labels build-paper` runs, it writes into
`<tree>/.bundles.staging-<12 hex>/` beside `bundles/` and moves the finished result into place.
That directory is beside, and not elsewhere, because the move is a rename, which needs both on one
filesystem. It is removed when the build succeeds, and removed best-effort on failure; any leftover
is reported by `qvla labels check` (axis_qvla.signals.paper.staging_dirs), and never read as part of
the tree.

Every probe here asks `kind_of`, which raises on an unreadable path instead of answering "absent"
(`Path.is_file` answers False on EACCES on some Python versions): a lock behind an unreadable
directory never reads as "not a published tree".

Stdlib only.
"""
from __future__ import annotations

import os
import pathlib
import stat

LOCK_NAME = "SIGNALS.lock.json"


class StagingError(ValueError):
    """A build was pointed into a published tree somewhere it must not write."""


def kind_of(path) -> str | None:
    """"dir", "file" (anything else that exists) or None for a path that does not exist. Only a path
    that cannot exist -- missing, or below something that is not a directory -- is None: any other
    OSError, EACCES on a directory above it first of all, propagates, so an unreadable path is never
    read as an absent one."""
    try:
        mode = os.stat(path).st_mode
    except (FileNotFoundError, NotADirectoryError):
        return None
    return "dir" if stat.S_ISDIR(mode) else "file"


def pack_root(path) -> pathlib.Path | None:
    """The published tree `path` lies in (the nearest directory holding SIGNALS.lock.json), or None."""
    p = pathlib.Path(path).resolve()
    for d in (p, *p.parents):
        if kind_of(d / LOCK_NAME) == "file":
            return d
    return None


def check_out(out, subtree: str, *, partial: bool = False) -> None:
    """Refuse `out` if it lies in a published tree anywhere but `<tree>/<subtree>`, or at all for a
    partial (local) build."""
    root = pack_root(out)
    if root is None:
        return
    if partial:
        raise StagingError(f"{out} is inside the published tree {root}; a partial build is a local build, "
                           f"write it outside that tree")
    if pathlib.Path(out).resolve() != root / subtree:
        raise StagingError(f"{out} is inside the published tree {root}; staging writes only {root / subtree}")

