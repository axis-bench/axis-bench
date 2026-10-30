"""What an eval run record says about the task files it read.

The CLI reads two files per task under `--task-data-root`: task.json (scene variants, for the
layout planner and the rollout backend) and checker_config.json (the success spec). A record
that stored only the root's path could not say WHICH task files scored a result: the same
directory name can hold different copies of the task files on two machines. So the record stores the sha256 of each file as read, and names
the root as a logical id, 'local:<root>/<rel>' against the configured host's declared roots --
never as a machine path. An unconfigured host or a root outside every declared root records
`root: None`; the file digests still identify the content.

Stdlib + axis_qvla.core only, so both the render venv and the tests import it.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

TASK_FILES = ("task.json", "checker_config.json")


def _sha256(p: Path) -> str | None:
    try:
        data = p.read_bytes()
    except FileNotFoundError:   # only a missing file is "not read"; EACCES etc. propagate
        return None
    return hashlib.sha256(data).hexdigest()


def logical_root(path: Path | str) -> str | None:
    """The root as 'local:<root>/<rel>' for this host ($AXIS_HOST), else None.

    The same on every host, a development box included: the record names the host's declared root,
    never the box (box ids are the publication tooling's, applied when a record leaves the box).
    """
    from axis_qvla.core.hosts import HostProfileError, load_host
    from axis_qvla.core.paths import PathOutsideRoots, Paths

    try:
        host = load_host()
    except HostProfileError:
        return None
    try:
        return "local:" + Paths(host).rel(Path(path))
    except PathOutsideRoots:
        return None


def task_data_record(root: Path | str, task_id: int) -> dict:
    """`{"root": <logical id or None>, "files": {name: sha256 or None}}` for one task."""
    d = Path(root) / f"task_{task_id}"
    return {"root": logical_root(root),
            "files": {name: _sha256(d / name) for name in TASK_FILES}}
