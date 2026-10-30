"""Read and write one served arm's per-seed rollout files, byte-exact.

Why: T0 recomputes every table from the raw `seed_<n>.json` files, and the eval fingerprint
hashes their exact bytes, so a reader may never re-serialise them. Two sources hold the same
files: a served directory on disk (the box layout) and a published seed-only tar. Both readers
return one `ServedRun`; `write_served_tar` makes the tar deterministic so its sha256 can be
locked.

Directories are listed with `os.scandir` only, and any `OSError` is fatal: `glob`/`iterdir`
turn an unreadable directory into an empty one.

Core tier: stdlib only.
"""
from __future__ import annotations

import dataclasses
import io
import json
import os
import re
import tarfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from axis_qvla.core.canon import sha256_file   # the one chunked sha256 helper

_TASK_DIR = re.compile(r"task_(\d+)")
_IGNORED = ("meta.json",)
# The one name a result file has: exactly `seed_<int>.json`, what the harness's write_seed writes.
# Every reader selects with this pattern -- the campaign tally and eval fingerprint
# (axis_qvla.bench.sim.record imports is_seed_file from here), this package's fingerprint and results
# files, T0's per-task seed check -- so a file can never count in one and not another. The old
# `seed_*.json` minus `*.diag.json` read seed_<n>.diag_objz.json as a result. ASCII digits spelled
# out: `\d` also matches other scripts' digits ("seed_\u0661.json"). Group 1 is the seed.
SEED_FILE = re.compile(r"seed_([0-9]+)\.json")
# What else a task directory holds under --diagnostics, never a result: the per-seed trace
# (seed_<n>.diag.json) and what the render and dual-sim backends derive from its path
# (seed_<n>.diag_objz.json, seed_<n>.diag_*.png, seed_<n>.diag.mp4), or the physics video
# (seed_<n>.mp4) -- axis_qvla.bench.sim.domain.diagnostic_path.
_SIDECAR = re.compile(r"seed_[0-9]+\.(?:diag[._].+|mp4)")


class SeedFileError(RuntimeError):
    """A served run cannot be read exactly: unreadable, malformed or unsafe."""


@dataclasses.dataclass(frozen=True)
class SeedFile:
    name: str
    data: bytes

    def record(self) -> dict:
        return json.loads(self.data)


@dataclasses.dataclass(frozen=True)
class ServedRun:
    served: str
    tasks: Mapping[int, tuple[SeedFile, ...]]

    def tallies(self) -> dict[int, tuple[int, int]]:
        return {tid: (sum(1 for f in files if f.record().get("success") is True), len(files))
                for tid, files in sorted(self.tasks.items())}


def is_seed_file(name: str) -> bool:
    """True for exactly `seed_<int>.json`: a rollout result (SEED_FILE)."""
    return SEED_FILE.fullmatch(name) is not None


def _keep(name: str) -> bool:
    """True for a seed file; False for meta.json or a diagnostics sidecar; raise for anything else,
    so a stray name (seed_x.json, a leftover temp file) is reported rather than skipped silently."""
    if is_seed_file(name):
        return True
    if name in _IGNORED or _SIDECAR.fullmatch(name):
        return False
    raise SeedFileError(f"unexpected file {name!r} in a task directory")


def read_served_dir(log_root: Path, served: str) -> ServedRun:
    root = Path(log_root) / served
    tasks: dict[int, tuple[SeedFile, ...]] = {}
    try:
        with os.scandir(root) as it:
            entries = sorted((e.name, e.is_dir(follow_symlinks=False)) for e in it)
        for name, is_dir in entries:
            m = _TASK_DIR.fullmatch(name)
            if m is None:
                continue
            if not is_dir:
                raise SeedFileError(f"{root / name} is not a directory")
            files = []
            with os.scandir(root / name) as it:
                seeds = sorted((e.name, e.is_file(follow_symlinks=False)) for e in it)
            for fname, is_file in seeds:
                if not _keep(fname):
                    continue
                if not is_file:
                    raise SeedFileError(f"{root / name / fname} is not a regular file")
                files.append(SeedFile(fname, (root / name / fname).read_bytes()))
            tasks[int(m.group(1))] = tuple(files)
    except OSError as exc:
        raise SeedFileError(f"cannot read {root}: {exc}") from exc
    return ServedRun(served, tasks)


def _member_parts(name: str, served: str) -> tuple[str, ...]:
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts:
        raise SeedFileError(f"unsafe tar member {name!r}")
    parts = p.parts[1:] if p.parts[:1] == (".",) else p.parts
    if not parts or parts[0] != served:
        raise SeedFileError(f"tar member {name!r} is not under {served}/")
    return parts


def read_served_tar(tar_path: Path, served: str) -> ServedRun:
    found: dict[int, dict[str, bytes]] = {}
    seen: set[str] = set()
    try:
        with tarfile.open(tar_path, mode="r:*") as tf:
            for ti in tf:
                parts = _member_parts(ti.name, served)
                key = "/".join(parts)
                if key in seen:
                    raise SeedFileError(f"duplicate tar member {ti.name!r}")
                seen.add(key)
                if ti.isdir():
                    if len(parts) == 2 and _TASK_DIR.fullmatch(parts[1]):
                        found.setdefault(int(parts[1][5:]), {})
                    elif len(parts) != 1:
                        raise SeedFileError(f"unexpected directory {ti.name!r}")
                    continue
                if not ti.isreg():
                    raise SeedFileError(f"tar member {ti.name!r} is not a regular file")
                m = _TASK_DIR.fullmatch(parts[1]) if len(parts) == 3 else None
                if m is None:
                    raise SeedFileError(f"tar member {ti.name!r} is not <served>/task_<tid>/<file>")
                if not _keep(parts[2]):
                    continue
                fh = tf.extractfile(ti)
                found.setdefault(int(m.group(1)), {})[parts[2]] = fh.read()
    except (OSError, tarfile.TarError) as exc:
        raise SeedFileError(f"cannot read {tar_path}: {exc}") from exc
    return ServedRun(served, {tid: tuple(SeedFile(n, d) for n, d in sorted(files.items()))
                              for tid, files in sorted(found.items())})


def _info(name: str, *, size: int, mtime: int, is_dir: bool) -> tarfile.TarInfo:
    ti = tarfile.TarInfo(name)
    ti.type = tarfile.DIRTYPE if is_dir else tarfile.REGTYPE
    ti.size, ti.mtime = size, mtime
    ti.mode = 0o755 if is_dir else 0o644
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    return ti


def write_served_tar(run: ServedRun, out: Path, *, mtimes: Mapping[str, int] | None = None) -> str:
    mt = dict(mtimes or {})
    out = Path(out)
    tmp = out.with_name(out.name + ".partial")
    with tarfile.open(tmp, mode="w", format=tarfile.USTAR_FORMAT) as tf:
        tf.addfile(_info(run.served, size=0, mtime=mt.get(run.served, 0), is_dir=True))
        for tid in sorted(run.tasks):
            d = f"{run.served}/task_{tid}"
            tf.addfile(_info(d, size=0, mtime=mt.get(d, 0), is_dir=True))
            for f in sorted(run.tasks[tid], key=lambda s: s.name):
                name = f"{d}/{f.name}"
                tf.addfile(_info(name, size=len(f.data), mtime=mt.get(name, 0), is_dir=False),
                           io.BytesIO(f.data))
    os.replace(tmp, out)
    return sha256_file(out)
