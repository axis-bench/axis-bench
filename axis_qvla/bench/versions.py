"""Benchmark versions: one directory per version, identified by the hash of everything in it.

Why: a benchmark is frozen once numbers are reported on it; new tasks ship as a NEW version
(conf/benchmarks/<version>/), never as edits to an old one. Its identity must therefore cover
every input that changes what a number means -- the declaration, the task and layout files, the
asset checksum manifest and the execution constants -- not only the declaration file, which is all a run
record named before. `version_sha256` is that identity: sha256 over the sorted lines
"<relative path>\\0<sha256 of the file>\\n" of every regular file in the directory except
VERSION.json. VERSION.json records it, with the per-file digests, when the version is frozen;
`check_version` names every file that no longer matches that record.

    python -m axis_qvla.bench.versions check conf/benchmarks/<version>
    python -m axis_qvla.bench.versions freeze conf/benchmarks/<new version>

Standard library only. Directories are listed with os.scandir; an OSError propagates (an
unreadable directory must never read as an empty one).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from axis_qvla.core.canon import dumps_canonical, sha256_file

BENCHMARKS_ROOT = Path(__file__).resolve().parents[2] / "conf" / "benchmarks"
DEFAULT_VERSION = "axis-heldout-v1"
VERSION_FILE = "VERSION.json"
VERSION_SCHEMA = "qvla.benchmark_version/1"


class VersionError(ValueError):
    """A version directory is missing or unreadable, or its record is malformed."""


def available_versions(root: Path | None = None) -> list[str]:
    with os.scandir(BENCHMARKS_ROOT if root is None else Path(root)) as it:
        return sorted(e.name for e in it if e.is_dir(follow_symlinks=False))


def version_dir(version: str, root: Path | None = None) -> Path:
    root = BENCHMARKS_ROOT if root is None else Path(root)
    if not version or "/" in version or version.startswith("."):
        raise VersionError(f"not a benchmark version name: {version!r}")
    d = root / version
    if not d.is_dir():
        raise VersionError(f"no benchmark version {version!r} under {root} "
                           f"(have: {', '.join(available_versions(root)) or 'none'})")
    return d


def content_files(vdir: Path) -> dict[str, str]:
    """relative posix path -> sha256 of every regular file under `vdir` except VERSION.json."""
    out: dict[str, str] = {}
    stack = [(Path(vdir), "")]
    while stack:
        d, rel = stack.pop()
        with os.scandir(d) as it:
            for e in it:
                r = f"{rel}/{e.name}" if rel else e.name
                if e.is_symlink():
                    raise VersionError(f"{Path(vdir) / r} is a symlink: a version's bytes could then "
                                       f"change without any file in it changing")
                if e.is_dir():
                    stack.append((Path(e.path), r))
                elif e.is_file() and r != VERSION_FILE:
                    out[r] = sha256_file(Path(e.path))
    return dict(sorted(out.items()))


def version_sha256(files: dict[str, str]) -> str:
    h = hashlib.sha256()
    for rel, sha in sorted(files.items()):
        h.update(f"{rel}\0{sha}\n".encode("utf-8"))
    return h.hexdigest()


def frozen_record(vdir: Path) -> dict | None:
    p = Path(vdir) / VERSION_FILE
    try:
        data = p.read_bytes()
    except FileNotFoundError:
        return None
    try:
        doc = json.loads(data)
    except json.JSONDecodeError as exc:
        raise VersionError(f"{p} is not valid JSON ({exc})") from None
    if (not isinstance(doc, dict) or doc.get("schema") != VERSION_SCHEMA
            or not isinstance(doc.get("files"), dict) or not isinstance(doc.get("sha256"), str)):
        raise VersionError(f"{p} is not a {VERSION_SCHEMA} record")
    return doc


def check_version(vdir: Path) -> list[str]:
    """Every way `vdir` disagrees with its own VERSION.json; [] when intact. A malformed
    VERSION.json (bad JSON, wrong schema, no sha256) is itself a named problem, not a traceback --
    `load_benchmark`/`bench check` print one clear line either way. Raises VersionError when the
    directory holds a symlink (`content_files`) and OSError when a file or directory in it cannot
    be read."""
    vdir = Path(vdir)
    try:
        rec = frozen_record(vdir)
    except VersionError as exc:
        return [str(exc)]
    if rec is None:
        return [f"{vdir.name}: not frozen (no {VERSION_FILE})"]
    out = []
    if rec.get("version") != vdir.name:
        out.append(f"{VERSION_FILE} names version {rec.get('version')!r}, the directory is {vdir.name!r}")
    want, have = rec["files"], content_files(vdir)
    for rel in sorted(set(want) | set(have)):
        if rel not in have:
            out.append(f"{rel}: frozen in {VERSION_FILE} but absent")
        elif rel not in want:
            out.append(f"{rel}: added after the version was frozen; a changed benchmark is a new version")
        elif want[rel] != have[rel]:
            out.append(f"{rel}: edited after the version was frozen (sha256 {have[rel][:12]}, frozen "
                       f"{want[rel][:12]}); a changed benchmark is a new version")
    if rec.get("sha256") != version_sha256(want):
        out.append(f"{VERSION_FILE}: its sha256 is not the hash of its own file list")
    return out


def freeze(vdir: Path) -> dict:
    vdir = Path(vdir)
    if (vdir / VERSION_FILE).exists():
        raise VersionError(f"{vdir.name} is already frozen; a changed benchmark is a new version directory")
    files = content_files(vdir)
    rec = {"schema": VERSION_SCHEMA, "version": vdir.name, "sha256": version_sha256(files), "files": files}
    (vdir / VERSION_FILE).write_bytes(dumps_canonical(rec))
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m axis_qvla.bench.versions", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "freeze"):
        sub.add_parser(name).add_argument("dir", type=Path)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "freeze":
            rec = freeze(a.dir)
            print(f"froze {rec['version']}: {rec['sha256']} ({len(rec['files'])} files)")
            return 0
        problems = check_version(a.dir)
    except (VersionError, OSError, ValueError) as exc:
        print(f"FAIL  {exc}")
        return 2
    for p in problems:
        print(f"FAIL  {p}")
    if not problems:
        print(f"{a.dir}: intact")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
