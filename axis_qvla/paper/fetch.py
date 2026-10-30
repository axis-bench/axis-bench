"""Where `qvla paper tables` reads the results pack from: a local directory or
`hf://<org>/<repo>@<rev>`.

Why: the same pipeline must run against the staged pack before upload and against the
published dataset after it, and every byte it reads must be checked against a digest it
already holds. `path()`/`read()` take the expected sha256 and refuse a mismatch
(`ShaMismatch`, an integrity failure, exit 2); any other failure to obtain bytes is a
`FetchError` (exit 3). A local pack with no file at the path raises `MissingFile`, a FetchError
subclass: still exit 3 for a byte `qvla paper tables` computes from (a rollout tar), but typed
apart from a path that exists and cannot be read (EACCES, a directory), so `qvla paper tables`
can make a missing REFERENCE -- a file cells.json says the pack publishes -- an integrity failure
without ever swallowing an unreadable one. An hf:// source resolves THROUGH the release checksum
manifest: the tag is replaced by the checksum manifest's pinned commit, every file -- cells.json
and expected/MANIFEST.json included -- is checked against its row in the checksum manifest, a
path the checksum manifest does not list is refused, and the cache is keyed by commit, so a moved
tag or an edited file on the Hub can never be accepted and a stale cache from another tag position
is never reused. The cache directory is chosen by the CLI. The token and endpoint are passed in by
the CLI; this module never reads the environment.

A mismatch between the downloaded bytes and the checksum-manifest row is caught by TYPE
(`axis_qvla.data.hf.HfDigestMismatch`), never by inspecting `download`'s
error text: `HfSource.path` passes the checksum manifest's own sha256 AND size as `download`'s
`expected_sha256`/`expected_size`, so a truncated or substituted file is caught before it is
ever compared to anything else.

Core tier: stdlib only.
"""
from __future__ import annotations

import dataclasses
import pathlib
import re
from typing import Protocol

from axis_qvla.paper.cells import safe_relpath
from axis_qvla.core.canon import sha256_file as _file_sha   # the one chunked sha256 helper
from axis_qvla.data.hf import HfDigestMismatch, HfHttpError, HfRepo, download

DEFAULT_ENDPOINT = "https://huggingface.co"
_HF = re.compile(r"hf://([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)@([A-Za-z0-9_.-]+)")


class FetchError(RuntimeError):
    """Bytes could not be obtained (missing file, I/O error, network)."""


class ShaMismatch(FetchError):
    """Bytes were obtained but do not match the digest they were requested with."""


class MissingFile(FetchError):
    """A local source holds no file at the path at all (ENOENT) -- never raised for a path that
    exists but cannot be read, which stays a plain FetchError."""


class Source(Protocol):
    def read(self, path: str, *, sha256: str | None) -> bytes: ...
    def path(self, path: str, *, sha256: str | None) -> pathlib.Path: ...


def _checked(rel: str) -> str:
    if not safe_relpath(rel):
        raise FetchError(f"unsafe path {rel!r}")
    return rel


def _os_reason(exc: OSError) -> str:
    """The OS's own text for `exc`, without the filename `open()` already appended to `str(exc)`
    -- callers put the path in their own message once, so it is never duplicated."""
    return exc.strerror or str(exc)


class _ReadViaPath:
    """`read` is `path` plus the bytes, for both sources: one implementation."""

    def read(self, path: str, *, sha256: str | None) -> bytes:
        try:
            return self.path(path, sha256=sha256).read_bytes()
        except OSError as exc:
            raise FetchError(f"cannot read {path}: {_os_reason(exc)}") from exc


@dataclasses.dataclass(frozen=True)
class LocalSource(_ReadViaPath):
    root: pathlib.Path

    def path(self, path: str, *, sha256: str | None) -> pathlib.Path:
        p = self.root / _checked(path)
        try:
            got = _file_sha(p)
        except FileNotFoundError as exc:
            raise MissingFile(f"cannot read {p}: {_os_reason(exc)}") from exc
        except OSError as exc:
            raise FetchError(f"cannot read {p}: {_os_reason(exc)}") from exc
        if sha256 is not None and got != sha256:
            raise ShaMismatch(f"{path}: sha256 {got[:16]} != expected {sha256[:16]}")
        return p


class LockRefused(ShaMismatch):
    """The path is not in the release lock, or the lock does not pin the repository."""


@dataclasses.dataclass(frozen=True)
class HfSource(_ReadViaPath):
    repo: HfRepo
    rev: str                                   # the lock's pinned COMMIT, never the tag
    cache: pathlib.Path
    token: str | None
    endpoint: str
    locked: tuple[tuple[str, str, int], ...] = ()   # (path, sha256, size) per file the lock lists

    def _lock_entry(self, path: str, sha256: str | None) -> tuple[str, int]:
        want = {p: (s, n) for p, s, n in self.locked}.get(path)
        if want is None:
            raise LockRefused(f"{self.repo.repo_id}:{path} not in v1.0.lock.json")
        sha, size = want
        if sha256 is not None and sha256 != sha:
            raise ShaMismatch(f"{path}: sha256 {sha256[:16]} requested, v1.0.lock.json pins {sha[:16]}")
        return sha, size

    def path(self, path: str, *, sha256: str | None) -> pathlib.Path:
        checked = _checked(path)
        sha, size = self._lock_entry(checked, sha256)
        dest = self.cache / self.repo.repo_id.replace("/", "__") / self.rev / checked
        try:
            cached_sha = _file_sha(dest)
        except FileNotFoundError:
            cached_sha = None
        except OSError as exc:
            raise FetchError(f"cannot read cached {dest}: {_os_reason(exc)}") from exc
        if cached_sha == sha:
            return dest
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            download(self.repo, self.rev, path, dest, expected_sha256=sha, expected_size=size,
                     token=self.token, endpoint=self.endpoint)
        except HfDigestMismatch as exc:
            raise ShaMismatch(str(exc)) from exc
        except HfHttpError as exc:
            raise FetchError(f"hf://{self.repo.repo_id}@{self.rev}/{path}: {exc}") from exc
        except OSError as exc:
            raise FetchError(f"hf://{self.repo.repo_id}@{self.rev}/{path}: {_os_reason(exc)}") from exc
        return dest


def open_source(spec: str, *, cache: pathlib.Path | None, token: str | None = None,
                endpoint: str | None = None, lock: dict | None = None) -> Source:
    if spec.startswith("hf://"):
        m = _HF.fullmatch(spec)
        if m is None:
            raise FetchError(f"bad results source {spec!r}; expected hf://<org>/<repo>@<rev>")
        if cache is None:
            raise FetchError("an hf:// results source needs a cache directory")
        if lock is None:
            raise FetchError("an hf:// results source resolves through the release checksum manifest (--checksums)")
        info = (lock.get("repos") or {}).get(m.group(1))
        if not info or info.get("tag") != m.group(2):
            raise LockRefused(f"{m.group(1)}@{m.group(2)} not in v1.0.lock.json")
        locked = tuple((str(f["path"]), str(f["sha256"]), int(f["size"])) for f in lock.get("files", [])
                       if f.get("repo") == m.group(1))
        return HfSource(HfRepo(m.group(1), "dataset"), str(info["commit"]), pathlib.Path(cache), token,
                        endpoint or DEFAULT_ENDPOINT, locked=locked)
    return LocalSource(pathlib.Path(spec))
