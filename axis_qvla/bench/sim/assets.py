"""Where the eval harness gets scene meshes and textures: a local directory, never a default URL.

WHY. The benchmark declares assets.fetch = "refuse": every file must already be under
--asset-dir (qvla-v1-bench assets/, listed in ASSETS.lock.json), and a miss raises
AssetFetchRefused by name, before a seed is spent. "http" serves a mirror, with no default URL.

With a lock, each store -- local or HTTP -- checks the size and sha256 of every file it hands
out, and any defect in the lock itself is AssetFetchRefused too: the rollout modules re-raise only
that past their per-seed error handlers, so anything else would be scored as a failed seed. The
builders' side of the contract (an emptied scene dir, a swallowed refusal re-raised, the CLI's
up-front check) is axis_qvla.bench.sim.asset_guard.

RefusingAssetStore reimplements LocalAssetStore's lookup (AXIS-Render asset_store.py:32-47)
in the stdlib, so the refusal is testable in any venv; the HTTP store is imported only when
asked for, from the render venv. Stdlib at module scope.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path

FETCH_MODES = ("refuse", "http")
# The rollout CLI exits with this when a scene asset is refused (missing under --asset-dir, absent
# from ASSETS.lock.json, or sha256-mismatched). Distinct from 3 (UNEVALUABLE): the task is fine,
# the asset tree is not. The queue worker and the campaign driver abort the run on it.
EXIT_ASSET_REFUSED = 4
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


class AssetFetchRefused(RuntimeError):
    """A scene asset is not in the local asset directory, and fetching is refused."""


def _normalize(path: str) -> str:
    """axis_replay_native.xml_dependency_resolver.normalize_path, verbatim in behaviour."""
    parts: list[str] = []
    for part in path.replace("\\", "/").split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)


def _candidates(normalized: str) -> list[str]:
    """LocalAssetStore._build_candidates: the reference path, minus 'mujoco-assets/', under 'scenes/'."""
    out = [normalized]
    if normalized.startswith("mujoco-assets/"):
        out.append(normalized[len("mujoco-assets/"):])
    if not normalized.startswith("scenes/"):
        out.append("scenes/" + normalized)
    return out


def _row_defect(row: object) -> str | None:
    if not isinstance(row, Mapping):
        return "is not an object"
    path, sha, size = row.get("path"), row.get("sha256"), row.get("size")
    if not isinstance(path, str) or not path.startswith("assets/"):
        return f"path {path!r} is not under assets/"
    if "\\" in path or any(p in ("", ".", "..") for p in path.split("/")):
        return f"path {path!r} is not a clean relative path"
    if not isinstance(sha, str) or not _SHA256_HEX.fullmatch(sha):
        return f"sha256 {sha!r} is not 64 lowercase hex digits"
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        return f"size {size!r} is not a non-negative integer"
    return None


def load_assets_lock(lock: str | os.PathLike[str] | Mapping) -> dict[str, dict]:
    """ASSETS.lock.json rows keyed by their 'assets/<rel>' path. Any defect -- unreadable, not
    JSON, no rows, a row with a bad path, sha256 or size, a duplicate -- is AssetFetchRefused."""
    where = "ASSETS.lock.json" if isinstance(lock, Mapping) else f"asset checksum manifest {lock}"
    if isinstance(lock, Mapping):
        doc: object = lock
    else:
        try:
            doc = json.loads(Path(lock).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:   # JSONDecodeError and UnicodeDecodeError are ValueErrors
            raise AssetFetchRefused(f"{where} is unreadable: {type(exc).__name__}: {exc}") from None
    files = doc.get("files") if isinstance(doc, Mapping) else None
    if not isinstance(files, list) or not files:
        raise AssetFetchRefused(f"{where}: no 'files' rows; not an asset checksum manifest")
    rows: dict[str, dict] = {}
    for i, row in enumerate(files):
        defect = _row_defect(row)
        if defect is None and row["path"] in rows:
            defect = f"path {row['path']!r} is listed twice"
        if defect is not None:
            raise AssetFetchRefused(f"{where}: files[{i}] {defect}")
        rows[row["path"]] = dict(row)
    return rows


def _verify(lock: Mapping[str, dict], cand: str, data: bytes) -> bytes:
    row = lock.get(f"assets/{cand}")
    if row is None:
        raise AssetFetchRefused(f"asset {cand!r} is not in ASSETS.lock.json")
    if len(data) != row["size"]:
        raise AssetFetchRefused(f"asset {cand!r}: size {len(data)} differs from ASSETS.lock.json")
    if hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise AssetFetchRefused(f"asset {cand!r}: sha256 differs from ASSETS.lock.json")
    return data


class _RefusalLatch:
    """Keeps the first refusal. axis-render's XML resolver calls the loader inside `except Exception:
    continue` (xml_dependency_resolver.resolve_xml_dependencies), so a refused include would reach
    MuJoCo as a missing file -- a compile error, recorded as an error seed. The builders re-raise
    it with asset_guard.raise_if_refused."""

    refused: AssetFetchRefused | None = None

    def fetch_file(self, file_path: str) -> bytes:
        try:
            return self._fetch(file_path)
        except AssetFetchRefused as exc:
            if self.refused is None:
                self.refused = exc
            raise

    def fetch_files(self, file_paths: list[str]) -> tuple[dict[str, bytes], list[str]]:
        return {_normalize(fp): self.fetch_file(fp) for fp in file_paths}, []


class RefusingAssetStore(_RefusalLatch):
    """LocalAssetStore's candidate rule with every fallback removed: a miss raises. With a lock,
    each file's size and sha256 are checked the first time it is opened."""

    def __init__(self, asset_dir: str | os.PathLike[str], lock: Mapping[str, dict] | None = None):
        self.asset_dir = Path(asset_dir)
        if not self.asset_dir.is_dir():
            raise AssetFetchRefused(f"asset directory not found: {self.asset_dir}")
        self.resolved: list[str] = []
        self._lock = lock
        self._verified: set[str] = set()

    def _fetch(self, file_path: str) -> bytes:
        normalized = _normalize(file_path)
        for cand in _candidates(normalized):
            p = self.asset_dir / cand
            try:
                if not p.is_file():
                    continue
                data = p.read_bytes()
            except OSError as exc:   # EACCES is a broken asset tree, not a missing file
                raise AssetFetchRefused(f"asset {cand!r} under {self.asset_dir} is unreadable: {exc}") from None
            self.resolved.append(cand)
            if self._lock is not None and cand not in self._verified:
                _verify(self._lock, cand, data)
                self._verified.add(cand)
            return data
        raise AssetFetchRefused(
            f"asset {normalized!r} is not under {self.asset_dir} and assets.fetch is 'refuse'. "
            f"Download qvla-v1-bench assets/ (listed in ASSETS.lock.json) into --asset-dir."
        )

    def close(self) -> None:
        return None


class LockedAssetStore(_RefusalLatch):
    """Another store (the HTTP one) whose every file is checked against the lock. The HTTP store
    fetches by the reference path itself; its lock row is the first LocalAssetStore candidate the
    lock lists, and a path the lock does not list is refused before anything is fetched."""

    def __init__(self, inner: object, lock: Mapping[str, dict]):
        self.inner, self._lock = inner, lock

    def _fetch(self, file_path: str) -> bytes:
        normalized = _normalize(file_path)
        cand = next((c for c in _candidates(normalized) if f"assets/{c}" in self._lock), None)
        if cand is None:
            raise AssetFetchRefused(f"asset {normalized!r} is not in ASSETS.lock.json")
        data = self.inner.fetch_file(normalized)
        if data is None:
            raise AssetFetchRefused(f"asset {normalized!r} is in ASSETS.lock.json but the store did not serve it")
        return _verify(self._lock, cand, data)

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if callable(close):
            close()


def make_asset_store(
    asset_dir: str | os.PathLike[str] | None, *, fetch: str, base_url: str | None = None,
    lock: str | os.PathLike[str] | Mapping | None = None,
) -> object:
    """The store a scene is materialised from, chosen by the declared fetch policy."""
    if fetch not in FETCH_MODES:
        raise ValueError(f"asset fetch must be one of {FETCH_MODES}, got {fetch!r}")
    rows = load_assets_lock(lock) if lock is not None else None
    if fetch == "refuse":
        if asset_dir is None:
            raise AssetFetchRefused(
                "assets.fetch is 'refuse' and no asset directory was given; pass --asset-dir")
        return RefusingAssetStore(asset_dir, rows)
    if not base_url:
        raise ValueError("assets.fetch 'http' needs an explicit base_url; there is no default")
    from axis_replay_native.asset_store import HttpAssetStore

    store = HttpAssetStore(base_url)
    return store if rows is None else LockedAssetStore(store, rows)


def bench_asset_fetch(declaration_path: str | os.PathLike[str]) -> str:
    """The declaration's assets.fetch; 'refuse' when the declaration has no assets block."""
    doc = json.loads(Path(declaration_path).read_text(encoding="utf-8"))
    fetch = str((doc.get("assets") or {}).get("fetch", "refuse"))
    if fetch not in FETCH_MODES:
        raise ValueError(f"{declaration_path}: assets.fetch must be one of {FETCH_MODES}, got {fetch!r}")
    return fetch
