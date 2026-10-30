"""The per-seed result schema and the run-level provenance that says what produced it.

Both harnesses already write `{seed, success, n_steps, sim_time, reason}` per seed, and
`report.py` / `compare_variants.py` parse that unchanged -- so the schema is fixed here, not
redesigned. What differs is provenance: the served-pi0.5 path writes a `meta.json` naming the
checkpoint and settings, and a local-checkpoint path would write none, leaving its numbers
untraceable to the checkpoint that produced them. One writer, used by both, keeps every result
traceable.

Extra per-seed fields are carried in the seed file itself rather than only in `meta.json`
because seed files get copied around: a guidance sweep's records are otherwise
indistinguishable, and a fresh-layout record must never be mistaken for a shipped-variant one.

Offline tier: stdlib only, plus the seed-file rule from axis_qvla.bench.results.seedfiles (core tier, itself
stdlib only). Writing and reading results must not require jax, torch or a simulator -- scoring a
finished run is a different job from producing it.
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
from pathlib import Path

# The one name a result file has: exactly `seed_<int>.json`, what write_seed writes. Everything else
# in a task directory -- meta.json, and under --diagnostics seed_<n>.diag.json, seed_<n>.diag_objz.json,
# PNG frames and MP4s -- is a sidecar; `seed_*.json` minus `*.diag.json` read seed_<n>.diag_objz.json
# (and a stray seed_x.json) as results: a phantom seed in the tally and a moved eval fingerprint.
# The rule is defined ONCE, in the results package (stdlib only), and imported here, so the tally,
# this fingerprint and the published results can never select different files.
from axis_qvla.bench.results.seedfiles import is_seed_file

# Written into every meta.json. Bump when a change makes new results non-comparable with old
# ones; a reader can then refuse to average across the boundary instead of doing it silently.
SCHEMA_VERSION = 1


class RecordError(ValueError):
    """A result could not be written or read back as a well-formed record."""


@dataclasses.dataclass(frozen=True)
class SeedResult:
    """One rollout outcome. Mirrors what the rollout backends already return."""

    seed: int
    success: bool
    n_steps: int
    sim_time: float
    reason: str

    def to_json(self) -> dict:
        return {
            "seed": int(self.seed),
            "success": bool(self.success),
            "n_steps": int(self.n_steps),
            "sim_time": float(self.sim_time),
            "reason": self.reason,
        }


def git_sha(repo_root: Path | None = None) -> str:
    """Short HEAD sha, or "unknown" -- provenance must never abort a finished eval.

    A rollout sweep can cost GPU-hours; refusing to write its results because `git` is absent
    or the tree is not a repository would throw away the expensive part to protect the cheap
    part. Every failure mode collapses to "unknown", which is honest and readable.
    """
    cmd = ["git", "rev-parse", "--short", "HEAD"]
    try:
        out = subprocess.run(
            cmd,
            cwd=str(repo_root) if repo_root else None,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if out.returncode != 0:
        return "unknown"
    try:
        return out.stdout.decode().strip() or "unknown"
    except UnicodeDecodeError:
        return "unknown"


def write_seed(dir_: Path, result: SeedResult, extra: dict | None = None) -> Path:
    """Write `seed_<n>.json`, merging `extra` provenance that must travel with the record.

    `extra` cannot overwrite a schema field: a caller passing `success` would otherwise
    reverse the outcome of a rollout through a side channel, and that failure is invisible in
    the output. It is refused loudly instead.
    """
    payload = result.to_json()
    if extra:
        clash = sorted(set(extra) & set(payload))
        if clash:
            raise RecordError(
                f"extra provenance may not shadow result fields {clash} -- these carry the "
                f"rollout outcome and must come from the rollout, not the caller"
            )
        payload.update(extra)

    path = dir_ / f"seed_{int(result.seed)}.json"
    _write_json(path, payload)
    return path


def write_meta(dir_: Path, meta: dict) -> Path:
    """Write the run-level `meta.json`, stamping schema version and git sha if absent.

    Both are set only when the caller has not supplied them, so a replayed or synthesised run
    can state its own provenance rather than silently acquiring this process's.
    """
    payload = dict(meta)
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("git_sha", git_sha())
    path = dir_ / "meta.json"
    _write_json(path, payload)
    return path


def _write_json(path: Path, payload: dict) -> None:
    """Atomic write: a reader must never see a truncated result file.

    Results are read by report tooling that may run while a sweep is still going, and a
    half-written seed file parses as a JSONDecodeError that looks like a corrupt run rather
    than a race. Written to a temp name in the same directory, then renamed -- rename is
    atomic within a filesystem, so a reader sees either the old file or the complete new one.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RecordError(
            f"cannot create {path.parent}: {exc.__class__.__name__}. Check that every parent "
            f"directory is traversable and writable."
        ) from None

    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    try:
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, path)
    except (OSError, TypeError, ValueError) as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise RecordError(
            f"cannot write {path}: {exc.__class__.__name__}: {exc}"
        ) from None


def seed_file_names(dir_: Path) -> list[str]:
    """The result files of one task directory, sorted by name; [] when it does not exist.

    The single selection rule: the tally (`read_seeds`) and the eval fingerprint
    (`axis_qvla.bench.sim.campaign.fingerprint`) both read this list, so a file can never count in one
    and not the other. Uses `os.scandir` rather than `Path.glob`: `glob` swallows
    `PermissionError` and would report an unreadable directory as an empty one -- a run that
    scores 0/N instead of failing. Only a missing directory means "no results".
    """
    try:
        with os.scandir(dir_) as entries:
            return sorted(e.name for e in entries if is_seed_file(e.name) and e.is_file())
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise RecordError(
            f"cannot list {dir_}: {exc.__class__.__name__}. Check that every parent "
            f"directory is traversable (e.g. not mode 000)."
        ) from None


def read_seeds(dir_: Path) -> list[dict]:
    """Every result (`seed_<int>.json`, see `seed_file_names`) in a task directory, ordered by seed."""
    records: list[dict] = []
    for name in seed_file_names(dir_):
        path = dir_ / name
        try:
            records.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError) as exc:
            raise RecordError(
                f"cannot read result {path}: {exc.__class__.__name__}. A partial or corrupt "
                f"seed file must not be counted as a failed rollout."
            ) from None

    return sorted(records, key=lambda r: int(r.get("seed", 0)))


def score(records: list[dict]) -> tuple[int, int]:
    """(successes, total). Kept next to the schema so counting cannot drift from writing."""
    return sum(1 for r in records if r.get("success")), len(records)
