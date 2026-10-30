"""Validation of two declaration fields, `compatible_declarations` and `dev_seeds`.

`compatible_declarations` lists older declarations whose results still score here, each with a
reason (the declaration the paper's results are stamped with); `dev_seeds` declares development
seeds that must never touch a test seed. Their checks live here so spec.py stays the loader.
BenchmarkError is defined here and re-exported by spec.py, so `axis_qvla.bench.spec.BenchmarkError`
is the same class for every caller.

Core tier: stdlib only.
"""
from __future__ import annotations

import re
from pathlib import Path

_HEX64 = re.compile(r"[0-9a-f]{64}")


class BenchmarkError(ValueError):
    """The declaration is missing, malformed, or internally inconsistent."""


def _is_int(x: object) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def parse_compatible(raw: object, where: Path) -> dict[str, str]:
    """`compatible_declarations`: [{sha256: 64 lower-case hex, reason: non-empty}], no repeats."""
    if raw is None:
        return {}
    if not isinstance(raw, list):
        raise BenchmarkError(f"{where}: compatible_declarations must be a list of {{sha256, reason}}")
    out: dict[str, str] = {}
    for e in raw:
        sha, why = (e.get("sha256"), e.get("reason")) if isinstance(e, dict) else (None, None)
        if not (isinstance(sha, str) and _HEX64.fullmatch(sha)):
            raise BenchmarkError(f"{where}: compatible_declarations entry needs a 64-hex sha256: {e!r:.90}")
        if not (isinstance(why, str) and why.strip()) or sha in out:
            raise BenchmarkError(f"{where}: compatible_declarations entry {sha[:16]} has no reason "
                                 f"or is repeated")
        out[sha] = why
    return out


def parse_dev(raw: object, where: Path) -> dict:
    """`dev_seeds` as read; its values are checked when seeds are asked for (dev_seed_tuple)."""
    if raw is not None and not isinstance(raw, dict):
        raise BenchmarkError(f"{where}: dev_seeds must be an object {{offset, n}}")
    return dict(raw or {})


def dev_seed_tuple(name: str, dev: dict, task_id: int, test_seeds: tuple[int, ...]) -> tuple[int, ...]:
    """tid*1000 + offset + i for i < n. Malformed or test-overlapping dev_seeds raise
    BenchmarkError: a dev run on a test seed tunes on the test set, and a bare KeyError would not
    say which file is wrong."""
    off, n = dev.get("offset"), dev.get("n")
    if not dev:
        raise BenchmarkError(f"{name}: no dev_seeds declared")
    if not (_is_int(off) and _is_int(n) and off >= 0 and n > 0):
        raise BenchmarkError(f"{name}: dev_seeds must be {{offset: int >= 0, n: int > 0}}, got {dev!r}")
    seeds = tuple(int(task_id) * 1000 + off + i for i in range(n))
    hit = sorted(set(seeds) & set(test_seeds))
    if hit:
        raise BenchmarkError(f"{name}: dev_seeds for task {task_id} overlap its test seeds "
                             f"({len(hit)} shared, first {hit[0]})")
    return seeds


EXECUTION_SCHEMA = "qvla.execution/1"
_EXEC_INT = ("fps", "replan", "max_steps", "n_stable_frames")
_EXEC_KEYS = {"schema", *_EXEC_INT, "noise_salt", "render", "grip_state", "eef_ik", "table_texture"}


def parse_execution(raw: object, where: Path) -> dict:
    """execution.json: the constants a rollout executes, each a declared value, nothing implied."""
    if not isinstance(raw, dict) or raw.get("schema") != EXECUTION_SCHEMA:
        raise BenchmarkError(f"{where}: not a {EXECUTION_SCHEMA} document")
    unknown = sorted(set(raw) - _EXEC_KEYS)
    missing = sorted(_EXEC_KEYS - set(raw))
    if unknown or missing:
        raise BenchmarkError(f"{where}: unknown keys {unknown}, missing keys {missing}")
    for k in _EXEC_INT:
        if not (_is_int(raw[k]) and raw[k] > 0):
            raise BenchmarkError(f"{where}: {k} must be a positive integer, got {raw[k]!r}")
    if not (_is_int(raw["noise_salt"]) and raw["noise_salt"] >= 0):
        raise BenchmarkError(f"{where}: noise_salt must be a non-negative integer")
    r = raw["render"]
    if not (isinstance(r, dict) and set(r) == {"width", "height"} and all(_is_int(v) and v > 0 for v in r.values())):
        raise BenchmarkError(f"{where}: render must be {{width, height}}, positive integers")
    for k in ("grip_state", "eef_ik"):
        if not (isinstance(raw[k], str) and raw[k]):
            raise BenchmarkError(f"{where}: {k} must be a non-empty string")
    if raw["table_texture"] is not None and not isinstance(raw["table_texture"], str):
        raise BenchmarkError(f"{where}: table_texture must be null or a string")
    return dict(raw)
