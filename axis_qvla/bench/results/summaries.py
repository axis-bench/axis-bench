"""Read a campaign root's arm_summary.json files into one table.

`bench score` is a READER, not a pooler: one row per arm as the driver wrote it, with the
two digests that say what was measured (the eval fingerprint over every seed file, and the
benchmark declaration's digest) so two rows can be compared by content. Pooling reads across
(checkpoint, protocol) and the paper tables are `axis_qvla.paper` (`axis_qvla.cli paper tables`).

Core tier: stdlib only.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

from axis_qvla.bench.sim.arms import FAILED_SUMMARY_FILE, OUT_PREFIX, SUMMARY_FILE


class ScoreError(RuntimeError):
    """The run root cannot be listed, or a summary cannot be read."""


@dataclasses.dataclass(frozen=True)
class ArmRow:
    arm: str
    served_arm: str
    all24_pct: float | None
    rollouts: int
    successes: int
    fingerprint: str
    benchmark_sha: str
    matches_benchmark: bool
    status: str      # "ok", "incomplete", "errored", or "FAILED" (0 rollouts)
    path: Path
    seed_set: str = "test"   # "dev" for a development read; a summary that predates the key is a test read


def _row(path: Path) -> ArmRow:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ScoreError(f"cannot read {path}: {exc.__class__.__name__}: {exc}") from None
    bench = d.get("benchmark") or {}
    # The driver stamped `manifest_sha256` before the benchmark package existed and `sha256`
    # (the declaration's digest) since; both are the same file's digest.
    sha = bench.get("sha256") or bench.get("manifest_sha256") or ""
    n = int(d.get("rollouts") or 0)
    if path.name == FAILED_SUMMARY_FILE or n == 0:
        status = "FAILED"
    elif d.get("errored_rollouts"):
        status = "errored"
    elif d.get("incomplete_tasks"):
        status = "incomplete"
    else:
        status = "ok"
    return ArmRow(
        arm=str(d.get("arm", path.parent.name)), served_arm=str(d.get("served_arm", "")),
        all24_pct=float(d["all24_pct"]) if d.get("all24_pct") is not None else None,
        rollouts=n, successes=int(d.get("successes") or 0),
        fingerprint=str(d.get("eval_fingerprint") or ""), benchmark_sha=str(sha),
        matches_benchmark=bool(bench.get("matches_manifest", False)),
        status=status, path=path, seed_set=str(d.get("seed_set") or "test"),
    )


def read_summaries(run_root: Path, arms: list[str] | None = None) -> list[ArmRow]:
    """One row per `<run_root>/out_<arm>/arm_summary.json` (or `.FAILED.json`), sorted by arm.

    `os.scandir`, never `glob`: glob swallows PermissionError and would report an unreadable
    campaign root as "no arms", which reads as nothing evaluated rather than nothing readable.
    """
    rows: list[ArmRow] = []
    wanted = set(arms) if arms else None
    try:
        with os.scandir(run_root) as entries:
            dirs = [Path(e.path) for e in entries if e.is_dir() and e.name.startswith(OUT_PREFIX)]
    except OSError as exc:
        raise ScoreError(f"cannot list {run_root}: {exc.__class__.__name__}") from None
    for d in sorted(dirs):
        arm = d.name[len(OUT_PREFIX):]
        if wanted is not None and arm not in wanted:
            continue
        for name in (SUMMARY_FILE, FAILED_SUMMARY_FILE):
            p = d / name
            try:
                present = p.is_file()
            except OSError as exc:
                raise ScoreError(f"cannot access {p}: {exc.__class__.__name__}") from None
            if present:
                rows.append(_row(p))
                break
    if wanted is not None:
        missing = sorted(wanted - {r.arm for r in rows})
        if missing:
            raise ScoreError(f"no arm_summary.json under {run_root} for: {', '.join(missing)}")
    return rows


def format_table(rows: list[ArmRow]) -> str:
    """Fixed-width text; `*` after the benchmark digest marks a task list that did NOT match
    the declaration, so a custom run cannot sit unmarked beside a real one, and the `seeds` column
    says which seed set each run read (a development read is never a test number)."""
    head = (f"{'run':22s} {'served':30s} {'all24%':>7s} {'n':>5s} {'seeds':5s} {'status':10s} "
            f"{'fingerprint':13s} {'benchmark':13s}")
    lines = [head]
    for r in rows:
        pct = f"{r.all24_pct:.1f}" if r.all24_pct is not None else "-"
        mark = "" if r.matches_benchmark else "*"
        lines.append(
            f"{r.arm:22s} {r.served_arm:30s} {pct:>7s} {r.rollouts:>5d} {r.seed_set:5s} {r.status:10s} "
            f"{r.fingerprint[:12]:13s} {(r.benchmark_sha[:12] + mark):13s}"
        )
    if any(not r.matches_benchmark for r in rows):
        lines.append("* task list did not match the tracked benchmark declaration")
    return "\n".join(lines)
