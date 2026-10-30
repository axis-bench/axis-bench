"""The seed loop both harnesses share, with provenance that survives a crash.

What varies between rollouts is (a) how a policy produces actions and
(b) which sim domain steps the scene. What does NOT vary is this: resolve the seeds, roll each
one out, write `seed_<n>.json`, tally, write `meta.json`. That invariant part is here, and it
takes the varying part as a `rollout_fn`.

`rollout_fn` is a plain callable rather than a Protocol-with-backend because the two heavy
backends cannot be imported into one process -- the physics path needs the offline environment and
the render path the eval environment. Keeping the seam at a callable means this module, and its
tests, need neither.

PROVENANCE IS WRITTEN TWICE, and that is deliberate. A `meta.json` written only after the last seed
leaves a sweep that dies on seed 14 of 20 with fourteen result files that name no checkpoint, no
server and no settings -- unattributable, therefore unusable, and the GPU-hours are wasted. Here
`meta.json` lands before the first rollout with `complete: false`, and is rewritten at the end with
`complete: true` plus the tally. A reader must treat `complete: false` as a partial run and refuse
to report it as a final number.

Offline tier: stdlib only.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from axis_qvla.bench.sim.record import SeedResult, read_seeds, score, write_meta, write_seed

# A rollout of one seed. Raising is allowed and is not swallowed -- see run_seeds.
RolloutFn = Callable[[int], SeedResult]
# Per-seed provenance that must travel inside the seed file (see record.write_seed).
ExtraFn = Callable[[int], Mapping[str, object]]


class RunnerError(RuntimeError):
    """The sweep could not be set up or completed as described."""


@dataclasses.dataclass(frozen=True)
class RunSpec:
    """One arm's sweep: which task, which seeds, where results go, and what produced them."""

    task_id: int
    arm: str
    seeds: tuple[int, ...]
    log_dir: Path
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.seeds:
            raise RunnerError(
                f"task {self.task_id} ({self.arm}): no seeds to evaluate. A run with zero "
                f"seeds would write 0/0, which is indistinguishable from a run that failed."
            )
        if len(set(self.seeds)) != len(self.seeds):
            dupes = sorted({s for s in self.seeds if list(self.seeds).count(s) > 1})
            raise RunnerError(
                f"task {self.task_id} ({self.arm}): duplicate seeds {dupes}. Each seed writes "
                f"one file, so a repeat silently overwrites its earlier result and shrinks "
                f"the denominator."
            )


def run_seeds(
    spec: RunSpec,
    rollout_fn: RolloutFn,
    *,
    extra_fn: ExtraFn | None = None,
    report: Callable[[str], None] = print,
) -> tuple[int, int]:
    """Roll out every seed, writing results as they land. Returns (successes, total).

    Results are written per seed rather than batched at the end so that a sweep killed by a
    SLURM time limit keeps everything it already paid for.

    A rollout that raises is NOT caught. A backend failure means the sim or the policy server
    is wrong, and continuing would write a column of zeros that reads like a policy outcome. The
    partial results and the `complete: false` meta.json remain on disk for diagnosis.
    """
    try:
        spec.log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RunnerError(
            f"cannot create {spec.log_dir}: {exc.__class__.__name__}. Check that every parent "
            f"directory is traversable and writable."
        ) from None

    base_meta = {
        "task_id": int(spec.task_id),
        "arm": spec.arm,
        "seeds": [int(s) for s in spec.seeds],
        **dict(spec.provenance),
    }
    write_meta(spec.log_dir, {**base_meta, "complete": False})

    n_done = 0
    for seed in spec.seeds:
        result = rollout_fn(int(seed))
        extra = dict(extra_fn(int(seed))) if extra_fn else None
        write_seed(spec.log_dir, result, extra=extra)
        n_done += 1
        report(
            f"  seed {seed}: {result.reason:>10} steps={result.n_steps} "
            f"t={result.sim_time:.1f}s"
        )

    records = read_seeds(spec.log_dir)
    n_success, n_total = score(records)
    write_meta(spec.log_dir, {
        **base_meta,
        "complete": True,
        "n_success": n_success,
        "n_seeds": n_total,
        "success_rate": n_success / n_total if n_total else 0.0,
    })
    pct = 100 * n_success / n_total if n_total else 0.0
    report(
        f"\ntask {spec.task_id} {spec.arm}: {n_success}/{n_total} success "
        f"({pct:.0f}%) -> {spec.log_dir}"
    )
    return n_success, n_total


def resolve_seeds(
    all_seeds: Iterable[int], n_seeds: int | None, only: Iterable[int] | None = None,
    *, seed_set: str = "test",
) -> tuple[int, ...]:
    """First `n_seeds` of a task's declared seeds of `seed_set` (its test or its development
    seeds, whichever `all_seeds` is), in declared order.

    Order is the manifest's, never sorted: the seed list is what the screener produced, and
    reordering would change which seeds a truncated `--n-seeds` run evaluates, making two runs
    that report the same N incomparable.

    `only` selects an explicit SUBSET, which is what lets a work queue hand one task's seeds to
    several workers. It is filtered against the declared list rather than trusted: a seed that the
    manifest never declared has no screened initial state, so evaluating it would silently add an
    off-benchmark condition to the denominator. The returned order stays the manifest's, so a
    subset run and a full run agree seed-for-seed.
    """
    seeds = [int(s) for s in all_seeds]
    if n_seeds is not None:
        if n_seeds <= 0:
            raise RunnerError(f"n_seeds must be positive, got {n_seeds}")
        seeds = seeds[:n_seeds]
    if only is not None:
        wanted = [int(s) for s in only]
        if not wanted:
            raise RunnerError("--seeds was given an empty list; nothing would be evaluated")
        undeclared = sorted(set(wanted) - set(seeds))
        if undeclared:
            raise RunnerError(
                f"seeds {undeclared} are not among this task's declared {seed_set} seeds "
                f"(first {len(seeds)}); a seed with no screened initial state would evaluate an "
                f"off-benchmark condition and still be counted in the denominator"
            )
        keep = set(wanted)
        seeds = [s for s in seeds if s in keep]
    return tuple(seeds)
