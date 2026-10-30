"""Does a re-read of a checkpoint reproduce its published read? A reproducibility check of the harness
on that checkpoint, with a stated tolerance, not a byte check.

Why: rollouts are not bitwise reproducible. The sampler noise of every request is a function of
(task, seed, call index) and XLA autotuning is pinned, but the renderer's multisample colour
resolve differs by +-1 LSB between renders, and a closed loop of up to 1,500 steps can amplify that
into a different outcome for a single rollout. Per-rollout agreement is therefore not the check --
differing rollouts are counted and reported, never tested. What two reads of one
checkpoint on the same seeds must agree on is the success rate, within THIS pair's own evaluation
error: |SR_new - SR_pub| <= 1.96 sqrt(se_pub^2 + se_new^2), each SE the per-task binomial SE of
its read. The tolerance belongs to this pair; it is not a band for any other comparison.

A re-read is reported next to the read it checks. It is never pooled with that read, and never a
training replicate (`contrast` refuses the same checkpoint twice on one side). The check applies to
ONE checkpoint read twice: two files that both name checkpoint content digests and share none are
refused, unless the caller says otherwise (`allow_different_checkpoints`, which the CLI prints).
A file built from a run names its directory's content digest (`eval_digest`); over exactly the
published file set it equals the published `pub_digest`, and a local copy holding more files
differs.

Core tier: stdlib only.
"""
from __future__ import annotations

import dataclasses
import math

from axis_qvla.bench.results.bootstrap import Z95
from axis_qvla.bench.results.checks import check_alike
from axis_qvla.bench.results.format import Results
from axis_qvla.bench.results.score import ArmScore, score

__all__ = ["Z95", "Agreement", "reeval"]
_DIGESTS = ("pub_digest", "eval_digest")


@dataclasses.dataclass(frozen=True)
class Agreement:
    published: ArmScore
    new: ArmScore
    diff: float               # new - published, percentage points
    se: float                 # sqrt(se_pub^2 + se_new^2): this pair's evaluation SE
    z: float | None           # None when both reads are certain (se == 0)
    agrees: bool              # |diff| <= Z95 * se (diff == 0 when se == 0)
    flipped: int = 0          # rollouts whose outcome differs between the reads: reported, not checked
    compared: int = 0         # rollouts both reads hold, over the view's tasks
    same_checkpoint: bool | None = None   # by content digest; None when either file names none
    # False only when the caller allowed it: the files name different checkpoints.

    @property
    def tolerance(self) -> float:
        return Z95 * self.se


def _digests(r: Results) -> set[str]:
    return {str(r.header.checkpoint.get(k)) for k in _DIGESTS if r.header.checkpoint.get(k)}


def reeval(published: Results, new: Results, bench, *, view: str = "all24",
           allow_different_checkpoints: bool = False) -> Agreement:
    da, db = _digests(published), _digests(new)
    same = None if not (da and db) else bool(da & db)
    if same is False and not allow_different_checkpoints:
        raise ValueError(f"not a re-read: {published.header.served} names checkpoint digests "
                         f"{sorted(d[:12] for d in da)} and {new.header.served} {sorted(d[:12] for d in db)}, none "
                         f"in common. The reproducibility check applies only to one checkpoint read twice (a "
                         f"local copy holding files beyond the published set digests differently); pass "
                         f"--allow-different-checkpoints to compare anyway")
    a, b = score(published, bench, view=view), score(new, bench, view=view)
    if a.seed_set != b.seed_set:
        raise ValueError(f"a re-read reads the seeds of the read it checks: {b.served} is a {b.seed_set} read, "
                         f"{a.served} a {a.seed_set} read")
    check_alike([published, new])
    tasks = {t for t, _, _ in a.tallies}
    pub = {(x.task, x.seed): x.success for x in published.rows if x.task in tasks}
    cur = {(x.task, x.seed): x.success for x in new.rows if x.task in tasks}
    both = set(pub) & set(cur)
    flipped = len([k for k in both if pub[k] != cur[k]])
    diff, se = b.pct - a.pct, math.sqrt(a.se ** 2 + b.se ** 2)
    if se == 0.0:
        return Agreement(a, b, diff, se, None, diff == 0.0, flipped, len(both), same)
    z = diff / se
    return Agreement(a, b, diff, se, z, abs(z) <= Z95, flipped, len(both), same)
