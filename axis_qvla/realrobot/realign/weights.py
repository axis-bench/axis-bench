"""AWR weights from phase curves: the rank (unbalanced) file and the phase-balanced file.

OFFLINE TIER: numpy only.

`build_awr_weights --reward phase` does two things: it FORMS the phase curve from the corpus's own
columns (object binding, hand distance, contacts) and then WEIGHTS it. REALALIGN's curves come from
the recordings (phase.py) or export-keyed curves, so only the second half applies, and it is reused
function by function:

- the RANK file (the ranking/binning input every tag and filter derives from) is the pinned path
  without `--phase-balanced`: `build_episode_weights` over the curve;
- the BALANCED file (the AWR runs) is the pinned `--phase-balanced` path: `phase_delta_clamped`,
  `_neutralised_mask_clamped`, then one corpus-level per-phase quantile match.

The two-pass block in `balanced_weights` mirrors `build_awr_weights.main` statement for statement:
that file is pinned (`conf/signals/generators.json`) and cannot be refactored without a full
rebuild of every locked artifact. `tests/unit/realrobot/realign/test_weights.py` runs both on one fixture and
requires identical bytes and identical scales, so the mirror cannot drift without failing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from axis_qvla.signals.blocks.awr import (
    DEFAULT_CAP, DEFAULT_HORIZON, DEFAULT_TAU, AwrWeightError, EpisodeWeights, awr_weight,
    build_episode_weights, episode_key, neutral_episode, write_manifest,
)
from axis_qvla.signals.methods.awr_chunk import _neutralised_mask_clamped, phase_delta_clamped
from axis_qvla.realrobot.realign.curves import PhaseCurve

REWARD_ID = "phase"
BINDING = "curves"  # enters the formula fingerprint: these weights were not formed from columns

# The pinned builder's per-phase fit-pool minimum (axis_qvla.signals.methods.awr_chunk). A smaller corpus -- a
# few rendered tasks -- passes min_pool explicitly; the value is then stamped into the scales, so an
# artifact built under a weaker guard says so. The default adds no stamp.
MIN_FIT_POOL = 10_000


def _ordered(curves: dict[tuple[int, int], PhaseCurve]):
    return sorted(curves.items(), key=lambda kv: (int(kv[0][0]), int(kv[0][1])))


def rank_weights(curves: dict[tuple[int, int], PhaseCurve], *, tau: float = DEFAULT_TAU,
                 cap: float = DEFAULT_CAP, horizon: int = DEFAULT_HORIZON) -> list[EpisodeWeights]:
    """Unbalanced weights: unclamped forward Delta, invalid windows neutral, neutral episodes all 1.0."""
    out = []
    for (tid, ep), c in _ordered(curves):
        if c.t0 is None:
            out.append(neutral_episode(int(tid), int(ep), len(c.curve)))
        else:
            out.append(build_episode_weights(int(tid), int(ep), np.asarray(c.curve, float), c.valid,
                                             horizon=horizon, tau=tau, cap=cap))
    return out


def balanced_weights(curves: dict[tuple[int, int], PhaseCurve], *, tau: float = DEFAULT_TAU,
                     cap: float = DEFAULT_CAP, horizon: int = DEFAULT_HORIZON, min_pool: int = MIN_FIT_POOL
                     ) -> tuple[list[EpisodeWeights], dict]:
    """Phase-balanced weights and the scales stamped into their provenance (pinned `--phase-balanced`)."""
    thr = float(np.log(cap) / tau)
    slots: list[tuple[tuple[int, int], tuple | None]] = []
    pool1, pool2 = [], []
    for (tid, ep), c in _ordered(curves):
        if c.t0 is None:
            slots.append(((int(tid), int(ep)), None))
            continue
        t0 = int(c.t0)
        d, ok = phase_delta_clamped(np.asarray(c.curve, float), t0, horizon)
        nm = _neutralised_mask_clamped(c.valid, t0, horizon)
        fit = ok & ~nm
        pool1.append(d[:t0][fit[:t0]])
        pool2.append(d[t0:][fit[t0:]])
        slots.append(((int(tid), int(ep)), (d, nm, t0)))
    if not pool1:
        raise AwrWeightError("no episode carries a contact onset; there is nothing to balance")
    p1, p2 = np.concatenate(pool1), np.concatenate(pool2)
    if min(p1.size, p2.size) < min_pool:
        raise AwrWeightError(
            f"per-phase fit pools too small (p1={p1.size} p2={p2.size}); the branch "
            f"taxonomy has collapsed one phase and a quantile fit would be noise")
    if not (np.isfinite(p1).all() and np.isfinite(p2).all()):
        raise AwrWeightError("non-finite delta in the scale-fit pool")
    target = float(np.mean(np.concatenate([p1, p2]) >= thr))
    if not 0.0 < target < 0.5:
        raise AwrWeightError(f"degenerate pooled capped rate {target:.4f}")
    q1 = float(np.quantile(p1, 1 - target))
    q2 = float(np.quantile(p2, 1 - target))
    if min(q1, q2) <= 1e-9 or not np.isfinite(q1 + q2):
        raise AwrWeightError(f"degenerate per-phase quantiles q1={q1} q2={q2}")
    sc1, sc2 = thr / q1, thr / q2
    scales = {"mode": "clamped-lennorm+quantile-match", "cap_threshold": thr,
              "target_capped_rate": target, "q1": q1, "q2": q2,
              "scale1": sc1, "scale2": sc2}
    if min_pool != MIN_FIT_POOL:
        scales["min_fit_pool"] = int(min_pool)
    eps = []
    for (tid, ep), item in slots:
        if item is None:
            eps.append(neutral_episode(tid, ep, len(curves[(tid, ep)].curve)))
            continue
        d, nm, t0 = item
        d = d.copy()
        d[:t0] *= sc1
        d[t0:] *= sc2
        w = np.where(nm, 1.0, awr_weight(d, tau, cap)).astype(np.float32)
        if not np.isfinite(w).all():
            raise AwrWeightError(f"non-finite balanced weight in task_{tid}--{ep}")
        eps.append(EpisodeWeights(key=episode_key(tid, ep), weights=w, n_neutral=int(nm.sum())))
    return eps, scales


def write_weights(path: Path, eps: list[EpisodeWeights], *, corpus: Path, balanced: bool,
                  scales: dict | None = None, tau: float = DEFAULT_TAU, cap: float = DEFAULT_CAP,
                  horizon: int = DEFAULT_HORIZON) -> dict:
    """The pinned artifact format, stamped with the corpus and with `binding="curves"`."""
    if bool(balanced) != (scales is not None):
        raise ValueError("phase-balanced weights must carry their scales, and only they may")
    header = write_manifest(Path(path), iter(eps), corpus=Path(corpus), reward_id=REWARD_ID, tau=tau,
                            cap=cap, horizon=horizon, binding=BINDING, phase_balanced=bool(balanced))
    if scales is not None:
        payload = json.loads(Path(path).read_text())
        payload["provenance"]["phase_balance"] = scales
        Path(path).write_text(json.dumps(payload))
    return header
