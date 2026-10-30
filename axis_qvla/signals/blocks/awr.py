"""Per-frame AWR weights for the pretraining corpus, one file per reward variant.

WHY THIS IS A SEPARATE OFFLINE PASS. AWR (WVM Eq E.7) weights each sampled row by
`w = min(exp(tau * Delta), cap)`, where `Delta` is a forward difference of a reward curve. The
reward is a geometric quantity computed from traces and MuJoCo contact; the trainer runs in the
online (jax) venv and cannot import any of that. So the weights are materialised here and read
there.

KEYED BY EPISODE, NEVER BY FLAT ROW INDEX. The trainer's flat index space is a function of the
task order AND every episode length across ~476 sub-datasets. A weight file keyed by flat index
silently goes wrong if one task or episode is added, dropped, or if the builder sorts task ids as
STRINGS ("1000" < "800") while the trainer sorts them as ints. Nothing downstream would report
it -- the run would simply apply the wrong weight to the wrong frame. Keying by
`task_<id>--<episode_index>`, the same key `pretrain_sampler.episode_key` builds, makes the
alignment checkable and lets the trainer resolve rows and weights in ONE loop.

NEUTRAL, NEVER SILENT. Any frame whose weight cannot be honestly computed gets EXACTLY 1.0 --
the vanilla-BC weight -- and is counted in the manifest. A weight of 0 would drop the frame from
the objective while leaving it in the batch, which is a training-set change disguised as a
weighting; a missing key would let the reader's default decide, which is how a silent zero
happens. Both are worse than saying "this frame is not reweighted".

MASKED ON `reward_valid`. Past the clip point the stored reward is held constant (not covered by
the reward curve) while the geometry keeps moving, so an advantage window touching such a row
carries no reward information. It is neutralised to the vanilla weight, identically for every
reward variant.

Offline tier: numpy/pandas/pyarrow. Run under the offline environment.
"""
from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

NEUTRAL = 1.0
DEFAULT_TAU = 10.0
DEFAULT_CAP = 2.0
DEFAULT_HORIZON = 10


class AwrWeightError(ValueError):
    """Weights could not be built as asked, or were asked for in a way that hides a bug."""


def episode_key(task_id: int, episode_index: int) -> str:
    """`task_<id>--<episode>`, matching `openpi.training.pretrain_sampler.episode_key`.

    Duplicated deliberately rather than imported: that module lives in the online tier and this
    one must run under the offline venv. A test pins the two against each other so the
    duplication cannot drift.
    """
    return f"task_{int(task_id)}--{int(episode_index)}"


def forward_delta(reward: np.ndarray, horizon: int = DEFAULT_HORIZON) -> np.ndarray:
    """WVM Eq E.4 on ONE episode: `Delta_t = V(min(t + H - 1, T - 1)) - V(t)`.

    One value per row, not per window: the trainer needs a weight for every sampled row, and the
    `T - H + 1` window-start form would leave the last H-1 rows of every
    episode unmapped.

    MUST be called per episode. A task parquet holds many episodes concatenated, so running this
    over a whole file computes advantages BETWEEN unrelated trajectories at every boundary.
    """
    r = np.asarray(reward, dtype=np.float64)
    if r.ndim != 1:
        raise AwrWeightError(f"reward must be 1-D, got shape {r.shape}")
    t = np.arange(len(r))
    return r[np.minimum(t + horizon - 1, len(r) - 1)] - r[t]


def awr_weight(delta: np.ndarray, tau: float = DEFAULT_TAU, cap: float = DEFAULT_CAP) -> np.ndarray:
    """WVM Eq E.7 pre-renormalisation: `w = min(exp(tau * Delta), cap)`."""
    return np.minimum(np.exp(tau * np.asarray(delta, dtype=np.float64)), cap).astype(np.float32)


def neutralise_invalid(
    weights: np.ndarray, valid: np.ndarray, horizon: int = DEFAULT_HORIZON
) -> np.ndarray:
    """Set w = 1.0 on every row whose advantage window touches an invalid reward row.

    Not just the invalid rows themselves: `Delta_t` reads `r[t + H - 1]`, so a VALID row whose
    horizon lands in the held-constant tail has no reward information in its advantage either.
    """
    w = np.array(weights, dtype=np.float32, copy=True)
    v = np.asarray(valid, dtype=bool)
    if len(w) != len(v):
        raise AwrWeightError(f"weights ({len(w)}) and valid ({len(v)}) differ in length")
    n = len(w)
    t = np.arange(n)
    head = np.minimum(t + horizon - 1, n - 1)
    # a window is trustworthy only if BOTH ends are valid; the interior cannot rescue it because
    # Delta only reads the two ends.
    bad = ~v | ~v[head]
    w[bad] = NEUTRAL
    return w


def neutralised_mask(valid: np.ndarray, horizon: int = DEFAULT_HORIZON) -> np.ndarray:
    """The rows `neutralise_invalid` would mask: the definition of 'neutral'.

    Counting neutral rows as `w == 1.0` conflates two different things: a row masked because its
    advantage could not be trusted, and a row whose advantage is genuinely zero (a flat reward over
    the horizon gives `exp(tau * 0) = 1.0`). On plateau-heavy curves the second is common, so the
    neutral fraction counts the masked rows only.
    """
    v = np.asarray(valid, dtype=bool)
    n = len(v)
    head = np.minimum(np.arange(n) + horizon - 1, n - 1)
    return ~v | ~v[head]


@dataclasses.dataclass(frozen=True)
class EpisodeWeights:
    key: str
    weights: np.ndarray
    n_neutral: int

    @property
    def n(self) -> int:
        return len(self.weights)


def build_episode_weights(
    task_id: int,
    episode_index: int,
    reward: np.ndarray,
    valid: np.ndarray,
    *,
    horizon: int = DEFAULT_HORIZON,
    tau: float = DEFAULT_TAU,
    cap: float = DEFAULT_CAP,
) -> EpisodeWeights:
    """Weights for one episode from its own reward curve."""
    d = forward_delta(reward, horizon)
    w = neutralise_invalid(awr_weight(d, tau, cap), valid, horizon)
    if not np.isfinite(w).all():
        raise AwrWeightError(f"non-finite weight in {episode_key(task_id, episode_index)}")
    return EpisodeWeights(
        key=episode_key(task_id, episode_index),
        weights=w,
        # count the MASK, not the value: a weight of exactly 1.0 can also mean "advantage is
        # genuinely zero", which is not neutralisation and must not be reported as such.
        n_neutral=int(neutralised_mask(valid, horizon).sum()),
    )


def neutral_episode(task_id: int, episode_index: int, length: int) -> EpisodeWeights:
    """An episode whose reward could not be formed: every row at vanilla-BC weight."""
    return EpisodeWeights(
        key=episode_key(task_id, episode_index),
        weights=np.full(int(length), NEUTRAL, dtype=np.float32),
        n_neutral=int(length),
    )


def manifest_payload(episodes: Iterator[EpisodeWeights], *, reward_id: str, tau: float, cap: float, horizon: int,
                     corpus: Path | None = None, corpus_fingerprint: str | None = None,
                     **extra_formula_params) -> dict:
    """`{key: [w, ...]}` plus the provenance needed to tell two arms apart.

    The header is not decoration: two runs that differ only in `reward_id` are the whole
    experiment, and a weights file that does not say which reward produced it cannot be
    attributed to an arm after the fact.
    """
    out: dict[str, list[float]] = {}
    total = neutral = 0
    for ep in episodes:
        if ep.key in out:
            raise AwrWeightError(f"duplicate episode key {ep.key}")
        out[ep.key] = [float(x) for x in ep.weights]
        total += ep.n
        neutral += ep.n_neutral
    payload = {"version": 1, "reward_id": reward_id, "tau": tau, "cap": cap, "horizon": horizon,
               "n_episodes": len(out), "n_rows": total, "n_neutral_rows": neutral, "weights": out}
    # STAMP the corpus and the formula: nothing else could tell whether the corpus had moved under
    # the weights or whether the maths had changed. extra_formula_params (binding, phase_balanced)
    # enter the FORMULA FINGERPRINT: two artifacts computed by different maths never share one.
    if corpus is not None or corpus_fingerprint is not None:
        from axis_qvla.signals.provenance import stamp

        stamp(payload, corpus=None if corpus is None else Path(corpus), fingerprint=corpus_fingerprint,
              reward_id=reward_id, tau=tau, cap=cap, horizon=horizon, **extra_formula_params)
    return payload


def write_manifest(path: Path, episodes: Iterator[EpisodeWeights], *, reward_id: str,
                   tau: float, cap: float, horizon: int, corpus: Path | None = None,
                   **extra_formula_params) -> dict:
    """Write `manifest_payload` to `path`; returns its header."""
    payload = manifest_payload(episodes, reward_id=reward_id, tau=tau, cap=cap, horizon=horizon, corpus=corpus,
                               **extra_formula_params)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return {k: v for k, v in payload.items() if k != "weights"}
