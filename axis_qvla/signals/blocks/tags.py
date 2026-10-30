"""π0.7 quality-tag maths: two-level dropout, the dense per-row tag array, and the
task-identity diagnostic.

OFFLINE TIER: numpy only. Never import jax, torch or lerobot here.

Why the dropout is BAKED IN rather than drawn per batch: the sibling schedule arms get their
determinism from an artifact, and a per-batch RNG is reproducible only by argument about RNG
state -- under `num_workers=8` the worker seed is not part of the run record at all. Baking it in
makes the realized dropout rate a number you read off disk.

Why TWO draws and not one: π0.7 drops the whole metadata block at 15% and each component at a
further 5%. With our single component the marginal is 1 - 0.85*0.95 = 0.1925, not 0.15 -- a fact
the paper must state. Both draws are consumed UNCONDITIONALLY so a future second component slots
in without moving the first component's stream.
"""

from __future__ import annotations

import numpy as np

from axis_qvla.signals.blocks.schedule import N_BINS

# π0.7's two levels, kept literally, matching the paper's reported numbers exactly.
DROP_WHOLE = 0.15
DROP_COMPONENT = 0.05
# The marginal with ONE component. Stated as a constant because it is what the paper reports and
# what every rate test compares against; deriving it inline three times invites a 0.85 typo.
TAGGED_FRACTION = (1.0 - DROP_WHOLE) * (1.0 - DROP_COMPONENT)

# Tag codes. 0 and 255 are chosen so neither can be confused with a quintile: `Quality: 0` is not
# a thing the tokenizer would ever see, and 255 is unreachable from `quality_bins`.
NO_TAG = 0            # dropped out -- the bare control prompt, i.e. the unconditional branch
NOT_TRAINABLE = 255   # never sampled; reading one online RAISES (see QualityTags.tag_for_row)


class ConstantQualityError(ValueError):
    """The quality label is constant over the population, so `H(q|task)/H(q)` is 0/0.

    A property of the corpus, not a malformed input: a consumer that accepts a constant-quality
    corpus recognises this refusal by its type. Every other refusal of the diagnostic is a plain
    ValueError."""


def two_level_dropout(n: int, *, seed: int, drop_whole: float = DROP_WHOLE,
                      drop_component: float = DROP_COMPONENT) -> tuple[np.ndarray, np.ndarray]:
    """Per-row `(drop_whole, drop_component)` boolean masks, in flat row order.

    Drawn as an `(n, 2)` block, which numpy fills row-major -- so the stream is exactly
    `rng.random(2n)` with the whole-block draw at even positions and the component draw at odd
    ones. That equivalence is what `test_the_two_levels_are_two_independent_draws_in_stream_order`
    pins, and it is the only way to tell a two-level scheme from a collapsed one after the fact.
    Rates of 0 give pure conditioning: the stream is still consumed, nothing is dropped.
    """
    for name, p in (("drop_whole", drop_whole), ("drop_component", drop_component)):
        if not 0.0 <= float(p) < 1.0:
            raise ValueError(f"{name} must be in [0, 1), got {p}")
    u = np.random.default_rng(int(seed)).random((int(n), 2))
    return u[:, 0] < drop_whole, u[:, 1] < drop_component


def dense_tags(
    episode_rows: list[np.ndarray],
    q_ep: np.ndarray,
    *,
    n_rows: int,
    seed: int,
    drop_whole: float = DROP_WHOLE,
    drop_component: float = DROP_COMPONENT,
) -> tuple[np.ndarray, dict]:
    """The dense `uint8 (n_rows,)` tag array plus the realized-rate block that describes it.

    Dense over the CONCATENATED CORPUS row space, not the trainable subset: the online side then
    looks a tag up by flat row index with no re-derivation of the index space. The schedule
    artifacts already record that trap; this one repeats the same choice for the same reason.
    """
    q = np.asarray(q_ep, dtype=np.int64)
    if len(episode_rows) != q.size:
        raise ValueError(
            f"{len(episode_rows)} episode row blocks but {q.size} quality bins; the two come from "
            f"one enumeration and a mismatch means they describe different episode sets"
        )
    if q.size and (q.min() < 1 or q.max() > N_BINS):
        raise ValueError(
            f"quality bin outside [1, {N_BINS}]: min={q.min()} max={q.max()}. A 0 would collide "
            f"with NO_TAG and silently turn a tagged row into an untagged one."
        )
    lengths = [len(b) for b in episode_rows]
    q_per_row = np.repeat(q, lengths).astype(np.uint8)
    return _dense_from_q_per_row(episode_rows, q_per_row, n_rows=n_rows, seed=seed,
                                 drop_whole=drop_whole, drop_component=drop_component)


def dense_tags_rowwise(
    episode_rows: list[np.ndarray],
    row_bins: list[np.ndarray],
    *,
    n_rows: int,
    seed: int,
    drop_whole: float = DROP_WHOLE,
    drop_component: float = DROP_COMPONENT,
) -> tuple[np.ndarray, dict]:
    """`dense_tags` with a PER-ROW bin instead of a broadcast per-episode one.

    `row_bins` is one int array per episode, parallel to `episode_rows` and the same length as
    each -- the shape `index_schedule.delta_quality_bins` returns, and using that function's
    output here is the point: the tag then conditions on exactly the row-level quality unit
    `delta_quality_bins` ranks, so the episode-vs-segment comparison is a granularity ablation and
    not also a second quality definition. The dropout stream is shared with `dense_tags` (same
    length, same seed), so the two granularities tag and drop EXACTLY the same rows and differ
    only in WHICH bin a tagged row carries.
    """
    if len(episode_rows) != len(row_bins):
        raise ValueError(
            f"{len(episode_rows)} episode row blocks but {len(row_bins)} bin blocks; the two come "
            f"from one enumeration and a mismatch means they describe different episode sets"
        )
    for i, (rows, bins) in enumerate(zip(episode_rows, row_bins)):
        if len(rows) != len(np.asarray(bins)):
            raise ValueError(
                f"episode {i} has {len(rows)} rows but {len(np.asarray(bins))} bins; a slipped "
                f"block would tag every later row with its neighbour's quality"
            )
    q_per_row = (
        np.concatenate([np.asarray(b, dtype=np.int64) for b in row_bins])
        if row_bins else np.empty(0, dtype=np.int64)
    )
    if q_per_row.size and (q_per_row.min() < 1 or q_per_row.max() > N_BINS):
        raise ValueError(
            f"quality bin outside [1, {N_BINS}]: min={q_per_row.min()} max={q_per_row.max()}. A 0 "
            f"would collide with NO_TAG and silently turn a tagged row into an untagged one."
        )
    return _dense_from_q_per_row(
        episode_rows, q_per_row.astype(np.uint8), n_rows=n_rows, seed=seed,
        drop_whole=drop_whole, drop_component=drop_component,
    )


def _dense_from_q_per_row(
    episode_rows: list[np.ndarray],
    q_per_row: np.ndarray,
    *,
    n_rows: int,
    seed: int,
    drop_whole: float = DROP_WHOLE,
    drop_component: float = DROP_COMPONENT,
) -> tuple[np.ndarray, dict]:
    """Shared tail of the two granularities: flat-index guards, ONE dropout stream, the tag
    array, and the realized-rate block. Kept in one place so the stream can never diverge
    between them -- a granularity ablation where the two arms also drew different dropout masks
    would be two experiments wearing one name."""
    flat = np.concatenate(episode_rows) if episode_rows else np.empty(0, dtype=np.int64)
    if flat.size and not np.all(np.diff(flat) > 0):
        raise ValueError(
            "episode row blocks are not strictly ascending across the pretraining data. The dropout stream "
            "is consumed in flat row order, so a re-ordered enumeration yields a DIFFERENT "
            "artifact from the same seed -- reproducibility that quietly is not."
        )
    if flat.size and (flat.min() < 0 or flat.max() >= int(n_rows)):
        raise ValueError(
            f"row {flat.max() if flat.size else -1} is outside the pretraining data of {n_rows} rows; the "
            f"tag array would be built against other pretraining data than the loader indexes"
        )

    whole, component = two_level_dropout(int(flat.size), seed=seed, drop_whole=drop_whole,
                                         drop_component=drop_component)
    dropped = whole | component

    tag = np.full(int(n_rows), NOT_TRAINABLE, dtype=np.uint8)
    tag[flat] = np.where(dropped, np.uint8(NO_TAG), q_per_row)

    n = max(int(flat.size), 1)
    meta = {
        "realized_drop_whole": float(whole.sum()) / n,
        "realized_drop_component": float(component.sum()) / n,
        "realized_tagged_fraction": float((~dropped).sum()) / n,
        "bin_row_counts": {
            str(b): int((tag[flat] == b).sum()) for b in range(1, N_BINS + 1)
        },
        "n_trainable": int(flat.size),
    }
    return tag, meta


def q_given_task_entropy_ratio(q_ep: np.ndarray, task_ids: np.ndarray, *, log2=np.log2) -> float:
    """`H(q | task) / H(q)` in bits, plug-in estimator over episodes.

    -> 1 means the tag is orthogonal to task identity; -> 0 means the tag IS the task. The
    diagnostic exists because a CFG gain from a tag that is a task label in disguise would be
    extra task capacity, not a quality signal -- and it is invisible in every loss curve, every
    guard and every coverage number, because CFG's sampling is provably identical to the control.

    Plug-in, deliberately: the bias-corrected readings (Miller-Madow and a permutation null) are
    computed separately. What belongs
    in an artifact is the census figure for the population the artifact describes, computed the
    same way every time so two artifacts are comparable.
    """
    q = np.asarray(q_ep).ravel()
    t = np.asarray(task_ids).ravel()
    if q.size != t.size:
        raise ValueError(f"{q.size} quality bins but {t.size} task ids")
    if q.size == 0:
        raise ValueError("no episodes: the entropy ratio is undefined")

    def _entropy(counts: np.ndarray) -> float:
        p = counts[counts > 0] / counts.sum()
        return float(-(p * log2(p)).sum())

    _, q_codes = np.unique(q, return_inverse=True)
    h_q = _entropy(np.bincount(q_codes))
    if h_q == 0.0:
        raise ConstantQualityError(
            "H(q) is zero: the quality label is constant over this population, so H(q|task)/H(q) "
            "is 0/0. Returning 0.0 would read as 'the tag is pure task identity', the opposite of "
            "what a constant tag means."
        )
    h_cond = 0.0
    for tid in np.unique(t):
        sel = t == tid
        h_cond += (sel.sum() / q.size) * _entropy(np.bincount(q_codes[sel]))
    return h_cond / h_q
