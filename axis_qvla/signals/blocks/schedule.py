"""Sampling maths for the filtered index schedules, and the quality ranking CFG conditions on.

OFFLINE TIER: numpy only. Never import jax, torch or lerobot here.

Why a PRECOMPUTED schedule rather than a runtime sampler: the experiment must report samples-seen
and unique-frames-seen per arm, and must resume byte-exactly. A stateful sampler makes both of
those an argument about RNG state; an artifact makes them a fact you can read off disk.

Quality is a GLOBAL quintile -- `quality_bins` over episodes, `delta_quality_bins` over rows --
not a per-task rank; see `quality_bins`' docstring for why per-task ranking was rejected. The
row-level ranking is the unit `axis_qvla.signals.methods.cfg_tags` conditions on; `draw_schedule` itself
only permutes or filters rows, so a schedule's `realized_draws_per_bin` REPORTS this ranking
rather than drawing from it.
"""

from __future__ import annotations

import math

import numpy as np

# Terminal q5:q1 ODDS of 4:1, range-matched to the reweight runs' 2.0/0.5 weight ratio -- the
# calibration of the weighted-draw mode some released schedule files were written with. No mode here
# reads these: they are plain constants because `axis_qvla.signals.legacy.schedule_config_hash`
# reproduces the config_hash of those files, and that hash is a function of these exact values. Do
# not repurpose them for a new mode without checking that reproduction first.
LAMBDA_MAX = math.log(4.0) / 4.0
RAMP_START_FRAC = 0.85
NEUTRAL_Q = 3
N_BINS = 5


def quality_bins(progress_ratio: np.ndarray, unlabelled: np.ndarray) -> np.ndarray:
    """Global quintile (1..N_BINS) of the episode progress-time ratio.

    `progress_ratio` is the share of an episode's trainable rows that make forward progress,
    read as `w > 1` on the stored float32 weight: that is Delta > 2^-24/tau (about 6e-9 at tau 10),
    not Delta > 0, because positive float noise below that rounds to w == 1.0 and counts as no
    progress (see `build_index_schedule.episode_stats_from_weights`). It is an ABSOLUTE measure, so
    unlike a within-task rank it is comparable across tasks -- which is what lets a task with a
    single episode receive an honest quality rather than being ranked against itself and forced to
    the bottom bin.

    Equal ratios are NOT averaged: a stable sort splits a tie that straddles a bin edge by corpus
    (task, episode) order, which keeps the artifact deterministic.

    Unlabelled episodes get exactly NEUTRAL_Q and are excluded from the ranking, so they neither
    receive nor consume a quantile slot.
    """
    q = np.full(len(progress_ratio), NEUTRAL_Q, dtype=np.int64)
    sel = ~unlabelled
    n = int(sel.sum())
    if n == 0:
        return q
    vals = progress_ratio[sel]
    order = np.argsort(vals, kind="stable")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n)
    # rank -> 1..N_BINS; ties break by stable order, which keeps the artifact deterministic
    q[sel] = np.minimum(N_BINS, 1 + (ranks * N_BINS) // max(n, 1))
    return q


def delta_quality_bins(delta_rows: list[np.ndarray]) -> list[np.ndarray]:
    """Global quintile (1..N_BINS) of the PER-ROW advantage, one array per episode.

    The sample-level counterpart of `quality_bins`, and the unit the CFG quality tags condition on
    (`axis_qvla.signals.methods.cfg_tags.row_quality_bins`). Binning a per-EPISODE score and
    broadcasting it to that episode's rows makes the ranking a statement about trajectories, while
    most of the weight variance lies WITHIN episodes rather than between them. WVM's advantage (Eq E.4) is
    defined per sample anchored at a frame, so a quality tag conditioning on it should rank the
    same unit the reweighting arm weights.

    Ranked, not thresholded. The reward curve is piecewise-constant, so `Delta` carries large atoms
    (a flat horizon gives exactly 0.0); a `>= percentile` test lands ON an atom and admits every tie
    with it, so the kept share can exceed the stated percentile. Ranking keeps the bin populations
    exact whatever the ties do.

    Ties break by (value, episode, offset) via a stable sort, so a rebuild reproduces the artifact.
    """
    if not delta_rows:
        return []
    lengths = [len(np.asarray(d)) for d in delta_rows]
    flat = np.concatenate([np.asarray(d, dtype=np.float64).ravel() for d in delta_rows]) \
        if any(lengths) else np.empty(0, dtype=np.float64)
    n = flat.size
    if n == 0:
        return [np.empty(0, dtype=np.int64) for _ in delta_rows]
    if not np.isfinite(flat).all():
        raise ValueError(
            "delta_quality_bins got a non-finite advantage; ranking would place NaN arbitrarily "
            "and silently tilt toward whichever bin it landed in"
        )
    order = np.argsort(flat, kind="stable")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n)
    bins = np.minimum(N_BINS, 1 + (ranks * N_BINS) // n)
    out, start = [], 0
    for ln in lengths:
        out.append(bins[start:start + ln])
        start += ln
    return out


def random_fraction_mask(
    row_w: list[np.ndarray], keep_frac: float, seed: int
) -> list[np.ndarray]:
    """Keep a UNIFORMLY RANDOM `keep_frac` of rows: the matched-count control of `drop_top`.

    `drop_top` keeps a fraction of the rows; this arm holds the row count fixed and varies ONLY
    the selection rule, which is the comparison that isolates the reward's ranking.

    The kept COUNT is `round(keep_frac * total)` -- computed identically to `top_fraction_mask`, and
    a test pins the two to the same cardinality.

    `row_w` is taken (rather than plain lengths) so the call site cannot accidentally pass a
    different row set than the one `top_fraction_mask` ranked; the VALUES are deliberately ignored.
    """
    if not 0.0 < keep_frac <= 1.0:
        raise ValueError(f"keep_frac must be in (0, 1], got {keep_frac}")
    lengths = [len(np.asarray(w)) for w in row_w]
    total = int(sum(lengths))
    if total == 0:
        return [np.zeros(n, dtype=bool) for n in lengths]
    n_keep = int(round(keep_frac * total))
    rng = np.random.default_rng(seed)
    keep_flat = np.zeros(total, dtype=bool)
    keep_flat[rng.choice(total, size=n_keep, replace=False)] = True
    out, start = [], 0
    for n in lengths:
        out.append(keep_flat[start:start + n])
        start += n
    return out


def broadcast_episode_bins(q_ep: np.ndarray, episode_rows: list[np.ndarray]) -> list[np.ndarray]:
    """Per-episode bins repeated across each episode's rows: the episode-level counterpart of
    `delta_quality_bins`'s per-row ranking.

    Kept as a tested, reusable function rather than re-derived by hand at each call site that wants
    an episode-level quality broadcast to rows.
    """
    if len(q_ep) != len(episode_rows):
        raise ValueError(
            f"q_ep has {len(q_ep)} entries but there are {len(episode_rows)} episodes; they must be "
            f"1:1 by position"
        )
    return [np.full(len(rows), int(q), dtype=np.int64) for q, rows in zip(q_ep, episode_rows)]


# `drop` IS WVM Filtered-BC, Eq E.6 at kappa = 0.0: a PER-ROW keep of `Delta >= 0`, applied when
# `episode_rows` is built. The name is kept deliberately even though "Filtered BC" is what the
# paper calls it -- `meta["mode"]` is written into every schedule artifact on disk and checked
# against `DataConfig.pretrain_expected_mode`, so renaming the mode would make every existing
# artifact unloadable until rebuilt, for no change in behaviour.
#
# READ IT AS FILTERED-BC, NOT AS AN EPISODE-LEVEL RULE. Next to `quality_labels.DROP_WHOLE` (which
# really does drop whole episodes, for the CFG tier) the name invites exactly that misreading; it
# has never dropped an episode. `episode_stats_from_weights` derives its keep mask as `w >= 1.0`,
# and since `w = min(exp(tau*Delta), cap)` is monotone and stored as float32, that is `Delta >= 0`
# per row up to float noise (w == 1.0 absorbs |tau*Delta| below ~2^-25..2^-24) -- cap or no cap.
MODES = ("vanilla", "drop", "drop_top", "drop_random")

# Modes whose supervision is a row-level keep/drop of the drawable set rather than a draw order.
FILTER_MODES = ("drop", "drop_top", "drop_random")

# WVM's percentile Filtered-BC keeps the top 70% (Eq E.6). That cut is nearly INERT here: 75.7% of
# this corpus's rows are already advantaged, so "top 70%" and "Delta >= 0" select almost the same
# rows and the two arms would not be distinguishable.
#
# 0.50, not WVM's 0.70: rows at the weight cap share one value and cannot be ranked among
# themselves, and 0.50 sits outside that tied group while keeping the two reward variants' kept
# sets separable.
#
# The fraction is a build-time parameter: state it in the arm's spec rather than relying on the
# default, because changing this default changes what a flagless build means.
DEFAULT_KEEP_TOP_FRAC = 0.50


def top_fraction_mask(row_w: list[np.ndarray], keep_frac: float) -> list[np.ndarray]:
    """Boolean keep masks retaining the top `keep_frac` of ROWS ranked by weight, corpus-wide.

    GLOBAL, not per task. WVM ranks "over the training set" and reports one threshold per suite
    (Eq E.6, Table E.5), so this follows the paper. Ranking by `w` is ranking by the Delta the
    weights were built from -- for the approach-aware arm that is the per-phase SCALED Delta --
    except inside the atoms the float32 weight introduces: the cap, and w == 1.0, which also holds
    rows whose Delta is float noise. Passed episode-constant weights (episode filtering), it ranks
    the MAPPED episode weight, which is monotone in P_e; episodes sharing one weight keep corpus
    order, and the boundary episode keeps its leading rows.

    A GLOBAL cut can in principle empty a task whose advantages are uniformly small; re-measure
    with `task_retention.py` if the corpus or the fraction changes to confirm no task is lost
    outright, since that is a property of the corpus and the fraction, not of this function.

    Ranked, not thresholded, for the reason spelled out in `delta_quality_bins`: `Delta` has a large
    atom at exactly 0, and a `>= quantile` test admits every row tied with the cut point rather than
    exactly `keep_frac` of them. Ties break by (-w, episode, offset), so a rebuild reproduces the
    same corpus.
    """
    if not 0.0 < keep_frac <= 1.0:
        raise ValueError(f"keep_frac must be in (0, 1], got {keep_frac}")
    lengths = [len(np.asarray(w)) for w in row_w]
    total = int(sum(lengths))
    if total == 0:
        return [np.zeros(n, dtype=bool) for n in lengths]
    flat = np.concatenate([np.asarray(w, dtype=np.float64).ravel() for w in row_w])
    if not np.isfinite(flat).all():
        raise ValueError(
            "top_fraction_mask got a non-finite weight; a NaN compares False against every "
            "threshold and would be dropped silently rather than refused"
        )
    n_keep = int(round(keep_frac * total))
    # argsort on -w puts the largest first; 'stable' makes the tie order (episode, offset).
    order = np.argsort(-flat, kind="stable")
    keep_flat = np.zeros(total, dtype=bool)
    keep_flat[order[:n_keep]] = True
    out, start = [], 0
    for n in lengths:
        out.append(keep_flat[start:start + n])
        start += n
    return out


def _permutation_batches(flat: np.ndarray, rng: np.random.Generator, batch: int):
    """Endless `batch`-sized slices of a shuffled pass over `flat`, re-permuted when exhausted.

    This is `openpi.training.slb_variant_sampler.RowSampler.__iter__` in numpy: one
    `permutation` per pass, consumed in order, a fresh permutation for the next pass -- i.e.
    WITHOUT replacement, which is what the plain-BC run trained on, so every arm covers the
    frames as the control does (with replacement, ~37% of the distinct frames at this step
    budget would go unseen).

    `RowSampler(rows, seed=...)` defaults `num_samples` to `len(rows)`, so every epoch is exactly
    one permutation and the DataLoader supplies the next pass by re-iterating, which is the loop
    below; `RowSampler`'s `num_samples > len(rows)` tail-fill (`torch.randint`) does not occur. Every
    pass is a permutation, also for arms whose budget exceeds one pass (both drop arms: 1.21 and
    1.16 epochs).
    """
    pool = np.empty(0, dtype=np.int64)
    cursor = 0
    while True:
        chunk = np.empty(batch, dtype=np.int64)
        filled = 0
        while filled < batch:
            if cursor >= pool.size:
                pool = flat[rng.permutation(flat.size)]
                cursor = 0
            take = min(batch - filled, pool.size - cursor)
            chunk[filled:filled + take] = pool[cursor:cursor + take]
            cursor += take
            filled += take
        yield chunk


def draw_schedule(
    episode_rows: list[np.ndarray],
    *,
    mode: str,
    total_steps: int,
    batch: int,
    seed: int,
) -> np.ndarray:
    """The exact rows every step will see, as int64 (total_steps, batch).

    `vanilla` and `drop` are the same draw over different row sets -- the filtering happens when
    `episode_rows` is built, so this function never needs to know the keep rule. Every surviving
    mode draws a PERMUTATION over its drawable rows per pass (`_permutation_batches`): uniform,
    without replacement, so an arm covers its drawable set exactly as the control does and differs
    from it in supervision alone, never in which frames it sees.

    This function does not rank rows by quality -- that ranking (`delta_quality_bins`,
    `broadcast_episode_bins`) is reported in a schedule's meta block and consumed by the CFG
    quality tags, but no current mode draws from it.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    if not episode_rows or all(len(r) == 0 for r in episode_rows):
        raise ValueError("no trainable rows: every episode is empty")

    rng = np.random.default_rng(seed)
    nonempty = [i for i, r in enumerate(episode_rows) if len(r)]
    flat = np.concatenate([episode_rows[i] for i in nonempty])

    stream = _permutation_batches(flat, rng, batch)
    out = np.empty((total_steps, batch), dtype=np.int64)
    for t in range(total_steps):
        out[t] = next(stream)
    return out
