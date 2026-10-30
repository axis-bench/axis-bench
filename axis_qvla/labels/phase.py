"""The approach-aware phase curve of Eq. (3): approach progress, then manipulation progress.

For an episode whose task object is bound (its pose column named by `object_body` in `body_names`),
the curve rises from 0 to LAMBDA while the hand closes on the object, and from LAMBDA + BONUS to 1 as
the manipulation progresses after stable contact -- at least MIN_RUN frames of target contact by a
finger OR the hand/palm. Edge branches instead of abort gates: stable contact in
[EDGE_FRAMES, T - EDGE_FRAMES] after an approach of at least MIN_APPROACH_M gives the two bands;
earlier contact, or no measurable approach, is all manipulation from frame 0; contact never or only
at the tail is all approach, normalised by the closest approach; no approach and no contact anchor
returns None and the AWR scorer declares the episode unscorable. Manipulation progress is backed out
of the stored reward of Eq. (1) with Eq. (1)'s own weights.

The contact term 1 - tanh(K * d) of a float32 distance -- every corpus episode's, since the corpus
stores its poses as float32 -- is the portable float32 tanh (axis_qvla.signals.numerics.tanh_f32),
so the curve, and the approach-aware weights built from it, are the same bytes on every CPU; any
other dtype keeps `np.tanh`, uncast.

Offline tier: numpy, pandas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd  # noqa: F401  (the moved functions annotate their frame table with it)

# Eq. (1)'s own weights and contact sharpness: the curve backs progress out of the reward they built.
from axis_qvla.labels.reward import CONTACT_TANH_K as K, W_CONTACT_V2 as W_CONTACT, W_PLACE_V2 as W_PROGRESS
from axis_qvla.signals.numerics import tanh_f32

LAMBDA, BONUS, MIN_RUN, EPS = 0.5, 0.10, 3, 1e-6
EDGE_FRAMES = 5             # contact this close to either end is not a mid-episode phase boundary
MIN_APPROACH_M = 0.02       # an approach shorter than this (m) is no approach
TARGET_CONTACT = ("contact_target_finger", "contact_target_eef")    # finger OR hand/palm


def target_contact(frames) -> np.ndarray:
    """Per-frame contact with the task object by a finger or the hand (palm)."""
    return frames[TARGET_CONTACT[0]].to_numpy(bool) | frames[TARGET_CONTACT[1]].to_numpy(bool)


def first_contact(mask: np.ndarray, min_run: int) -> int | None:
    if not mask.any():
        return None
    d = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    ok = [a for a, b in zip(s, e) if b - a >= min_run]
    return int(ok[0]) if ok else None


def _contact_tanh(x: np.ndarray) -> np.ndarray:
    """`np.tanh` for the contact proxy, with a float32 argument computed by the portable kernel.

    The corpus pose columns are float32, so `tanh(K * d)` of every corpus episode is a float32
    tanh. numpy computes that with its own SIMD kernel on a CPU with FMA and with the C library's
    tanhf on one without, which rounds 44.6% of the corpus's inputs 1-3 ulp differently and moves
    the approach-aware weights; the locked weights were written by the kernel. `tanh_f32` is that
    kernel on any host (axis_qvla.signals.numerics.tanh_f32). Any other dtype, such as
    synthetic float64 distances, keeps `np.tanh` and is never cast: no released file is computed
    from one.
    """
    x = np.asarray(x)
    return tanh_f32(x) if x.dtype == np.float32 else np.tanh(x)


def banded_phase_curve(r: np.ndarray, d_ho: np.ndarray,
                       tgt: np.ndarray) -> tuple[np.ndarray, int] | None:
    """The two-phase curve for a BOUND object, with edge BRANCHES instead of abort gates.

    Returns `(curve, t0_eff)` where `t0_eff` is the band boundary the advantage machinery must
    use (0 = all-manipulation, len(r) = all-approach). The branches:

      * stable contact mid-episode  -> the standard two bands
      * contact at/near the start (or no measurable approach: the hand begins engaged, e.g. a
        cut segment)                -> ALL-MANIPULATION from frame 0, gain measured from
        progress(0). A failed manipulation then scores LOW instead of being silenced.
      * contact only at the very end, or never  -> ALL-APPROACH: the episode's content is how
        it approached; normalised by the closest approach achieved.
      * no approach and no contact anchor at all -> None (honest neutral).

    The +0.10 bonus separates the bands; advantage windows never cross t0_eff, so it shapes
    serving-time readability, not weights.
    """
    T = len(r)
    t0 = first_contact(tgt, MIN_RUN)
    contact = 1.0 - _contact_tanh(K * d_ho)
    progress = np.clip((r - W_CONTACT * contact) / W_PROGRESS, 0.0, 1.0)
    head = 1.0 - LAMBDA - BONUS

    def approach_curve(denom):
        return LAMBDA * np.clip((d_ho[0] - d_ho) / denom, 0.0, 1.0)

    def manip_curve(base):
        return LAMBDA + BONUS + head * np.clip(progress - base, 0.0, 1.0)

    if t0 is not None and EDGE_FRAMES <= t0 <= T - EDGE_FRAMES:
        denom = float(d_ho[0] - d_ho[t0])
        if denom >= MIN_APPROACH_M:
            out = np.empty(T)
            out[:t0] = approach_curve(denom)[:t0]
            out[t0:] = manip_curve(progress[t0])[t0:]
            return out, t0
        # contact fine but the hand started engaged: all-manipulation
        return manip_curve(progress[0]), 0
    if t0 is not None and t0 < EDGE_FRAMES:
        return manip_curve(progress[0]), 0
    # contact never (or only at the tail): all-approach, if any approach happened
    denom = float(d_ho[0] - d_ho.min())
    if denom >= MIN_APPROACH_M:
        return approach_curve(denom), T
    return None


def phase_reward(g: pd.DataFrame, meta_row) -> tuple[np.ndarray, int] | None:
    """The contact-phase reward for ONE episode via the NAMES binding, or None."""
    r = g["reward"].to_numpy(float)
    tgt = target_contact(g)
    names = meta_row.get("body_names")
    obj = meta_row.get("object_body")
    if names is None or obj is None:
        return None
    names = list(names)
    if obj not in names:
        return None
    oi = names.index(obj)
    bp = np.stack(g["observation.body_pose"].to_numpy())
    nb = bp.shape[1] // 7
    if oi >= nb:
        return None
    B = bp.reshape(len(g), nb, 7)
    hand = np.stack(g["observation.eef_pose"].to_numpy())[:, :3]
    d_ho = np.linalg.norm(B[:, oi, :3] - hand, axis=1)
    return banded_phase_curve(r, d_ho, tgt)
