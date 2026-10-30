"""The approach-aware phase curve (Eq. 3) and its edge branches; the AWR builder reads this module's curve."""
from __future__ import annotations

import numpy as np
import pandas as pd

from axis_qvla.labels import phase, reward
from axis_qvla.signals.methods import awr_chunk


def test_the_awr_builder_and_the_labels_share_one_phase_curve():
    assert awr_chunk.banded_phase_curve is phase.banded_phase_curve
    assert awr_chunk.phase_reward is phase.phase_reward and awr_chunk.first_contact is phase.first_contact
    assert (phase.W_PROGRESS, phase.W_CONTACT, phase.K) == (reward.W_PLACE_V2, reward.W_CONTACT_V2, reward.CONTACT_TANH_K)


def _episode(t0, T=60, d0=0.4, dc=0.05):
    d = np.concatenate([np.linspace(d0, dc, max(t0, 1)), np.full(T - max(t0, 1), dc)])
    p = np.concatenate([np.zeros(max(t0, 1)), np.linspace(0.1, 0.9, T - max(t0, 1))])
    r = phase.W_PROGRESS * p + phase.W_CONTACT * (1.0 - np.tanh(phase.K * d))
    tgt = np.zeros(T, bool)
    if t0 < T:
        tgt[t0:] = True
    return r, d, tgt


def test_stable_contact_mid_episode_gives_two_bands():
    r, d, tgt = _episode(30)
    curve, t0 = phase.banded_phase_curve(r, d, tgt)
    assert t0 == 30 and curve[29] <= phase.LAMBDA < curve[30]


def test_contact_before_the_edge_is_all_manipulation():
    r, d, tgt = _episode(phase.EDGE_FRAMES - 1)
    assert phase.banded_phase_curve(r, d, tgt)[1] == 0


def test_no_contact_is_all_approach_and_no_approach_is_unscored():
    r, d, _ = _episode(30)
    never = np.zeros(len(r), bool)
    assert phase.banded_phase_curve(r, d, never)[1] == len(r)
    flat = np.full(len(r), 0.3)
    assert phase.banded_phase_curve(r, flat, never) is None
    short = 0.3 - np.linspace(0, phase.MIN_APPROACH_M / 2, len(r))
    assert phase.banded_phase_curve(r, short, never) is None


def test_contact_shorter_than_the_minimum_run_is_not_stable():
    mask = np.zeros(20, bool)
    mask[5:5 + phase.MIN_RUN - 1] = True
    assert phase.first_contact(mask, phase.MIN_RUN) is None
    mask[12:12 + phase.MIN_RUN] = True
    assert phase.first_contact(mask, phase.MIN_RUN) == 12


def test_palm_contact_counts_as_target_contact():
    frames = pd.DataFrame({"contact_target_finger": [False, False, True], "contact_target_eef": [False, True, False]})
    assert phase.target_contact(frames).tolist() == [False, True, True]


def test_mid_episode_contact_after_too_short_an_approach_is_all_manipulation():
    """The hand began engaged: contact at frame 30 after closing in by 1 cm, less than MIN_APPROACH_M."""
    r, d, tgt = _episode(30, d0=0.05 + phase.MIN_APPROACH_M / 2)
    curve, t0 = phase.banded_phase_curve(r, d, tgt)
    assert t0 == 0 and curve[0] == phase.LAMBDA + phase.BONUS and curve.min() >= phase.LAMBDA + phase.BONUS


def test_contact_only_at_the_tail_is_all_approach():
    """Stable contact that starts after T - EDGE_FRAMES is no phase boundary: the episode is its
    approach, normalised by the closest approach."""
    T = 60
    r, d, tgt = _episode(T - phase.MIN_RUN, T=T)
    assert phase.first_contact(tgt, phase.MIN_RUN) == T - phase.MIN_RUN > T - phase.EDGE_FRAMES
    curve, t0 = phase.banded_phase_curve(r, d, tgt)
    assert t0 == T and curve[0] == 0.0 and curve.max() == phase.LAMBDA
