"""The names-bound phase reward's two edge branches, on synthetic episodes whose truth is known by
construction: contact from the first frames scores the whole episode as manipulation, and an
episode whose contact never fires is scored on its approach.

Offline tier: numpy/pandas only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from axis_qvla.signals.methods.awr_chunk import phase_reward


def _episode(n=120, t0=40, n_bodies=3, grasped=1, hand_start=(0.6, 0.0, 0.4)):
    """A synthetic episode: hand approaches body `grasped`, contacts at t0, carries it away.

    Bodies other than `grasped` stay put. Returns (frame df, body positions (T,B,3))."""
    rng = np.random.default_rng(0)
    T, B = n, n_bodies
    pos0 = np.array([[0.1 * (i + 1), 0.1 * i, 0.02] for i in range(B)])
    bodies = np.tile(pos0[None, :, :], (T, 1, 1)).astype(float)
    hand = np.zeros((T, 3))
    hand[0] = hand_start
    # approach: linear to the grasped body by t0
    for t in range(t0):
        a = t / max(t0 - 1, 1)
        hand[t] = (1 - a) * np.asarray(hand_start) + a * pos0[grasped]
    # carry: hand + object move together after t0
    for t in range(t0, T):
        d = 0.004 * (t - t0)
        hand[t] = pos0[grasped] + [d, d, 0.1 * min(1, (t - t0) / 20)]
        bodies[t, grasped] = hand[t] - [0, 0, 0.01]
    quat = np.tile(np.array([1.0, 0, 0, 0]), (T, B, 1))
    bp = np.concatenate([bodies, quat], axis=2).reshape(T, B * 7)
    contact = np.zeros(T, bool)
    contact[t0:] = True
    reward = np.clip(np.linspace(0, 1, T) + rng.normal(0, 0.005, T), 0, 1)
    df = pd.DataFrame({
        "reward": reward,
        "reward_valid": np.ones(T, bool),
        "contact_target_finger": contact,
        "contact_target_eef": np.zeros(T, bool),
        "observation.body_pose": [bp[t] for t in range(T)],
        "observation.eef_pose": [np.concatenate([hand[t], [1, 0, 0, 0]]) for t in range(T)],
    })
    return df


NAMES = ["cup_0", "cup_1", "plate_0"]


def test_branch_start_engaged_is_all_manipulation():
    # Contact from (nearly) frame 0: a cut segment beginning at the object. The episode must
    # get an all-manipulation curve from frame 0, not a neutral abstention.
    g = _episode(grasped=1, t0=2)
    res = phase_reward(g, {"body_names": NAMES, "object_body": "cup_1"})
    assert res is not None
    curve, t0_eff = res
    assert t0_eff == 0
    assert curve.min() >= 0.6 - 1e-9


def test_branch_no_contact_is_all_approach():
    # Contact never fires: the demo is scored on its approach, normalised by the closest
    # approach achieved -- not silenced.
    g = _episode(grasped=1)
    g = g.copy()
    g["contact_target_finger"] = np.zeros(len(g), bool)
    g["contact_target_eef"] = np.zeros(len(g), bool)
    res = phase_reward(g, {"body_names": NAMES, "object_body": "cup_1"})
    assert res is not None
    curve, t0_eff = res
    assert t0_eff == len(g)
    assert curve.max() <= 0.5 + 1e-9
    assert curve.max() >= 0.4  # the approach genuinely sweeps the band
