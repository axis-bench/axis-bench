"""The approach-aware contact proxy of float32 distances is the portable float32 tanh, on every host.

The corpus pose columns are float32, so `c = 1 - tanh(5 d)` of every paper episode is a float32
tanh. numpy computes that with its SIMD kernel where the CPU has FMA and with the C library's tanhf
where it has not, 1-3 ulp apart, and the locked approach-aware weights were written by the kernel
(axis_qvla.signals.numerics). A float32 distance must therefore never reach `np.tanh`; other
dtypes (synthetic float64 data) keep it, uncast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from axis_qvla.signals.methods import awr_chunk as baw
from axis_qvla.signals.numerics import tanh_f32

META = {"body_names": ["plate_0", "cup_0"], "object_body": "cup_0"}


def _episode(T=60, tc=30):
    d = np.concatenate([np.linspace(0.40, 0.05, tc), np.full(T - tc, 0.05)])
    p = np.concatenate([np.zeros(tc), np.linspace(0.1, 0.9, T - tc)])
    r = 0.8 * p + 0.2 * (1.0 - np.tanh(5.0 * d))
    tgt = np.zeros(T, bool)
    tgt[tc:] = True
    return r, d, tgt


def _spy_on_numpy_tanh(monkeypatch) -> list:
    seen, real = [], np.tanh

    def spy(x, *args, **kwargs):
        seen.append(np.asarray(x).dtype)
        return real(x, *args, **kwargs)

    monkeypatch.setattr(np, "tanh", spy)
    return seen


def test_a_float32_distance_never_reaches_numpy_tanh(monkeypatch):
    r, d, tgt = _episode()
    seen = _spy_on_numpy_tanh(monkeypatch)
    curve, t0 = baw.banded_phase_curve(r, d.astype(np.float32), tgt)
    assert seen == []
    baw.banded_phase_curve(r, d, tgt)
    assert seen == [np.dtype(np.float64)]           # float64 keeps numpy's tanh, uncast


def test_the_float32_curve_is_the_kernels_curve():
    r, d, tgt = _episode()
    d32 = d.astype(np.float32)
    curve, t0 = baw.banded_phase_curve(r, d32, tgt)
    c = 1.0 - tanh_f32(baw.K * d32)
    progress = np.clip((r - baw.W_CONTACT * c) / baw.W_PROGRESS, 0.0, 1.0)
    head = 1.0 - baw.LAMBDA - baw.BONUS
    assert t0 == 30
    np.testing.assert_array_equal(curve[30:], baw.LAMBDA + baw.BONUS + head * np.clip(progress - progress[30], 0, 1)[30:])


def test_float32_pose_columns_give_float32_distances(monkeypatch):
    """The corpus stores `observation.body_pose` / `observation.eef_pose` as list<float>: the
    distance the names binding computes from them is float32, as in the paper's generator."""
    r, d, tgt = _episode()
    T = len(r)
    obj = np.tile([0.0, 0, 0, 1, 0, 0, 0], (T, 1))
    other = np.tile([0.5, 0.5, 0, 1, 0, 0, 0], (T, 1))
    bp = np.concatenate([other, obj], axis=1).astype(np.float32)
    eef = np.zeros((T, 7), np.float32)
    eef[:, 0], eef[:, 3] = d, 1.0
    g = pd.DataFrame({"reward": r, "reward_valid": np.ones(T, bool), "contact_target_finger": tgt,
                      "contact_target_eef": np.zeros(T, bool), "observation.body_pose": list(bp),
                      "observation.eef_pose": list(eef)})
    seen = _spy_on_numpy_tanh(monkeypatch)
    curve, t0 = baw.phase_reward(g, META)
    assert seen == [] and t0 == 30
    np.testing.assert_array_equal(curve, baw.banded_phase_curve(r, d.astype(np.float32), tgt)[0])
