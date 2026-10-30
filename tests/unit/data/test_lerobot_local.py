"""Local LeRobot v3.0 signal extraction (offline venv).

Pins the two properties the ranges build silently depends on: raw 9-D and converted
DROID-8D datasets must produce the SAME idle signal, and episode splitting must refuse
a file whose rows are not grouped by episode rather than emit plausible garbage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from axis_qvla.data.lerobot_local import DatasetSignals, state_to_idle_signal


def _write_dataset(root, state, episode_index):
    d = root / "data" / "chunk-000"
    d.mkdir(parents=True)
    pd.DataFrame(
        {
            "observation.state": list(np.asarray(state, dtype=np.float32)),
            "episode_index": np.asarray(episode_index, dtype=np.int64),
        }
    ).to_parquet(d / "file-000.parquet")
    return root


def _raw_state(n, rng):
    """9-D raw state: 7 arm joints in rad + two finger widths in metres."""
    arm = np.cumsum(rng.normal(scale=0.01, size=(n, 7)), axis=0)
    width = np.linspace(0.04, 0.012, n)[:, None]
    return np.concatenate([arm, width, width], axis=1)


def test_raw_and_droid8d_states_give_the_same_signal():
    """The converter only rescales the gripper, so ranges must not depend on which
    copy of a dataset was read."""
    rng = np.random.default_rng(0)
    raw = _raw_state(200, rng)
    sig = state_to_idle_signal(raw)
    assert sig.shape == (200, 8)
    # Feeding the 8-D result back is a no-op -- that is what makes the two paths agree.
    assert np.array_equal(state_to_idle_signal(sig), sig)
    assert np.allclose(sig[:, :7], raw[:, :7])


def test_gripper_channel_is_normalised_to_unit_range():
    rng = np.random.default_rng(1)
    sig = state_to_idle_signal(_raw_state(200, rng))
    assert sig[:, 7].min() == pytest.approx(0.0, abs=1e-6)
    assert sig[:, 7].max() == pytest.approx(1.0, abs=1e-6)


def test_frozen_gripper_falls_back_to_arm_only():
    """A dataset whose gripper never moves cannot be calibrated; dropping the channel
    beats crashing a corpus-wide build."""
    rng = np.random.default_rng(2)
    raw = _raw_state(50, rng)
    raw[:, 7:] = 0.04
    assert state_to_idle_signal(raw).shape == (50, 7)


def test_rejects_unexpected_state_width():
    with pytest.raises(ValueError, match="state"):
        state_to_idle_signal(np.zeros((10, 5)))


def test_episodes_split_on_episode_index(tmp_path):
    rng = np.random.default_rng(3)
    ep = np.repeat([0, 1, 2], [10, 20, 30])
    _write_dataset(tmp_path, _raw_state(60, rng), ep)
    got = [(i, len(s)) for i, s in DatasetSignals.load(tmp_path).episodes()]
    assert got == [(0, 10), (1, 20), (2, 30)]


def test_unordered_episode_index_raises(tmp_path):
    """Silently accepting a shuffled file would yield ranges that look fine and are wrong."""
    rng = np.random.default_rng(4)
    _write_dataset(tmp_path, _raw_state(30, rng), np.repeat([1, 0, 1], 10))
    with pytest.raises(ValueError, match="non-decreasing"):
        list(DatasetSignals.load(tmp_path).episodes())


def test_missing_data_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        DatasetSignals.load(tmp_path)
