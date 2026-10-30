"""axis_qvla.data.convert_droid_actions: raw 9-D LeRobot state to DROID-8D state and joint-velocity actions."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from axis_qvla.data.convert_droid_actions import _stack, convert_dataset, gripper_closedness, joint_velocity, to_droid8d_frame

_STAT_KEYS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")


def test_stack_matches_the_naive_per_row_cast():
    """`_stack` is the fast path for a list-of-vector column; it must stay equivalent
    to stacking row by row, including the float32 -> float64 widening."""
    rows = np.random.default_rng(0).random((64, 9)).astype(np.float32)
    got = _stack(pd.Series(list(rows)))
    assert got.dtype == np.float64
    assert np.array_equal(got, np.stack([np.asarray(r, dtype=np.float64) for r in rows]))


def test_stack_of_an_empty_column_is_empty():
    assert _stack(pd.Series([], dtype=object)).shape == (0, 0)


def test_joint_velocity_per_episode_resets():
    # 2 episodes of 3 frames; arm dim 7 all equal to a ramp on dim 0
    arm = np.zeros((6, 7))
    arm[:, 0] = [0.0, 0.1, 0.2,   1.0, 1.3, 1.6]
    ep = np.array([0, 0, 0, 1, 1, 1])
    v = joint_velocity(arm, ep, fps=15.0)
    assert v.shape == (6, 7)
    # within ep0: (0.1-0)*15=1.5, (0.2-0.1)*15=1.5, last=0
    assert np.allclose(v[:3, 0], [1.5, 1.5, 0.0])
    # within ep1: (1.3-1.0)*15=4.5, (1.6-1.3)*15=4.5, last=0
    assert np.allclose(v[3:, 0], [4.5, 4.5, 0.0])


def test_gripper_closedness_open_closed_endpoints():
    open_w, closed_w = 0.04, 0.0
    # internal width = ½(g0+g1); pick g1=0 so width=½g0 -> g0=2*target_width
    g0 = np.array([0.08, 0.0, 0.04])  # internal widths 0.04(open), 0.0(closed), 0.02(mid)
    g1 = np.zeros_like(g0)
    c = gripper_closedness(g0, g1, open_w, closed_w)
    assert np.isclose(c[0], 0.0)   # open -> 0
    assert np.isclose(c[1], 1.0)   # closed -> 1
    assert 0.0 < c[2] < 1.0


# to_droid8d_frame works on a DataFrame, not a dataset directory: the materializer builds one task's
# frames in memory and must emit DROID-8D state and action without a second pass over the disk.
# Same maths as convert_dataset.

def test_to_droid8d_frame_produces_8d_state_and_action():
    n = 6
    rng = np.random.default_rng(0)
    arm = np.cumsum(rng.normal(size=(n, 7)) * 0.01, axis=0)          # smooth arm trajectory
    fingers = np.stack([np.linspace(0.04, 0.0, n), np.linspace(0.04, 0.0, n)], axis=1)  # closing
    state9 = np.hstack([arm, fingers]).astype(np.float32)
    df = pd.DataFrame({
        "observation.state": list(state9),
        "action": list(state9),                      # action mirrors state here (position targets)
        "episode_index": [0, 0, 0, 1, 1, 1],
        "reward": np.arange(n, dtype=np.float32),     # a baked column that must pass through untouched
    })
    out = to_droid8d_frame(df, fps=15.0)
    S = np.stack(out["observation.state"].values); A = np.stack(out["action"].values)
    assert S.shape == (n, 8) and A.shape == (n, 8)
    # state arm part is unchanged; gripper part is closedness in [0,1]
    np.testing.assert_allclose(S[:, :7], arm, atol=1e-5)
    assert (S[:, 7] >= 0).all() and (S[:, 7] <= 1).all()
    # action arm part is joint VELOCITY (forward diff * fps), zeroed at each episode's last frame
    assert np.allclose(A[2, :7], 0.0) and np.allclose(A[5, :7], 0.0), "episode-last velocity must be 0"
    np.testing.assert_allclose(A[0, :7], (arm[1] - arm[0]) * 15.0, atol=1e-4)
    # gripper closes over time -> closedness increases
    assert S[-1, 7] > S[0, 7]
    # baked columns survive
    assert list(out["reward"]) == list(df["reward"])


def test_to_droid8d_frame_gripper_calibration_is_whole_dataframe():
    # A single glitch frame must not blow up the scale (99/1 quantiles, not min/max).
    n = 100
    arm = np.zeros((n, 7), np.float32)
    w = np.full((n, 2), 0.02, np.float32); w[50] = 999.0   # one glitch
    df = pd.DataFrame({"observation.state": list(np.hstack([arm, w]).astype(np.float32)),
                       "action": list(np.hstack([arm, w]).astype(np.float32)),
                       "episode_index": [0] * n})
    out = to_droid8d_frame(df, fps=15.0)
    S = np.stack(out["observation.state"].values)
    assert np.isfinite(S).all() and (S[:, 7] <= 1).all() and (S[:, 7] >= 0).all()


def test_state_width_is_a_checked_contract_not_an_index():
    """9-D in, 8-D out; anything else fails with a message that names the layout.

    A LIBERO render batch declares a 9-D state where earlier derived corpora carry 8-D, so a build
    can genuinely be handed either. Without this check, an 8-D input would reach `st[:, 8]` and
    raise a bare numpy IndexError naming neither the column nor the corpus -- and an auto-build
    over hundreds of tasks turns that into one unreadable traceback per task.

    An 8-D input is the interesting case: it is an ALREADY-CONVERTED corpus, so the error has to
    say "do not run this twice" rather than "bad width".
    """
    def frame(width, n=30):
        rng = np.random.default_rng(0)
        st = rng.normal(size=(n, width)).astype(np.float32)
        if width > 7:
            st[:, 7:] = np.abs(st[:, 7:]) * 0.02
        return pd.DataFrame({"observation.state": list(st), "action": list(st),
                             "episode_index": np.zeros(n, int), "frame_index": np.arange(n)})

    out = to_droid8d_frame(frame(9), fps=15.0)
    assert np.vstack(out["observation.state"]).shape[1] == 8
    assert np.vstack(out["action"]).shape[1] == 8

    for bad in (7, 8):
        with pytest.raises(ValueError, match="9-D layout"):
            to_droid8d_frame(frame(bad), fps=15.0)


# convert_dataset writes an on-disk dataset: meta/info.json, meta/stats.json and
# meta/episodes/*.parquet, not just the data parquet. Rewriting info.json's
# observation.state/action shapes to [8] while leaving the two stats files at the source's
# 9-element values makes LeRobot's normaliser crash at the first training step, comparing an 8-D
# batch to a 9-D stats tensor (normalize_processor.py:362). These tests pin that all three agree:
# meta/stats.json and meta/episodes carry the SAME 8-D shape info.json declares, with values
# recomputed from the converted rows, and every other feature's stats -- and meta/stats.json's
# formatting -- left untouched.

def _direct_stats(arr: np.ndarray) -> dict[str, list]:
    """The same stats dict convert_dataset writes, recomputed independently from `arr` -- the
    oracle these tests check convert_dataset's written files against."""
    quantiles = {"q01": .01, "q10": .10, "q50": .50, "q90": .90, "q99": .99}
    a = np.asarray(arr, dtype=np.float64)
    out = {"min": a.min(axis=0).tolist(), "max": a.max(axis=0).tolist(), "mean": a.mean(axis=0).tolist(),
           "std": a.std(axis=0).tolist(), "count": [len(a)]}
    for key, q in quantiles.items():
        out[key] = np.quantile(a, q, axis=0).tolist()
    return out


def _toy_lerobot_task(root, lengths=(5, 4), *, seed: int = 0):
    """A minimal on-disk 9-D LeRobot v3.0 task: meta/info.json, a populated meta/stats.json
    (observation.state, action, and the unrelated observation.ee_state, as a published AXIS task
    carries them), meta/episodes/.../file-000.parquet with the matching per-episode `stats/`
    columns, one data parquet, and an empty videos/ dir for convert_dataset's symlink.
    `observation.ee_state` is a feature convert_dataset never touches, carried along so its stats
    being left alone is checked, not assumed.
    """
    rng = np.random.default_rng(seed)
    n = sum(lengths)
    ep = np.repeat(np.arange(len(lengths)), lengths)
    arm = np.cumsum(rng.normal(scale=0.05, size=(n, 7)), axis=0)
    fingers = (0.02 + 0.015 * np.sin(np.linspace(0.0, 6.0, n)))[:, None] * np.ones((1, 2))
    state9 = np.concatenate([arm, fingers], axis=1).astype(np.float32)
    action9 = (state9 + rng.normal(scale=0.001, size=state9.shape)).astype(np.float32)
    ee = rng.normal(size=(n, 3)).astype(np.float32)             # untouched by convert_dataset

    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "videos").mkdir(parents=True)

    features = {
        "observation.state": {"dtype": "float32", "shape": [9], "names": ["state"]},
        "action": {"dtype": "float32", "shape": [9], "names": ["action"]},
        "observation.ee_state": {"dtype": "float32", "shape": [3], "names": ["ee"]},
    }
    (root / "meta" / "info.json").write_text(json.dumps(
        {"codebase_version": "v3.0", "fps": 15, "total_episodes": len(lengths), "total_frames": n,
         "features": features}, indent=2))

    stats = {"observation.state": _direct_stats(state9), "action": _direct_stats(action9),
              "observation.ee_state": _direct_stats(ee)}
    (root / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))

    ep_rows = {"episode_index": np.arange(len(lengths)), "length": list(lengths),
               "dataset_from_index": np.concatenate([[0], np.cumsum(lengths)[:-1]]).tolist(),
               "dataset_to_index": np.cumsum(lengths).tolist()}
    for feature, arr in (("observation.state", state9), ("action", action9), ("observation.ee_state", ee)):
        per_ep = [_direct_stats(arr[ep == e]) for e in range(len(lengths))]
        for key in _STAT_KEYS:
            dtype = np.int64 if key == "count" else np.float64
            ep_rows[f"stats/{feature}/{key}"] = [np.asarray(d[key], dtype=dtype) for d in per_ep]
    pd.DataFrame(ep_rows).to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    pd.DataFrame({"observation.state": list(state9), "action": list(action9), "episode_index": ep,
                  "frame_index": np.concatenate([np.arange(k) for k in lengths])}).to_parquet(
        root / "data" / "chunk-000" / "file-000.parquet")
    return root


def test_convert_dataset_rewrites_dataset_level_stats_to_8d(tmp_path):
    dst = tmp_path / "dst"
    convert_dataset(_toy_lerobot_task(tmp_path / "src"), dst, fps=15.0)
    info = json.loads((dst / "meta" / "info.json").read_text())
    stats = json.loads((dst / "meta" / "stats.json").read_text())
    for feature in ("observation.state", "action"):
        (width,) = info["features"][feature]["shape"]
        assert width == 8
        for key, arr in stats[feature].items():
            assert len(arr) == (1 if key == "count" else width), f"{feature}/{key} is {len(arr)}-long"


def test_convert_dataset_dataset_level_stats_match_a_direct_recomputation(tmp_path):
    dst = tmp_path / "dst"
    convert_dataset(_toy_lerobot_task(tmp_path / "src"), dst, fps=15.0)
    df = pd.read_parquet(dst / "data" / "chunk-000" / "file-000.parquet")
    want = {"observation.state": _direct_stats(np.stack(df["observation.state"].to_numpy())),
            "action": _direct_stats(np.stack(df["action"].to_numpy()))}
    got = json.loads((dst / "meta" / "stats.json").read_text())
    for feature, w in want.items():
        for key, wv in w.items():
            np.testing.assert_allclose(got[feature][key], wv, rtol=1e-9, atol=1e-9)


def test_convert_dataset_stats_json_changes_only_the_converted_features(tmp_path):
    """Every byte of meta/stats.json outside the observation.state/action entries -- the other
    feature's values AND the file's json.dumps formatting -- is exactly the source's."""
    src = _toy_lerobot_task(tmp_path / "src")
    dst = tmp_path / "dst"
    convert_dataset(src, dst, fps=15.0)
    src_stats = json.loads((src / "meta" / "stats.json").read_text())
    dst_stats = json.loads((dst / "meta" / "stats.json").read_text())
    expected = dict(src_stats)
    expected["observation.state"] = dst_stats["observation.state"]
    expected["action"] = dst_stats["action"]
    assert json.dumps(expected, indent=2) == (dst / "meta" / "stats.json").read_text()
    assert dst_stats["observation.state"] != src_stats["observation.state"]   # actually changed


def test_convert_dataset_episode_stats_are_8d(tmp_path):
    lengths = (5, 4)
    dst = tmp_path / "dst"
    convert_dataset(_toy_lerobot_task(tmp_path / "src", lengths=lengths), dst, fps=15.0)
    eps = pd.read_parquet(dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    assert len(eps) == len(lengths)
    for feature in ("observation.state", "action"):
        for key in _STAT_KEYS:
            col = f"stats/{feature}/{key}"
            want = 1 if key == "count" else 8
            assert all(len(v) == want for v in eps[col]), col


def test_convert_dataset_episode_stats_match_a_direct_recomputation(tmp_path):
    lengths = (5, 4)
    dst = tmp_path / "dst"
    convert_dataset(_toy_lerobot_task(tmp_path / "src", lengths=lengths), dst, fps=15.0)
    df = pd.read_parquet(dst / "data" / "chunk-000" / "file-000.parquet")
    state8 = np.stack(df["observation.state"].to_numpy())
    action8 = np.stack(df["action"].to_numpy())
    ep = df["episode_index"].to_numpy()
    eps = pd.read_parquet(dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet").set_index("episode_index")
    for e in range(len(lengths)):
        row = eps.loc[e]
        for feature, arr8 in (("observation.state", state8), ("action", action8)):
            want = _direct_stats(arr8[ep == e])
            for key, wv in want.items():
                np.testing.assert_allclose(row[f"stats/{feature}/{key}"], wv, rtol=1e-9, atol=1e-9)


def test_convert_dataset_leaves_an_unchanged_features_episode_stats_alone(tmp_path):
    src = _toy_lerobot_task(tmp_path / "src")
    dst = tmp_path / "dst"
    convert_dataset(src, dst, fps=15.0)
    src_eps = pd.read_parquet(src / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    dst_eps = pd.read_parquet(dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    touched = {f"stats/{f}/{k}" for f in ("observation.state", "action") for k in _STAT_KEYS}
    for col in src_eps.columns:
        if col in touched:
            continue
        for a, b in zip(src_eps[col], dst_eps[col], strict=True):
            if isinstance(a, np.ndarray):
                np.testing.assert_array_equal(a, b)
            else:
                assert a == b
