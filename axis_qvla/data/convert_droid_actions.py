"""Offline: rewrite an AXIS LeRobot dataset's action+state columns to the DROID
8D layout (arm joint velocity + gripper closedness). Runs under the offline environment.
Video is symlinked; the data parquet, meta/info.json and -- for whichever of
`observation.state`/`action` the source's meta/stats.json and meta/episodes already carry
statistics for -- their stats are rewritten; episode/row structure (and the sidecar
frame-alignment) is unchanged.
"""
from __future__ import annotations

import glob
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def joint_velocity(state_arm: np.ndarray, episode_index: np.ndarray, fps: float) -> np.ndarray:
    """Forward-difference arm joint velocity, zeroed at each episode's last frame."""
    state_arm = np.asarray(state_arm, dtype=np.float64)
    ep = np.asarray(episode_index)
    v = np.zeros_like(state_arm)
    v[:-1] = (state_arm[1:] - state_arm[:-1]) * fps
    # zero the last frame of each episode (no successor within the episode)
    boundary = np.concatenate([ep[1:] != ep[:-1], [True]])
    v[boundary] = 0.0
    return v.astype(np.float32)


def gripper_calibration(width: np.ndarray) -> tuple[float, float]:
    """(open, closed) finger width from the 99%/1% quantiles of the WHOLE dataset.

    Quantiles rather than min/max so a single glitched frame cannot set the scale, and
    whole-dataset rather than per-episode so an episode that never grips does not get its
    measurement noise stretched over the full [0, 1] closedness range.
    """
    width = np.asarray(width, dtype=np.float64)
    return float(np.quantile(width, 0.99)), float(np.quantile(width, 0.01))


def gripper_closedness(g0, g1, open_w: float, closed_w: float) -> np.ndarray:
    """Two finger widths -> DROID's scalar closedness in [0, 1] (0 = open, 1 = closed)."""
    width = 0.5 * (np.asarray(g0, dtype=np.float64) + np.asarray(g1, dtype=np.float64))
    denom = open_w - closed_w
    if abs(denom) < 1e-8:
        raise ValueError("gripper open/closed widths too close to calibrate")
    return np.clip((open_w - width) / denom, 0.0, 1.0).astype(np.float32)


def _stack(col) -> np.ndarray:
    """List-of-vector pandas column -> (N, D) float64.

    `np.stack` over a Python list up-casts each of the N rows separately; `np.vstack` on
    the column's object array does one pass and one cast. Measured 2.4x faster on a
    390 328-row column (0.55 s -> 0.23 s), which matters because convert_dataset calls
    this twice per parquet file.
    """
    if len(col) == 0:
        return np.zeros((0, 0), dtype=np.float64)
    return np.asarray(np.vstack(np.asarray(col)), dtype=np.float64)


def to_droid8d_frame(df, fps: float = 15.0):
    """Rewrite a task's in-memory frames to the DROID-8D layout, in place-ish (returns the df).

    Same math as ``convert_dataset`` but on a DataFrame the materializer already holds, so the
    conversion happens in the materialization pass with no second on-disk convert. ``df`` must
    carry ``observation.state`` (9-D: 7 arm joints + 2 finger widths), ``action`` (9-D, same
    layout — position targets), and ``episode_index`` (renumbered is fine; only boundaries matter).
    All other columns (the baked reward/primitive columns) pass through untouched.

    Gripper is calibrated over the WHOLE df (99/1 quantiles of mean finger width), matching
    ``convert_dataset`` — per derived task, since each derived task is a standalone dataset.
    """
    st = _stack(df["observation.state"])
    ac = _stack(df["action"])
    ep = df["episode_index"].to_numpy()
    # STATE WIDTH IS A CONTRACT, checked rather than indexed into. Columns 7 and 8 are the two
    # finger widths; on a narrower state `st[:, 8]` raises a bare numpy IndexError that says
    # nothing about which corpus, which column layout, or what was expected -- and an auto-build
    # driving hundreds of tasks turns that into an unreadable traceback per task.
    #
    # A corpus's state width is NOT stable across render batches: a raw batch declares 9-D and a
    # derived corpus 8-D, so this is a real intake question and not a defensive formality.
    # 9-D raw (7 arm joints + 2 finger widths) is the INPUT this
    # function converts FROM; 8-D DROID is what it produces, so being handed 8-D usually means
    # someone is converting an already-converted corpus a second time.
    for name, arr in (("observation.state", st), ("action", ac)):
        if arr.ndim != 2 or arr.shape[1] != 9:
            width = "x".join(str(d) for d in arr.shape[1:]) or "scalar"
            raise ValueError(
                f"{name} is {width}-wide; to_droid8d_frame converts FROM the raw 9-D layout "
                f"(7 arm joints + 2 finger widths) TO DROID-8D. An 8-D input is already "
                f"converted -- do not run this twice. Measure the source state width before building."
            )
    open_w, closed_w = gripper_calibration(0.5 * (st[:, 7] + st[:, 8]))
    new_state = np.concatenate(
        [st[:, :7], gripper_closedness(st[:, 7], st[:, 8], open_w, closed_w)[:, None]], axis=1
    ).astype(np.float32)
    new_action = np.concatenate(
        [joint_velocity(st[:, :7], ep, fps),
         gripper_closedness(ac[:, 7], ac[:, 8], open_w, closed_w)[:, None]], axis=1
    ).astype(np.float32)
    out = df.copy()
    out["observation.state"] = list(new_state)
    out["action"] = list(new_action)
    return out


_QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}


def _feature_stats(arr: np.ndarray) -> dict[str, np.ndarray]:
    """One feature's LeRobot stats dict -- the same keys meta/stats.json and meta/episodes'
    `stats/<feature>/*` columns already carry for every feature this module does not touch: min,
    max, mean, std (population, ddof=0), count, and the 1st/10th/50th/90th/99th percentiles, one
    value per column of `arr` (count is the one exception, a length-1 array).

    Computed directly over the rows given -- exact, not the pinned LeRobot line's out-of-core
    histogram approximation (`lerobot.datasets.compute_stats.RunningQuantileStats`) -- since
    convert_dataset already holds a task's whole converted array in memory, both at the per-episode
    grain and over the whole dataset, so there is no need to approximate.
    """
    arr = np.asarray(arr, dtype=np.float64)
    stats: dict[str, np.ndarray] = {
        "min": arr.min(axis=0),
        "max": arr.max(axis=0),
        "mean": arr.mean(axis=0),
        "std": arr.std(axis=0),
        "count": np.array([len(arr)], dtype=np.int64),
    }
    for key, q in _QUANTILES.items():
        stats[key] = np.quantile(arr, q, axis=0)
    return stats


def _patch_dataset_stats(meta_dir: Path, feature_stats: dict[str, dict[str, np.ndarray]]) -> None:
    """Overwrite `feature_stats`'s entries in meta/stats.json with the converted (8-D) values.

    Every other feature's entry, and the file's formatting, is left byte-for-byte as copied from
    the source: the file is loaded, only the named keys are replaced, and it is written back with
    the same `json.dumps(..., indent=2)` this module already uses for info.json, which round-trips
    an unmodified entry to the same bytes it started with. A source with no meta/stats.json, or no
    entry for a given feature, is left alone -- convert_dataset never introduces a stats entry the
    source never had.
    """
    path = meta_dir / "stats.json"
    if not path.exists():
        return
    stats = json.loads(path.read_text())
    changed = False
    for feature, stat_dict in feature_stats.items():
        if feature not in stats:
            continue
        stats[feature] = {key: np.asarray(value).tolist() for key, value in stat_dict.items()}
        changed = True
    if changed:
        path.write_text(json.dumps(stats, indent=2))


def _patch_episode_stats(meta_dir: Path, per_episode: dict[int, dict[str, dict[str, np.ndarray]]]) -> None:
    """Overwrite the `stats/<feature>/<key>` columns of every meta/episodes row with the converted
    values for the episode it describes.

    LeRobot's own merge tool (`lerobot.scripts.axis_merge_datasets`, which is a thin wrapper over
    `lerobot.datasets.aggregate.aggregate_datasets`) reads these per-episode columns -- not
    meta/stats.json -- to build a merged copy's own dataset-level stats
    (`aggregate_datasets.finalize_aggregation` calls `aggregate_stats` over every source's
    `LeRobotDatasetMetadata.stats`, which `dataset_metadata.py` loads from meta/stats.json, but the
    per-episode table it carries forward unchanged is these `stats/` columns). A stale 9-D value
    here would reach a merged copy even after meta/stats.json is fixed. A row for an episode this
    call has no stats for, or a parquet with no `stats/` columns at all, is left untouched.
    """
    episodes_dir = meta_dir / "episodes"
    if not episodes_dir.is_dir():
        return
    features = sorted({feature for stats in per_episode.values() for feature in stats})
    keys = sorted({key for stats in per_episode.values() for stat in stats.values() for key in stat})
    for f in sorted(episodes_dir.glob("**/*.parquet")):
        df = pd.read_parquet(f)
        episode_ids = df["episode_index"].to_numpy()
        touched = False
        for feature in features:
            for key in keys:
                col = f"stats/{feature}/{key}"
                if col not in df.columns:
                    continue
                # Reassign the WHOLE column from a plain list, not `.at[row, col] = array`: pandas'
                # per-cell `.at` squeezes a length-1 array value down to a 0-D scalar (silently --
                # the count column, [N], becomes bare N), which pyarrow then refuses to write back
                # as the list<int64> column it already is.
                values = df[col].tolist()
                dtype = np.int64 if key == "count" else np.float64
                for i, eidx in enumerate(episode_ids):
                    stat_dict = per_episode.get(int(eidx), {}).get(feature)
                    if stat_dict is None:
                        continue
                    values[i] = np.asarray(stat_dict[key], dtype=dtype)
                    touched = True
                df[col] = values
        if touched:
            df.to_parquet(f)


def convert_dataset(src_root, dst_root, fps: float = 15.0) -> dict:
    src, dst = Path(src_root), Path(dst_root)
    dst.mkdir(parents=True, exist_ok=True)
    # meta: copy everything, then symlink videos, then patch info.json shapes.
    if (dst / "meta").exists():
        shutil.rmtree(dst / "meta")
    shutil.copytree(src / "meta", dst / "meta")
    if (dst / "videos").exists() or (dst / "videos").is_symlink():
        (dst / "videos").unlink() if (dst / "videos").is_symlink() else shutil.rmtree(dst / "videos")
    (dst / "videos").symlink_to((src / "videos").resolve(), target_is_directory=True)
    info = json.loads((dst / "meta" / "info.json").read_text())
    for k in ("observation.state", "action"):
        if k in info.get("features", {}):
            info["features"][k]["shape"] = [8]
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    data_files = sorted(glob.glob(str(src / "data" / "**" / "*.parquet"), recursive=True))
    # Calibrate the gripper once over every frame's finger widths. Project to the state
    # column only: the full row carries the camera intrinsics/extrinsics (16 floats each)
    # that this pass has no use for, and the files are read again below anyway.
    finger_widths = [
        # np.ascontiguousarray, not a bare slice: the slice is a VIEW, so each element of
        # this list would pin its full (N, 9) float64 base array until the concatenate
        # below -- 28 MB resident instead of 6 MB for one dataset, scaling with file count.
        np.ascontiguousarray(
            _stack(pd.read_parquet(f, columns=["observation.state"])["observation.state"])[:, 7:9]
        )
        for f in data_files
    ]
    all_widths = np.concatenate(finger_widths)
    open_w, closed_w = gripper_calibration(0.5 * (all_widths[:, 0] + all_widths[:, 1]))

    n_frames = 0
    eps, states, actions = [], [], []       # accumulated across files, to stats the whole task
    for f in data_files:
        rel = Path(f).relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_parquet(f)
        st = _stack(df["observation.state"]); ac = _stack(df["action"])
        ep = df["episode_index"].to_numpy()
        new_state = np.concatenate(
            [st[:, :7], gripper_closedness(st[:, 7], st[:, 8], open_w, closed_w)[:, None]], axis=1
        ).astype(np.float32)
        new_action = np.concatenate(
            [joint_velocity(st[:, :7], ep, fps),
             gripper_closedness(ac[:, 7], ac[:, 8], open_w, closed_w)[:, None]], axis=1
        ).astype(np.float32)
        df["observation.state"] = list(new_state)
        df["action"] = list(new_action)
        df.to_parquet(out, compression="zstd")
        n_frames += len(df)
        eps.append(ep); states.append(new_state); actions.append(new_action)

    # The converted state/action are 8-D, but meta/stats.json and meta/episodes' `stats/` columns
    # were just copied wholesale from the 9-D source (above) and still say 9: LeRobot's normaliser
    # then fails at the first training step comparing an 8-D batch to a 9-D stats tensor. Recompute
    # both from the converted rows, over the whole task (dataset-level) and per episode.
    if eps:
        ep_all, state_all, action_all = np.concatenate(eps), np.concatenate(states), np.concatenate(actions)
        dataset_stats = {"observation.state": _feature_stats(state_all), "action": _feature_stats(action_all)}
        per_episode = {
            int(e): {"observation.state": _feature_stats(state_all[ep_all == e]),
                     "action": _feature_stats(action_all[ep_all == e])}
            for e in np.unique(ep_all)
        }
        _patch_dataset_stats(dst / "meta", dataset_stats)
        _patch_episode_stats(dst / "meta", per_episode)

    return {"n_frames": n_frames, "open_w": open_w, "closed_w": closed_w,
            "src": str(src), "dst": str(dst)}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--fps", type=float, default=15.0)
    a = ap.parse_args()
    print(convert_dataset(a.src, a.dst, a.fps))


if __name__ == "__main__":
    main()
