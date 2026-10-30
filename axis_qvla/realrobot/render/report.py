"""Reported, never gated: how close a render is to a reference render of the same attempt, and how much
of the DROID-8D view's velocity sits at its clip bound.

Render tier (numpy, pandas; PyAV and scipy only inside the image metrics, so the DROID-8D readout needs
neither). Joint parity (all frames, and the knot frames alone), the gripper channel, PSNR and SSIM per
camera on decoded frames, and the per-channel mean colour difference (palette). The reference is another
render of the same attempt, e.g. a second render of your own. The DROID-8D view clips joint velocity to
+-1 rad/s (droid8d.py); how many entries sit at that bound is a property of the view alone, recorded
next to the hard gates' results, as is each episode's largest arm-joint jump between consecutive frames.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from axis_qvla.realrobot.render import rig


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Wang et al. 2004 on luma, Gaussian window sigma 1.5."""
    from scipy.ndimage import gaussian_filter
    w = np.array([0.299, 0.587, 0.114])
    x, y = a.astype(np.float64) @ w, b.astype(np.float64) @ w
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my = gaussian_filter(x, 1.5), gaussian_filter(y, 1.5)
    sxx = gaussian_filter(x * x, 1.5) - mx * mx
    syy = gaussian_filter(y * y, 1.5) - my * my
    sxy = gaussian_filter(x * y, 1.5) - mx * my
    return float(np.mean(((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))))


def _parquets(task: Path, sub: str) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p) for p in sorted((Path(task) / sub).rglob("*.parquet"))], ignore_index=True)


def _episode_rows(task: Path, episode: int) -> pd.DataFrame:
    df = _parquets(task, "data")
    return df[df["episode_index"] == episode]


def episode_frames(task: Path, key: str, episode: int) -> np.ndarray:
    """The decoded (T, H, W, 3) uint8 frames of one episode of video `key` (observation.images.<camera>).

    LeRobot v3 addressing, as a loader reads it: the file is info.json's `video_path` filled with the
    episode row's `videos/<key>/chunk_index` and `file_index`, and the episode is the frames whose
    presentation time lies in [from_timestamp, to_timestamp) of that file (half a frame of tolerance).
    The reader seeks to from_timestamp and selects by pts, never by counting frames from the file's start.
    """
    import av
    task = Path(task)
    info = json.loads((task / "meta" / "info.json").read_text())
    meta = _parquets(task, "meta/episodes")
    rows = meta[meta["episode_index"] == episode]
    if rows.empty:
        raise KeyError(f"{task}: no episode {episode} in meta/episodes")
    row = rows.iloc[0]
    path = task / info["video_path"].format(video_key=key, chunk_index=int(row[f"videos/{key}/chunk_index"]),
                                            file_index=int(row[f"videos/{key}/file_index"]))
    lo, hi = float(row[f"videos/{key}/from_timestamp"]), float(row[f"videos/{key}/to_timestamp"])
    fps, n = float(info["fps"]), int(row["length"])
    half = 0.5 / fps
    out, times = [], []
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        c.seek(max(int((lo - half) / s.time_base), 0), stream=s, backward=True)
        for f in c.decode(s):
            t = float(f.pts * s.time_base)
            if t >= hi - half:
                break
            if t >= lo - half:
                out.append(f.to_ndarray(format="rgb24"))
                times.append(t)
    want = lo + np.arange(n) / fps
    if len(out) != n or np.abs(np.asarray(times) - want).max(initial=0.0) > half:
        raise ValueError(f"{path}: {len(out)} frames in [{lo}, {hi}) s for episode {episode}'s {n} at {fps:g} fps")
    return np.stack(out)


def report_episode(task: Path, episode: int, ref_task: Path, ref_episode: int) -> dict:
    ours, ref = _episode_rows(Path(task), episode), _episode_rows(Path(ref_task), ref_episode)
    if len(ours) != len(ref):
        raise ValueError(f"episode {episode} has {len(ours)} frames, the reference {len(ref)}: "
                         f"not renders of the same attempt")
    st = np.stack(ours["observation.state"].to_numpy()).astype(np.float64)
    rs = np.stack(ref["observation.state"].to_numpy()).astype(np.float64)
    ac = np.stack(ours["action"].to_numpy()).astype(np.float64)
    ra = np.stack(ref["action"].to_numpy()).astype(np.float64)
    knots = np.asarray(json.loads((Path(task) / "meta" / "axis_episodes" /
                                   f"episode_{episode:06d}.json").read_text())["knot_rows"])
    after_home = knots[1:] if len(knots) > 1 else knots       # knot 0 is the home, not a solve
    out = {"frames": [len(st), len(rs)],
           "joint_parity": {"all_rows_max_rad": float(np.abs(st - rs).max()),
                            "all_rows_median_rad": float(np.median(np.abs(st - rs).max(1))),
                            "knot_rows_max_rad": float(np.abs(st[knots] - rs[knots]).max()),
                            "knot_rows_median_rad": float(np.median(np.abs(st[after_home] - rs[after_home]).max(1)))},
           "gripper_max_abs_ctrl": float(np.abs(ac[:, 7] - ra[:, 7]).max()), "cameras": {}}
    out["reference_starts_at_home"] = bool(np.abs(rs[0] - np.asarray(rig.PIXELMATCH_HOME)).max() < 1e-3)
    for key_ in rig.CAMERA_KEYS:
        key = f"observation.images.{key_}"
        if not (Path(ref_task) / "videos" / key).is_dir():
            out["cameras"][key_] = {"not_compared": "the reference copy holds no videos"}
            continue
        a, b = episode_frames(Path(task), key, episode), episode_frames(Path(ref_task), key, ref_episode)
        out["cameras"][key_] = {
            "psnr_db_mean": float(np.mean([psnr(x, y) for x, y in zip(a, b)])),
            "ssim_mean": float(np.mean([ssim(x, y) for x, y in zip(a, b)])),
            "palette_abs_mean_diff": (np.abs(a.reshape(-1, 3).mean(0) - b.reshape(-1, 3).mean(0)) / 255.0).tolist()}
    return out


def write_report(task: Path, episode: int, ref_task: Path, ref_episode: int, out: Path) -> dict:
    rep = report_episode(task, episode, ref_task, ref_episode)
    Path(out).write_text(json.dumps(rep, indent=1))
    return rep


def droid8d_velocity(droid8d_root: Path) -> dict:
    """Per task and in total: rows, joint-velocity entries at the +-1 bound, and rows with any entry there."""
    doc = json.loads((Path(droid8d_root) / "roots.json").read_text())
    base = Path(droid8d_root) / doc.get("base", ".")
    tasks = {}
    for tid, rel in sorted(doc["roots"].items(), key=lambda kv: int(kv[0])):
        df = _parquets(base / rel, "data")
        vel = np.stack(df["action"].to_numpy()).reshape(-1, 8)[:, :7]
        at = np.abs(vel) >= 1.0
        tasks[str(tid)] = {"rows": int(len(vel)), "velocity_at_bound": int(at.sum()), "rows_at_bound": int(at.any(1).sum()),
                           "fraction_rows_at_bound": float(at.any(1).mean()) if len(vel) else 0.0}
    rows = sum(t["rows"] for t in tasks.values())
    at_rows = sum(t["rows_at_bound"] for t in tasks.values())
    return {"tasks": tasks, "rows": rows, "velocity_at_bound": sum(t["velocity_at_bound"] for t in tasks.values()),
            "rows_at_bound": at_rows, "fraction_rows_at_bound": at_rows / rows if rows else 0.0}


def joint_jumps(task: Path) -> dict[int, dict]:
    """Per episode: the largest change of one arm joint between consecutive frames (rad), the later frame's
    index within the episode, and the joint (0-6). A large value is a reconfiguration of the arm, which
    the training video shows as a jump."""
    df = _parquets(Path(task), "data")
    out = {}
    for e, rows in df.groupby("episode_index"):
        st = np.stack(rows.sort_values("frame_index")["observation.state"].to_numpy()).reshape(-1, 7).astype(np.float64)
        d = np.abs(np.diff(st, axis=0))
        if not d.size:
            out[int(e)] = {"max_rad": 0.0, "frame": None, "joint": None}
            continue
        f, j = np.unravel_index(int(np.argmax(d)), d.shape)
        out[int(e)] = {"max_rad": float(d[f, j]), "frame": int(f) + 1, "joint": int(j)}
    return out
