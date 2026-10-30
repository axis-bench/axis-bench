"""The renderer's hard gates, checked on what was written.

Render tier (numpy, pandas, pyarrow, PyAV). Each gate reads the task directory (and the per-episode
sidecar for the inputs it must recompute against) and returns {"ok", "failures", ...}:
- structure: info.json features == the published schema; info.json and every sidecar name the workspace
  kept beside the render (workspace.toml); parquet columns and fixed-size widths; meta/episodes columns
  incl. attempt_id, and every sidecar names the attempt its meta/episodes row does; each recorded row on
  its frame and each episode as long as resample.frame_grid makes it; frame_index, index and timestamp; every
  video h264 640x360 yuv420p holding exactly the total frames;
- intrinsics: every row's K is its camera's (float32-exact); the exterior pose is the workspace profile's;
- home: every frame up to the first recorded row's time is the pixelmatch home, float32-exact (frames
  0..2 of a post-step recording; frames 0..1 when the first row falls between frames 1 and 2);
- tcp: the convergence knot -- the first knot after knot 0 whose pinch (from the written ee_state) is
  within 3 mm of the source TCP -- comes no later than knot 8, and at least 60% of knots are within 2 cm
  of the source TCP, the IK's own acceptance recomputed from the written poses. The IK solves each knot
  toward 1.5 mm of the source TCP; some recordings move the hand where the Panda with the 2F-85 cannot
  follow, and there the render keeps the pose the IK selects, which can be several centimetres from the
  source TCP. How far, and on how many knots, is reported per episode (tcp_reported), never gated;
- gripper: action[7] is the causal ZOH of 255 * (1 - openness) on the recorded time base, and the
  settled extremes match the expected separations;
- label_tag: the render's curve store has one curve of exactly each episode's length, and rank weights
  and pure-conditioning tags built over the rendered DROID-8D corpus equal those built over a
  metadata-only view of the same attempts with the same curves.
Joint parity, PSNR/SSIM and palette are reported by report.py, never gated. run_gates records under
"reported" the pinch's distance from the source TCP after convergence, the largest arm-joint jump between
consecutive frames, and the count of DROID-8D velocities at the clip bound.

The per-episode sidecar is meta/axis_episodes/episode_<index>.json: LeRobot v3's episode_index lays the
task out, and attempt_id is written inside every artifact (the structure gate checks they agree).
"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from axis_qvla.realrobot.render import resample, rig, writer
from axis_qvla.realrobot.render.profile import FILE as PROFILE_FILE, load_profile

EPISODE_COLUMNS = ("episode_index", "tasks", "length", "data/chunk_index", "data/file_index", "dataset_from_index",
                   "dataset_to_index", "meta/episodes/chunk_index", "meta/episodes/file_index", "attempt_id")
_SIDECAR = re.compile(r"episode_(\d+)")


class GateError(RuntimeError):
    """At least one hard gate failed; the message names every failing gate."""


def _result(fails: list[str], **extra) -> dict:
    return {"ok": not fails, "failures": fails[:50], "n_failures": len(fails), **extra}


def _table(task: Path) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p) for p in sorted((task / "data").rglob("*.parquet"))], ignore_index=True)


def _col(df: pd.DataFrame, name: str, width: int) -> np.ndarray:
    return np.stack(df[name].to_numpy()).reshape(-1, width)


def _episodes(task: Path) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p) for p in sorted((task / "meta" / "episodes").rglob("*.parquet"))],
                     ignore_index=True)


def _sidecars(task: Path) -> dict[int, dict]:
    out = {}
    for p in sorted((task / "meta" / "axis_episodes").glob("episode_*.json")):
        m = _SIDECAR.fullmatch(p.stem)
        if m is not None:
            out[int(m.group(1))] = json.loads(p.read_text())
    return out


def _with_sidecars(task: Path, df: pd.DataFrame, fails: list[str]):
    """(episode index, sidecar, rows) per episode of the data; an episode without a sidecar is a failure."""
    sides = _sidecars(task)
    for e, rows in df.groupby("episode_index"):
        if int(e) not in sides:
            fails.append(f"episode {e}: no sidecar")
            continue
        yield int(e), sides[int(e)], rows


def gate_structure(task: Path) -> dict:
    fails = []
    info = json.loads((task / "meta" / "info.json").read_text())
    workspace = load_profile(task.parent / PROFILE_FILE).id
    kept = f"the render keeps workspace {workspace} ({PROFILE_FILE})"
    if info.get("features") != writer.FEATURES:
        fails.append("info.json features differ from the published schema")
    if info.get("workspace_profile") != workspace:
        fails.append(f"info.json workspace_profile {info.get('workspace_profile')!r}; {kept}")
    if info.get("fps") != rig.FPS or info.get("codebase_version") != writer.CODEBASE_VERSION:
        fails.append(f"fps {info.get('fps')} / codebase {info.get('codebase_version')}")
    for p in sorted((task / "data").rglob("*.parquet")):
        s = pq.read_schema(p)
        for name, f in writer.FEATURES.items():
            if f["dtype"] == "video":
                continue
            if name not in s.names:
                fails.append(f"{p.name}: no column {name}")
            elif f["dtype"] == "float32" and getattr(s.field(name).type, "list_size", 1) != f["shape"][0] \
                    and name != "timestamp":
                fails.append(f"{p.name}: {name} is {s.field(name).type}")
    ep = _episodes(task)
    missing = [c for c in EPISODE_COLUMNS if c not in ep.columns]
    fails += [f"meta/episodes has no {c}" for c in missing]
    if "attempt_id" in ep.columns and ep["attempt_id"].duplicated().any():
        dup = sorted({int(a) for a in ep.loc[ep["attempt_id"].duplicated(), "attempt_id"]})
        fails.append(f"attempt {dup[:5]} rendered more than once: attempt-keyed artifacts would collide")
    df = _table(task)
    sides = _sidecars(task)
    if sorted(sides) != sorted(int(e) for e in ep["episode_index"]):
        fails.append(f"sidecars for episodes {sorted(sides)[:10]}, meta/episodes lists {sorted(ep['episode_index'])[:10]}")
    for _, row in ep.iterrows():
        e = int(row["episode_index"])
        side = sides.get(e)
        if side is None:
            continue
        if "attempt_id" in row and int(side.get("attempt_id", -1)) != int(row["attempt_id"]):
            fails.append(f"episode {e}: sidecar attempt {side.get('attempt_id')}, meta/episodes attempt {row['attempt_id']}")
        if side.get("workspace") != workspace:
            fails.append(f"episode {e}: sidecar workspace {side.get('workspace')!r}; {kept}")
        grid, want_rows = resample.frame_grid(np.asarray(side["row_times"], np.float64))
        if list(side["knot_rows"]) != want_rows.tolist():
            fails.append(f"episode {e}: knot rows are not the frames of the recorded times")
        if int(row["length"]) != len(grid):
            fails.append(f"episode {e}: {row['length']} frames; its last recorded row is frame {int(want_rows[-1])}")
        rows = df[df["episode_index"] == e]
        if len(rows) != int(row["length"]):
            fails.append(f"episode {e}: {len(rows)} data rows, meta/episodes says {row['length']}")
        if not np.array_equal(rows["frame_index"].to_numpy(), np.arange(len(rows))):
            fails.append(f"episode {e}: frame_index is not 0..T-1")
        if not np.array_equal(rows["timestamp"].to_numpy(), (np.arange(len(rows)) / rig.FPS).astype(np.float32)):
            fails.append(f"episode {e}: timestamps are not frame_index / 15")
    if not np.array_equal(df["index"].to_numpy(), np.arange(len(df))):
        fails.append("index is not contiguous")
    for key in writer.VIDEO_KEYS:
        files = sorted((task / "videos" / key).rglob("*.mp4"))
        if not files:
            fails.append(f"{key}: no video")
        n = 0
        for p in files:
            try:
                n += writer.video_frame_count(p)
            except (ValueError, OSError) as exc:
                fails.append(f"{key}: {exc}")
        if files and n != len(df):
            fails.append(f"{key}: {n} video frames for {len(df)} rows")
    return _result(fails, frames=len(df), episodes=len(ep))


def gate_intrinsics(task: Path, cameras: tuple[rig.Camera, ...]) -> dict:
    fails = []
    df = _table(task)
    for cam in cameras:
        k = _col(df, f"observation.cameras.{cam.key}.intrinsics", 9)
        want = np.asarray(rig.intrinsics(cam), np.float32)
        if not np.array_equal(k, np.tile(want, (len(k), 1))):
            fails.append(f"{cam.key}: intrinsics differ from its camera's on {int((k != want).any(1).sum())} rows")
    ext = cameras[0]
    front = _col(df, f"observation.cameras.{ext.key}.camera_to_world", 16).reshape(-1, 4, 4)
    want = np.eye(4)
    want[:3, :3], want[:3, 3] = rig.quat_to_mat(ext.quat), ext.pos
    if np.abs(front - want).max() > 1e-5:
        fails.append("the exterior camera pose differs from the workspace profile")
    return _result(fails)


def gate_home(task: Path) -> dict:
    fails = []
    df = _table(task)
    home = np.asarray(rig.PIXELMATCH_HOME, np.float32)
    for e, side, rows in _with_sidecars(task, df, fails):
        rt = np.asarray(side["row_times"], np.float64)
        n = int(np.count_nonzero(resample.frame_grid(rt)[0] <= rt[0] + resample.TIME_EPS))
        st = _col(rows, "observation.state", 7)[:n]
        if not np.array_equal(st, np.tile(home, (len(st), 1))):
            fails.append(f"episode {e}: frames 0..{n - 1}, up to the first recorded row, are not the pixelmatch home")
    return _result(fails)


def _pinch_errors(task: Path, fails: list[str]):
    """(episode index, sidecar, per-knot distance in metres from the written pinch to the source TCP)."""
    for e, side, rows in _with_sidecars(task, _table(task), fails):
        ee = _col(rows, "observation.ee_state", 7).astype(np.float64)[side["knot_rows"]]
        pinch = np.asarray(side["pinch_in_base"])
        world = np.stack([p[:3] + np.asarray(rig.quat_to_mat(p[3:])) @ pinch for p in ee])
        yield e, side, np.linalg.norm(world - np.asarray(side["source_tcp"]), axis=1)


def _convergence_knot(err: np.ndarray) -> int | None:
    hits = np.flatnonzero(err[1:] <= rig.TCP_GATE_M)
    return int(hits[0]) + 1 if hits.size else None


def gate_tcp(task: Path) -> dict:
    fails, knots = [], {}
    for e, side, err in _pinch_errors(task, fails):
        k = _convergence_knot(err)
        reach = float(np.mean(err < rig.IK.reach_tol) * 100.0)
        knots[e] = {"convergence_knot": k, "reach_pct": reach, "ik_convergence_knot": side.get("convergence_knot")}
        if k is None or k > rig.CONVERGE_WITHIN_KNOTS:
            fails.append(f"episode {e}: the TCP converges at knot {k}, after knot {rig.CONVERGE_WITHIN_KNOTS}")
        if reach < rig.IK.min_reach_pct:
            fails.append(f"episode {e}: {reach:.1f}% of knots within {rig.IK.reach_tol * 100:.0f} cm of the source "
                         f"TCP, under the minimum of {rig.IK.min_reach_pct:.0f}%")
    return _result(fails, episodes=knots)


def tcp_reported(task: Path) -> dict[int, dict]:
    """Per episode, from the convergence knot on: the largest and median pinch error (mm), and how many
    knots are over the convergence threshold (source poses the arm cannot reach)."""
    out = {}
    for e, _side, err in _pinch_errors(task, []):
        k = _convergence_knot(err)
        after = err[k:] if k is not None else None
        out[e] = {"convergence_knot": k,
                  "max_mm_after": None if after is None else float(after.max() * 1000),
                  "median_mm_after": None if after is None else float(np.median(after) * 1000),
                  "knots_over_gate_after": None if after is None else int(np.count_nonzero(after > rig.TCP_GATE_M)),
                  "knots_after": None if after is None else int(after.size)}
    return out


def gate_gripper(task: Path, *, open_separation: float = rig.GRIPPER_OPEN_SEPARATION_M,
                 closed_separation: float = rig.GRIPPER_CLOSED_SEPARATION_M) -> dict:
    fails = []
    df = _table(task)
    for e, side, rows in _with_sidecars(task, df, fails):
        kt, lead = np.asarray(side["row_times"], np.float64), float(side["lead_time"])
        u = np.asarray(side["openness"])
        ctrl = 255.0 * (1.0 - u)
        want = resample.zoh(np.r_[lead, kt], np.r_[ctrl[:1], ctrl], resample.frame_grid(kt)[0])
        got = _col(rows, "action", 8)[:, 7].astype(np.float64)
        if got.shape != want.shape or np.abs(got - want).max() > 1e-3:
            fails.append(f"episode {e}: action[7] is not the ZOH of 255 * (1 - openness)")
        g = side["gripper"]
        if (float(g["open_ctrl"]), float(g["closed_ctrl"])) != (0.0, 255.0):
            fails.append(f"episode {e}: ctrl range {g['open_ctrl']}..{g['closed_ctrl']}, expected 0 (open) .. 255")
        if abs(g["open_separation"] - open_separation) > rig.GRIPPER_SEPARATION_TOL_M or \
                abs(g["closed_separation"] - closed_separation) > rig.GRIPPER_SEPARATION_TOL_M:
            fails.append(f"episode {e}: settled separations {g['open_separation']:.5f}/{g['closed_separation']:.5f}")
    return _result(fails)


def _tags_and_weights(roots_path: Path, curves, work: Path):
    """(ranges, rank weights by key, tag rows or None, the builder's ValueError or None) over one corpus.

    The pinned tag builder refuses a corpus whose quality label is constant (one episode, or every
    episode in one bin): its tag/task entropy diagnostic is 0/0. That refusal (ConstantQualityError) is
    a property of the corpus, so the gate accepts it when both corpora get it and still compares ranges
    and weights. Every other refusal fails the gate.
    """
    from axis_qvla.realrobot.realign.ranges import tail_ranges, write_ranges
    from axis_qvla.realrobot.realign.tags import build_pure_tags
    from axis_qvla.realrobot.realign.weights import rank_weights, write_weights
    base = (roots_path.parent / json.loads(roots_path.read_text()).get("base", ".")).resolve()
    work.mkdir(parents=True, exist_ok=True)
    ranges = tail_ranges(roots_path)
    write_ranges(work / "ranges.json", ranges)
    eps = rank_weights(curves)
    write_weights(work / "awr_weights_phase_rank.json", eps, corpus=base, balanced=False)
    try:
        build_pure_tags(work / "awr_weights_phase_rank.json", roots_path, work / "ranges.json", work / "quality_phase.npz")
    except ValueError as exc:
        return ranges, {e.key: e.weights for e in eps}, None, exc
    with np.load(work / "quality_phase.npz") as z:
        tag = z["tag"]
    return ranges, {e.key: e.weights for e in eps}, tag, None


def gate_label_tag(droid8d_root: Path, render_store: Path, attempt_curves: dict) -> dict:
    from axis_qvla.realrobot.realign.curves import CurveError, read_curve_store
    from axis_qvla.realrobot.realign.view import ViewEpisode, write_meta_view
    from axis_qvla.signals.blocks.tags import ConstantQualityError
    fails = []
    roots = Path(droid8d_root) / "roots.json"
    try:
        store = read_curve_store(render_store)
    except CurveError as exc:
        return _result([f"the render's curve store ({render_store}): {exc}"])
    doc = json.loads(roots.read_text())
    base = Path(droid8d_root) / doc.get("base", ".")
    view_eps, view_curves, corpus_keys = [], {}, set()
    for tid, rel in doc["roots"].items():
        meta = pd.concat([pd.read_parquet(p) for p in sorted((base / rel / "meta" / "episodes").rglob("*.parquet"))])
        for ep, length, aid, tasks in zip(meta["episode_index"], meta["length"], meta["attempt_id"], meta["tasks"]):
            corpus_keys.add((int(tid), int(ep)))
            c = store.get((int(tid), int(ep)))
            if c is None or len(c.curve) != int(length):
                fails.append(f"task {tid} episode {ep}: no curve of {int(length)} rows in the render's store")
                continue
            if (int(tid), int(aid)) not in attempt_curves:
                fails.append(f"task {tid} episode {ep}: attempt {aid} has no attempt-keyed curve")
                continue
            a = attempt_curves[(int(tid), int(aid))]
            differs = [n for n, same in (("curve", np.array_equal(c.curve, a.curve)),
                                         ("valid", np.array_equal(c.valid, a.valid)), ("t0", c.t0 == a.t0)) if not same]
            if differs:   # t0 and the mask feed `realign build` (balanced weights) though not the rank file
                fails.append(f"task {tid} episode {ep}: the store's {'/'.join(differs)} differ from attempt {aid}'s")
                continue
            view_eps.append(ViewEpisode(int(tid), int(ep), int(length), int(aid), str(list(tasks)[0])))
            view_curves[(int(tid), int(ep))] = attempt_curves[(int(tid), int(aid))]
    extra = sorted(set(store) - corpus_keys)
    if extra:
        fails.append(f"the render's store holds curves for episodes the corpus does not: {extra[:5]}")
    if fails:
        return _result(fails)
    with tempfile.TemporaryDirectory(dir=Path(render_store).parent) as tmp:
        r_a, w_a, tag_a, why_a = _tags_and_weights(roots, store, Path(tmp) / "render")
        view_roots = write_meta_view(Path(tmp) / "view", view_eps)
        r_b, w_b, tag_b, why_b = _tags_and_weights(view_roots, view_curves, Path(tmp) / "view_out")
    if r_a != r_b:
        fails.append("trainable ranges differ between the render and the attempt view")
    if set(w_a) != set(w_b) or any(not np.array_equal(w_a[k], w_b[k]) for k in w_a):
        fails.append("rank weights differ between the render and the attempt view")
    extra = {"rows": None if tag_a is None else int(len(tag_a))}
    const_a, const_b = (isinstance(why, ConstantQualityError) for why in (why_a, why_b))
    for name, why, const in (("render", why_a, const_a), ("attempt view", why_b, const_b)):
        if why is not None and not const:
            fails.append(f"the tag builder refused the {name}: {why}")
    if (why_a is None) != (why_b is None):
        fails.append(f"the tag builder refused only one corpus: render {why_a!s}, attempt view {why_b!s}")
    elif const_a and const_b:
        extra["tags_not_built"] = "the quality label is constant over this corpus (H(q) = 0): " + str(why_a)
    elif why_a is not None:
        pass                                    # both refused, at least one not for H(q) = 0: failed above
    elif tag_a.shape != tag_b.shape:
        fails.append(f"the render enumerates {len(tag_a)} rows, the attempt view {len(tag_b)}")
    elif not np.array_equal(tag_a, tag_b):
        fails.append(f"tags differ on {int((tag_a != tag_b).sum())} rows between the render and the attempt view")
    return _result(fails, **extra)


def _gripper_lock(tasks: list[Path]) -> dict:
    """Whether every episode was rendered with the locked 2F-85 (the sidecars' lock commit), and which.
    The gripper gate measures pad separations, which an unlocked 2F-85-shaped model can match, so a pass
    certifies the locked model only when this says `locked`."""
    commits = {json.loads(side.read_text(encoding="utf-8")).get("gripper", {}).get("lock_commit")
               for task in tasks for side in sorted((task / "meta" / "axis_episodes").glob("episode_*.json"))}
    return {"locked": bool(commits) and None not in commits, "commits": sorted(c for c in commits if c)}


def run_gates(out_root: Path, *, droid8d_root: Path | None = None, render_store: Path | None = None,
              attempt_curves_dir: Path | None = None, open_separation: float = rig.GRIPPER_OPEN_SEPARATION_M,
              closed_separation: float = rig.GRIPPER_CLOSED_SEPARATION_M) -> dict:
    from axis_qvla.realrobot.render.labels import read_attempt_store
    from axis_qvla.realrobot.render.report import droid8d_velocity, joint_jumps
    label_args = (droid8d_root, render_store, attempt_curves_dir)
    if any(a is not None for a in label_args) and any(a is None for a in label_args):
        raise ValueError("the label/tag gate needs all three of droid8d_root, render_store and attempt_curves_dir "
                         "(--droid8d, --curves and --attempt-curves)")
    tasks = list(writer.task_dirs(out_root).values())
    if not tasks:
        raise FileNotFoundError(f"no task_* directories under {out_root}")
    cameras = load_profile(Path(out_root) / PROFILE_FILE).cameras
    report: dict = {"tasks": {}}
    for task in tasks:
        report["tasks"][task.name.split("_")[1]] = {
            "structure": gate_structure(task), "intrinsics": gate_intrinsics(task, cameras), "home": gate_home(task),
            "tcp": gate_tcp(task),
            "gripper": gate_gripper(task, open_separation=open_separation, closed_separation=closed_separation)}
    report["gripper_lock"] = _gripper_lock(tasks)
    report["reported"] = {
        "tcp": {t.name.split("_")[1]: {str(e): r for e, r in tcp_reported(t).items()} for t in tasks},
        "joint_jump": {t.name.split("_")[1]: {str(e): r for e, r in joint_jumps(t).items()} for t in tasks}}
    if droid8d_root is not None:
        report["label_tag"] = gate_label_tag(droid8d_root, render_store, read_attempt_store(attempt_curves_dir))
        report["reported"]["droid8d_velocity"] = droid8d_velocity(droid8d_root)
    (Path(out_root) / "render_gates.json").write_text(json.dumps(report, indent=1))
    failed = [f"task {t} {g}" for t, gs in report["tasks"].items() for g, r in gs.items() if not r["ok"]]
    if "label_tag" in report and not report["label_tag"]["ok"]:
        failed.append("label_tag")
    if failed:
        raise GateError(f"hard gates failed: {', '.join(failed)}")
    return report
