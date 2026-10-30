"""Gates over a built REALALIGN signal directory: every artifact describes the same corpus and rows.

OFFLINE TIER: numpy, pandas.

These gates are structural and exact. Each one checks a property whose violation trains silently on the
wrong thing:

- G1 alignment: every corpus episode has exactly one curve, of exactly its length, and attempt ids are
  unique within a task. A curve attached by position to a re-ordered export breaks this when the
  swapped episodes differ in length, and nothing downstream can see it. The curve store is keyed by
  episode index and carries no attempt id, so G1 cannot catch a swap of equal-length episodes: curves
  are attached by attempt where they enter (`render labels`, and `build --export-curves` through the
  alignment map), never here.
- G2 coverage: curves are finite and in [0, 1] where valid; a neutral episode (no contact onset) carries
  no valid row. Counts are reported, never thresholded.
- G3 artifacts: every weights file covers every episode with the right lengths, is stamped against THIS
  corpus, and describes `n_rows` rows. Tags and schedule describe the same row space; the schedule draws
  only trainable rows from a (20000, 64) drop-top plan over the episode weights.
- G4 labels: pure conditioning (no NO_TAG on a trainable row, 255 exactly on non-trainable rows), one tag
  per episode, unlabelled episodes at the neutral bin, and scorable episodes split into quintiles whose
  counts differ by at most one.

G1/G2 always run; G3 checks each selected run's files and G4 belongs to 'conditioning'.

Float32 semantics are the pinned code's: progress is `w > 1` on the stored float32 weight.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from axis_qvla.signals.methods import schedule as bis
from axis_qvla.signals.provenance import corpus_fingerprint
from axis_qvla.signals.blocks.awr import episode_key
from axis_qvla.signals.blocks.schedule import N_BINS, NEUTRAL_Q
from axis_qvla.signals.blocks.tags import NO_TAG, NOT_TRAINABLE
from axis_qvla.realrobot.realign.curves import CurveError, read_curve_store
from axis_qvla.realrobot.realign.runs import RUNS, check_runs_built

KEY = re.compile(r"task_\d+--\d+")
PLAN = (20000, 64)


class GateError(RuntimeError):
    """At least one gate failed; the message names every failing gate."""


def _corpus(out: Path) -> tuple[Path, list[tuple[int, int, int, int]]]:
    """`(base, [(task_id, episode_index, global_start, length)])` in loader order."""
    roots = out / "roots.json"
    doc = json.loads(roots.read_text())
    base = roots.parent / doc.get("base", ".")
    rows, start = [], 0
    for tid, root in bis._ordered_roots(roots):
        episodes, total = bis._episode_offsets(root)
        for ep, (local, length) in sorted(episodes.items()):
            rows.append((int(tid), int(ep), start + int(local), int(length)))
        start += int(total)
    return base, rows


def _attempt_ids(base: Path) -> dict[int, list[int]]:
    import pandas as pd
    out: dict[int, list[int]] = {}
    for task_dir in sorted(base.glob("task_*")):
        for part in sorted(task_dir.glob("meta/episodes/**/*.parquet")):
            out.setdefault(int(task_dir.name.split("_")[1]), []).extend(
                int(a) for a in pd.read_parquet(part, columns=["attempt_id"])["attempt_id"])
    return out


def _g1(base, corpus, curves) -> dict:
    fails = []
    want = {(t, e): n for t, e, _s, n in corpus}
    for k, n in want.items():
        if k not in curves:
            fails.append(f"no curve for task_{k[0]}--{k[1]}")
        elif len(curves[k].curve) != n:
            fails.append(f"task_{k[0]}--{k[1]}: curve has {len(curves[k].curve)} frames, corpus {n}")
    fails += [f"curve for task_{t}--{e} is not in the corpus" for t, e in sorted(set(curves) - set(want))]
    for tid, ids in _attempt_ids(base).items():
        if len(ids) != len(set(ids)):
            fails.append(f"task {tid}: duplicate attempt ids")
    return {"ok": not fails, "failures": fails[:50], "n_failures": len(fails), "episodes": len(want)}


def _g2(curves) -> dict:
    fails, neutral = [], 0
    for (t, e), c in sorted(curves.items()):
        if c.t0 is None:
            neutral += 1
            if c.valid.any():
                fails.append(f"task_{t}--{e}: neutral (no onset) but has valid rows")
            continue
        v = c.curve[c.valid]
        if not np.isfinite(c.curve).all():
            fails.append(f"task_{t}--{e}: non-finite curve")
        elif v.size and (v.min() < 0.0 or v.max() > 1.0):
            fails.append(f"task_{t}--{e}: valid reward outside [0, 1]")
    n = len(curves)
    return {"ok": not fails, "failures": fails[:50], "n_failures": len(fails), "curves": n - neutral,
            "neutral": neutral, "coverage": (n - neutral) / max(n, 1)}


def _weights_check(path: Path, corpus, fingerprint: str, n_rows: int, fails: list, *, reward_id: str,
                   balanced: bool | None) -> dict:
    doc = json.loads(path.read_text())
    w = doc.get("weights", {})
    name = path.name
    if doc.get("reward_id") != reward_id:
        fails.append(f"{name}: reward_id {doc.get('reward_id')!r}, expected {reward_id!r}")
    if int(doc.get("n_rows", -1)) != n_rows:
        fails.append(f"{name}: n_rows {doc.get('n_rows')} != corpus rows {n_rows}")
    prov = doc.get("provenance") or {}
    if "corpus_fingerprint" not in prov or "params" not in prov:
        fails.append(f"{name}: no full provenance stamp")
    elif prov["corpus_fingerprint"] != fingerprint:
        fails.append(f"{name}: stamped against another corpus")
    if balanced is not None and prov.get("params", {}).get("phase_balanced") is not balanced:
        fails.append(f"{name}: phase_balanced is not {balanced}")
    if balanced and "phase_balance" not in prov:
        fails.append(f"{name}: balanced weights without their scales")
    bad_keys = [k for k in w if not KEY.fullmatch(k)]
    if bad_keys:
        fails.append(f"{name}: malformed keys, e.g. {bad_keys[:3]}")
    for t, e, _s, n in corpus:
        k = episode_key(t, e)
        if k not in w:
            fails.append(f"{name}: no weights for {k}")
        elif len(w[k]) != n:
            fails.append(f"{name}: {k} has {len(w[k])} weights for {n} frames")
    return w


def _g3(out, corpus, base, runs: tuple[str, ...] = RUNS) -> dict:
    """Artifact checks of the selected runs' files: the rank file and tags ('conditioning'), the balanced
    file ('awr'), the episode file and the drop-top schedule ('filtering'). `trainable_rows` comes from
    `ranges.json` alone and is always reported."""
    fails: list[str] = []
    fingerprint = corpus_fingerprint(base)
    n_rows = sum(n for *_x, n in corpus)
    if "conditioning" in runs:
        _weights_check(out / "awr_weights_phase_rank.json", corpus, fingerprint, n_rows, fails,
                       reward_id="phase", balanced=False)
    if "awr" in runs:
        _weights_check(out / "awr_weights_phase.json", corpus, fingerprint, n_rows, fails,
                       reward_id="phase", balanced=True)
    if "filtering" in runs:
        _weights_check(out / "awr_weights_phase_ep.json", corpus, fingerprint, n_rows, fails,
                       reward_id="phase_ep", balanced=None)
    if "conditioning" in runs:
        tags = np.load(out / "quality_phase.npz", allow_pickle=False)
        if tags["tag"].shape != (n_rows,):
            fails.append(f"quality_phase.npz: {tags['tag'].shape[0]} rows, corpus {n_rows}")
    trainable = _trainable_mask(out, corpus, n_rows)
    drop_top_unique = None
    if "filtering" in runs:
        sched = np.load(out / "schedule_drop_top.npz", allow_pickle=False)
        meta = json.loads(str(sched["meta"]))
        if meta.get("mode") != "drop_top" or meta.get("reward_id") != "phase_ep":
            fails.append(f"schedule: mode {meta.get('mode')!r} reward {meta.get('reward_id')!r}")
        if tuple(sched["rows"].shape) != PLAN:
            fails.append(f"schedule: plan {tuple(sched['rows'].shape)}, expected {PLAN}")
        if int(meta.get("n_rows", -1)) != n_rows:
            fails.append(f"schedule: n_rows {meta.get('n_rows')} != corpus rows {n_rows}")
        drawn = np.unique(sched["rows"])
        if drawn.size and (drawn.min() < 0 or drawn.max() >= n_rows or not trainable[drawn].all()):
            fails.append("schedule: draws a row that is not trainable")
        drop_top_unique = meta.get("unique_episodes")
    return {"ok": not fails, "failures": fails[:50], "n_failures": len(fails), "n_rows": n_rows,
            "trainable_rows": int(trainable.sum()), "drop_top_unique_episodes": drop_top_unique}


def _trainable_mask(out, corpus, n_rows) -> np.ndarray:
    ranges = json.loads((out / "ranges.json").read_text())
    mask = np.zeros(n_rows, bool)
    for t, e, start, n in corpus:
        frames = bis._frames_in_spans(ranges.get(episode_key(t, e), []), n)
        mask[start + frames] = True
    return mask


def _g4(out, corpus, runs: tuple[str, ...] = RUNS) -> dict:
    """Pure-conditioning label checks: 'conditioning''s own gate, ok with nothing read when 'conditioning'
    is not selected (only it writes quality_phase.npz)."""
    if "conditioning" not in runs:
        return {"ok": True, "failures": [], "n_failures": 0, "scorable_episode_counts": {}}
    fails: list[str] = []
    n_rows = sum(n for *_x, n in corpus)
    tag = np.load(out / "quality_phase.npz", allow_pickle=False)["tag"]
    meta = json.loads(str(np.load(out / "quality_phase.npz", allow_pickle=False)["meta"]))
    if meta.get("pure_conditioning") is not True:
        fails.append("quality_phase.npz is not marked pure_conditioning")
    trainable = _trainable_mask(out, corpus, n_rows)
    if (tag[trainable] == NO_TAG).any():
        fails.append(f"{int((tag[trainable] == NO_TAG).sum())} trainable rows carry NO_TAG")
    if not np.array_equal(tag == NOT_TRAINABLE, ~trainable):
        fails.append("NOT_TRAINABLE (255) does not mark exactly the non-trainable rows")
    rank = json.loads((out / "awr_weights_phase_rank.json").read_text())["weights"]
    counts = {b: 0 for b in range(1, N_BINS + 1)}
    for t, e, start, n in corpus:
        rows = trainable[start:start + n]
        if not rows.any():
            continue
        q = np.unique(tag[start:start + n][rows])
        if q.size != 1:
            fails.append(f"task_{t}--{e}: {q.size} different tags within one episode")
            continue
        unlabelled = bool(np.all(np.asarray(rank[episode_key(t, e)], np.float32) == 1.0))
        if unlabelled:
            if int(q[0]) != NEUTRAL_Q:
                fails.append(f"task_{t}--{e}: unlabelled episode tagged {int(q[0])}, not {NEUTRAL_Q}")
        elif 1 <= int(q[0]) <= N_BINS:
            counts[int(q[0])] += 1
    if max(counts.values()) - min(counts.values()) > 1:
        fails.append(f"scorable episodes are not split into quintiles: {counts}")
    return {"ok": not fails, "failures": fails[:50], "n_failures": len(fails),
            "scorable_episode_counts": {str(k): v for k, v in counts.items()}}


def run_gates(out_dir: Path, *, write: bool = True, runs: tuple[str, ...] = RUNS) -> dict:
    """Run G1-G4 over `out_dir` and raise `GateError` if any fails. Writes `gates.json` into `out_dir`
    unless `write` is False: a caller that only needs the verdict over a directory it does not own
    (`realign compile`, re-checking a `build --out` it merely reads) passes `write=False` so the
    gates run read-only and never touch a read-only or shared build."""
    out = Path(out_dir)
    check_runs_built(out, runs)
    base, corpus = _corpus(out)
    try:
        curves = read_curve_store(out / "curves")
    except CurveError as exc:
        curves = {}
        report = {"G1": {"ok": False, "failures": [str(exc)], "n_failures": 1}}
    else:
        report = {"G1": _g1(base, corpus, curves)}
    report["G2"] = _g2(curves)
    report["G3"] = _g3(out, corpus, base, runs)
    report["G4"] = _g4(out, corpus, runs)
    if write:
        (out / "gates.json").write_text(json.dumps(report, indent=1))
    failed = [g for g, r in report.items() if not r["ok"]]
    if failed:
        first = "; ".join(f"{g}: {report[g]['failures'][:2]}" for g in failed)
        raise GateError(f"gates failed: {', '.join(failed)} -- {first}")
    return report
