"""Per-episode phase-reward curves, keyed by the TRAINING episode index.

OFFLINE TIER: numpy only.

Curves enter keyed by attempt (an attempt store, or export-keyed curves through their alignment map) and are
kept keyed by the training episode in one store per task: `ep_<i>` (float32 curve), `valid_<i>` (bool) and
`t0_<i>` (contact onset; absent for a neutral episode). The `ep_`/`t0_` names are those of
`build_awr_weights --dump-curves`. An attempt store holds the same curves keyed by
`(task_id, attempt_id)` before a render gives each attempt its episode: `task_<tid>.npz` with
`attempt_<aid>` = float32 (2, T) [curve; valid] and `t0_<aid>` unless neutral.
"""
from __future__ import annotations

import dataclasses
import fnmatch
import json
import os
import re
from pathlib import Path

import numpy as np

_MAP_KEY = re.compile(r"task_(\d+)--(\d+)")
_EXPORT_CURVE = re.compile(r"phase_ep_(\d+)")
_STORE_CURVE = re.compile(r"ep_(\d+)")
_STORE_FILE = re.compile(r"task_(\d+)_phase_curves")
_ATTEMPT = re.compile(r"attempt_(\d+)")


class CurveError(ValueError):
    """A curve that cannot be attached to exactly one training episode, or is malformed."""


@dataclasses.dataclass(frozen=True)
class PhaseCurve:
    """One episode's phase reward: the curve, its validity, and the contact onset (None = neutral)."""

    curve: np.ndarray
    valid: np.ndarray
    t0: int | None

    def __post_init__(self):
        if self.curve.ndim != 1 or self.curve.shape != self.valid.shape:
            raise CurveError(f"curve {self.curve.shape} and validity {self.valid.shape} must be equal 1-D")
        if self.t0 is not None and not 0 <= int(self.t0) <= len(self.curve):
            raise CurveError(f"t0={self.t0} outside the episode's [0, {len(self.curve)}]")


def _store_files(directory: Path, pattern: str) -> list[Path]:
    """The files of `directory` matching `pattern`, refusing a directory that is absent or holds none:
    either one reads as a store without curves, which then fails for another reason. Listing raises on
    an unreadable directory instead of reading it as empty."""
    d = Path(directory)
    if not d.is_dir():
        raise CurveError(f"{d}: no such directory")
    files = sorted(p for p in d.iterdir() if fnmatch.fnmatchcase(p.name, pattern))
    if not files:
        raise CurveError(f"{d} holds no {pattern} file")
    return files


def _training_index(alignment_map: Path) -> dict[tuple[int, int], tuple[int, int]]:
    """`(task_id, export_episode_index) -> (training episode_index, train_frame_count)`."""
    doc = json.loads(Path(alignment_map).read_text())
    out: dict[tuple[int, int], tuple[int, int]] = {}
    for key, v in doc["episodes"].items():
        m = _MAP_KEY.fullmatch(key)
        if m is None:
            raise CurveError(f"alignment-map key {key!r} is not task_<id>--<episode>")
        export = (int(m.group(1)), int(v["export_episode_index"]))
        if export in out:
            raise CurveError(f"two training episodes claim task {export[0]} export episode {export[1]}")
        out[export] = (int(m.group(2)), int(v["train_frame_count"]))
    return out


def load_export_curves(curves_dir: Path, alignment_map: Path) -> dict[tuple[int, int], PhaseCurve]:
    """Export-keyed curves (`task_<tid>.npz`: `phase_ep_<x>` (2, T), `t0_ep_<x>`), keyed by training episode.

    Refuses a curve the map does not know, a length that disagrees with the map, and a map episode
    left without a curve: each is a silent mis-attachment otherwise.
    """
    index = _training_index(alignment_map)
    out: dict[tuple[int, int], PhaseCurve] = {}
    for f in _store_files(curves_dir, "task_*.npz"):
        tid = int(f.stem.split("_")[1])
        with np.load(f, allow_pickle=False) as z:
            names = set(z.files)
            for name in sorted(names):
                m = _EXPORT_CURVE.fullmatch(name)
                if m is None:
                    continue
                x = int(m.group(1))
                if (tid, x) not in index:
                    raise CurveError(f"task {tid} export episode {x} is not in the alignment map")
                ep, length = index[(tid, x)]
                arr = z[name]
                if arr.shape != (2, length):
                    raise CurveError(f"task {tid} export episode {x}: curve {arr.shape}, map says (2, {length})")
                if not np.isin(arr[1], (0.0, 1.0)).all():
                    raise CurveError(f"task {tid} export episode {x}: validity is not binary")
                t0 = int(z[f"t0_ep_{x}"]) if f"t0_ep_{x}" in names else None
                out[(tid, ep)] = PhaseCurve(arr[0].astype(np.float32), arr[1] > 0.5, t0)
    missing = sorted({(t, e) for (t, _x), (e, _n) in index.items()} - set(out))
    if missing:
        raise CurveError(f"{len(missing)} alignment-map episodes have no curve, e.g. {missing[:3]}")
    return out


def write_curve_store(out_dir: Path, curves: dict[tuple[int, int], PhaseCurve]) -> None:
    """One `task_<tid>_phase_curves.npz` per task, keyed by training episode."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_task: dict[int, dict[int, PhaseCurve]] = {}
    for (tid, ep), c in curves.items():
        by_task.setdefault(int(tid), {})[int(ep)] = c
    for tid, eps in sorted(by_task.items()):
        arrs: dict[str, np.ndarray] = {}
        for ep, c in sorted(eps.items()):
            arrs[f"ep_{ep}"] = np.asarray(c.curve, np.float32)
            arrs[f"valid_{ep}"] = np.asarray(c.valid, bool)
            if c.t0 is not None:
                arrs[f"t0_{ep}"] = np.int64(c.t0)
        np.savez_compressed(out / f"task_{tid}_phase_curves.npz", **arrs)


def read_curve_store(store_dir: Path) -> dict[tuple[int, int], PhaseCurve]:
    """Inverse of `write_curve_store`. A curve stored without `valid_<i>` is refused, not defaulted."""
    out: dict[tuple[int, int], PhaseCurve] = {}
    for f in _store_files(store_dir, "task_*_phase_curves.npz"):
        m = _STORE_FILE.fullmatch(f.stem)
        if m is None:
            continue
        tid = int(m.group(1))
        with np.load(f, allow_pickle=False) as z:
            names = set(z.files)
            for name in sorted(names):
                mm = _STORE_CURVE.fullmatch(name)
                if mm is None:
                    continue
                ep = int(mm.group(1))
                if f"valid_{ep}" not in names:
                    raise CurveError(f"{f.name}: ep_{ep} has no valid_{ep}; a store needs every mask")
                t0 = int(z[f"t0_{ep}"]) if f"t0_{ep}" in names else None
                out[(tid, ep)] = PhaseCurve(z[name].astype(np.float32), z[f"valid_{ep}"].astype(bool), t0)
    return out


def read_dumped_curve_store(store_dir: Path, roots_path: Path) -> dict[tuple[int, int], PhaseCurve]:
    """A store written by the pinned AWR builder's --dump-curves: `ep_<i>` and `t0_<i>` only.

    Its validity mask is the corpus's own `reward_valid` column -- the mask the dumping builder read --
    and an episode the builder found unscorable is not dumped at all (it was weighted neutral). Both are
    restored here from the corpus, and only here: a store that carries its masks goes through
    read_curve_store, which still refuses a missing one.
    """
    import pandas as pd
    from axis_qvla.signals.methods import schedule as bis

    dumped: dict[tuple[int, int], tuple[np.ndarray, int | None]] = {}
    for f in _store_files(store_dir, "task_*_phase_curves.npz"):
        m = _STORE_FILE.fullmatch(f.stem)
        if m is None:
            continue
        with np.load(f, allow_pickle=False) as z:
            if any(n.startswith("valid_") for n in z.files):
                raise CurveError(f"{f.name} carries validity masks: not a dump; read it with read_curve_store")
            for name in z.files:
                mm = _STORE_CURVE.fullmatch(name)
                if mm is not None:
                    ep = int(mm.group(1))
                    dumped[(int(m.group(1)), ep)] = (z[name].astype(np.float32),
                                                     int(z[f"t0_{ep}"]) if f"t0_{ep}" in z.files else None)
    out: dict[tuple[int, int], PhaseCurve] = {}
    for tid, root in bis._ordered_roots(roots_path):
        parts = sorted(Path(root).glob("data/**/*.parquet"))
        if not parts:
            raise CurveError(f"task {tid}: no data parquet with reward_valid under {root}")
        df = pd.concat([pd.read_parquet(p, columns=["episode_index", "frame_index", "reward_valid"]) for p in parts])
        for ep, g in df.groupby("episode_index"):
            valid = g.sort_values("frame_index")["reward_valid"].to_numpy(bool)
            key = (int(tid), int(ep))
            if key in dumped:
                curve, t0 = dumped.pop(key)
                if len(curve) != len(valid):
                    raise CurveError(f"task_{tid}--{ep}: dumped curve of {len(curve)} rows for {len(valid)} frames")
                out[key] = PhaseCurve(curve, valid, t0)
            else:
                out[key] = PhaseCurve(np.zeros(len(valid), np.float32), np.zeros(len(valid), bool), None)
    if dumped:
        raise CurveError(f"{len(dumped)} dumped curves name episodes the corpus does not have, e.g. {sorted(dumped)[:3]}")
    return out


def write_attempt_store(out: Path, curves: dict[tuple[int, int], PhaseCurve]) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    by_task: dict[int, dict] = {}
    for (tid, aid), c in sorted(curves.items()):
        arrs = by_task.setdefault(int(tid), {})
        arrs[f"attempt_{aid}"] = np.stack([np.asarray(c.curve, np.float32), np.asarray(c.valid, np.float32)])
        if c.t0 is not None:
            arrs[f"t0_{aid}"] = np.int64(c.t0)
    for tid, arrs in by_task.items():
        # Whole or not at all: `qvla realrobot labels` keeps a task whose file exists, so a half-written one
        # must never carry the final name. The temporary name does not match read_attempt_store's glob.
        part = out / f".task_{tid}.npz.partial"
        try:
            with open(part, "wb") as f:
                np.savez_compressed(f, **arrs)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        os.replace(part, out / f"task_{tid}.npz")


def read_attempt_store(store: Path) -> dict[tuple[int, int], PhaseCurve]:
    """Inverse of `write_attempt_store`. An absent directory is refused, naming it, and an unreadable one
    raises from its listing: neither reads as a store without curves. An existing empty directory is an
    empty store."""
    d = Path(store)
    try:
        files = sorted(p for p in d.iterdir() if fnmatch.fnmatchcase(p.name, "task_*.npz"))
    except (FileNotFoundError, NotADirectoryError):
        raise CurveError(f"{d}: no such directory; give the attempt store `realign phase --out` writes") from None
    out: dict[tuple[int, int], PhaseCurve] = {}
    for f in files:
        tid = int(f.stem.split("_")[1])
        with np.load(f, allow_pickle=False) as z:
            names = set(z.files)
            for name in sorted(names):
                m = _ATTEMPT.fullmatch(name)
                if m is None:
                    continue
                aid, arr = int(m.group(1)), z[name]
                t0 = int(z[f"t0_{aid}"]) if f"t0_{aid}" in names else None
                out[(tid, aid)] = PhaseCurve(arr[0].astype(np.float32), arr[1] > 0.5, t0)
    return out
