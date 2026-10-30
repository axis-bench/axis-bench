"""Phase curves for a rendered corpus: attempt-keyed in, training-episode-keyed out.

Render tier (numpy, pandas). The phase reward is a fact of the recorded trajectory: the label pipeline
computes it from the source trace and places it on the render frame base (recorded row k is frame
rint((t_k - t_0) * 15) + 2; rows before the first anchor are invalid). It arrives keyed by (task_id,
attempt_id) in an attempt store -- `task_<tid>.npz` with `attempt_<aid>` = float32 (2, T) [curve; valid]
and `t0_<aid>` unless neutral -- and leaves as the REALALIGN curve store (axis_qvla.realrobot.realign.curves),
keyed by the episode index each attempt received in THIS corpus. Nothing is keyed by position.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from axis_qvla.realrobot.realign.curves import (PhaseCurve, load_export_curves, read_attempt_store, write_attempt_store,
                                           write_curve_store)


def attempt_store_from_export(curves_dir: Path, alignment_map: Path, out: Path) -> int:
    """Curves keyed by export index, keyed by attempt through their alignment map."""
    by_training = load_export_curves(curves_dir, alignment_map)
    doc = json.loads(Path(alignment_map).read_text())
    attempt: dict[tuple[int, int], int] = {}
    seen: set[tuple[int, int]] = set()
    for k, v in doc["episodes"].items():
        tid, ep, aid = int(k.split("--")[0].split("_")[1]), int(k.split("--")[1]), int(v["attempt_id"])
        if (tid, aid) in seen:      # two curves for one attempt: one of them would be dropped silently
            raise ValueError(f"task {tid} attempt {aid} is mapped to two training episodes")
        seen.add((tid, aid))
        attempt[(tid, ep)] = aid
    curves = {(tid, attempt[(tid, ep)]): c for (tid, ep), c in by_training.items()}
    write_attempt_store(out, curves)
    return len(curves)


def render_curve_store(corpus_root: Path, attempt_curves: dict[tuple[int, int], PhaseCurve], out: Path) -> int:
    roots = json.loads((Path(corpus_root) / "roots.json").read_text())
    base = Path(corpus_root) / roots.get("base", ".")
    keyed: dict[tuple[int, int], PhaseCurve] = {}
    for tid, rel in roots["roots"].items():
        meta = pd.concat([pd.read_parquet(p) for p in sorted((base / rel / "meta" / "episodes").rglob("*.parquet"))])
        for ep, length, aid in zip(meta["episode_index"], meta["length"], meta["attempt_id"]):
            key = (int(tid), int(aid))
            if key not in attempt_curves:
                raise KeyError(f"no phase curve for task {tid} attempt {aid}")
            c = attempt_curves[key]
            if len(c.curve) != int(length):
                raise ValueError(f"task {tid} attempt {aid}: curve of {len(c.curve)} rows for {int(length)} frames")
            keyed[(int(tid), int(ep))] = c
    write_curve_store(out, keyed)
    return len(keyed)
