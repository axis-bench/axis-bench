"""The flat LeRobot views of the quality labels and training weights: frame plans and per-frame
AWR weights.

OFFLINE TIER: numpy + pandas only.

SmolVLA and X-VLA train from the merged LeRobot dataset, whose frame index is the concatenation of
the per-task pretraining data in task-id order -- the same flat index space the pi0.5 loader
samples. Their fork consumes dumb arrays keyed by that index, so these files carry no logic of their
own:

  rows   `lerobot_rows.npz` (key `rows`, int64): every trainable (non-idle) frame, in the walk
         order of `build_index_schedule.episode_rows`. With a subset ranges file it writes the
         scaling plans `lerobot_rows_s{1000,2500}.npz` the same way.
  flat   `lerobot_awr_flat_<reward>.npz` (key `weights`, float32, one slot per pretraining
         frame): the AWR weight of each trainable frame, NaN on every frame no plan samples -- so
         a frame the row plan never names cannot be weighted by accident.
  ft-rows `ft_rows_all.npz`: `arange(n)` over the finetune dataset, which keeps every frame.

PROVENANCE. The released rows and chunk-level flats come from openpi's `plan_rows_from_roots`,
the subset plans from `build_index_schedule.episode_rows` (then sorted). This module walks the
pretraining data with the primitives `episode_rows` uses (`_ordered_roots`, `_episode_offsets`,
`_frames_in_spans`), which agree with openpi's planner over the whole pretraining data, and
reproduces all eight released files byte for byte.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from axis_qvla.signals.methods import schedule as bis
from axis_qvla.signals.blocks.awr import episode_key


def trainable_blocks(roots_path, ranges_path) -> tuple[list[tuple[str, np.ndarray, np.ndarray, int]], int]:
    """`[(episode_key, global_rows, local_frames, length), ...]` and the corpus row count.

    The walk is `episode_rows`' own, minus the weights: an episode with no trainable frame is
    skipped, and an episode's rows sit at `global_base + dataset_from_index + frame`.
    """
    ranges = json.loads(Path(ranges_path).read_text())
    blocks = []
    global_base = 0
    for task_id, root in bis._ordered_roots(roots_path):
        episodes, total = bis._episode_offsets(root)
        for ep_index in sorted(episodes):
            local_start, length = episodes[ep_index]
            key = episode_key(task_id, ep_index)
            frames = bis._frames_in_spans(ranges.get(key, []), length)
            if frames.size:
                blocks.append((key, global_base + local_start + frames, frames, int(length)))
        global_base += total
    return blocks, global_base


def plan_rows(roots_path, ranges_path) -> np.ndarray:
    """Every trainable frame's flat index, in walk order (strictly increasing for a sorted
    ranges file, which the refusal below insists on -- the fork treats `rows` as a set)."""
    blocks, _ = trainable_blocks(roots_path, ranges_path)
    rows = (np.concatenate([b[1] for b in blocks]) if blocks else np.empty(0)).astype(np.int64)
    if rows.size > 1 and not bool((np.diff(rows) > 0).all()):
        raise ValueError(f"{ranges_path} yields rows that are not strictly increasing; the spans "
                         f"overlap or are out of order, so a row would be planned twice")
    return rows


def flat_weights(roots_path, ranges_path, weights_path) -> np.ndarray:
    """One float32 slot per corpus frame: the episode's weight on a trainable frame, NaN elsewhere."""
    payload = json.loads(Path(weights_path).read_text())
    by_key = payload["weights"] if isinstance(payload, dict) and "weights" in payload else payload
    blocks, n_rows = trainable_blocks(roots_path, ranges_path)
    declared = payload.get("n_rows") if isinstance(payload, dict) else None
    if declared is not None and int(declared) != int(n_rows):
        raise ValueError(f"{weights_path} describes {declared} pretraining rows but the roots walk "
                         f"enumerates {n_rows}; the two describe different pretraining data")
    flat = np.full(int(n_rows), np.nan, dtype=np.float32)
    for key, rows, frames, length in blocks:
        if key not in by_key:
            raise ValueError(f"{key} has trainable rows but no weight in {weights_path}")
        w = np.asarray(by_key[key], dtype=np.float64)
        if w.size != length:
            raise ValueError(f"{key}: {w.size} weights for a {length}-frame episode")
        if not np.isfinite(w[frames]).all():
            raise ValueError(f"{key}: non-finite weight on a trainable frame")
        flat[rows] = w[frames].astype(np.float32)
    return flat


def _save(path, **arrays) -> Path:
    """`np.savez` (uncompressed, fixed zip timestamps): the locked files were written this way."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **arrays)
    return p


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="axis_qvla.signals.views.lerobot", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("rows", "flat"):
        p = sub.add_parser(name)
        p.add_argument("--roots", required=True, help="roots-index JSON of the pretraining data")
        p.add_argument("--ranges", required=True, help="non-idle (or subset) sample-ranges JSON")
        p.add_argument("--out", required=True)
        p.add_argument("--corpus", default=None,
                       help=f"pretraining-data directory, overriding the roots index's base "
                            f"(${bis.CORPUS_ROOT_ENV})")
    sub.choices["flat"].add_argument("--weights", required=True, help="AWR weights JSON")
    ft = sub.add_parser("ft-rows")
    ft.add_argument("--n-rows", type=int, required=True, help="frames in the finetune dataset")
    ft.add_argument("--out", required=True)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if getattr(a, "corpus", None):
        os.environ[bis.CORPUS_ROOT_ENV] = str(Path(a.corpus))
    if a.cmd == "rows":
        rows = plan_rows(a.roots, a.ranges)
        _save(a.out, rows=rows)
        print(json.dumps({"out": str(a.out), "n_rows": int(rows.size)}))
    elif a.cmd == "flat":
        flat = flat_weights(a.roots, a.ranges, a.weights)
        _save(a.out, weights=flat)
        print(json.dumps({"out": str(a.out), "total_frames": int(flat.size),
                          "n_rows": int(np.isfinite(flat).sum())}))
    else:
        if a.n_rows <= 0:
            raise ValueError("--n-rows must be positive")
        _save(a.out, rows=np.arange(a.n_rows, dtype=np.int64))
        print(json.dumps({"out": str(a.out), "n_rows": int(a.n_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
