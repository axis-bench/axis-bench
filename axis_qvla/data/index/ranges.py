"""Non-idle sample ranges computed from a tree of per-task LeRobot datasets (the pretraining data or
a finetune set).

Offline environment.

    python -m axis_qvla.data.index.ranges --corpus <dir> --out <ranges>.json

The idle signal is `observation.state` (8-D DROID state; its forward difference times the frame
rate IS the action label), so the ranges are derivable from the published data alone. The filter is
axis_qvla.labels.idle_filter.nonidle_ranges with the parameters recorded in
axis_qvla.labels.idle_filter.IDLE_FILTER (forward-aligned: a frame is idle when the action trained at it
is ~zero).

KEY FORMAT is `task_<task_id>--<episode_index>`, the key every label artifact joins on. Tasks are
visited in sorted DIRECTORY-NAME order (task_1000 before task_800), which is the key order of the
written file; a frame index keeps that order so it can write the file back byte for byte.
"""

from __future__ import annotations

import argparse
import json
import pathlib


def task_ranges(task_dir: pathlib.Path, *, forward: bool = True, **params) -> dict[str, list]:
    """`{episode_key: [[start, end], ...]}` for one derived task tree."""
    from axis_qvla.labels.idle_filter import nonidle_ranges
    from axis_qvla.data.lerobot_local import DatasetSignals

    task_id = int(task_dir.name.split("_")[1])
    out: dict[str, list] = {}
    for ep_idx, signal in DatasetSignals.load(task_dir).episodes():
        ranges = nonidle_ranges(signal, forward=forward, **params)
        out[f"task_{task_id}--{ep_idx}"] = [[int(s), int(e)] for s, e in ranges]
    return out


def build(corpus: pathlib.Path, *, forward: bool = True, **params) -> dict:
    tasks = sorted(corpus.glob("task_*"))
    if not tasks:
        raise ValueError(f"{corpus} holds no task_* trees")
    ranges: dict[str, list] = {}
    kept = total = 0
    for t in tasks:
        r = task_ranges(t, forward=forward, **params)
        ranges.update(r)
        for spans in r.values():
            kept += sum(e - s for s, e in spans)
    # Retention is the number worth printing: a filter that silently keeps everything, or almost
    # nothing, looks identical in the output file.
    for t in tasks:
        total += int(json.loads((t / "meta" / "info.json").read_text())["total_frames"])
    return {"ranges": ranges, "n_episodes": len(ranges), "kept_rows": kept, "total_rows": total}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="axis_qvla.data.index.ranges", description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    corpus = pathlib.Path(a.corpus)
    res = build(corpus)
    pathlib.Path(a.out).write_text(json.dumps(res["ranges"]))
    frac = res["kept_rows"] / res["total_rows"] if res["total_rows"] else 0.0
    print(f"{res['n_episodes']} episodes -> {a.out}")
    print(f"  kept {res['kept_rows']:,} of {res['total_rows']:,} rows ({frac:.1%} retention)")
    if not 0.2 <= frac <= 0.98:
        print(f"  WARNING retention {frac:.1%} is outside the range a sane idle filter produces; "
              f"check the signal before training on this")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
