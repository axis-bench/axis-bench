"""`python -m axis_qvla.realrobot.realdata {idle-ranges,tail,crop224,training-set}`: the FR3 real-finetune data steps.

    idle-ranges   the FR3 idle filter over a LeRobot v2.1 dataset: write its kept segments as a
                  source_segments.json and, with --export, the filtered dataset itself (one kept
                  segment per episode) -- the export-time step behind the published 392 episodes
    tail          the tail-10 chunk-start ranges of a dataset: openpi's filter_dict_path file, the
                  one frame rule the stage-2 configs applied
    crop224       rebuild a letterboxed dataset's images as centre-crop-224 from its recordings
    training-set  a DROID-layout recording in the per-task layout the π0.5 loader reads, with its
                  tail-10 ranges

Paths given here are only read or written, never stored: what the tools record is the source
dataset's directory name and manifest-relative recording paths.

A refusal -- a directory that is not a LeRobot v2.1 dataset, an export of a video-backed copy, an
output that exists, a recordings manifest with gaps, a recording without the DROID columns -- is one
FAIL line on stderr and exit 2, and it comes before anything is written.

Offline tier.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from axis_qvla.realrobot.realdata import tail as tail_mod
from axis_qvla.realrobot.realdata.crop224 import WINDOW
from axis_qvla.realrobot.realdata.idle_filter import FR3_RULE, IdleRule
from axis_qvla.realrobot.realdata.lerobot_v21 import Dataset, RealdataError


def _idle_ranges(a: argparse.Namespace) -> int:
    from axis_qvla.realrobot.realdata.export import check_exportable, dataset_segments, export, provenance, write_provenance

    ds = Dataset.open(a.dataset)
    if a.export:
        check_exportable(ds, a.export)
    rule = IdleRule(grip_guard=a.grip_guard, grip_delta=a.grip_delta)
    segments = dataset_segments(ds, rule)
    write_provenance(a.out, provenance(ds.root.name, segments, rule))
    before = sum(int(e["length"]) for e in ds.episodes)
    after = sum(s.length for s in segments)
    print(f"{len(ds.episodes)} episodes / {before} frames -> {len(segments)} episodes / {after} frames "
          f"(dropped {before - after} = {100 * (before - after) / max(before, 1):.1f}%); wrote {a.out}")
    if a.export:
        done = export(ds, segments, a.export, rule)
        print(f"exported {done['episodes']} episodes / {done['frames']} frames to {a.export}")
    return 0


def _tail(a: argparse.Namespace) -> int:
    episodes = Path(a.dataset) / "meta" / "episodes.jsonl"
    if not episodes.is_file():
        raise RealdataError(f"{a.dataset}: no meta/episodes.jsonl; give a LeRobot v2.1 dataset directory")
    ranges = tail_mod.tail_ranges(tail_mod.episode_lengths(episodes), a.n)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tail_mod.dumps(ranges), encoding="utf-8")
    print(f"{len(ranges)} episodes, {tail_mod.chunk_starts(ranges)} chunk starts "
          f"(the last {a.n} frames of each never start a chunk); wrote {out}")
    return 0


def _crop224(a: argparse.Namespace) -> int:
    from axis_qvla.realrobot.realdata.crop224_dataset import rebuild

    report = rebuild(Dataset.open(a.dataset), a.recordings, a.out, a.window, a.episodes)
    cols = [c for ep in report["episodes"] for c in ep["cols"].values()]
    worst = max((c["mse_p95"] for c in cols), default=0.0)
    outside = sum(c["rows_outside_recording"] for c in cols)
    print(f"{len(report['episodes'])} episodes rebuilt at {a.out}; worst p95 match MSE {worst:.2f}, "
          f"{outside} rows outside their recording" + ("; smoke subset: no meta written" if a.episodes else ""))
    return 0


def _training_set(a: argparse.Namespace) -> int:
    from axis_qvla.realrobot.realdata.training_set import write_training_set

    got = write_training_set(a.dataset, a.out, tail=a.tail)
    print(f"{got['episodes']} episodes / {got['frames']} frames of {got['tasks']} instructions -> {a.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m axis_qvla.realrobot.realdata", description=__doc__.split("\n\nOffline tier")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("idle-ranges", help="the idle filter: kept segments, optionally the exported dataset")
    p.add_argument("--dataset", type=Path, required=True, help="LeRobot v2.1 dataset directory (never modified)")
    p.add_argument("--out", type=Path, required=True, help="where to write source_segments.json")
    p.add_argument("--export", type=Path, help="also write the filtered dataset here (must not exist)")
    p.add_argument("--grip-guard", type=int, default=FR3_RULE.grip_guard, help="frames protected each side of a gripper event")
    p.add_argument("--grip-delta", type=float, default=FR3_RULE.grip_delta, help="|gripper command change| that is an event")
    p.set_defaults(run=_idle_ranges)
    p = sub.add_parser("tail", help="tail-10 chunk-start ranges (openpi filter_dict_path file)")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=tail_mod.TAIL, help="frames at the end of an episode that start no chunk")
    p.set_defaults(run=_tail)
    p = sub.add_parser("crop224", help="rebuild a letterboxed dataset's images as centre-crop-224")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--recordings", type=Path, required=True, help="manifest JSON: {episode: {column: relative path}}")
    p.add_argument("--out", type=Path, required=True, help="output dataset directory (must not exist)")
    p.add_argument("--window", type=int, default=WINDOW, help="search +-this many recorded frames per row")
    p.add_argument("--episodes", type=int, nargs="+", help="smoke test: only these episodes, no meta")
    p.set_defaults(run=_crop224)
    p = sub.add_parser("training-set", help="a DROID-layout recording in the per-task layout the π0.5 loader reads")
    p.add_argument("--dataset", type=Path, required=True, help="LeRobot v2.1 recording in the DROID layout (never modified)")
    p.add_argument("--out", type=Path, required=True, help="training-set directory (must not exist)")
    p.add_argument("--tail", type=int, default=tail_mod.TAIL, help="frames at the end of an episode that start no chunk")
    p.set_defaults(run=_training_set)
    a = ap.parse_args(argv)
    try:
        return a.run(a)
    except (RealdataError, FileNotFoundError, FileExistsError) as exc:
        print(f"FAIL  {exc}", file=sys.stderr)
        return 2
