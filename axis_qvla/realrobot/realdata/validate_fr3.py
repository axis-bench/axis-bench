"""Reproduce the FR3 real-finetune data facts from the public data, with axis_qvla.realrobot.realdata.

    PYTHONPATH=$PWD $OFFLINE axis_qvla/realrobot/realdata/validate_fr3.py \
        --recording <the 250-episode FR3 recording, any LeRobot v2.1 copy with its state/actions parquet> \
        --published <the published 392-episode filtered FR3 dataset> \
        --tail-reference <the tail-10 filter file the stage-2 configs used> \
        [--work <scratch dir for the export check>] [--stats]

Each check prints PASS or FAIL; the exit code is 1 if any fails.
  segments  the idle filter over the recording: 250 episodes / 66,463 frames -> 392 / 62,680, and the segments,
            criterion and constants equal the published meta/source_segments.json
  export    (--work) the export of the recording's parquet columns: episodes.jsonl byte-equal to the
            published one, every non-image column equal row for row, every non-image statistic equal
  tail      tail-10 of the published episodes: 58,760 chunk starts, byte-equal to --tail-reference
            (md5 9e47e3c8...), and the same bytes from the export's own episode lengths
  flags     the recording's `done` is True on exactly the last frame of every episode, and
            `is_success` and `intervene_flag` are constant True
  stats     (--stats) every line of the published meta/episodes_stats.jsonl, images included,
            recomputed from its parquet: byte-equal
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

from axis_qvla.realrobot.realdata.export import dataset_segments, export, provenance
from axis_qvla.realrobot.realdata.lerobot_v21 import VIDEO, Dataset, RealdataError, episode_stats, read_jsonl
from axis_qvla.realrobot.realdata.tail import chunk_starts, dumps, episode_lengths, tail_ranges


@dataclasses.dataclass(frozen=True)
class Facts:
    recording: tuple[int, int]
    published: tuple[int, int]
    chunk_starts: int
    tail10_md5: str


# The published FR3 numbers: 392 idle-filtered SEGMENTS of 250 recordings.
FR3 = Facts((250, 66_463), (392, 62_680), 58_760, "9e47e3c8f0fe9207a7f34f6d84147fb9")


class Checks:
    def __init__(self) -> None:
        self.failed: list[str] = []

    def __call__(self, name: str, ok: bool, detail: str) -> bool:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
        if not ok:
            self.failed.append(name)
        return ok


def _size(ds: Dataset) -> tuple[int, int]:
    return len(ds.episodes), sum(int(e["length"]) for e in ds.episodes)


def check_segments(check: Checks, rec: Dataset, pub: Dataset, *, facts: Facts = FR3) -> list:
    check("recording size", _size(rec) == facts.recording, f"{_size(rec)} (expected {facts.recording})")
    segments = dataset_segments(rec)
    got = (len(segments), sum(s.length for s in segments))
    check("idle split", got == facts.published, f"{facts.recording} -> {got} (expected {facts.published})")
    ours = provenance(rec.root.name, segments)
    theirs = json.loads((pub.root / "meta" / "source_segments.json").read_text(encoding="utf-8"))
    for key in ("criterion", "constants", "episodes"):
        check(f"source_segments.{key}", ours[key] == theirs[key], "equal" if ours[key] == theirs[key] else "differs")
    return segments


def check_flags(check: Checks, rec: Dataset) -> None:
    import numpy as np

    bad = []
    for e in rec.episodes:
        ep = int(e["episode_index"])
        t = rec.read_episode(ep, ["done", "is_success", "intervene_flag"])
        done = np.asarray(t.column("done").to_pylist(), dtype=bool)
        last = np.zeros(len(done), dtype=bool)
        last[-1:] = True
        if not (np.array_equal(done, last) and all(t.column(c).to_pylist() == [True] * t.num_rows
                                                   for c in ("is_success", "intervene_flag"))):
            bad.append(ep)
    check("flags", not bad, f"{len(rec.episodes) - len(bad)}/{len(rec.episodes)} episodes: done on the last frame "
                            f"only, is_success and intervene_flag constant True" + (f"; first bad {bad[:5]}" if bad else ""))


def check_export(check: Checks, rec: Dataset, pub: Dataset, segments: list, work: Path, *,
                 facts: Facts = FR3) -> Path:
    """Export the recording's parquet columns (its videos are not in the parquet) and compare."""
    view = dataclasses.replace(rec, info={**rec.info, "features": {k: f for k, f in rec.features.items()
                                                                     if f["dtype"] != VIDEO}})
    out = work / "export"
    done = export(view, segments, out)
    check("export size", (done["episodes"], done["frames"]) == facts.published, f"{done['episodes']} / {done['frames']}")
    same = (out / "meta" / "episodes.jsonl").read_bytes() == (pub.root / "meta" / "episodes.jsonl").read_bytes()
    check("episodes.jsonl bytes", same, "identical" if same else "differ")
    ours, theirs = read_jsonl(out / "meta" / "episodes_stats.jsonl"), read_jsonl(pub.root / "meta" / "episodes_stats.jsonl")
    exported, cols = Dataset.open(out), list(view.features)
    bad_rows, bad_stats = [], []
    for ep, (o, t) in enumerate(zip(ours, theirs, strict=True)):
        a, b = exported.read_episode(ep), pub.read_episode(ep, cols)
        if a.column_names != cols or not all(a.column(c).equals(b.column(c)) for c in cols):
            bad_rows.append(ep)
        if any(o["stats"][k] != t["stats"][k] for k in o["stats"]):
            bad_stats.append(ep)
    check("rows", not bad_rows, f"{len(ours) - len(bad_rows)}/{len(ours)} episodes equal on {len(view.features)} columns")
    check("non-image stats", not bad_stats, f"{len(ours) - len(bad_stats)}/{len(ours)} episodes equal")
    return out


def check_tail(check: Checks, pub: Dataset, reference: Path, export_dir: Path | None, *,
               facts: Facts = FR3) -> None:
    ranges = tail_ranges(episode_lengths(pub.root / "meta" / "episodes.jsonl"))
    text = dumps(ranges).encode()
    check("tail-10 starts", chunk_starts(ranges) == facts.chunk_starts, f"{chunk_starts(ranges)} (expected {facts.chunk_starts})")
    check("tail-10 md5", hashlib.md5(text).hexdigest() == facts.tail10_md5, hashlib.md5(text).hexdigest())
    check("tail-10 bytes", text == Path(reference).read_bytes(), f"against {Path(reference).name}")
    if export_dir is not None:
        own = dumps(tail_ranges(episode_lengths(export_dir / "meta" / "episodes.jsonl"))).encode()
        check("tail-10 from our export", own == text, "identical" if own == text else "differs")


def check_stats(check: Checks, pub: Dataset) -> None:
    lines = (pub.root / "meta" / "episodes_stats.jsonl").read_text(encoding="utf-8").splitlines()
    bad = [ep for ep, line in enumerate(lines)
           if json.dumps({"episode_index": ep, "stats": episode_stats(pub.read_episode(ep), pub.features)}) != line]
    check("episodes_stats.jsonl lines", not bad, f"{len(lines) - len(bad)}/{len(lines)} byte-equal; first bad {bad[:5]}")


def run(recording: Path, published: Path, tail_reference: Path, *, work: Path | None = None, stats: bool = False,
        facts: Facts = FR3) -> int:
    check = Checks()
    rec, pub = Dataset.open(recording), Dataset.open(published)
    check("published size", _size(pub) == facts.published, f"{_size(pub)}")
    segments = check_segments(check, rec, pub, facts=facts)
    check_flags(check, rec)
    export_dir = check_export(check, rec, pub, segments, work, facts=facts) if work else None
    check_tail(check, pub, tail_reference, export_dir, facts=facts)
    if stats:
        check_stats(check, pub)
    print(f"{'ALL PASS' if not check.failed else 'FAILED: ' + ', '.join(check.failed)}")
    return 1 if check.failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", type=Path, required=True)
    ap.add_argument("--published", type=Path, required=True)
    ap.add_argument("--tail-reference", type=Path, required=True)
    ap.add_argument("--work", type=Path, help="empty scratch directory for the export check")
    ap.add_argument("--stats", action="store_true", help="recompute every published statistic (decodes images)")
    a = ap.parse_args(argv)
    try:
        return run(a.recording, a.published, a.tail_reference, work=a.work, stats=a.stats)
    except (RealdataError, FileNotFoundError, FileExistsError) as exc:
        print(f"FAIL  {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
