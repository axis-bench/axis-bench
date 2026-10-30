"""Tail-10 chunk-start ranges: the last 10 frames of an episode never start an action chunk.

WHY: a chunk that starts in an episode's last frames runs past its end, where the loader repeats the
last frame -- a label of holding still. DROID's recipe therefore never starts a chunk in the last
10 frames of a range (openpi's filter_last_n_in_ranges = 10); the frames themselves stay, as the
targets of earlier chunks.

This is the ONLY frame rule the FR3 stage-2 configs applied: their filter file (openpi's
`DataConfig.filter_dict_path`) holds exactly one range [0, L - 10] for each of the 392 exported
episodes, 58,760 chunk starts. The idle filter had already run, physically, at export
(axis_qvla.realrobot.realdata.idle_filter). openpi's script that wrote that file would also drop idle runs and
episodes under 16 frames; neither occurs on the exported dataset, so this uniform rule is what the
file holds, byte for byte (axis_qvla/realrobot/realdata/validate_fr3.py). An episode of L <= 10 frames
has no start and gets an empty list, as in openpi.

Format: json.dump of {str(episode_index): [[start, end]]} with the default separators, keys in
ascending episode order, no trailing newline -- what openpi's script writes.

Rewritten from the dataset producers' FR3 data pipeline (credited in NOTICE). Stdlib only.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

TAIL = 10


def tail_ranges(lengths: Mapping[int, int], n: int = TAIL) -> dict[str, list[list[int]]]:
    """{episode: [[0, L - n]]}, or [] when the episode has no more than n frames."""
    if n < 0:
        raise ValueError(f"tail length must be >= 0, got {n}")
    out: dict[str, list[list[int]]] = {}
    for ep in sorted(lengths):
        length = int(lengths[ep])
        if length < 0:
            raise ValueError(f"episode {ep}: negative length {length}")
        out[str(ep)] = [[0, length - n]] if length > n else []
    return out


def dumps(ranges: Mapping[str, list[list[int]]]) -> str:
    """The filter file's exact text."""
    return json.dumps(ranges)


def chunk_starts(ranges: Mapping[str, list[list[int]]]) -> int:
    """How many rows may start a chunk."""
    return sum(end - start for spans in ranges.values() for start, end in spans)


def episode_lengths(episodes_jsonl: Path) -> dict[int, int]:
    """{episode_index: length} from a LeRobot v2.1 meta/episodes.jsonl."""
    out: dict[int, int] = {}
    with Path(episodes_jsonl).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            ep = int(rec["episode_index"])
            if ep in out:
                raise ValueError(f"{Path(episodes_jsonl).name}: episode {ep} listed twice")
            out[ep] = int(rec["length"])
    return out
