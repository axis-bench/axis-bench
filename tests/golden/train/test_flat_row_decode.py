"""The finetune row space's 86,912 flat rows, decoded over the published bench, are the frames the
openpi line trains them as.

Binding compiles a pi0.5 finetune's tags keyed on each flat row's (episode, frame), read from
axis_qvla.train.ft_bundle.episode_table. The openpi line reads flat row r as the r-th row of its ranges
walk over the roots index (`plan_rows_from_roots`), a row of the concatenated per-task datasets,
and logs the sha256 of that row map (`rows_sha256`, little-endian int64 in flat-row order) when it
binds the bundle. Over the published `ft_demos/splits/qual_v2.{roots,ranges}.json` and the 24
finetune datasets' episode tables this golden requires:

  - the walk keeps 86,912 of the 230,088 frames, and its row map has the sha256 the openpi line
    logs for these files, FLAT_ROWS_SHA256;
  - every flat row's (episode, frame) in the episode table is the episode and frame its dataset
    row holds, so the tags binding compiles land on the frames the line reads;
  - no flat row is its own dataset row: a decode that read concat row r would pass a row count.

Marked slow; `--bench` names a qvla-v1-bench tree (at least `ft_demos/splits/` and each finetune
task's `meta/`), read through tests.conftest.require_option, which fails rather than skips when it
is missing.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from axis_qvla.train import ft_bundle
from tests.conftest import require_option

FLAT_ROWS = 86_912
DATASET_ROWS = 230_088
FLAT_ROWS_SHA256 = "fd3fd494a0287ac4aa84459bf80fbdc891ca7bd06db5530678382c7e4faa61d1"
ROOTS = "ft_demos/splits/qual_v2.roots.json"
RANGES = "ft_demos/splits/qual_v2.ranges.json"


def _walk(bench: pathlib.Path):
    """(dataset row of each flat row, episode of each dataset row, frame of each dataset row), read
    from the files alone: tasks in ascending id, each task's episodes in its meta/episodes order
    (every parquet there, sorted by path), each episode's spans in file order clamped to the
    episode; a task's rows follow the earlier tasks', an episode's the earlier episodes'."""
    import numpy as np
    import pandas as pd

    roots = json.loads((bench / ROOTS).read_text(encoding="utf-8"))
    roots = roots.get("roots", roots)
    spans = json.loads((bench / RANGES).read_text(encoding="utf-8"))
    flat, episode, frame, base = [], [], [], 0
    for tid, rel in sorted(roots.items(), key=lambda kv: int(kv[0])):
        files = sorted((bench / rel / "meta" / "episodes").rglob("*.parquet"))
        assert files, f"{bench / rel} has no meta/episodes"
        table = pd.concat([pd.read_parquet(f, columns=["episode_index", "dataset_from_index", "length"])
                           for f in files], ignore_index=True)
        offset = 0
        for e, start, n in zip(table["episode_index"], table["dataset_from_index"], table["length"], strict=True):
            e, n = int(e), int(n)
            episode.append(np.full(n, e, dtype=np.int64))
            frame.append(np.arange(n, dtype=np.int64))
            kept = [(max(0, int(s)), min(int(t), n)) for s, t in spans.get(f"task_{int(tid)}--{e}", ())]
            kept = [(lo, hi) for lo, hi in kept if hi > lo]
            assert not kept or int(start) == offset, f"task {tid} episode {e} is not where its rows sit"
            flat += [np.arange(base + offset + lo, base + offset + hi, dtype=np.int64) for lo, hi in kept]
            offset += n
        base += offset
    return np.concatenate(flat), np.concatenate(episode), np.concatenate(frame)


@pytest.mark.slow
def test_the_published_finetune_rows_decode_to_the_frames_the_openpi_line_reads(request):
    import numpy as np

    bench = pathlib.Path(require_option(request.config, "--bench"))
    rows, episode_of, frame_of = _walk(bench)
    assert (rows.size, episode_of.size) == (FLAT_ROWS, DATASET_ROWS)
    assert hashlib.sha256(np.ascontiguousarray(rows, dtype="<i8").tobytes()).hexdigest() == FLAT_ROWS_SHA256
    ep, fr, _ = ft_bundle.episode_table(bench / ROOTS, bench, bench / RANGES)
    assert np.array_equal(ep, episode_of[rows]) and np.array_equal(fr, frame_of[rows])
    assert not (rows == np.arange(FLAT_ROWS)).any()
