"""The roots and ranges builders the row spaces read: which tasks are in, which frames are trained on."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from axis_qvla.data.index import ranges as idx_ranges
from axis_qvla.data.index import roots as idx_roots
from axis_qvla.labels import idle_filter
from axis_qvla.labels.idle_filter import nonidle_ranges


def _task(root, tid, *, info=True, tasks=True, video=True, col_type=int):
    d = root / f"task_{tid}"
    (d / "meta").mkdir(parents=True)
    (d / "data" / "chunk-000").mkdir(parents=True)
    if info:
        (d / "meta" / "info.json").write_text("{}")
    if tasks:
        pd.DataFrame({"task": ["x"]}).to_parquet(d / "meta" / "tasks.parquet")
    if video:
        (d / "videos" / "cam").mkdir(parents=True)
        (d / "videos" / "cam" / "file-000.mp4").write_bytes(b"")
    pd.DataFrame({"a": [col_type(1)]}).to_parquet(d / "data" / "chunk-000" / "file-000.parquet")


def test_select_roots_reports_exclusions_and_unusable_tasks(tmp_path):
    src = tmp_path / "corpus"
    for tid in (800, 1000):
        _task(src, tid)
    _task(src, 809)                        # held out
    _task(src, 900, info=False)
    _task(src, 901, tasks=False)
    _task(src, 902, video=False)
    _task(src, 903, col_type=float)        # minority parquet schema
    rep = idx_roots.select_roots(str(src))
    assert list(rep.roots) == ["1000", "800"]                   # sorted DIRECTORY names
    assert rep.excluded == (809,) and 1889 in rep.missing
    assert rep.unusable == (900, 901, 902, 903)
    payload = idx_roots.roots_payload(rep, src_root=str(src), out=str(tmp_path / "index" / "roots.json"))
    assert payload == {"version": 2, "base": "../corpus", "roots": {"1000": "task_1000", "800": "task_800"}}


def test_main_writes_the_v2_index(tmp_path):
    src = tmp_path / "corpus"
    _task(src, 800)
    out = tmp_path / "roots.json"
    assert idx_roots.main(["--src-root", str(src), "--out", str(out)]) == 0
    assert json.loads(out.read_text()) == {"version": 2, "base": "corpus", "roots": {"800": "task_800"}}


def test_ranges_are_the_forward_idle_filter_of_each_episode(tmp_path):
    corpus = tmp_path / "corpus"
    rng = np.random.default_rng(0)
    moving = np.cumsum(rng.normal(0, 0.05, (40, 8)), axis=0)
    paused = np.concatenate([moving[:12], np.repeat(moving[12:13], 10, axis=0), moving[12:30]])
    signals = {}
    for tid, eps in ((800, [moving, paused]), (1000, [moving[:25]])):
        d = corpus / f"task_{tid}"
        (d / "data" / "chunk-000").mkdir(parents=True)
        (d / "meta").mkdir(parents=True)
        rows = [{"episode_index": e, "observation.state": s.tolist()} for e, ep in enumerate(eps) for s in ep]
        pd.DataFrame(rows).to_parquet(d / "data" / "chunk-000" / "file-000.parquet")
        (d / "meta" / "info.json").write_text(json.dumps({"total_frames": sum(len(x) for x in eps)}))
        signals.update({f"task_{tid}--{e}": ep for e, ep in enumerate(eps)})
    res = idx_ranges.build(corpus)
    assert list(res["ranges"]) == ["task_1000--0", "task_800--0", "task_800--1"]
    for key, sig in signals.items():
        assert res["ranges"][key] == [[s, e] for s, e in nonidle_ranges(sig, forward=True)], key
    assert res["total_rows"] == 40 + 40 + 25


def test_the_recorded_idle_filter_is_the_one_the_ranges_use():
    f = idle_filter.IDLE_FILTER
    assert f["forward"] is True
    assert (f["idle_eps"], f["min_idle_len"], f["min_non_idle_len"], f["trim_last_n"]) == (
        idle_filter.DROID_IDLE_EPS, idle_filter.MIN_IDLE_LEN, idle_filter.MIN_NON_IDLE_LEN, idle_filter.TRIM_LAST_N)
