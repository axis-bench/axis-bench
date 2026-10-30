"""build_scaling_subsets: nested seeded episode subsets, ranges kept verbatim, budgets 1-epoch-matched."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from axis_qvla.signals.methods import schedule as bis
from axis_qvla.signals.methods import scaling_subsets as bss


def test_subsets_are_nested_prefixes_of_one_permutation():
    keys = [f"task_1--{i}" for i in range(20)]
    s = bss.scaling_subsets(keys, [5, 12], seed=0)
    assert s[12][:5] == s[5] and len(set(s[12])) == 12
    assert s == bss.scaling_subsets(keys, [5, 12], seed=0)


def test_run_scaling_writes_filtered_ranges_and_a_manifest(tmp_path, monkeypatch):
    monkeypatch.delenv(bis.CORPUS_ROOT_ENV, raising=False)
    lens = [10, 8, 6, 12, 9, 7]
    d = tmp_path / "task_900" / "meta" / "episodes" / "chunk-000"
    d.mkdir(parents=True)
    pd.DataFrame({"episode_index": range(6), "dataset_from_index": np.r_[0, np.cumsum(lens)[:-1]],
                  "length": lens}).to_parquet(d / "file-000.parquet")
    roots = tmp_path / "roots.json"
    roots.write_text(json.dumps({"version": 2, "base": ".", "roots": {"900": "task_900"}}))
    ranges = {f"task_900--{e}": [[1, n - 1]] for e, n in enumerate(lens)}
    (tmp_path / "ranges.json").write_text(json.dumps(ranges))
    w = tmp_path / "w.json"
    w.write_text(json.dumps({"reward_id": "phase", "tau": 10.0, "cap": 2.0, "horizon": 10, "n_rows": sum(lens),
                             "weights": {f"task_900--{e}": [1.5] * n for e, n in enumerate(lens)}}))
    out = tmp_path / "out"
    assert bss.main(["--roots", str(roots), "--ranges", str(tmp_path / "ranges.json"), "--weights", str(w),
                     "--out-dir", str(out), "--sizes", "2", "4", "--batch", "4"]) == 0
    man = json.loads((out / "scaling_manifest.json").read_text())
    for size in (2, 4):
        sub = json.loads((out / f"scaling_{size}.ranges.json").read_text())
        assert list(sub) == [k for k in ranges if k in sub]          # source key order kept
        assert all(sub[k] == ranges[k] for k in sub)                  # spans verbatim
        rows = sum(n - 2 for k, n in zip(ranges, lens) if k in sub)
        assert man["subsets"][str(size)]["trainable_rows"] == rows
        assert man["subsets"][str(size)]["steps_1epoch"] == rows // 4
    assert man["full_corpus"]["trainable_rows"] == sum(n - 2 for n in lens)
    assert (out / "scaling_2.ranges.json").read_text().startswith("{\n \"")   # indent=1, as released
