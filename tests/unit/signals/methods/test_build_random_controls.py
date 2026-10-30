"""build_random_controls: the random-50% arms keep the drop-top row count and differ only in WHICH rows."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from axis_qvla.signals.methods import schedule as bis
from axis_qvla.signals.methods import random50 as brc


@pytest.fixture
def toy(tmp_path, monkeypatch):
    monkeypatch.delenv(bis.CORPUS_ROOT_ENV, raising=False)
    lens = [12, 9, 15, 7, 11]
    d = tmp_path / "task_900" / "meta" / "episodes" / "chunk-000"
    d.mkdir(parents=True)
    pd.DataFrame({"episode_index": range(5), "dataset_from_index": np.r_[0, np.cumsum(lens)[:-1]],
                  "length": lens}).to_parquet(d / "file-000.parquet")
    roots = tmp_path / "roots.json"
    roots.write_text(json.dumps({"version": 2, "base": ".", "roots": {"900": "task_900"}}))
    ranges = tmp_path / "ranges.json"
    ranges.write_text(json.dumps({f"task_900--{e}": [[0, n]] for e, n in enumerate(lens)}))
    rng = np.random.default_rng(3)
    tmpl = tmp_path / "awr_weights.json"
    tmpl.write_text(json.dumps({"version": 1, "reward_id": "v2", "tau": 10.0, "cap": 2.0, "horizon": 10,
                                "n_episodes": 5, "n_rows": sum(lens), "n_neutral_rows": 0,
                                "weights": {f"task_900--{e}": (0.5 + rng.random(n)).tolist()
                                            for e, n in enumerate(lens)}}))
    return roots, ranges, tmpl


def test_synthetic_weights_carry_no_reward_and_name_their_template(toy, tmp_path):
    _, _, tmpl = toy
    ep = json.loads(brc.synthetic_weights(tmpl, tmp_path / "e.json", "rand50_ep", per_episode_random=True,
                                          template_name="awr_weights_v2.json").read_text())
    row = json.loads(brc.synthetic_weights(tmpl, tmp_path / "r.json", "rand50_row",
                                           per_episode_random=False).read_text())
    assert ep["provenance"]["keys_from"] == "awr_weights_v2.json" and row["provenance"]["keys_from"] == "awr_weights.json"
    assert ep["provenance"]["carries_reward"] is False
    for k, v in ep["weights"].items():
        assert len(set(v)) == 1 and 1.0 < v[0] < 2.0
    assert {x for v in row["weights"].values() for x in v} == {1.5}


def test_controls_keep_exactly_the_drop_top_row_count(toy, tmp_path):
    roots, ranges, tmpl = toy
    out = tmp_path / "out"
    assert brc.main(["--template", str(tmpl), "--roots", str(roots), "--ranges", str(ranges),
                     "--out-dir", str(out), "--total-steps", "6", "--batch", "9"]) == 0
    assert bis.main(["--mode", "drop_top", "--weights", str(tmpl), "--roots", str(roots), "--ranges", str(ranges),
                     "--out", str(out / "top.npz"), "--total-steps", "6", "--batch", "9", "--keep-top-frac", "0.5"]) == 0
    metas = {p.name: json.loads(str(np.load(p)["meta"])) for p in out.glob("*.npz")}
    kept = {n: m["unique_frames"] for n, m in metas.items()}
    assert len(set(kept.values())) == 1 and kept["top.npz"] == round(0.5 * 54)
    ep = metas["lerobot_schedule_drop_top_rand50_ep_smolvla.npz"]
    assert ep["reward_id"] == "rand50_ep" and ep["seed"] == 0
    assert metas["lerobot_schedule_drop_random_rand50_row_smolvla.npz"]["seed"] == 42


def test_controls_are_reproducible_byte_for_byte(toy, tmp_path):
    roots, ranges, tmpl = toy
    for i in (1, 2):
        brc.main(["--template", str(tmpl), "--roots", str(roots), "--ranges", str(ranges),
                  "--out-dir", str(tmp_path / f"o{i}"), "--total-steps", "4", "--batch", "5"])
    for p in (tmp_path / "o1").iterdir():
        assert p.read_bytes() == (tmp_path / "o2" / p.name).read_bytes()
