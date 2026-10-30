"""Imported scores record the digest of the bytes they were read from, and cover every frame."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from axis_qvla.signals.imported import ImportedScores
from axis_qvla.signals.rowspace import episode_keys
from axis_qvla.signals.testing import toy_rowspace


def test_json_scores_record_their_bytes_digest(tmp_path):
    p = tmp_path / "w.json"
    p.write_text(json.dumps({"reward_id": "v2", "tau": 10.0, "weights": {"task_800--0": [1.0, 2.0]}}))
    s = ImportedScores.from_json(p)
    assert s.sha256 == hashlib.sha256(p.read_bytes()).hexdigest() and s.name == "w.json"
    assert s.header == {"reward_id": "v2", "tau": 10.0}
    np.testing.assert_array_equal(s.values().values["task_800--0"], [1.0, 2.0])
    with pytest.raises(ValueError, match="sha256"):
        ImportedScores.from_json(p, expect_sha256="0" * 64)
    (tmp_path / "bad.json").write_text("[1, 2]")
    with pytest.raises(ValueError, match="weights"):
        ImportedScores.from_json(tmp_path / "bad.json")


def test_parquet_scores_must_cover_every_frame_of_every_episode(tmp_path):
    rs = toy_rowspace()
    rows = [{"task_id": t, "episode_index": e, "frame_index": f, "score": float(f)}
            for t, e, n in ((800, 0, 6), (800, 1, 4), (800, 2, 5), (1000, 0, 8)) for f in range(n)]
    p = tmp_path / "s.parquet"
    pd.DataFrame(rows).to_parquet(p)
    s = ImportedScores.from_parquet(p, rs, reward_id="gvl")
    assert list(s.payload["weights"]) == episode_keys(rs) and s.header["reward_id"] == "gvl"
    assert s.payload["weights"]["task_1000--0"] == [float(f) for f in range(8)]
    pd.DataFrame(rows[:-1]).to_parquet(p)
    with pytest.raises(ValueError, match="frames"):
        ImportedScores.from_parquet(p, rs, reward_id="gvl")


def test_parquet_scores_name_only_the_row_spaces_episodes(tmp_path):
    rs = toy_rowspace()
    rows = [{"task_id": t, "episode_index": e, "frame_index": f, "score": float(f)}
            for t, e, n in ((800, 0, 6), (800, 1, 4), (800, 2, 5), (1000, 0, 8)) for f in range(n)]
    p = tmp_path / "s.parquet"
    pd.DataFrame(rows + [{"task_id": 900, "episode_index": 0, "frame_index": 0, "score": 1.0}]).to_parquet(p)
    with pytest.raises(ValueError, match="task_900--0 is not an episode of frame index toy"):
        ImportedScores.from_parquet(p, rs, reward_id="gvl")
    pd.DataFrame([r for r in rows if (r["task_id"], r["episode_index"]) != (800, 2)]).to_parquet(p)
    with pytest.raises(ValueError, match="1 episodes have no frames scored \\(first: task_800--2\\)"):
        ImportedScores.from_parquet(p, rs, reward_id="gvl")
