"""The episode-granularity AWR artifact must be attributable (distinct fingerprint), faithful
(hand-checkable P-ratio rank onto the row-weight marginal), and refuse a source it cannot trust."""
from __future__ import annotations

import json

import pytest

from axis_qvla.signals.provenance import FORMULA_VERSION, formula_fingerprint
from axis_qvla.signals.methods.awr_episode import main


def _source(tmp_path, *, with_provenance=True):
    weights = {
        "task_800--0": [2.0, 1.0],          # P = 0.5
        "task_800--1": [1.0, 1.0, 1.0],     # neutral, stays 1.0
        "task_1000--0": [0.5, 0.5],         # P = 0
    }
    params = {"binding": "names", "cap": 2.0, "horizon": 10, "tau": 10.0}
    payload = {
        "version": 1, "reward_id": "phase", "tau": 10.0, "cap": 2.0, "horizon": 10,
        "n_episodes": 3, "n_rows": 7, "n_neutral_rows": 3, "weights": weights,
    }
    if with_provenance:
        payload["provenance"] = {
            "corpus_fingerprint": "sha256:feedbeef",
            "formula_fingerprint": formula_fingerprint("phase", **params),
            "formula_version": FORMULA_VERSION,
            "reward_id": "phase",
            "params": params,
        }
    p = tmp_path / "awr_weights_phase.json"
    p.write_text(json.dumps(payload))
    return p


def _ranges(tmp_path):
    """Every row of every episode trainable -- the denominators then match the raw arrays."""
    rg = tmp_path / "ranges.json"
    rg.write_text(json.dumps({
        "task_800--0": [[0, 2]], "task_800--1": [[0, 3]], "task_1000--0": [[0, 2]],
    }))
    return rg


def test_main_pratio_marginal_matches_the_row_arm_histogram(tmp_path):
    """Hand-solved: informative episodes t1000-0 (P=0) and t800-0 (P=0.5); sorted row marginal
    [0.5, 0.5, 1.0, 2.0]; row-mass midpoints 1 and 3 -> weights 0.5 and 2.0. Neutral episode
    stays 1.0 and is outside both the ordering and the marginal."""
    src = _source(tmp_path)
    out = tmp_path / "awr_weights_phase_ep.json"
    assert main(["--weights", str(src), "--out", str(out),
                 "--ranges", str(_ranges(tmp_path))]) == 0
    got = json.loads(out.read_text())
    assert got["reward_id"] == "phase_ep"
    assert got["episode_score"] == "pratio_marginal"
    assert got["weights"]["task_800--0"] == [2.0, 2.0]
    assert got["weights"]["task_800--1"] == [1.0, 1.0, 1.0]
    assert got["weights"]["task_1000--0"] == [0.5, 0.5]
    assert got["n_rows"] == 7 and got["n_neutral_rows"] == 3
    prov = got["provenance"]
    assert prov["params"]["episode_score"] == "pratio_marginal"
    # self-consistent under pretrain_dataset's recompute-and-compare...
    assert prov["formula_fingerprint"] == formula_fingerprint("phase_ep", **prov["params"])
    # ...and can never collide with the source's
    assert prov["formula_fingerprint"] != json.loads(src.read_text())["provenance"]["formula_fingerprint"]
    assert prov["corpus_fingerprint"] == "sha256:feedbeef"
    assert prov["params"]["granularity"] == "episode"
    assert prov["params"]["source_reward_id"] == "phase"


def test_main_refuses_a_source_without_full_provenance(tmp_path):
    src = _source(tmp_path, with_provenance=False)
    with pytest.raises(ValueError, match="provenance"):
        main(["--weights", str(src), "--out", str(tmp_path / "awr_weights_phase_ep.json"),
              "--ranges", str(_ranges(tmp_path))])


def test_pratio_denominator_is_trainable_rows_only(tmp_path):
    """Catches: idle frames entering the P denominator. Restricting task_800--0's spans to its
    FIRST row (w=2.0 -> P=1.0) must rank it above a full-span episode with P=0.5 would have --
    and its weight comes from the trainable-row marginal [0.5, 0.5, 2.0], midpoints: t1000-0
    (P=0, 2 rows, mid 1 -> 0.5), t800-0 (P=1, 1 row, mid 2.5 -> idx 2 -> 2.0)."""
    src = _source(tmp_path)
    rg = tmp_path / "ranges.json"
    rg.write_text(json.dumps({
        "task_800--0": [[0, 1]], "task_800--1": [[0, 3]], "task_1000--0": [[0, 2]],
    }))
    out = tmp_path / "awr_weights_phase_ep.json"
    assert main(["--weights", str(src), "--out", str(out), "--ranges", str(rg)]) == 0
    got = json.loads(out.read_text())
    assert got["weights"]["task_800--0"] == [2.0, 2.0]
    assert got["weights"]["task_1000--0"] == [0.5, 0.5]


def test_pratio_without_ranges_refuses(tmp_path):
    src = _source(tmp_path)
    with pytest.raises(ValueError, match="ranges"):
        main(["--weights", str(src), "--out", str(tmp_path / "awr_weights_phase_ep.json")])
