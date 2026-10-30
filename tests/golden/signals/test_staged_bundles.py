"""The staged bundles in the corpus pack (--pretraining-data) are the declared ones: the manifest records
this declaration and lock, maps every arm to the bundle conf/signals/paper_bundles.toml names for
it, and records each bundle as declared; each bundle carries its entry's seed, budget, method and
row space; every bundle and row space verifies, and all 30 locked files re-export byte for byte."""
from __future__ import annotations

import hashlib
import json

import pytest

from axis_qvla.signals import paper
from axis_qvla.signals.bundle import load_bundle
from axis_qvla.signals.legacy import SIGNALS_LOCK


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.slow
def test_the_staged_bundles_are_the_declared_ones_and_reexport_every_locked_file(signals_pack, tmp_path):
    man = json.loads((signals_pack / "bundles" / "manifest.json").read_text())
    entries = paper.load_conf()
    assert (len(man["bundles"]), len(man["arms"]), len(man["rowspaces"])) == (53, 153, 4)
    assert (man["conf_sha256"], man["signals_lock_sha256"]) == (_sha(paper.CONF), _sha(SIGNALS_LOCK))
    assert man["arms"] == {a: n for n, e in entries.items() for a in e["arms"]}
    assert {n: (b["method"], b["format"], b["rowspace"], sorted(b["exports"]), b["paper_arm"])
            for n, b in man["bundles"].items()} == \
           {n: (e["method"], e["format"], e["rowspace"], sorted(e["exports"]), e.get("paper_arm", True))
            for n, e in entries.items()}
    for name, b in man["bundles"].items():
        e, bundle = entries[name], load_bundle(signals_pack / "bundles" / b["id"])
        assert (bundle.seed, [bundle.budget.steps, bundle.budget.global_batch], bundle.program["method"]) == \
               (e["seed"], e["budget"], paper.PROGRAM_METHODS.get(e["method"], e["method"])), name
        assert bundle.rowspace.name == e["rowspace"], name
    assert paper.check(signals_pack / "bundles", scratch=tmp_path) == []


@pytest.mark.slow
def test_paper_bundles_declare_their_consumers_contract(signals_pack):
    man = json.loads((signals_pack / "bundles" / "manifest.json").read_text())
    for name, b in man["bundles"].items():
        bundle = load_bundle(signals_pack / "bundles" / b["id"])
        if b["method"] in ("awr_chunk", "awr_episode"):
            assert bundle.reducer == "global_ratio", name
        if b["method"] == "awr_imported":
            assert bundle.reducer == "per_rank_ratio", name
        if b["method"] == "cfg":
            assert bundle.requires == frozenset({"plan", "cond_prompt"}), name
            assert [s.fields for s in bundle.serve] == [{}, {"quality": "5"}], name
