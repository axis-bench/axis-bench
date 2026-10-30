"""The paper-bundle declaration covers every arm and every locked file, and builds a program for each
entry -- checked against SIGNALS.lock alone, no data."""
from __future__ import annotations

import collections
import json

import pytest

from axis_qvla.signals import paper
from axis_qvla.signals.imported import ImportedScores
from axis_qvla.signals.legacy import SIGNALS_LOCK, legacy_target
from axis_qvla.signals.testing import toy_rowspace

ENTRIES = paper.load_conf()
LOCK = json.loads(SIGNALS_LOCK.read_text())
CONSUMED = collections.defaultdict(set)
for _row in LOCK["files"]:
    for _t in _row["trained"]:
        CONSUMED[_t["arm"]].add(_row["id"])


def test_every_trained_arm_is_mapped_exactly_once():
    arms = [a for e in ENTRIES.values() for a in e["arms"]]
    assert len(arms) == len(set(arms)) == 153
    assert set(arms) == set(CONSUMED)


def test_every_locked_file_is_exported_and_every_arm_gets_what_it_consumed():
    exported = {i for e in ENTRIES.values() for i in e["exports"]}
    assert exported == {r["id"] for r in LOCK["files"]}
    for e in ENTRIES.values():
        for arm in e["arms"]:
            assert CONSUMED[arm] <= set(e["exports"]), (arm, e["name"])


@pytest.mark.parametrize("name", sorted(ENTRIES))
def test_every_entry_resolves_its_targets_and_builds_a_program(name):
    e = ENTRIES[name]
    for lock_id in e["exports"]:
        legacy_target(lock_id, e["format"])                       # raises on a format the file is not read in
    stub = ImportedScores({"reward_id": "v2", "weights": {}}, "0" * 64, "stub")
    rowspaces = {n: toy_rowspace() for n in {x["rowspace"] for x in ENTRIES.values()} | {"libero5k_v2_d8"}}
    program, _labels = paper.program_for(e, ENTRIES, rowspaces=rowspaces, scores=lambda _id: stub, corpus="corpus")
    assert program.spec()["method"] in {"bc", "awr_chunk", "awr_episode", "awr_imported", "cfg_tags", "drop_top50", "random50"}
    if e.get("control"):
        assert ENTRIES[e["control"]]["method"] == "random50"


def test_budgets_are_the_paper_budgets():
    budgets = collections.Counter((tuple(e["budget"]), e["rowspace"], e["format"]) for e in ENTRIES.values())
    allowed = {((20605, 64), "libero5k_v2_d8", "openpi-legacy"), ((20621, 64), "libero5k_v2_d8", "lerobot-legacy"),
               ((4168, 64), "libero5k_v2_d8.sub1000", "openpi-legacy"), ((10472, 64), "libero5k_v2_d8.sub2500", "openpi-legacy"),
               ((4168, 64), "libero5k_v2_d8.sub1000", "lerobot-legacy"), ((10472, 64), "libero5k_v2_d8.sub2500", "lerobot-legacy"),
               ((6790, 64), "heldout24_ft", "lerobot-legacy")}
    assert set(budgets) <= allowed
