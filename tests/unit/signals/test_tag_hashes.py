"""The four locked tag files' config hashes recompute from the recorded generator digest, the inputs'
digests and the metadata dialect of the builder version that wrote each file (conf/ only)."""
from __future__ import annotations

import json

import pytest

from axis_qvla.signals import legacy

LOCK = json.loads(legacy.SIGNALS_LOCK.read_text())
SPEC = json.loads(legacy.GENERATORS.read_text())
ROWS = {r["id"]: r for r in LOCK["files"]}
TAGS = [i for i, e in SPEC["files"].items() if e["generator"].startswith("tags@")]


@pytest.mark.parametrize("lock_id", TAGS)
def test_every_tag_files_config_hash_recomputes(lock_id):
    argv = SPEC["files"][lock_id]["argv"]
    t = legacy.legacy_target(lock_id, "openpi-legacy")
    weights = argv[argv.index("--weights") + 1][len("{lock:"):-1]
    roots = SPEC["inputs"]["roots_v2"]["sha256"] if t.roots_form == "v2" else ROWS["index.roots_d8"]["sha256"]
    granularity = argv[argv.index("--granularity") + 1] if "--granularity" in argv else "episode"
    got = legacy.tags_config_hash(dialect=legacy.TAG_DIALECTS[t.generator], seed=0,
                                  generator_sha256=t.generator_sha256, weights_sha256=ROWS[weights]["sha256"],
                                  roots_sha256=roots, ranges_sha256=ROWS["index.ranges"]["sha256"],
                                  granularity=granularity, plateau="rank")
    assert got == ROWS[lock_id]["generator"]["config_hash"]


def test_the_replay_map_names_the_four_locked_tag_files():
    assert sorted(TAGS) == ["approach_aware.chunk.tags", "approach_aware.episode.tags", "object_centric.chunk.tags",
                            "object_centric.episode.tags"]


def test_each_dialect_keeps_exactly_the_keys_its_builder_wrote():
    live = {"reward_id": "phase", "granularity": "row", "seed": 0, "n_episodes": 3, "plateau": "rank",
            "episodes_multibin_fraction": 0.5}
    assert list(legacy.tags_meta_for(live, "episode")) == ["reward_id", "seed", "n_episodes", "episodes_multibin_fraction"]
    assert list(legacy.tags_meta_for(live, "row-v1")) == ["reward_id", "granularity", "seed", "n_episodes",
                                                          "episodes_multibin_fraction"]
    assert legacy.tags_meta_for(live, "row-v2") == live
