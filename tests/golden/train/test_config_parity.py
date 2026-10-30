"""Config hash-parity: every paper ArmSpec, with its recipe, reproduces the run it names.

conf/release/paper_configs.lock.json was built once from the training commits; the repo-only
tests below need nothing else (no data, no GPU, no backbone venv). An arm may depart from its run
only where it declares the difference under [paper.differences], with both values and a reason.

A bundle's own content (its exact id, and the row space it was compiled over) has no ground truth
in the lock at all -- only the staged bundle manifest carries it -- so the bundle- and
rowspace-identity checks run separately, over `--pretraining-data`, in
test_the_arm_matches_the_staged_bundle_manifest: marked slow, and reads
tests.conftest.require_option, which fails rather than skips when the option is missing.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from axis_qvla.train.armspec import load_armspec
from axis_qvla.train.lerobot_cfg import LINE_FIELDS
from axis_qvla.train.parity import check_arm, load_lock, load_signals_lock, lock_sha256, project_lerobot, project_openpi
from axis_qvla.train.paper import RUNS_DIR, arms_for_paper_config, find_arm_by_checkpoint, paper_arm_files
from axis_qvla.train.recipes import load_recipe
from tests.conftest import require_option

LOCK = load_lock()
SIGNALS = load_signals_lock()
ARMS = paper_arm_files()
INDEX = json.loads((RUNS_DIR / "index.json").read_text())


def test_one_arm_per_paper_run():
    ids = {load_armspec(p).id for p in ARMS}
    assert len(ARMS) == 185 and ids == set(LOCK["pi05"]) | set(LOCK["lerobot"])
    assert set(INDEX["arms"]) == ids


@pytest.mark.parametrize("path", ARMS, ids=lambda p: p.stem)
def test_the_arm_reproduces_its_run(path):
    spec = load_armspec(path)
    assert check_arm(spec, LOCK, SIGNALS) == []
    assert path.relative_to(RUNS_DIR).as_posix() == INDEX["arms"][spec.id]["file"]


@pytest.fixture(scope="module")
def manifest(request) -> dict:
    pack = require_option(request.config, "--pretraining-data")
    return json.loads((pathlib.Path(pack) / "bundles" / "manifest.json").read_text())


@pytest.mark.slow
@pytest.mark.parametrize("path", ARMS, ids=lambda p: p.stem)
def test_the_arm_matches_the_staged_bundle_manifest(path, manifest):
    spec = load_armspec(path)
    assert check_arm(spec, LOCK, SIGNALS, manifest=manifest) == []


@pytest.mark.parametrize("arm_id", sorted(LOCK["lerobot"]))
def test_the_hash_of_a_lerobot_projection_is_the_lock_hash(arm_id):
    """A field the LeRobot line added after the paper's runs is emitted only when an arm declares it,
    so every paper arm's config stays the one its run recorded."""
    spec = load_armspec(RUNS_DIR / INDEX["arms"][arm_id]["file"])
    projected = project_lerobot(spec, load_recipe(spec.preset), SIGNALS)
    assert not set(LINE_FIELDS) & set(projected)
    assert lock_sha256(projected) == LOCK["lerobot"][arm_id]["sha256"]


def test_the_hash_of_a_pi05_projection_is_the_lock_hash():
    spec = load_armspec(find_arm_by_checkpoint("pi05/ft/pi05_axis_heldout_qual_v2_bc__qual_v2_bc"))
    assert lock_sha256(project_openpi(spec, load_recipe(spec.preset), SIGNALS)) == LOCK["pi05"][spec.id]["sha256"]


def test_a_parity_failure_is_reported_field_by_field():
    import dataclasses
    spec = load_armspec(find_arm_by_checkpoint("pi05/ft/pi05_axis_heldout_qual_v2_bc__qual_v2_bc"))
    bad = dataclasses.replace(spec, seed=7)
    assert check_arm(bad, LOCK, SIGNALS) == ["seed: arm 7.0 != lock 42.0"]


def test_the_table_one_replicate_pairing():
    """Seed-43/44 replicates pair on the pretraining seed; the finetune seed is recorded beside it.
    The BC replicate is the one whose seed-43 member was also finetuned with seed 43."""
    groups = {g["family"]: g["members"] for g in INDEX["replicates"] if g["backbone"] == "pi05-openpi"}
    bc = groups["pi05_axis_pretrain_d8_paper_5k__libero5k_d8_bc ft:plain"]
    assert [(m["pretrain_seed"], m["ft_seed"]) for m in bc] == [(42, 42), (43, 43), (44, 42)]
    assert len(groups) == 6 and all(len(v) == 3 for v in groups.values())


def test_serving_finds_the_arm_of_a_cell():
    paths = arms_for_paper_config("pi05_axis_heldout_qual_v2_bc")
    assert sorted(load_armspec(p).paper["checkpoint"] for p in paths) == [
        "pi05/ft/pi05_axis_heldout_qual_v2_bc__qual_v2_bc", "pi05/ft/pi05_axis_heldout_qual_v2_bc__seed44_bc"]
