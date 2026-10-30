"""Recipes hold the paper's hyper-parameters; an arm may change only what its recipe declares."""
from __future__ import annotations

import dataclasses
import json
import re
import shutil

import pytest

from axis_qvla.train.armspec import loads_armspec
from axis_qvla.train.recipes import (
    RECIPES_DIR, RecipeError, check_arm_against_recipe, check_recipe_against_track, list_recipes,
    load_bases, load_recipe, load_rowspaces, load_template, load_track, resolved_fields,
)

H = "b" * 64
# Split so the private box-host token never lands contiguous in this file's own bytes: the test only
# needs the COMPILED regex to match it in a generated template, which string concatenation preserves.
_HOST = "more" "vlas"
MACHINE = re.compile(r"/(disk|localdisk|home|bigdata|rhome|data|mnt)/|" + _HOST)


def _arm(preset: str, stage: str = "finetune", backbone: str = "pi05-openpi", track: str = "paper",
         overrides: str = "") -> str:
    return (f'[arm]\nid = "x.{stage}.y"\nbackbone = "{backbone}"\nstage = "{stage}"\ntrack = "{track}"\n\n'
            f'[data]\nrowspace = "heldout24_ft#sha256:{H}"\n\n[recipe]\npreset = "{preset}"\nseed = 42\n'
            + (f"\n[recipe.overrides]\n{overrides}\n" if overrides else ""))


PAPER_AND_SMALL = {f"{b}.{s}.{t}" for b, s in (("pi05", "pretrain.full"), ("pi05", "finetune.lora"),
                                                ("smolvla", "pretrain"), ("smolvla", "finetune"),
                                                ("xvla", "pretrain"), ("xvla", "finetune")) for t in ("paper", "small")}
REALROBOT = {"pi05.pretrain.lora.realrobot", "pi05.finetune.lora.realrobot", "pi05.finetune.lora.realrobot_only"}
SMOKE = {"pi05.pretrain.full.smoke", "smolvla.pretrain.smoke", "xvla.pretrain.smoke"}


def test_every_recipe_loads_and_matches_its_track():
    ids = list_recipes()
    assert set(ids) == PAPER_AND_SMALL | REALROBOT | SMOKE
    for rid in ids:
        r = load_recipe(rid)
        check_recipe_against_track(r, load_track(r.track))
        assert r.paper_budget == (r.track in ("paper", "realrobot")), rid
        if r.template is not None:
            assert load_template(r)["policy"]["type"] == r.backbone.split("-")[0]


def test_the_paper_pi05_recipes_carry_the_launched_values():
    pt, ft = load_recipe("pi05.pretrain.full.paper"), load_recipe("pi05.finetune.lora.paper")
    assert (pt.steps, pt.global_batch, ft.steps, ft.global_batch) == (20605, 64, 13580, 32)
    assert pt.fields["lr_schedule.warmup_steps"] == 2060 and ft.fields["lr_schedule.warmup_steps"] == 500
    assert pt.fields["ema_decay"] == 0.999 and ft.fields["ema_decay"] is None
    assert ft.fields["lr_schedule.decay_steps"] == 13580 and ft.fields["lr_schedule.decay_lr"] == 2.5e-05
    assert ft.fields["model.paligemma_variant"] == "gemma_2b_lora" and ft.structure["freeze"] == "lora"
    assert pt.structure["freeze"] == "none" and pt.fields["data.default_prompt"] is None


def test_the_realrobot_presets_carry_the_lines_values():
    """The real-robot line's stage 1 and stage 2, as each preset's own description states them:
    steps, batch, warmup, LR, LoRA, vision."""
    want = {"pi05.pretrain.lora.realrobot": (20000, 1000, "lora_frozen_vision"),
            "pi05.finetune.lora.realrobot": (16000, 1600, "lora"),
            "pi05.finetune.lora.realrobot_only": (16000, 1600, "lora")}
    for rid, (steps, warmup, freeze) in want.items():
        r = load_recipe(rid)
        assert (r.track, r.paper_budget, r.steps, r.global_batch) == ("realrobot", True, steps, 64), rid
        assert (r.fields["lr_schedule.warmup_steps"], r.fields["lr_schedule.peak_lr"],
                r.fields["lr_schedule.decay_lr"], r.fields["lr_schedule.decay_steps"]) == (warmup, 2.5e-05, 2.5e-06, steps)
        assert r.fields["model.paligemma_variant"] == "gemma_2b_lora" and r.structure["freeze"] == freeze, rid
        assert r.fields["model.action_horizon"] == 15 and r.fields["data.quality_tag"] is None, rid


def test_the_smoke_presets_are_20_steps_of_8_on_one_gpu_and_otherwise_the_small_ones():
    for rid in SMOKE:
        r = load_recipe(rid)
        assert (r.track, r.paper_budget, r.steps, r.global_batch, r.per_rank_batch, r.world) == ("smoke", False, 20, 8, 8, 1)
        small = load_recipe(rid.replace(".smoke", ".small"))
        assert r.template == small.template
        if r.backbone == "pi05-openpi":
            assert r.structure["fsdp_devices"] == 1 and {**r.structure, "fsdp_devices": 8} == dict(small.structure)
            assert {**r.fields, "batch_size": 64, "num_train_steps": 2000} == dict(small.fields)


def test_the_lerobot_templates_are_neutral():
    for rid in ("smolvla.pretrain.paper", "smolvla.finetune.paper", "xvla.pretrain.paper", "xvla.finetune.paper"):
        t = load_template(load_recipe(rid))
        assert not MACHINE.search(json.dumps(t)), rid
        assert t["seed"] == 0 and t["output_dir"] is None and t["dataset"]["root"] is None
        assert t["policy"]["pretrained_path"] is None and t["sample_weighting"] is None
        assert all(t[k] is None for k in ("axis_rows_path", "axis_schedule_path", "axis_quality_path"))
    x = load_template(load_recipe("xvla.pretrain.paper"))
    assert (x["policy"]["chunk_size"], x["policy"]["domain_id"]) == (30, 20)


def test_resolved_fields_apply_declared_overrides_only():
    r = load_recipe("pi05.finetune.lora.paper")
    spec = loads_armspec(_arm(r.id, overrides='"lr_schedule.warmup_steps" = 100'))
    assert resolved_fields(r, spec)["lr_schedule.warmup_steps"] == 100
    with pytest.raises(RecipeError, match="not a field of recipe"):
        resolved_fields(r, loads_armspec(_arm(r.id, overrides='"lr_schedule.warmup" = 100')))


def test_an_arm_must_match_its_recipe():
    r = load_recipe("pi05.finetune.lora.paper")
    with pytest.raises(RecipeError, match="run stage"):
        check_arm_against_recipe(loads_armspec(_arm(r.id, stage="pretrain")), r)
    with pytest.raises(RecipeError, match="run track"):
        check_arm_against_recipe(loads_armspec(_arm(r.id, track="small")), r)


def test_recipe_refusals(tmp_path):
    shutil.copytree(RECIPES_DIR, tmp_path / "r")
    p = tmp_path / "r" / "pi05.finetune.lora.paper.toml"
    good = p.read_text()
    for edit, match in (
        (good.replace('paper_budget = true', 'paper_budget = true\nsteps = 1'), "unknown key"),
        (good.replace("per_rank_batch = 32", "per_rank_batch = 16"), "per_rank_batch 16 x world 1"),
        (good.replace('freeze = "lora"', 'freeze = "vision"'), "structure.freeze"),
        (good.replace('nulls = ["fields.ema_decay"', 'nulls = ["fields.batch_size"'), "nulls entry"),
    ):
        p.write_text(edit)
        with pytest.raises(RecipeError, match=match):
            load_recipe("pi05.finetune.lora.paper", root=tmp_path / "r")
    with pytest.raises(RecipeError, match="must match"):
        load_recipe("../etc/passwd")


def test_a_small_recipe_cannot_claim_the_paper_budget():
    r = load_recipe("pi05.finetune.lora.small")
    with pytest.raises(RecipeError, match="budgets are not the paper's"):
        check_recipe_against_track(dataclasses.replace(r, paper_budget=True), load_track("small"))


def test_the_old_validated_key_is_refused_naming_paper_budget(tmp_path):
    shutil.copytree(RECIPES_DIR, tmp_path / "r")
    for p in (tmp_path / "r" / "pi05.finetune.lora.paper.toml", tmp_path / "r" / "tracks" / "paper.toml"):
        p.write_text(p.read_text().replace("paper_budget = true", "validated = true"))
    with pytest.raises(RecipeError, match="`validated` is now `paper_budget`"):
        load_recipe("pi05.finetune.lora.paper", root=tmp_path / "r")
    with pytest.raises(RecipeError, match="`validated` is now `paper_budget`"):
        load_track("paper", root=tmp_path / "r")


def test_bases_and_rowspaces():
    bases = load_bases()
    assert bases["pi05_droid"].local == "openpi/openpi-assets/checkpoints/pi05_droid/params"
    assert bases["pi05_droid"].upstream == "gs://openpi-assets/checkpoints/pi05_droid/params"
    assert bases["smolvlm2_500m"].local == "" and bases["xvla_base"].local == "lerobot/xvla-base"
    rs = load_rowspaces()
    assert rs["libero5k_v2_d8"].n_rows == 1906469 and rs["heldout24_ft"].n_rows == 86912
    assert rs["libero5k_v2_d8"].corpus_base == "episodes" and rs["heldout24_ft"].lerobot_local
    assert {r.repo for r in rs.values()} == {"qvla-v1-corpus", "qvla-v1-bench", "local"}
    assert {n: (r.lerobot_copy, r.action_source) for n, r in rs.items()} == {
        "libero5k_v2_d8": ("published", None), "libero5k_v2_d8.sub1000": ("published", None),
        "libero5k_v2_d8.sub2500": ("published", None), "heldout24_ft": ("cut", "action"),
        "smoke_2x3": ("plain", None), "realworld_pretrain": ("plain", None), "realworld_finetune": ("plain", None)}


def test_a_local_frame_index_needs_no_frame_count():
    lay = load_rowspaces()
    assert lay["realworld_pretrain"].repo == "local" and lay["realworld_pretrain"].n_rows is None
    assert lay["libero5k_v2_d8"].n_rows == 1906469


_ROWSPACE = ('repo = "{repo}"\nroots = "r.json"\nranges = "g.json"\ncorpus_base = "."\nlerobot_dataset = "m"\n'
             'n_rows = 10\n')


@pytest.mark.parametrize("body, match", [
    ('lerobot_copy = "built"\n', "lerobot_copy one of"),
    ('lerobot_copy = "cut"\n', "plus action_source"),
    ('lerobot_copy = "cut"\naction_source = ""\n', "action_source must name the column"),
    ('lerobot_copy = "plain"\naction_source = "action"\n', "plus action_source when lerobot_copy"),
    ('lerobot_local = true\n', "needs exactly"),
])
def test_a_rowspace_declares_what_its_lerobot_copy_is(tmp_path, body, match):
    (tmp_path / "rowspaces.toml").write_text(f'schema = "qvla.rowspaces/1"\n[x]\n{_ROWSPACE.format(repo="qvla-v1-bench")}{body}')
    with pytest.raises(RecipeError, match=match):
        load_rowspaces(root=tmp_path)


def test_a_local_rowspace_has_no_published_copy(tmp_path):
    (tmp_path / "rowspaces.toml").write_text(
        f'schema = "qvla.rowspaces/1"\n[x]\n{_ROWSPACE.format(repo="local")}lerobot_copy = "published"\n')
    with pytest.raises(RecipeError, match="local frame index, so its LeRobot copy cannot be published"):
        load_rowspaces(root=tmp_path)
    (tmp_path / "rowspaces.toml").write_text(
        f'schema = "qvla.rowspaces/1"\n[x]\n{_ROWSPACE.format(repo="local")}lerobot_copy = "plain"\n')
    assert load_rowspaces(root=tmp_path)["x"].lerobot_local


def test_only_a_local_frame_index_may_omit_its_frame_count(tmp_path):
    body = _ROWSPACE.replace("n_rows = 10\n", "") + 'lerobot_copy = "plain"\n'
    (tmp_path / "rowspaces.toml").write_text(f'schema = "qvla.rowspaces/1"\n[x]\n{body.format(repo="local")}')
    assert load_rowspaces(root=tmp_path)["x"].n_rows is None
    (tmp_path / "rowspaces.toml").write_text(f'schema = "qvla.rowspaces/1"\n[x]\n{body.format(repo="qvla-v1-bench")}')
    with pytest.raises(RecipeError, match="needs exactly"):
        load_rowspaces(root=tmp_path)
