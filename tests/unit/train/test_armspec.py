"""An ArmSpec names everything a run depends on, by content; anything it does not know is refused."""
from __future__ import annotations

import dataclasses

import pytest

from axis_qvla.train.armspec import (
    ArmSpec, ArmSpecError, armspec_sha256, as_dict, dumps_armspec, load_armspec, loads_armspec,
)

H = "a" * 64
FULL = f'''
[arm]
id = "pi05.finetune.example"
backbone = "pi05-openpi"
stage = "finetune"
track = "paper"
paper_ref = "pi05_axis_heldout_qual_v2_bc"

[init]
ckpt = "axisrobotics/qvla-v1-pi05:pretrain/example/20604#sha256:{H}"

[data]
rowspace = "heldout24_ft#sha256:{H}"
action_space = "droid8"
image = "center_crop"
asset_id = "Devon018/Franka-Datasets-v2"

[signal]
bundle = "sha256:{H}"

[norm]
artifact = "heldout24_qual_v2#sha256:{H}"

[recipe]
preset = "pi05.finetune.lora.paper"
seed = 42
nulls = ["lr_schedule.decay_lr"]

[recipe.overrides]
"lr_schedule.warmup_steps" = 500
"sample_weighting" = {{ type = "axis_flat", kappa = 0.01 }}

[paper]
checkpoint = "pi05/ft/example"
launch_overrides = ["--lr-schedule.warmup-steps=500"]
signals = ["ft:plain"]
pretrain_seed = 42
cells = ["bc|bare"]

[paper.differences."weight_loader.params_path"]
lock = '"base:pi05_droid"'
arm = '"base:pi05_base"'
reason = "example"
'''


def _minimal(**over) -> str:
    base = {"id": "pi05.pretrain.x", "backbone": "pi05-openpi", "stage": "pretrain", "track": "small"}
    base.update(over)
    arm = "\n".join(f'{k} = "{v}"' for k, v in base.items())
    return f'[arm]\n{arm}\n\n[data]\nrowspace = "r#sha256:{H}"\n\n[recipe]\npreset = "p.q"\nseed = 1\n'


def test_the_contract_fields_come_first_in_contract_order():
    names = [f.name for f in dataclasses.fields(ArmSpec)]
    assert names[:11] == ["id", "backbone", "stage", "track", "init", "rowspace", "bundle", "norm",
                          "preset", "seed", "overrides"]


def test_a_full_arm_parses_to_its_fields():
    s = loads_armspec(FULL)
    assert (s.id, s.backbone, s.stage, s.track, s.seed) == (
        "pi05.finetune.example", "pi05-openpi", "finetune", "paper", 42)
    assert s.init.endswith(f"#sha256:{H}") and s.bundle == f"sha256:{H}"
    assert s.rowspace == f"heldout24_ft#sha256:{H}" and s.norm == f"heldout24_qual_v2#sha256:{H}"
    assert dict(s.overrides) == {"lr_schedule.warmup_steps": 500, "lr_schedule.decay_lr": None,
                                 "sample_weighting": {"type": "axis_flat", "kappa": 0.01}}
    assert dict(s.data) == {"action_space": "droid8", "image": "center_crop",
                            "asset_id": "Devon018/Franka-Datasets-v2"}
    assert s.paper["signals"] == ["ft:plain"] and s.paper["pretrain_seed"] == 42
    assert s.paper["differences"]["weight_loader.params_path"]["arm"] == '"base:pi05_base"'


def test_optional_tables_default_to_none():
    s = loads_armspec(_minimal())
    assert (s.init, s.bundle, s.norm, s.paper_ref) == (None, None, None, None)
    assert dict(s.overrides) == {} and dict(s.paper) == {}


def test_dumps_round_trips_and_is_stable():
    s = loads_armspec(FULL)
    text = dumps_armspec(s)
    assert loads_armspec(text) == s
    assert dumps_armspec(loads_armspec(text)) == text


def test_the_digest_follows_every_field():
    s = loads_armspec(FULL)
    assert armspec_sha256(s) == armspec_sha256(loads_armspec(dumps_armspec(s)))
    variants = {
        "id": "pi05.finetune.other",                                   # [arm]
        "backbone": "smolvla-lerobot",                                  # [arm]
        "stage": "pretrain",                                            # [arm]
        "track": "small",                                               # [arm]
        "paper_ref": "other_ref",                                       # [arm]
        "init": s.init.replace(H, "b" * 64),                            # [init]
        "rowspace": s.rowspace.replace(H, "b" * 64),                    # [data]
        "data": {**s.data, "image": "letterbox224"},                    # [data]
        "bundle": s.bundle.replace(H, "b" * 64),                        # [signal]
        "norm": s.norm.replace(H, "b" * 64),                            # [norm]
        "preset": "pi05.finetune.lora.other",                           # [recipe]
        "seed": 43,                                                     # [recipe]
        "overrides": {**s.overrides, "lr_schedule.warmup_steps": 999},  # [recipe.overrides]
        "paper": {**s.paper, "pretrain_seed": 99},                      # [paper]
    }
    for field, value in variants.items():
        other = dataclasses.replace(s, **{field: value})
        assert armspec_sha256(other) != armspec_sha256(s), f"changing {field} did not change the digest"
    assert as_dict(s)["overrides"]["sample_weighting"] == {"type": "axis_flat", "kappa": 0.01}


@pytest.mark.parametrize("text, match", [
    (_minimal() + "\n[extra]\nx = 1\n", "unknown table"),
    (_minimal().replace('track = "small"', 'track = "small"\nsteps = 5'), r"\[arm\] has unknown key"),
    (_minimal().replace("seed = 1", "seed = 1\nsteps = 5"), r"\[recipe\] has unknown key"),
    (_minimal().replace('preset = "p.q"\n', ""), r"\[recipe\] is missing \['preset'\]"),
    (_minimal(backbone="pi0-jax"), "arm.backbone"),
    (_minimal(stage="train"), "arm.stage"),
    (_minimal().replace(f"r#sha256:{H}", "r#sha256:abc"), "data.rowspace"),
    (_minimal() + '\n[signal]\nbundle = "abc"\n', "signal.bundle"),
    (_minimal() + '\n[init]\nckpt = "/abs/path"\n', "init.ckpt"),
    (_minimal().replace("seed = 1", "seed = true"), "recipe.seed"),
    (_minimal().replace("seed = 1", 'seed = 1\n\n[recipe.overrides]\n"bad key" = 1'), "dotted field"),
    (_minimal().replace("seed = 1", 'seed = 1\nnulls = ["a.b"]\n\n[recipe.overrides]\n"a.b" = 1'), "both in recipe.nulls"),
    (_minimal().replace("seed = 1", 'seed = 1\nnulls = ["a.b", "a.b"]'), "duplicated in recipe.nulls"),
    (_minimal().replace("seed = 1", 'seed = 1\nnulls = ["bad key"]'), r"recipe\.nulls key"),
    (_minimal().replace("seed = 1", "seed = 1\noverrides = 5"), "recipe.overrides must be a table"),
    (_minimal().replace("seed = 1", 'seed = 1\noverrides = "ab"'), "recipe.overrides must be a table"),
    (_minimal().replace("seed = 1", 'seed = 1\noverrides = [["a.b", 1]]'), "recipe.overrides must be a table"),
    (_minimal() + '\n[norm]\nartifact = "local:a/../b"\n', "not a valid local: reference"),
    (_minimal() + '\n[norm]\nartifact = "local:a//b"\n', "not a valid local: reference"),
    (_minimal() + '\n[norm]\nartifact = "local:a/./b"\n', "not a valid local: reference"),
    (_minimal(track="papr"), r"arm\.track 'papr' has no file"),
    (_minimal().replace("seed = 1", 'seed = 1\nnulls = ["a.b"]\n\n[recipe.overrides]\n"a" = { x = 1 }'), "overlap"),
    (_minimal().replace("seed = 1", 'seed = 1\n\n[recipe.overrides]\n"a" = { x = 1 }\n"a.y" = 2'), "overlap"),
    # An unquoted dotted key (`a.b = 1`) nests instead of naming the flat field "a.b"; here it
    # collides with a quoted sibling of the same parent, so the ambiguity is refused outright.
    (_minimal().replace("seed = 1", 'seed = 1\n\n[recipe.overrides]\na.b = 1\n"a.c" = 2'), "overlap"),
    (_minimal() + '\n[paper]\ncheckpoint = "x"\n', r"\[paper\] is only allowed on track 'paper'"),
    (FULL.replace('reason = "example"\n', ""), "paper.differences"),
    (FULL.replace('cells = ["bc|bare"]\n', 'cells = ["bc|bare"]\nnotes = ["x"]\n'), r"\[paper\] has unknown key"),
    (FULL.replace("lock = '\"base:pi05_droid\"'", "lock = 'not json'"), "JSON value"),
    ("[arm\n", "not valid TOML"),
])
def test_malformed_arms_are_refused(text, match):
    with pytest.raises(ArmSpecError, match=match):
        loads_armspec(text)


LOCAL = (f'[arm]\nid = "realrobot.finetune.cond"\nbackbone = "pi05-openpi"\nstage = "finetune"\n'
         f'track = "realrobot"\n\n[init]\nckpt = "run:realrobot.pretrain.cond"\n\n'
         f'[data]\nrowspace = "local:realalign_droid8d"\n\n[signal]\nbundle = "local:realalign.cond"\n\n'
         f'[norm]\nartifact = "local:norm/fr3"\n\n[recipe]\npreset = "pi05.finetune.lora.realrobot"\nseed = 0\n')


def test_local_references_are_declarations_off_the_paper_track():
    s = loads_armspec(LOCAL)
    assert (s.rowspace, s.bundle, s.norm, s.init) == (
        "local:realalign_droid8d", "local:realalign.cond", "local:norm/fr3", "run:realrobot.pretrain.cond")
    assert loads_armspec(dumps_armspec(s)) == s


def test_local_on_paper_track_is_refused():
    """Pins which rule fires: the paper-track prohibition, on the first local: field in order
    (data.rowspace, before signal.bundle and norm.artifact are ever reached)."""
    text = LOCAL.replace('track = "realrobot"', 'track = "paper"')
    with pytest.raises(ArmSpecError, match=r"data\.rowspace 'local:realalign_droid8d': a paper run"):
        loads_armspec(text)


def test_local_on_paper_track_wins_over_a_malformed_local_name():
    """Carried fix: on track paper, the paper-track rule is checked before local: syntax, so a
    malformed local: value (here, a `..` segment) still reports the useful reason, not a verdict
    on its well-formedness."""
    text = LOCAL.replace('track = "realrobot"', 'track = "paper"').replace(
        '"local:realalign_droid8d"', '"local:a/../b"')
    with pytest.raises(ArmSpecError, match=r"data\.rowspace 'local:a/\.\./b': a paper run"):
        loads_armspec(text)


def test_local_on_init_on_paper_track_is_refused():
    """init.ckpt is checked first of all four local: fields; a local: init on track paper must be
    refused by the SAME rule as rowspace/bundle/norm, not silently accepted."""
    text = LOCAL.replace('track = "realrobot"', 'track = "paper"').replace(
        'ckpt = "run:realrobot.pretrain.cond"', 'ckpt = "local:realrobot.pretrain.cond"')
    with pytest.raises(ArmSpecError, match=r"init\.ckpt 'local:realrobot\.pretrain\.cond': a paper run"):
        loads_armspec(text)


def test_a_local_name_with_an_empty_segment_is_refused():
    """Pins the malformed-local rule (not the paper-track rule): track stays `realrobot`."""
    text = LOCAL.replace('"local:realalign.cond"', '"local:"')
    with pytest.raises(ArmSpecError, match=r"signal\.bundle 'local:' is not a valid local: reference"):
        loads_armspec(text)


def test_a_local_name_with_a_leading_slash_is_refused():
    text = LOCAL.replace('"local:norm/fr3"', '"local:/abs"')
    with pytest.raises(ArmSpecError, match=r"norm\.artifact 'local:/abs' is not a valid local: reference"):
        loads_armspec(text)


def test_load_names_the_file(tmp_path):
    p = tmp_path / "arm.toml"
    p.write_text(_minimal(backbone="nope"))
    with pytest.raises(ArmSpecError, match=str(p)):
        load_armspec(p)
    with pytest.raises(ArmSpecError, match="cannot read"):
        load_armspec(tmp_path / "absent.toml")


def test_load_refuses_a_file_that_is_not_utf8(tmp_path):
    p = tmp_path / "bad_encoding.toml"
    p.write_bytes(b"[arm]\nid = \"x\xff\xfe\"\n")
    with pytest.raises(ArmSpecError, match=str(p)):
        load_armspec(p)
    with pytest.raises(ArmSpecError, match="cannot read"):
        load_armspec(p)


def test_load_names_the_file_for_a_non_table_overrides(tmp_path):
    """The loader's own promise ('every refusal names the file') held even when the raw TOML
    value crashes a naive `dict(...)` call with a raw TypeError instead of an ArmSpecError."""
    p = tmp_path / "arm.toml"
    p.write_text(_minimal().replace("seed = 1", "seed = 1\noverrides = 5"))
    with pytest.raises(ArmSpecError, match=str(p)):
        load_armspec(p)
