"""The two run specs a method's author never writes: its pretraining on the paper's budget, and the
paper's fine-tune of it on the 24 held-out tasks, bare or (--cond) tagged and evaluated both ways.
Writing them reads no data; what the labels decide is completed when the run is bound."""
from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np
import pytest

import axis_qvla.train.finetune as ftcli
import axis_qvla.train.pretrain as ptcli
from axis_qvla.core.canon import dumps_canonical, sha256_bytes
from axis_qvla.cli import main
from axis_qvla.data.layout import CORPUS, default_dest
from axis_qvla.signals.bundle import rowspace_id, write_bundle
from axis_qvla.signals.compile import compile_program
from axis_qvla.signals.imported import ImportedScores
from axis_qvla.signals.legacy import legacy_target
from axis_qvla.signals.methods.bc import BC
from axis_qvla.signals.methods import schedule as bis
from axis_qvla.signals.schema import Budget
from axis_qvla.train import ids, names
from axis_qvla.train.armspec import dumps_armspec, load_armspec, loads_armspec
from axis_qvla.train.method_runs import (FAMILIES, PAPER_SEED, MethodRunError, complete_from_labels, finetune_run_id,
                                    finetune_spec, lerobot_overrides, paper_finetune_seed, pretrain_spec)
from axis_qvla.train.paper import RUNS_DIR
from axis_qvla.train.recipes import load_recipe
from experiments.examples.keep_top import keep_top
from tests.unit.signals.toy_corpus import write_toy_corpus
from tests.unit.train._arms import arm_toml, host
from tests.unit.train._bundles import finetune_corpus, make_bundle

EXAMPLES = pathlib.Path(__file__).resolve().parents[3] / "experiments" / "examples" / "runs"


@pytest.fixture
def toy(tmp_path, monkeypatch):
    monkeypatch.delenv(bis.CORPUS_ROOT_ENV, raising=False)
    return dataclasses.replace(write_toy_corpus(tmp_path / "toy").rs, name="libero5k_v2_d8")


def _keep_top(toy, tmp_path):
    scores = ImportedScores.from_json(tmp_path / "toy" / "w.json")
    return compile_program(keep_top(0.5), scores, toy, Budget(20, 8), 42)[0]


@pytest.mark.parametrize("family, backbone", [("smolvla", "smolvla-lerobot"), ("xvla", "xvla-lerobot")])
def test_the_lerobot_overrides_are_derived_from_the_labels_files(toy, tmp_path, family, backbone):
    over = lerobot_overrides(backbone, _keep_top(toy, tmp_path), load_recipe(f"{family}.pretrain.paper"))
    example = load_armspec(EXAMPLES / f"{family}.pretrain.keep_top.toml").overrides
    assert over.pop("steps") == 20
    assert over == {k: v for k, v in example.items() if k != "steps"}     # what the shipped example types


def test_a_pretrain_spec_is_written_from_names_alone():
    spec = pretrain_spec("smolvla", "example.keep_top")
    assert (spec.id, spec.bundle, spec.preset, spec.rowspace) == (
        "smolvla.pretrain.example.keep_top", "name:example.keep_top", "smolvla.pretrain.paper", "libero5k_v2_d8")
    assert dict(spec.overrides) == {}                                    # completed from the labels at bind
    assert pretrain_spec("pi05", "m", run="r").id == "pi05.pretrain.r"
    with pytest.raises(MethodRunError, match="pi05, smolvla, xvla"):
        pretrain_spec("pi0", "m")


def test_binding_completes_a_lerobot_spec_from_its_labels_and_keeps_its_own_overrides(toy, tmp_path):
    _, paths = host(tmp_path)
    root = ids.local_bundles_root(paths)
    root.mkdir(parents=True)
    bid = write_bundle(_keep_top(toy, tmp_path), root)
    spec = dataclasses.replace(pretrain_spec("smolvla", "example.keep_top"), bundle=f"sha256:{bid}",
                               overrides={"batch_size": 32})
    done = complete_from_labels(spec, paths, smoke=True)
    assert done.overrides["axis_expected_mode"] == "drop_top" and done.overrides["batch_size"] == 32


def _labels(paths, steps, batch):
    b = make_bundle(plan=np.zeros((steps, batch), dtype=np.int64))
    prog = {**b.program, "method": "example.keep_top"}
    root = ids.local_bundles_root(paths)
    root.mkdir(parents=True, exist_ok=True)
    return write_bundle(dataclasses.replace(b, program=prog, program_sha=sha256_bytes(dumps_canonical(prog))), root)


def test_labels_off_the_paper_budget_are_refused_with_the_compile_flags(tmp_path):
    _, paths = host(tmp_path)
    spec = dataclasses.replace(pretrain_spec("pi05", "example.keep_top"), bundle=f"sha256:{_labels(paths, 6, 2)}")
    with pytest.raises(MethodRunError, match="--steps 20605 --batch 64"):
        complete_from_labels(spec, paths, smoke=False)
    assert complete_from_labels(spec, paths, smoke=True) == spec           # a smoke trains a prefix


@pytest.mark.parametrize("backbone, family, cond, overrides", [
    ("pi05-openpi", "pi05", False, {}),
    ("pi05-openpi", "pi05", True, {"data.quality_tag": 5}),
    ("smolvla-lerobot", "smolvla", False, {}),
    ("smolvla-lerobot", "smolvla", True, {"axis_quality_constant_tag": 5, "axis_quality_token_budget": 48}),
    ("xvla-lerobot", "xvla", False, {}),
    ("xvla-lerobot", "xvla", True, {"axis_quality_constant_tag": 5, "axis_quality_token_budget": 50}),
])
def test_the_finetune_is_the_papers_recipe_from_the_run(backbone, family, cond, overrides):
    ft = finetune_spec(pretrain_spec(family, "mine"), cond=cond)
    assert ft.id == f"{family}.finetune.mine" + (".cond" if cond else "") and ft.init == f"run:{family}.pretrain.mine"
    assert ft.stage == "finetune" and dict(ft.overrides) == overrides and not ft.paper and ft.paper_ref is None
    assert ft.rowspace == "heldout24_ft" and "sha256:" not in dumps_armspec(ft)     # every input by name
    assert "asset_id" not in ft.data                     # the recipe's data.repo_id names the same dataset
    assert ft.bundle == (None if family == "pi05" else "name:lerobot.ft_rows_all")


def test_a_generated_lerobot_finetune_names_the_released_row_order_which_resolves_to_the_papers(tmp_path):
    _, paths = host(tmp_path)
    ft = finetune_spec(pretrain_spec("smolvla", "mine"), cond=False)
    resolved, named = names.resolve_names(ft, paths)
    paper = load_armspec(RUNS_DIR / "smolvla" / "finetune" / "mvft_smolvla_bc.toml")
    assert resolved.bundle == paper.bundle and named["bundle"] == {"name": ft.bundle, "id": paper.bundle}
    assert names.is_published_labels(ft.bundle) and not names.is_published_labels("name:example.keep_top")


def test_a_generated_lerobot_finetune_renders_on_the_released_row_order_as_it_is(tmp_path, monkeypatch, capsys):
    """End to end: `qvla train finetune --from` writes the spec, `qvla train <spec> --render` binds it.
    The released row order is a fine-tune's plan (6,790 x 64), so completing it as a method's labels
    (complete_from_labels: the pretraining budget, derived overrides) would refuse it; train_spec skips
    that for released labels (names.is_published_labels). The release's row order and frame index stand
    in as a fixture's, compiled as conf/signals/paper_bundles.toml declares lerobot.ft.rows, under the
    released names."""
    hosts, paths = host(tmp_path)
    pre = loads_armspec(arm_toml(id="smolvla.pretrain.mine", backbone="smolvla-lerobot", stage="pretrain",
                                 preset="smolvla.pretrain.paper", rowspace="libero5k_v2_d8",
                                 init="base:smolvlm2_500m", norm=False, seed=0))
    run = ids.run_root(paths, pre.id)
    last = run / "train" / "checkpoints" / "last"
    (last / "training_state").mkdir(parents=True)
    (last / "pretrained_model").mkdir()
    (run / "run_spec.toml").write_text(dumps_armspec(pre))
    (last / "training_state" / "training_step.json").write_text('{"step": 20605}')
    (last / "pretrained_model" / "train_config.json").write_text('{"steps": 20605}')
    (last / "pretrained_model" / "model.safetensors").write_bytes(b"")
    assert main(["train", "finetune", "--from", pre.id, "--host", str(hosts), "--print"]) == 0
    ft = paths.runs / "smolvla.finetune.mine"
    spec = load_armspec(ft / "spec.toml")
    assert (spec.bundle, spec.rowspace) == ("name:lerobot.ft_rows_all", "heldout24_ft")

    rs = finetune_corpus(paths, spans={"task_900--0": [[0, 40]], "task_901--0": [[0, 24]]}, n_rows=64)
    rows = BC("lerobot-axisrowsampler", exports=(legacy_target("lerobot.ft_rows_all", "lerobot-legacy"),))
    bundle, _ = compile_program(rows, None, rs, Budget(6790, 64), 0)
    bid = write_bundle(bundle, default_dest(CORPUS, paths, "bundles"))
    monkeypatch.setitem(names.PUBLISHED_LABELS, "lerobot.ft_rows_all", bid)
    monkeypatch.setitem(names.PUBLISHED_ROWSPACES, "heldout24_ft", rowspace_id(rs))
    completed = []
    monkeypatch.setattr("axis_qvla.train.method_runs.complete_from_labels", lambda *a, **k: completed.append(a) or a[0])
    capsys.readouterr()
    assert main(["train", str(ft / "spec.toml"), "--host", str(hosts), "--render"]) == 0, capsys.readouterr().out
    assert completed == []
    assert json.loads((ft / "names.json").read_text())["bundle"] == {"name": "name:lerobot.ft_rows_all",
                                                                     "id": f"sha256:{bid}"}
    cfg = json.loads((ft / "train_config.json").read_text())
    # The recipe's steps as they are; the seed the paper's SmolVLA convention gives pretrain seed 0 (0).
    assert (cfg["steps"], cfg["seed"], cfg["axis_expected_mode"]) == (6790, 0, None)
    assert cfg["axis_rows_path"] == str(ft / "signals" / "v1" / "lerobot" / "ft_rows_all.npz")


def test_lerobot_labels_off_the_paper_pretraining_budget_are_refused_unless_smoke(tmp_path):
    """SmolVLA and X-VLA replay a plan of the paper's 20,605 x 64, as pi0.5 does (not their preset's 20,621)."""
    _, paths = host(tmp_path)
    spec = dataclasses.replace(pretrain_spec("smolvla", "example.keep_top"), bundle=f"sha256:{_labels(paths, 6, 64)}")
    with pytest.raises(MethodRunError, match=r"compiled for 6 steps x 64; the paper trains it for 20605 x 64: "
                                             r"compile them again with --steps 20605 --batch 64"):
        complete_from_labels(spec, paths, smoke=False)
    with pytest.raises(MethodRunError, match="lerobot-legacy"):     # a smoke passes the budget, then exports
        complete_from_labels(spec, paths, smoke=True)


def test_finetune_ids_follow_the_pretrains():
    assert finetune_run_id("pi05.pretrain.example.keep_top", cond=False) == "pi05.finetune.example.keep_top"
    assert finetune_run_id("xvla.pretrain.a", cond=True, run="b") == "xvla.finetune.b.cond"
    with pytest.raises(MethodRunError, match="not a pretraining run id"):
        finetune_run_id("pi05.finetune.a", cond=False)


def test_print_plans_pretrain_then_finetune_with_no_data(tmp_path, capsys):
    """The docs' CI stand-in: nothing compiled, nothing trained, no environment built."""
    hosts, paths = host(tmp_path)
    assert main(["train", "pretrain", "--labels", "example.keep_top", "--backbone", "pi05",
                 "--host", str(hosts), "--print"]) == 0
    pre = paths.runs / "pi05.pretrain.example.keep_top" / "spec.toml"
    assert capsys.readouterr().out.splitlines() == [
        f"OK    wrote {pre}", f"OK    pi05.pretrain.example.keep_top: {pre}", f"qvla train {pre} --host {hosts}",
        f"NEXT  qvla train finetune --from pi05.pretrain.example.keep_top --host {hosts}"]
    assert main(["train", "finetune", "--from", "pi05.pretrain.example.keep_top", "--cond",
                 "--host", str(hosts), "--print"]) == 0
    ft = paths.runs / "pi05.finetune.example.keep_top.cond"
    out = capsys.readouterr().out.splitlines()
    assert out[-2:] == [f"qvla train {ft / 'spec.toml'} --host {hosts}",
                        f"NEXT  qvla bench eval --from-run pi05.finetune.example.keep_top.cond --host {hosts}"]
    assert load_armspec(ft / "spec.toml").init == "run:pi05.pretrain.example.keep_top"
    assert json.loads((ft / "eval.json").read_text())["serving"] == ["bare", "cond"]


def _finished_pretrain(paths):
    spec = loads_armspec(arm_toml(id="pi05.pretrain.mine", stage="pretrain", preset="pi05.pretrain.full.paper",
                                  rowspace="libero5k_v2_d8", init="base:pi05_droid"))
    run = ids.run_root(paths, spec.id)
    (run / "train" / "final" / "params").mkdir(parents=True)
    (run / "train" / "final" / "assets").mkdir()
    (run / "run_spec.toml").write_text(dumps_armspec(spec))
    (run / "train" / "train_summary.json").write_text('{"trained_steps": 20605, "declared_steps": 20605}')
    return spec


def test_finetune_trains_from_a_finished_pretrain_and_names_the_evaluation(tmp_path, monkeypatch, capsys):
    hosts, paths = host(tmp_path)
    _finished_pretrain(paths)
    calls = []
    monkeypatch.setattr(ftcli, "train_spec", lambda path, **kw: calls.append((path, kw)) or 0)
    assert main(["train", "finetune", "--from", "pi05.pretrain.mine", "--host", str(hosts)]) == 0
    run = paths.runs / "pi05.finetune.mine"
    assert calls == [(run / "spec.toml", {"host": str(hosts), "fsdp_devices": None, "mode": "run"})]
    assert json.loads((run / "eval.json").read_text())["serving"] == ["bare"]
    assert capsys.readouterr().out.splitlines()[-1] == f"NEXT  qvla bench eval --from-run pi05.finetune.mine --host {hosts}"


def test_pretrain_trains_the_spec_it_wrote(tmp_path, monkeypatch):
    hosts, paths = host(tmp_path)
    calls = []
    monkeypatch.setattr(ptcli, "train_spec", lambda path, **kw: calls.append((path, kw["mode"])) or 0)
    assert main(["train", "pretrain", "--labels", "example.keep_top", "--backbone", "xvla", "--host", str(hosts)]) == 0
    assert calls == [(paths.runs / "xvla.pretrain.example.keep_top" / "spec.toml", "run")]


@pytest.mark.parametrize("argv", [["--labels", "Mine"], ["--labels", "mine", "--run", "../elsewhere"]])
def test_a_name_that_makes_no_valid_run_is_refused_before_anything_is_written(tmp_path, capsys, argv):
    hosts, paths = host(tmp_path)
    assert main(["train", "pretrain", *argv, "--backbone", "pi05", "--host", str(hosts), "--print"]) == 2
    assert capsys.readouterr().out.startswith("FAIL  ") and not paths.runs.exists()
    with pytest.raises(MethodRunError, match="not a pretraining run id"):
        finetune_run_id("pi05.pretrain.../../x", cond=False)


def test_a_planned_spec_is_never_replaced_by_another(tmp_path, capsys):
    hosts, paths = host(tmp_path)
    argv = ["train", "pretrain", "--labels", "mine", "--backbone", "smolvla", "--host", str(hosts), "--print"]
    assert main(argv) == 0 and main(argv) == 0                           # the same plan again is fine
    assert main(argv[:5] + ["xvla", "--run", "mine"] + argv[6:]) == 0      # another backbone, another id
    spec = paths.runs / "smolvla.pretrain.mine" / "spec.toml"
    spec.write_text(spec.read_text().replace("seed = 0", "seed = 1"))
    capsys.readouterr()
    assert main(argv) == 2
    assert "already holds another run spec: pass --run <another name>" in capsys.readouterr().out


def test_finetune_refuses_an_unfinished_pretrain_naming_the_resume(tmp_path, capsys):
    hosts, paths = host(tmp_path)
    _finished_pretrain(paths)
    (paths.runs / "pi05.pretrain.mine" / "train" / "train_summary.json").write_text(
        '{"trained_steps": 10, "declared_steps": 20605}')
    assert main(["train", "finetune", "--from", "pi05.pretrain.mine", "--host", str(hosts)]) == 2
    out = capsys.readouterr().out
    run = paths.runs / "pi05.pretrain.mine"
    assert out.strip() == (f"FAIL  pi05.pretrain.mine has not finished: it trained 10 of 20605 steps; when it is not "
                           f"still training, continue it with `qvla train {run / 'run_spec.toml'} --resume`")
    assert not (paths.runs / "pi05.finetune.mine").exists()


def test_finetune_of_a_run_nobody_planned_names_the_pretrain_command(tmp_path, capsys):
    hosts, _ = host(tmp_path)
    assert main(["train", "finetune", "--from", "pi05.pretrain.nobody", "--host", str(hosts), "--print"]) == 2
    out = capsys.readouterr().out
    assert "holds neither run_spec.toml nor spec.toml" in out and "`qvla train pretrain` plans one" in out


def test_the_pretrain_seed_defaults_to_the_papers_first_replicate_and_another_names_the_run():
    """--seed sets the replicate key `results contrast` pairs on (the pretrain seed); its default is
    each backbone's first Table I replicate, the shipped example's seed, and only another seed suffixes
    the id, which the fine-tune inherits while keeping the paper's own fine-tune seed."""
    t1 = json.loads((RUNS_DIR.parent / "paper.json").read_text())["table1"]["replicates"]
    for family in FAMILIES:
        assert PAPER_SEED[family] == load_armspec(EXAMPLES / f"{family}.pretrain.keep_top.toml").seed
        assert PAPER_SEED[family] == t1[family][0]["seed"]
        assert pretrain_spec(family, "mine").seed == PAPER_SEED[family]
        assert pretrain_spec(family, "mine", seed=PAPER_SEED[family]).id == f"{family}.pretrain.mine"
    s43 = pretrain_spec("pi05", "mine", seed=43)
    assert (s43.id, s43.seed) == ("pi05.pretrain.mine.s43", 43)
    assert pretrain_spec("xvla", "mine", run="r", seed=4).id == "xvla.pretrain.r.s4"
    ft, paper_ft = finetune_spec(s43, cond=True), finetune_spec(pretrain_spec("pi05", "mine"), cond=True)
    assert (ft.id, ft.seed) == ("pi05.finetune.mine.s43.cond", paper_ft.seed)
    with pytest.raises(MethodRunError, match="--seed"):
        pretrain_spec("pi05", "mine", seed=-1)


PI05_TABLE = "the paper's pi0.5 fine-tunes, pretrain seed -> fine-tune seed: 42 -> 42; 43 -> 42 (bc43: 43); 44 -> 42"
LEROBOT_TABLE = "pretrain seed -> fine-tune seed: 0 -> 0; 1 -> 1; 2 -> 2"


@pytest.mark.parametrize("family, seed, ft, why", [
    ("pi05", 43, 42, PI05_TABLE), ("pi05", 42, 42, PI05_TABLE),
    ("pi05", 7, 42, f"{PI05_TABLE}; any other pretrain seed -> 42"),
    ("smolvla", 1, 1, f"the paper's SmolVLA fine-tunes, {LEROBOT_TABLE}"),
    ("xvla", 2, 2, None), ("smolvla", 0, 0, None),
    ("xvla", 9, 9, f"the paper's X-VLA fine-tunes, {LEROBOT_TABLE}; any other pretrain seed -> itself"),
])
def test_the_finetune_seed_is_the_papers_convention_per_backbone(family, seed, ft, why):
    """Read from the paper's fine-tune run specs: pi0.5 fine-tunes use 42 at every pretrain seed, and
    bc43 uses 43; SmolVLA and X-VLA use the pretraining's seed (cells.json svla_bc_sd1 {1, 1}, r1xbc ... {0, 0})."""
    spec = finetune_spec(pretrain_spec(family, "mine", seed=seed), cond=False)
    assert spec.seed == ft
    got, text = paper_finetune_seed(FAMILIES[family], seed)
    assert got == ft and (why is None or text == why)


def test_a_finetune_run_name_keeps_the_pretrainings_seed_suffix(tmp_path):
    s43 = pretrain_spec("pi05", "mine", seed=43)
    assert finetune_spec(s43, cond=False, run="other").id == "pi05.finetune.other.s43"
    assert finetune_spec(s43, cond=True, run="other.s43").id == "pi05.finetune.other.s43.cond"
    assert finetune_spec(pretrain_spec("pi05", "mine"), cond=False, run="other").id == "pi05.finetune.other"
    hosts, paths = host(tmp_path)
    assert main(["train", "pretrain", "--labels", "mine", "--backbone", "smolvla", "--seed", "1",
                 "--host", str(hosts), "--print"]) == 0
    assert main(["train", "finetune", "--from", "smolvla.pretrain.mine.s1", "--run", "x", "--host", str(hosts),
                 "--print"]) == 0
    ft = load_armspec(paths.runs / "smolvla.finetune.x.s1" / "spec.toml")
    assert ft.seed == 1 and json.loads((paths.runs / ft.id / "eval.json").read_text())["pretrain_seed"] == 1
