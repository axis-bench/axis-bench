"""The paper's vocabulary on the command line: each renamed flag parses into the attribute the
command reads, and its old spelling is refused, never accepted as an alias."""
from __future__ import annotations

import pytest

from axis_qvla.cli import build_parser


def _parse(argv: list[str]):
    return build_parser().parse_args(argv)


def _refused(argv: list[str], capsys) -> str:
    with pytest.raises(SystemExit) as exc:
        _parse(argv)
    assert exc.value.code == 2
    return capsys.readouterr().err


def test_bench_eval_files_results_under_a_run(capsys):
    assert _parse(["bench", "eval", "--policy", "127.0.0.1:9000", "--run", "mine"]).arm == "mine"
    assert "--arm" in _refused(["bench", "eval", "--policy", "127.0.0.1:9000", "--arm", "mine"], capsys)


def test_bench_score_selects_runs(capsys):
    assert _parse(["bench", "score", "--run-root", "r1", "--run", "a", "--run", "b"]).arm == ["a", "b"]
    assert "--arm" in _refused(["bench", "score", "--run-root", "r1", "--arm", "a"], capsys)


def test_results_build_names_the_run(capsys):
    a = _parse(["results", "build", "--run-root", "r1", "--run", "mine", "--ft-seed", "0",
                "--pretrain-seed", "none", "--out", "o.json"])
    assert (a.run_root, a.arm) == ("r1", "mine")
    # --run is required, so the old spelling is refused as the missing new one.
    assert "required: --run" in _refused(["results", "build", "--run-root", "r1", "--arm", "mine", "--ft-seed", "0",
                                          "--pretrain-seed", "none", "--out", "o.json"], capsys)


def test_the_rollout_cli_takes_a_run_name(capsys):
    from axis_qvla.bench.sim.cli import build_parser as rollout_parser

    base = ["--task-id", "806", "--variant", "v", "--checkpoint-dir", "ck",
            "--log-root", "l", "--work-dir", "w", "--task-data-root", "t"]
    assert rollout_parser().parse_args([*base, "--run-name", "hx_806_v"]).arm_name == "hx_806_v"
    with pytest.raises(SystemExit) as exc:
        rollout_parser().parse_args([*base, "--arm-name", "hx_806_v"])
    assert exc.value.code == 2 and "--arm-name" in capsys.readouterr().err


def test_the_driver_takes_a_run(capsys):
    from axis_qvla.bench.sim import campaign

    with pytest.raises(SystemExit) as exc:
        campaign.main(["--arm", "probe", "--tasks", "806"])
    assert exc.value.code == 2 and "required: --run" in capsys.readouterr().err


def test_train_takes_a_run_spec():
    p = build_parser()
    assert p.parse_args(["train", "x.toml"]).arm.name == "x.toml"
    train = next(a for a in p._actions if a.dest == "command").choices["train"]
    assert "run_spec" in train.format_usage()


def test_env_takes_an_environment(capsys):
    from axis_qvla.train.envs import main as env_main

    with pytest.raises(SystemExit) as exc:
        env_main(["plan", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "environment" in out and "tier" not in out


def test_a_rendered_run_holds_its_run_spec(tmp_path):
    from axis_qvla.train.armspec import loads_armspec
    from axis_qvla.train.generate import render
    from axis_qvla.train.recipes import load_recipe
    from tests.unit.train._arms import arm_toml, host

    _, paths = host(tmp_path)
    spec = loads_armspec(arm_toml())
    files, _ = render(spec, load_recipe(spec.preset), paths)
    assert "run_spec.toml" in files and "armspec.toml" not in files


def test_the_labels_group_compiles_from_the_pretraining_data(capsys):
    a = _parse(["labels", "build-paper", "--pretraining-data", "p", "--episodes", "p/episodes", "--out", "o"])
    assert (str(a.pack), str(a.corpus), str(a.out)) == ("p", "p/episodes", "o")
    assert "invalid choice: 'signals'" in _refused(["signals", "check", "x"], capsys)
    # --pretraining-data is required, so the old spelling is refused as the missing new one.
    assert "required: --pretraining-data" in _refused(
        ["labels", "build-paper", "--pack", "p", "--episodes", "e", "--out", "o"], capsys)


def test_labels_check_reads_only_the_compiled_tree(capsys):
    assert str(_parse(["labels", "check", "compiled"]).bundles) == "compiled"
    assert "--pretraining-data" in _refused(["labels", "check", "compiled", "--pretraining-data", "p"], capsys)


def test_the_awr_builder_binds_by_names_only(capsys):
    import sys

    from axis_qvla.signals.methods import awr_chunk

    argv = ["awr_chunk", "--reward", "phase", "--out", "o.json", "--index", "i.json", "--binding", "mover"]
    old, sys.argv = sys.argv, argv
    try:
        with pytest.raises(SystemExit) as exc:
            awr_chunk.main()
    finally:
        sys.argv = old
    assert exc.value.code == 2 and "invalid choice: 'mover'" in capsys.readouterr().err


def test_label_study_and_bench_check_read_the_pretraining_data(capsys):
    assert str(_parse(["paper", "label-study", "--pretraining-data", "p"]).corpus) == "p"
    assert _parse(["bench", "check", "--pretraining-prompts", "f.json"]).corpus_prompts == "f.json"
    # --pretraining-data is required, so the old spelling is refused as the missing new one.
    assert "required: --pretraining-data" in _refused(["paper", "label-study", "--corpus", "p"], capsys)
    assert "--corpus-prompts" in _refused(["bench", "check", "--corpus-prompts", "f.json"], capsys)


def test_the_traces_sample_reads_episodes(capsys):
    from axis_qvla.labels import traces_sample

    with pytest.raises(SystemExit) as exc:
        traces_sample.main(["select", "--corpus", "c", "--roots", "r.json"])
    assert exc.value.code == 2 and "--episodes" in capsys.readouterr().err


def test_the_paper_prompt_list_is_named_for_the_pretraining_data():
    from tests.docs.runner import REPO

    assert (REPO / "experiments" / "paper" / "pretraining_prompts.json").is_file()
    assert not (REPO / "experiments" / "paper" / "corpus_prompts.json").exists()


def test_data_and_paper_read_the_checksum_manifest_and_the_results_map(capsys):
    a = _parse(["data", "get", "results", "--checksums", "l.json", "--results-map", "c.json"])
    assert (str(a.lock), str(a.cells)) == ("l.json", "c.json")
    t = _parse(["paper", "tables", "--results", "p", "--checksums", "l.json", "--results-map", "c.json"])
    assert (t.lock, t.cells) == ("l.json", "c.json")
    assert "--lock" in _refused(["data", "verify", "--lock", "l.json"], capsys)
    assert "--cells" in _refused(["paper", "tables", "--results", "p", "--cells", "c.json"], capsys)


def test_the_driver_and_the_rollout_cli_check_assets_against_the_checksum_manifest(capsys):
    from axis_qvla.bench import inputs
    from axis_qvla.bench.sim.cli import build_parser as rollout_parser

    assert any(t.startswith("--asset-checksums=") for t in inputs.path_argv(None, None, None, asset_lock="A.json"))
    base = ["--task-id", "806", "--variant", "v", "--checkpoint-dir", "ck",
            "--log-root", "l", "--work-dir", "w", "--task-data-root", "t"]
    assert str(rollout_parser().parse_args([*base, "--asset-checksums", "A.json"]).asset_lock) == "A.json"
    with pytest.raises(SystemExit):
        rollout_parser().parse_args([*base, "--asset-lock", "A.json"])
    assert "--asset-lock" in capsys.readouterr().err


def test_the_constants_page_speaks_the_paper_s_vocabulary():
    from tests.docs import constants_page

    titles = [t for t, _ in constants_page.REGISTRIES]
    assert titles == ["Labels and training weights", "Benchmark and scoring", "Reproducing the paper's tables",
                      "Training"]
    names = {c.name for _, cs in constants_page.constants() for c in cs}
    assert {"PAPER_RESULTS", "PRETRAINING_FRAMES", "FINETUNE_FRAMES", "FILTER_KEEP_FRAC"} <= names
    assert not names & {"T0_CELLS", "CORPUS_ROWS", "FINETUNE_ROWS", "KEEP_TOP_FRAC"}


def test_label_study_is_the_command_and_fig4_its_hidden_alias(capsys):
    """`qvla paper label-study` names the result by what it shows; `paper fig4` (the name before v1.0)
    still runs it and is listed nowhere."""
    from axis_qvla.cli import build_parser

    assert _parse(["paper", "label-study", "--pretraining-data", "p"]).func is _parse(
        ["paper", "fig4", "--pretraining-data", "p"]).func
    with pytest.raises(SystemExit):
        build_parser().parse_args(["paper", "--help"])
    out = capsys.readouterr().out
    assert "label-study" in out and "fig4" not in out
    with pytest.raises(SystemExit):                       # nor in a mistyped command's refusal
        build_parser().parse_args(["paper", "nonsense"])
    err = capsys.readouterr().err
    assert "invalid choice: 'nonsense'" in err and "label-study" in err and "fig4" not in err
