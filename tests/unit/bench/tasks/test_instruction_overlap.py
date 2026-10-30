"""Instruction overlap between the benchmark and the pretraining corpus: reported, with the
episodes it touches. Task ids never overlap (test_heldout_leakage.py); instructions do, for three
tasks, and the paper says so."""
from __future__ import annotations

import json
from pathlib import Path

from axis_qvla.bench.spec import load_benchmark
from axis_qvla.bench.tasks.leakage import instruction_overlap, load_corpus_prompts, normalise_instruction, read_roots

REPO = Path(__file__).resolve().parents[4]
PROMPTS = REPO / "experiments" / "paper" / "pretraining_prompts.json"
ROOTS = REPO / "tests" / "golden" / "bench" / "roots_libero_5k_v2_d8.json"


def test_the_rule_folds_case_whitespace_and_a_corpus_variant_number():
    assert normalise_instruction("Rearrange Three Items 2", variant=True) == normalise_instruction(" rearrange  three items")
    assert normalise_instruction("Fit lid onto container ") == "fit lid onto container"
    assert normalise_instruction("Stack 2 cups", variant=True) == "stack 2 cups"   # only a TRAILING number
    assert normalise_instruction("task 806") == "task 806"                         # benchmark side: kept


def test_the_published_corpus_prompts_cover_the_corpus():
    corpus = load_corpus_prompts(PROMPTS)
    assert set(corpus) == set(read_roots(ROOTS)) and len(corpus) == 731
    assert sum(t["n_episodes"] for t in corpus.values()) == 5000


def test_three_instructions_four_corpus_tasks_22_episodes():
    found = instruction_overlap(load_benchmark(), load_corpus_prompts(PROMPTS))
    assert [(o.task_id, o.corpus_tasks) for o in found] == [
        (1426, (("1427", 4), ("1605", 4))), (1459, (("1152", 10),)), (1527, (("1528", 4),))]
    assert sum(o.episodes for o in found) == 22


def test_a_synthetic_overlap(tmp_path):
    from tests.unit.bench.results.fixtures import make_bench

    b = load_benchmark("fixture_bench", conf_dir=make_bench(tmp_path / "conf", [806, 1050], 3))
    # The corpus spells the benchmark's own instruction in another case, as a second variant.
    p = tmp_path / "prompts.json"
    p.write_text(json.dumps({"schema": "qvla.corpus_prompts/1",
                             "tasks": {"5": {"prompt": f"{b.task(806).prompt.upper()} 2", "n_episodes": 3}}}))
    [o] = instruction_overlap(b, load_corpus_prompts(p))
    assert (o.task_id, o.episodes) == (806, 3)


def test_bench_check_reports_an_unreadable_prompt_list_as_a_failure_not_a_traceback(tmp_path, capsys):
    from axis_qvla.cli import main

    bad = tmp_path / "bad.json"
    for path, text in ((tmp_path / "absent.json", None), (bad, "{not json"),
                       (bad, json.dumps({"schema": "other/1"})),
                       (bad, json.dumps({"schema": "qvla.corpus_prompts/1", "tasks": {"5": {}}}))):
        if text is not None:
            path.write_text(text)
        assert main(["bench", "check", "--pretraining-prompts", str(path)]) == 2
        out = capsys.readouterr().out
        assert f"FAIL  --pretraining-prompts {path} cannot be read as a prompt list" in out
