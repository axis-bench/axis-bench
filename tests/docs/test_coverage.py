"""Coverage of the at-scale blocks (tests/docs/coverage.py): the keys, the static rule, and the
measured one, where the shims' log decides."""
from __future__ import annotations

import pathlib

from tests.docs import blocks as bl
from tests.docs import coverage, runner


def _page(tmp_path: pathlib.Path, text: str) -> pathlib.Path:
    page = tmp_path / "page.md"
    page.write_text(text)
    return page


def test_a_command_key_is_what_ci_must_run():
    assert bl.command_key("qvla realrobot render render --episodes e.json --out r") == "qvla realrobot render render"
    assert bl.command_key("qvla train my_arm.toml --smoke") == "qvla train"
    assert bl.command_key("! qvla realrobot realdata crop224 --dataset d") == "qvla realrobot realdata crop224"
    assert bl.command_key("AXIS_X=1 python my_tools/my_method.py --toy") == "python my_tools/my_method.py"
    assert bl.command_key("nohup env A=1 python3.11 -m my.mod --x") == "python -m my.mod"
    assert bl.command_key("third_party/lerobot/.venv/bin/python -m lerobot.scripts.axis_merge_datasets --out m") == \
        "python -m lerobot.scripts.axis_merge_datasets"
    assert [bl.command_key(x) for x in ("hf download x", "pip install -e .", "mkdir -p d", "# qvla x")] == \
        [None] * 4


def test_commands_are_keyed_one_per_pipeline_command_across_continuations():
    text = "qvla train a.toml | tee log\npython \\\n  -m my.mod --x\ncat a | qvla bench check > out\n"
    assert bl.commands(text) == ["qvla train", "python -m my.mod", "qvla bench check"]


def test_a_scale_command_is_covered_by_the_same_command_a_fetch_or_its_checker():
    page = ("```bash gpu\nqvla data get corpus --dest c\nqvla train a.toml\n"
            "python -m lerobot.scripts.axis_merge_datasets --out m a b\n```\n"
            "```bash ci\nqvla train a.toml --smoke\nqvla train merge-check d/a d/b\n```\n")
    assert bl.problems(page) == []
    assert coverage.needed([bl.blocks(page)[0].text]) == {"qvla train", "qvla train merge-check"}


def test_the_shims_log_each_command_a_block_starts_and_not_what_those_start(tmp_path):
    page = _page(tmp_path, "```bash ci\nqvla --help > /dev/null\n"
                           "python -c 'import subprocess; subprocess.run([\"qvla\", \"--help\"], check=True)' > /dev/null\n"
                           "python3 -c 'pass'\n```\n")
    assert runner.run_page(page, ("ci",), tmp_path / "run") == []
    logged = (tmp_path / "run" / coverage.RAN).read_text().splitlines()
    assert [line.split()[0] for line in logged] == ["qvla", "python", "python3"]
    assert coverage.ran("\n".join(logged)) == {"qvla", "python -c"}


def test_a_scale_command_a_run_block_names_but_never_runs_fails_the_page(tmp_path):
    """The block names `qvla train env plan openpi`, but the command it runs is not one the shims
    start for it (here the shim is told it is nested, so it logs nothing): the measured rule fails it."""
    gpu = "```bash gpu\nqvla train env plan openpi\n```\n"
    ran = _page(tmp_path, gpu + "```bash ci\nqvla train env plan openpi --help > /dev/null\n```\n")
    assert bl.problems(ran.read_text()) == [] and runner.run_page(ran, ("ci",), tmp_path / "a") == []
    hidden = _page(tmp_path, gpu + "```bash ci\nQVLA_DOCS_NESTED=1 qvla train env plan openpi --help > /dev/null\n```\n")
    assert bl.problems(hidden.read_text()) == []
    assert runner.run_page(hidden, ("ci",), tmp_path / "b") == [
        "page.md: `qvla train env plan openpi` is named by a `ci` block but never ran"]
