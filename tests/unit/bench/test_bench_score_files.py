"""`qvla bench score` over results files: pooled and per-task success with bootstrap intervals;
--run-root keeps the arm-summary table, with the seed set each arm read."""
from __future__ import annotations

import json
from pathlib import Path

from axis_qvla.bench import spec
from axis_qvla.bench.results.format import dumps_results
from axis_qvla.bench.spec import load_benchmark
from axis_qvla.cli import main
from tests.unit.bench.results.fixtures import make_bench, make_dev_results, make_results


def test_files_get_pooled_and_per_task_intervals(tmp_path: Path, capsys):
    conf = make_bench(tmp_path / "conf", [806, 1050], 3)
    bench = load_benchmark("fixture_bench", conf_dir=conf)
    f = tmp_path / "a.json"
    f.write_bytes(dumps_results(make_results({806: [True, True, False], 1050: [False, False, True]}, bench=bench)))
    common = ["--benchmark", "fixture_bench", "--bench-conf-dir", str(conf)]
    assert main(["bench", "score", str(f), "--bootstrap", "300", "--per-task", *common]) == 0
    out = capsys.readouterr().out
    assert "armx_dualsim\tall24\t50.00 ±" in out and "95% CI [" in out and "B=300" in out
    assert "  task 806\t2/3\t66.7  [" in out and "  task 1050\t1/3\t33.3  [" in out
    # per task: the Wilson score interval, which needs no resampling
    assert "95% Wilson score interval" in out
    assert "  task 806\t2/3\t66.7  [20.8, 93.9]" in out and "  task 1050\t1/3\t33.3  [6.1, 79.2]" in out
    assert main(["bench", "score", str(f), "--bootstrap", "0", "--per-task", *common]) == 0
    out = capsys.readouterr().out
    assert "CI" not in out and "  task 806\t2/3\t66.7  [20.8, 93.9]" in out
    assert main(["bench", "score", str(f), "--bootstrap", "-1", *common]) == 2


def test_a_refused_file_leaves_no_partial_table(tmp_path: Path, capsys):
    conf = make_bench(tmp_path / "conf", [806, 1050], 3)
    bench = load_benchmark("fixture_bench", conf_dir=conf)
    good, off = tmp_path / "good.json", tmp_path / "off.json"
    good.write_bytes(dumps_results(make_results({806: [True] * 3, 1050: [False] * 3}, bench=bench)))
    doc = json.loads(good.read_bytes())
    doc["header"]["execution"]["off_protocol"] = True
    off.write_text(json.dumps(doc))
    common = ["--benchmark", "fixture_bench", "--bench-conf-dir", str(conf), "--bootstrap", "20"]
    assert main(["bench", "score", str(good), str(off), *common]) == 2
    out = capsys.readouterr().out
    assert out.startswith("FAIL") and "armx_dualsim\tall24" not in out


def test_a_development_read_and_an_off_protocol_read_say_so(tmp_path: Path, capsys):
    conf = make_bench(tmp_path / "conf", [806, 1050], 3, extra={"dev_seeds": {"offset": 500, "n": 3}})
    bench = load_benchmark("fixture_bench", conf_dir=conf)
    dev, off = tmp_path / "dev.json", tmp_path / "off.json"
    dev.write_bytes(dumps_results(make_dev_results(bench, {806: [True] * 3, 1050: [False] * 3})))
    r = make_results({806: [True] * 3, 1050: [False] * 3}, bench=bench)
    doc = json.loads(dumps_results(r))
    doc["header"]["execution"]["off_protocol"] = True
    off.write_text(json.dumps(doc))
    common = ["--benchmark", "fixture_bench", "--bench-conf-dir", str(conf), "--bootstrap", "50"]
    assert main(["bench", "score", str(dev), *common]) == 0
    assert "[development seeds]" in capsys.readouterr().out
    assert main(["bench", "score", str(off), *common]) == 2
    assert "--allow-off-protocol" in capsys.readouterr().out
    assert main(["bench", "score", str(off), "--allow-off-protocol", *common]) == 0
    assert "OFF-PROTOCOL" in capsys.readouterr().out


def test_files_and_a_run_root_are_one_or_the_other(tmp_path: Path, capsys):
    assert main(["bench", "score"]) == 2
    assert main(["bench", "score", "x.json", "--run-root", str(tmp_path)]) == 2
    assert "not both" in capsys.readouterr().out
    assert main(["bench", "score", "x.json", "--run", "a"]) == 2
    assert "--run selects runs under --run-root" in capsys.readouterr().out


def test_the_run_root_table_refuses_the_options_it_does_not_use(tmp_path: Path, capsys):
    for flag in (["--view", "all24"], ["--bootstrap", "0"], ["--per-task"], ["--allow-off-protocol"],
                 ["--benchmark", "franka_heldout24"]):
        assert main(["bench", "score", "--run-root", str(tmp_path), *flag]) == 2
        out = capsys.readouterr().out
        assert f"{flag[0]} apply to results files" in out


def test_the_run_root_table_names_each_arms_seed_set(tmp_path: Path, capsys):
    ident = {**spec.load_benchmark().identity(), "matches_manifest": True}
    for arm, extra in (("mine", {}), ("mine_dev", {"seed_set": "dev"})):
        d = tmp_path / f"out_{arm}"
        d.mkdir()
        (d / "arm_summary.json").write_text(json.dumps({
            "arm": arm, "served_arm": f"{arm}_dualsim", "all24_pct": 50.0, "rollouts": 1200, "successes": 600,
            "eval_fingerprint": "ab" * 32, "benchmark": ident, "incomplete_tasks": [], "errored_rollouts": 0,
            **extra}))
    assert main(["bench", "score", "--run-root", str(tmp_path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split()[:5] == ["run", "served", "all24%", "n", "seeds"]
    rows = {ln.split()[0]: ln.split() for ln in lines[1:]}
    assert rows["mine"][4] == "test" and rows["mine_dev"][4] == "dev"
