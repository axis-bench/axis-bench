"""`python -m axis_qvla.realrobot.fr3 {check,norm}`, end to end on the shipped record and a synthetic FR3 dataset."""
from __future__ import annotations

import json

import pytest

from axis_qvla.realrobot.fr3.cli import main
from axis_qvla.realrobot.realdata.lerobot_v21 import Dataset
from tests.unit.realrobot.fr3.fr3_sets import fr3_set
from tests.unit.realrobot.fr3.test_checks import _handshake


def test_check_with_presets_passes_on_the_shipped_line(capsys):
    assert main(["check", "--presets"]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == "ALL PASS"


def test_norm_then_check_norm_and_serve(tmp_path, capsys):
    ds = Dataset.open(fr3_set(tmp_path / "fr3_392"))
    out = tmp_path / "norm" / "norm_stats.json"
    assert main(["norm", "--dataset", str(ds.root), "--embodiment", "fr3", "--out", str(out)]) == 0
    assert set(json.loads(out.read_text())["norm_stats"]) == {"state", "actions"}
    (tmp_path / "hs.json").write_text(json.dumps(_handshake()))
    assert main(["check", "--norm", str(out), "--dataset", str(ds.root),
                 "--run", "pi05.finetune.realrobot_cond", "--handshake", str(tmp_path / "hs.json")]) == 0
    (tmp_path / "hs.json").write_text(json.dumps(_handshake(state_order=["gripper", *[f"q{i}" for i in range(7)]])))
    assert main(["check", "--dataset", str(ds.root), "--run", "pi05.finetune.realrobot_cond",
                 "--handshake", str(tmp_path / "hs.json")]) == 1
    out_lines = capsys.readouterr().out.splitlines()
    assert out_lines[-1] == "1 problem" and out_lines[-2].startswith("FAIL  state order")


def test_a_failing_check_exits_1_and_names_each_problem(tmp_path, capsys):
    (tmp_path / "arms").mkdir()
    assert main(["check", "--runs", str(tmp_path / "arms")]) == 1
    out = capsys.readouterr().out.splitlines()
    assert out[-1] == "9 problems" and all(x.startswith("FAIL  ") and x.endswith("no ArmSpec") for x in out[:-1])


@pytest.mark.parametrize("argv, needs", [
    (["check", "--handshake", "hs.json"], "--handshake needs --run and --dataset"),
    (["check", "--norm", "n.json"], "--norm needs --dataset"),
    (["check", "--run", "pi05.finetune.realrobot_cond"], "--run needs --handshake"),
    (["check", "--dataset", "d"], "--dataset needs --handshake or --norm"),
    (["check", "--embodiment", "realalign_sim"], "--embodiment needs --norm or --handshake"),
])
def test_an_incomplete_check_is_refused_by_the_parser(argv, needs, capsys):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2 and needs in capsys.readouterr().err


@pytest.mark.parametrize("embodiment", ["fr3", "realalign_sim"])
def test_a_dataset_that_is_not_there_is_a_named_refusal(tmp_path, capsys, embodiment):
    nowhere = tmp_path / "nowhere"
    assert main(["norm", "--dataset", str(nowhere), "--embodiment", embodiment, "--out", str(tmp_path / "n.json")]) == 2
    assert capsys.readouterr().err.startswith(f"FAIL  {nowhere}: cannot read the {embodiment} training set")
    assert not (tmp_path / "n.json").exists()
    (tmp_path / "n.json").write_text('{"norm_stats": {}}')
    assert main(["check", "--norm", str(tmp_path / "n.json"), "--dataset", str(nowhere), "--embodiment", embodiment]) == 1
    assert f"FAIL  {nowhere}: cannot read the {embodiment} training set" in capsys.readouterr().out


def test_a_server_of_your_demonstrations_is_checked_from_a_check_server_report(tmp_path, capsys):
    from tests.unit.realrobot.fr3.test_checks import _demos
    demos = _demos(tmp_path)
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"endpoint": "127.0.0.1:8000", "ok": False, "handshake": _handshake(prompt_suffix="")}))
    assert main(["check", "--run", "pi05.finetune.realrobot_only", "--handshake", str(report), "--dataset", str(demos),
                 "--embodiment", "fr3_droid"]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == "ALL PASS"
    assert main(["check", "--run", "pi05.finetune.realrobot_cond", "--handshake", str(report), "--dataset", str(demos),
                 "--embodiment", "fr3_droid"]) == 1
    assert "FAIL  prompt suffix ''" in capsys.readouterr().out
