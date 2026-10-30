"""`qvla labels`: the group is registered, `export` writes compiled labels' legacy files, and `build-paper`
reports a refused input as one FAIL line."""
from __future__ import annotations

import numpy as np
import pytest

from axis_qvla.cli import main
from axis_qvla.signals.bundle import write_bundle
from axis_qvla.signals.paper import DeclaredFieldMissing
from axis_qvla.signals.program import RefusedDtype, Unscorable
from axis_qvla.signals.testing import TOY_TRAINABLE, toy_bundle


@pytest.mark.parametrize("argv", [["labels", "build-paper", "--help"], ["labels", "export", "--help"],
                                  ["labels", "check", "--help"]])
def test_every_signals_command_has_help(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 0


def test_export_writes_the_legacy_files(tmp_path, capsys):
    bid = write_bundle(toy_bundle(), tmp_path / "b")
    assert main(["labels", "export", str(tmp_path / "b" / bid), "--format", "lerobot-legacy",
                 "--out", str(tmp_path / "o")]) == 0
    with np.load(tmp_path / "o" / "signals" / "toy" / "lerobot_rows.npz") as z:
        assert z["rows"].tolist() == list(TOY_TRAINABLE)
    assert main(["labels", "export", str(tmp_path / "b" / bid), "--format", "openpi-legacy",
                 "--out", str(tmp_path / "o")]) == 2
    assert "no openpi-legacy files" in capsys.readouterr().out


@pytest.mark.parametrize("exc,line", [
    (DeclaredFieldMissing("source"), "FAIL  KeyError: 'source' is missing (an edited declaration, or a "
                                     "checksum-manifest or input record without it)"),
    (RefusedDtype("task_800--0: column 'observation.body_pose' holds ['float64'], not float32"),
     "FAIL  TypeError: task_800--0: column 'observation.body_pose' holds ['float64'], not float32"),
    (Unscorable("task_800--0", "no stable target contact"), "FAIL  Unscorable: task_800--0: no stable target contact"),
    (ValueError("--episodes must be the published pretraining episodes"), "FAIL  --episodes must be the published pretraining episodes"),
])
def test_build_paper_reports_a_refused_input_as_one_fail_line_with_exit_2(tmp_path, monkeypatch, capsys, exc, line):
    """An edited declaration (DeclaredFieldMissing), a refused dtype (RefusedDtype) or an episode a
    scorer cannot score (Unscorable) ends the command with one named FAIL line and exit 2, never a
    traceback."""
    from axis_qvla.signals import paper

    def refuse(**_kw):
        raise exc

    monkeypatch.setattr(paper, "build", refuse)
    argv = ["labels", "build-paper", "--pretraining-data", str(tmp_path), "--episodes", str(tmp_path),
            "--out", str(tmp_path / "o")]
    assert main(argv) == 2
    assert capsys.readouterr().out == line + "\n"


@pytest.mark.parametrize("exc", [KeyError("source"), TypeError("unexpected shape")])
def test_a_bug_shaped_error_raised_elsewhere_is_not_rewritten(tmp_path, monkeypatch, exc):
    """A bare KeyError or TypeError -- not the DeclaredFieldMissing/RefusedDtype subclass `build`
    raises for a refused input -- is a bug, not a refusal: `main` does not catch it, so it
    propagates with its own traceback instead of becoming a FAIL line."""
    from axis_qvla.signals import paper

    def buggy(**_kw):
        raise exc

    monkeypatch.setattr(paper, "build", buggy)
    argv = ["labels", "build-paper", "--pretraining-data", str(tmp_path), "--episodes", str(tmp_path),
            "--out", str(tmp_path / "o")]
    with pytest.raises(type(exc), match=str(exc.args[0])):
        main(argv)
