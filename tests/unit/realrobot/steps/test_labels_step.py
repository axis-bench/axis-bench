"""Step 4: the label chain on the layout. Phase is resumable by task, gates stop it, and no digest is printed."""
from __future__ import annotations

import re

import numpy as np
import pytest

from axis_qvla.cli import main
from axis_qvla.realrobot.layout import layout
from axis_qvla.realrobot.realign import curves as curves_mod
from axis_qvla.realrobot.realign.curves import PhaseCurve, read_dumped_curve_store, write_curve_store
from axis_qvla.realrobot.render.gates import GateError
from axis_qvla.realrobot.steps import labels as step
from tests.unit.realrobot.fr3.local_fixture import _add_droid8d_columns
from tests.unit.realrobot.realign.test_driver import _dumped_roots_fixture
from tests.unit.realrobot.steps.hosts import qvla_toml


def _rendered(tmp_path, monkeypatch):
    """The layout with a six-episode training view (the realign driver's synthetic corpus) and stand-ins for
    the three render-side stages: phase and gates do nothing, and the curve store is the corpus's own."""
    toml = qvla_toml(tmp_path)
    lay = layout(str(toml))
    roots, store, _tid, _n = _dumped_roots_fixture(lay.base / "fx")
    _add_droid8d_columns(lay.base / "fx" / "corpus")
    lay.d8.symlink_to("fx/corpus", target_is_directory=True)
    lay.render.mkdir()
    lay.source.mkdir()
    (lay.source / "manifest.json").write_text("{}")
    calls = []
    monkeypatch.setattr(step, "_phase", lambda d8, source, att: calls.append("phase") or
                        {"episodes": 6, "neutral": 0, "computed": 1, "kept": 0})
    monkeypatch.setattr(step, "_curves", lambda d8, att, out: calls.append("curves") or
                        write_curve_store(out, read_dumped_curve_store(store, roots)) or 6)
    monkeypatch.setattr(step, "_gates", lambda *a: calls.append("gates"))
    return str(toml), lay, calls


def test_the_chain_runs_in_order_and_names_compiled_labels_by_run(tmp_path, monkeypatch, capsys):
    toml, lay, calls = _rendered(tmp_path, monkeypatch)
    assert main(["realrobot", "labels", "--host", toml]) == 0
    assert calls == ["phase", "curves", "gates"]
    assert (lay.built / "runs.json").is_file() and (lay.bundles / "realign_compiled.json").is_file()
    out = capsys.readouterr().out
    assert "compiled labels realrobot.bc, realrobot.conditioning ->" in out
    assert re.search(r"[0-9a-f]{64}", out) is None


def test_a_failed_render_gate_stops_the_chain(tmp_path, monkeypatch):
    toml, lay, _calls = _rendered(tmp_path, monkeypatch)

    def fail_gates(*a):
        raise GateError("home gate: frame 2 moved")

    monkeypatch.setattr(step, "_gates", fail_gates)
    assert main(["realrobot", "labels", "--host", toml]) == 1
    assert not lay.built.exists()


def test_a_missing_render_names_the_step_that_writes_it(tmp_path, capsys):
    toml = qvla_toml(tmp_path)
    assert main(["realrobot", "labels", "--host", str(toml)]) == 2
    assert "qvla realrobot rerender" in capsys.readouterr().err


def test_phase_resumes_by_task(tmp_path, monkeypatch):
    import json
    from axis_qvla.realrobot.realign import phase
    d8, att = tmp_path / "d8", tmp_path / "attempts"
    d8.mkdir()
    (d8 / "roots.json").write_text(json.dumps({"version": 2, "base": ".", "roots": {"1": "task_1", "2": "task_2"}}))
    curves_mod.write_attempt_store(att, {(1, 70): PhaseCurve(np.zeros(4, np.float32), np.zeros(4, bool), None)})
    asked = []
    monkeypatch.setattr(phase, "phase_curves", lambda roots, data, tasks: asked.append(tasks) or
                        {(2, 71): PhaseCurve(np.ones(5, np.float32), np.ones(5, bool), 2)})
    assert step._phase(d8, tmp_path / "source", att) == {"episodes": 2, "neutral": 1, "computed": 1, "kept": 1}
    assert asked == [[2]]


def test_the_attempt_store_is_written_whole_or_not_at_all(tmp_path, monkeypatch):
    def boom(f, **arrs):
        f.write(b"half")
        raise OSError("disk full")

    monkeypatch.setattr(curves_mod.np, "savez_compressed", boom)
    with pytest.raises(OSError):
        curves_mod.write_attempt_store(tmp_path, {(3, 9): PhaseCurve(np.zeros(2, np.float32), np.zeros(2, bool), None)})
    assert not (tmp_path / "task_3.npz").exists()
