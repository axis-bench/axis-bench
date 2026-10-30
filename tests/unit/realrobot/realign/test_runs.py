"""Run-selective `realign build/gates/compile`.

A build that only wants {bc, conditioning} does not fit phase-balanced AWR or produce the drop-top
filtering schedule: those are the 'awr'/'filtering' runs' own preconditions. So curves that are fine for
conditioning (real per-episode progress variation, valid quintiles) build for it even when they are
degenerate for AWR: a pooled capped rate of exactly 0 in the phase-balanced fit, which lowering
`--min-fit-pool` cannot rescue because the refusal is about the RATE, not the pool size.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from axis_qvla.realrobot.realign.runs import UnbuiltRunError
from axis_qvla.realrobot.realign.bundles import compile_realign, write_bundles
from axis_qvla.realrobot.realign.cli import build, main
from axis_qvla.realrobot.realign.gates import run_gates
from axis_qvla.signals.blocks.awr import AwrWeightError

TASKS = (800, 912, 1450)
N_EP, T = 70, 180
SLOPE = 0.3 / 140             # the approach ramp's rate; scaled by t0 (not a fixed 0.3 endpoint)
NOISE_STD = 0.004             # the same per-frame noise the full-signal export_dir uses
MANIP_AMP = (0.0005, 0.001)   # ~300-1400x smaller than a typical export_dir's (0.2, 0.7)


def _export_curves(tmp_path):
    """Curves with real per-episode progress variation (quintiles 1-5 all populated -- rank weights
    only need ORDER, not scale) but whose advantage never nears the AWR cap threshold
    ln(2)/10 ~= 0.0693 in either phase: valid for conditioning, degenerate for AWR."""
    rng = np.random.default_rng(3)
    curves_dir = tmp_path / "export_dir"
    curves_dir.mkdir()
    amap, prompts = {}, {}
    for tid in TASKS:
        prompts[str(tid)] = f"do task {tid}"
        export_of = rng.permutation(N_EP)
        arrs = {"_provenance": np.array("{}")}
        for ep in range(N_EP):
            x = int(export_of[ep])
            length = T + int(rng.integers(-40, 41))
            amap[f"task_{tid}--{ep}"] = {"attempt_id": 10_000 * tid + ep, "export_episode_index": x,
                                         "train_frame_count": length, "export_frame_count": length}
            if ep % 23 == 5:  # neutral: no onset, nothing valid
                arrs[f"phase_ep_{x}"] = np.zeros((2, length), np.float32)
                continue
            t0 = int(rng.integers(length // 4, 3 * length // 4))
            approach = np.linspace(0.0, SLOPE * t0, t0) + rng.normal(0, NOISE_STD, t0)
            steps = rng.gamma(0.6, 1.0, length - t0)
            manip = SLOPE * t0 + rng.uniform(*MANIP_AMP) * np.cumsum(steps) / steps.sum()
            curve = np.clip(np.r_[approach, manip], 0, 1).astype(np.float32)
            valid = np.ones(length, np.float32)
            valid[length - int(rng.integers(0, 6)):] = 0.0
            arrs[f"phase_ep_{x}"] = np.stack([curve, valid])
            arrs[f"t0_ep_{x}"] = np.int64(t0)
        np.savez(curves_dir / f"task_{tid}.npz", **arrs)
    (tmp_path / "map.json").write_text(json.dumps({"episodes": amap}))
    (tmp_path / "prompts.json").write_text(json.dumps(prompts))
    return curves_dir, tmp_path / "map.json", tmp_path / "prompts.json"


def test_the_fixture_is_degenerate_for_awr_even_at_min_fit_pool_one(tmp_path):
    """Sanity check on the fixture itself: the phase-balanced fit sees a pooled capped rate of
    exactly 0 in both phases, and a smaller --min-fit-pool does not rescue it."""
    from axis_qvla.realrobot.realign.curves import load_export_curves
    from axis_qvla.realrobot.realign.weights import balanced_weights
    export_dir, amap, prompts = _export_curves(tmp_path)
    curves = load_export_curves(export_dir, amap)
    with pytest.raises(AwrWeightError, match=r"degenerate pooled capped rate 0\.0000"):
        balanced_weights(curves, min_pool=1)


def test_a_full_runs_build_still_refuses_on_this_fixture(tmp_path):
    """The default (every run) build fits AWR, so a curve set degenerate for AWR fails it. Run
    selection changes what a build NEEDS, never whether AWR's own refusal fires when AWR is asked for."""
    export_dir, amap, prompts = _export_curves(tmp_path)
    with pytest.raises(AwrWeightError, match=r"degenerate pooled capped rate 0\.0000"):
        build(tmp_path / "out", export_curves=export_dir, alignment_map=amap, prompts=prompts, min_fit_pool=1)
    assert not (tmp_path / "out" / "gates.json").exists()


def test_a_build_for_bc_and_conditioning_does_not_fit_awr_or_need_the_filtering_schedule(tmp_path):
    export_dir, amap, prompts = _export_curves(tmp_path)
    out = tmp_path / "out"
    summary = build(out, export_curves=export_dir, alignment_map=amap, prompts=prompts,
                    min_fit_pool=1, runs=("bc", "conditioning"))
    assert summary["episodes"] == len(TASKS) * N_EP and "phase_balance" not in summary
    assert not (out / "awr_weights_phase.json").exists()
    assert not (out / "awr_weights_phase_ep.json").exists()
    assert not (out / "schedule_drop_top.npz").exists()
    for name in ("roots.json", "ranges.json", "awr_weights_phase_rank.json", "quality_phase.npz", "gates.json"):
        assert (out / name).exists(), name

    gates = json.loads((out / "gates.json").read_text())
    assert set(gates) == {"G1", "G2", "G3", "G4"} and all(g["ok"] for g in gates.values()), gates

    assert json.loads((out / "runs.json").read_text())["runs"] == ["bc", "conditioning"]

    compiled = compile_realign(out, runs=("bc", "conditioning"))
    assert set(compiled) == {"bc", "conditioning"}

    ids = write_bundles(out, tmp_path / "bundles", runs=("bc", "conditioning"))
    assert set(ids) == {"realalign_bc", "realalign_conditioning"}


def test_cli_build_with_runs_bc_conditioning_succeeds_through_argparse(tmp_path, capsys):
    export_dir, amap, prompts = _export_curves(tmp_path)
    out = tmp_path / "out"
    code = main(["build", "--out", str(out), "--export-curves", str(export_dir), "--alignment-map", str(amap),
                 "--prompts", str(prompts), "--min-fit-pool", "1", "--runs", "bc", "conditioning"])
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert "phase_balance" not in summary


def test_compile_refuses_a_run_the_build_did_not_produce(tmp_path, capsys):
    export_dir, amap, prompts = _export_curves(tmp_path)
    out = tmp_path / "out"
    build(out, export_curves=export_dir, alignment_map=amap, prompts=prompts, min_fit_pool=1, runs=("bc", "conditioning"))
    code = main(["compile", "--built", str(out), "--out", str(tmp_path / "b"), "--runs", "awr"])
    err = capsys.readouterr().err
    assert code == 2 and "awr" in err and "bc" in err and "conditioning" in err
    assert not (tmp_path / "b").exists()


def test_gates_defaults_to_the_build_s_own_recorded_runs(tmp_path, capsys):
    export_dir, amap, prompts = _export_curves(tmp_path)
    out = tmp_path / "out"
    build(out, export_curves=export_dir, alignment_map=amap, prompts=prompts, min_fit_pool=1, runs=("bc", "conditioning"))
    assert main(["gates", "--out", str(out)]) == 0
    assert json.loads(capsys.readouterr().out) == {"G1": True, "G2": True, "G3": True, "G4": True}


def test_build_rejects_an_unknown_run_name(tmp_path):
    export_dir, amap, prompts = _export_curves(tmp_path)
    with pytest.raises(ValueError, match="nope"):
        build(tmp_path / "out", export_curves=export_dir, alignment_map=amap, prompts=prompts, runs=("bc", "nope"))
    assert not (tmp_path / "out").exists()


def test_the_python_api_also_refuses_a_run_the_build_did_not_produce(tmp_path):
    """The named refusal is not CLI-only: `run_gates`/`compile_realign`/`write_bundles` called
    directly in Python raise `UnbuiltRunError` naming the missing run and what the build has."""
    export_dir, amap, prompts = _export_curves(tmp_path)
    out = tmp_path / "out"
    build(out, export_curves=export_dir, alignment_map=amap, prompts=prompts, min_fit_pool=1, runs=("bc", "conditioning"))

    with pytest.raises(UnbuiltRunError, match=r"\['awr'\].*\['bc', 'conditioning'\]"):
        run_gates(out, runs=("bc", "awr"))

    with pytest.raises(UnbuiltRunError, match=r"\['awr'\].*\['bc', 'conditioning'\]"):
        compile_realign(out, runs=("bc", "awr"))

    with pytest.raises(UnbuiltRunError, match=r"\['awr'\].*\['bc', 'conditioning'\]"):
        write_bundles(out, tmp_path / "bundles", runs=("bc", "awr"))
    assert not (tmp_path / "bundles").exists()
