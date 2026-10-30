"""End to end: export-keyed curves -> gated artifacts, and every gate catching what it exists to catch."""
from __future__ import annotations

import json

import numpy as np
import pytest

from axis_qvla.realrobot.realign.cli import build, main
from axis_qvla.realrobot.realign.gates import GateError, run_gates

TASKS = (800, 912, 1450)
N_EP, T = 70, 180


def _export_curves(tmp_path):
    """Curves keyed by a SHUFFLED export index, an alignment map, prompts."""
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
            approach = np.linspace(0.0, 0.3, t0) + rng.normal(0, 0.004, t0)
            steps = rng.gamma(0.6, 1.0, length - t0)
            manip = 0.3 + rng.uniform(0.2, 0.7) * np.cumsum(steps) / steps.sum()
            curve = np.clip(np.r_[approach, manip], 0, 1).astype(np.float32)
            valid = np.ones(length, np.float32)
            valid[length - int(rng.integers(0, 6)):] = 0.0
            arrs[f"phase_ep_{x}"] = np.stack([curve, valid])
            arrs[f"t0_ep_{x}"] = np.int64(t0)
        np.savez(curves_dir / f"task_{tid}.npz", **arrs)
    (tmp_path / "map.json").write_text(json.dumps({"episodes": amap}))
    (tmp_path / "prompts.json").write_text(json.dumps(prompts))
    return curves_dir, tmp_path / "map.json", tmp_path / "prompts.json"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("realign")
    export_dir, amap, prompts = _export_curves(tmp)
    out = tmp / "out"
    summary = build(out, export_curves=export_dir, alignment_map=amap, prompts=prompts)
    return out, summary


ARTIFACTS = ("roots.json", "ranges.json", "awr_weights_phase_rank.json", "awr_weights_phase.json",
             "awr_weights_phase_ep.json", "quality_phase.npz", "schedule_drop_top.npz", "gates.json")


def test_one_command_writes_every_artifact_and_the_gates_pass(built):
    out, summary = built
    for name in ARTIFACTS:
        assert (out / name).exists(), name
    gates = json.loads((out / "gates.json").read_text())
    assert all(g["ok"] for g in gates.values()), gates
    assert summary["episodes"] == len(TASKS) * N_EP and summary["neutral"] == 9


def test_episode_weights_and_schedule_are_the_unchanged_tools_outputs(built):
    out, _ = built
    ep = json.loads((out / "awr_weights_phase_ep.json").read_text())
    assert ep["reward_id"] == "phase_ep"
    sched = np.load(out / "schedule_drop_top.npz")
    meta = json.loads(str(sched["meta"]))
    assert meta["mode"] == "drop_top" and meta["reward_id"] == "phase_ep" and sched["rows"].shape == (20000, 64)
    # keep_fraction is the REALIZED share of trainable rows kept: whole episodes, so near 0.5, not equal.
    assert abs(meta["keep_fraction"] - 0.5) < 0.01 and meta["seed"] == 0


def test_a_set_corpus_root_override_is_refused(tmp_path, monkeypatch):
    export_dir, amap, prompts = _export_curves(tmp_path)
    monkeypatch.setenv("AXIS_PRETRAIN_CORPUS_ROOT", str(tmp_path))
    with pytest.raises(RuntimeError, match="AXIS_PRETRAIN_CORPUS_ROOT"):
        build(tmp_path / "out", export_curves=export_dir, alignment_map=amap, prompts=prompts)


def _copy(out, tmp_path):
    import shutil
    dst = tmp_path / "copy"
    shutil.copytree(out, dst)
    return dst


def _gate_fails(out, gate):
    with pytest.raises(GateError) as err:
        run_gates(out)
    assert gate in str(err.value)
    report = json.loads((out / "gates.json").read_text())
    assert not report[gate]["ok"]


def test_g1_catches_a_curve_of_the_wrong_length(built, tmp_path):
    out = _copy(built[0], tmp_path)
    f = out / "curves" / "task_800_phase_curves.npz"
    z = dict(np.load(f))
    z["ep_0"] = z["ep_0"][:-1]
    z["valid_0"] = z["valid_0"][:-1]
    np.savez_compressed(f, **z)
    _gate_fails(out, "G1")


def test_g3_catches_a_missing_weights_entry(built, tmp_path):
    out = _copy(built[0], tmp_path)
    p = out / "awr_weights_phase.json"
    doc = json.loads(p.read_text())
    del doc["weights"]["task_912--3"]
    p.write_text(json.dumps(doc))
    _gate_fails(out, "G3")


def test_g3_catches_a_weights_file_from_another_corpus(built, tmp_path):
    out = _copy(built[0], tmp_path)
    p = out / "awr_weights_phase_rank.json"
    doc = json.loads(p.read_text())
    doc["provenance"]["corpus_fingerprint"] = "sha256:" + "0" * 32
    p.write_text(json.dumps(doc))
    _gate_fails(out, "G3")


def test_g4_catches_a_dropped_tag_on_a_trainable_row(built, tmp_path):
    out = _copy(built[0], tmp_path)
    p = out / "quality_phase.npz"
    z = dict(np.load(p))
    tag = z["tag"].copy()
    tag[int(np.flatnonzero(tag != 255)[0])] = 0
    z["tag"] = tag
    np.savez(p, **z)
    _gate_fails(out, "G4")


def test_the_tag_artifact_names_a_rename_stable_builder(built):
    out, _ = built
    meta = json.loads(str(np.load(out / "quality_phase.npz")["meta"]))
    assert meta["builder"] == "qvla.realrobot.realign.tags/1" and meta["pure_conditioning"] is True


def test_a_few_episodes_build_with_an_explicit_fit_pool(tmp_path):
    import shutil
    export_dir, amap, prompts = _export_curves(tmp_path)
    doc = json.loads(amap.read_text())
    doc["episodes"] = {k: v for k, v in doc["episodes"].items() if k.startswith("task_800--") and int(k.split("--")[1]) < 6}
    amap.write_text(json.dumps(doc))
    kept = {v["export_episode_index"] for v in doc["episodes"].values()}
    z = dict(np.load(export_dir / "task_800.npz"))
    for k in list(z):
        if k.startswith(("phase_ep_", "t0_ep_")) and int(k.rsplit("_", 1)[1]) not in kept:
            del z[k]
    shutil.rmtree(export_dir)
    export_dir.mkdir()
    np.savez(export_dir / "task_800.npz", **z)
    with pytest.raises(Exception, match="pools too small"):
        build(tmp_path / "o1", export_curves=export_dir, alignment_map=amap, prompts=prompts)
    summary = build(tmp_path / "o2", export_curves=export_dir, alignment_map=amap, prompts=prompts, min_fit_pool=100)
    assert summary["episodes"] == 6 and summary["phase_balance"]["min_fit_pool"] == 100


def _dumped_roots_fixture(tmp_path, tid=700, n_ep=6, T=180, seed=11):
    """A rendered corpus (real `reward_valid` data parquet, `meta/episodes`, `meta/tasks`) plus an AWR
    `--dump-curves`-style store keyed directly by the TRAINING episode_index -- no export re-keying,
    unlike the export_dir route. This is the actual production shape of
    `--roots ROOTS --curves-store STORE --valid-from-data`: a render's own corpus plus the dump a
    signal-tooling run wrote over it. The last episode carries no contact onset and is not dumped at
    all, exactly as the pinned builder leaves an unscorable episode out of its dump.
    """
    import pandas as pd
    rng = np.random.default_rng(seed)
    base = tmp_path / "corpus"
    task_dir = base / f"task_{tid}"
    (task_dir / "data" / "chunk-000").mkdir(parents=True)
    (task_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    rows, ep_meta, dump = [], [], {}
    offset = 0
    for ep in range(n_ep):
        length = T + int(rng.integers(-20, 21))
        valid = np.ones(length, bool)
        valid[length - int(rng.integers(0, 4)):] = False
        rows += [(ep, f, bool(valid[f])) for f in range(length)]
        ep_meta.append((ep, offset, length, 9000 + ep))
        if ep != n_ep - 1:  # every episode but the last carries an onset and is dumped
            t0 = int(rng.integers(length // 4, 3 * length // 4))
            approach = np.linspace(0.0, 0.3, t0) + rng.normal(0, 0.004, t0)
            steps = rng.gamma(0.6, 1.0, length - t0)
            manip = 0.3 + rng.uniform(0.2, 0.7) * np.cumsum(steps) / steps.sum()
            dump[f"ep_{ep}"] = np.clip(np.r_[approach, manip], 0, 1).astype(np.float32)
            dump[f"t0_{ep}"] = np.int64(t0)
        offset += length
    pd.DataFrame(rows, columns=["episode_index", "frame_index", "reward_valid"]).to_parquet(
        task_dir / "data" / "chunk-000" / "file-000.parquet")
    pd.DataFrame(ep_meta, columns=["episode_index", "dataset_from_index", "length", "attempt_id"]
                ).to_parquet(task_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    pd.DataFrame({"task": [f"do task {tid}"]}).to_parquet(task_dir / "meta" / "tasks.parquet")
    roots = tmp_path / "roots.json"
    roots.write_text(json.dumps({"version": 2, "base": "corpus", "roots": {str(tid): f"task_{tid}"}}))
    store = tmp_path / "dump"
    store.mkdir()
    np.savez(store / f"task_{tid}_phase_curves.npz", **dump)
    return roots, store, tid, n_ep


def test_valid_from_data_builds_end_to_end_through_the_cli_and_refuses_a_default_pool(tmp_path, capsys):
    """The one route the rest of this suite never drives through the real CLI entry point: a
    rendered corpus's own `reward_valid` plus an AWR `--dump-curves` store, through
    `cli.main(argv)`, with both flags (`--valid-from-data`, `--min-fit-pool`).

    The SAME dump store, without `--valid-from-data`, goes through `read_curve_store` instead --
    which refuses it (no `valid_<i>` keys) -- so this one fixture exercises BOTH sides of the
    `read_dumped_curve_store(...) if valid_from_data else read_curve_store(...)` branch in
    `cli.build`: swap it either way and one of the two assertions below flips.
    """
    roots, store, tid, n_ep = _dumped_roots_fixture(tmp_path)

    # Branch 1: --valid-from-data NOT given -- the masked reader refuses this dump-only store
    # (a named refusal: exit 2 and a FAIL line, never a traceback).
    assert main(["build", "--out", str(tmp_path / "o0"), "--roots", str(roots), "--curves-store", str(store)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("FAIL  ") and "has no valid_0" in err

    # Branch 2: --valid-from-data given, but the default --min-fit-pool (10,000) is far too large
    # for 6 episodes -- the same clear refusal, reached through argparse.
    assert main(["build", "--out", str(tmp_path / "o1"), "--roots", str(roots), "--curves-store", str(store),
                 "--valid-from-data"]) == 2
    assert "pools too small" in capsys.readouterr().err

    # Branch 2, lowered: both flags together, through the real argparse entry point.
    out = tmp_path / "o2"
    assert main(["build", "--out", str(out), "--roots", str(roots), "--curves-store", str(store),
                 "--valid-from-data", "--min-fit-pool", "100"]) == 0

    gates = json.loads((out / "gates.json").read_text())
    assert all(g["ok"] for g in gates.values()), gates
    scales = json.loads((out / "awr_weights_phase.json").read_text())["provenance"]["phase_balance"]
    assert scales["min_fit_pool"] == 100

    # The written curve store proves the masks and the neutral episode came from the CORPUS, not
    # from the (mask-free) dump: this is `read_dumped_curve_store`'s own contract, now checked at
    # the end of a real CLI build rather than only in isolation.
    from axis_qvla.realrobot.realign.curves import read_curve_store
    import pandas as pd
    written = read_curve_store(out / "curves")
    assert not written[(tid, n_ep - 1)].valid.any() and written[(tid, n_ep - 1)].t0 is None
    scored_ep = n_ep - 2
    corpus_valid = pd.read_parquet(tmp_path / "corpus" / f"task_{tid}" / "data" / "chunk-000" / "file-000.parquet")
    corpus_valid = corpus_valid[corpus_valid["episode_index"] == scored_ep].sort_values(
        "frame_index")["reward_valid"].to_numpy(bool)
    np.testing.assert_array_equal(written[(tid, scored_ep)].valid, corpus_valid)


@pytest.mark.parametrize("cmd, flag", [("gates", "--out"), ("compile", "--built")])
def test_gates_and_compile_refuse_a_directory_that_is_not_a_build(tmp_path, capsys, cmd, flag):
    """A mistyped directory is a named refusal (exit 2), never a FileNotFoundError traceback."""
    def run(d):
        argv = [cmd, flag, str(d)] + (["--out", str(tmp_path / "bundles")] if cmd == "compile" else [])
        return main(argv), capsys.readouterr().err

    code, err = run(tmp_path / "nowhere")
    assert code == 2 and f"{flag} {tmp_path / 'nowhere'}: no runs.json" in err and "realign build --out" in err
    (tmp_path / "half").mkdir()
    (tmp_path / "half" / "roots.json").write_text("{}")
    code, err = run(tmp_path / "half")
    assert code == 2 and f"{tmp_path / 'half'}: no runs.json" in err
    (tmp_path / "half" / "runs.json").write_text(json.dumps({"runs": ["bc"]}))
    code, err = run(tmp_path / "half")
    assert code == 2 and "is not a `realign build --out` directory" in err and "'ranges.json'" in err
    assert "'roots.json'" not in err and not (tmp_path / "bundles").exists()


def test_the_gates_command_exits_1_naming_the_failed_gate(built, tmp_path, capsys):
    out = _copy(built[0], tmp_path)
    assert main(["gates", "--out", str(out)]) == 0
    assert json.loads(capsys.readouterr().out) == {"G1": True, "G2": True, "G3": True, "G4": True}
    p = out / "awr_weights_phase.json"
    doc = json.loads(p.read_text())
    del doc["weights"]["task_912--3"]
    p.write_text(json.dumps(doc))
    assert main(["gates", "--out", str(out)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("FAIL  gates failed: G3") and str(out / "gates.json") in err


# --- what produces the training signals refuses a build the gates reject --------------------------

def _compile(out, tmp_path, capsys):
    code = main(["compile", "--built", str(out), "--out", str(tmp_path / "bundles")])
    return code, capsys.readouterr().err


def test_compile_refuses_a_build_whose_coverage_gate_fails_and_writes_nothing(built, tmp_path, capsys):
    out = _copy(built[0], tmp_path)
    f = out / "curves" / "task_800_phase_curves.npz"
    z = dict(np.load(f))
    z["ep_0"] = z["ep_0"].copy()
    z["ep_0"][int(np.flatnonzero(z["valid_0"])[0])] = 1.7          # one valid reward outside [0, 1]
    np.savez_compressed(f, **z)
    code, err = _compile(out, tmp_path, capsys)
    assert code == 2 and err.startswith("FAIL  ") and "G2" in err and "outside [0, 1]" in err
    assert not (tmp_path / "bundles").exists()


def test_compile_refuses_a_build_whose_balanced_weights_were_swapped_and_writes_nothing(built, tmp_path, capsys):
    out = _copy(built[0], tmp_path)
    (out / "awr_weights_phase.json").write_text((out / "awr_weights_phase_rank.json").read_text())
    code, err = _compile(out, tmp_path, capsys)
    assert code == 2 and "G3" in err and "phase_balanced is not True" in err
    assert not (tmp_path / "bundles").exists()


def test_build_reports_a_failed_gate_as_a_fail_line(tmp_path, capsys):
    export_dir, amap, prompts = _export_curves(tmp_path)
    f = export_dir / "task_800.npz"
    z = dict(np.load(f))
    name = next(k for k in sorted(z) if k.startswith("phase_ep_") and z[k][1].any())
    z[name] = z[name].copy()
    z[name][0, int(np.flatnonzero(z[name][1])[0])] = 1.7
    np.savez(f, **z)
    argv = ["build", "--out", str(tmp_path / "out"), "--export-curves", str(export_dir), "--alignment-map", str(amap),
            "--prompts", str(prompts)]
    assert main(argv) == 1
    err = capsys.readouterr().err
    assert err.startswith("FAIL  gates failed: G2") and str(tmp_path / "out" / "gates.json") in err


def test_a_flag_combination_build_cannot_use_is_refused_and_creates_nothing(tmp_path, capsys):
    assert main(["build", "--out", str(tmp_path / "out"), "--export-curves", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("FAIL  ") and "--export-curves needs --alignment-map and --prompts" in err
    export_dir, amap, prompts = _export_curves(tmp_path)
    assert main(["build", "--out", str(tmp_path / "out"), "--export-curves", str(export_dir), "--alignment-map", str(amap),
                 "--prompts", str(prompts), "--valid-from-data"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("FAIL  ") and "--valid-from-data reads a --curves-store" in err
    assert not (tmp_path / "out").exists()


def test_a_missing_input_of_build_is_a_named_refusal(tmp_path, capsys):
    argv = ["build", "--out", str(tmp_path / "out"), "--roots", str(tmp_path / "no_roots.json"),
            "--curves-store", str(tmp_path / "no_curves")]
    assert main(argv) == 2
    err = capsys.readouterr().err
    assert err.startswith("FAIL  ") and "no_curves" in err
    assert not (tmp_path / "out").exists(), "a missing input must be refused before --out is created"


def test_a_malformed_input_of_build_is_also_refused_before_out_is_created(tmp_path, capsys):
    """Not just an absent path: a `--roots` file that exists but is not valid JSON is read (and so
    refused) before `--out` is created too, the same as a missing one."""
    export_dir, amap, prompts = _export_curves(tmp_path)
    curves_store = tmp_path / "store"
    from axis_qvla.realrobot.realign.curves import load_export_curves, write_curve_store
    write_curve_store(curves_store, load_export_curves(export_dir, amap))
    roots = tmp_path / "roots.json"
    roots.write_text("not json")
    argv = ["build", "--out", str(tmp_path / "out"), "--roots", str(roots), "--curves-store", str(curves_store)]
    assert main(argv) == 2
    err = capsys.readouterr().err
    assert err.startswith("FAIL  ")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("cmd", ["build", "gates", "compile"])
def test_every_command_refuses_a_set_corpus_root_override(tmp_path, monkeypatch, capsys, cmd):
    """The override redirects the roots index's base in some readers and not others, so the gates or
    the compiled labels would describe two corpora."""
    monkeypatch.setenv("AXIS_PRETRAIN_CORPUS_ROOT", str(tmp_path))
    argv = {"build": ["build", "--out", str(tmp_path / "o"), "--roots", "r", "--curves-store", "c"],
            "gates": ["gates", "--out", str(tmp_path / "o")],
            "compile": ["compile", "--built", str(tmp_path / "o"), "--out", str(tmp_path / "b")]}[cmd]
    assert main(argv) == 2
    assert "AXIS_PRETRAIN_CORPUS_ROOT is set" in capsys.readouterr().err
    assert not (tmp_path / "o").exists() and not (tmp_path / "b").exists()


def test_a_moved_build_whose_roots_base_does_not_resolve_is_named(tmp_path, capsys):
    """A build over --roots keeps its corpus by a path relative to the build; moved alone, it is
    refused naming that base rather than failing inside the gates."""
    from axis_qvla.realrobot.realign.cli import BUILT
    out = tmp_path / "moved"
    out.mkdir()
    for name in BUILT:
        (out / name).write_text("{}")
    (out / "runs.json").write_text(json.dumps({"runs": ["bc", "conditioning", "awr", "filtering"]}))
    (out / "roots.json").write_text(json.dumps({"version": 2, "base": "../R", "roots": {}}))
    assert main(["gates", "--out", str(out)]) == 2
    err = capsys.readouterr().err
    assert "../R" in err and "does not resolve" in err
