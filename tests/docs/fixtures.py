"""Small synthetic inputs for the tutorials' `ci` blocks: `python -m tests.docs.fixtures <kind> <out>`.

Each kind writes, at <out>, the same shape of input a tutorial's full-size block reads from published
data. It is built by the fixture the unit tests of that tool already use, so a tutorial and those
tests exercise one input, and so it needs the `test` extra. The numbers mean nothing.

Core tier at import; each kind imports what it needs.
"""
from __future__ import annotations

import argparse
import json
import pathlib


def toy_scores(out: pathlib.Path) -> None:
    """A scores file over the toy row space of axis_qvla.signals.testing: the weights of the toy corpus
    the signal tests use (tests/unit/signals/toy_corpus.py)."""
    from tests.unit.signals.toy_corpus import WEIGHTS

    doc = {"version": 1, "reward_id": "toy", "tau": 10.0, "cap": 2.0, "horizon": 10,
           "n_episodes": len(WEIGHTS), "n_rows": sum(map(len, WEIGHTS.values())), "weights": WEIGHTS}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc))


def smoke_corpus(out: pathlib.Path) -> None:
    """The smoke fixture's input: corpus tasks 2013 and 2232, three episodes each, in the published
    per-task layout (the corpus the training smoke tests build, tests/unit/train/test_smoke.py)."""
    from tests.unit.train.test_smoke import _corpus

    _corpus(out)


def keep_top_smoke(out: pathlib.Path) -> None:
    """The example method's SmolVLA and X-VLA run specs at the smoke fixture's scale, at
    <out>/{smolvla,xvla}.pretrain.keep_top.toml: experiments/examples/runs/<name>.pretrain.keep_top.toml
    on the smoke track and the smoke fixture's frame index, naming the example method compiled over that
    frame index (keep 0.5, 20 x 8, seed 42, a fixed score per frame) into this machine's compiled labels.
    The smoke fixture (`qvla train smoke-fixture`) is built first, under the roots of ./qvla.toml."""
    import dataclasses

    import numpy as np

    from axis_qvla.data.userconf import user_paths
    from axis_qvla.signals.bundle import load_rowspace, write_bundle
    from axis_qvla.signals.compile import compile_program
    from axis_qvla.signals.imported import ImportedScores
    from axis_qvla.signals.rowspace import episode_keys
    from axis_qvla.signals.schema import Budget
    from axis_qvla.train import ids
    from axis_qvla.train.armspec import dumps_armspec, load_armspec
    from axis_qvla.train.smoke import BATCH, SMOKE_DIR, STEPS
    from experiments.examples.keep_top import keep_top

    paths = user_paths(None)
    smoke = load_armspec(ids.local_derived(paths, SMOKE_DIR) / "runs" / "smolvla.toml")
    rs = load_rowspace(ids.rowspace_record_dirs(paths, smoke.rowspace)[1])
    rng = np.random.default_rng(0)
    scores = {k: rng.uniform(size=int(n)).tolist() for k, n in zip(episode_keys(rs), rs.arrays["episode_length"])}
    bundle, _ = compile_program(keep_top(0.5), ImportedScores({"weights": scores}, "", "keep-top-smoke"), rs,
                                Budget(STEPS, BATCH), 42)
    bid = write_bundle(bundle, ids.local_bundles_root(paths))
    out.mkdir(parents=True)
    for name in ("smolvla", "xvla"):
        spec = load_armspec(pathlib.Path("experiments/examples/runs") / f"{name}.pretrain.keep_top.toml")
        over = {k: spec.overrides[k] for k in ("axis_expected_mode", "axis_expected_reward")}
        spec = dataclasses.replace(spec, track="smoke", preset=f"{name}.pretrain.smoke", rowspace=smoke.rowspace,
                                   bundle=f"sha256:{bid}", overrides={**over, "axis_expected_frames": rs.n_rows})
        (out / f"{name}.pretrain.keep_top.toml").write_text(dumps_armspec(spec))


def traces_sample(out: pathlib.Path) -> None:
    """A traces sample of one task and two episodes, laid out like qvla-v1-corpus/traces_sample/: the
    synthetic trace and pnp goal leaf the reward tests use (tests/unit/labels/test_reward.py)."""
    from tests.unit.labels.test_reward import LEAF, write_trace

    (out / "traces").mkdir(parents=True)
    (out / "goals_per_attempt").mkdir()
    episodes = []
    for index, (attempt, offset) in enumerate(((2, 0.05), (3, 0.12))):
        write_trace(out / "traces" / f"task_1_attempt_{attempt}.jsonl", offset=offset)
        goals = {"n_steps": 25, "top_first_pass_frame": 19, "leaves": [LEAF], "attempt_id": attempt}
        (out / "goals_per_attempt" / f"task_1_attempt_{attempt}.json").write_text(json.dumps(goals))
        episodes.append({"attempt_id": attempt, "episode_index": index, "length": 80})
    (out / "manifest.json").write_text(json.dumps({"schema": "qvla.traces_sample/1", "tasks": [
        {"task_id": 1, "family": "put", "prompt": "put the cup on the plate", "episodes": episodes}]}))


def fr3_recording(out: pathlib.Path) -> None:
    """A synthetic FR3 recording (LeRobot v2.1, three episodes) at <out>/rec_250: the dataset the
    real-data tool tests use (tests/unit/realrobot/realdata/test_export.py)."""
    from tests.unit.realrobot.realdata.test_export import _world

    out.mkdir(parents=True)
    _world(out)


def realign_export_curves(out: pathlib.Path) -> None:
    """Phase curves of three tasks, keyed by a shuffled export index, with their alignment
    map and prompts, at <out>/{export_dir,map.json,prompts.json} (tests/unit/realrobot/realign/test_driver.py)."""
    from tests.unit.realrobot.realign.test_driver import _export_curves

    out.mkdir(parents=True)
    _export_curves(out)


def realign_corpus(out: pathlib.Path) -> None:
    """A rendered corpus of six episodes (its own `reward_valid`) and an AWR `--dump-curves` store over
    it, at <out>/{roots.json,corpus,dump} (tests/unit/realrobot/realign/test_driver.py)."""
    from tests.unit.realrobot.realign.test_driver import _dumped_roots_fixture

    out.mkdir(parents=True)
    _dumped_roots_fixture(out)


def render_source(out: pathlib.Path) -> None:
    """A one-object MuJoCo scene and a 12-row recorded trace of it, at <out>/{scene.xml,trace.jsonl}:
    the renderer tests' scene (tests/unit/realrobot/render/fixtures.py). Render tier."""
    from tests.unit.realrobot.render import fixtures as rf

    rf.write_trace(out / "trace.jsonl", rf.write_scene(out), n=12)


def attempt_curves(out: pathlib.Path) -> None:
    """Attempt-keyed phase curves for that trace's 36 render frames, as an attempt store. Render tier."""
    from axis_qvla.realrobot.render import labels
    from tests.unit.realrobot.render import fixtures as rf

    labels.write_attempt_store(out, rf.attempt_curves(36))


def realworld_data(out: pathlib.Path) -> None:
    """A six-episode copy of the published real-world data's layout over the renderer's fixture scene
    (tests/unit/realrobot/realworld_fixture.py): enough drawable rows for one batch of the compiled
    labels' row sampler, as `qvla realrobot labels` needs (tests/smoke/realrobot/test_steps_smoke.py)."""
    from tests.unit.realrobot.realworld_fixture import write_realworld

    write_realworld(out, attempts=(7001, 7002, 7003, 7004, 7005, 7006))


def workspace_photo(out: pathlib.Path) -> None:
    """A 1280 x 720 frame of the reference workspace from its exterior camera, standing in for a real
    camera's: the overlay check's input. Render tier."""
    import numpy as np

    from axis_qvla.realrobot.render import overlay, profile, rig
    from tests.unit.realrobot.render import fixtures as rf

    work = out.parent / f"{out.stem}_inputs"
    rgb, _ = overlay.render_view(rf.write_scene(work / "scene"), rf.write_gripper(work / "gripper"),
                                 profile.load_profile(), rig.PIXELMATCH_HOME)
    overlay.write_png(out, np.repeat(np.repeat(rgb, 2, axis=0), 2, axis=1))


def realworld_local(out: pathlib.Path) -> None:
    """The real-world tutorial's local layout in the cache root <out>: derived/realworld/{d8,built,demos},
    compiled labels in bundles/, norm statistics in derived/norm/ (tests/unit/realrobot/fr3/local_fixture.py)."""
    from tests.unit.realrobot.fr3.local_fixture import build_cache

    build_cache(out, out.parent / "droid_recording")


def droid_recording(out: pathlib.Path) -> None:
    """A three-episode DROID-layout recording of two instructions, as openpi's DROID converter writes it
    (tests/unit/realrobot/realdata/test_training_set.py)."""
    from tests.unit.realrobot.realdata.test_training_set import write_droid_recording

    write_droid_recording(out)


def droid_frames(out: pathlib.Path) -> None:
    """Two 1280 x 720 camera frames, <out>/{exterior,wrist}.png, as openpi's DROID example client captures
    them (tests/unit/realrobot/realdata/droid_raw.py)."""
    from PIL import Image

    from tests.unit.realrobot.realdata.droid_raw import raw_frame

    out.mkdir(parents=True)
    for camera, name in enumerate(("exterior", "wrist")):
        Image.fromarray(raw_frame(0, 3, camera)).save(out / f"{name}.png")


def fr3_server_report(out: pathlib.Path) -> None:
    """A `qvla bench check-server --report` file of a policy server of `pi05.finetune.realrobot_only`,
    standing in for your server's: the handshake the openpi line declares for that run (centre-cropped
    224 frames, the model's state order, DROID's joint-velocity command at 15 Hz, 15 actions, no
    quality tag)."""
    import json

    from tests.unit.realrobot.fr3.test_checks import _handshake

    out.write_text(json.dumps({"endpoint": "127.0.0.1:8000", "ok": False, "handshake": _handshake(prompt_suffix="")}))


def method_results(out: pathlib.Path) -> None:
    """The results files `qvla bench eval --from-run` writes for the example method and its control,
    at <out>/out_pi05.finetune.example.keep_top{,.control}/results.json, over the contrast tests'
    fixture pack (tests/golden/paper/fixture_pack: 2 tasks x 3 seeds), at pretraining seed 42."""
    from tests.golden.paper.fixtures import user_results

    for arm, outcomes in (("pi05.finetune.example.keep_top", None),
                          ("pi05.finetune.example.keep_top.control", {806: [True, True, False], 1050: [False, False, True]})):
        kw = {} if outcomes is None else {"outcomes": outcomes}
        user_results(out / f"out_{arm}" / "results.json", arm=arm, pretrain=42, ft=42, **kw)

KINDS = {"toy-scores": toy_scores, "smoke-corpus": smoke_corpus, "keep-top-smoke": keep_top_smoke,
         "traces-sample": traces_sample, "fr3-recording": fr3_recording, "realign-export-curves": realign_export_curves,
         "realign-corpus": realign_corpus, "render-source": render_source, "attempt-curves": attempt_curves,
         "realworld-data": realworld_data, "workspace-photo": workspace_photo, "realworld-local": realworld_local,
         "droid-recording": droid_recording, "droid-frames": droid_frames, "fr3-server-report": fr3_server_report,
         "method-results": method_results}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tests.docs.fixtures", description=__doc__.split("\n\n")[0])
    ap.add_argument("kind", choices=sorted(KINDS))
    ap.add_argument("out", type=pathlib.Path)
    a = ap.parse_args(argv)
    if a.out.exists():
        ap.error(f"{a.out} exists: a fixture is written fresh")
    KINDS[a.kind](a.out)
    print(f"OK    {a.kind} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
