"""Tiny served-run trees, benchmarks, headers and results files for the results and paper tests
(2 tasks x 3 seeds by default)."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from axis_qvla.bench.results.fingerprint import eval_fingerprint, units_sha256
from axis_qvla.bench.results.format import Results, ResultsHeader, ResultsRow, ServeCondition
from axis_qvla.bench.results.seedfiles import read_served_dir
from axis_qvla.bench.spec import Benchmark, load_benchmark

MD = {"supports_client_noise": True, "action_horizon": 10, "action_dim": 32}
DEV = {"dev_seeds": {"offset": 500, "n": 3}}
EXECUTION = {"schema": "qvla.execution/1", "fps": 15, "replan": 8, "max_steps": 1500, "n_stable_frames": 45,
             "noise_salt": 1096304979, "render": {"width": 640, "height": 360}, "grip_state": "closedness",
             "eef_ik": "single",
             "table_texture": None}
TASKS = [806, 1050]


def seed_bytes(seed: int, success: bool, n_steps: int = 100) -> bytes:
    rec = {"seed": seed, "success": success, "n_steps": n_steps, "sim_time": n_steps / 15,
           "reason": "success" if success else "timeout", "guidance_scale": 0.0,
           "cfg_conditioning": False, "eval_layouts": True, "base_variant_index": 0}
    return json.dumps(rec, indent=2).encode()


def make_served(log_root: Path, served: str, outcomes: dict[int, list[bool]]) -> Path:
    """<log_root>/<served>/task_<tid>/seed_<tid*1000+i>.json plus a meta.json and a .diag.json."""
    for tid, results in outcomes.items():
        d = log_root / served / f"task_{tid}"
        d.mkdir(parents=True, exist_ok=True)
        for i, ok in enumerate(results):
            seed = tid * 1000 + i
            (d / f"seed_{seed}.json").write_bytes(seed_bytes(seed, ok, 100 + i))
        (d / "meta.json").write_text(json.dumps({"replan": 8, "max_steps": 1500, "fps": 15}))
        (d / f"seed_{tid * 1000}.diag.json").write_text("{}")
    return log_root / served


def make_bench(conf_dir: Path, task_ids: list[int], n_seeds: int, *, name: str = "fixture_bench",
               extra: dict | None = None) -> Path:
    """A self-consistent benchmark declaration (three files) for `load_benchmark(conf_dir=...)`."""
    conf_dir.mkdir(parents=True, exist_ok=True)
    decl = {"name": name, "version": "fixture-v1", "n_seeds": n_seeds, "dual_sim": True,
            "action_space": "droid8", "image_geometry": "raw", "xla_pin": "--xla_gpu_autotune_level=0",
            "tasks_file": f"{name}.tasks.json", "layouts_file": f"{name}.layouts.json",
            "task_ids": task_ids,
            "subsets": {"informative_20": task_ids, "usable_12": task_ids[:1]}, **(extra or {})}
    tasks = {"tasks": {str(t): {"task_name_canonical": f"task {t}", "task_name": None,
                                "test_seeds": [t * 1000 + i for i in range(n_seeds)],
                                "gripper": {"open_w": 0.08, "closed_w": 0.0}} for t in task_ids}}
    layouts = {"tasks": {str(t): {"train": [0]} for t in task_ids}}
    (conf_dir / f"{name}.json").write_text(json.dumps(decl, indent=2))
    (conf_dir / f"{name}.tasks.json").write_text(json.dumps(tasks, indent=2))
    (conf_dir / f"{name}.layouts.json").write_text(json.dumps(layouts, indent=2))
    return conf_dir


def header(fp: str, *, served: str = "armx_dualsim", seed: int | None = 42, digest: str = "d" * 64,
           pretrain: int | None = None) -> ResultsHeader:
    """A header for the fixture benchmark of TASKS x 3 seeds (eval fingerprint `fp`)."""
    return ResultsHeader(
        bench={"name": "fixture_bench", "declaration_sha256": "0" * 64, "units_sha256": units_sha256(TASKS, 3)},
        arm=served.removesuffix("_dualsim"), served=served,
        training_seed=None if seed is None else {"pretrain": pretrain, "ft": seed},
        checkpoint={"ref": "axisrobotics/qvla-v1-pi05@v1.0:ft/x/13579", "pub_digest": digest},
        serve=ServeCondition(fields={}, uncond_fields=None, beta=1.0),
        execution={"replan": 8, "max_steps": 1500, "fps": 15}, image_geometry="raw", action_space="droid8",
        harness=None, eval_lock_sha256=None, gpu=None, xla_pin="--xla_gpu_autotune_level=0",
        noise_salt=1096304979, eval_fingerprint=fp)


def make_results(outcomes: dict[int, list[bool]], *, seed: int | None = 42, digest: str = "d" * 64,
                 arm: str = "armx_dualsim", pretrain: int | None = None, bench: Benchmark | None = None) -> Results:
    """A results file of `outcomes` (task -> success per seed, seeds tid*1000 + i); with `bench`, its
    header names that benchmark by identity."""
    rows = tuple(ResultsRow(t, t * 1000 + i, ok, 100, 6.6, "success" if ok else "timeout")
                 for t in sorted(outcomes) for i, ok in enumerate(outcomes[t]))
    h = header("f" * 64, served=arm, seed=seed, digest=digest, pretrain=pretrain)
    if bench is not None:
        # A valid results file names the benchmark by identity: name, work-unit list digest AND
        # the declaration's own content hash, each checked.
        h = dataclasses.replace(h, bench={"name": bench.name, "declaration_sha256": bench.sha256,
                                          "units_sha256": units_sha256(bench.task_ids, bench.n_seeds)})
    return Results(h, rows)


def make_dev_results(bench: Benchmark, outcomes: dict[int, list[bool]]) -> Results:
    """`make_results` read on the development seeds (tid*1000 + offset + i), its header saying so."""
    off = int(bench.dev["offset"])
    r = make_results(outcomes, bench=bench)
    rows = tuple(ResultsRow(t, t * 1000 + off + i, ok, 100, 6.6, "success" if ok else "timeout")
                 for t in sorted(outcomes) for i, ok in enumerate(outcomes[t]))
    return Results(dataclasses.replace(r.header, bench={
        **r.header.bench, "seed_set": "dev", "units_sha256": units_sha256(bench.task_ids, bench.n_seeds, offset=off)}),
        rows)


def make_run(tmp_path: Path, *, md=MD, meta_extra=None, errored=0, served="mine_dualsim", arm="mine",
             seed_set="test", extra=None, summary_extra=None, provenance_extra=None, execution_json=False):
    """A finished evaluation on disk: a 2-task x 3-seed served run under <tmp_path>/logs and the
    arm's arm_summary.json under <tmp_path>/r1/out_<arm>, as the driver leaves them; returns the
    benchmark's conf dir and the benchmark. Shared by the results-build and `bench eval` tests."""
    conf = make_bench(tmp_path / "conf", [806, 1050], 3, extra={**DEV, **(extra or {})})
    if execution_json:
        (conf / "execution.json").write_text(json.dumps(EXECUTION))
    bench = load_benchmark("fixture_bench", conf_dir=conf)
    logs = tmp_path / "logs"
    make_served(logs, served, {806: [True, False, True], 1050: [False, False, True]})
    if seed_set == "dev":   # the same outcomes on the development seeds tid*1000 + 500 + i
        for t in (806, 1050):
            d = logs / served / f"task_{t}"
            for i in range(3):
                (d / f"seed_{t * 1000 + i}.json").rename(d / f"seed_{t * 1000 + 500 + i}.json")
                rec = json.loads((d / f"seed_{t * 1000 + 500 + i}.json").read_text())
                (d / f"seed_{t * 1000 + 500 + i}.json").write_bytes(
                    seed_bytes(t * 1000 + 500 + i, rec["success"], rec["n_steps"]))
    for t in (806, 1050):
        meta = {"task_id": t, "fps": 15.0, "replan": 8, "max_steps": 1500, "n_stable_frames": 45,
                "server_metadata": md, "domain": "dualsim", **((meta_extra or {}).get(t, {}))}
        (logs / served / f"task_{t}" / "meta.json").write_text(json.dumps(meta))
    fp = eval_fingerprint(read_served_dir(logs, served), bench.task_ids)
    out = tmp_path / "r1" / f"out_{arm}"
    out.mkdir(parents=True)
    (out / "arm_summary.json").write_text(json.dumps({
        "arm": arm, "served_arm": served, "log_root": str(logs), "checkpoint": "my-checkpoint",
        "xla_flags": "--xla_gpu_autotune_level=0", "eval_fingerprint": fp, "errored_rollouts": errored,
        "n_seeds": 3, "benchmark": {**bench.identity(), "matches_manifest": True}, "seed_set": seed_set,
        "provenance": {"checkpoint": {"sha256": "c" * 64}, "noise_salt": 1096304979,
                       "env_pins": {"AXIS_EVAL_IMAGE_GEOMETRY": "raw", "AXIS_EVAL_ACTION_SPACE": "droid8"},
                       "sources": {"repo": {"head": "a" * 40, "dirty": False},
                                   "files": {"axis_qvla/bench/sim/campaign.py": "d" * 64}},
                       **(provenance_extra or {})},
        **(summary_extra or {})}))
    return conf, bench
