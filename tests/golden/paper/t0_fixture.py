"""A two-cell results pack (2 tasks x 3 seeds) for the T0 tests.

Each cell names, beside its rollout tar, a `qvla.results/v1` file, and the first cell a run
manifest; the second names none, as a cell whose run published no run record (T0 verifies every
reference a cell names). The builder is
deterministic: tests/golden/paper/fixture_pack is exactly what it builds, plus the expected/ T0
writes for it (test_t0_pinning pins the two together).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from axis_qvla.core.canon import dumps_canonical
from axis_qvla.bench.results.fingerprint import eval_fingerprint, units_sha256
from axis_qvla.bench.results.format import ResultsHeader, ServeCondition, dumps_results, results_from_run
from axis_qvla.bench.results.seedfiles import ServedRun, read_served_dir, write_served_tar
from tests.unit.bench.results.fixtures import make_bench, make_served

TASKS = [806, 1050]
N_SEEDS = 3
CELLS = [("bc|bare", "r1fbc", {806: [True, False, True], 1050: [False, False, True]}),
         ("cfg_ph_ep|cond", "r1fcfg", {806: [True, True, True], 1050: [False, True, True]})]
WITH_MANIFEST = {"r1fbc"}   # r1fcfg names no run manifest


def cells_doc(entries: list[dict], decl_sha: str = "0" * 64) -> dict:
    """A cell map over `entries`; `decl_sha` is the fixture declaration's digest (validation-only
    callers such as tests/unit/paper/test_cells.py may leave the placeholder)."""
    return {"schema": "qvla.cells/1", "release": "v1.0",
            "bench": {"name": "fixture_bench", "r1_declaration_sha256": decl_sha, "task_ids": TASKS,
                      "n_seeds": N_SEEDS, "units_sha256": units_sha256(TASKS, N_SEEDS)},
            "cells": entries}


def _file(pack: Path, rel: str, data: bytes) -> dict:
    """Write `data` at pack/rel and return its cells.json reference (path + sha256, no size, as
    in the paper's map)."""
    p = pack / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return {"path": rel, "sha256": hashlib.sha256(data).hexdigest()}


def _results(sr: ServedRun, run: str, ckpt_ref: str, fp: str, decl_sha: str) -> bytes:
    header = ResultsHeader(
        bench={"name": "fixture_bench", "declaration_sha256": decl_sha, "units_sha256": units_sha256(TASKS, N_SEEDS)},
        arm=run, served=sr.served, training_seed={"pretrain": 42, "ft": 42},
        checkpoint={"ref": ckpt_ref, "pub_digest": None},
        serve=ServeCondition(fields={}, uncond_fields=None, beta=1.0),
        execution={"replan": 8, "max_steps": 1500, "fps": 15}, image_geometry="raw", action_space="droid8",
        harness=None, eval_lock_sha256=None, gpu=None, xla_pin="--xla_gpu_autotune_level=0",
        noise_salt=None, eval_fingerprint=fp)
    return dumps_results(results_from_run(sr, header, task_ids=TASKS))


def build_pack(tmp: Path) -> tuple[Path, Path, Path]:
    """-> (bench conf dir, pack dir, committed cells.json path). The pack has no expected/ yet."""
    conf = make_bench(tmp / "conf", TASKS, N_SEEDS)
    decl_sha = hashlib.sha256((conf / "fixture_bench.json").read_bytes()).hexdigest()
    pack = tmp / "pack"
    (pack / "rollouts" / "s2").mkdir(parents=True)
    entries = []
    for cell, run, outcomes in CELLS:
        served = f"{run}_dualsim"
        make_served(tmp / "logs", served, outcomes)
        sr = read_served_dir(tmp / "logs", served)
        tar = pack / "rollouts" / "s2" / f"{served}.tar"
        sha = write_served_tar(sr, tar)
        fp = eval_fingerprint(sr, TASKS)
        ckpt_ref = f"axisrobotics/qvla-v1-pi05@v1.0:ft/{run}/13579"
        manifest = dumps_canonical({"arm": run, "units_sha256": units_sha256(TASKS, N_SEEDS)})
        entry = {"cell": cell, "run": run, "box": "s2", "served": served, "backbone": "pi05",
                "training_seed": {"pretrain": 42, "ft": 42},   # Table I's pi0.5 seed-42 replicate
                "serve": {"fields": {}, "uncond_fields": None, "beta": 1.0},
                "eval_fingerprint": fp,
                "rollouts": {"path": f"rollouts/s2/{served}.tar", "sha256": sha, "size": tar.stat().st_size},
                "results": _file(pack, f"results/s2/{served}.json", _results(sr, run, ckpt_ref, fp, decl_sha)),
                "checkpoint": {"id": f"pi05/ft/{run}", "ref": ckpt_ref},
                "eval_command": f"qvla bench eval --run {run}",
                "lineage": {"train_config": None, "train_overrides": [], "step": "13579",
                            "norm_stats_sha256": None, "base": None, "init": None,
                            "arm": f"pi05.finetune.{run}"}}
        if run in WITH_MANIFEST:
            entry["manifest"] = _file(pack, f"runs/s2/{run}/run_manifest.json", manifest)
        entries.append(entry)
    data = dumps_canonical(cells_doc(entries, decl_sha))
    (pack / "cells.json").write_bytes(data)
    ref = tmp / "paper_fixture_cells.json"
    ref.write_bytes(data)
    return conf, pack, ref


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()
