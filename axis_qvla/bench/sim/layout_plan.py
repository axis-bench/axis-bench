"""Resolve every eval seed's FRESH layout up front, before a single rollout runs.

WHY PLAN AHEAD INSTEAD OF SAMPLING PER SEED
    `eval_layouts.generate_layout` can legitimately fail: a task whose homogeneous training
    set spans a single scene variant has no arrangement that is simultaneously novel and
    difficulty-matched to training (task 1628 is the measured case). Discovering that on seed 7
    of 20 leaves a task
    half-evaluated, and the tempting repair -- fall back to the shipped scene variant for
    the remaining seeds -- silently reintroduces the training/eval layout overlap the fresh
    layouts exist to remove, unevenly across arms.

    So resolve the whole seed set first and let `UnsupportedTaskError` out. The caller
    records the task as UNEVALUABLE and evaluates none of its seeds.

Offline (json/numpy only): imports cleanly under the offline environment and the eval environment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from axis_qvla.bench.sim.layouts import (
    LayoutSpec,
    UnsupportedTaskError,
    generate_layout,
    generate_layout_faithful,
)

# Process exit status for "this task has no fair eval layout". Distinct from 1 so a caller can
# tell an unevaluable task apart from a crashed rollout without parsing logs.
EXIT_UNEVALUABLE = 3


def adapt_layout_manifest(manifest: Path | str, out_dir: Path) -> Path:
    """`manifest`, reshaped for this module's own list-based "tasks" reader, when it is the
    benchmark's layouts file (`axis_qvla.bench.spec.Benchmark.layouts_path`, e.g.
    `franka_heldout24.layouts.json`), which keys "tasks" by id instead -- a no-op (the same path,
    unread) on an already list-shaped manifest, such as an adapted copy.

    CALL THIS ONCE, before `plan_layouts`, never inside it or inside `_is_single_variant`: the
    campaign driver applies it once per run, before any worker starts (`axis_qvla.bench.sim.campaign`,
    around its own `harness_layout` call), and the single-task CLI applies it before planning.
    `_is_single_variant` and the generators in `axis_qvla.bench.sim.layouts` read the list shape
    (`tests/unit/bench/sim/test_eval_queue_driver.py` pins the `AttributeError` a direct call on
    the keyed file raises): that is the queue worker's contract.

    A local import: `axis_qvla.bench.sim.campaign` pulls in the whole campaign driver (workqueue,
    subprocess handling) for one small function, and this module's own docstring promises
    `json`/`numpy` only.
    """
    from axis_qvla.bench.sim.campaign import harness_layout_manifest

    return harness_layout_manifest(Path(manifest), Path(out_dir))


def _is_single_variant(task_id: int, manifest: Path | str) -> bool:
    """True if this task's training set fixes exactly ONE scene_variant_index.

    Such a task is the homogeneous variant-selection build: its eval must reproduce
    training-time generation (same variant meshes + small jitter), not the novelty/parity
    sampler, which is unsatisfiable against a single variant.
    """
    man = json.loads(Path(manifest).read_text())
    task = next((t for t in man.get("tasks", []) if int(t.get("task_id", -1)) == int(task_id)), None)
    if task is None:
        return False
    return len({int(d["scene_variant_index"]) for d in task.get("train", []) if "scene_variant_index" in d}) == 1


def plan_layouts(
    task_id: int, seeds: Iterable[int], *, manifest: Path | str, task_data_root: Path | str,
) -> dict[int, LayoutSpec]:
    """`{seed: LayoutSpec}` for every seed, or raise.

    Auto-selects the generator by manifest shape: a single-variant (homogeneous) training set
    gets the variant-faithful in-distribution generator; a multi-variant set gets the
    novelty/difficulty-parity sampler. No try/except by design -- see the module docstring;
    `UnsupportedTaskError` must reach the caller. `task_data_root` is where task_<id>/task.json
    lives.
    """
    gen = generate_layout_faithful if _is_single_variant(task_id, manifest) else generate_layout
    return {
        int(s): gen(int(task_id), int(s), homogeneous_manifest=manifest,
                    task_data_root=task_data_root)
        for s in seeds
    }


def write_unevaluable(
    log_dir: Path,
    *,
    task_id: int,
    arm: str,
    manifest: Path | str,
    exc: UnsupportedTaskError,
    seeds: Iterable[int],
) -> Path:
    """Record the task as unevaluable so its absence from a scorecard is explained.

    A missing directory looks identical to a job that never ran; this file says which task,
    under which manifest, and with the sampler's own diagnosis of the binding constraint.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / "unevaluable.json"
    out.write_text(
        json.dumps(
            {
                "task_id": int(task_id),
                "arm": arm,
                "eval_layouts": True,
                "layout_manifest": str(manifest),
                "unevaluable": True,
                "n_seeds_requested": len(list(seeds)),
                "error_type": type(exc).__name__,
                "reason": " ".join(str(exc).split()),
            },
            indent=2,
        )
        + "\n"
    )
    return out
