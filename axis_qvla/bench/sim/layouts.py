"""Generate FRESH eval layouts inside the training layout distribution.

WHY GENERATE INSTEAD OF REUSING SHIPPED VARIANTS
    Two problems with picking eval scenes from a task's ~40 shipped scene variants:

    1. Memorization. A shipped scene variant can also appear in that task's training set,
       so a rollout on it could succeed by replaying a memorized layout. Success would then
       measure recall, not competence.
    2. Distribution drift. The variants do not share one object set, so an arbitrary
       variant may show object instances training never saw.

    Sampling layouts ourselves fixes both at once: we fix the task's homogeneous object
    set (so no drift) and resample object XY uniformly inside the SAME box the training
    demos span (so the same randomization range), while guaranteeing the exact layout is
    novel. That is in-distribution but unseen -- the condition under which a variant
    bake-off actually measures generalization.

WHAT IT DOES NOT DO
    It does not invent new objects, change the object set, or widen the range. Every
    sampled layout stays inside the measured training support; only the arrangement is new.

Offline (json/numpy). Run under the offline environment.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

# Objects are point-sampled; this is the minimum centre-to-centre XY gap enforced between
# any two objects so a sampled layout cannot spawn one object inside another. It is a
# conservative proxy for real mesh extents (we do not have per-object bounding boxes here),
# deliberately larger than typical object radii for these tabletop scenes.
MIN_OBJECT_SEPARATION_M = 0.12

# Rejection budget, NOT a constraint: raising it cannot admit a layout the constraints
# reject, it only stops us from giving up on a feasible one. 2000 was too tight -- measured
# across all 25 trainable tasks x 50 test seeds, the worst seed (task 1440) needs 2283
# tries, so 2000 silently dropped eval seeds on that task alone. 50k leaves >20x headroom
# and still costs ~2 s for a task's full seed set.
MAX_REJECTION_TRIES = 50_000

# A sampled layout must differ from every TRAINING layout by at least this much (target
# object, XY). Without it a sample can land on a memorized arrangement, which is the exact
# failure mode this module exists to avoid. Small enough to stay well inside the box.
MIN_NOVELTY_M = 0.05

# A target-pair distance interval narrower than this is treated as degenerate (see
# _degenerate_pairs): 1 mm is far below the layout randomization scale, so an interval that
# tight carries no difficulty information and cannot be hit by rejection sampling.
DEGENERATE_BOUND_M = 1e-3


def _variants(task_id: int, task_data_root: Path | str) -> list[dict]:
    """The task's scene variants, read from `task_data_root` (the directory holding task_<id>/)."""
    root = Path(task_data_root)
    return json.loads((root / f"task_{task_id}" / "task.json").read_text())["scene_variants"]


class UnsupportedTaskError(RuntimeError):
    """No eval layout exists for this task under the current (deliberately strict) rules.

    A distinct type so callers can separate "this task cannot be evaluated fairly" from a
    genuine bug in scene building. Subclasses RuntimeError so existing `except RuntimeError`
    handlers keep working.
    """


@dataclasses.dataclass(frozen=True)
class LayoutSpec:
    """A generated eval layout: the template variant plus resampled object XY."""

    task_id: int
    seed: int
    base_variant_index: int
    object_xy: dict[str, tuple[float, float]]

    def to_json(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "base_variant_index": self.base_variant_index,
            "object_xy": {k: list(v) for k, v in self.object_xy.items()},
        }


def _pair_bounds(
    support: dict[str, np.ndarray], names: list[str], target: str | None
) -> dict[tuple[str, str], tuple[float, float]]:
    """[min,max] XY distance observed across TRAINING layouts, for pairs involving TARGET.

    Only target-involving pairs are bounded. Constraining all pairs is over-determined --
    rejection sampling cannot satisfy 6 simultaneous interval constraints inside a 0.5 m
    box (measured: 0 valid samples in 2000 tries) -- and distractor-to-distractor spacing
    does not affect how hard the task is. What matters is that the target is never closer
    to another object than training ever placed it.
    """
    out: dict[tuple[str, str], tuple[float, float]] = {}
    if not target:
        return out
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if target not in (a, b):
                continue
            pa, pb = support.get(a), support.get(b)
            if pa is None or pb is None or len(pa) != len(pb):
                continue
            d = np.linalg.norm(pa - pb, axis=1)
            if len(d):
                out[(a, b)] = (float(d.min()), float(d.max()))
    return out


def _pairs_within(placed: dict[str, np.ndarray], bounds: dict[tuple[str, str], tuple[float, float]]) -> bool:
    for (a, b), (lo, hi) in bounds.items():
        if a in placed and b in placed:
            d = float(np.linalg.norm(placed[a] - placed[b]))
            if d < lo or d > hi:
                return False
    return True


def _degenerate_pairs(
    bounds: dict[tuple[str, str], tuple[float, float]]
) -> list[tuple[tuple[str, str], float]]:
    """Target pairs whose training distance range is a POINT, not an interval.

    Happens when a task's homogeneous training set covers a single scene variant: every
    demo then shares one layout, so `hi - lo == 0` and difficulty parity demands an exact
    distance match that continuous sampling can never produce. Detecting it up front
    matters because the generic "no valid layout after N tries" message blames
    min_separation / novelty, which would send someone to loosen the wrong knob -- and
    loosening those is precisely what makes eval easier than training.
    """
    return [(k, hi - lo) for k, (lo, hi) in bounds.items() if hi - lo < DEGENERATE_BOUND_M]


def _training_support(task_id: int, variants: list[dict], variant_indices: list[int]) -> dict[str, np.ndarray]:
    """Per-object [min,max] XY box observed across the task's TRAINING variants."""
    per: dict[str, list[np.ndarray]] = {}
    for i in variant_indices:
        objs = (variants[i].get("initial_state") or {}).get("objects") or {}
        for name, o in objs.items():
            pos = o.get("pos")
            if pos:
                per.setdefault(name, []).append(np.asarray(pos[:2], dtype=float))
    return {k: np.asarray(v) for k, v in per.items() if v}


def generate_layout(
    task_id: int,
    seed: int,
    *,
    homogeneous_manifest: Path | str,
    min_separation_m: float = MIN_OBJECT_SEPARATION_M,
    task_data_root: Path | str,
) -> LayoutSpec:
    """Sample one fresh layout for `task_id` inside its training support.

    Deterministic in `seed`. Raises if a valid (non-overlapping) arrangement cannot be
    found -- silently returning an overlapping layout would spawn interpenetrating objects
    and corrupt the rollout.
    """
    man = json.loads(Path(homogeneous_manifest).read_text())
    task = next((t for t in man["tasks"] if int(t["task_id"]) == int(task_id)), None)
    if task is None:
        raise KeyError(f"task {task_id} not in {homogeneous_manifest}")

    variants = _variants(task_id, task_data_root)
    train_vidx = sorted({int(d["scene_variant_index"]) for d in task["train"]})
    support = _training_support(task_id, variants, train_vidx)
    if not support:
        raise ValueError(f"task {task_id}: no object positions in training variants")

    rng = np.random.default_rng((int(task_id) << 20) ^ int(seed))
    base = train_vidx[int(seed) % len(train_vidx)]

    # Sample inside the task's DECLARED box, not the observed per-object min/max. Observed
    # support collapses to a point when a task's homogeneous set covers a single training
    # variant (measured: task 1628 has exactly 1), which would emit the same layout for
    # every seed -- and that layout is a training layout. The box is the allowed
    # randomization range and is always non-degenerate.
    center = np.asarray(task["layout_center_xy"], dtype=float)
    half = float(task["layout_box_m"]) / 2.0
    names = sorted(support)

    # Training layouts of the target object, to enforce novelty against.
    target = task.get("target_body")
    train_target_xy = support.get(target) if target else None
    pair_bounds = _pair_bounds(support, names, target)

    degenerate = _degenerate_pairs(pair_bounds)
    if degenerate:
        pairs = ", ".join(f"{a}-{b} (width {w*1e3:.2f} mm)" for (a, b), w in degenerate)
        raise UnsupportedTaskError(
            f"task {task_id}: difficulty parity is unsatisfiable -- the training layouts span "
            f"{len(next(iter(support.values())))} distinct arrangement(s) ({len(train_vidx)} scene "
            f"variant(s)), so the target-pair distance range collapses to a point: {pairs}. "
            "No novel layout can match it. Fix upstream (choose a homogeneous object set that "
            "covers >1 scene variant for this task), not by widening the bound here."
        )

    rejects = {"separation": 0, "novelty": 0, "pair_bounds": 0}
    for _ in range(MAX_REJECTION_TRIES):
        placed: dict[str, np.ndarray] = {}
        ok = True
        for n in names:
            p = rng.uniform(center - half, center + half)
            if any(np.linalg.norm(p - q) < min_separation_m for q in placed.values()):
                ok = False
                break
            placed[n] = p
        if not ok:
            rejects["separation"] += 1
            continue
        if train_target_xy is not None and target in placed:
            d = np.linalg.norm(train_target_xy - placed[target], axis=1).min()
            if d < MIN_NOVELTY_M:
                rejects["novelty"] += 1
                continue  # too close to a memorized arrangement
        # DIFFICULTY PARITY: keep every pairwise object distance within the range training
        # spanned. Uniform sampling in the box otherwise produces arrangements strictly
        # easier than any training layout -- measured on task 1424, 9/20 samples put the
        # target closer to its goal than the closest training layout (13.1 cm vs a 26.2 cm
        # training minimum). That shortens the task and would inflate success rates,
        # unequally across variants. Layouts must be novel, not easier.
        if pair_bounds and not _pairs_within(placed, pair_bounds):
            rejects["pair_bounds"] += 1
            continue
        return LayoutSpec(
            task_id=int(task_id),
            seed=int(seed),
            base_variant_index=int(base),
            object_xy={n: (float(p[0]), float(p[1])) for n, p in placed.items()},
        )
    raise UnsupportedTaskError(
        f"task {task_id} seed {seed}: no valid layout after {MAX_REJECTION_TRIES} tries in a "
        f"{2*half:.2f} m box with {len(names)} objects; rejected by "
        f"separation={rejects['separation']}, novelty={rejects['novelty']}, "
        f"pair_bounds={rejects['pair_bounds']}. The dominant cause names the binding "
        "constraint -- report the task as unsupported rather than loosening it."
    )


def generate_layout_faithful(
    task_id: int,
    seed: int,
    *,
    homogeneous_manifest: Path | str,
    jitter_m: float = 0.05,
    min_separation_m: float = MIN_OBJECT_SEPARATION_M,
    task_data_root: Path | str,
) -> LayoutSpec:
    """Eval layout for a HOMOGENEOUS single-variant training set: reproduce training-time
    generation instead of the novelty/difficulty-parity sampler.

    `generate_layout` deliberately samples layouts that are NOVEL and difficulty-matched vs a
    DIVERSE training set -- but a homogeneous manifest fixes ONE scene_variant_index per task, whose
    training scenes are that variant's fixed object arrangement plus <=5 cm per-object jitter.
    Against a single variant the novelty+parity constraints are unsatisfiable (the pairwise distance
    range collapses to a point -> UnsupportedTaskError), and worse, pointing `generate_layout` at a
    MULTI-variant manifest spawns the WRONG meshes. This generator instead spawns the SAME variant's
    meshes at the SAME nominal positions with fresh <=`jitter_m` jitter -- i.e. an in-distribution
    eval draw, which is the correct test for a homogeneous set. No novelty gate: same-distribution
    generalization (new jittered layouts, new visual randomization) IS what we are measuring.
    """
    man = json.loads(Path(homogeneous_manifest).read_text())
    task = next((t for t in man["tasks"] if int(t["task_id"]) == int(task_id)), None)
    if task is None:
        raise KeyError(f"task {task_id} not in {homogeneous_manifest}")
    vidx = sorted({int(d["scene_variant_index"]) for d in task["train"]})
    if len(vidx) != 1:
        raise ValueError(
            f"task {task_id}: generate_layout_faithful expects a single training scene "
            f"variant, got {vidx}. Use generate_layout for multi-variant training sets."
        )
    base = vidx[0]
    variants = _variants(task_id, task_data_root)
    objs = (variants[base].get("initial_state") or {}).get("objects") or {}
    nominal = {n: np.asarray(o["pos"][:2], dtype=float) for n, o in objs.items() if o.get("pos")}
    if not nominal:
        raise ValueError(f"task {task_id} variant {base}: no object positions")

    rng = np.random.default_rng((int(task_id) << 20) ^ int(seed))
    names = sorted(nominal)
    for _ in range(MAX_REJECTION_TRIES):
        placed = {n: nominal[n] + rng.uniform(-jitter_m, jitter_m, size=2) for n in names}
        pts = list(placed.values())
        if all(
            np.linalg.norm(pts[i] - pts[j]) >= min_separation_m
            for i in range(len(pts)) for j in range(i + 1, len(pts))
        ):
            return LayoutSpec(
                task_id=int(task_id),
                seed=int(seed),
                base_variant_index=int(base),
                object_xy={n: (float(p[0]), float(p[1])) for n, p in placed.items()},
            )
    raise UnsupportedTaskError(
        f"task {task_id} seed {seed}: jittered layout kept overlapping after "
        f"{MAX_REJECTION_TRIES} tries (jitter {jitter_m} m, {len(names)} objects). The nominal "
        f"variant-{base} arrangement may already violate the {min_separation_m} m separation."
    )


def apply_layout(initial_state: dict, layout: LayoutSpec) -> dict:
    """Return a copy of `initial_state` with object XY replaced by the sampled layout.

    Z is deliberately left untouched: the render scene raises objects onto the table by
    `free_joint_qpos_z_offset` afterwards, and resting height is object-specific.
    """
    import copy

    out = copy.deepcopy(initial_state or {})
    objs = out.get("objects") or {}
    for name, xy in layout.object_xy.items():
        o = objs.get(name)
        if not o or not o.get("pos"):
            continue
        pos = list(o["pos"])
        pos[0], pos[1] = float(xy[0]), float(xy[1])
        o["pos"] = pos
    return out


def scene_for_layout(payload: dict, layout, task_id: int) -> tuple[str | None, dict | None, int]:
    """The scene an eval layout names: its base variant's mjcf_xml, and that variant's initial_state
    with the layout applied (the task-level values when the variant lacks them). This is the whole
    layout branch of render_scene.build_render_scene, kept here because render_scene imports
    robosuite: a scene digest can hash THIS function in the offline venv, so it covers the code
    the render path runs."""
    variants = payload.get("scene_variants") or []
    if not 0 <= layout.base_variant_index < len(variants):
        raise ValueError(f"task {task_id}: layout base variant {layout.base_variant_index} "
                         f"is outside its {len(variants)} scene variants")
    base = variants[layout.base_variant_index]
    state = apply_layout(base.get("initial_state") or payload.get("initial_state"), layout)
    return base.get("mjcf_xml") or payload.get("mjcf_xml"), state, layout.base_variant_index
