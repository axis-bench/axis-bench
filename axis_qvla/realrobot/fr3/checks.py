"""Checks that keep the FR3 line's ArmSpecs, presets, norm statistics and serving consistent with its record.

Each returns a list of problems; empty means it holds.
- check_runs: the runs directory holds exactly one ArmSpec per run of line.toml; the ArmSpec loader
  accepts each; each is field for field the ArmSpec the record declares (axis_qvla.realrobot.fr3.line.run_spec);
  and it fits its recipe: same backbone, stage and track, overrides only of fields the recipe declares,
  no quality tag beside a named bundle (axis_qvla.train.recipes.resolved_fields,
  axis_qvla.train.generate.quality_tag_problem);
- check_presets: the three realrobot presets, read through axis_qvla.train.recipes.load_recipe, carry the
  record's budget, warmup, rates, action horizon, LoRA variants, freeze and image crop, fit the realrobot
  track's budget, state paper_budget = true (the record's budgets are the paper's runs'), name every model field the paper's LoRA finetune preset names (the
  openpi line refuses a config missing one), declare the `data.quality_tag` field the tagged finetune overrides,
  augment the images exactly when the record's stage does (`sim_image_aug`), and name their stage's data
  units (`data.action_units`, the stage embodiment's `action.units`);
- check_units: a finetune run initialised from a pretrain run crosses from the sim units to the FR3 units
  only when the record declares that transfer; each stage's embodiment has the record's units; the
  FR3 embodiment's units are the ones the recording declares (axis_qvla.realrobot.realdata.state_order.ACTION_UNITS:
  DROID's normalised joint-velocity command in [-1, 1] at 15 Hz, target q + 0.2 v, absolute gripper in [0, 1]);
  and the rendered view's fps and range are its producer's (axis_qvla.realrobot.render.rig: FPS, the +-1 rad/s
  VELOCITY_CLIP of the DROID-8D view);
- check_record: the record against itself and its embodiments: every finetune run's serve condition and
  init resolve, every pretrain run names a signal, each stage's image is its embodiment's frame geometry
  (or, for fr3, whose stored frames carry their geometry, the stage names the set it describes: the
  serve check measures that set's pixels), the FR3 embodiment's state and action layouts are the ones
  the recording declares, and the rendered view's layouts and frame size are the renderer's.

check_serve (a policy server's handshake, axis_qvla.realrobot.fr3.serve_check) and check_norm (a norm file,
axis_qvla.realrobot.fr3.norm_check) live in their own modules; the command line reads every check from here.

Offline tier (numpy, pyarrow).
"""
from __future__ import annotations

import os
from pathlib import Path

from axis_qvla.realrobot.fr3.embodiment import EmbodimentError, load_embodiment
from axis_qvla.realrobot.fr3.line import IMAGES, PRESETS, Line, kind, run_spec
from axis_qvla.realrobot.fr3.norm_check import check_norm
from axis_qvla.realrobot.fr3.producers import layout_problems, unit_problems
from axis_qvla.realrobot.fr3.serve_check import check_serve, training_geometry

__all__ = ["PRESETS", "IMAGES", "check_runs", "check_presets", "check_units", "check_record", "check_serve",
           "check_norm", "training_geometry"]

# The paper's LoRA finetune preset: its model.* keys are every Pi0Config field the openpi line requires.
PAPER_PRESET = "pi05.finetune.lora.paper"
_ARMSPEC_FIELDS = ("id", "backbone", "stage", "track", "paper_ref", "init", "rowspace", "bundle", "norm",
                   "preset", "seed")
_UNSET = object()


def _show(v: object) -> str:
    return "unset" if v is _UNSET else repr(v)


def _run_files(runs_dir: Path) -> dict[str, Path]:
    """{stem: path} of every .toml in `runs_dir`. os.scandir, so an unreadable directory raises
    instead of reading as an empty one."""
    with os.scandir(runs_dir) as it:
        return {e.name[:-5]: Path(e.path) for e in it if e.name.endswith(".toml") and e.is_file()}


def check_runs(line: Line, runs_dir: Path) -> list[str]:
    from axis_qvla.train.armspec import ArmSpecError, load_armspec
    from axis_qvla.train.generate import quality_tag_problem
    from axis_qvla.train.recipes import RecipeError, load_recipe, resolved_fields

    try:
        files = _run_files(Path(runs_dir))
    except OSError as exc:
        return [f"{runs_dir}: cannot list the runs directory ({exc.__class__.__name__}: {exc})"]
    problems = [f"{name}.toml is not in the line record" for name in sorted(set(files) - set(line.runs))]
    for run_id in sorted(line.runs):
        if run_id not in files:
            problems.append(f"{run_id}: no ArmSpec")
            continue
        try:
            a = load_armspec(files[run_id])
        except ArmSpecError as exc:
            problems.append(str(exc))
            continue
        want = run_spec(line, run_id)
        problems += [f"{run_id}: {f} {getattr(a, f)!r}, the record says {getattr(want, f)!r}"
                     for f in _ARMSPEC_FIELDS if getattr(a, f) != getattr(want, f)]
        for k in sorted(set(a.data) | set(want.data)):
            if a.data.get(k) != want.data.get(k):
                problems.append(f"{run_id}: data.{k} {a.data.get(k)!r}, the record says {want.data.get(k)!r}")
        for k in sorted(set(a.overrides) | set(want.overrides)):
            got, wanted = a.overrides.get(k, _UNSET), want.overrides.get(k, _UNSET)
            if got != wanted:
                problems.append(f"{run_id}: override {k} = {_show(got)}, the record says {_show(wanted)} "
                                f"(quality_tag = {str(line.runs[run_id].get('quality_tag', False)).lower()})")
        try:
            tag = quality_tag_problem(a, resolved_fields(load_recipe(a.preset), a))
        except RecipeError as exc:
            problems.append(str(exc))
            continue
        if tag:
            problems.append(f"{run_id}: {tag}")
    return problems


def check_presets(line: Line, recipes: Path | None = None) -> list[str]:
    """`recipes` is the recipe directory to read (default: conf/recipes)."""
    from axis_qvla.train.recipes import RECIPES_DIR, RecipeError, check_recipe_against_track, load_recipe, load_track

    root = RECIPES_DIR if recipes is None else Path(recipes)
    problems = []
    try:
        track = load_track("realrobot", root=root)
        model_keys = {k for k in load_recipe(PAPER_PRESET, root=root).fields if k.startswith("model.")}
    except RecipeError as exc:
        return [str(exc)]
    for k, stage in (("pretrain", line.stage1), ("finetune", line.stage2), ("finetune_only", line.stage2)):
        try:
            r = load_recipe(PRESETS[k], root=root)
            check_recipe_against_track(r, track)
        except RecipeError as exc:
            problems.append(str(exc))
            continue
        got = {"track": r.track, "paper_budget": r.paper_budget, "steps": r.steps, "global_batch": r.global_batch,
               "warmup": r.fields.get("lr_schedule.warmup_steps"), "peak_lr": r.fields.get("lr_schedule.peak_lr"),
               "decay_lr": r.fields.get("lr_schedule.decay_lr"), "action_horizon": r.fields.get("model.action_horizon"),
               "paligemma_variant": r.fields.get("model.paligemma_variant"),
               "action_expert_variant": r.fields.get("model.action_expert_variant"),
               "freeze": r.structure.get("freeze"), "image_center_crop": r.fields.get("data.image_center_crop")}
        want = {"track": "realrobot", "paper_budget": True, "steps": stage["steps"], "global_batch": stage["global_batch"],
                "warmup": stage["warmup"], "peak_lr": stage["peak_lr"], "decay_lr": stage["decay_lr"],
                "action_horizon": stage["action_horizon"], "paligemma_variant": "gemma_2b_lora",
                "action_expert_variant": "gemma_300m_lora", "freeze": stage["freeze"],
                "image_center_crop": stage["image"] == "center_crop"}
        problems += [f"{r.id}: {f} = {got[f]!r}, the record says {want[f]!r}" for f in want if got[f] != want[f]]
        problems += [f"{r.id}: no {k} ({PAPER_PRESET} names every model field the openpi line requires)"
                     for k in sorted(model_keys - set(r.fields))]
        if "data.quality_tag" not in r.fields:
            problems.append(f"{r.id}: no data.quality_tag field (the tagged finetune overrides it)")
        aug, want_aug = r.fields.get("data.sim_image_aug", False), stage.get("sim_image_aug", False)
        if aug != want_aug:
            problems.append(f"{r.id}: data.sim_image_aug {aug!r}, the record's {k} stage says {want_aug!r}")
        try:
            units = load_embodiment(stage["embodiment"]).action_units
        except EmbodimentError as exc:
            problems.append(str(exc))
            continue
        got_units = r.fields.get("data.action_units")
        if got_units != units:
            problems.append(f"{r.id}: data.action_units {got_units!r}, the {k} data's units are {units!r}")
    return problems


def check_units(line: Line, run_id: str) -> list[str]:
    spec = line.runs[run_id]
    problems = []
    if kind(spec) == "finetune":
        if line.units["stage1"] != line.units["stage2"] and line.units.get("declared_transfer") is not True:
            problems.append(f"{run_id}: finetunes from {line.units['stage1']} to {line.units['stage2']} units "
                            f"without a declared transfer")
    keys = {"pretrain": ("stage1",), "finetune": ("stage1", "stage2"), "finetune_only": ("stage2",)}[kind(spec)]
    for key in keys:
        try:
            emb = load_embodiment(getattr(line, key)["embodiment"])
        except EmbodimentError as exc:
            problems.append(str(exc))
            continue
        if emb.action_units != line.units[key]:
            problems.append(f"{emb.id}: embodiment units {emb.action_units!r} differ from the record's "
                            f"{key} units {line.units[key]!r}")
        problems += unit_problems(emb)
    return problems


def _stage_image_problems(name: str, stage: dict, emb) -> list[str]:
    """A stage's image against its embodiment: the embodiment's fixed geometry, or, where the frames carry
    their geometry in their stored pixels (fr3), the stage must name the set its image describes."""
    if stage.get("image") not in IMAGES:
        return [f"{name}: image {stage.get('image')!r} is none of {', '.join(IMAGES)}"]
    if emb.geometry_default is None:
        if not stage.get("dataset"):
            return [f"{name}: image {stage['image']!r} is a property of the {emb.id} frames' stored pixels; "
                    f"the stage names no dataset"]
        return []
    if IMAGES[stage["image"]] != emb.geometry_default:
        return [f"{name}: image {stage['image']!r}, the {emb.id} frames are {emb.geometry_default!r}"]
    return []


def check_record(line: Line) -> list[str]:
    from axis_qvla.train.recipes import load_bases

    problems = []
    bases = load_bases()
    pretrains = {r for r, s in line.runs.items() if s["stage"] == "pretrain"}
    for name, stage in (("stage1", line.stage1), ("stage2", line.stage2)):
        try:
            emb = load_embodiment(stage["embodiment"])
        except EmbodimentError as exc:
            problems.append(str(exc))
            continue
        problems += _stage_image_problems(name, stage, emb)
        problems += layout_problems(emb)
    if line.stage1["init"].removeprefix("base:") not in bases:
        problems.append(f"stage1: init {line.stage1['init']!r} is not a base of conf/recipes/bases.toml")
    for run_id, s in sorted(line.runs.items()):
        if s["stage"] == "pretrain":
            if not s.get("signal"):
                problems.append(f"{run_id}: a pretrain run names no signal")
            if "serve" in s:
                problems.append(f"{run_id}: a pretrain run names a serve condition; only finetune runs are served")
            continue
        init = s.get("init", "")
        if not ((init.startswith("base:") and init[5:] in bases) or (init.startswith("run:") and init[4:] in pretrains)):
            problems.append(f"{run_id}: init {init!r} is neither a base nor a pretrain run of the record")
        if s.get("serve") not in line.serve:
            problems.append(f"{run_id}: serve condition {s.get('serve')!r} is not in [serve]")
    return problems
