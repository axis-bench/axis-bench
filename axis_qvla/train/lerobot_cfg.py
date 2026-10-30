"""`qvla train` for SmolVLA and X-VLA: an ArmSpec rendered as a LeRobot train_config.json.

The recipe's template is the train_config.json the paper's plain-BC run recorded, with every
per-arm and per-machine value neutral. This module applies the recipe budget, the arm's overrides,
its seed, its init and its signal files, and the machine's paths. The result is handed to the
LeRobot line as `lerobot-train --config_path=<file>`, which loads it the way a resumed run reloads
its own recorded config, so the run sees exactly these values.

An arm may not override a field this module writes itself (OWNED_FIELDS): a signal file binding
exports from the bundle, or the seed, output and job names, dataset copy and init it sets from the
ArmSpec and the machine. Such an override would either put a file binding never checked into the
run (the LeRobot line's signal lock passes a file it does not know) or be replaced without a word.
Per-frame loss weights a run trains with come only from the bundle: the LeRobot line's other
weighters build weights from files of their own (`rabc` reads a progress file, named or found
beside the dataset), which binding never loads. So a config's `sample_weighting` is null unless
the bundle exports a weights file, and then its type is `axis_flat`, the weighter that reads it.

Core tier: stdlib only.
"""
from __future__ import annotations

import copy
import pathlib
from collections.abc import Mapping

from axis_qvla.core.paths import Paths
from axis_qvla.train import ids
from axis_qvla.train.armspec import ArmSpec
from axis_qvla.train.generate import SMOKE_STEPS, GenerateError, check_renderable, steps_for
from axis_qvla.train.recipes import Recipe, check_arm_against_recipe, load_template

LEROBOT_BACKBONES = ("smolvla-lerobot", "xvla-lerobot")
# The train_config.json keys a signal file is passed through (the LeRobot line's own flags).
WEIGHTS_FIELD = "sample_weighting.extra_params.weights_path"
SIGNAL_FIELDS = ("axis_rows_path", "axis_schedule_path", "axis_quality_path", WEIGHTS_FIELD)
# The keys this module sets from the ArmSpec and the machine after the overrides are applied.
GENERATOR_FIELDS = ("seed", "output_dir", "job_name", "dataset.root", "policy.pretrained_path")
OWNED_FIELDS = SIGNAL_FIELDS + GENERATOR_FIELDS
# The LeRobot line's weighter for per-frame loss weights (sample_weighting.type).
AXIS_FLAT = "axis_flat"
# Fields of the LeRobot line's train config that the recorded templates predate, each with the value
# a config without it trains with (the line's default). An arm may override one, and the override is
# written; otherwise the field is left out, so a paper arm's config stays the one its run recorded.
# axis_quality_unconditional = false admits pure conditioning: a quality file with no untagged row.
LINE_FIELDS: dict[str, object] = {"axis_quality_unconditional": True}


def apply_line_field(cfg: dict, key: str, value: object, *, source: str) -> None:
    default = LINE_FIELDS[key]
    if type(value) is not type(default):
        raise GenerateError(f"{source}: {key!r} is a {type(default).__name__}, got {value!r}")
    cfg[key] = value


def apply_override(cfg: dict, key: str, value: object, *, source: str) -> None:
    """Set dotted `key` in `cfg`. Its first component must be a key of the template; a missing or
    null intermediate becomes a table (e.g. `sample_weighting.type` under a null sample_weighting)."""
    parts = key.split(".")
    if parts[0] not in cfg:
        raise GenerateError(f"{source}: {key!r}: {parts[0]!r} is not a field of the LeRobot train config")
    node = cfg
    for p in parts[:-1]:
        nxt = node.get(p)
        if nxt is None:
            nxt = node[p] = {}
        elif not isinstance(nxt, dict):
            raise GenerateError(f"{source}: {key!r}: {p!r} is a value, not a table")
        node = nxt
    node[parts[-1]] = copy.deepcopy(value)


def owned_override_problems(spec: ArmSpec) -> list[str]:
    """Every override of a field this module writes itself, or of a table holding one."""
    out = []
    for key in sorted(spec.overrides):
        hit = [f for f in OWNED_FIELDS if key == f or f.startswith(key + ".") or key.startswith(f + ".")]
        if hit:
            who = ("binding writes from the compiled labels' exported label files" if set(hit) <= set(SIGNAL_FIELDS)
                   else "qvla train sets from the run spec and this machine")
            out.append(f"recipe: the override {key!r} sets {', '.join(hit)}, which {who}; drop the override")
    return out


def weighting_problem(cfg: Mapping[str, object], signal_fields) -> str | None:
    """Why the config's `sample_weighting` does not fit the exported signal files, or None: it is
    null unless the bundle exports per-frame weights, and AXIS_FLAT (the weighter that reads them)
    when it does."""
    sw = cfg.get("sample_weighting")
    kind = sw.get("type") if isinstance(sw, dict) else None
    weighted = WEIGHTS_FIELD in set(signal_fields)
    if weighted and kind != AXIS_FLAT:
        return (f"recipe: the compiled labels export per-frame loss weights, which the pinned LeRobot fork "
                f"applies only with sample_weighting.type = {AXIS_FLAT!r} (the config has {kind!r}); declare it "
                f"in [recipe.overrides]")
    if not weighted and sw is not None:
        return (f"recipe: the config sets sample_weighting (type {kind!r}) but the compiled labels export no "
                f"per-frame weights file; a run's loss weights come only from its compiled labels, so drop the "
                f"sample_weighting overrides")
    return None


def resolved_template(spec: ArmSpec, recipe: Recipe) -> dict:
    """The recipe's template with its budget and the arm's overrides applied (no signal files, no paths)."""
    cfg = copy.deepcopy(load_template(recipe))
    cfg["steps"] = recipe.steps
    cfg["batch_size"] = recipe.per_rank_batch
    cfg["parallelism"]["dp_replicate"] = recipe.world
    for key in sorted(spec.overrides):
        (apply_line_field if key in LINE_FIELDS else apply_override)(cfg, key, spec.overrides[key], source=spec.id)
    return cfg


def override_problems(spec: ArmSpec, recipe: Recipe, signal_fields) -> list[str]:
    """Every reason the arm's overrides do not fit the run it renders with `signal_fields` exported
    (owned_override_problems, weighting_problem); [] when they fit. Binding lists these with its own."""
    out = owned_override_problems(spec)
    if not out:
        problem = weighting_problem(resolved_template(spec, recipe), signal_fields)
        out += [problem] if problem else []
    return out


def lerobot_config(spec: ArmSpec, recipe: Recipe, paths: Paths, *, signal_files: Mapping[str, object],
                   init_value: object = None, smoke: bool = False) -> dict:
    """`init_value` replaces the resolved policy.pretrained_path (parity uses a logical token)."""
    if spec.backbone not in LEROBOT_BACKBONES:
        raise GenerateError(f"{spec.id}: lerobot_config renders SmolVLA / X-VLA runs, not {spec.backbone}")
    check_renderable(spec)
    check_arm_against_recipe(spec, recipe)
    if spec.init is None:
        raise GenerateError(f"{spec.id}: a LeRobot run needs init.ckpt")
    unknown = sorted(set(signal_files) - set(SIGNAL_FIELDS))
    if unknown:
        raise GenerateError(f"{spec.id}: {unknown} are not LeRobot label fields {list(SIGNAL_FIELDS)}")
    problems = override_problems(spec, recipe, signal_files)
    if problems:
        raise GenerateError(f"{spec.id}: " + "; ".join(problems))
    cfg = resolved_template(spec, recipe)
    cfg["steps"] = steps_for(int(cfg["steps"]), smoke=smoke)
    if smoke:
        cfg["save_freq"] = SMOKE_STEPS
    cfg["seed"] = spec.seed
    for key in sorted(signal_files):
        apply_override(cfg, key, signal_files[key], source=spec.id)
    rs = ids.rowspace_dirs(paths, spec.rowspace)
    cfg["dataset"]["root"] = str(rs["lerobot_dataset"])
    cfg["output_dir"] = str(ids.lerobot_output_dir(paths, spec.id, smoke=smoke))
    cfg["job_name"] = spec.id
    cfg["policy"]["pretrained_path"] = (init_value if init_value is not None
                                        else ids.lerobot_init(paths, spec.init, spec.backbone))
    per_rank, world = int(cfg["batch_size"]), int(cfg["parallelism"]["dp_replicate"])
    if per_rank * world != recipe.global_batch:
        raise GenerateError(f"{spec.id}: per-rank batch {per_rank} x world {world} != the declared global "
                            f"batch {recipe.global_batch}")
    return cfg


def lerobot_argv(config_path: pathlib.Path, world: int) -> list[str]:
    """`torchrun --standalone` picks a free rendezvous port, so two runs on one box do not collide."""
    return ["torchrun", "--standalone", f"--nproc-per-node={world}", "-m", "lerobot.scripts.lerobot_train",
            f"--config_path={config_path}"]
