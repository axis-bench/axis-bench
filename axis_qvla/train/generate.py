"""`qvla train` config generation: an ArmSpec and its recipe, rendered for one backbone.

pi0.5 arms render to a JSON description of an openpi TrainConfig, which the openpi line
(third_party/openpi) builds with its training script (`--config-json <file>`) and serves with its
policy server (`policy:checkpoint --policy.config=<file>`; `openpi_config(..., serve=True)`, what
the benchmark's reference server renders for a published checkpoint); SmolVLA and X-VLA arms render
to a LeRobot train_config.json (axis_qvla.train.lerobot_cfg). Generation is pure: it reads the ArmSpec,
the recipe files and the host's roots, never the environment and never the state of the
filesystem, so one arm generates the same bytes on every machine apart from the root prefixes.
A run naming a `local:` input is refused here.

Two launch options change how a pi0.5 run starts, never what it trains, so neither is an ArmSpec
or recipe field and neither reaches the config hash parity compares: `fsdp_devices` (the devices
each copy of the train state is sharded over; the preset's by default) and `resume` (continue the
run in its checkpoint directory from its last saved step). Both are recorded in run.json.

Core tier: stdlib only.
"""
from __future__ import annotations

from collections.abc import Mapping

from axis_qvla.core.canon import dumps_canonical
from axis_qvla.core.paths import Paths
from axis_qvla.train import ids
from axis_qvla.train.armspec import ArmSpec, armspec_sha256, dumps_armspec
from axis_qvla.train.recipes import Recipe, check_arm_against_recipe, resolved_fields

OPENPI_SCHEMA = "qvla.openpi_train_config/1"
RUN_SCHEMA = "qvla.train_run/2"
SMOKE_STEPS = 20
MIN_STEPS = 100

# The resolved-field keys openpi_config actually reads: three bare scalars, the "data.*"
# selections (the last two only when the recipe sets them), and every key under these group
# prefixes (passed through whole).
_EMITTED_SCALARS = frozenset({"batch_size", "num_train_steps", "ema_decay", "data.repo_id", "data.image_center_crop",
                              "data.default_prompt", "data.sim_image_aug", "data.action_units"})
_OPTIONAL_DATA = ("data.sim_image_aug", "data.action_units")
_EMITTED_GROUPS = ("model.", "lr_schedule.", "optimizer.")
# A recipe field this generator never reads because the pre-launch binding step consumes it when
# it compiles the arm's plan (axis_qvla.train.bind, through axis_qvla.train.ft_bundle): the per-arm constant
# quality-conditioning tag, which is baked into the signal bundle, never into the openpi
# TrainConfig. Not "dropped" -- used elsewhere -- so overriding it is allowed.
CONSUMED_AT_BIND = frozenset({"data.quality_tag"})


class GenerateError(ValueError):
    """An arm cannot be rendered: wrong backbone, a missing input, or a budget below the floor."""


def _check_every_field_is_used(spec_id: str, f: Mapping[str, object]) -> None:
    """Refuse a resolved field (a recipe default or an arm override) that `openpi_config` would
    silently drop: one that is neither rendered into the config nor read at bind time. Without
    this, overriding an unread field changes nothing and raises nothing."""
    dropped = sorted(k for k in f if k not in _EMITTED_SCALARS and k not in CONSUMED_AT_BIND
                     and not k.startswith(_EMITTED_GROUPS))
    if dropped:
        raise GenerateError(f"{spec_id}: recipe field(s) {dropped} are neither rendered into the "
                            f"openpi config nor consumed at bind time; an override of one of them "
                            f"would silently change nothing")


def steps_for(steps: int, *, smoke: bool) -> int:
    """A smoke run trains SMOKE_STEPS steps; a real run refuses fewer than MIN_STEPS."""
    if smoke:
        return SMOKE_STEPS
    if steps < MIN_STEPS:
        raise GenerateError(f"{steps} steps is below the {MIN_STEPS}-step floor for a real run; "
                            f"pass --smoke for a {SMOKE_STEPS}-step check")
    return steps


def check_renderable(spec: ArmSpec) -> None:
    """A run that names an input `local:` cannot be pinned by content, so it is not rendered."""
    local = [f for f, v in (("init", spec.init), ("rowspace", spec.rowspace), ("bundle", spec.bundle),
                            ("norm", spec.norm)) if isinstance(v, str) and v.startswith("local:")]
    if local:
        raise GenerateError(f"{spec.id}: names {', '.join(local)} by a local: name, which qvla train cannot pin; "
                            f"`qvla realrobot pretrain` and `qvla realrobot finetune` (`qvla realrobot fr3 local-runs`) "
                            f"write the run specs that name this machine's inputs")


def openpi_asset_id(spec: ArmSpec, fields: Mapping[str, object]) -> str:
    """The asset id a pi0.5 config renders (its norm statistics sit at `<norm dir>/<asset id>/`):
    the arm's [data].asset_id, else the recipe's data.repo_id."""
    return spec.data.get("asset_id", fields["data.repo_id"])


def quality_tag_problem(spec: ArmSpec, fields: Mapping[str, object]) -> str | None:
    """Why a pi0.5 arm's `data.quality_tag` would change nothing, or None. The tag is compiled only
    into the finetune signal of an arm that names no bundle; a named bundle's codes are its own."""
    if spec.bundle is not None and fields.get("data.quality_tag") is not None:
        return ("recipe: data.quality_tag is set but the run names compiled labels, whose codes are their own; "
                "drop the override or the labels")
    return None


def _group(fields: Mapping[str, object], prefix: str) -> dict[str, object]:
    return {k[len(prefix) + 1:]: v for k, v in fields.items() if k.startswith(prefix + ".")}


def fsdp_devices_for(spec: ArmSpec, recipe: Recipe, fsdp_devices: int | None) -> int:
    """The devices a pi0.5 run shards its train state over: `fsdp_devices` when given, else the
    preset's. It must divide the global batch, since the launch's device count must both be a
    multiple of it and divide the batch (the openpi line checks the device count when it starts)."""
    if fsdp_devices is None:
        return int(recipe.structure["fsdp_devices"])
    if isinstance(fsdp_devices, bool) or not isinstance(fsdp_devices, int) or fsdp_devices < 1:
        raise GenerateError(f"{spec.id}: --fsdp-devices must be a positive integer, got {fsdp_devices!r}")
    batch = int(resolved_fields(recipe, spec)["batch_size"])
    if batch % fsdp_devices:
        raise GenerateError(f"{spec.id}: --fsdp-devices {fsdp_devices} does not divide the global batch {batch}; "
                            f"the devices a run shards over divide the devices it runs on, which divide the batch")
    return fsdp_devices


def openpi_config(spec: ArmSpec, recipe: Recipe, paths: Paths, *, smoke: bool = False,
                  bundle: str | None = None, bundle_dir: str | None = None, serve: bool = False,
                  fsdp_devices: int | None = None) -> dict:
    """`bundle` / `bundle_dir` are what the run was bound to (`Binding.bundle`, `.bundle_dir`); they
    default to the arm's own bundle in the published tree. A pi0.5 finetune arm that names none gets
    its bundle when it is bound. `serve=True` renders the config a policy server builds for one of
    the arm's checkpoints: no bundle is read, so none is needed, and nothing is trained, so the
    real-run step floor does not apply (a smoke arm's checkpoint is served with its recipe's step
    count, as a real arm's is). `fsdp_devices` replaces the preset's sharding (`fsdp_devices_for`);
    the global batch stays the recipe's."""
    if spec.backbone != "pi05-openpi":
        raise GenerateError(f"{spec.id}: openpi_config renders pi0.5 runs, not {spec.backbone}")
    check_renderable(spec)
    check_arm_against_recipe(spec, recipe)
    if serve and smoke:
        raise GenerateError(f"{spec.id}: a serve config is not a smoke run")
    bundle = bundle or spec.bundle
    if bundle is None and not serve:
        raise GenerateError(f"{spec.id}: the run names no compiled labels, so its finetune plan is compiled when "
                            f"it is bound; drop --no-bind")
    if spec.bundle is not None and bundle != spec.bundle:
        raise GenerateError(f"{spec.id}: bound to {bundle} but the run names {spec.bundle}")
    if bundle is not None and bundle_dir is None:
        bundle_dir = str(ids.bundle_dirs(paths, bundle)[0])
    for what, value in (("init.ckpt", spec.init), ("norm.artifact", spec.norm)):
        if value is None:
            raise GenerateError(f"{spec.id}: a pi0.5 run needs {what}")
    f = resolved_fields(recipe, spec)
    _check_every_field_is_used(spec.id, f)
    problem = quality_tag_problem(spec, f)
    if problem:
        raise GenerateError(f"{spec.id}: {problem}")
    steps = int(f["num_train_steps"]) if serve else steps_for(int(f["num_train_steps"]), smoke=smoke)
    rs = ids.rowspace_dirs(paths, spec.rowspace)
    st = recipe.structure
    return {
        "schema": OPENPI_SCHEMA,
        "name": spec.id,
        "exp_name": "serve" if serve else ("smoke" if smoke else "train"),
        "seed": spec.seed,
        "batch_size": f["batch_size"],
        "num_train_steps": steps,
        "ema_decay": f["ema_decay"],
        "model": _group(f, "model"),
        "lr_schedule": _group(f, "lr_schedule"),
        "optimizer": _group(f, "optimizer"),
        "freeze": st["freeze"],
        "fsdp_devices": fsdp_devices_for(spec, recipe, fsdp_devices),
        "num_workers": st["num_workers"],
        "save_interval": steps if smoke else st["save_interval"],
        "keep_period": st["keep_period"],
        "log_interval": 1 if smoke else st["log_interval"],
        "wandb_enabled": False,
        "overwrite": smoke,
        "resume": False,
        "checkpoint_base_dir": str(paths.runs),
        "assets_base_dir": str(ids.run_root(paths, spec.id) / "assets"),
        "weight_loader": {"params_path": ids.openpi_init_params(paths, spec.init)},
        "data": {
            "repo_id": f["data.repo_id"],
            "asset_id": openpi_asset_id(spec, f),
            "image_center_crop": f["data.image_center_crop"],
            "default_prompt": f["data.default_prompt"],
            "prompt_from_task": st["prompt_from_task"],
            "roots_index": str(rs["roots"]),
            # The openpi line reads a flat row space's rows as frames of the ranges walk over the
            # roots index (axis_qvla.train.ft_bundle.FLAT_ROW_DECODE), so the file is named, not inferred.
            "ranges_path": str(rs["ranges"]),
            "corpus_base": str(rs["corpus_base"]),
            "bundle": None if serve else bundle_dir,
            "bundle_id": None if serve else bundle,
            "norm_stats_dir": str(ids.norm_dir(paths, spec.norm)),
            "smoke": smoke,
            **{k[len("data."):]: f[k] for k in _OPTIONAL_DATA if k in f},
        },
        "identity": {"arm": spec.id, "armspec_sha256": armspec_sha256(spec), "preset": recipe.id,
                     "init": spec.init, "rowspace": spec.rowspace, "bundle": bundle, "norm": spec.norm},
    }


def render(spec: ArmSpec, recipe: Recipe, paths: Paths, *, smoke: bool = False,
           signal_files: Mapping[str, str] | None = None, bundle: str | None = None,
           bundle_dir: str | None = None, fsdp_devices: int | None = None,
           resume: bool = False) -> tuple[dict[str, bytes], list[str]]:
    """The files of the run directory (name -> bytes) and the command that trains the run.

    The command runs from the backbone's checkout (third_party/openpi or third_party/lerobot) with
    that checkout's venv. `fsdp_devices` and `resume` are pi0.5 launch options (module docstring):
    the first is rendered into the config's `fsdp_devices`, the second only into the command
    (`--resume` after the config), and run.json records both. A real run's files (`run_spec.toml`, the backbone config, `run.json`) sit
    directly at `<runs>/<run id>/`. A smoke's do NOT: `<runs>/<run id>/smoke/` is the smoke's own
    output/checkpoint directory (LeRobot's `output_dir`, openpi's `exp_name="smoke"` checkpoint
    path -- axis_qvla.train.ids.lerobot_output_dir) -- a backbone either refuses to train into an
    existing one (LeRobot, resume=false) or deletes and recreates it on every launch (openpi), so a
    rendered config written there would be refused or destroyed before training even starts. A
    smoke's files -- all three, `run_spec.toml` included, so the two modes never share a file -- key
    under the sibling `<runs>/<run id>/rendered-smoke/` instead, which no backbone ever writes
    into. It sets the one environment variable each pinned fork reads its base weights through
    (openpi's OPENPI_DATA_HOME, LeRobot's HF_HOME with HF_HUB_OFFLINE=1) to the local copy of the
    base weights (axis_qvla.train.ids).
    """
    from axis_qvla.train.lerobot_cfg import lerobot_argv, lerobot_config

    # Where THIS call writes its own rendered files -- never the smoke output/checkpoint
    # directory itself (ids.lerobot_output_dir), which a backbone owns exclusively.
    file_dir = ids.rendered_dir(paths, spec.id, smoke=smoke)

    def key(name: str) -> str:
        return (file_dir / name).relative_to(ids.run_root(paths, spec.id)).as_posix()

    files = {key("run_spec.toml"): dumps_armspec(spec).encode()}
    if spec.backbone == "pi05-openpi":
        if resume and smoke:
            raise GenerateError(f"{spec.id}: a smoke run starts over at every launch; it is not resumed")
        cfg = openpi_config(spec, recipe, paths, smoke=smoke, bundle=bundle, bundle_dir=bundle_dir,
                            fsdp_devices=fsdp_devices)
        files[key("openpi_config.json")] = dumps_canonical(cfg)
        argv = ["env", f"OPENPI_DATA_HOME={ids.openpi_home(paths)}",
                "python", "scripts/train.py", "--config-json", str(file_dir / "openpi_config.json"),
                *(["--resume"] if resume else [])]
        sharded = cfg["fsdp_devices"]
    else:
        if fsdp_devices is not None or resume:
            raise GenerateError(f"{spec.id}: --fsdp-devices and --resume are pi0.5 launch options; a "
                                f"{spec.backbone} run's GPUs are its recipe's world")
        cfg = lerobot_config(spec, recipe, paths, signal_files=signal_files or {}, smoke=smoke)
        files[key("train_config.json")] = dumps_canonical(cfg)
        argv = ["env", f"HF_HOME={ids.hf_home(paths)}", "HF_HUB_OFFLINE=1",
                *lerobot_argv(file_dir / "train_config.json", int(cfg["parallelism"]["dp_replicate"]))]
        sharded = None
    files[key("run.json")] = dumps_canonical({"schema": RUN_SCHEMA, "arm": spec.id, "armspec_sha256": armspec_sha256(spec),
                                              "preset": recipe.id, "smoke": smoke, "fsdp_devices": sharded,
                                              "resume": resume, "argv": argv})
    return files, argv
