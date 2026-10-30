"""What `qvla bench eval --from-run <run id>` serves for a run trained here: the training package's
interface to the benchmark.

Why: a run trained here is evaluated the way the paper's checkpoints are. That means the right
reference server, its checkpoint, the config that server builds, and the serving modes the run was
trained for: bare, plus cond for a run fine-tuned with `--cond`. The benchmark should not re-derive
any of that from the run directory's layout. `serve_spec` returns it whole. It refuses a run that
`qvla train` did not write, did not train its whole budget, or is a smoke run, and names why. A run
that is only not ready yet (not there, not finished, a step file missing) raises `RunNotReady`, so a
caller can tell "train it first" from a defect (a malformed eval.json, a run spec in another run's
directory, a config that does not render, a step file that is there but unreadable or not JSON),
which raises the plain `ServeSpecError`.

The seeds and the pretraining run come from eval.json when `qvla train finetune` wrote one. Without
it, a fine-tune of a run trained here (`init = run:<id>`) takes that run and its recorded seed, and a
fine-tune of a published checkpoint takes the seed the paper run that trained that checkpoint
records (experiments/paper/runs: a pretraining's seed, a fine-tune's `pretrain_seed`; None when no
paper run trained it).

A finished run directory, <runs>/<run id>/:
  spec.toml          the spec `qvla train pretrain|finetune` wrote, inputs by name (when it wrote one)
  run_spec.toml      the spec that trained, every input by content            (render)
  names.json         each name and the id it resolved to                     (qvla train)
  run.json, binding.json, openpi_config.json | train_config.json (+ signals/)
  eval.json          serving modes and the pretraining run (qvla train finetune)
  serve_config.json  pi0.5: the config the reference server builds         (serve_spec)
  train/             pi0.5: <step>/{params,train_state,assets}, final -> last step, train_summary.json;
                     LeRobot: checkpoints/<step>/{pretrained_model,training_state}, checkpoints/last

Core tier: stdlib only.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import types
from collections.abc import Mapping

from axis_qvla.core.canon import dumps_canonical
from axis_qvla.core.paths import Paths
from axis_qvla.train import ids
from axis_qvla.train.armspec import ArmSpecError, load_armspec
from axis_qvla.train.runs import SHORT_NAMES, checkpoint_dir, load_run_spec, unfinished

EVAL_FILE = "eval.json"
EVAL_SCHEMA = "qvla.train_eval/1"
SERVE_CONFIG = "serve_config.json"
QUALITY_TAG = 5                 # the tag the paper's conditioned results were served and fine-tuned with
BACKENDS = {"pi05-openpi": "openpi", "smolvla-lerobot": "smolvla", "xvla-lerobot": "xvla"}
FAMILIES = SHORT_NAMES          # backbone -> ServeSpec.backbone ("pi05" | "smolvla" | "xvla")


@dataclasses.dataclass(frozen=True)
class ServeMode:
    name: str                       # "bare" | "cond": the row of the results table
    serving: str                    # `qvla bench eval --serving`: "bare" | "conditioned"
    fields: Mapping[str, str]       # cells.json `serve.fields` form: {} or {"quality": "5"}


MODES = {"bare": ServeMode("bare", "bare", types.MappingProxyType({})),
         "cond": ServeMode("cond", "conditioned", types.MappingProxyType({"quality": str(QUALITY_TAG)}))}


@dataclasses.dataclass(frozen=True)
class ServeSpec:
    run_id: str
    backbone: str                   # "pi05" | "smolvla" | "xvla"
    backend: str                    # `qvla bench eval --backend`: "openpi" | "smolvla" | "xvla"
    stage: str                      # "pretrain" | "finetune"
    checkpoint: pathlib.Path        # pi0.5: the step dir (params/, assets/); LeRobot: pretrained_model/
    config: pathlib.Path | None     # pi0.5: serve_config.json (`--config <file>.json`); LeRobot: None
    modes: tuple[ServeMode, ...]    # bare first
    ft_seed: int | None             # the fine-tune's training seed (None for a pretraining run)
    pretrain_seed: int | None       # the pretraining's (None when neither this machine nor the release records it)
    pretrain: str | None            # the run id it was fine-tuned from (None: not a run trained here)
    config_bytes: bytes | None = None   # pi0.5: serve_config.json's bytes, also when serve_spec wrote no file


class ServeSpecError(ValueError):
    """The directory is not a finished run `qvla train` wrote."""


class RunNotReady(ServeSpecError):
    """The run is not there yet, has not finished, or lacks a file its last step must leave: train it
    (the message names how), as opposed to a defect in what is there."""


def _write(dest: pathlib.Path, data: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)


def write_eval_record(run_dir, *, pretrain: str, pretrain_seed: int, cond: bool) -> pathlib.Path:
    """Record how `qvla train finetune` means the run to be evaluated: bare, plus cond when it was
    fine-tuned with the quality tag (`--cond`), and the pretraining run it started from."""
    if isinstance(pretrain_seed, bool) or not isinstance(pretrain_seed, int):
        raise ValueError(f"pretrain_seed must be an integer, got {pretrain_seed!r}")
    dest = pathlib.Path(run_dir) / EVAL_FILE
    _write(dest, dumps_canonical({"schema": EVAL_SCHEMA, "pretrain": pretrain, "pretrain_seed": pretrain_seed,
                                  "serving": ["bare", "cond"] if cond else ["bare"], "quality_tag": QUALITY_TAG}))
    return dest


def _eval_record(run_dir: pathlib.Path) -> dict:
    """The run's eval.json, or {} (bare only) when it has none."""
    path = run_dir / EVAL_FILE
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, NotADirectoryError):
        return {}
    except (OSError, ValueError) as exc:
        raise ServeSpecError(f"cannot read {path} ({exc.__class__.__name__})") from None
    serving = doc.get("serving") if isinstance(doc, dict) else None
    if not (isinstance(doc, dict) and doc.get("schema") == EVAL_SCHEMA and isinstance(serving, list)
            and serving and serving[0] == "bare" and len(set(serving)) == len(serving)
            and set(serving) <= set(MODES)):
        raise ServeSpecError(f"{path} is not a {EVAL_SCHEMA} record (serving: bare first, then cond)")
    seed, pretrain = doc.get("pretrain_seed"), doc.get("pretrain")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ServeSpecError(f"{path}: pretrain_seed is {seed!r}, not an integer or null")
    if pretrain is not None and not isinstance(pretrain, str):
        raise ServeSpecError(f"{path}: pretrain is {pretrain!r}, not a run id or null")
    return doc


def _published_pretrain_seed(init: str) -> int | None:
    """The pretraining seed of published checkpoint `init` (`<org>/qvla-v1-<backbone>:<path>/<step>`),
    as the paper run that trained it records it (its [paper] checkpoint is `<backbone>/<path>`): a
    pretraining run's own seed, a fine-tune's `pretrain_seed`. None when no one paper run trained it."""
    from axis_qvla.train.paper import find_arm_by_checkpoint

    try:
        _, repo, path, _ = ids.published(init)
        arm = load_armspec(find_arm_by_checkpoint(f"{repo.removeprefix('qvla-v1-')}/{path.rsplit('/', 1)[0]}"))
    except (ids.IdError, LookupError, ArmSpecError, OSError):
        return None
    seed = arm.seed if arm.stage == "pretrain" else arm.paper.get("pretrain_seed")
    return seed if isinstance(seed, int) and not isinstance(seed, bool) else None


def _pretraining(spec) -> tuple[str | None, int | None]:
    """(pretraining run id, its seed) of a fine-tune with no eval.json, from its init."""
    init = spec.init or ""
    if init.startswith("run:"):
        return init[len("run:"):], None
    return None, _published_pretrain_seed(init)


def serve_spec(run_dir: pathlib.Path, paths: Paths, *, write: bool = True) -> ServeSpec:
    """How to serve the finished run at `run_dir` (<runs>/<run id>). For pi0.5 it renders the server's
    config (ServeSpec.config_bytes) and, when `write`, writes it to <run_dir>/serve_config.json
    (ServeSpec.config names that path either way); `write=False` writes nothing."""
    from axis_qvla.train.generate import GenerateError, openpi_config
    from axis_qvla.train.recipes import RecipeError, load_recipe
    from axis_qvla.train.runs import RUN_SPEC, not_a_file

    run_dir = pathlib.Path(run_dir)
    if not_a_file(run_dir / RUN_SPEC) == "is not there":
        raise RunNotReady(f"{run_dir} is not a run `qvla train` wrote yet: it holds no {RUN_SPEC}; train it "
                          f"first (`qvla train <spec>`, or `qvla train pretrain|finetune`)")
    try:
        spec = load_run_spec(run_dir)
    except ArmSpecError as exc:
        raise ServeSpecError(f"{run_dir} is not a run `qvla train` wrote: {exc}") from None
    home = ids.run_root(paths, spec.id)
    if run_dir.resolve() != home.resolve():
        raise ServeSpecError(f"{run_dir} holds run {spec.id}, whose directory is {home}")
    why = unfinished(spec.id, spec.backbone, paths)
    if why is not None:
        error = RunNotReady if why.not_ready else ServeSpecError
        raise error(why.message(f"{spec.id} cannot be served", home, spec.backbone))
    record = _eval_record(run_dir)
    config = data = None
    if spec.backbone == "pi05-openpi":
        try:
            data = dumps_canonical(openpi_config(spec, load_recipe(spec.preset), paths, serve=True))
        except (GenerateError, RecipeError, ValueError) as exc:
            raise ServeSpecError(f"{spec.id}: {exc}") from None
        config = home / SERVE_CONFIG
        if write:
            _write(config, data)
    finetune = spec.stage == "finetune"
    if not finetune:
        pretrain, pretrain_seed = None, spec.seed
    elif record:
        pretrain, pretrain_seed = record.get("pretrain"), record.get("pretrain_seed")
    else:
        pretrain, pretrain_seed = _pretraining(spec)
        if pretrain is not None:
            try:
                pretrain_seed = load_run_spec(ids.run_root(paths, pretrain)).seed
            except ArmSpecError:
                pretrain_seed = None        # its run is not on this machine: the seed is not recorded here
    return ServeSpec(run_id=spec.id, backbone=FAMILIES[spec.backbone], backend=BACKENDS[spec.backbone],
                     stage=spec.stage, checkpoint=checkpoint_dir(paths, spec.id, spec.backbone), config=config,
                     modes=tuple(MODES[m] for m in record.get("serving", ["bare"])),
                     ft_seed=spec.seed if finetune else None, pretrain_seed=pretrain_seed, pretrain=pretrain,
                     config_bytes=data)
