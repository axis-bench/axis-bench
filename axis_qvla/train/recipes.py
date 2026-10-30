"""Recipe presets, tracks, base weights and row-space layouts: the tables an ArmSpec names.

A recipe holds the training fields of one (backbone, stage, track). For pi0.5 they are openpi
TrainConfig fields under their dotted openpi names ([fields]); for SmolVLA and X-VLA the recipe
points at a LeRobot train_config.json template (conf/recipes/lerobot/) that holds every field the
paper's runs recorded, with the per-arm and per-machine ones neutral. An ArmSpec may override only
fields its recipe declares, so a misspelt override is refused instead of adding a field nobody
reads. A track fixes the budget (steps x global batch) per backbone and stage. `paper_budget` states a
fact, on a track and on each preset: its budgets are the ones the paper's runs trained with (true for
the paper and real-robot tracks, false for small and smoke).

Core tier: stdlib only.
"""
from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import re
import tomllib
import types
from collections.abc import Mapping

from axis_qvla.train.armspec import BACKBONES, STAGES, ArmSpec

RECIPES_DIR = pathlib.Path(__file__).resolve().parents[2] / "conf" / "recipes"
RECIPE_SCHEMA = "qvla.recipe/1"
TRACK_SCHEMA = "qvla.track/1"
# "lora_frozen_vision" is the real-robot line's stage 1 (LoRA on both experts, vision tower frozen).
FREEZES = ("none", "lora", "lora_frozen_vision")
_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_TOP = frozenset({"schema", "id", "backbone", "stage", "track", "paper_budget", "description", "nulls",
                  "template", "budget", "fields", "structure"})
_BUDGET = ("steps", "global_batch", "per_rank_batch", "world")
_STRUCTURE = {
    "pi05-openpi": frozenset({"freeze", "fsdp_devices", "num_workers", "save_interval", "keep_period",
                              "log_interval", "prompt_from_task"}),
    "smolvla-lerobot": frozenset(),
    "xvla-lerobot": frozenset(),
}
_TEMPLATE_KEYS = frozenset({"steps", "batch_size", "seed", "parallelism", "policy", "dataset", "output_dir",
                            "job_name"})


class RecipeError(ValueError):
    """A recipe, track or table is malformed, or an arm does not fit its recipe."""


@dataclasses.dataclass(frozen=True)
class Recipe:
    id: str
    backbone: str
    stage: str
    track: str
    paper_budget: bool          # its budgets are the paper's runs' (a fact, not a judgment)
    description: str
    steps: int
    global_batch: int
    per_rank_batch: int
    world: int
    fields: Mapping[str, object]
    structure: Mapping[str, object]
    template: str | None


@dataclasses.dataclass(frozen=True)
class Budget:
    steps: int
    global_batch: int


@dataclasses.dataclass(frozen=True)
class Track:
    name: str
    paper_budget: bool
    budgets: Mapping[tuple[str, str], Budget]


@dataclasses.dataclass(frozen=True)
class Base:
    """A base model an arm may start from. `local` is what the run reads: for pi0.5 a params
    directory inside the base-weights copy, for LeRobot the Hub id the recorded train_config.json
    carries ("" when the policy loads its own backbone), resolved offline from that copy's Hugging
    Face cache. `upstream` is where it is published; conf/recipes/bases.lock.json pins the exact
    files `qvla data get base-weights` fetches from there."""

    name: str
    backbone: str
    local: str
    upstream: str


# What the merged LeRobot copy SmolVLA and X-VLA read is: "published" (a lock-verified repository's
# file), "cut" (built on this machine by `axis_merge_datasets --ranges` from the row space's ranges
# file, so binding checks its cut record) or "plain" (built on this machine by a plain merge).
LEROBOT_COPIES = ("published", "cut", "plain")


@dataclasses.dataclass(frozen=True)
class RowSpaceLayout:
    """Where a row space's files are: `repo` is a published repository (placed by the data
    layout's default-destination helper) or "local" (this machine's derived data,
    local_dest("derived")). `lerobot_copy` says what the merged LeRobot copy is (LEROBOT_COPIES); a
    copy built here sits in the derived data even when the rest is published. A "cut" copy's
    `action_source` is the column its action is read from. `n_rows` is the frame count a published
    frame index records; a local one (built on this machine) records none (None)."""

    name: str
    repo: str
    roots: str
    ranges: str
    corpus_base: str
    lerobot_dataset: str
    lerobot_copy: str
    n_rows: int | None
    action_source: str | None = None

    @property
    def lerobot_local(self) -> bool:
        """The merged LeRobot copy is built on this machine (in the derived data)."""
        return self.lerobot_copy != "published"


def _read(path: pathlib.Path) -> dict:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RecipeError(f"{path}: cannot read ({exc.__class__.__name__}: {exc})") from None
    except tomllib.TOMLDecodeError as exc:
        raise RecipeError(f"{path}: not valid TOML ({exc})") from None


def _positive_int(v: object, where: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
        raise RecipeError(f"{where} must be a positive integer, got {v!r}")
    return v


def load_template_file(path: pathlib.Path) -> dict:
    """A LeRobot train_config.json template: parsed, with the keys the generator writes present."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RecipeError(f"{path}: cannot read ({exc.__class__.__name__}: {exc})") from None
    except ValueError as exc:
        raise RecipeError(f"{path}: not valid JSON ({exc})") from None
    missing = sorted(_TEMPLATE_KEYS - set(doc))
    if missing:
        raise RecipeError(f"{path}: template lacks {missing}")
    return doc


def load_recipe(preset: str, *, root: pathlib.Path = RECIPES_DIR) -> Recipe:
    if not isinstance(preset, str) or not _ID.fullmatch(preset):
        raise RecipeError(f"recipe id {preset!r} must match {_ID.pattern}")
    path = root / f"{preset}.toml"
    doc = _read(path)
    if "validated" in doc:
        raise RecipeError(f"{path}: `validated` is now `paper_budget` (true when the budgets are the paper's runs')")
    extra = sorted(set(doc) - _TOP)
    if extra:
        raise RecipeError(f"{path}: unknown key(s) {extra}; allowed: {sorted(_TOP)}")
    if doc.get("schema") != RECIPE_SCHEMA:
        raise RecipeError(f"{path}: schema must be {RECIPE_SCHEMA!r}")
    if doc.get("id") != preset:
        raise RecipeError(f"{path}: id {doc.get('id')!r} is not the file name {preset!r}")
    backbone, stage, track = doc.get("backbone"), doc.get("stage"), doc.get("track")
    if backbone not in BACKBONES or stage not in STAGES or not isinstance(track, str):
        raise RecipeError(f"{path}: backbone/stage/track {backbone!r}/{stage!r}/{track!r} are not valid")
    if not isinstance(doc.get("paper_budget"), bool):
        raise RecipeError(f"{path}: paper_budget must be true or false")
    budget = doc.get("budget", {})
    if sorted(budget) != sorted(_BUDGET):
        raise RecipeError(f"{path}: [budget] needs exactly {list(_BUDGET)}")
    steps, gb, prb, world = (_positive_int(budget[k], f"{path}: budget.{k}") for k in _BUDGET)
    if prb * world != gb:
        raise RecipeError(f"{path}: per_rank_batch {prb} x world {world} != global_batch {gb}")
    fields, structure = dict(doc.get("fields", {})), dict(doc.get("structure", {}))
    for ref in doc.get("nulls", []):
        table, _, key = str(ref).partition(".")
        target = {"fields": fields, "structure": structure}.get(table)
        if target is None or not key or key in target:
            raise RecipeError(f"{path}: nulls entry {ref!r} must name a key absent from [fields] or [structure]")
        target[key] = None
    unknown = sorted(set(structure) - _STRUCTURE[backbone])
    if unknown:
        raise RecipeError(f"{path}: [structure] has unknown key(s) {unknown}")
    template = doc.get("template")
    if backbone == "pi05-openpi":
        if template is not None:
            raise RecipeError(f"{path}: a pi0.5 recipe keeps its fields in [fields], not a template")
        if fields.get("batch_size") != gb or fields.get("num_train_steps") != steps:
            raise RecipeError(f"{path}: fields batch_size/num_train_steps must equal the budget")
        if structure.get("freeze") not in FREEZES:
            raise RecipeError(f"{path}: structure.freeze must be one of {list(FREEZES)}")
    else:
        if fields:
            raise RecipeError(f"{path}: a LeRobot recipe keeps its fields in its template")
        if not isinstance(template, str):
            raise RecipeError(f"{path}: a LeRobot recipe needs a template path")
        load_template_file(root / template)
    return Recipe(id=preset, backbone=backbone, stage=stage, track=track, paper_budget=doc["paper_budget"],
                  description=str(doc.get("description", "")), steps=steps, global_batch=gb,
                  per_rank_batch=prb, world=world, fields=types.MappingProxyType(fields),
                  structure=types.MappingProxyType(structure), template=template)


def load_template(recipe: Recipe, *, root: pathlib.Path = RECIPES_DIR) -> dict:
    if recipe.template is None:
        raise RecipeError(f"{recipe.id} has no LeRobot template")
    return load_template_file(root / recipe.template)


def list_recipes(*, root: pathlib.Path = RECIPES_DIR) -> list[str]:
    """Every preset id under `root`. os.scandir, not Path.glob: glob turns an unreadable directory
    into an empty list, which would read as 'no recipes'."""
    try:
        with os.scandir(root) as it:
            return sorted(e.name[:-5] for e in it if e.name.endswith(".toml")
                          and e.name not in ("bases.toml", "rowspaces.toml"))
    except OSError as exc:
        raise RecipeError(f"cannot list {root}: {exc.__class__.__name__}: {exc}") from None


def load_track(name: str, *, root: pathlib.Path = RECIPES_DIR) -> Track:
    path = root / "tracks" / f"{name}.toml"
    doc = _read(path)
    if "validated" in doc:
        raise RecipeError(f"{path}: `validated` is now `paper_budget` (true when the budgets are the paper's runs')")
    if doc.get("schema") != TRACK_SCHEMA or doc.get("name") != name or not isinstance(doc.get("paper_budget"), bool):
        raise RecipeError(f"{path}: needs schema {TRACK_SCHEMA!r}, name {name!r} and a boolean paper_budget")
    extra = sorted(set(doc) - {"schema", "name", "paper_budget", "description", "budgets"})
    if extra:
        raise RecipeError(f"{path}: unknown key(s) {extra}")
    budgets: dict[tuple[str, str], Budget] = {}
    for backbone, stages in doc.get("budgets", {}).items():
        if backbone not in BACKBONES:
            raise RecipeError(f"{path}: unknown backbone {backbone!r}")
        for stage, b in stages.items():
            if stage not in STAGES or not isinstance(b, dict) or sorted(b) != ["global_batch", "steps"]:
                raise RecipeError(f"{path}: budgets.{backbone}.{stage} must be {{steps, global_batch}}")
            budgets[(backbone, stage)] = Budget(_positive_int(b["steps"], f"{path}: steps"),
                                                _positive_int(b["global_batch"], f"{path}: global_batch"))
    return Track(name=name, paper_budget=doc["paper_budget"], budgets=types.MappingProxyType(budgets))


def check_recipe_against_track(recipe: Recipe, track: Track) -> None:
    b = track.budgets.get((recipe.backbone, recipe.stage))
    if b is None:
        raise RecipeError(f"track {track.name} declares no budget for {recipe.backbone} {recipe.stage}")
    if (b.steps, b.global_batch) != (recipe.steps, recipe.global_batch):
        raise RecipeError(f"{recipe.id}: budget {recipe.steps} x {recipe.global_batch} is not track "
                          f"{track.name}'s {b.steps} x {b.global_batch}")
    if recipe.paper_budget and not track.paper_budget:
        raise RecipeError(f"{recipe.id}: track {track.name}'s budgets are not the paper's, so its recipe's are not either "
                          f"(paper_budget = true)")


def check_arm_against_recipe(spec: ArmSpec, recipe: Recipe) -> None:
    for what, a, r in (("backbone", spec.backbone, recipe.backbone), ("stage", spec.stage, recipe.stage),
                       ("track", spec.track, recipe.track)):
        if a != r:
            raise RecipeError(f"{spec.id}: run {what} is {a!r} but recipe {recipe.id} is for {r!r}")


def resolved_fields(recipe: Recipe, spec: ArmSpec) -> dict[str, object]:
    """A pi0.5 recipe's fields with the arm's overrides applied; an override the recipe does not
    declare is refused."""
    if recipe.template is not None:
        raise RecipeError(f"{recipe.id}: LeRobot recipes resolve through their template")
    check_arm_against_recipe(spec, recipe)
    out = dict(recipe.fields)
    for k, v in spec.overrides.items():
        if k not in out:
            raise RecipeError(f"{spec.id}: override {k!r} is not a field of recipe {recipe.id}")
        out[k] = v
    return out


def _bases_doc(root: pathlib.Path) -> tuple[pathlib.Path, dict]:
    path = root / "bases.toml"
    doc = _read(path)
    if doc.pop("schema", None) != "qvla.bases/3":
        raise RecipeError(f"{path}: schema must be 'qvla.bases/3'")
    return path, {"bases": doc}


def load_bases(*, root: pathlib.Path = RECIPES_DIR) -> dict[str, Base]:
    path, doc = _bases_doc(root)
    out = {}
    for name, t in doc["bases"].items():
        if not isinstance(t, dict) or sorted(t) != ["backbone", "local", "upstream"] or t["backbone"] not in BACKBONES:
            raise RecipeError(f"{path}: [{name}] needs exactly backbone, local, upstream")
        out[name] = Base(name=name, backbone=t["backbone"], local=str(t["local"]), upstream=str(t["upstream"]))
    return out


def load_rowspaces(*, root: pathlib.Path = RECIPES_DIR) -> dict[str, RowSpaceLayout]:
    path = root / "rowspaces.toml"
    doc = _read(path)
    if doc.pop("schema", None) != "qvla.rowspaces/1":
        raise RecipeError(f"{path}: schema must be 'qvla.rowspaces/1'")
    keys = ["corpus_base", "lerobot_copy", "lerobot_dataset", "n_rows", "ranges", "repo", "roots"]
    out = {}
    for name, t in doc.items():
        copy = t.get("lerobot_copy") if isinstance(t, dict) else None
        local = isinstance(t, dict) and t.get("repo") == "local"
        need = [k for k in keys if not (k == "n_rows" and local)]
        want = sorted(need + (["n_rows"] if local and "n_rows" in t else []) + (["action_source"] if copy == "cut" else []))
        if not isinstance(t, dict) or sorted(t) != want or copy not in LEROBOT_COPIES:
            raise RecipeError(f"{path}: [{name}] needs exactly {keys} (a local one may omit n_rows), plus "
                              f"action_source when lerobot_copy is \"cut\" (lerobot_copy one of {list(LEROBOT_COPIES)})")
        if copy == "cut" and not (isinstance(t["action_source"], str) and t["action_source"]):
            raise RecipeError(f"{path}: [{name}] action_source must name the column the cut copy's action is read from")
        if t["repo"] == "local" and copy == "published":
            raise RecipeError(f"{path}: [{name}] is a local frame index, so its LeRobot copy cannot be published")
        out[name] = RowSpaceLayout(name=name, repo=t["repo"], roots=t["roots"], ranges=t["ranges"],
                                   corpus_base=t["corpus_base"], lerobot_dataset=t["lerobot_dataset"],
                                   lerobot_copy=copy,
                                   n_rows=_positive_int(t["n_rows"], f"{path}: {name}.n_rows") if "n_rows" in t else None,
                                   action_source=t.get("action_source"))
    return out
