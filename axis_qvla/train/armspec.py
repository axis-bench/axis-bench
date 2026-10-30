"""ArmSpec: the one file that says what a training run is.

An arm names the backbone and stage, the weights it starts from, the row space and signal bundle
it samples, the normalisation statistics, the recipe preset and seed, and every field it overrides
in that preset. Each input is named by content (`#sha256:` digests and bundle ids) or by name: a
published frame index or normalisation statistics by its name, and compiled labels as
`name:<method>`. axis_qvla.train.names resolves a name to its content id before anything reads the
spec. Either way the same file trains the same run on any machine, and a misspelt key is refused
rather than silently trained as the preset's default. Off track `paper`, an input that ships
nowhere (the real-robot line's data) may be named `local:<name>` instead: it cannot be hash-pinned,
so such an arm is a declaration that `qvla train` refuses to render into a config.

    [arm]     id, backbone, stage, track, paper_ref (optional)
    [init]    ckpt                                      (optional table)
    [data]    rowspace, action_space, image, asset_id   (rowspace required)
    [signal]  bundle                                    (optional table)
    [norm]    artifact                                  (optional table)
    [recipe]  preset, seed, nulls (dotted fields set to null), [recipe.overrides] (dotted fields)
    [paper]   provenance of a paper run; only on track "paper"

Core tier: stdlib only (tomllib needs Python >= 3.11).
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import pathlib
import re
import tomllib
import types
from collections.abc import Mapping

from axis_qvla.core.canon import dumps_canonical, sha256_bytes

BACKBONES = ("pi05-openpi", "smolvla-lerobot", "xvla-lerobot")
STAGES = ("pretrain", "finetune")
# Independent of axis_qvla.train.recipes.RECIPES_DIR (importing it here would be circular: recipes.py
# imports BACKBONES/STAGES/ArmSpec from this module), computed the same way.
_TRACKS_DIR = pathlib.Path(__file__).resolve().parents[2] / "conf" / "recipes" / "tracks"

_SHA = r"[0-9a-f]{64}"
_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_TRACK = re.compile(r"^[a-z][a-z0-9_-]*$")
_CONTENT = re.compile(rf"^[A-Za-z0-9][A-Za-z0-9._/-]*#sha256:{_SHA}$")
_BUNDLE = re.compile(rf"^sha256:{_SHA}$")
# The by-name forms (axis_qvla.train.names resolves them before anything reads the spec): a published
# frame index or normalisation statistics by its name, compiled labels by their method's name.
_REF_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_BUNDLE_NAME = re.compile(r"^name:[a-z0-9][a-z0-9._-]*$")
_LOCAL = re.compile(r"^local:[A-Za-z0-9][A-Za-z0-9._/-]*$")
_INIT = re.compile(
    rf"^(?:base:[a-z0-9][a-z0-9_.-]*"
    rf"|run:[a-z0-9][a-z0-9._-]*"
    rf"|[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*:[A-Za-z0-9][A-Za-z0-9_./-]*#sha256:{_SHA})$")
_DOTTED = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")

# table -> (required keys, optional keys); any other table or key is refused.
_TABLES: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "arm": (frozenset({"id", "backbone", "stage", "track"}), frozenset({"paper_ref"})),
    "init": (frozenset({"ckpt"}), frozenset()),
    "data": (frozenset({"rowspace"}), frozenset({"action_space", "image", "asset_id"})),
    "signal": (frozenset({"bundle"}), frozenset()),
    "norm": (frozenset({"artifact"}), frozenset()),
    "recipe": (frozenset({"preset", "seed"}), frozenset({"overrides", "nulls"})),
    "paper": (frozenset(), frozenset({"checkpoint", "registered_config", "train_config", "launch_overrides",
                                      "signals", "pretrain_seed", "cells", "differences"})),
}
_REQUIRED_TABLES = ("arm", "data", "recipe")
_PAPER_TYPES = {"checkpoint": str, "registered_config": str, "train_config": str, "launch_overrides": list,
                "signals": list, "pretrain_seed": int, "cells": list, "differences": dict}
_DIFFERENCE_KEYS = ("lock", "arm", "reason")


class ArmSpecError(ValueError):
    """An ArmSpec file is malformed, names an unknown table or key, or carries an invalid id."""


def _frozen(d: Mapping | None) -> Mapping:
    return types.MappingProxyType(dict(d or {}))


@dataclasses.dataclass(frozen=True)
class ArmSpec:
    id: str
    backbone: str
    stage: str
    track: str
    init: str | None
    rowspace: str
    bundle: str | None
    norm: str | None
    preset: str
    seed: int
    overrides: Mapping[str, object]
    paper_ref: str | None = None
    data: Mapping[str, str] = dataclasses.field(default_factory=lambda: _frozen(None))
    paper: Mapping[str, object] = dataclasses.field(default_factory=lambda: _frozen(None))


def load_armspec(path: pathlib.Path) -> ArmSpec:
    """Read and validate one ArmSpec file; every refusal names the file."""
    path = pathlib.Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ArmSpecError(f"{path}: cannot read ({exc.__class__.__name__}: {exc})") from None
    return loads_armspec(text, source=str(path))


def loads_armspec(text: str, *, source: str = "<string>") -> ArmSpec:
    try:
        doc = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ArmSpecError(f"{source}: not valid TOML ({exc})") from None
    _check_shape(doc, source)
    arm, data, recipe = doc["arm"], doc["data"], doc["recipe"]
    raw_overrides = recipe.get("overrides", {})
    if not isinstance(raw_overrides, dict):
        raise ArmSpecError(f"{source}: recipe.overrides must be a table")
    overrides = dict(raw_overrides)
    nulls = recipe.get("nulls", [])
    if not isinstance(nulls, list) or not all(isinstance(k, str) for k in nulls):
        raise ArmSpecError(f"{source}: recipe.nulls must be a list of dotted field names")
    seen: set[str] = set()
    for key in nulls:
        if key in seen:
            raise ArmSpecError(f"{source}: {key!r} is duplicated in recipe.nulls")
        seen.add(key)
        if key in overrides:
            raise ArmSpecError(f"{source}: {key!r} is both in recipe.nulls and in [recipe.overrides]")
        if not _DOTTED.fullmatch(key):
            raise ArmSpecError(f"{source}: recipe.nulls key {key!r} must be a dotted field name")
    _check_no_overlap(list(overrides) + nulls, source)
    for key in nulls:
        overrides[key] = None
    spec = ArmSpec(
        id=arm["id"], backbone=arm["backbone"], stage=arm["stage"], track=arm["track"],
        init=doc.get("init", {}).get("ckpt"), rowspace=data["rowspace"],
        bundle=doc.get("signal", {}).get("bundle"), norm=doc.get("norm", {}).get("artifact"),
        preset=recipe["preset"], seed=recipe["seed"], overrides=_frozen(overrides),
        paper_ref=arm.get("paper_ref"),
        data=_frozen({k: v for k, v in data.items() if k != "rowspace"}),
        paper=_frozen(doc.get("paper")),
    )
    _validate(spec, source)
    return spec


def _check_shape(doc: dict, source: str) -> None:
    unknown = sorted(set(doc) - set(_TABLES))
    if unknown:
        raise ArmSpecError(f"{source}: unknown table(s) {unknown}; allowed: {sorted(_TABLES)}")
    missing = [t for t in _REQUIRED_TABLES if t not in doc]
    if missing:
        raise ArmSpecError(f"{source}: missing table(s) {missing}")
    for name, table in doc.items():
        if not isinstance(table, dict):
            raise ArmSpecError(f"{source}: [{name}] must be a table, got {type(table).__name__}")
        required, optional = _TABLES[name]
        extra = sorted(set(table) - required - optional)
        if extra:
            raise ArmSpecError(f"{source}: [{name}] has unknown key(s) {extra}; "
                               f"allowed: {sorted(required | optional)}")
        lacking = sorted(required - set(table))
        if lacking:
            raise ArmSpecError(f"{source}: [{name}] is missing {lacking}")


def _check_value(v: object, where: str, bad) -> None:
    if v is None or isinstance(v, (str, bool, int)):
        return
    if isinstance(v, float):
        if not math.isfinite(v):
            raise bad(f"{where} must be finite")
        return
    if isinstance(v, list):
        for i, x in enumerate(v):
            _check_value(x, f"{where}[{i}]", bad)
        return
    if isinstance(v, dict):
        for k, x in v.items():
            _check_value(x, f"{where}.{k}", bad)
        return
    raise bad(f"{where} has unsupported type {type(v).__name__}")


def _valid_local(value: str) -> bool:
    """`local:<name>`; refuses `.` or `..` as a `/`-separated segment of `<name>`, and an empty
    segment (a leading, trailing or doubled `/`)."""
    if not _LOCAL.fullmatch(value):
        return False
    segments = value.split(":", 1)[1].split("/")
    return all(seg not in ("", ".", "..") for seg in segments)


def _check_no_overlap(keys: list[str], source: str) -> None:
    """No two declared [recipe.overrides]/nulls keys may name the same dotted field by two
    routes: a parent key set as a whole value (an inline table, or an unquoted dotted TOML key
    that silently nests, e.g. `model.action_dim = 5` written where `"model.action_dim" = 5` was
    meant) and a quoted dotted key naming one of its children have no defined precedence, so
    both being set at once is refused rather than guessed."""
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a != b and (b.startswith(a + ".") or a.startswith(b + ".")):
                raise ArmSpecError(f"{source}: recipe.overrides/nulls keys {a!r} and {b!r} overlap "
                                   f"(one is a dotted-path prefix of the other); set only the more "
                                   f"specific one")


def _validate(s: ArmSpec, source: str) -> None:
    def bad(what: str) -> ArmSpecError:
        return ArmSpecError(f"{source}: {what}")

    for field, value in (("arm.id", s.id), ("arm.backbone", s.backbone), ("arm.stage", s.stage),
                         ("arm.track", s.track), ("data.rowspace", s.rowspace), ("recipe.preset", s.preset)):
        if not isinstance(value, str):
            raise bad(f"{field} must be a string, got {type(value).__name__}")
    if not _NAME.fullmatch(s.id):
        raise bad(f"arm.id {s.id!r} must match {_NAME.pattern}")
    if s.backbone not in BACKBONES:
        raise bad(f"arm.backbone {s.backbone!r} is not one of {list(BACKBONES)}")
    if s.stage not in STAGES:
        raise bad(f"arm.stage {s.stage!r} is not one of {list(STAGES)}")
    if not _TRACK.fullmatch(s.track):
        raise bad(f"arm.track {s.track!r} must match {_TRACK.pattern}")
    try:
        with os.scandir(_TRACKS_DIR) as entries:
            has_track_file = any(e.name == f"{s.track}.toml" for e in entries)
    except OSError as exc:
        raise bad(f"cannot list conf/recipes/tracks/: {exc.__class__.__name__}: {exc}") from None
    if not has_track_file:
        raise bad(f"arm.track {s.track!r} has no file under conf/recipes/tracks/ "
                  f"(a track name that resembles a real one, e.g. a typo, is refused rather than "
                  f"silently unlocking local: references)")
    if s.paper_ref is not None and not (isinstance(s.paper_ref, str) and s.paper_ref):
        raise bad("arm.paper_ref must be a non-empty string")
    refs = (("init.ckpt", s.init, (_INIT,), "base:<name>, run:<run id> or <org>/<repo>:<path>#sha256:<64 hex>"),
            ("data.rowspace", s.rowspace, (_CONTENT, _REF_NAME),
             "<name>#sha256:<64 hex> or a published frame index's name"),
            ("signal.bundle", s.bundle, (_BUNDLE, _BUNDLE_NAME), "sha256:<64 hex> or name:<compiled labels' name>"),
            ("norm.artifact", s.norm, (_CONTENT, _REF_NAME), "<name>#sha256:<64 hex> or published statistics' name"))
    for field, value, patterns, form in refs:
        if value is None:
            continue
        if isinstance(value, str) and value.startswith("local:"):
            # The paper-track rule fires before local: syntax validation: on track paper, the
            # useful message is "never local:", not a verdict on whether the local: name is
            # well-formed (which does not matter there at all).
            if s.track == "paper":
                raise bad(f"{field} {value!r}: a paper run names every input by content, never local:")
            if not _valid_local(value):
                raise bad(f"{field} {value!r} is not a valid local: reference "
                          f"(local:<name>, no '.', '..' or empty segment)")
            continue
        if not (isinstance(value, str) and any(p.fullmatch(value) for p in patterns)):
            raise bad(f"{field} {value!r} must be {form}"
                      + ("" if s.track == "paper" else " or local:<name>"))
    if not _NAME.fullmatch(s.preset):
        raise bad(f"recipe.preset {s.preset!r} must match {_NAME.pattern}")
    if isinstance(s.seed, bool) or not isinstance(s.seed, int) or not 0 <= s.seed < 2**32:
        raise bad(f"recipe.seed must be an integer in [0, 2**32), got {s.seed!r}")
    for k, v in s.data.items():
        if not (isinstance(v, str) and v):
            raise bad(f"data.{k} must be a non-empty string")
    for k, v in s.overrides.items():
        if not _DOTTED.fullmatch(k):
            raise bad(f"recipe.overrides key {k!r} must be a dotted field name")
        _check_value(v, f"recipe.overrides.{k}", bad)
    if s.paper and s.track != "paper":
        raise bad(f"[paper] is only allowed on track 'paper', not {s.track!r}")
    for k, v in s.paper.items():
        want = _PAPER_TYPES[k]
        if isinstance(v, bool) or not isinstance(v, want):
            raise bad(f"paper.{k} must be a {want.__name__}")
        if want is list and not all(isinstance(x, str) for x in v):
            raise bad(f"paper.{k} must be a list of strings")
    for field, d in s.paper.get("differences", {}).items():
        if not _DOTTED.fullmatch(field):
            raise bad(f"paper.differences key {field!r} must be a dotted field name")
        if not isinstance(d, dict) or sorted(d) != sorted(_DIFFERENCE_KEYS) \
                or not all(isinstance(d[x], str) for x in d):
            raise bad(f"paper.differences.{field} must hold exactly the string keys {list(_DIFFERENCE_KEYS)}")
        for side in ("lock", "arm"):
            try:
                json.loads(d[side])
            except ValueError:
                raise bad(f"paper.differences.{field}.{side} must be a JSON value") from None


def as_dict(spec: ArmSpec) -> dict:
    """A plain, JSON-able copy (mapping proxies and nested TOML tables become dicts)."""
    out = {f.name: getattr(spec, f.name) for f in dataclasses.fields(spec)}
    for k in ("overrides", "data", "paper"):
        out[k] = json.loads(json.dumps(dict(out[k])))
    return out


def armspec_sha256(spec: ArmSpec) -> str:
    return sha256_bytes(dumps_canonical(as_dict(spec)))


def _k(key: str) -> str:
    return key if _BARE_KEY.fullmatch(key) else _v(key)


def _v(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_v(x) for x in v) + "]"
    if isinstance(v, Mapping):
        return "{ " + ", ".join(f"{_k(k)} = {_v(v[k])}" for k in sorted(v)) + " }" if v else "{}"
    raise ArmSpecError(f"cannot write a value of type {type(v).__name__}")


def dumps_armspec(spec: ArmSpec) -> str:
    """Canonical TOML for `spec`: fixed table order, sorted keys, one trailing newline."""
    lines = ["[arm]", f"id = {_v(spec.id)}", f"backbone = {_v(spec.backbone)}",
             f"stage = {_v(spec.stage)}", f"track = {_v(spec.track)}"]
    if spec.paper_ref is not None:
        lines.append(f"paper_ref = {_v(spec.paper_ref)}")
    if spec.init is not None:
        lines += ["", "[init]", f"ckpt = {_v(spec.init)}"]
    lines += ["", "[data]", f"rowspace = {_v(spec.rowspace)}"]
    lines += [f"{_k(k)} = {_v(spec.data[k])}" for k in sorted(spec.data)]
    if spec.bundle is not None:
        lines += ["", "[signal]", f"bundle = {_v(spec.bundle)}"]
    if spec.norm is not None:
        lines += ["", "[norm]", f"artifact = {_v(spec.norm)}"]
    lines += ["", "[recipe]", f"preset = {_v(spec.preset)}", f"seed = {_v(spec.seed)}"]
    nulls = sorted(k for k, v in spec.overrides.items() if v is None)
    if nulls:
        lines.append(f"nulls = {_v(nulls)}")
    values = sorted(k for k, v in spec.overrides.items() if v is not None)
    if values:
        lines += ["", "[recipe.overrides]"] + [f"{_k(k)} = {_v(spec.overrides[k])}" for k in values]
    if spec.paper:
        plain = {k: v for k, v in spec.paper.items() if k != "differences"}
        lines += ["", "[paper]"] + [f"{_k(k)} = {_v(plain[k])}" for k in sorted(plain)]
        diffs = spec.paper.get("differences", {})
        for field in sorted(diffs):
            lines += ["", f"[paper.differences.{_k(field)}]"]
            lines += [f"{k} = {_v(diffs[field][k])}" for k in _DIFFERENCE_KEYS]
    return "\n".join(lines) + "\n"
