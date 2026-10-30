"""Logical ids of training inputs, resolved to paths on this machine through `Paths` alone.

Every input an ArmSpec names is location-free: a published checkpoint
(`<org>/<repo>:<path>#sha256:<digest>`), a base model (`base:<name>`, conf/recipes/bases.toml), a
local run (`run:<arm id>`), a bundle (`sha256:<id>`), a row space or a normalisation artifact
(`<name>#sha256:<digest>`). Where each lives is the data layout (axis_qvla.data.layout), the one
module `qvla data get` writes by and every reader opens by:

  published data   `default_dest` / `checkpoint_dest` / `norm_stats_dest`: a dataset repository
                   (corpus, bench) at <corpus>/<repo>, a model repository (checkpoints, base
                   weights) at <cache>/checkpoints/<repo>, each file at its repository path;
  local data       `local_dest`: what this machine computed itself -- bundles and row spaces
                   compiled at bind time (<cache>/bundles), the smoke fixture, norm statistics it
                   computed, merged LeRobot copies it built (<cache>/derived) -- never inside a
                   mirrored repository, whose bytes the release lock pins.

A bundle or a stored row space may be either, so `bundle_dirs` / `rowspace_record_dirs` give both
places and binding (the pre-launch step a run goes through before training starts) takes the one
that exists, published first. This module never looks at the filesystem and never reads the
environment.

Core tier: stdlib only.
"""
from __future__ import annotations

import pathlib
import re

from axis_qvla.core.paths import Paths
from axis_qvla.data.layout import (CORPUS, CORPUS_BUNDLES, base_weights_dest, checkpoint_dest, default_dest, local_dest,
                              norm_stats_dest)
from axis_qvla.train.recipes import Base, load_bases, load_rowspaces

# The directory a smoke's own files go to, beside the real run's (`rendered_dir`).
RENDERED_SMOKE = "rendered-smoke"
# Normalisation statistics a published repository carries; any other name is computed locally.
PUBLISHED_NORMS = frozenset({"heldout24_qual_v2", "libero5k_v2_d8"})
_PUBLISHED = re.compile(r"^(?P<org>[^/:#]+)/(?P<repo>[^/:#]+):(?P<path>[^#]+)#sha256:(?P<sha>[0-9a-f]{64})$")
_CONTENT = re.compile(r"^(?P<name>[^#]+)#sha256:(?P<sha>[0-9a-f]{64})$")
_BUNDLE = re.compile(r"^sha256:(?P<sha>[0-9a-f]{64})$")


class IdError(ValueError):
    """A logical id has the wrong form or names a table entry that does not exist."""


def content_id(cid: str) -> tuple[str, str]:
    """`<name>#sha256:<hex>` -> (name, hex)."""
    m = _CONTENT.fullmatch(cid or "")
    if not m:
        raise IdError(f"{cid!r} is not <name>#sha256:<64 hex>")
    return m["name"], m["sha"]


def published(ckpt: str) -> tuple[str, str, str, str]:
    """`<org>/<repo>:<path>#sha256:<hex>` -> (org, repo, path, hex)."""
    m = _PUBLISHED.fullmatch(ckpt or "")
    if not m:
        raise IdError(f"{ckpt!r} is not <org>/<repo>:<path>#sha256:<64 hex>")
    return m["org"], m["repo"], m["path"], m["sha"]


def checkpoint_dir(paths: Paths, ckpt: str) -> pathlib.Path:
    org, repo, path, _ = published(ckpt)
    return checkpoint_dest(f"{org}/{repo}:{path}", paths)


def local_derived(paths: Paths, name: str) -> pathlib.Path:
    """This machine's own derived data `name` (<cache>/derived/<name>)."""
    return local_dest("derived", name, paths)


def local_bundles_root(paths: Paths) -> pathlib.Path:
    """Where bundles and row spaces compiled on this machine are written (<cache>/bundles)."""
    return local_dest("bundles", ".", paths)


def base(init: str, backbone: str) -> Base:
    name = init.removeprefix("base:")
    bases = load_bases()
    if name not in bases:
        raise IdError(f"{init!r}: no base {name!r} in conf/recipes/bases.toml")
    if bases[name].backbone != backbone:
        raise IdError(f"{init!r} is a {bases[name].backbone} base, not a {backbone} one")
    return bases[name]


def base_weights_root(paths: Paths) -> pathlib.Path:
    """The local copy of the base weights (`qvla data get base-weights`)."""
    return base_weights_dest(paths)


def openpi_home(paths: Paths) -> pathlib.Path:
    """OPENPI_DATA_HOME for a run: openpi's download cache inside the base-weights copy, so the
    tokenizer (and any base) resolves offline."""
    return base_weights_root(paths) / "openpi"


def hf_home(paths: Paths) -> pathlib.Path:
    """HF_HOME for a LeRobot run: the Hugging Face cache inside the base-weights copy."""
    return base_weights_root(paths) / "hf"


def run_root(paths: Paths, arm_id: str) -> pathlib.Path:
    return paths.runs / arm_id


def rendered_dir(paths: Paths, arm_id: str, *, smoke: bool = False) -> pathlib.Path:
    """Where `qvla train` keeps one mode's own files (rendered config, run record, binding, exported
    signal files): `<runs>/<arm id>/` for a real run, the sibling `<runs>/<arm id>/rendered-smoke/`
    for a smoke, so the two modes never share a file and a smoke never writes into the directory a
    backbone owns (`lerobot_output_dir`)."""
    run = run_root(paths, arm_id)
    return run / RENDERED_SMOKE if smoke else run


def lerobot_output_dir(paths: Paths, arm_id: str, *, smoke: bool = False) -> pathlib.Path:
    """Where the backbone writes a run's own checkpoints: a LeRobot run's `output_dir`, and the
    directory openpi's checkpoint convention (`<runs>/<name>/<exp_name>`, exp_name "train" or
    "smoke") resolves to. `qvla train` itself never writes into this directory and never creates
    it: LeRobot refuses to train into an existing `output_dir` when resume is false, openpi refuses
    an existing real-run checkpoint directory unless it resumes, and openpi's smoke mode deletes and
    recreates its checkpoint directory on every launch, so a rendered config sitting here would be
    refused by the first two and destroyed by the last. A smoke's own files go to the sibling directory
    `rendered_dir` names instead (`<runs>/<arm id>/rendered-smoke/`), which no backbone ever writes
    into."""
    return run_root(paths, arm_id) / ("smoke" if smoke else "train")


def openpi_init_params(paths: Paths, init: str) -> str:
    """The params directory a pi0.5 run starts from."""
    if init.startswith("base:"):
        return str(base_weights_root(paths) / base(init, "pi05-openpi").local)
    if init.startswith("run:"):
        return str(run_root(paths, init[len("run:"):]) / "train" / "final" / "params")
    return str(checkpoint_dir(paths, init) / "params")


def lerobot_init(paths: Paths, init: str, backbone: str) -> str | None:
    """policy.pretrained_path for a LeRobot run; None when the policy loads its own backbone. A base
    is named by its Hub id, which the run resolves offline from `hf_home`."""
    if init.startswith("base:"):
        return base(init, backbone).local or None
    if init.startswith("run:"):
        return str(run_root(paths, init[len("run:"):]) / "train" / "checkpoints" / "last" / "pretrained_model")
    return str(checkpoint_dir(paths, init) / "pretrained_model")


def bundle_dirs(paths: Paths, bundle: str) -> tuple[pathlib.Path, pathlib.Path]:
    """(published, local) directories of bundle `sha256:<id>`; binding takes the one that exists."""
    m = _BUNDLE.fullmatch(bundle or "")
    if not m:
        raise IdError(f"{bundle!r} is not sha256:<64 hex>")
    return default_dest(CORPUS, paths, f"{CORPUS_BUNDLES}/{m['sha']}"), local_dest("bundles", m["sha"], paths)


def rowspace_record_dirs(paths: Paths, rowspace: str) -> tuple[pathlib.Path, pathlib.Path]:
    """(published, local) directories of the stored row space `<name>#sha256:<id>`."""
    _, digest = content_id(rowspace)
    return (default_dest(CORPUS, paths, f"{CORPUS_BUNDLES}/rowspaces/{digest}"),
            local_dest("bundles", f"rowspaces/{digest}", paths))


def norm_dir(paths: Paths, norm: str) -> pathlib.Path:
    """The directory holding `<asset_id>/norm_stats.json` for this normalisation artifact: where
    the release publishes it (`norm_stats_dest`) when it does, else this machine's own."""
    name, _ = content_id(norm)
    if name in PUBLISHED_NORMS:
        return norm_stats_dest(name, paths)
    return local_derived(paths, f"norm/{name}")


def rowspace_dirs(paths: Paths, rowspace: str) -> dict[str, pathlib.Path]:
    name, _ = content_id(rowspace)
    layouts = load_rowspaces()
    if name not in layouts:
        raise IdError(f"frame index {name!r} is not in conf/recipes/rowspaces.toml")
    lay = layouts[name]

    def at(rel: str, local: bool) -> pathlib.Path:
        return local_derived(paths, rel) if local else default_dest(lay.repo, paths, rel)

    local = lay.repo == "local"
    return {"roots": at(lay.roots, local), "ranges": at(lay.ranges, local),
            "corpus_base": at(lay.corpus_base, local),
            "lerobot_dataset": at(lay.lerobot_dataset, local or lay.lerobot_local)}
