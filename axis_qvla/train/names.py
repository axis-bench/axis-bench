"""Run-spec inputs named by name, resolved to the content ids every other `qvla train` step reads.

Why: a run spec should say "the paper's pretraining frame index" and "the labels I compiled as
example.keep_top", not carry two 64-hex digests that someone keeps in sync by hand. Names are
resolved here, once, right after the spec is read. `qvla train` records each name with the id it
resolved to (names.json beside the rendered files). Binding then verifies the ids exactly as for a
spec that wrote them out: the labels' content against their directory, their frame index against the
run's, the statistics' sha256 against the file.

  frame index, statistics  the release's ids (PUBLISHED_*: the ones every paper run spec names)
  compiled labels          the release's (PUBLISHED_LABELS: the fine-tune's row order the paper's
                           SmolVLA / X-VLA fine-tunes read), else the name the compile recorded
                           (<cache>/bundles/names/<name>.json), else the one compiled-labels directory
                           under <cache>/bundles whose program is that method
  this machine's own       a name a step recorded for what it built (record_local_name: the real-world
  frame index, statistics  line's frame indexes and norm statistics, <cache>/bundles/names/<kind>/<name>.json)

Core tier: stdlib only.
"""
from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import re

from axis_qvla.core.paths import Paths
from axis_qvla.train import ids
from axis_qvla.train.armspec import ArmSpec

PUBLISHED_ROWSPACES = {
    "libero5k_v2_d8": "2da88145ce3a175dda17cd99bcec9685acfb9fa611e20a284688fe75b76810c9",
    "libero5k_v2_d8.sub1000": "4bdaf94a64b992196f9f02ad01216d5b441013288ef662dc1b5fb00c6e0e4006",
    "libero5k_v2_d8.sub2500": "f451a87072baf26b87313f1e4d597c800091c059177156c36ec6a031ad8a64a9",
    "heldout24_ft": "813de4957527d3394cadc9e564f9718769c82415c9091f4971ba5e47641960f9",
}
PUBLISHED_NORMS = {
    "libero5k_v2_d8": "2fad5134b75426dea10eded094b192a455eb9e557a43d19033dd10c374aebfa6",
    "heldout24_qual_v2": "b1808cdd0c7000f30b93eff8246e814e5e90e1a8dbb33a83be47936500fa6454",
}
# The released compiled labels a run spec names by name: the held-out fine-tune's row order the paper's
# seed-0 SmolVLA / X-VLA fine-tunes read (SIGNALS.lock.json lerobot.ft_rows_all), so a generated
# fine-tune spec names it as `name:lerobot.ft_rows_all`. A release name wins over a method's own.
PUBLISHED_LABELS = {
    "lerobot.ft_rows_all": "a83c563e831e3f157532a46f4d4c997f25dc07959e73d59ef97b153212fe548e",
}
LABEL_PREFIX = "name:"
NAMES_DIR = "names"
NAMES_FILE = "names.json"
NAME_SCHEMA = "qvla.labels_name/1"
LOCAL_NAME_SCHEMA = "qvla.local_name/1"
LOCAL_KINDS = {"rowspace": "frame index", "norm": "normalisation statistics"}
_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class NameResolutionError(ValueError):
    """A name in a run spec names nothing, or more than one thing, on this machine."""


def _published(kind: str, table: dict[str, str], ref: str, paths: Paths | None = None) -> str:
    if "#" in ref or ref.startswith("local:"):
        return ref
    if ref in table:
        return f"{ref}#sha256:{table[ref]}"
    local = _local_name(paths, kind, ref) if paths is not None else None
    if local is not None:
        return local
    raise NameResolutionError(f"{LOCAL_KINDS[kind]} {ref!r} is not a published one ({', '.join(sorted(table))})"
                              + (" nor one this machine recorded" if paths is not None else "")
                              + "; name any other by content, <name>#sha256:<id>")


def resolve_rowspace(ref: str, paths: Paths | None = None) -> str:
    """A frame index by name: the release's, else (given `paths`) one this machine recorded."""
    return _published("rowspace", PUBLISHED_ROWSPACES, ref, paths)


def resolve_norm(ref: str, paths: Paths | None = None) -> str:
    """Normalisation statistics by name: the release's, else (given `paths`) ones this machine recorded."""
    return _published("norm", PUBLISHED_NORMS, ref, paths)


def _local_path(root, kind: str, name: str) -> pathlib.Path:
    if not _NAME.fullmatch(name):        # the name becomes a file name
        raise NameResolutionError(f"{name!r} is not a {LOCAL_KINDS[kind]} name ({_NAME.pattern})")
    return pathlib.Path(root) / NAMES_DIR / kind / f"{name}.json"


def record_local_name(root, kind: str, ref: str) -> pathlib.Path:
    """Make the name of `ref` (`<name>#sha256:<id>`, a frame index or normalisation statistics this
    machine built) resolve to it under `root` (<cache>/bundles): a run spec then names it by name,
    and `qvla train` records the id it resolved to in the run's names.json. The latest record wins,
    as for compiled labels. A release name is refused: it always resolves to the release's."""
    if kind not in LOCAL_KINDS:
        raise NameResolutionError(f"{kind!r} is not one of {', '.join(LOCAL_KINDS)}")
    name, _ = ids.content_id(ref)
    if name in (PUBLISHED_ROWSPACES if kind == "rowspace" else PUBLISHED_NORMS):
        raise NameResolutionError(f"{name!r} names the release's {LOCAL_KINDS[kind]}; it cannot be recorded")
    dest = _local_path(root, kind, name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".{name}.json.tmp-{os.getpid()}"
    tmp.write_text(json.dumps({"schema": LOCAL_NAME_SCHEMA, "kind": kind, "name": name, "ref": ref},
                              sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(dest)
    return dest


def _local_name(paths: Paths, kind: str, name: str) -> str | None:
    path = _local_path(ids.local_bundles_root(paths), kind, name)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, NotADirectoryError):
        return None
    except (OSError, ValueError) as exc:
        raise NameResolutionError(f"cannot read {path} ({exc.__class__.__name__})") from None
    ref = doc.get("ref") if isinstance(doc, dict) else None
    try:
        ok = isinstance(ref, str) and ids.content_id(ref)[0] == name
    except ids.IdError:
        ok = False
    if not ok:
        raise NameResolutionError(f"{path} records no {LOCAL_KINDS[kind]} named {name!r}")
    return ref


def published_name(ref: str, table: dict[str, str]) -> str:
    """`<name>#sha256:<id>` -> `<name>` when that id is the release's for the name; else `ref`."""
    name, _, digest = ref.partition("#sha256:")
    return name if table.get(name) == digest else ref


def published_labels_name(ref: str) -> str:
    """`sha256:<id>` -> `name:<name>` when that id is released compiled labels (PUBLISHED_LABELS); else `ref`."""
    hit = [n for n, d in PUBLISHED_LABELS.items() if ref == f"sha256:{d}"]
    return f"{LABEL_PREFIX}{hit[0]}" if hit else ref


def is_published_labels(ref: str | None) -> bool:
    """`ref` names released compiled labels by name (`name:<a PUBLISHED_LABELS name>`)."""
    return bool(ref) and ref.startswith(LABEL_PREFIX) and ref[len(LABEL_PREFIX):] in PUBLISHED_LABELS


def check_labels_name(name: str) -> None:
    """Refuse a name `record_labels_name` would refuse, before a compile spends its time: one that is
    not a file name of the form a run spec names, or that names the release's compiled labels."""
    if not _NAME.fullmatch(name):
        raise NameResolutionError(f"{name!r} is not a compiled-labels name ({_NAME.pattern})")
    if name in PUBLISHED_LABELS:
        raise NameResolutionError(f"{name!r} names the release's compiled labels; name your method otherwise")


def record_labels_name(root, name: str, bundle_id: str) -> pathlib.Path:
    """Make `name` resolve to compiled labels `bundle_id` (64 hex) under `root`. A compile calls this
    after writing them, so its latest compile is the one a run spec naming the method trains. A
    released name is refused: `resolve_labels` answers it from PUBLISHED_LABELS before any record, so
    a record under it would never be read, and the compile that wrote it would train the release's.
    The record keeps the method the labels' manifest names when it is not `name` (the real-world line
    records its built-in programs as `realrobot.<run>`), and resolving checks the directory against it."""
    check_labels_name(name)
    if not (isinstance(bundle_id, str) and _HEX64.fullmatch(bundle_id)):
        raise NameResolutionError(f"{bundle_id!r} is not a compiled-labels id (64 lowercase hex characters)")
    d = pathlib.Path(root) / NAMES_DIR
    d.mkdir(parents=True, exist_ok=True)
    dest, tmp = d / f"{name}.json", d / f".{name}.json.tmp-{os.getpid()}"
    doc = {"schema": NAME_SCHEMA, "name": name, "id": f"sha256:{bundle_id}"}
    method = _method(pathlib.Path(root) / bundle_id)
    if method is not None and method != name:
        doc["method"] = method
    tmp.write_text(json.dumps(doc, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(dest)
    return dest


def _method(directory: pathlib.Path) -> str | None:
    """The method a compiled-labels directory's manifest records; None when it holds no manifest."""
    path = directory / "manifest.json"
    try:
        man = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, NotADirectoryError):
        return None
    except (OSError, ValueError) as exc:
        raise NameResolutionError(f"cannot read {path} ({exc.__class__.__name__})") from None
    prog = man.get("program") if isinstance(man, dict) else None
    return prog.get("method") if isinstance(prog, dict) else None


def resolve_labels(paths: Paths, ref: str) -> str:
    """`name:<method>` -> `sha256:<id>` of this machine's compiled labels of that method; any other
    form unchanged."""
    if not ref.startswith(LABEL_PREFIX):
        return ref
    name, root = ref[len(LABEL_PREFIX):], ids.local_bundles_root(paths)
    if name in PUBLISHED_LABELS:
        return f"sha256:{PUBLISHED_LABELS[name]}"
    if not _NAME.fullmatch(name):        # the name becomes a file name under `root`
        raise NameResolutionError(f"{ref!r}: {name!r} is not a compiled-labels name ({_NAME.pattern})")
    alias = root / NAMES_DIR / f"{name}.json"
    try:
        doc = json.loads(alias.read_text(encoding="utf-8"))
    except (FileNotFoundError, NotADirectoryError):
        doc = None
    except (OSError, ValueError) as exc:
        raise NameResolutionError(f"cannot read {alias} ({exc.__class__.__name__})") from None
    if doc is not None:
        bid = doc.get("id") if isinstance(doc, dict) else None
        if not (isinstance(bid, str) and _ID.fullmatch(bid)):
            raise NameResolutionError(f"{alias} names no compiled-labels id")
        if _method(root / bid[len("sha256:"):]) != doc.get("method", name):
            raise NameResolutionError(f"{alias} names compiled labels that are not under {root}: "
                                      f"compile {name} again")
        return bid
    try:
        with os.scandir(root) as it:
            dirs = [pathlib.Path(e.path) for e in it if len(e.name) == 64 and e.is_dir()]
    except FileNotFoundError:
        dirs = []
    except OSError as exc:
        raise NameResolutionError(f"cannot list {root} ({exc.__class__.__name__})") from None
    hits = sorted(d.name for d in dirs if _method(d) == name)
    if len(hits) == 1:
        return f"sha256:{hits[0]}"
    if not hits:
        raise NameResolutionError(f"no compiled labels named {name!r} under {root}: compile your method first "
                                  f"(experiments/examples/keep_top.py shows how) with its output there")
    raise NameResolutionError(f"{len(hits)} compiled labels under {root} are named {name!r} and none is recorded "
                              f"as the current one: compile {name} again (that records it), or remove the others")


def resolve_names(spec: ArmSpec, paths: Paths) -> tuple[ArmSpec, dict[str, dict[str, str]]]:
    """`spec` with every input given by name resolved to its content id, and {field: {name, id}} for
    each one that was resolved (empty when the spec names everything by content)."""
    out = {"rowspace": resolve_rowspace(spec.rowspace, paths),
           "norm": None if spec.norm is None else resolve_norm(spec.norm, paths),
           "bundle": None if spec.bundle is None else resolve_labels(paths, spec.bundle)}
    out = {f: v for f, v in out.items() if v is not None}
    names = {f: {"name": getattr(spec, f), "id": v} for f, v in out.items() if v != getattr(spec, f)}
    return dataclasses.replace(spec, **out), names
