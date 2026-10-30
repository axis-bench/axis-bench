"""The paper's signals as SignalBundles: compile every arm's bundle from the published corpus.

conf/signals/paper_bundles.toml declares one bundle per distinct training signal of the paper --
its method, row space, budget, planner seed and the locked files it exports -- and the arms that
trained on it; SmolVLA and X-VLA arms share bundles because they consumed the same files. `build`
compiles every entry, refuses any bundle whose legacy export is not byte-equal to its locked file
(there is no tolerance: a training line refuses any byte difference), and writes the bundles, their
row spaces under `rowspaces/` and `manifest.json` (arm -> bundle). It builds in a staging directory
beside `out` and moves the result into `out` only after every check has passed, the manifest last:
a build that fails leaves `out` as it was, and `check` reports what a killed one left. Inside the
published corpus tree it writes only its bundles/ (axis_qvla.signals.staging); a partial (`only`) build
is a local build and is refused inside that tree. `--corpus` must be the published per-task corpus
(qvla-v1-corpus/episodes): before compiling anything, `build` requires every row space
SIGNALS.lock.json records to have the corpus fingerprint, row count and trainable-row count
recorded there. `check` verifies a written tree against the declaration and the lock
(axis_qvla.signals.paper_check).

    qvla labels build-paper --pretraining-data <qvla-v1-corpus> --episodes <qvla-v1-corpus>/episodes --out <dir>

Offline tier: numpy, pandas.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import tempfile
import tomllib
import uuid

from axis_qvla.core.canon import dumps_canonical
from axis_qvla.labels.corpus import CorpusLabels
from axis_qvla.signals.bundle import (BundleError, bundle_id, export_legacy, load_bundle, load_rowspace, rowspace_id,
                                 write_bundle, write_rowspace)
from axis_qvla.signals.compile import compile_program
from axis_qvla.signals.imported import ImportedScores
from axis_qvla.signals.legacy import GENERATORS, REPO, SIGNALS_LOCK, legacy_target
from axis_qvla.signals.methods.awr import AwrChunk, AwrEpisode, AwrImported
from axis_qvla.signals.methods.bc import BC
from axis_qvla.signals.methods.cfg import CfgTags
from axis_qvla.signals.methods.filtering import DropTop50, Random50
from axis_qvla.signals.rowspace import paper_rowspaces
from axis_qvla.signals.schema import Budget
from axis_qvla.signals.staging import check_out, kind_of

CONF = REPO / "conf" / "signals" / "paper_bundles.toml"
NAMES_TABLE = REPO / "conf" / "signals" / "episode_body_names.json"
CONF_SCHEMA = "qvla.paper_bundles/1"
MANIFEST_SCHEMA = "qvla.signal_bundles/1"
METHODS = ("bc", "awr_chunk", "awr_episode", "awr_imported", "cfg", "drop_top50", "random50")
# A declared method -> the `method` its compiled program records (`spec()`), where the two differ.
PROGRAM_METHODS = {"cfg": "cfg_tags"}
REQUIRED = ("name", "method", "format", "rowspace", "budget", "seed", "planner", "exports", "arms")
# What a row space's record in SIGNALS.lock.json (`rowspaces`) pins about the corpus it was cut from.
CORPUS_FIELDS = ("corpus_fingerprint", "n_rows", "n_trainable")
PUBLISHED_CORPUS = "--episodes must be the published pretraining episodes (qvla-v1-corpus/episodes)"


class PaperBundleError(ValueError):
    """A paper bundle is mis-declared, or its legacy export is not its locked file."""


class DeclaredFieldMissing(KeyError):
    """A declared bundle entry, a row-space reference, or a lock record it names lacks the field
    this key names. Raised only where `build` reads one entry's own declaration and the lock
    records it points at, so a caller that means to report refused inputs by name, not a bug's
    traceback, can catch this type without also catching an unrelated KeyError."""


def sha256_of(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def load_conf(path=CONF) -> dict[str, dict]:
    doc = tomllib.loads(pathlib.Path(path).read_text())
    if doc.get("schema") != CONF_SCHEMA:
        raise PaperBundleError(f"{path}: schema {doc.get('schema')!r} is not {CONF_SCHEMA!r}")
    entries: dict[str, dict] = {}
    for e in doc["bundle"]:
        missing = [k for k in REQUIRED if k not in e]
        if missing:
            raise PaperBundleError(f"{path}: a [[bundle]] entry ({e.get('name', 'unnamed')}) has no "
                                   f"{', '.join(missing)}")
        if e["name"] in entries:
            raise PaperBundleError(f"{path}: [[bundle]] {e['name']} is declared twice")
        if e["method"] not in METHODS:
            raise PaperBundleError(f"{path}: {e['name']} has unknown method {e['method']!r}")
        entries[e["name"]] = e
    return entries


def check_corpus(rowspaces, lock_doc, corpus) -> None:
    """Refuse row spaces built from another corpus than the published one: each row space the lock
    records (`rowspaces` of SIGNALS.lock.json) must have the corpus fingerprint, row count and
    trainable-row count recorded there. Another corpus would address other frames, so its plans and
    weights would be another signal under the paper's names. A flat row space (`kind != "corpus"`,
    e.g. `heldout24_ft`) is not cut from `--corpus` at all and carries no corpus fingerprint to
    compare, so it is skipped here: blaming `--corpus` for a flat row space's fields would name the
    wrong input."""
    for name, want in sorted(lock_doc.get("rowspaces", {}).items()):
        rs = rowspaces.get(name)
        if rs is None:
            raise PaperBundleError(f"SIGNALS.lock.json records frame index {name}, which the build does not cut")
        if rs.kind != "corpus":
            continue
        diff = [f"{k} {getattr(rs, k)!r} (the checksum manifest's is {want[k]!r})" for k in CORPUS_FIELDS
                if k in want and getattr(rs, k) != want[k]]
        if diff:
            raise PaperBundleError(f"{corpus}: frame index {name} has {'; '.join(diff)}; {PUBLISHED_CORPUS}")


def awr_chunk_inputs(scorer: str, generators=GENERATORS) -> dict[str, str]:
    """The inputs an awr_chunk bundle records, as conf/signals/generators.json pins them: the corpus
    fingerprint and, for the approach-aware (`phase`) scorer, the digest of the names table that
    binds each episode's object."""
    pins = json.loads(pathlib.Path(generators).read_text())["inputs"]
    out = {"corpus_fingerprint": pins["corpus"]["corpus_fingerprint"]}
    if scorer == "phase":
        out["episode_body_names"] = pins["episode_names"]["reward_input_sha256"]
    return out


def _random50(entry, scores):
    return Random50(entry["granularity"], scores(entry["source"]), template_name=entry.get("template_name"),
                    exports=tuple(legacy_target(i, entry["format"]) for i in entry["exports"]))


def program_for(entry, entries, *, rowspaces, scores, corpus):
    """The program and labels of one declared bundle; `scores(lock_id) -> ImportedScores`."""
    exports = tuple(legacy_target(i, entry["format"]) for i in entry["exports"])
    method = entry["method"]
    if method == "bc":
        return BC(entry["planner"], exports=exports), None
    if method == "awr_chunk":
        phase = entry["scorer"] == "phase"
        pin = awr_chunk_inputs(entry["scorer"]).get("episode_body_names")
        labels = CorpusLabels(pathlib.Path(corpus), names_table=NAMES_TABLE if phase else None, names_sha256=pin)
        return AwrChunk(entry["scorer"], entry["planner"], entry["reducer"], phase_balanced=phase, exports=exports), labels
    if method == "awr_episode":
        return AwrEpisode(entry["planner"], entry["reducer"], exports=exports), scores(entry["source"])
    if method == "awr_imported":
        return AwrImported(entry["planner"], entry["reducer"], exports=exports), scores(entry["source"])
    if method == "cfg":
        return CfgTags(entry["granularity"], entry["planner"], tag_rows=rowspaces[entry["tag_rowspace"]],
                       exports=exports), scores(entry["source"])
    if method == "random50":
        return _random50(entry, scores), None
    control = None
    if entry.get("control"):
        if entry["control"] not in entries:
            raise PaperBundleError(f"{entry['name']}: control {entry['control']} is not declared")
        control = entries[entry["control"]]
    return (DropTop50(control=None if control is None else _random50(control, scores),
                      control_seed=0 if control is None else int(control["seed"]), exports=exports),
            scores(entry["source"]))


def verify_exports(bundle, fmt, rows_by_id, scratch) -> dict[str, str]:
    """Export the bundle's `fmt` files under `scratch` and require each to be its locked file, byte for byte."""
    got = {}
    by_path = {f.path: f for f in bundle.legacy if f.format == fmt}
    for p in export_legacy(bundle, fmt, scratch):
        lf = by_path[p.relative_to(scratch).as_posix()]
        row = rows_by_id.get(lf.lock_id)
        if row is None:
            raise PaperBundleError(f"{lf.path}: its id {lf.lock_id!r} is not a file of SIGNALS.lock.json")
        digest = sha256_of(p)
        if digest != row["sha256"]:
            raise PaperBundleError(f"{lf.lock_id}: the export ({digest[:16]}) is not the file SIGNALS.lock.json "
                                   f"lists ({row['sha256'][:16]})")
        got[lf.lock_id] = digest
    return got


def staging_dirs(out) -> list[pathlib.Path]:
    """The staging directories a build of `out` left beside it (one that was killed). A missing
    parent holds none; any other OSError listing it propagates -- an unreadable parent is not an
    empty one, and `Path.glob` would have answered "none" for it."""
    out = pathlib.Path(out)
    prefix = f".{out.name}.staging-"
    try:
        names = sorted(os.listdir(out.parent))
    except FileNotFoundError:
        return []
    return [out.parent / n for n in names if n.startswith(prefix)]


def _move_into(staging: pathlib.Path, out: pathlib.Path) -> None:
    """Put a finished build at `out`: one rename where `out` does not exist yet; otherwise entry by
    entry, keeping an entry already there that verifies (ids are content addresses, so it holds the
    same bytes), with the manifest last."""
    if kind_of(out) is None:
        os.rename(staging, out)
        return
    for sub, load in (("rowspaces", load_rowspace), ("", load_bundle)):
        (out / sub).mkdir(parents=True, exist_ok=True)
        for p in sorted((staging / sub).iterdir()):
            if p.name in ("rowspaces", "manifest.json"):
                continue
            dest = out / sub / p.name
            if kind_of(dest) == "dir":
                try:
                    load(dest)
                    continue
                except BundleError:
                    shutil.rmtree(dest)
            os.rename(p, dest)
    if kind_of(staging / "manifest.json") == "file":
        os.replace(staging / "manifest.json", out / "manifest.json")


def build(*, pack, corpus, out, only=None, log=print, conf=CONF, lock=SIGNALS_LOCK) -> dict:
    """Compile the declared bundles (all, or the names in `only`) and write them to `out`; returns
    the manifest. `conf` and `lock` are the declaration and the lock, the repository's by default."""
    pack, out, conf, lock = pathlib.Path(pack), pathlib.Path(out), pathlib.Path(conf), pathlib.Path(lock)
    if sha256_of(pack / "SIGNALS.lock.json") != sha256_of(lock):
        raise PaperBundleError(f"{pack}/SIGNALS.lock.json is not this repository's SIGNALS.lock.json")
    check_out(out, "bundles", partial=bool(only))
    lock_doc = json.loads(lock.read_text())
    entries, rows_by_id = load_conf(conf), {r["id"]: r for r in lock_doc["files"]}
    unknown = sorted(set(only or ()) - set(entries))
    if unknown:
        raise PaperBundleError(f"no compiled labels are declared as {', '.join(unknown)}")
    rowspaces = paper_rowspaces(pack, corpus)
    check_corpus(rowspaces, lock_doc, corpus)
    cache: dict[str, ImportedScores] = {}

    def scores(lock_id: str) -> ImportedScores:
        if lock_id not in cache:
            row = rows_by_id[lock_id]
            cache[lock_id] = ImportedScores.from_json(pack / row["path"], name=lock_id, expect_sha256=row["sha256"])
        return cache[lock_id]

    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.parent / f".{out.name}.staging-{uuid.uuid4().hex[:12]}"
    staging.mkdir()
    try:
        for rs in rowspaces.values():
            write_rowspace(rs, staging / "rowspaces")
        built: dict[str, dict] = {}
        for name, e in entries.items():
            if only and name not in only:
                continue
            try:
                program, labels = program_for(e, entries, rowspaces=rowspaces, scores=scores, corpus=corpus)
                entry_rs, budget, seed = rowspaces[e["rowspace"]], Budget(*e["budget"]), int(e["seed"])
            except KeyError as exc:
                # `program_for` and the three lookups above read only `name`'s declared fields,
                # the row space it names, and the lock record a `source` names -- never the
                # method's own data. A missing key means one of those, not a bug elsewhere, so it
                # is named and not a traceback (axis_qvla.signals.cli). `compile_program` below runs
                # the method's own scorer/selector/planner/weighter/conditioner, which reads the
                # corpus and imported scores, not the declaration; a KeyError there is a bug and
                # is not caught here, so it (like `verify_exports`'s, further down) keeps its own
                # traceback.
                raise DeclaredFieldMissing(*exc.args) from exc
            bundle, controls = compile_program(program, labels, entry_rs, budget, seed)
            with tempfile.TemporaryDirectory(dir=staging) as scratch:
                exports = verify_exports(bundle, e["format"], rows_by_id, pathlib.Path(scratch))
            built[name] = {"id": write_bundle(bundle, staging), "method": e["method"], "format": e["format"],
                           "rowspace": e["rowspace"], "exports": exports, "controls": [bundle_id(c) for c in controls],
                           "paper_arm": bool(e.get("paper_arm", True)), "arms": sorted(e["arms"])}
            log(f"{name}: {built[name]['id'][:16]}  {', '.join(sorted(exports))}")
        for name, b in built.items():
            ctl = entries[name].get("control")
            if ctl in built and b["controls"] != [built[ctl]["id"]]:
                raise PaperBundleError(f"{name}: its random control is not the declared {ctl}")
        man = {"schema": MANIFEST_SCHEMA, "signals_lock_sha256": sha256_of(lock), "conf_sha256": sha256_of(conf),
               "rowspaces": {n: rowspace_id(rs) for n, rs in rowspaces.items()},
               "bundles": {n: {k: v for k, v in b.items() if k != "arms"} for n, b in built.items()},
               "arms": {a: n for n, b in built.items() for a in b["arms"]}}
        if not only:
            (staging / "manifest.json").write_bytes(dumps_canonical(man))
        _move_into(staging, out)
    except BaseException:
        # The build's own error is the one to report; a staging directory this cannot remove stays
        # beside `out`, where `check` names it.
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if kind_of(staging) is not None:        # what a merge into an existing `out` did not move
        shutil.rmtree(staging)
    return man


def check(bundles, *, scratch=None, conf=CONF, lock=SIGNALS_LOCK) -> list[str]:
    """Problems with a written compiled-labels tree, checked against the declaration `conf` and the
    checksum manifest (axis_qvla.signals.paper_check.check_tree)."""
    from axis_qvla.signals.paper_check import check_tree

    return check_tree(bundles, scratch=scratch, conf=conf, lock=lock)
