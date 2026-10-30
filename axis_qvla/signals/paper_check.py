"""Verify a written bundles tree against the paper's declaration and the lock, naming every problem.

A consumer resolves an arm to `bundles[arms[arm]]["id"]`, and the seed replicates of one signal
export the same locked files, so a manifest that maps an arm to another seed, or a bundle record
that points at another bundle's directory, still passes every check made against the lock alone.
`check_tree` therefore checks the tree against the declaration (conf/signals/paper_bundles.toml).
- The manifest records the declaration's digest, its bundle names and its arm -> bundle map.
- Every loaded bundle has the seed, budget, row space, method, planner, granularity, scorer, loss
  reducer and exported lock ids its entry declares, and the controls its entry names. A bundle
  whose entry names a `source` read that locked file with the lock's digest; an awr_chunk bundle,
  which reads the corpus instead, recorded exactly the corpus fingerprint (and, for the
  approach-aware scorer, the names-table digest) that conf/signals/generators.json pins.
- Every locked file re-exports byte for byte, with the digest the manifest records.
- The tree holds nothing the manifest does not name, and no staging directory of a killed build is
  left beside it.
A malformed manifest -- a missing field, a bundle or row-space id that is not a content address, an
arm naming an absent bundle, an unknown row space -- is reported by name, never raised.

Offline tier: numpy.
"""
from __future__ import annotations

import json
import pathlib
import re
import tempfile

from axis_qvla.signals.bundle import load_bundle, load_rowspace, rowspace_id
from axis_qvla.signals.legacy import GENERATORS
from axis_qvla.signals.paper import (MANIFEST_SCHEMA, PROGRAM_METHODS, awr_chunk_inputs, kind_of, load_conf, sha256_of,
                                staging_dirs, verify_exports)

BUNDLE_ID = re.compile(r"[0-9a-f]{64}")
RECORD = {"id": str, "method": str, "format": str, "rowspace": str, "exports": dict, "controls": list,
          "paper_arm": bool}


def _short(digest) -> str:
    return "nothing" if digest is None else str(digest)[:16]


def _digests(inputs) -> dict[str, str]:
    return {k: _short(v) for k, v in sorted(inputs.items())}


def _shape(man) -> list[str]:
    if not isinstance(man, dict) or man.get("schema") != MANIFEST_SCHEMA:
        got = man.get("schema") if isinstance(man, dict) else type(man).__name__
        return [f"manifest schema {got!r} is not {MANIFEST_SCHEMA!r}"]
    problems = [f"manifest {k} is missing or not a string" for k in ("signals_lock_sha256", "conf_sha256")
                if not isinstance(man.get(k), str)]
    for k in ("rowspaces", "arms"):
        v = man.get(k)
        if not (isinstance(v, dict) and all(isinstance(x, str) for x in v.values())):
            problems.append(f"manifest {k} is not a mapping of names to strings")
    if not isinstance(man.get("bundles"), dict):
        problems.append("manifest bundles is not a mapping")
    return problems


def _record(name, b) -> list[str]:
    """What is wrong with one manifest record's shape. Its id and its controls' ids must be
    compiled-labels ids (content addresses: 64 lowercase hex characters) before any of them is
    joined onto the tree."""
    if not isinstance(b, dict):
        return [f"{name}: its manifest record is not a mapping"]
    bad = sorted(k for k, t in RECORD.items() if not isinstance(b.get(k), t))
    if not bad and not all(isinstance(v, str) for v in [*b["exports"].values(), *b["controls"]]):
        bad = ["exports", "controls"]
    if bad:
        return [f"{name}: manifest record fields {bad} are missing or mistyped"]
    ids = [("id", b["id"]), *(("control", c) for c in b["controls"])]
    return [f"{name}: its {what} {v!r} is not a compiled-labels id (64 lowercase hex characters)"
            for what, v in ids if not BUNDLE_ID.fullmatch(v)]


def _names_and_arms(man, entries) -> list[str]:
    problems = [f"{n}: declared, but the manifest has no such compiled labels"
                for n in sorted(set(entries) - set(man["bundles"]))]
    problems += [f"{n}: in the manifest, but no such compiled labels are declared"
                 for n in sorted(set(man["bundles"]) - set(entries))]
    want = {a: n for n, e in entries.items() for a in e["arms"]}
    for arm in sorted(set(want) | set(man["arms"])):
        got = man["arms"].get(arm)
        absent = got is not None and got not in man["bundles"]
        if got != want.get(arm):
            problems.append(f"run {arm}: the manifest maps it to {got}"
                            f"{' (not compiled labels of the manifest)' * absent}, "
                            f"the declaration to {want.get(arm)}")
        elif absent:
            problems.append(f"run {arm}: its compiled labels {got} are not in the manifest")
    return problems


def _rowspaces(bundles, man, entries) -> list[str]:
    problems = []
    for name, rid in man["rowspaces"].items():
        if not BUNDLE_ID.fullmatch(rid):        # a content address, as a compiled-labels id is: never joined unchecked
            problems.append(f"frame index {name}: its id {rid!r} is not a frame-index id (64 lowercase hex characters)")
            continue
        try:
            rs = load_rowspace(bundles / "rowspaces" / rid)
        except (ValueError, OSError) as exc:
            problems.append(f"frame index {name}: {exc}")
            continue
        if rs.name != name:
            problems.append(f"frame index {name}: its record is named {rs.name}")
    declared = {e["rowspace"] for e in entries.values()} | {e["tag_rowspace"] for e in entries.values()
                                                             if "tag_rowspace" in e}
    problems += [f"frame index {n}: declared, but the manifest has no such frame index"
                 for n in sorted(declared - set(man["rowspaces"]))]
    return problems


def _declared(name, e, b, bundle, man, rows_by_id, generators=GENERATORS) -> list[str]:
    """What compiled labels compiled from entry `e` record about themselves, against the entry (and,
    for an awr_chunk compiled-labels entry, its inputs against the generators.json pins)."""
    problems = []

    def differ(what, got, want):
        if got != want:
            problems.append(f"{name}: {what} is {got!r}, the declaration says {want!r}")

    for key in ("method", "format", "rowspace"):
        differ(f"the manifest's {key}", b[key], e[key])
    differ("the manifest's paper_arm", b["paper_arm"], bool(e.get("paper_arm", True)))
    params = bundle.program.get("params", {})
    compiled = PROGRAM_METHODS.get(b["method"], b["method"])      # against the record, so a declaration
    if bundle.program.get("method") != compiled:                  # edit is reported once, above
        problems.append(f"{name}: the program's method is {bundle.program.get('method')!r}, and the manifest's "
                        f"{b['method']!r} compiles to {compiled!r}")
    differ("the seed", int(bundle.seed), int(e["seed"]))
    differ("the budget", [bundle.budget.steps, bundle.budget.global_batch], list(e["budget"]))
    differ("the frame index", rowspace_id(bundle.rowspace), man["rowspaces"].get(e["rowspace"]))
    differ("the planner", params.get("planner"), e["planner"])
    for key in ("granularity", "scorer"):
        if key in e:
            differ(f"the {key}", params.get(key), e[key])
    if "reducer" in e:
        differ("the loss reducer", bundle.reducer, e["reducer"])
    if "tag_rowspace" in e:
        differ("the tag frame index", params.get("tag_rowspace"), man["rowspaces"].get(e["tag_rowspace"]))
    if "source" in e:
        want = (rows_by_id.get(e["source"]) or {}).get("sha256")
        got = params.get("template_sha256") if e["method"] == "random50" else bundle.inputs.get(e["source"])
        if got != want:
            problems.append(f"{name}: it read {e['source']} as {_short(got)}, "
                            f"the checksum manifest's is {_short(want)}")
    if e["method"] == "awr_chunk":
        got, want = dict(bundle.inputs), awr_chunk_inputs(e.get("scorer"), generators)
        if got != want:
            problems.append(f"{name}: its inputs are {_digests(got)}, generators.json pins {_digests(want)}")
    exports = sorted(f.lock_id for f in bundle.legacy)            # every format: the format is checked above
    if exports != sorted(e["exports"]):
        problems.append(f"{name}: it exports {exports}, the declaration says {sorted(e['exports'])}")
    return problems


def _bundle(name, b, e, man, bundles, rows_by_id, scratch, generators) -> list[str]:
    problems = _record(name, b)
    if problems or e is None:
        return problems
    try:
        bundle = load_bundle(bundles / b["id"])
    except (ValueError, OSError) as exc:
        return [f"{name}: {exc}"]
    problems = _declared(name, e, b, bundle, man, rows_by_id, generators)
    if list(bundle.controls) != b["controls"]:
        problems.append(f"{name}: the manifest records controls {[_short(c) for c in b['controls']]}, "
                        f"the compiled labels name {[_short(c) for c in bundle.controls]}")
    ctl = e.get("control")
    if ctl and isinstance(man["bundles"].get(ctl), dict) and b["controls"] != [man["bundles"][ctl].get("id")]:
        problems.append(f"{name}: its controls are {[_short(c) for c in b['controls']]}, the declared control "
                        f"{ctl} is {_short(man['bundles'][ctl].get('id'))}")
    problems += [f"{name}: control {c} is missing" for c in b["controls"] if kind_of(bundles / c) != "dir"]
    try:
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            got = verify_exports(bundle, b["format"], rows_by_id, pathlib.Path(tmp))
    except (ValueError, OSError) as exc:
        return problems + [f"{name}: {exc}"]
    for lock_id in sorted(set(got) | set(b["exports"])):
        if got.get(lock_id) != b["exports"].get(lock_id):
            problems.append(f"{name}: {lock_id} re-exports as {_short(got.get(lock_id))}, the manifest records "
                            f"{_short(b['exports'].get(lock_id))}")
    return problems


def _strays(bundles, man) -> list[str]:
    valid = [b for b in man["bundles"].values() if isinstance(b, dict)]
    named = {"manifest.json", "rowspaces"} | {b["id"] for b in valid if isinstance(b.get("id"), str)}
    named |= {c for b in valid if isinstance(b.get("controls"), list) for c in b["controls"] if isinstance(c, str)}
    problems = [f"{bundles.name}/{p.name}: not named by the manifest" for p in sorted(bundles.iterdir())
                if p.name not in named]
    if kind_of(bundles / "rowspaces") == "dir":
        problems += [f"{bundles.name}/rowspaces/{p.name}: not named by the manifest"
                     for p in sorted((bundles / "rowspaces").iterdir()) if p.name not in set(man["rowspaces"].values())]
    try:
        left = staging_dirs(bundles)
    except OSError as exc:              # unreadable is not empty: say that nothing could be ruled out
        return problems + [f"{bundles.parent}: cannot be listed ({exc.strerror}), so a staging directory a killed "
                           f"build left beside {bundles.name}/ cannot be ruled out"]
    return problems + [f"{p.name}: the staging directory of a build that did not finish" for p in left]


def _lock_coverage(man, lock, rows_by_id) -> list[str]:
    problems = []
    trained = {t["arm"] for r in lock["files"] for t in r["trained"]}
    if set(man["arms"]) != trained:
        problems.append(f"runs differ from SIGNALS.lock's trained runs: {sorted(set(man['arms']) ^ trained)[:5]}")
    records = {n: b for n, b in man["bundles"].items() if not _record(n, b)}
    exported = {i for b in records.values() for i in b["exports"]}
    unexported, unlocked = sorted(set(rows_by_id) - exported), sorted(exported - set(rows_by_id))
    if unexported:
        problems.append(f"files SIGNALS.lock.json lists that no compiled labels export: {unexported}")
    if unlocked:
        problems.append(f"the manifest records exports of files SIGNALS.lock.json does not have: {unlocked}")
    for arm, name in man["arms"].items():
        if name not in records:
            continue
        need = {r["id"] for r in lock["files"] if any(t["arm"] == arm for t in r["trained"])}
        if not need <= set(records[name]["exports"]):
            problems.append(f"{arm}: its compiled labels {name} do not export "
                            f"{sorted(need - set(records[name]['exports']))}")
    return problems


def check_tree(bundles, *, scratch=None, conf, lock, generators=GENERATORS) -> list[str]:
    """Every problem of the module docstring in the tree at `bundles`; an empty list means none."""
    bundles, conf, lock_path = pathlib.Path(bundles), pathlib.Path(conf), pathlib.Path(lock)
    man = json.loads((bundles / "manifest.json").read_text())
    problems = _shape(man)
    if problems:
        return problems
    entries, lock = load_conf(conf), json.loads(lock_path.read_text())
    rows_by_id = {r["id"]: r for r in lock["files"]}
    if man["signals_lock_sha256"] != sha256_of(lock_path):
        problems.append("the manifest was built against another SIGNALS.lock.json")
    if man["conf_sha256"] != sha256_of(conf):
        problems.append(f"the manifest was built from another declaration (conf_sha256 {_short(man['conf_sha256'])}, "
                        f"{conf.name} is {_short(sha256_of(conf))})")
    problems += _names_and_arms(man, entries) + _rowspaces(bundles, man, entries)
    for name, b in man["bundles"].items():
        problems += _bundle(name, b, entries.get(name), man, bundles, rows_by_id, scratch, generators)
    return problems + _strays(bundles, man) + _lock_coverage(man, lock, rows_by_id)
