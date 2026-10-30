"""Write, load and export SignalBundles: content-addressed directories of canonical JSON and .npy.

A bundle is `<root>/<id>/` holding `manifest.json` and one `.npy` per array, never a pickle. The
manifest is canonical JSON (axis_qvla.core.canon) and `id` is the sha256 of its bytes. It describes
every array by dtype, shape and a sha256 over dtype, shape and bytes, so the id covers the arrays
too, and `load_bundle` refuses a directory whose name, manifest or any array does not verify.
Arrays are named `plan`, `weights`, `cond`, `rowspace.<name>` and `legacy.<name>`; a legacy
record's `$array` leaves use those names. A row space is stored the same way on its own
(`write_rowspace`, `load_rowspace`) and embedded in every bundle that indexes it.

Offline tier: numpy.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pathlib
import shutil
import uuid
from collections.abc import Mapping
from typing import Literal

import numpy as np

from axis_qvla.core.canon import dumps_canonical, sha256_bytes
from axis_qvla.signals import legacy
from axis_qvla.signals.schema import (FORMATS, ROWSPACE_SCHEMA, SCHEMA, Budget, CondVocab, LegacyFile, RowSpace,
                                 ServeCondition, SignalBundle)

MANIFEST = "manifest.json"


class BundleError(ValueError):
    """A bundle or row space does not verify, or cannot be written or exported as asked."""


@contextlib.contextmanager
def _reraise_as_bundle_error(path):
    """Every way a directory on disk can fail to be a bundle -- a missing manifest or array file,
    a truncated .npy, a manifest missing a key a reader needs -- is a BundleError, never a raw
    OSError/KeyError/ValueError/EOFError, so a consumer can catch one exception for all of them."""
    try:
        yield
    except BundleError:
        raise
    except (OSError, ValueError, EOFError, KeyError, TypeError, AttributeError) as exc:
        raise BundleError(f"{path}: not a valid compiled-labels directory ({exc})") from exc


def array_digest(a) -> str:
    """sha256 over dtype, shape and C-order bytes: identity of an array, not of an .npy header."""
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(f"{a.dtype.str}|{','.join(str(d) for d in a.shape)}|".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _describe(name: str, a) -> dict:
    a = np.asarray(a)
    if a.dtype.kind == "O":
        raise BundleError(f"{name}: object arrays cannot be stored without pickle")
    return {"dtype": a.dtype.str, "shape": list(a.shape), "sha256": array_digest(a)}


def _rowspace_record(rs: RowSpace) -> tuple[dict, dict]:
    arrays = {k: np.asarray(v) for k, v in sorted(rs.arrays.items())}
    for k in arrays:
        if not legacy.ARRAY_KEY.fullmatch(k):
            raise BundleError(f"frame-index array name {k!r} must match {legacy.ARRAY_KEY.pattern}")
    rec = {"schema": ROWSPACE_SCHEMA, "name": rs.name, "kind": rs.kind, "n_rows": int(rs.n_rows),
           "n_trainable": int(rs.n_trainable), "corpus_fingerprint": rs.corpus_fingerprint,
           "source": rs.source, "roots": rs.roots, "roots_path": rs.roots_path,
           "roots_codec": rs.roots_codec, "ranges_path": rs.ranges_path,
           "ranges_codec": rs.ranges_codec, "prompts": list(rs.prompts),
           "idle_filter": dict(rs.idle_filter), "heldout_excluded": [int(t) for t in rs.heldout_excluded],
           "arrays": {k: _describe(k, v) for k, v in arrays.items()}}
    return rec, arrays


def rowspace_id(rs: RowSpace) -> str:
    return sha256_bytes(dumps_canonical(_rowspace_record(rs)[0]))


def _rowspace_from(rec: Mapping, arrays: Mapping) -> RowSpace:
    return RowSpace(name=rec["name"], kind=rec["kind"], n_rows=rec["n_rows"], n_trainable=rec["n_trainable"],
                    corpus_fingerprint=rec["corpus_fingerprint"], source=rec["source"], roots=rec["roots"],
                    roots_path=rec["roots_path"], roots_codec=rec["roots_codec"],
                    ranges_path=rec["ranges_path"], ranges_codec=rec["ranges_codec"],
                    prompts=tuple(rec["prompts"]), idle_filter=dict(rec["idle_filter"]),
                    heldout_excluded=tuple(rec["heldout_excluded"]), arrays=dict(arrays))


def bundle_arrays(b: SignalBundle) -> dict:
    """Every array of a bundle, under the name its manifest and its legacy records use."""
    out = {"plan": np.asarray(b.plan)}
    if b.weights is not None:
        out["weights"] = np.asarray(b.weights)
    if b.cond is not None:
        out["cond"] = np.asarray(b.cond)
    for k, v in b.rowspace.arrays.items():
        if not legacy.ARRAY_KEY.fullmatch(k):
            raise BundleError(f"frame-index array name {k!r} must match {legacy.ARRAY_KEY.pattern}")
        out[f"rowspace.{k}"] = np.asarray(v)
    for k, v in b.legacy_arrays.items():
        if not legacy.ARRAY_KEY.fullmatch(k):
            raise BundleError(f"legacy array name {k!r} must match {legacy.ARRAY_KEY.pattern}")
        out[f"legacy.{k}"] = np.asarray(v)
    return out


def _check_array_shapes(budget: Budget, rowspace: RowSpace, vocab: CondVocab | None, arrays: Mapping) -> None:
    """`plan` is int64[S,B] with S,B the budget (steps, global_batch); `weights` is float32[N] with
    N the row space's n_rows; `cond` is uint8[S,B,F] with S,B plan's and F the vocab's field count.
    Raises BundleError naming the field."""
    plan = np.asarray(arrays["plan"])
    want_plan = (budget.steps, budget.global_batch)
    if plan.dtype != np.dtype(np.int64) or plan.shape != want_plan:
        raise BundleError(f"plan: must be int64{list(want_plan)}, got {plan.dtype}{list(plan.shape)}")
    if "weights" in arrays:
        w = np.asarray(arrays["weights"])
        want_w = (rowspace.n_rows,)
        if w.dtype != np.dtype(np.float32) or w.shape != want_w:
            raise BundleError(f"weights: must be float32{list(want_w)}, got {w.dtype}{list(w.shape)}")
    if "cond" in arrays:
        if vocab is None:
            raise BundleError("cond: an array is present but the manifest's vocab is null")
        c = np.asarray(arrays["cond"])
        want_c = (*want_plan, len(vocab.fields))
        if c.dtype != np.dtype(np.uint8) or c.shape != want_c:
            raise BundleError(f"cond: must be uint8{list(want_c)}, got {c.dtype}{list(c.shape)}")


def _legacy_array_refs(doc) -> set[str]:
    """Every array name a legacy record's doc references through `$array`, `$episode_floats` or
    `$spans` nodes."""
    refs: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, Mapping):
            if set(node) == {"$o"}:
                for _, v in node["$o"]:
                    walk(v)
                return
            if len(node) == 1:
                (kind, arg), = node.items()
                if kind == "$array":
                    refs.add(arg)
                elif kind in ("$episode_floats", "$spans"):
                    if not isinstance(arg, Mapping):
                        raise BundleError(f"{kind} node must be a mapping, got {arg!r}")
                    keys = ("lengths", "values") if kind == "$episode_floats" else ("counts", "spans")
                    missing = [k for k in keys if k not in arg]
                    if missing:
                        raise BundleError(f"{kind} node is missing {missing}")
                    refs.update(arg[k] for k in keys)
            return
        if isinstance(node, list):
            for v in node:
                walk(v)

    walk(doc)
    return refs


def _check_legacy_records(b: SignalBundle, arrays: Mapping) -> None:
    """At write time: every `$array` (or table) reference in a legacy record names an array the
    bundle actually has, and no two legacy records share a `path` (export_legacy would silently
    let the second overwrite the first on disk)."""
    seen: set[str] = set()
    for f in b.legacy:
        if f.path in seen:
            raise BundleError(f"legacy path {f.path!r} is used by more than one record")
        seen.add(f.path)
        missing = sorted(ref for ref in _legacy_array_refs(f.doc) if ref not in arrays)
        if missing:
            raise BundleError(f"legacy record {f.path!r} references missing array(s) {missing}")


def _vocab_record(v: CondVocab | None):
    if v is None:
        return None
    return {"fields": list(v.fields), "prompts": list(v.prompts),
            "text": {f: [[int(c), t] for c, t in sorted(v.text[f].items())] for f in v.fields}}


def _serve_record(s: ServeCondition) -> dict:
    return {"fields": dict(s.fields), "beta": float(s.beta),
            "uncond_fields": None if s.uncond_fields is None else dict(s.uncond_fields)}


def _manifest(b: SignalBundle) -> tuple[bytes, dict]:
    rs_rec, _ = _rowspace_record(b.rowspace)
    arrays = bundle_arrays(b)
    _check_array_shapes(b.budget, b.rowspace, b.vocab, arrays)
    _check_legacy_records(b, arrays)
    man = {"schema": SCHEMA, "program": b.program, "program_sha": b.program_sha,
           "rowspace": {**rs_rec, "id": sha256_bytes(dumps_canonical(rs_rec))},
           "budget": {"steps": b.budget.steps, "global_batch": b.budget.global_batch},
           "seed": int(b.seed), "reducer": b.reducer, "vocab": _vocab_record(b.vocab),
           "serve": [_serve_record(s) for s in b.serve], "controls": list(b.controls),
           "requires": sorted(b.requires), "diagnostics": {k: float(v) for k, v in b.diagnostics.items()},
           "inputs": dict(b.inputs),
           "legacy": [{"format": f.format, "lock_id": f.lock_id, "path": f.path, "codec": f.codec, "doc": f.doc}
                      for f in b.legacy],
           "arrays": {k: _describe(k, v) for k, v in sorted(arrays.items())}}
    try:
        return dumps_canonical(man), arrays
    except (TypeError, ValueError) as exc:
        raise BundleError(f"the manifest is not canonical JSON: {exc}") from exc


def bundle_id(b: SignalBundle) -> str:
    return sha256_bytes(_manifest(b)[0])


def _write_record(root, raw: bytes, arrays: Mapping) -> str:
    rid = sha256_bytes(raw)
    root = pathlib.Path(root)
    dest = root / rid
    if dest.is_dir():
        _read_record(dest)                      # an existing copy must verify; then this is a no-op
        return rid
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f".tmp-{rid[:12]}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    tmp.mkdir()
    try:
        for name, a in arrays.items():
            np.save(tmp / f"{name}.npy", np.ascontiguousarray(a), allow_pickle=False)
        (tmp / MANIFEST).write_bytes(raw)
        os.replace(tmp, dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return rid


def _read_record(path) -> tuple[dict, dict]:
    path = pathlib.Path(path)
    with _reraise_as_bundle_error(path):
        raw = (path / MANIFEST).read_bytes()
        rid = sha256_bytes(raw)
        if path.name != rid:
            raise BundleError(f"{path}: the directory name is not the sha256 of its manifest ({rid})")
        man = json.loads(raw)
        if dumps_canonical(man) != raw:
            raise BundleError(f"{path}: {MANIFEST} is not canonical JSON")
        arrays = {}
        for name, desc in man["arrays"].items():
            a = np.load(path / f"{name}.npy", allow_pickle=False)
            if _describe(name, a) != desc:
                raise BundleError(f"{path}: array {name} does not match its manifest entry")
            arrays[name] = a
        named = {f"{n}.npy" for n in man["arrays"]} | {MANIFEST}
        stray = sorted(p.name for p in path.iterdir() if p.name not in named)
        if stray:
            raise BundleError(f"{path}: files the manifest does not name: {stray}")
        return man, arrays


def write_rowspace(rs: RowSpace, root) -> str:
    rec, arrays = _rowspace_record(rs)
    return _write_record(root, dumps_canonical(rec), arrays)


def load_rowspace(path) -> RowSpace:
    man, arrays = _read_record(path)
    with _reraise_as_bundle_error(path):
        if man.get("schema") != ROWSPACE_SCHEMA:
            raise BundleError(f"{path}: schema {man.get('schema')!r} is not {ROWSPACE_SCHEMA!r}")
        return _rowspace_from(man, arrays)


def write_bundle(bundle: SignalBundle, root: pathlib.Path) -> str:
    raw, arrays = _manifest(bundle)
    return _write_record(root, raw, arrays)


def load_bundle(path: pathlib.Path) -> SignalBundle:
    man, arrays = _read_record(path)
    with _reraise_as_bundle_error(path):
        if man.get("schema") != SCHEMA:
            raise BundleError(f"{path}: schema {man.get('schema')!r} is not {SCHEMA!r}")
        rs_rec = {k: v for k, v in man["rowspace"].items() if k != "id"}
        rs = _rowspace_from(rs_rec, {k[len("rowspace."):]: v for k, v in arrays.items() if k.startswith("rowspace.")})
        if rowspace_id(rs) != man["rowspace"]["id"]:
            raise BundleError(f"{path}: the embedded frame index does not match its id")
        v = man["vocab"]
        vocab = None if v is None else CondVocab(tuple(v["fields"]), {f: {int(c): t for c, t in v["text"][f]} for f in v["fields"]},
                                                 tuple(v["prompts"]))
        budget = Budget(man["budget"]["steps"], man["budget"]["global_batch"])
        _check_array_shapes(budget, rs, vocab, arrays)
        return SignalBundle(
            program=man["program"], program_sha=man["program_sha"], rowspace=rs,
            budget=budget, seed=man["seed"],
            plan=arrays["plan"], weights=arrays.get("weights"), reducer=man["reducer"], cond=arrays.get("cond"),
            vocab=vocab,
            serve=tuple(ServeCondition(s["fields"], s["uncond_fields"], s["beta"]) for s in man["serve"]),
            controls=tuple(man["controls"]), requires=frozenset(man["requires"]), diagnostics=man["diagnostics"],
            inputs=man["inputs"],
            legacy=tuple(LegacyFile(f["format"], f["lock_id"], f["path"], f["codec"], f["doc"]) for f in man["legacy"]),
            legacy_arrays={k[len("legacy."):]: v for k, v in arrays.items() if k.startswith("legacy.")})


def export_legacy(bundle: SignalBundle, fmt: Literal["openpi-legacy", "lerobot-legacy"],
                  out: pathlib.Path) -> list[pathlib.Path]:
    """Write the bundle's `fmt` files under `out` at their published relative paths."""
    if fmt not in FORMATS:
        raise BundleError(f"format must be one of {FORMATS}, got {fmt!r}")
    files = [f for f in bundle.legacy if f.format == fmt]
    if not files:
        have = sorted({f.format for f in bundle.legacy}) or ["none"]
        raise BundleError(f"these compiled labels have no {fmt} files (it has: {', '.join(have)})")
    arrays = bundle_arrays(bundle)
    _check_array_shapes(bundle.budget, bundle.rowspace, bundle.vocab, arrays)
    written = []
    for f in files:
        data = legacy.render(f, arrays)
        p = pathlib.Path(out) / f.path
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.name}.tmp-{uuid.uuid4().hex[:8]}")
        tmp.write_bytes(data)
        os.replace(tmp, p)
        written.append(p)
    return written
