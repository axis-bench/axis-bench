"""Write and compare `expected/`, the frozen T0 outputs, and report the FIRST difference.

Why: `--check` must say where a rebuild diverged, not just that it did. `expected/` holds every
T0 output plus `MANIFEST.json` {schema, files: {relpath: sha256}} in canonical bytes. The
comparison reports, first, the first differing file in a fixed order (pooled cells, then the
in-text numbers, then the rest by path) with a pointer into it: the first differing cell of
`pooled_r1.json`, the first differing claim of `claims.json` (or its Table I block), or the
first differing line of any other text file. The expected directory is listed with `os.scandir` so a stray file is
reported, and any OSError propagates to the caller (exit 3), never an empty listing. A file
matching `axis_qvla.data.hf.download`'s own `.<name>.part-<pid>` temp-file pattern is
skipped by the walk ONLY when the caller passes `skip_part_files=True`: a download killed
mid-transfer can leave one behind in `axis_qvla.paper.fetch.HfSource`'s
persistent `--cache` directory, and it names no real published file there, so counting it as a
stray there would be a false DIFFERS with nothing to fix. `t0.check` passes `skip_part_files=True`
only for an `HfSource` (that download cache); a LOCAL pack's `expected/` is never touched by
`download`, so a `.part-` file there is an ordinary stray like any other and must still DIFFER.

Core tier: stdlib only.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
from collections.abc import Mapping

from axis_qvla.core.canon import dumps_canonical
from axis_qvla.core.canon import sha256_bytes as _sha   # the one sha helper

EXPECTED_SCHEMA = "qvla.expected/1"
MANIFEST = "MANIFEST.json"
_FIRST = ("pooled_r1.json", "claims.json")
_PART_FILE = re.compile(r"^\..+\.part-\d+$")


def write_expected(outputs: Mapping[str, bytes], out_dir: pathlib.Path) -> pathlib.Path:
    out_dir = pathlib.Path(out_dir)
    for rel, data in outputs.items():
        p = out_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    doc = {"schema": EXPECTED_SCHEMA, "files": {rel: _sha(outputs[rel]) for rel in sorted(outputs)}}
    (out_dir / MANIFEST).write_bytes(dumps_canonical(doc))
    return out_dir


def _walk(root: pathlib.Path, prefix: str = "", *, skip_part_files: bool = False) -> list[str]:
    out = []
    with os.scandir(root) as it:
        for e in sorted(it, key=lambda e: e.name):
            if skip_part_files and _PART_FILE.fullmatch(e.name):
                continue
            rel = prefix + e.name
            if e.is_dir(follow_symlinks=False):
                out += _walk(pathlib.Path(e.path), rel + "/", skip_part_files=skip_part_files)
            else:
                out.append(rel)
    return out


def _detail(rel: str, got: bytes, want: bytes) -> str:
    if rel == "pooled_r1.json":
        g, w = json.loads(got), json.loads(want)
        for k in list(g) + [k for k in w if k not in g]:
            if g.get(k) != w.get(k):
                return f"first differing cell {k!r}"
        return "same cells, different bytes (key order or formatting)"
    if rel == "claims.json":
        g, w = json.loads(got), json.loads(want)
        gc, wc = g.get("claims", {}), w.get("claims", {})
        for k in list(gc) + [k for k in wc if k not in gc]:
            if gc.get(k) != wc.get(k):
                return f"first differing claim {k!r}"
        if g.get("table1") != w.get("table1"):
            return "table1 (Table I per-seed values)"
        return "same claims, different bytes (header, key order or formatting)"
    gl, wl = got.decode("utf-8", "replace").split("\n"), want.decode("utf-8", "replace").split("\n")
    line = next((i for i, (a, b) in enumerate(zip(gl, wl)) if a != b), min(len(gl), len(wl)))
    return f"first differing line {line + 1}"


def compare_expected(outputs: Mapping[str, bytes], expected_dir: pathlib.Path, *,
                     skip_part_files: bool = False) -> list[str]:
    expected_dir = pathlib.Path(expected_dir)
    manifest = json.loads((expected_dir / MANIFEST).read_bytes())
    if not isinstance(manifest, dict) or manifest.get("schema") != EXPECTED_SCHEMA:
        schema = manifest.get("schema") if isinstance(manifest, dict) else None
        return [f"{MANIFEST}: schema is {schema!r}, need {EXPECTED_SCHEMA}"]
    want = manifest.get("files")
    problems = []
    on_disk = set(_walk(expected_dir, skip_part_files=skip_part_files)) - {MANIFEST}
    for rel in sorted(on_disk - set(want)):
        problems.append(f"expected/{rel} is not listed in {MANIFEST}")
    order = [r for r in _FIRST if r in set(want) | set(outputs)]
    order += sorted((set(want) | set(outputs)) - set(_FIRST))
    diffs = []
    for rel in order:
        if rel not in outputs:
            diffs.append(f"{rel}: in expected/ but not produced")
        elif rel not in want:
            diffs.append(f"{rel}: produced but not in expected/")
        elif _sha(outputs[rel]) != want[rel]:
            ref = (expected_dir / rel).read_bytes()
            if _sha(ref) != want[rel]:
                diffs.append(f"expected/{rel} does not match {MANIFEST}")
            else:
                diffs.append(f"{rel} differs: {_detail(rel, outputs[rel], ref)}")
    return diffs + problems
