"""Write the pretrain roots index: which per-task LeRobot datasets the loader concatenates.

This is where the held-out split is enforced: the loader trains on exactly the roots named here.
`select_roots` decides which task directories are usable and reports what it excluded and what it
could not find, because a silently-empty exclusion is indistinguishable from a working one, and
that is how a held-out task ends up in pretraining. `roots_payload` builds the v2 index: roots
relative to a `base` that is itself relative to the index file, so the tree survives a copy to
another machine. axis_qvla.signals.rowspace reads the index back into a frame index.

Standard library at import; pyarrow is imported where parquet schemas are compared.

    python -m axis_qvla.data.index.roots --src-root <dir> --out <roots>.json
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import re

# The held-out benchmark tasks that also exist in the corpus build; none may be pretrained on.
HELDOUT_EXCLUDED = (809, 811, 813, 866, 945, 953, 966, 1426, 1459, 1889)


@dataclasses.dataclass(frozen=True)
class RootsReport:
    roots: dict          # task id (str) -> task directory, in sorted-directory-name order
    excluded: tuple      # held-out ids present in src-root and left out
    missing: tuple       # held-out ids not present in src-root
    unusable: tuple      # dropped: no meta/info.json, no meta/tasks.parquet, no video, or a minority schema


def select_roots(src_root, *, exclude_ids=HELDOUT_EXCLUDED) -> RootsReport:
    excluded = {int(x) for x in exclude_ids}
    roots, skipped, unusable = {}, [], []
    schema_of: dict[int, tuple] = {}
    for d in sorted(glob.glob(os.path.join(src_root, "task_*"))):
        m = re.search(r"task_(\d+)$", os.path.basename(d))
        if not m:
            continue
        tid = int(m.group(1))
        if tid in excluded:
            skipped.append(tid)
            continue
        if not os.path.isfile(os.path.join(d, "meta", "info.json")):
            unusable.append(tid)
            continue
        # meta/tasks.parquet is what the loader reads the prompt from; a task without it takes the
        # whole norm-stats pass down with a bare FileNotFoundError.
        if not os.path.isfile(os.path.join(d, "meta", "tasks.parquet")):
            unusable.append(tid)
            continue
        # A task with no video files cannot be loaded locally: LeRobotDataset falls back to the Hub
        # and dies with an error that names neither the task nor the real problem.
        if not glob.glob(os.path.join(d, "videos", "**", "*.mp4"), recursive=True):
            unusable.append(tid)
            continue
        roots[str(tid)] = d
        # Column NAMES matching is not enough: a task whose finalize step failed keeps variable-length
        # columns, and the loader casts every shard to one schema, so a single straggler kills the
        # whole concatenation. Compare (name, type) and drop the outliers here.
        try:
            import pyarrow.parquet as pq
            p = sorted(glob.glob(os.path.join(d, "data", "**", "*.parquet"), recursive=True))
            if p:
                s = pq.read_schema(p[0])
                schema_of[tid] = tuple(sorted((n, str(t)) for n, t in zip(s.names, s.types)))
        except Exception:  # noqa: BLE001 - a task we cannot even read is dropped below
            schema_of[tid] = ()
    if schema_of:
        counts: dict[tuple, int] = {}
        for sig in schema_of.values():
            counts[sig] = counts.get(sig, 0) + 1
        majority = max(counts, key=counts.get)
        for t in sorted(t for t, sig in schema_of.items() if sig != majority):
            roots.pop(str(t), None)
            unusable.append(t)
    return RootsReport(roots=roots, excluded=tuple(sorted(skipped)),
                       missing=tuple(sorted(excluded - set(skipped))), unusable=tuple(sorted(unusable)))


def roots_payload(report: RootsReport, *, src_root, out) -> dict:
    out_dir = os.path.dirname(os.path.abspath(out)) or "."
    src_abs = os.path.abspath(src_root)
    return {"version": 2, "base": os.path.relpath(src_abs, out_dir),
            "roots": {tid: os.path.relpath(os.path.abspath(d), src_abs) for tid, d in report.roots.items()}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="axis_qvla.data.index.roots", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-root", required=True,
                    help="derived per-task LeRobot build root holding task_<id>/ directories")
    ap.add_argument("--out", required=True, help="where to write the roots index JSON")
    ap.add_argument("--exclude-ids", default=",".join(str(t) for t in HELDOUT_EXCLUDED))
    a = ap.parse_args(argv)
    report = select_roots(a.src_root, exclude_ids=[int(x) for x in a.exclude_ids.split(",") if x.strip()])
    payload = roots_payload(report, src_root=a.src_root, out=a.out)
    with open(a.out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"roots written: {len(report.roots)} tasks -> {a.out} "
          f"(v2, base={payload['base']!r} relative to the index file)")
    print(f"held-out excluded ({len(report.excluded)}): {list(report.excluded)}")
    if report.missing:
        print(f"  note: {len(report.missing)} excluded id(s) not present in src-root: {list(report.missing)}")
    if report.unusable:
        print(f"  dropped as unloadable: {list(report.unusable)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
