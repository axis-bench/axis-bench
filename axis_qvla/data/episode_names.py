"""The per-episode object binding the approach-aware reward reads, shipped as one pinned table.

Stdlib only at import, so the shipped table is read and digested without numpy;
`corpus_table` imports pandas inside the function (offline tier).

WHY THIS TABLE SHIPS. Eq. (3)'s d_t is the hand's distance to the episode's OBJECT, and
`build_awr_weights --binding names` finds that object's pose column through two per-episode meta
fields: `body_names` (the order the episode's pose columns were stacked in) and `object_body`.
Neither the pretraining data's fingerprint nor the weights header covers these two fields, so a
copy of the data whose meta carries other names would rebuild different weights under the same
fingerprint. Hence the table ships as `conf/signals/episode_body_names.json`, its content digest
is pinned in `conf/signals/generators.json`, and the builder's `--names-map` (read the names from
it) and `--names-sha256` (refuse any other names) make a rebuild bind exactly these objects.

The digest is over CONTENT -- the (key, body_names, object_body) triples in (task, episode)
order -- so pretraining data whose own meta carries these names digests to the same value as the
table. `target_body` is not part of it: the reward never reads it.

    python -m axis_qvla.data.episode_names digest --index <roots index> [--names-map <table>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from collections.abc import Mapping

SCHEMA = "qvla.episode_body_names/1"
# (body_names, object_body); either may be None in a corpus whose meta lacks the field.
Entry = tuple


def key_order(key: str) -> tuple[int, int]:
    """`task_<id>--<episode>` -> (id, episode): the corpus order every artifact walks in."""
    task, ep = key.split("--")
    return int(task.removeprefix("task_")), int(ep)


def _norm(names, obj) -> Entry:
    names = None if names is None or isinstance(names, (str, float)) else [str(x) for x in names]
    return names, (obj if isinstance(obj, str) else None)


def table_digest(table: Mapping[str, Entry]) -> str:
    """sha256 over the (key, body_names, object_body) triples in (task, episode) order."""
    rows = [[k, *_norm(*table[k])] for k in sorted(table, key=key_order)]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def write_table(table: Mapping[str, Entry], path, *, about: str) -> str:
    """Write the table, factoring out the body prefix every episode shares (the robot links) and
    numbering the distinct scene lists, one scene and one episode per line. Returns its digest."""
    keys = sorted(table, key=key_order)
    rows = [_norm(*table[k]) for k in keys]
    if any(n is None or o is None for n, o in rows):
        raise ValueError("every episode needs body_names and object_body to be written")
    prefix = list(rows[0][0])
    for n, _o in rows[1:]:
        while n[:len(prefix)] != prefix:
            prefix.pop()
    scenes: list[list[str]] = []
    index: dict[tuple, int] = {}
    eps = []
    for k, (n, o) in zip(keys, rows):
        scene = tuple(n[len(prefix):])
        if scene not in index:
            index[scene] = len(scenes)
            scenes.append(list(scene))
        eps.append(f"  {json.dumps(k)}: [{index[scene]}, {json.dumps(o)}]")
    digest = table_digest(table)
    lines = ["{", f' "schema": {json.dumps(SCHEMA)},', f' "about": {json.dumps(about)},',
             f' "reward_input_sha256": {json.dumps(digest)},', f' "robot_prefix": {json.dumps(prefix)},',
             ' "scenes": [', ",\n".join(f"  {json.dumps(s)}" for s in scenes), " ],",
             ' "episodes": {', ",\n".join(eps), " }", "}"]
    pathlib.Path(path).write_text("\n".join(lines) + "\n")
    return digest


def load_table(path) -> dict[str, Entry]:
    """The table written by `write_table` (its digest is pinned in
    conf/signals/generators.json)."""
    doc = json.loads(pathlib.Path(path).read_text())
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {doc.get('schema')!r} is not {SCHEMA!r}")
    prefix, scenes = doc["robot_prefix"], doc["scenes"]
    return {k: (prefix + scenes[i], obj) for k, (i, obj) in doc["episodes"].items()}


def corpus_table(index) -> dict[str, Entry]:
    """The table a corpus's own `meta/episodes` holds, for every task of a roots index."""
    import glob

    import pandas as pd

    index = pathlib.Path(index)
    idx = json.loads(index.read_text())
    base = index.parent / idx.get("base", ".")
    out: dict[str, Entry] = {}
    for tid, rel in sorted(idx["roots"].items(), key=lambda kv: int(kv[0])):
        for f in sorted(glob.glob(str(base / rel / "meta" / "episodes" / "**" / "*.parquet"), recursive=True)):
            df = pd.read_parquet(f)
            names = df["body_names"] if "body_names" in df else [None] * len(df)
            objs = df["object_body"] if "object_body" in df else [None] * len(df)
            for ep, n, o in zip(df["episode_index"], names, objs):
                out[f"task_{int(tid)}--{int(ep)}"] = _norm(n, o)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="axis_qvla.data.episode_names", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    dg = sub.add_parser("digest", help="digest the names a build over this roots index would bind")
    dg.add_argument("--index", required=True, help="roots index JSON of the pretraining data")
    dg.add_argument("--names-map", default=None, help="a shipped table to bind instead of the meta")
    a = ap.parse_args(argv)
    table = corpus_table(a.index)
    if a.names_map:
        shipped = load_table(a.names_map)
        table = {k: shipped.get(k, (None, None)) for k in table}
    print(f"{len(table)} episodes, sha256 {table_digest(table)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
