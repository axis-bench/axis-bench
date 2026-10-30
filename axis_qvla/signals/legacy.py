"""Write the files the paper's training runs consumed, byte for byte, from legacy records.

A legacy record (axis_qvla.signals.schema.LegacyFile) is a codec plus an ordered tree
(axis_qvla.signals.otree) whose array leaves name bundle arrays. `npz` writes `numpy.savez` members in
tree order: `$array` (a named array), `$json` (a 0-d string holding `json.dumps` of a tree: the
`meta` member of every schedule and tag file) or `$strings` (a unicode array: a tag file's
`prompts`). The json codecs write `json.dumps` of the decoded tree with the original writer's
indent: none (weights, ranges), 1 (subset ranges) or 2 plus a newline (roots indexes). Two nodes
keep per-episode tables out of a manifest's JSON: `$episode_floats` (per-episode float lists from
one float32 array, as the AWR weight files store them) and `$spans` (per-episode [start, end) lists,
as the ranges files store them).

`decompose` is the inverse, for a file that already exists: each of the 30 locked files
decomposes and renders back to its locked sha256 (tests/golden/signals/test_legacy_roundtrip.py).
`legacy_target` is what a built-in needs to write one locked file again: its path and kind from
the lock, and from the generator map (conf/signals/generators.json) the generator digest its metadata
records and the roots index its config hash folded in.

Offline tier: numpy.
"""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import io
import json
import pathlib
import re
import zipfile
from collections.abc import Mapping

import numpy as np

from axis_qvla.signals import otree
from axis_qvla.signals.blocks.schedule import LAMBDA_MAX, N_BINS, RAMP_START_FRAC
from axis_qvla.signals.schema import CODECS, LegacyFile

ARRAY_KEY = re.compile(r"[a-z0-9_]+")
# How each kind of locked file (SIGNALS.lock `kind`) is consumed: the LeRobot views by the LeRobot
# fork, the tag files by both forks, everything else by openpi. The first entry is the default.
KIND_FORMATS = {"rowspace": ("openpi-legacy",), "awr_weights": ("openpi-legacy",),
                "schedule": ("openpi-legacy",), "quality_tags": ("openpi-legacy", "lerobot-legacy"),
                "lerobot_rows": ("lerobot-legacy",), "lerobot_awr_flat": ("lerobot-legacy",),
                "lerobot_schedule": ("lerobot-legacy",)}


class LegacyError(ValueError):
    """A legacy record cannot be rendered, or a file cannot be decomposed into one."""


def codec_for_path(path: str) -> str:
    """The codec a published signal file was written with, read off its name."""
    name = path.rsplit("/", 1)[-1]
    if name.endswith(".npz"):
        return "npz"
    if name.startswith("roots_"):
        return "json-indent2-nl"
    if "/subsets/" in f"/{path}":
        return "json-indent1"
    return "json"


def _array(arrays: Mapping, name: str) -> np.ndarray:
    if name not in arrays:
        raise LegacyError(f"no array named {name!r}")
    return np.asarray(arrays[name])


def _offsets(n_keys: int, counts: np.ndarray, total: int, kind: str) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 1 or counts.size != n_keys or (counts < 0).any() or int(counts.sum()) != total:
        raise LegacyError(f"{kind}: {n_keys} keys and {counts.size} counts summing to "
                          f"{int(counts.sum())} do not describe {total} values")
    return np.concatenate([[0], np.cumsum(counts)])


def _resolver(arrays: Mapping):
    def resolve(kind: str, arg):
        if kind == "$episode_floats":
            keys, values = list(arg["keys"]), _array(arrays, arg["values"])
            off = _offsets(len(keys), _array(arrays, arg["lengths"]), values.size, kind)
            return {k: values[off[i]:off[i + 1]].tolist() for i, k in enumerate(keys)}
        if kind == "$spans":
            keys, spans = list(arg["keys"]), _array(arrays, arg["spans"]).reshape(-1, 2)
            off = _offsets(len(keys), _array(arrays, arg["counts"]), len(spans), kind)
            return {k: spans[off[i]:off[i + 1]].tolist() for i, k in enumerate(keys)}
        raise LegacyError(f"{kind} is not a node the json codecs know")
    return resolve


def _npz_member(node, arrays: Mapping) -> np.ndarray:
    if not (isinstance(node, Mapping) and len(node) == 1):
        raise LegacyError(f"an npz member must be one $-node, got {node!r}")
    (kind, arg), = node.items()
    if kind == "$array":
        return _array(arrays, arg)
    if kind == "$json":
        return np.array(json.dumps(otree.decode(arg)))
    if kind == "$strings":
        return np.array([str(s) for s in arg])
    raise LegacyError(f"{kind} is not an npz member node")


def render(lf: LegacyFile, arrays: Mapping) -> bytes:
    """The bytes of one legacy file."""
    if lf.codec not in CODECS:
        raise LegacyError(f"unknown codec {lf.codec!r}")
    if lf.codec == "npz":
        members = {name: _npz_member(node, arrays) for name, node in lf.doc["$o"]}
        buf = io.BytesIO()
        np.savez(buf, **members)
        return buf.getvalue()
    obj = otree.decode(lf.doc, _resolver(arrays))
    if lf.codec == "json":
        return json.dumps(obj).encode()
    if lf.codec == "json-indent1":
        return json.dumps(obj, indent=1).encode()
    return (json.dumps(obj, indent=2) + "\n").encode()


def _is_spans(v) -> bool:
    return isinstance(v, list) and all(
        isinstance(p, list) and len(p) == 2 and all(isinstance(i, int) and not isinstance(i, bool) for i in p)
        for p in v)


def decompose(data: bytes, *, path: str, lock_id: str | None, fmt: str, name: str) -> tuple[LegacyFile, dict]:
    """A legacy record plus its arrays (keyed `<name>_<member>`, referenced as `legacy.<key>`).

    Renders its own result before returning and refuses anything that does not render back to
    `data` byte for byte (e.g. a JSON object with a duplicate key, which `json.loads` silently
    collapses): a decomposition this function could not itself verify is not one a caller should
    trust."""
    if not ARRAY_KEY.fullmatch(name):
        raise LegacyError(f"array name {name!r} must match {ARRAY_KEY.pattern}")
    codec = codec_for_path(path)
    arrays: dict[str, np.ndarray] = {}
    if codec == "npz":
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            members = [i.filename[:-len(".npy")] for i in zf.infolist()]
        pairs = []
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            for m in members:
                a = z[m]
                if a.dtype.kind == "U" and a.shape == ():
                    pairs.append([m, {"$json": otree.encode(json.loads(str(a)))}])
                elif a.dtype.kind == "U":
                    pairs.append([m, {"$strings": [str(s) for s in a.tolist()]}])
                else:
                    arrays[f"{name}_{m}"] = a
                    pairs.append([m, {"$array": f"legacy.{name}_{m}"}])
        lf = LegacyFile(fmt, lock_id, path, codec, {"$o": pairs})
    else:
        obj = json.loads(data)
        if isinstance(obj, dict) and isinstance(obj.get("weights"), dict):
            pairs = []
            for k, v in obj.items():
                if k != "weights":
                    pairs.append([k, otree.encode(v)])
                    continue
                keys = list(v)
                flat = [x for key in keys for x in v[key]]
                values = np.asarray(flat, dtype=np.float32)
                if values.astype(np.float64).tolist() != flat:
                    raise LegacyError(f"{path}: a weight is not a float32 value; it cannot be stored as one")
                arrays[f"{name}_lengths"] = np.asarray([len(v[key]) for key in keys], dtype=np.int64)
                arrays[f"{name}_values"] = values
                pairs.append([k, {"$episode_floats": {"keys": keys, "lengths": f"legacy.{name}_lengths",
                                                      "values": f"legacy.{name}_values"}}])
            doc = {"$o": pairs}
        elif isinstance(obj, dict) and obj and all(_is_spans(v) for v in obj.values()):
            keys = list(obj)
            arrays[f"{name}_counts"] = np.asarray([len(obj[k]) for k in keys], dtype=np.int64)
            arrays[f"{name}_spans"] = np.asarray([s for k in keys for s in obj[k]], dtype=np.int64).reshape(-1, 2)
            doc = {"$spans": {"keys": keys, "counts": f"legacy.{name}_counts", "spans": f"legacy.{name}_spans"}}
        else:
            doc = otree.encode(obj)
        lf = LegacyFile(fmt, lock_id, path, codec, doc)
    rendered = render(lf, {f"legacy.{k}": v for k, v in arrays.items()})
    if rendered != data:
        raise LegacyError(f"{path}: this decomposition does not render back to its input bytes")
    return lf, arrays


REPO = pathlib.Path(__file__).resolve().parents[2]
GENERATORS = REPO / "conf" / "signals" / "generators.json"
SIGNALS_LOCK = REPO / "conf" / "release" / "SIGNALS.lock.json"


@dataclasses.dataclass(frozen=True)
class LegacyTarget:
    """One locked file a bundle writes: where it goes, in which format, and what its generator
    recorded. Everything but `format` is read from the lock and the generator map."""

    lock_id: str
    path: str
    kind: str
    sha256: str
    format: str
    generator: str
    generator_sha256: str | None     # what the paper's generator stamped into meta, if anything
    roots_form: str | None           # which roots index its config hash folded in: "d8" (stored) or "v2"
    numerics: str | None             # e.g. "log2-extended" (axis_qvla.signals.numerics)
    tolerance: tuple[tuple[str, float], ...] | None   # sorted (key, bound) pairs; a mapping is converted
    template_name: str | None        # the random-50 controls record their template under this name

    def __post_init__(self) -> None:
        # Pairs, not the generator map's dict: a frozen target must hash, and must not alias the cached
        # map every later target is read from.
        if self.tolerance is not None:
            pairs = sorted((str(k), float(v)) for k, v in dict(self.tolerance).items())
            object.__setattr__(self, "tolerance", tuple(pairs))


@functools.lru_cache(maxsize=None)
def _maps() -> tuple[dict, dict]:
    return json.loads(GENERATORS.read_text()), json.loads(SIGNALS_LOCK.read_text())


def legacy_target(lock_id: str, fmt: str) -> LegacyTarget:
    spec, lock = _maps()
    row = next((r for r in lock["files"] if r["id"] == lock_id), None)
    if row is None:
        raise LegacyError(f"{lock_id} is not a file SIGNALS.lock.json lists")
    allowed = KIND_FORMATS[row["kind"]]
    if fmt not in allowed:
        raise LegacyError(f"{lock_id} is a {row['kind']} file, consumed as {' or '.join(allowed)}, not {fmt}")
    entry = spec["files"][lock_id]
    gen = spec["generators"][entry["generator"]]
    argv = list(entry["argv"])
    roots_form = None
    for flag in ("--roots", "--index"):
        if flag in argv:
            roots_form = {"{roots_v2}": "v2", "{lock:index.roots_d8}": "d8"}.get(argv[argv.index(flag) + 1])
    template_name = argv[argv.index("--template-name") + 1] if "--template-name" in argv else None
    return LegacyTarget(lock_id=lock_id, path=row["path"], kind=row["kind"], sha256=row["sha256"], format=fmt,
                        generator=entry["generator"], generator_sha256=gen.get("generator_sha256"),
                        roots_form=roots_form, numerics=entry.get("numerics"), tolerance=entry.get("tolerance"),
                        template_name=template_name)


def schedule_config_hash(*, mode: str, keep: float, steps: int, batch: int, seed: int, generator_sha256: str,
                         weights_sha256: str, roots_sha256: str, ranges_sha256: str) -> str:
    """The schedule builder's config hash (axis_qvla/signals/methods/schedule.py `_config_hash`) over the
    RECORDED generator digest and the digests of the inputs actually read."""
    payload = {"mode": mode, "keep_top_frac": float(keep), "total_steps": int(steps), "batch": int(batch),
               "seed": int(seed), "n_bins": N_BINS, "lambda_max": LAMBDA_MAX, "ramp_start_frac": RAMP_START_FRAC,
               "generator_digest": generator_sha256, "weights_digest": weights_sha256,
               "roots_digest": roots_sha256, "ranges_digest": ranges_sha256}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# The tag builder's metadata layout per recorded version (the build_quality_labels sources generators.json records):
# the builder source of 1076500, run as tags@d0c3388 and tags@7d41f7f to write both episode-tag files,
# has no `granularity` key and does not fold it into the config hash; 89c8c46 (approach-aware chunk
# tags) adds `granularity`; 888d0d2 (object-centric chunk tags) also adds `plateau`, as the live
# builder does.
TAG_DIALECTS = {"tags@d0c3388": "episode", "tags@7d41f7f": "episode",
                "tags@89c8c46": "row-v1", "tags@888d0d2": "row-v2"}


def tags_meta_for(meta: Mapping, dialect: str) -> dict:
    """The live tag builder's metadata (axis_qvla/signals/methods/cfg_tags.py `quality_meta`) cut to the
    keys the recorded builder version wrote, in the same order."""
    drop = {"episode": ("granularity", "plateau"), "row-v1": ("plateau",), "row-v2": ()}[dialect]
    return {k: v for k, v in meta.items() if k not in drop}


def tags_config_hash(*, dialect: str, seed: int, generator_sha256: str, weights_sha256: str, roots_sha256: str,
                     ranges_sha256: str, granularity: str, plateau: str) -> str:
    """The tag builder's config hash for one recorded version, over the RECORDED generator digest."""
    payload = {"seed": int(seed), "n_bins": N_BINS, "generator_digest": generator_sha256,
               "weights_digest": weights_sha256, "roots_digest": roots_sha256, "ranges_digest": ranges_sha256}
    if dialect in ("row-v1", "row-v2"):
        payload["granularity"] = str(granularity)
    if dialect == "row-v2":
        payload["plateau"] = str(plateau)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
