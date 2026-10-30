"""A LeRobot dataset's statistics against its own feature shapes, checked before training reads them.

LeRobot normalises each feature with the statistics in meta/stats.json (its normaliser reads only
that file). A merge builds the merged copy's meta/stats.json from every input's meta/stats.json and
copies every input's per-episode `stats/<feature>/<stat>` columns (meta/episodes) forward, changing
only those of the index columns it renumbers. A conversion that rewrites a feature (to the DROID 8-D state and action, say) and its info.json shape
but leaves the source's statistics behind makes a dataset whose `observation.state` is declared 8-D
and normalised with 9 values: the LeRobot line stops at the first training step with a tensor-size
mismatch in its normaliser, after the GPU is allocated. `stats_problems` finds that before: every statistic of a
numeric feature (all but `count`, one value per dataset or episode) must have the feature's
declared shape, at both levels.

Core tier at import; pyarrow is imported where the episode table is read.
"""
from __future__ import annotations

import glob
import json
import pathlib

# Features whose statistics are per-channel image summaries, never the feature's own shape.
VISUAL = frozenset({"video", "image"})
# The template feature types LeRobot normalises from the dataset statistics a run trains on.
_TRAINED_TYPES = frozenset({"STATE", "ACTION", "ENV"})


def _shape(value) -> tuple[int, ...]:
    """The shape of a nested list of numbers (stdlib numpy.shape for well-formed values)."""
    out = []
    while isinstance(value, (list, tuple)):
        out.append(len(value))
        value = value[0] if value else None
    return tuple(out)


def trained_features(template: dict) -> list[str]:
    """The non-visual input and output features of a LeRobot train config's policy."""
    policy = template.get("policy") or {}
    feats = {**(policy.get("input_features") or {}), **(policy.get("output_features") or {})}
    return sorted(k for k, v in feats.items() if isinstance(v, dict) and v.get("type") in _TRAINED_TYPES)


# What a mismatch at each level leads to: LeRobot's normaliser reads only meta/stats.json; a merge
# copies the per-episode columns into the merged copy unchanged.
_CONSEQUENCE = {
    "dataset": "LeRobot's normaliser reads these and stops at the first training step",
    "episode": "a merge copies these per-episode columns unchanged, so every copy merged from this "
               "dataset keeps them",
}


def _mismatch(feature: str, stat: str, got: tuple, want: tuple, where: str, level: str) -> str:
    return (f"{where}: {feature!r} {stat} has {list(got)} values but meta/info.json declares the shape "
            f"{list(want)}: the statistics were not recomputed when the feature was converted, and "
            f"{_CONSEQUENCE[level]}; rebuild the dataset from a conversion that recomputes them "
            f"(axis_qvla.data.convert_droid_actions does, at both levels)")


def stats_problems(dataset_dir, features=None) -> list[str]:
    """Every feature (of `features`, else every numeric feature) whose statistics in meta/stats.json
    or in meta/episodes' per-episode columns do not have its declared shape; one line per feature and
    level. A dataset without meta/info.json, or without statistics for a feature, has nothing to
    compare and is not refused here."""
    root = pathlib.Path(dataset_dir)
    try:
        info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    declared = {k: v for k, v in (info.get("features") or {}).items()
                if isinstance(v, dict) and v.get("dtype") not in VISUAL and isinstance(v.get("shape"), list)}
    names = sorted(declared) if features is None else [f for f in features if f in declared]
    out = []
    stats_file = root / "meta" / "stats.json"
    try:
        stats = json.loads(stats_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        stats = {}
    for f in names:
        want = tuple(declared[f]["shape"])
        for stat, value in sorted((stats.get(f) or {}).items()):
            if stat != "count" and _shape(value) != want:
                out.append(_mismatch(f, stat, _shape(value), want, f"{stats_file}", "dataset"))
                break
    files = sorted(glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
    if not files or not names:
        return out
    import pyarrow.parquet as pq

    for f in names:
        want = tuple(declared[f]["shape"])
        found = None
        for path in files:
            cols = [c for c in pq.read_schema(path).names if c.startswith(f"stats/{f}/") and not c.endswith("/count")]
            if not cols:
                continue
            table = pq.read_table(path, columns=cols + ["episode_index"])
            for col in cols:
                for ep, value in zip(table.column("episode_index").to_pylist(), table.column(col).to_pylist()):
                    if value is not None and _shape(value) != want:
                        found = _mismatch(f, col.rsplit("/", 1)[1], _shape(value), want,
                                          f"{root}/meta/episodes (episode {ep})", "episode")
                        break
                if found:
                    break
            if found:
                break
        if found:
            out.append(found)
    return out


def copy_problems(dataset_dir, template: dict) -> list[str]:
    """For binding: the merged copy a LeRobot run reads, checked for the features its policy
    normalises (`trained_features` of the run's train config)."""
    try:
        return [f"rowspace: {p}" for p in stats_problems(dataset_dir, trained_features(template))]
    except (OSError, ValueError) as exc:
        return [f"rowspace: cannot read the statistics of {dataset_dir}: {exc}"]
