"""Legacy records for the files the paper's runs consumed, built from a program's intermediates.

One builder per kind of locked file: the row space's index files and LeRobot row plan
(`rowspace_files`), the filtering schedules (`schedule_file`), the AWR weights (`awr_json_file`),
their LeRobot flat view (`flat_file`) and the CFG quality tags (`tags_file`). Each takes the
LegacyTarget of its lock id (axis_qvla.signals.legacy.legacy_target), so where the file goes, which
generator digest its metadata records and which roots index its config hash folds in all come from
the generator map (conf/signals/generators.json), by sha256.
`legacy_exports` is the one dispatcher every built-in calls: it picks each target's builder by kind
and refuses a target whose builder needs an intermediate the program did not produce.

Offline tier: numpy.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from axis_qvla.signals import otree
from axis_qvla.signals.blocks.awr import DEFAULT_HORIZON
from axis_qvla.signals.blocks.schedule import delta_quality_bins
from axis_qvla.signals.blocks.tags import q_given_task_entropy_ratio
from axis_qvla.signals.legacy import (TAG_DIALECTS, LegacyError, LegacyTarget, schedule_config_hash, tags_config_hash,
                                 tags_meta_for)
from axis_qvla.signals.methods.schedule import schedule_meta
from axis_qvla.signals.numerics import EXTENDED, log2_extended, log2_libm
from axis_qvla.signals.rowspace import ranges_doc, ranges_sha256, roots_sha256, trainable_rows
from axis_qvla.signals.schema import Budget, LegacyFile, RowSpace

SCHEDULE_KINDS = ("schedule", "lerobot_schedule")
# The log2 a tag file's entropy ratio is re-evaluated with, by the numerics its generator map entry declares.
RATIO_LOG2 = {"log2-extended": log2_extended, "log2-libm": log2_libm}


def rowspace_files(target: LegacyTarget, rows: RowSpace) -> tuple[LegacyFile, dict]:
    """An index file of the row space (its roots index or ranges file), or its LeRobot row plan."""
    if target.kind == "lerobot_rows":
        return (LegacyFile(target.format, target.lock_id, target.path, "npz",
                           {"$o": [["rows", {"$array": "legacy.rows"}]]}), {"rows": trainable_rows(rows)})
    if target.kind == "rowspace" and target.path == rows.roots_path:
        return LegacyFile(target.format, target.lock_id, target.path, rows.roots_codec, rows.roots), {}
    if target.kind == "rowspace" and target.path == rows.ranges_path:
        return LegacyFile(target.format, target.lock_id, target.path, rows.ranges_codec, ranges_doc(rows)), {}
    raise LegacyError(f"{target.lock_id} ({target.path}) is not an index file or row plan of frame index {rows.name}")


def schedule_file(target: LegacyTarget, *, plan, selection, mode: str, keep: float, budget: Budget, seed: int,
                  header, weights_sha256: str, rows: RowSpace) -> LegacyFile:
    """A filtering schedule: `rows` is the plan; `meta` is the schedule builder's own
    (axis_qvla/signals/methods/schedule.py `schedule_meta`) with the recorded generator digest and a
    config hash recomputed from it and the digests of the inputs read."""
    if target.kind not in SCHEDULE_KINDS:
        raise LegacyError(f"{target.lock_id} is a {target.kind} file, not a schedule")
    if target.generator_sha256 is None or target.roots_form is None:
        raise LegacyError(f"{target.lock_id}: the generator map records no generator digest or roots input")
    config_hash = schedule_config_hash(
        mode=mode, keep=keep, steps=budget.steps, batch=budget.global_batch, seed=seed,
        generator_sha256=target.generator_sha256, weights_sha256=weights_sha256,
        roots_sha256=roots_sha256(rows, target.roots_form), ranges_sha256=ranges_sha256(rows))
    meta = schedule_meta(
        np.asarray(plan), mode=mode, seed=int(seed), total_steps=budget.steps, batch=budget.global_batch,
        row_bins=delta_quality_bins([np.asarray(w) for w in selection.row_w]),
        episode_rows=[np.asarray(b) for b in selection.blocks],
        keep_fraction=selection.kept / max(selection.n_trainable, 1), config_hash=config_hash,
        reward_id=header["reward_id"], horizon=int(header.get("horizon", DEFAULT_HORIZON)), n_rows=rows.n_rows)
    meta["generator_sha256"] = target.generator_sha256
    return LegacyFile(target.format, target.lock_id, target.path, "npz",
                      {"$o": [["rows", {"$array": "plan"}], ["meta", {"$json": otree.encode(meta)}]]})


def awr_json_file(target: LegacyTarget, payload, *, name: str) -> tuple[LegacyFile, dict]:
    """An AWR weights file: `json.dumps(payload)`, its per-episode weights rebuilt from one float32
    array in the payload's own key order. A weight float32 cannot hold is refused: the file would
    not render back to the payload."""
    if target.kind != "awr_weights":
        raise LegacyError(f"{target.lock_id} is a {target.kind} file, not AWR weights")
    weights = payload["weights"]
    keys = list(weights)
    lengths = np.asarray([len(weights[k]) for k in keys], dtype=np.int64)
    exact = (np.concatenate([np.asarray(weights[k], dtype=np.float64) for k in keys])
             if keys else np.empty(0, dtype=np.float64))
    values = exact.astype(np.float32)
    if not np.array_equal(values.astype(np.float64), exact):
        raise LegacyError(f"{target.lock_id}: a weight is not a float32 value; it cannot be stored as one")
    pairs = []
    for k, v in payload.items():
        if k == "weights":
            pairs.append([k, {"$episode_floats": {"keys": keys, "lengths": f"legacy.{name}_lengths",
                                                  "values": f"legacy.{name}_values"}}])
        else:
            pairs.append([k, otree.encode(v)])
    return (LegacyFile(target.format, target.lock_id, target.path, "json", {"$o": pairs}),
            {f"{name}_lengths": lengths, f"{name}_values": values})


def flat_file(target: LegacyTarget) -> LegacyFile:
    """A LeRobot flat-weights file: the bundle's own `weights` (float32 per corpus row, NaN off-plan)."""
    if target.kind != "lerobot_awr_flat":
        raise LegacyError(f"{target.lock_id} is a {target.kind} file, not a LeRobot flat-weights file")
    return LegacyFile(target.format, target.lock_id, target.path, "npz", {"$o": [["weights", {"$array": "weights"}]]})


def tags_file(target: LegacyTarget, *, tag_array: str, prompts, meta_live: dict, weights_sha256: str,
              tag_rows: RowSpace, seed: int, granularity: str, plateau: str, ratio_units) -> LegacyFile:
    """A quality-tag file: the dense tags, the corpus instructions, and the live builder's metadata
    cut to the recorded version's keys, with its generator digest, a config hash recomputed from it,
    and -- where the generator map declares log2 numerics -- the entropy ratio re-evaluated over
    `ratio_units` (the `(q_units, task_units)` pair of axis_qvla.signals.methods.cfg_tags.entropy_units)
    with that log2 (axis_qvla.signals.numerics): `log2-extended`, a correctly rounded log2, or
    `log2-libm`, glibc 2.28's double log2, which is evaluated the same way on every host. On a host
    whose long double is plain double the log2-extended float cannot be reproduced, so that record
    is refused rather than written with a value that differs from the locked file; a bundle
    compiled elsewhere still exports the locked bytes on any host, because the record stores the
    value."""
    if target.kind != "quality_tags":
        raise LegacyError(f"{target.lock_id} is a {target.kind} file, not quality tags")
    if target.numerics == "log2-extended" and not EXTENDED:
        raise LegacyError(f"{target.lock_id}: this host's long double has no extended mantissa, so the entropy "
                          f"ratio SIGNALS.lock.json lists cannot be reproduced here; compile these labels on an "
                          f"x86-64 host")
    if target.numerics is not None and target.numerics not in RATIO_LOG2:
        raise LegacyError(f"{target.lock_id}: numerics {target.numerics!r} is not a log2 a tag file is evaluated with")
    dialect = TAG_DIALECTS.get(target.generator)
    if dialect is None or target.generator_sha256 is None or target.roots_form is None:
        raise LegacyError(f"{target.lock_id}: no recorded tag-builder version for generator {target.generator!r}")
    meta = tags_meta_for(meta_live, dialect)
    meta["config_hash"] = tags_config_hash(
        dialect=dialect, seed=seed, generator_sha256=target.generator_sha256, weights_sha256=weights_sha256,
        roots_sha256=roots_sha256(tag_rows, target.roots_form), ranges_sha256=ranges_sha256(tag_rows),
        granularity=granularity, plateau=plateau)
    meta["generator_sha256"] = target.generator_sha256
    if target.numerics is not None:
        q_units, task_units = ratio_units
        meta["q_given_task_entropy_ratio"] = float(q_given_task_entropy_ratio(
            q_units, task_units, log2=RATIO_LOG2[target.numerics]))
    return LegacyFile(target.format, target.lock_id, target.path, "npz", {"$o": [
        ["tag", {"$array": f"legacy.{tag_array}"}], ["prompts", {"$strings": [str(s) for s in prompts]}],
        ["meta", {"$json": otree.encode(meta)}]]})


def legacy_exports(targets, rows: RowSpace, *, schedule: Mapping | None = None, awr_doc: tuple | None = None,
                   weighted: bool = False, tags: tuple | None = None) -> tuple[tuple[LegacyFile, ...], dict]:
    """Every target's legacy record and the arrays they name, each built by its kind's builder.

    Index files and row plans need only the row space. The others need what the program produced,
    and a program that did not produce it passes nothing, so the target is refused rather than
    written from something else: a schedule needs the program's draw (`schedule`, `schedule_file`'s
    keyword arguments), an AWR weights file the document the program computed (`awr_doc`, its
    payload and the name its arrays are stored under), the LeRobot flat view per-row loss
    weights (`weighted`), and a quality-tag file the program's dense tags (`tags`, the uint8 tag
    array and `tags_file`'s other keyword arguments; the array is stored as `tags`)."""
    legacy, arrays = [], {}
    for t in targets:
        if t.kind in SCHEDULE_KINDS:
            if schedule is None:
                raise LegacyError(f"{t.lock_id} is a {t.kind} file, and this program draws no schedule")
            lf, a = schedule_file(t, rows=rows, **schedule), {}
        elif t.kind == "awr_weights":
            if awr_doc is None:
                raise LegacyError(f"{t.lock_id} is an AWR weights file, and this program writes none")
            payload, name = awr_doc
            lf, a = awr_json_file(t, payload, name=name)
        elif t.kind == "lerobot_awr_flat":
            if not weighted:
                raise LegacyError(f"{t.lock_id} is a flat-weights file, and this program weighs no rows")
            lf, a = flat_file(t), {}
        elif t.kind == "quality_tags":
            if tags is None:
                raise LegacyError(f"{t.lock_id} is a quality-tag file, and this program tags no rows")
            tag, kw = tags
            lf, a = tags_file(t, tag_array="tags", **kw), {"tags": tag}
        else:
            lf, a = rowspace_files(t, rows)
        legacy.append(lf)
        arrays.update(a)
    return tuple(legacy), arrays
