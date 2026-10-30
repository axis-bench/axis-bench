"""Pre-launch binding: what `qvla train` refuses before any GPU time.

A run binds one set of compiled labels. Binding loads them (axis_qvla.signals.bundle.load_bundle
verifies their content id against their directory and every array's dtype, shape and digest) and
refuses, with every problem listed at once:
  - a frame-index mismatch: the labels were compiled over another frame index than the run trains
    on;
  - a missing capability: the labels require something the backbone's consumer does not do;
  - weights reduced another way than the backbone reduces them (pi0.5 `global_ratio`, SmolVLA and
    X-VLA `per_rank_ratio`);
  - a plan that does not fit the run: its batch is not the global batch, it has fewer steps than
    the run trains, or more steps without --smoke (only a smoke run trains a prefix);
  - a per-rank batch that does not multiply out to the global batch;
  - a planned frame without a finite weight, a forbidden code (255), a code with no text, more
    conditioned fields than the consumer reads;
  - guidance at serve time with no unconditional (code 0) sample to guide away from. Pure
    conditioning (REALALIGN-style tags, no dropout) is accepted when the labels serve without
    guidance;
  - for SmolVLA / X-VLA, compiled labels with no lerobot-legacy file, a file no train_config field
    reads, two files for one field, a run-spec override of a field qvla train writes itself, and a
    `sample_weighting` that is not null for labels exporting no weights or not `axis_flat` for
    labels that do (axis_qvla.train.lerobot_cfg.override_problems); and for a file no paper run read, a
    run seed other than the one a row set's plan was drawn with, a tag file with no untagged frame
    without `axis_quality_unconditional = false`, and an `axis_expected_mode` or
    `axis_expected_reward` other than the one its schedule or tag file records;
  - a `data.quality_tag` on a pi0.5 run spec that names compiled labels
    (axis_qvla.train.generate.quality_tag_problem: only the finetune labels binding compiles read it, so
    it would change nothing).
The schema already refuses weights without a reducer and codes without a vocabulary
(axis_qvla.signals.schema.SignalBundle), so those are not repeated here.

Where the compiled labels come from: a run spec names them by content id, read from the
pretraining data's bundles/ when that copy has them, else from this machine's local bundles
(axis_qvla.train.ids.bundle_dirs; same id, same bytes). A pi0.5 finetune run that names none trains on
the paper's finetune plan, compiled here from its recipe (axis_qvla.train.ft_bundle) and written to the
local bundles (axis_qvla.train.ids.local_bundles_root), never into a mirrored repository the checksum
manifest lists. The compile
reads the roots index, the ranges file and each task's meta/ files only once each is the file the
repository pins in BENCH_FILES.json (axis_qvla.train.cut_copy.pinned_input_problems), so the digests
it records are the release's, not whatever the local mirror holds. A run naming a `local:` input is
refused first (axis_qvla.train.generate.check_renderable).

Binding also checks what the generated config will open: the init checkpoint (a local run only if it
trained its whole declared budget), the norm-stats digest and the frame-index files -- for a pi0.5
flat frame index, the roots index and ranges file the labels' `inputs` name by digest, and the
decode rule its program records, since its flat indices are frames of the ranges walk over them
(axis_qvla.train.ft_bundle.FLAT_ROW_DECODE); for a merged LeRobot copy this machine cuts, every field of
its cut record (axis_qvla.train.cut_copy); for a plain copy not built yet, whether the pinned LeRobot
fork's merge would accept the frame index's task directories at all, and for one `qvla train` merged,
that it was merged from the roots index on disk (axis_qvla.train.plain_copy); for a merged copy that is
there, that the statistics of the features the policy normalises have those features' declared shapes
(axis_qvla.train.lerobot_stats). Nothing is written until every check passes. It writes beside the mode's
rendered files (axis_qvla.train.ids.rendered_dir, so a smoke never overwrites a real run's): binding.json, and for SmolVLA / X-VLA the labels' lerobot-legacy files at their
published relative paths, each mapped to the train_config field the pinned LeRobot fork reads it
from. A paper run's released file keeps its released name (the fork's label checksum guard then logs
the released id); any other file is prefixed with its compiled-labels id, so the guard never takes
it for another generation of a paper file.

Core tier at import (stdlib only); numpy and axis_qvla.signals are imported inside the functions.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import pathlib
from collections.abc import Mapping

from axis_qvla.core.canon import dumps_canonical
from axis_qvla.core.paths import Paths
from axis_qvla.data.layout import default_dest
from axis_qvla.train import ids
from axis_qvla.train.armspec import ArmSpec
from axis_qvla.train.cut_copy import CUT_RECORD, check_cut_copy, pinned_input_problems  # noqa: F401  (CUT_RECORD: re-export)
from axis_qvla.train.ft_bundle import FLAT_ROW_DECODE, INPUT_RANGES, INPUT_ROOTS
from axis_qvla.train.lerobot_stats import copy_problems as copy_stats_problems
from axis_qvla.train.plain_copy import record_problems as plain_record_problems
from axis_qvla.train.plain_copy import rowspace_problems as plain_copy_problems
from axis_qvla.train.generate import GenerateError, check_renderable, openpi_asset_id, quality_tag_problem, steps_for
from axis_qvla.train.recipes import (Recipe, RecipeError, check_arm_against_recipe, load_recipe, load_rowspaces,
                                load_template, resolved_fields)
from axis_qvla.train.runs import finished_problems, not_a_dir as _not_a_dir

CAPABILITIES = {
    "pi05-openpi": frozenset({"plan", "loss_weights", "cond_prompt", "guidance"}),
    "smolvla-lerobot": frozenset({"plan", "loss_weights", "cond_prompt"}),
    "xvla-lerobot": frozenset({"plan", "loss_weights", "cond_prompt"}),
}
# The reducer each backbone applies to weighted losses: openpi normalises over the global batch;
# the LeRobot line divides by the per-GPU weight sum, then DDP averages the GPUs.
REDUCERS = {"pi05-openpi": "global_ratio", "smolvla-lerobot": "per_rank_ratio", "xvla-lerobot": "per_rank_ratio"}
# For a LeRobot legacy file no paper run read (lock_id None): file name prefix -> the train_config
# field the LeRobot line reads it from. A paper file's field is its SIGNALS.lock row's
# `consumers.lerobot` (a CFG tag file is published as `.../tags.npz`, so its name alone says little).
LEGACY_FIELDS = (("lerobot_rows", "axis_rows_path"), ("ft_rows_all", "axis_rows_path"),
                 ("lerobot_awr_flat_", "sample_weighting.extra_params.weights_path"),
                 ("lerobot_schedule_", "axis_schedule_path"), ("quality_", "axis_quality_path"),
                 ("tags", "axis_quality_path"))
BINDING_SCHEMA = "qvla.binding/1"
BINDING_FILE = "binding.json"


class BindError(ValueError):
    """The arm's bundle or inputs do not fit the run; nothing has been launched."""


@dataclasses.dataclass(frozen=True)
class Binding:
    """What a run was bound to. `bundle` is the bundle it trains on (the arm's own, or the one
    binding compiled) and `bundle_dir` the directory it was read from or written to; for SmolVLA /
    X-VLA `signal_files` maps train_config fields to the exported legacy files."""

    arm: str
    bundle: str
    bundle_dir: str
    rowspace: str
    steps: int
    global_batch: int
    per_rank_batch: int
    world: int
    reducer: str
    signal_files: Mapping[str, str]


def legacy_field(record, signals_lock: dict) -> str:
    """The train_config field a LeRobot legacy record (axis_qvla.signals.schema.LegacyFile) goes to."""
    if record.lock_id is not None:
        row = next((f for f in signals_lock["files"] if f["id"] == record.lock_id), None)
        field = None if row is None else row.get("consumers", {}).get("lerobot")
        if field is None:
            raise BindError(f"{record.lock_id} is not a LeRobot label file in SIGNALS.lock.json")
        return field
    name = record.path.rsplit("/", 1)[-1]
    for prefix, field in LEGACY_FIELDS:
        if name.startswith(prefix):
            return field
    raise BindError(f"{name!r} is not a legacy LeRobot label file the pinned LeRobot fork reads")


def run_shape(spec: ArmSpec, recipe: Recipe, *, smoke: bool) -> tuple[int, int, int, int]:
    """(steps, global batch, per-rank batch, world) the run will train with: the recipe's budget
    with the arm's overrides applied the way the config generator applies them, and its step floor
    (axis_qvla.train.generate.steps_for), so a budget the render refuses is refused before binding."""
    if spec.backbone == "pi05-openpi":
        f = resolved_fields(recipe, spec)
        steps, batch = int(f["num_train_steps"]), int(f["batch_size"])
        return steps_for(steps, smoke=smoke), batch, batch, 1
    from axis_qvla.train.lerobot_cfg import apply_override

    shape = {"steps": recipe.steps, "batch_size": recipe.per_rank_batch,
             "parallelism": {"dp_replicate": recipe.world}}
    for key in sorted(spec.overrides):
        if key.split(".", 1)[0] in shape:
            apply_override(shape, key, spec.overrides[key], source=spec.id)
    try:
        steps, per_rank = int(shape["steps"]), int(shape["batch_size"])
        world = int(shape["parallelism"]["dp_replicate"])
    except (KeyError, TypeError, ValueError):
        raise GenerateError(f"{spec.id}: the overrides leave no integer steps, batch_size and "
                            f"parallelism.dp_replicate") from None
    return steps_for(steps, smoke=smoke), recipe.global_batch, per_rank, world


def check_bundle(spec: ArmSpec, bundle, *, steps: int, global_batch: int, per_rank: int, world: int,
                 smoke: bool) -> list[str]:
    """Every reason `bundle` cannot train this run; [] when it fits. Pure: no filesystem."""
    import numpy as np

    from axis_qvla.signals.bundle import rowspace_id

    out = []
    rs = bundle.rowspace
    got = f"{rs.name}#sha256:{rowspace_id(rs)}"
    if got != spec.rowspace:
        out.append(f"rowspace: the labels were compiled over frame index {got}, the run trains on {spec.rowspace}")
    missing = sorted(set(bundle.requires) - CAPABILITIES[spec.backbone])
    if missing:
        out.append(f"capability: the compiled labels require {missing}, which the {spec.backbone} consumer does not implement")
    if bundle.weights is not None and bundle.reducer != REDUCERS[spec.backbone]:
        out.append(f"reducer: {bundle.reducer!r}; {spec.backbone} reduces weighted losses as {REDUCERS[spec.backbone]!r}")
    if per_rank * world != global_batch:
        out.append(f"batch: per-rank batch {per_rank} x world {world} != global batch {global_batch}")
    plan = np.asarray(bundle.plan)
    if plan.dtype != np.int64 or plan.ndim != 2:
        return out + [f"plan: expected int64 [steps, batch], got {plan.dtype} {plan.shape}"]
    n_steps, batch = plan.shape
    if batch != global_batch:
        out.append(f"plan: batch {batch} != the run's global batch {global_batch}")
    if n_steps < steps:
        out.append(f"plan: {n_steps} steps, the run trains {steps}")
    elif n_steps > steps and not smoke:
        out.append(f"plan: {n_steps} steps but the run trains {steps}; only a smoke run trains a prefix")
    used = plan[:steps]
    n_rows = int(rs.n_rows)
    if used.size and (int(used.min()) < 0 or int(used.max()) >= n_rows):
        out.append(f"plan: rows outside [0, {n_rows})")
        used = used[(used >= 0) & (used < n_rows)]
    if bundle.weights is not None:
        w = np.asarray(bundle.weights)
        if w.shape != (n_rows,) or w.dtype != np.float32:
            out.append(f"weights: expected float32 [{n_rows}], got {w.dtype} {w.shape}")
        elif used.size and not np.isfinite(w[used]).all():
            out.append("weights: a planned row has no finite weight")
    if bundle.cond is not None:
        c = np.asarray(bundle.cond)
        fields = tuple(bundle.vocab.fields)
        if c.dtype != np.uint8 or c.ndim != 3 or c.shape != (n_steps, batch, len(fields)):
            out.append(f"cond: expected uint8 [{n_steps}, {batch}, {len(fields)}], got {c.dtype} {c.shape}")
        else:
            cu = c[:steps]
            if bool((cu == 255).any()):
                out.append("cond: a planned sample carries the forbidden code 255")
            for i, f in enumerate(fields):
                unknown = sorted(set(np.unique(cu[..., i]).tolist()) - set(bundle.vocab.text[f]) - {0, 255})
                if unknown:
                    out.append(f"cond: field {f!r} codes {unknown} have no vocabulary text")
            if spec.backbone == "pi05-openpi" and len(fields) != 1:
                out.append(f"cond: {len(fields)} condition fields; the pi0.5 consumer conditions on one")
            if any(s.guided for s in bundle.serve) and not bool((cu == 0).any()):
                out.append("cond: the compiled labels serve with guidance but no planned sample is unconditional (code 0)")
    return out


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_run_init(spec: ArmSpec, paths: Paths) -> list[str]:
    """A local run is an init only if it trained its whole declared budget (never a smoke run)."""
    return finished_problems(spec.init, spec.backbone, paths)


def _check_init(spec: ArmSpec, paths: Paths) -> list[str]:
    if spec.init is None:
        return []
    if spec.init.startswith("run:"):
        return _check_run_init(spec, paths)
    pi05 = spec.backbone == "pi05-openpi"
    try:
        if spec.init.startswith("base:"):
            if not pi05:
                return []    # a LeRobot base is a Hub id, resolved offline from the base-weights copy
            want, hint = (pathlib.Path(ids.openpi_init_params(paths, spec.init)),
                          f"`qvla data get base-weights --include {spec.init[len('base:'):]}`")
        else:
            want = ids.checkpoint_dir(paths, spec.init) / ("params" if pi05 else "pretrained_model")
            hint = f"fetch {spec.init} with `qvla data get`"
    except ValueError as exc:
        return [f"init: {exc}"]
    why = _not_a_dir(want)
    return [] if why is None else [f"init: {want} {why} ({hint})"]


def _check_norm(spec: ArmSpec, paths: Paths, recipe: Recipe | None) -> list[str]:
    """The norm statistics the openpi config opens: `<norm dir>/<asset id>/norm_stats.json`, with
    the asset id the config generator renders. LeRobot configs read no norm artifact."""
    if spec.norm is None or spec.backbone != "pi05-openpi":
        return []
    try:
        _, digest = ids.content_id(spec.norm)
        asset = openpi_asset_id(spec, resolved_fields(recipe or load_recipe(spec.preset), spec))
        f = ids.norm_dir(paths, spec.norm) / str(asset) / "norm_stats.json"
    except ValueError as exc:
        return [f"norm: {exc}"]
    try:
        got = _sha256(f)
    except OSError as exc:
        return [f"norm: cannot read {f} ({exc.__class__.__name__})"]
    return [] if got == digest else [f"norm: {f} has sha256 {got[:12]}, the run names {digest[:12]}"]


def _check_flat_inputs(bundle, lay: Mapping[str, pathlib.Path], roots_sha: str) -> list[str]:
    """A flat pi0.5 row r is the r-th row of the ranges walk over the roots index
    (axis_qvla.train.ft_bundle.FLAT_ROW_DECODE), so the same plan means other frames over other files or
    under another rule: the bundle must record that rule, the one the openpi line reads its rows
    by, and name, by digest, the roots index and ranges file this machine's layout holds."""
    params = bundle.program.get("params") if isinstance(bundle.program, Mapping) else None
    decode = params.get("row_decode") if isinstance(params, Mapping) else None
    out = [] if decode == FLAT_ROW_DECODE else [
        f"rowspace: the compiled labels decode their flat rows as {decode!r}; the pinned openpi fork reads flat "
        f"row r as {FLAT_ROW_DECODE!r}"]
    want = {k: bundle.inputs.get(k) for k in (INPUT_ROOTS, INPUT_RANGES)}
    missing = [k for k, v in want.items() if not v]
    if missing:
        return out + [f"rowspace: the labels name no {' or '.join(missing)} digest in its inputs, so which frames "
                      f"its flat rows are cannot be checked against {lay['roots']} and {lay['ranges']}"]
    try:
        ranges_sha = _sha256(lay["ranges"])
    except OSError as exc:
        return out + [f"rowspace: cannot read {lay['ranges']} ({exc.__class__.__name__})"]
    return out + [f"rowspace: {path} has sha256 {got[:12]}, the labels' flat rows were cut by the {key} {want[key][:12]}"
                  for key, path, got in ((INPUT_ROOTS, lay["roots"], roots_sha), (INPUT_RANGES, lay["ranges"], ranges_sha))
                  if got != want[key]]


def _check_rowspace(spec: ArmSpec, paths: Paths, bundle, recipe: Recipe | None = None, *,
                    copy_pending: bool = False) -> list[str]:
    try:
        rs = ids.rowspace_dirs(paths, spec.rowspace)
    except ValueError as exc:
        return [f"rowspace: {exc}"]
    if spec.backbone != "pi05-openpi":
        lay = load_rowspaces()[ids.content_id(spec.rowspace)[0]]
        why = _not_a_dir(rs["lerobot_dataset"])
        if why is not None:
            # A copy not built yet: say now whether the line's merge would accept its inputs. With
            # `copy_pending` the merge is printed before the training command, so only that is checked.
            pending = plain_copy_problems(rs) if lay.lerobot_copy == "plain" else []
            return pending if copy_pending else [f"rowspace: {rs['lerobot_dataset']} {why}"] + pending
        stats = copy_stats_problems(rs["lerobot_dataset"], load_template(recipe or load_recipe(spec.preset)))
        if lay.lerobot_copy == "plain":
            return stats + plain_record_problems(rs)
        if lay.lerobot_copy != "cut":
            return stats
        repo_root = ids.local_derived(paths, "") if lay.repo == "local" else default_dest(lay.repo, paths)
        return stats + check_cut_copy(lay, rs, repo_root)
    try:
        got = _sha256(rs["roots"])
    except OSError as exc:
        return [f"rowspace: cannot read {rs['roots']} ({exc.__class__.__name__})"]
    out = []
    if bundle is not None and bundle.rowspace.kind == "corpus":
        from axis_qvla.signals.rowspace import roots_bytes

        want = hashlib.sha256(roots_bytes(bundle.rowspace)).hexdigest()
        if got != want:
            out.append(f"rowspace: {rs['roots']} has sha256 {got[:12]}, the labels' frame index names {want[:12]}")
    elif bundle is not None:
        out += _check_flat_inputs(bundle, rs, got)
    why = _not_a_dir(rs["corpus_base"])
    if why is not None:
        out.append(f"rowspace: the episodes directory {rs['corpus_base']} {why}")
    return out


def check_inputs(spec: ArmSpec, paths: Paths, bundle, *, recipe: Recipe | None = None,
                 copy_pending: bool = False) -> list[str]:
    """What the generated config will open: present, and the bytes the arm names. `bundle` may be
    None (it could not be loaded): the checks that compare against it are then skipped.
    `copy_pending`: see `bind`."""
    return (_check_init(spec, paths) + _check_norm(spec, paths, recipe)
            + _check_rowspace(spec, paths, bundle, recipe, copy_pending=copy_pending))


def _first_present(dirs: tuple[pathlib.Path, pathlib.Path]) -> pathlib.Path:
    """The published directory when it is there, else the local one (published first: same id,
    same bytes, and the published copy is the lock-verified one). A published directory that
    cannot be read is refused, never passed over."""
    published, local = dirs
    try:
        os.stat(published)
    except (FileNotFoundError, NotADirectoryError):
        return local
    except OSError as exc:
        raise BindError(f"bundle: cannot read {published} ({exc.__class__.__name__})") from None
    return published


def compile_finetune_bundle(spec: ArmSpec, recipe: Recipe, paths: Paths):
    """The paper's pi0.5 finetune signal for an arm that names no bundle (axis_qvla.train.ft_bundle), over
    a flat finetune row space: its rows are the finetune set's trainable frames, so the episode
    table is cut by the layout's ranges file and both files' digests are recorded in the bundle."""
    from axis_qvla.signals.bundle import load_rowspace
    from axis_qvla.train import ft_bundle

    try:
        f = resolved_fields(recipe, spec)
        lay = ids.rowspace_dirs(paths, spec.rowspace)
        repo = load_rowspaces()[ids.content_id(spec.rowspace)[0]].repo
    except ValueError as exc:
        raise BindError(f"bundle: cannot compile the finetune's labels: {exc}") from None
    rdir = _first_present(ids.rowspace_record_dirs(paths, spec.rowspace))
    try:
        rs = load_rowspace(rdir)
        if rs.kind != "flat":
            raise BindError(f"bundle: {spec.rowspace} is a {rs.kind} frame index; binding compiles a "
                            f"finetune's labels over a flat finetune frame index only, so name labels compiled over it")
        unpinned = pinned_input_problems(lay, ids.local_derived(paths, "") if repo == "local"
                                         else default_dest(repo, paths), published=repo != "local")
        if unpinned:
            raise BindError("bundle: cannot compile the finetune's labels from files that are not the "
                            "repository's: "
                            + "; ".join(unpinned))
        ep, fr, prompts = ft_bundle.episode_table(lay["roots"], lay["corpus_base"], lay["ranges"])
        inputs = {INPUT_ROOTS: _sha256(lay["roots"]), INPUT_RANGES: _sha256(lay["ranges"])}
        return ft_bundle.compile_finetune(rs, episode_index=ep, frame_index=fr, prompts=prompts,
                                          steps=int(f["num_train_steps"]), batch=int(f["batch_size"]),
                                          seed=spec.seed, quality_tag=f.get("data.quality_tag"), inputs=inputs)
    except BindError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise BindError(f"bundle: cannot compile the finetune's labels from the frame index {rdir} and "
                        f"the episode roots {lay['roots']}: {exc}") from None


def _lerobot_records(bundle) -> tuple[list[tuple[str, object]], list[str]]:
    """`(train_config field, record)` of each lerobot-legacy file of `bundle`, and every reason
    they cannot be exported (listed with the run's other problems, before anything is written)."""
    from axis_qvla.signals.lerobot_files import refusal
    from axis_qvla.train.parity import load_signals_lock

    records = [f for f in bundle.legacy if f.format == "lerobot-legacy"]
    if not records:
        why = refusal(bundle)
        return [], ["export: the compiled labels carry no lerobot-legacy files, the only label files the pinned "
                    "LeRobot fork reads" + (f": they hold {why}" if why else
                                            "; a composed method carries them where that fork can train it "
                                            "(axis_qvla.signals.lerobot_files)")]
    lock = load_signals_lock()
    mapped, problems, seen = [], [], {}
    for rec in records:
        try:
            field = legacy_field(rec, lock)
        except BindError as exc:
            problems.append(f"export: {exc}")
            continue
        if field in seen:
            problems.append(f"export: {seen[field]} and {rec.path} both go to {field}")
            continue
        seen[field] = rec.path
        mapped.append((field, rec))
    return mapped, problems


def _composed_problems(spec: ArmSpec, recipe: Recipe, bundle, mapped: list[tuple[str, object]]) -> list[str]:
    """What the pinned LeRobot fork checks of a file no paper run read, when it starts: its row sampler
    draws a row set in the order the run's seed gives, which is the labels' plan only for the seed
    they were compiled with; a tag file with no untagged frame trains only with
    `axis_quality_unconditional = false`; and a schedule's `mode` and `reward_id`, and a tag file's
    `reward_id`, must be the run's `axis_expected_mode` and `axis_expected_reward`
    (axis_qvla.signals.lerobot_files)."""
    import numpy as np

    from axis_qvla.signals.lerobot_files import record_meta
    from axis_qvla.train.lerobot_cfg import resolved_template

    cfg, out = resolved_template(spec, recipe), []
    for field, rec in mapped:
        if rec.lock_id is not None:
            continue
        if field == "axis_rows_path" and spec.seed != bundle.seed:
            out.append(f"seed: the pinned LeRobot fork draws {rec.path} in the order its row sampler gives the run's "
                       f"seed {spec.seed}; the compiled labels' plan was drawn with seed {bundle.seed}")
        if field == "axis_quality_path" and cfg.get("axis_quality_unconditional", True) and not bool(
                (np.asarray(bundle.legacy_arrays.get("tags", ())) == 0).any()):
            out.append(f"recipe: {rec.path} tags every frame (no code 0), which the pinned LeRobot fork trains only "
                       f"as pure conditioning: declare axis_quality_unconditional = false in [recipe.overrides]")
        meta = record_meta(rec) or {}
        bound = (("axis_expected_mode", "mode"),) if field == "axis_schedule_path" else ()
        for key, name in bound + (("axis_expected_reward", "reward_id"),):
            want = meta.get(name)
            if want is not None and cfg.get(key) != want:
                out.append(f"recipe: {rec.path} records {name} {want!r}, which the pinned LeRobot fork requires "
                           f"as {key} (the config has {cfg.get(key)!r}); declare {key} = {want!r} in "
                           f"[recipe.overrides]")
    return out


def _export_lerobot(spec: ArmSpec, bundle, bundle_ref: str, out: pathlib.Path,
                    mapped: list[tuple[str, object]]) -> dict[str, str]:
    """Write the mapped lerobot-legacy files under `out`; train_config field -> file. A paper arm's
    locked file keeps its locked path; any other is prefixed with the bundle id."""
    from axis_qvla.signals.bundle import export_legacy

    prefix = bundle_ref[len("sha256:"):][:12]
    named = []
    for _, rec in mapped:
        if spec.track == "paper" and rec.lock_id is not None:
            named.append(rec)
        else:
            head, _, name = rec.path.rpartition("/")
            named.append(dataclasses.replace(rec, path=f"{head}/{prefix}-{name}" if head else f"{prefix}-{name}"))
    others = tuple(f for f in bundle.legacy if f.format != "lerobot-legacy")
    try:
        written = export_legacy(dataclasses.replace(bundle, legacy=tuple(named) + others), "lerobot-legacy", out)
    except (OSError, ValueError) as exc:
        raise BindError(f"{spec.id}: cannot export {bundle_ref} for the pinned LeRobot fork: {exc}") from None
    return {field: str(p) for (field, _), p in zip(mapped, written, strict=True)}


def binding_path(paths: Paths, arm_id: str, *, smoke: bool = False) -> pathlib.Path:
    return ids.rendered_dir(paths, arm_id, smoke=smoke) / BINDING_FILE


def bind(spec: ArmSpec, recipe: Recipe, paths: Paths, *, smoke: bool = False, record: bool = True,
         copy_pending: bool = False) -> Binding:
    """Bind `spec` to its bundle: raise BindError listing every problem, or write and return the
    Binding (module docstring). `record=False` checks and returns the Binding without writing
    binding.json, for a launch that must first be compared with the run's recorded files (a resume).
    `copy_pending`: the merged LeRobot copy is built by a merge printed before the training command
    (`qvla train --render`), so its absence is not a refusal."""
    from axis_qvla.signals.bundle import load_bundle, write_bundle
    from axis_qvla.train.lerobot_cfg import owned_override_problems, override_problems

    try:
        check_renderable(spec)
        check_arm_against_recipe(spec, recipe)
        steps, global_batch, per_rank, world = run_shape(spec, recipe, smoke=smoke)
    except (GenerateError, RecipeError) as exc:
        raise BindError(str(exc)) from None
    problems: list[str] = []
    if spec.backbone == "pi05-openpi":
        tag = quality_tag_problem(spec, resolved_fields(recipe, spec))
        problems += [tag] if tag else []
    bundle, bdir, compiled = None, None, False
    try:
        if spec.bundle is not None:
            bdir = _first_present(ids.bundle_dirs(paths, spec.bundle))
            try:
                bundle = load_bundle(bdir)
            except (OSError, ValueError) as exc:
                raise BindError(f"bundle: cannot load {spec.bundle} from {bdir}: {exc}") from None
        elif spec.backbone == "pi05-openpi" and spec.stage == "finetune":
            bundle, compiled = compile_finetune_bundle(spec, recipe, paths), True
        else:
            raise BindError("bundle: the run names no compiled labels, and only a pi0.5 finetune's are compiled from "
                            "its recipe")
    except BindError as exc:
        problems.append(str(exc))
    mapped: list[tuple[str, object]] = []
    if bundle is not None:
        problems += check_bundle(spec, bundle, steps=steps, global_batch=global_batch, per_rank=per_rank,
                                 world=world, smoke=smoke)
        if spec.backbone != "pi05-openpi":
            mapped, more = _lerobot_records(bundle)
            problems += more
            try:
                problems += override_problems(spec, recipe, [field for field, _ in mapped])
                problems += _composed_problems(spec, recipe, bundle, mapped)
            except GenerateError as exc:
                problems.append(str(exc))
    elif spec.backbone != "pi05-openpi":
        problems += owned_override_problems(spec)
    problems += check_inputs(spec, paths, bundle, recipe=recipe, copy_pending=copy_pending)
    if problems:
        raise BindError(f"{spec.id}: refused before launch:\n  " + "\n  ".join(problems))
    if compiled:
        root = ids.local_bundles_root(paths)
        try:
            bundle_ref = "sha256:" + write_bundle(bundle, root)
        except (OSError, ValueError) as exc:
            raise BindError(f"{spec.id}: cannot write the compiled finetune labels under {root}: {exc}") from None
        bdir = ids.bundle_dirs(paths, bundle_ref)[1]
    else:
        bundle_ref = spec.bundle
    out = ids.rendered_dir(paths, spec.id, smoke=smoke)
    signal_files = {} if spec.backbone == "pi05-openpi" else _export_lerobot(spec, bundle, bundle_ref, out, mapped)
    binding = Binding(arm=spec.id, bundle=bundle_ref, bundle_dir=str(bdir), rowspace=spec.rowspace, steps=steps,
                      global_batch=global_batch, per_rank_batch=per_rank, world=world,
                      reducer=bundle.reducer, signal_files=signal_files)
    if not record:
        return binding
    dest = binding_path(paths, spec.id, smoke=smoke)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_bytes(dumps_canonical({"schema": BINDING_SCHEMA, **dataclasses.asdict(binding), "smoke": smoke}))
    tmp.replace(dest)
    return binding
