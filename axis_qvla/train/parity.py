"""Config hash-parity of the paper runs against conf/release/paper_configs.lock.json.

The lock holds, for every paper training run, its configuration projected onto the fields the
paper's runs were compared on. pi0.5 rows use openpi's dotted TrainConfig names; SmolVLA / X-VLA
rows are the recorded train_config.json flattened to dotted keys, minus LEROBOT_NOT_COMPARED.
Fields the training code names by content are logical values: `{"init": <ArmSpec init>}`,
`{"rowspace": <name>}`, `{"signal": <SIGNALS.lock id>}`. Numbers compare by value (an int and a
float that are equal project to the same bytes), because the recorded configs spell 10 and 10.0
for the same setting.

Core tier: stdlib only.
"""
from __future__ import annotations

import json
import pathlib

from axis_qvla.core.canon import dumps_canonical, sha256_bytes

REPO = pathlib.Path(__file__).resolve().parents[2]
LOCK_PATH = REPO / "conf" / "release" / "paper_configs.lock.json"
SIGNALS_LOCK_PATH = REPO / "conf" / "release" / "SIGNALS.lock.json"
LOCK_SCHEMA = "qvla.paper_configs_lock/1"

# Keys of a LeRobot train_config.json that do not change what is trained: where outputs and logs
# go, how often a checkpoint is written, evaluation during training, data-loader worker plumbing,
# the local dataset location and the video decoder the machine has.
LEROBOT_NOT_COMPARED = (
    "output_dir", "job_name", "resume", "checkpoint_path", "log_freq", "save_freq", "save_checkpoint",
    "save_checkpoint_to_hub", "env", "env_eval_freq", "eval", "eval_steps", "max_eval_samples", "wandb",
    "job", "num_workers", "prefetch_factor", "persistent_workers", "dataloader_multiprocessing_context",
    "dataset.root", "dataset.video_backend", "policy.device", "policy.push_to_hub", "policy.repo_id",
    "policy.private", "policy.tags", "policy.license", "reward_model",
)


def flatten(d: dict, prefix: str = "") -> dict[str, object]:
    """Nested dict -> {dotted key: leaf}; lists and empty dicts are leaves."""
    out: dict[str, object] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict) and v:
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def not_compared(key: str) -> bool:
    return any(key == p or key.startswith(p + ".") for p in LEROBOT_NOT_COMPARED)


def normalized(v: object) -> object:
    """Numbers as floats (bools untouched), recursively."""
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list):
        return [normalized(x) for x in v]
    if isinstance(v, dict):
        return {k: normalized(x) for k, x in v.items()}
    raise TypeError(f"cannot compare a value of type {type(v).__name__}")


def lock_sha256(fields: dict[str, object]) -> str:
    return sha256_bytes(dumps_canonical(normalized(fields)))


def _json_or_none(v: object) -> object:
    """`json.loads(v)`, or `None` for anything that is not a JSON-string -- used where a malformed
    declared value must be reported as a mismatch, never raise."""
    if not isinstance(v, str):
        return None
    try:
        return json.loads(v)
    except ValueError:
        return None


def load_lock(path: pathlib.Path = LOCK_PATH) -> dict:
    doc = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != LOCK_SCHEMA:
        raise ValueError(f"{path}: schema is {doc.get('schema')!r}, expected {LOCK_SCHEMA!r}")
    return doc


# --- the run side (an ArmSpec plus its recipe, projected the way the lock rows are) -------------

# Fork-only fields of the openpi line the paper trained on, at the inert values every paper run had.
# The clean openpi line has none of them (knowledge insulation and the EEF / joint-9 action spaces
# are not part of it), so a run projects them as constants.
DROPPED_INERT = {"model.knowledge_insulation": False, "model.ki_fast_loss_weight": 1.0,
                 "data.eef_action": False, "data.joint9_action": False}
FT_SIGNALS = ("ft:plain", "ft:tag5")
_OPENPI_SIGNAL_FIELD = {"awr_weights": "data.awr_weights", "schedule": "data.schedule_path",
                        "quality_tags": "data.quality_path"}
_ABSENT = "<absent>"
# The published normalisation statistics every paper pretrain/finetune cites, by stage -- the
# NAME half of [norm].artifact (the digest half is checked against the registry's own
# norm_stats_sha256, never invented here: see _check_norm).
NORM_NAME = {"pretrain": "libero5k_v2_d8", "finetune": "heldout24_qual_v2"}
# data.image_center_crop (a recipe field, already compared) <-> [data].image (an ArmSpec-level
# declaration no renderer reads yet): still refused on a mismatch, since it is the one field
# [data].image corresponds to.
_IMAGE_FOR_CENTER_CROP = {True: "center_crop", False: "raw"}
# data.eef_action / data.joint9_action (both always False -- DROPPED_INERT) <-> [data].action_space:
# the one combination every paper run used encodes "droid8"; same pattern as [data].image above.
_ACTION_SPACE_FOR = {(False, False): "droid8"}


def load_signals_lock(path: pathlib.Path = SIGNALS_LOCK_PATH) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def lerobot_rowspace(stage: str, signals: list[str]) -> str:
    """The row-space name a LeRobot run's `paper.signals` implies: the lock row carries no
    `rowspace` field for SmolVLA / X-VLA (unlike pi0.5), so this is the one place -- shared by the
    writer (which assigns `[data].rowspace` from it) and check_arm (which verifies it) -- that
    says what a LeRobot run's signals mean for its row space."""
    if stage == "finetune":
        return "heldout24_ft"
    for s, name in (("lerobot.rows_s1000", "libero5k_v2_d8.sub1000"), ("lerobot.rows_s2500", "libero5k_v2_d8.sub2500")):
        if s in signals:
            return name
    return "libero5k_v2_d8"


def _signal_rows(spec, signals_lock: dict) -> list[dict]:
    files = {f["id"]: f for f in signals_lock["files"]}
    out = []
    for s in spec.paper.get("signals", ()):
        if s in FT_SIGNALS:
            # ft:plain / ft:tag5 name pi0.5's own quality-tag presentation (tied to its
            # data.quality_tag field); a LeRobot run -- pretrain or finetune -- has no such
            # concept at all, so both the backbone and the stage are checked.
            if spec.backbone != "pi05-openpi" or spec.stage != "finetune":
                raise ValueError(f"{spec.id}: signal {s!r} names a pi0.5 finetune's quality-tag "
                                 f"presentation; this run is {spec.backbone!r}, stage {spec.stage!r}")
            continue
        if s not in files:
            raise ValueError(f"{spec.id}: signal {s!r} is not in SIGNALS.lock")
        out.append(files[s])
    return out


def project_openpi(spec, recipe, signals_lock: dict) -> dict[str, object]:
    from axis_qvla.train.ids import content_id
    from axis_qvla.train.recipes import resolved_fields

    out = dict(resolved_fields(recipe, spec))
    out["seed"] = spec.seed
    out["weight_loader.params_path"] = spec.init
    out.update(DROPPED_INERT)
    rowspace, _ = content_id(spec.rowspace)
    out["data.roots_index"] = {"rowspace": rowspace}
    out["data.ranges_path"] = {"rowspace": rowspace}
    for field in _OPENPI_SIGNAL_FIELD.values():
        out[field] = None
    seen_fields: dict[str, str] = {}
    for row in _signal_rows(spec, signals_lock):
        if row["kind"] not in _OPENPI_SIGNAL_FIELD:
            raise ValueError(f"{spec.id}: {row['id']} ({row['kind']}) is not a pi0.5 signal")
        field = _OPENPI_SIGNAL_FIELD[row["kind"]]
        if field in seen_fields:
            raise ValueError(f"{spec.id}: paper.signals names two signals for {field} "
                             f"({seen_fields[field]!r} and {row['id']!r}); the second would silently "
                             f"overwrite the first")
        seen_fields[field] = row["id"]
        out[field] = {"signal": row["id"]}
    out["data.awr_required"] = out["data.awr_weights"] is not None
    out["data.schedule_required"] = out["data.schedule_path"] is not None
    out["data.quality_required"] = out["data.quality_path"] is not None
    out["data.expected_mode"] = "drop_top" if out["data.schedule_path"] is not None else None
    # The finetune presets declare data.quality_tag (a run overrides it); the pretrain presets do not.
    out.setdefault("data.quality_tag", None)
    if spec.stage == "finetune":
        tags = [s for s in spec.paper.get("signals", ()) if s in FT_SIGNALS]
        if len(tags) != 1:
            raise ValueError(f"{spec.id}: a pi0.5 finetune run's paper.signals must name exactly one "
                             f"of {FT_SIGNALS}, got {tags}")
        wants_tag5 = tags[0] == "ft:tag5"
        if (out["data.quality_tag"] == 5) != wants_tag5:
            raise ValueError(f"{spec.id}: paper.signals says {tags[0]!r} but data.quality_tag is "
                             f"{out['data.quality_tag']!r}")
    return out


def parity_paths():
    """A synthetic `Paths` with no real machine behind it, so `lerobot_config` has somewhere to
    put a dataset-root string while comparing; `/parity/...` is not a real root (never resolved,
    never written to) and is not one of the no-absolute-paths ratchet's tracked machine prefixes.
    Public, not underscore-prefixed, so another module that renders through the same placeholder
    needs no private cross-module import to reach it."""
    from axis_qvla.core.hosts import HostProfile
    from axis_qvla.core.paths import Paths

    return Paths(HostProfile(name="parity", launcher="local", venvs={},
                             roots={k: pathlib.Path("/parity") / k for k in ("corpus", "cache", "scratch", "runs")}))


def project_lerobot(spec, recipe, signals_lock: dict) -> dict[str, object]:
    from axis_qvla.train.lerobot_cfg import lerobot_config

    signal_files: dict[str, object] = {}
    for row in _signal_rows(spec, signals_lock):
        field = row["consumers"]["lerobot"]
        if field in signal_files:
            raise ValueError(f"{spec.id}: paper.signals names two signals for {field}; the second "
                             f"would silently overwrite the first")
        signal_files[field] = {"signal": row["id"]}
    cfg = lerobot_config(spec, recipe, parity_paths(), signal_files=signal_files, init_value={"init": spec.init})
    flat = {k: v for k, v in flatten(cfg).items() if not not_compared(k)}
    for key, tag in [(k, "signal") for k in signal_files] + [("policy.pretrained_path", "init")]:
        if f"{key}.{tag}" in flat:
            flat[key] = {tag: flat.pop(f"{key}.{tag}")}
    return flat


def compare(projected: dict, row: dict, differences: dict) -> list[str]:
    """Field-by-field: equal, or declared under [paper.differences] with both values and a reason.

    A declared difference is itself refused, rather than trusted, when it is malformed: missing
    `arm`/`lock`, an empty or missing `reason`, or a value that is not valid JSON -- reported, not
    raised, so one bad entry does not crash the whole check. A `[paper.differences]` entry for a
    field neither the projection nor the lock carries is reported too (it declares nothing real)."""
    lock = row["fields"]
    problems = []
    touched = set()
    for key in sorted(set(projected) | set(lock)):
        a, b = normalized(projected.get(key, _ABSENT)), normalized(lock.get(key, _ABSENT))
        d = differences.get(key)
        if d is not None:
            touched.add(key)
        if a == b:
            if d is not None:
                problems.append(f"{key}: declared different, but the arm and the lock agree ({a!r})")
            continue
        if d is None:
            problems.append(f"{key}: arm {a!r} != lock {b!r}")
            continue
        if not isinstance(d, dict) or "arm" not in d or "lock" not in d:
            problems.append(f"{key}: declared difference is malformed (needs arm, lock and reason)")
            continue
        if not (isinstance(d.get("reason"), str) and d["reason"].strip()):
            problems.append(f"{key}: declared difference has no reason")
            continue
        try:
            arm_val, lock_val = normalized(json.loads(d["arm"])), normalized(json.loads(d["lock"]))
        except (ValueError, TypeError) as exc:
            problems.append(f"{key}: declared difference's arm/lock is not valid JSON "
                            f"({exc.__class__.__name__})")
            continue
        if arm_val != a or lock_val != b:
            problems.append(f"{key}: declared difference does not match (arm {a!r}, lock {b!r})")
    problems += [f"{key}: declared a [paper.differences] entry for a field neither the arm nor the "
                f"lock has" for key in sorted(set(differences) - touched)]
    return problems


def _pretrain_norm_anchor(lock: dict) -> str | None:
    """`<name>#sha256:<digest>` of the single norm digest every OTHER pi0.5 pretrain row in the
    lock records -- the only value a pretrain with no registry digest of its own may cite as its
    declared difference. None if the lock does not have exactly one such digest to anchor to
    (never a fabricated fallback: the digest comes from the lock itself, not from this file)."""
    digests = {r["norm"] for r in lock["pi05"].values() if ".pretrain." in r["arm"] and r["norm"]}
    if len(digests) != 1:
        return None
    return f"{NORM_NAME['pretrain']}#sha256:{next(iter(digests))}"


def _check_norm(spec, row: dict, norm_diff: object, lock: dict) -> list[str]:
    """[norm].artifact: its name, and its digest against the registry's, or -- the two pi0.5
    pretrains the registry records no digest for -- against the single digest every other pi0.5
    pretrain in the lock shares, declared exactly (both the `arm` and the `lock` half) under
    [paper.differences.norm] with a reason. A run whose digest IS recorded may not also declare a
    norm difference: that would excuse nothing real."""
    from axis_qvla.train.ids import content_id

    if spec.norm is None:
        return [f"{spec.id}: no [norm].artifact"]
    problems = []
    norm_name, norm_digest = content_id(spec.norm)
    want_name = NORM_NAME.get(spec.stage)
    if want_name and norm_name != want_name:
        problems.append(f"norm name {norm_name} != expected {want_name}")
    if row["norm"]:
        if norm_digest != row["norm"]:
            problems.append(f"norm {spec.norm} does not carry the checkpoint's norm digest {row['norm']}")
        if norm_diff is not None:
            problems.append(f"{spec.id}: [paper.differences.norm] is declared, but the registry "
                            f"records a norm digest for this checkpoint; nothing excuses it")
        return problems
    if not (isinstance(norm_diff, dict) and isinstance(norm_diff.get("reason"), str) and norm_diff["reason"].strip()):
        problems.append(f"{spec.id}: no norm digest recorded for this run (the registry carries "
                        f"none for this checkpoint); accept only via [paper.differences.norm] "
                        f"with a reason, never by inventing one")
        return problems
    anchor = _pretrain_norm_anchor(lock)
    if anchor is None:
        problems.append(f"{spec.id}: cannot anchor the declared norm difference: the lock has no "
                        f"single digest every other pi0.5 pretrain shares")
        return problems
    declared_arm = _json_or_none(norm_diff.get("arm"))
    # The lock half must be the literal string "null" -- not merely a value _json_or_none also
    # maps to None, which it does for a missing or malformed "lock" key too.
    if spec.norm != anchor or declared_arm != anchor or norm_diff.get("lock") != "null":
        problems.append(f"{spec.id}: [paper.differences.norm] must declare arm={anchor!r}, "
                        f"lock=null (every other pi0.5 pretrain's digest); got "
                        f"arm={norm_diff.get('arm')!r}, lock={norm_diff.get('lock')!r}, and "
                        f"[norm].artifact is {spec.norm!r}")
    return problems


def _check_pi05(spec, row: dict, recipe, signals_lock: dict, differences: dict, lock: dict) -> tuple[dict, list[str]]:
    """pi0.5-only checks: may raise ValueError (an unresolvable projection), caught by check_arm."""
    projected = project_openpi(spec, recipe, signals_lock)   # may raise
    problems = []
    rs_name = _content_name(spec.rowspace)
    if rs_name != row["rowspace"]:
        problems.append(f"rowspace {rs_name} != lock {row['rowspace']}")
    problems += _check_norm(spec, row, differences.pop("norm", None), lock)
    asset_id = spec.data.get("asset_id", row["fields"]["data.repo_id"])
    if asset_id != row["fields"]["data.repo_id"]:
        problems.append(f"data.asset_id {asset_id!r} != lock data.repo_id {row['fields']['data.repo_id']!r}")
    image = spec.data.get("image")
    want_image = _IMAGE_FOR_CENTER_CROP.get(row["fields"]["data.image_center_crop"])
    if image is not None and want_image is not None and image != want_image:
        problems.append(f"data.image {image!r} does not agree with data.image_center_crop="
                        f"{row['fields']['data.image_center_crop']!r} (expected {want_image!r})")
    action_space = spec.data.get("action_space")
    want_action_space = _ACTION_SPACE_FOR.get((row["fields"]["data.eef_action"], row["fields"]["data.joint9_action"]))
    if action_space is not None and want_action_space is not None and action_space != want_action_space:
        problems.append(f"data.action_space {action_space!r} does not agree with "
                        f"data.eef_action={row['fields']['data.eef_action']!r}/"
                        f"data.joint9_action={row['fields']['data.joint9_action']!r} "
                        f"(expected {want_action_space!r})")
    if spec.paper_ref != row["config"]:
        problems.append(f"paper_ref {spec.paper_ref!r} != lock config {row['config']!r}")
    if spec.paper.get("registered_config") != row["config"]:
        problems.append(f"paper.registered_config {spec.paper.get('registered_config')!r} != "
                        f"lock config {row['config']!r}")
    if spec.paper.get("train_config") != row["train_config"]:
        problems.append(f"paper.train_config {spec.paper.get('train_config')!r} != "
                        f"lock train_config {row['train_config']!r}")
    if list(spec.paper.get("launch_overrides", ())) != list(row["launch_overrides"]):
        problems.append(f"paper.launch_overrides {list(spec.paper.get('launch_overrides', ()))!r} "
                        f"!= lock {list(row['launch_overrides'])!r}")
    return projected, problems


def _check_lerobot(spec, row: dict, recipe, signals_lock: dict) -> tuple[dict, list[str]]:
    """SmolVLA / X-VLA-only checks: may raise ValueError (an unresolvable projection), caught by
    check_arm."""
    projected = project_lerobot(spec, recipe, signals_lock)   # may raise
    problems = []
    implied = lerobot_rowspace(spec.stage, spec.paper.get("signals", ()))
    if _content_name(spec.rowspace) != implied:
        problems.append(f"rowspace {_content_name(spec.rowspace)} != what paper.signals implies ({implied})")
    if spec.paper_ref is not None:
        # paper_ref names a pi0.5 openpi eval config (cells.json's --config); a LeRobot run has
        # none -- arms_for_paper_config's own pi0.5-only filter depends on this staying None.
        problems.append(f"{spec.id}: paper_ref {spec.paper_ref!r} must be None for a LeRobot run")
    return projected, problems


def _content_name(cid: str) -> str:
    from axis_qvla.train.ids import content_id

    return content_id(cid)[0]


_PI05_SIGNAL_KEYS = ("data.awr_weights", "data.schedule_path", "data.quality_path")


def _row_signal_ids(row: dict, backbone: str) -> set[str]:
    """Every signal id the LOCK ROW's own fields name -- the ground truth the exports check
    anchors to, not `spec.paper.signals` (an ArmSpec-level copy a test, or a bug, can mutate
    independently of what the row actually recorded)."""
    if backbone == "pi05-openpi":
        return {row["fields"][k]["signal"] for k in _PI05_SIGNAL_KEYS if row["fields"].get(k)}
    from axis_qvla.train.lerobot_cfg import SIGNAL_FIELDS

    return {row["fields"][k]["signal"] for k in SIGNAL_FIELDS if isinstance(row["fields"].get(k), dict)}


def _check_manifest(spec, row: dict, manifest: dict) -> list[str]:
    """What a staged bundle manifest (schema qvla.signal_bundles/1) says about this run,
    cross-checked against the run itself: the bundle IDENTITY -- checking only its rowspace or
    format would accept a swap between two runs that legitimately share both, a seed twin or a
    pi0.5/LeRobot twin -- that the manifest marks it a paper bundle, its declared row space and
    export format agree with the run, that every signal the lock row's own fields name is actually
    exported, and that the manifest's row-space digest agrees with the run's `[data].rowspace`.
    The lock carries none of this itself (it was built from training-time configs, never a bundle
    id), so this check only runs when a manifest is given."""
    from axis_qvla.train.ids import content_id
    from axis_qvla.train.paper import signals_key

    problems = []
    rs_name, rs_digest = content_id(spec.rowspace)
    rs_table = manifest.get("rowspaces", {})
    if rs_name not in rs_table:
        problems.append(f"{spec.id}: rowspace {rs_name!r} is not in the bundle manifest's rowspaces")
    elif rs_table[rs_name] != rs_digest:
        problems.append(f"{spec.id}: rowspace {rs_name} digest {rs_digest} != manifest {rs_table[rs_name]!r}")

    run = signals_key(spec.id)
    bundle_name = manifest.get("arms", {}).get(run)
    is_pi05_ft = spec.backbone == "pi05-openpi" and spec.stage == "finetune"
    if is_pi05_ft:
        if bundle_name is not None:
            problems.append(f"{spec.id}: a pi0.5 finetune has no bundle, but the manifest names "
                            f"{bundle_name!r} for run {run!r}")
        return problems
    if bundle_name is None:
        problems.append(f"{spec.id}: run {run!r} is not in the bundle manifest")
        return problems
    b = manifest.get("bundles", {}).get(bundle_name, {})
    want_bundle = "sha256:" + b.get("id", "")
    if spec.bundle != want_bundle:
        problems.append(f"{spec.id}: bundle {spec.bundle!r} != manifest {want_bundle!r} ({bundle_name})")
    if b.get("paper_arm") is not True:
        problems.append(f"{spec.id}: bundle {bundle_name} paper_arm is {b.get('paper_arm')!r}, not True")
    if b.get("rowspace") != rs_name:
        problems.append(f"{spec.id}: bundle {bundle_name} rowspace {b.get('rowspace')!r} != {rs_name!r}")
    want_format = "openpi-legacy" if spec.backbone == "pi05-openpi" else "lerobot-legacy"
    if b.get("format") != want_format:
        problems.append(f"{spec.id}: bundle {bundle_name} format {b.get('format')!r} != {want_format!r}")
    exports = b.get("exports")
    if not isinstance(exports, dict):
        problems.append(f"{spec.id}: bundle {bundle_name} has no exports mapping")
    else:
        missing = _row_signal_ids(row, spec.backbone) - set(exports)
        if missing:
            problems.append(f"{spec.id}: bundle {bundle_name} exports are missing {sorted(missing)}")
    return problems


def check_arm(spec, lock: dict, signals_lock: dict, manifest: dict | None = None) -> list[str]:
    """Every way `spec` departs from the paper run it names; [] when it reproduces it.

    `manifest` is the signal compiler's staged bundle manifest (schema qvla.signal_bundles/1);
    when given, `_check_manifest` also cross-checks the run's bundle identity against it -- the
    lock alone has no ground truth for a bundle's content. Any ValueError raised while projecting
    the run (an unresolvable signal, a duplicate-field signal, a presentation tag the run's
    backbone or stage cannot carry) is reported as one list entry naming the run, not propagated,
    so the writer's summary shows every run's problems rather than stopping at the first one."""
    from axis_qvla.train.recipes import load_recipe

    checkpoint = spec.paper.get("checkpoint")
    if not checkpoint:
        return [f"{spec.id}: no [paper].checkpoint"]
    table = lock["pi05"] if spec.backbone == "pi05-openpi" else lock["lerobot"]
    row = table.get(spec.id)
    if row is None:
        return [f"{spec.id}: no row in the paper-config lock"]
    problems = [] if row["checkpoint"] == checkpoint else [f"checkpoint {checkpoint} != lock {row['checkpoint']}"]
    # A bundle is what a run actually trained on; without a manifest this is only a necessary
    # invariant (the lock itself carries no bundle id). _check_manifest, when a manifest is given,
    # checks the bundle's actual identity.
    is_pi05_ft = spec.backbone == "pi05-openpi" and spec.stage == "finetune"
    if (spec.bundle is None) != is_pi05_ft:
        problems.append(f"bundle {spec.bundle!r}: must be None for a pi0.5 finetune (its plan is "
                        f"compiled at bind time) and non-None for every other run")
    try:
        manifest_problems = _check_manifest(spec, row, manifest) if manifest is not None else []
        differences = dict(spec.paper.get("differences", {}))
        recipe = load_recipe(spec.preset)
        if spec.backbone == "pi05-openpi":
            projected, more = _check_pi05(spec, row, recipe, signals_lock, differences, lock)
        else:
            projected, more = _check_lerobot(spec, row, recipe, signals_lock)
    except ValueError as exc:
        # Every raise site in _signal_rows/project_openpi/project_lerobot already prefixes its
        # own message with the run id; a raise from content_id (a malformed rowspace/norm
        # reference) does not, so this adds it only where it is not already there.
        msg = str(exc)
        if not msg.startswith(f"{spec.id}:"):
            msg = f"{spec.id}: {msg}"
        return problems + [msg]
    return problems + manifest_problems + more + compare(projected, row, differences)
