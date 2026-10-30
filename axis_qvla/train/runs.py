"""A run directory under <runs>: the spec it recorded, whether it trained its whole budget, and
where its checkpoint is; and the base-model files a run needs on this machine.

Why: three readers need the same facts. Binding a run that starts from another (`init = run:<id>`),
`qvla train finetune --from <id>`, and serving a run (axis_qvla.train.serve) all ask whether the run
finished and where its checkpoint is. One place answers, by the files each backbone writes:
pi0.5 `train/train_summary.json` and `train/final/{params,assets}`, LeRobot
`train/checkpoints/last/` with its `pretrained_model/model.safetensors`. Each refusal says what to
run next (`unfinished`), so its readers never word the fix themselves.

A run starting from a base model reads it from the base-weights copy, offline. `qvla train` fetches
a base that copy lacks before it binds (`fetch_base`, what `qvla data get base-weights --include
<name>` does, verified against conf/recipes/bases.lock.json), so a run never starts without it. A
run starting from a published checkpoint (or from a run trained here that did) still loads files of
its backbone's base when it starts: pi0.5 the PaliGemma tokenizer, SmolVLA the SmolVLM2 backbone its
processor comes from, X-VLA its base with the BART tokenizer. Those are the files a reference
server of that backbone reads (axis_qvla.data.bases.SERVING, the one table `qvla data get base-weights
--serving <backbone>` fetches by), and `qvla train` fetches them the same way (`fetch_serving`).

Core tier: stdlib only.
"""
from __future__ import annotations

import dataclasses
import os
import pathlib
import stat

from axis_qvla.core.paths import Paths
from axis_qvla.train import ids
from axis_qvla.train.armspec import ArmSpec, ArmSpecError, load_armspec
from axis_qvla.train.cut_copy import read_json

RUN_SPEC = "run_spec.toml"      # what render writes: the spec that trained, every input by content
PLANNED_SPEC = "spec.toml"      # what `qvla train pretrain|finetune` writes before the run trains
SHORT_NAMES = {"pi05-openpi": "pi05", "smolvla-lerobot": "smolvla", "xvla-lerobot": "xvla"}   # backbone -> short name


def all_ints(doc: object, *keys: str) -> bool:
    """Every key of `doc` is an integer (a missing one is not: None == None proves nothing)."""
    return isinstance(doc, dict) and all(isinstance(doc.get(k), int) and not isinstance(doc.get(k), bool)
                                         for k in keys)


def _stat_problem(path: pathlib.Path, want_dir: bool) -> str | None:
    try:
        st = os.stat(path)
    except (FileNotFoundError, NotADirectoryError):
        return "is not there"
    except OSError as exc:
        return f"cannot be read ({exc.__class__.__name__})"
    if want_dir:
        return None if stat.S_ISDIR(st.st_mode) else "is not a directory"
    return None if stat.S_ISREG(st.st_mode) else "is not a file"


def not_a_dir(path: pathlib.Path) -> str | None:
    """None when `path` is a directory, else why not. A stat failure other than absence (EACCES)
    is reported as such, never read as "not there"."""
    return _stat_problem(path, True)


def not_a_file(path: pathlib.Path) -> str | None:
    """None when `path` is a regular file, else why not (EACCES named, as `not_a_dir`)."""
    return _stat_problem(path, False)


def load_run_spec(run_dir: pathlib.Path) -> ArmSpec:
    """The spec run `run_dir` trained (its run_spec.toml); ArmSpecError when it has none."""
    return load_armspec(pathlib.Path(run_dir) / RUN_SPEC)


def load_planned_or_run_spec(run_dir: pathlib.Path) -> ArmSpec:
    """The spec the run trained (run_spec.toml), else the one `qvla train pretrain|finetune` planned
    (spec.toml): `qvla train finetune --print` plans from a pretraining run that has not run yet."""
    for name in (RUN_SPEC, PLANNED_SPEC):
        path = pathlib.Path(run_dir) / name
        try:
            os.stat(path)
        except (FileNotFoundError, NotADirectoryError):
            continue
        return load_armspec(path)
    raise ArmSpecError(f"{run_dir} holds neither {RUN_SPEC} nor {PLANNED_SPEC}: no run of that id was "
                       f"planned or trained on this machine (`qvla train pretrain` plans one and prints its id)")


def trainable_spec(run_dir: pathlib.Path) -> pathlib.Path:
    """The spec file to name in a `qvla train` line for the run at `run_dir`: the one `qvla train
    pretrain|finetune` planned (inputs by name) when it is there, else the one that trained."""
    planned = pathlib.Path(run_dir) / PLANNED_SPEC
    try:
        os.stat(planned)
        return planned
    except OSError:
        return pathlib.Path(run_dir) / RUN_SPEC


def checkpoint_dir(paths: Paths, run_id: str, backbone: str) -> pathlib.Path:
    """What a finished run's checkpoint is served and fine-tuned from: pi0.5 the final step's
    directory (params/, assets/), LeRobot the last checkpoint's pretrained_model/."""
    train = ids.run_root(paths, run_id) / "train"
    return train / "final" if backbone == "pi05-openpi" else train / "checkpoints" / "last" / "pretrained_model"


@dataclasses.dataclass(frozen=True)
class Unfinished:
    """Why a run is not a finished, non-smoke run, and what to run next.

    `why` leaves out the run id; `verb` says whether it reads after the id ("trained 10 of 20 steps")
    or after a colon (a path). `kind` picks the fix: "steps" (it has not trained, or not recorded,
    its whole budget), "smoke", "files" (it recorded its budget, but a file it must leave is not
    there), or "defect" (a file is there but cannot be read, is not JSON, or records no integer step:
    not a run that is still on its way, so training on cannot fix it)."""

    why: str
    kind: str
    verb: bool = False

    def about(self, subject: str) -> str:
        return f"{subject} {self.why}" if self.verb else f"{subject}: {self.why}"

    @property
    def not_ready(self) -> bool:
        """The run is on its way (train or continue it); False for a defect in what is there."""
        return self.kind != "defect"

    def message(self, head: str, run_dir: pathlib.Path, backbone: str) -> str:
        """`<head>: <why>; <fix>`, e.g. `<id> cannot be served: it trained 10 of 20605 steps; ...`."""
        return f"{head}: {'it ' if self.verb else ''}{self.why}; {self.fix(run_dir, backbone)}"

    def fix(self, run_dir: pathlib.Path, backbone: str) -> str:
        spec, train = trainable_spec(run_dir), pathlib.Path(run_dir) / "train"
        again = f"remove {train} yourself (qvla train never deletes a directory) and train it again with `qvla train {spec}`"
        if self.kind == "defect":
            return f"qvla train and the backbone write that file readable and well-formed: repair it, or {again}"
        if self.kind == "smoke":
            return f"train it without --smoke: `qvla train {spec}`"
        if self.kind == "files":
            return again
        if backbone == "pi05-openpi":
            return f"when it is not still training, continue it with `qvla train {spec} --resume`"
        return f"when it is not still training, {again} (a LeRobot run is not resumed)"


def _kind(why: str, absent: str) -> str:
    """`absent` when a file is only not there yet; "defect" when it is there but unreadable, not
    JSON or of the wrong type."""
    return absent if why == "is not there" else "defect"


def unfinished(run_id: str, backbone: str, paths: Paths) -> Unfinished | None:
    """Why run `run_id` is not a finished, non-smoke run of `backbone`; None when it is."""
    train = ids.run_root(paths, run_id) / "train"
    if backbone == "pi05-openpi":
        summary = train / "train_summary.json"
        s, why = read_json(summary)
        if why is not None:
            return Unfinished(f"{summary} {why}", _kind(why, "steps"))
        if not all_ints(s, "trained_steps", "declared_steps"):
            return Unfinished(f"{summary} records no integer trained_steps and declared_steps", "defect")
        if s["trained_steps"] != s["declared_steps"]:
            return Unfinished(f"trained {s['trained_steps']} of {s['declared_steps']} steps", "steps", verb=True)
        if s.get("smoke"):
            return Unfinished(f"is a smoke run ({summary})", "smoke", verb=True)
        final = pathlib.Path(ids.openpi_init_params(paths, f"run:{run_id}")).parent
        for sub in ("params", "assets"):        # assets/: the norm stats a fine-tune or server reads
            why = not_a_dir(final / sub)
            if why is not None:
                return Unfinished(f"{final / sub} {why}", _kind(why, "files"))
        return None
    last = train / "checkpoints" / "last"
    out = {}
    for name, rel, key in (("step", "training_state/training_step.json", "step"),
                           ("config", "pretrained_model/train_config.json", "steps")):
        doc, why = read_json(last / rel)
        if why is not None:
            return Unfinished(f"{last / rel} {why}", _kind(why, "steps"))
        if not all_ints(doc, key):
            return Unfinished(f"{last / rel} records no integer {key}", "defect")
        out[name] = doc[key]
    if out["step"] != out["config"]:
        return Unfinished(f"trained {out['step']} of {out['config']} steps", "steps", verb=True)
    model = last / "pretrained_model" / "model.safetensors"
    why = not_a_file(model)
    return None if why is None else Unfinished(f"{model} {why}", _kind(why, "files"))


def finished_problems(init: str, backbone: str, paths: Paths) -> list[str]:
    """Why run `init` (`run:<id>`) is not a finished, non-smoke run of `backbone`; [] when it is."""
    u = unfinished(init[len("run:"):], backbone, paths)
    return [] if u is None else [u.about(f"init: {init}")]


def _run_init(spec: ArmSpec, paths: Paths) -> str:
    """The init of `spec`, or for a run starting from a run trained here, that run's (whose policy
    loads the same backbone's files offline); "" for none, or a `run:` init whose run is not there
    (binding refuses that, naming it)."""
    init = spec.init or ""
    if init.startswith("run:"):
        try:
            init = load_run_spec(ids.run_root(paths, init[len("run:"):])).init or ""
        except ArmSpecError:
            return ""
    return init


def init_base(spec: ArmSpec, paths: Paths) -> str | None:
    """The base model `spec` needs in the base-weights copy: its own `base:<name>`, or, for a run that
    starts from a run trained here, that run's. None for a published checkpoint, or a `run:` init
    whose run is not there."""
    init = _run_init(spec, paths)
    return init[len("base:"):] if init.startswith("base:") else None


def init_serving(spec: ArmSpec, paths: Paths) -> str | None:
    """The backbone ("pi05" | "smolvla" | "xvla") whose base files `spec` loads beside a published
    checkpoint it starts from (directly, or through a run trained here); None otherwise."""
    try:
        ids.published(_run_init(spec, paths))
    except ids.IdError:
        return None
    return SHORT_NAMES[spec.backbone]


def _missing(lock: dict, paths: Paths) -> list[str]:
    """The files of every base in `lock` this machine's copy does not hold. Presence only: `qvla data
    get base-weights` verified each file's sha256 when it wrote it, and it renames a file into place
    only once verified. A file that cannot be stat'ed for another reason (EACCES) counts as missing,
    so the fetch names it rather than the run failing on it later."""
    root, out = ids.base_weights_root(paths), []
    for blocks in lock["bases"].values():
        for block in blocks:
            for row in block["files"]:
                path = root / block["dest"] / row["name"]
                try:
                    os.stat(path)
                except OSError:
                    out.append(str(path))
    return out


def missing_base_files(name: str, paths: Paths) -> list[str]:
    """The files of base `name` (conf/recipes/bases.lock.json) this machine's copy does not hold."""
    from axis_qvla.data import bases

    lock = bases.load()
    return _missing({**lock, "bases": {name: lock["bases"].get(name, ())}}, paths)


def missing_serving_files(backbone: str, paths: Paths) -> list[str]:
    """The base files `backbone`'s policy loads beside a published checkpoint that this machine's
    copy does not hold."""
    from axis_qvla.data import bases

    return _missing(bases.serving_lock(bases.load(), backbone), paths)


def _fetch(lock: dict, names: list[str], paths: Paths) -> None:
    from axis_qvla.data import bases
    from axis_qvla.data.hf import endpoint_from_env

    bases.fetch(lock, names, ids.base_weights_root(paths), hf_endpoint=endpoint_from_env(),
                gcs_endpoint=bases.GCS_ENDPOINT)


def fetch_base(name: str, paths: Paths) -> None:
    """Fetch base `name` into this machine's copy, verified against the lock: what
    `qvla data get base-weights --include <name>` does (network)."""
    from axis_qvla.data import bases

    _fetch(bases.load(), [name], paths)


def fetch_serving(backbone: str, paths: Paths) -> None:
    """Fetch the base files `backbone`'s policy loads beside a published checkpoint, verified against
    the lock: what `qvla data get base-weights --serving <backbone>` does (network)."""
    from axis_qvla.data import bases

    lock = bases.serving_lock(bases.load(), backbone)
    _fetch(lock, list(lock["bases"]), paths)
