"""Run specs of the real-world line that name this machine's inputs: the ones `qvla train` trains.

The shipped run specs (experiments/realrobot/runs/) name their pretraining data, compiled labels and norm
statistics `local:`: they exist only where you build them. Built in the local layout of the real-world
tutorial (<cache>/derived/realworld/: d8/, built/, demos/; compiled labels in <cache>/bundles/; norm
statistics in <cache>/derived/norm/), `write_local_runs` writes one run spec per run of the line record,
in <cache>/derived/realworld/runs/, with each input named by NAME, never by digest: the pretraining
frame index (the `realign build` directory's roots and ranges) and the fine-tuning one (every trainable
frame of the demonstrations' tail-10 ranges, its source naming the digest of the demonstrations' roots,
ranges, metadata and frame tables, so rebuilt demonstrations give another id) are written as frame-index
records into <cache>/bundles/rowspaces/ and recorded under their names (realworld_pretrain,
realworld_finetune), the norm statistics under theirs (norm_realalign_sim, norm_fr3_droid) with the
sha256 of the file checked here, and the compiled labels are named `name:realrobot.<run>` as `qvla
realrobot labels` recorded them (axis_qvla.train.names). `qvla train` resolves each name and records the id
it resolved to in the run's names.json. Before any run spec is written, each stage's norm statistics are checked
against that stage's data (axis_qvla.realrobot.fr3.norm_check: d8/ as realalign_sim, demos/ as fr3_droid),
and a run whose training layout, compiled labels or norm statistics are missing, or whose statistics
describe other data, is refused by name with the command that writes the input.

Core tier at import (stdlib only); the builders are imported inside the function.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path

from axis_qvla.core.canonical import canonical_name
from axis_qvla.realrobot.fr3.line import Line, run_spec
from axis_qvla.realrobot.layout import LOCAL, NORMS, SERVE_CONFIG

LINE = Path(__file__).resolve().parents[3] / "experiments" / "realrobot" / "line.toml"
LABELS_PREFIX = "realalign_"    # realign compile's manifest keys: realalign_<run>; recorded as realrobot.<run>
# The command that writes each input of the local layout, named when the input is missing.
WRITES = {"d8": "qvla realrobot rerender", "built": "qvla realrobot labels",
          "demos": "qvla realrobot demos <your DROID recordings>"}
# Recorded in the fine-tuning frame index, so part of its id: the module keeps the development root.
FINETUNE_SOURCE = canonical_name("realrobot.fr3.local_runs") + ": the demonstrations' tail-10 frames"


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _demos_digest(demos: Path) -> str:
    """sha256 over the demonstrations' roots, ranges and every file of each task's data/ and meta/ (the
    frame tables and their metadata; the videos are left out), by relative path. os.walk raises on an
    unreadable directory."""
    def fail(exc: OSError) -> None:
        raise exc

    files = [demos / "roots.json", demos / "ranges.json"]
    for task in sorted(p for p in demos.iterdir() if p.is_dir()):
        for sub in ("data", "meta"):
            files += sorted(Path(r) / f for r, _, names in os.walk(task / sub, onerror=fail) for f in names)
    h = hashlib.sha256()
    for f in files:
        h.update(f"{f.relative_to(demos).as_posix()}\0{_sha(f)}\n".encode())
    return h.hexdigest()


def _layout(derived: Path, name: str) -> Path:
    path = derived / name
    if not path.is_dir():
        raise ValueError(f"no {path}: run `{WRITES[name].format(path=path)}` first")
    return path


def labels_name(key: str) -> str:
    """The name `qvla realrobot labels` records the compiled labels of manifest key `key` under."""
    return "realrobot." + key.removeprefix(LABELS_PREFIX)


def load_local_run(path: Path, paths):
    """A run spec `write_local_runs` wrote, every name resolved to the content id it names on this
    machine (what `qvla train` binds)."""
    from axis_qvla.train.armspec import load_armspec
    from axis_qvla.train.names import resolve_names

    return resolve_names(load_armspec(path), paths)[0]


def write_local_runs(paths, line: Line, *, runs=None) -> dict[str, Path]:
    """`{run id: written run spec}` for `runs` (default: every run of `line`). ValueError, naming the run
    and the input, when a run's training layout, compiled labels or norm statistics are not built, or its
    norm statistics do not describe its stage's data."""
    from axis_qvla.realrobot.realign.bundles import MANIFEST_FILE, realign_rowspace
    from axis_qvla.signals.bundle import write_rowspace
    from axis_qvla.signals.rowspace import flat_rowspace
    from axis_qvla.train import ids
    from axis_qvla.realrobot.fr3.norm_check import check_norm
    from axis_qvla.train.armspec import dumps_armspec
    from axis_qvla.train.ft_bundle import episode_table
    from axis_qvla.train.generate import openpi_asset_id
    from axis_qvla.train.names import LABEL_PREFIX, record_labels_name, record_local_name
    from axis_qvla.train.recipes import load_recipe, resolved_fields

    unknown = sorted(set(runs or ()) - set(line.runs))
    if unknown:
        raise ValueError(f"{', '.join(unknown)}: not a run of the line record (runs: {', '.join(sorted(line.runs))})")
    derived = ids.local_derived(paths, LOCAL)
    bundles = ids.local_bundles_root(paths)
    specs = {r: run_spec(line, r) for r in sorted(runs or line.runs)}

    compiled: dict[str, str] = {}
    if any(s.bundle is not None for s in specs.values()):
        manifest = bundles / MANIFEST_FILE
        if not manifest.is_file():
            raise ValueError(f"no compiled labels at {manifest}: run `qvla realrobot labels` first")
        compiled = json.loads(manifest.read_text(encoding="utf-8"))["runs"]
    stages = {s.stage for s in specs.values()}
    data = {stage: _layout(derived, NORMS[stage][1]) for stage in sorted(stages)}
    built = _layout(derived, "built") if "pretrain" in stages else None
    local, checked = {}, set()
    for run_id, spec in specs.items():
        name, _dir, embodiment = NORMS[spec.stage]
        asset = openpi_asset_id(spec, resolved_fields(load_recipe(spec.preset), spec))
        norm_file = ids.local_derived(paths, f"norm/{name}") / str(asset) / "norm_stats.json"
        rerun = f"qvla realrobot fr3 norm --dataset {data[spec.stage]} --embodiment {embodiment}"
        if not norm_file.is_file():
            raise ValueError(f"{run_id}: no norm statistics at {norm_file}: run `{rerun}` first")
        if norm_file not in checked:
            stale = check_norm(norm_file, data[spec.stage], embodiment)
            if stale:
                raise ValueError(f"{run_id}: the norm statistics at {norm_file} do not describe "
                                 f"{data[spec.stage]} ({'; '.join(stale[:3])}): re-run `{rerun}`")
            checked.add(norm_file)
        bundle = None
        if spec.bundle is not None:
            key = spec.bundle.removeprefix("local:")
            if key not in compiled:
                raise ValueError(f"{run_id}: `realign compile` wrote no {key} (it wrote {', '.join(sorted(compiled))})")
            record_labels_name(bundles, labels_name(key), compiled[key])
            bundle = f"{LABEL_PREFIX}{labels_name(key)}"
        record_local_name(bundles, "norm", f"{name}#sha256:{_sha(norm_file)}")
        local[run_id] = (spec, bundle, name)

    records = bundles / "rowspaces"
    index: dict[str, str] = {}
    if built is not None:
        index["pretrain"] = f"realworld_pretrain#sha256:{write_rowspace(realign_rowspace(built), records)}"
        record_local_name(bundles, "rowspace", index["pretrain"])
    if "finetune" in stages:
        demos = data["finetune"]
        episodes, _frames, _prompts = episode_table(demos / "roots.json", demos, demos / "ranges.json")
        rows = flat_rowspace(name="realworld_finetune", n_rows=len(episodes),
                             source=f"{FINETUNE_SOURCE} of sha256:{_demos_digest(demos)}")
        index["finetune"] = f"realworld_finetune#sha256:{write_rowspace(rows, records)}"
        record_local_name(bundles, "rowspace", index["finetune"])

    out: dict[str, Path] = {}
    for run_id, (spec, bundle, norm) in local.items():
        path = derived / "runs" / f"{run_id}.toml"
        path.parent.mkdir(parents=True, exist_ok=True)
        rowspace = index[spec.stage].partition("#")[0]          # by name; recorded above
        path.write_text(dumps_armspec(dataclasses.replace(spec, rowspace=rowspace, bundle=bundle, norm=norm)),
                        encoding="utf-8")
        out[run_id] = path
    return out


def write_serve_config(spec_path: Path, out: Path | None, paths) -> tuple[str, Path]:
    """(run id, file): the openpi config a policy server of the run spec `spec_path` starts with. It is
    the run's training config with nothing bound (axis_qvla.train.generate.openpi_config(serve=True)), so the
    image crop, action horizon and action units are the ones it trained with. It is written to `out`,
    default <runs>/<run id>/serve.json. ValueError for a run spec that names an input `local:`."""
    from axis_qvla.core.canon import dumps_canonical
    from axis_qvla.train import ids
    from axis_qvla.train.generate import openpi_config
    from axis_qvla.train.recipes import load_recipe

    spec = load_local_run(spec_path, paths)
    cfg = openpi_config(spec, load_recipe(spec.preset), paths, serve=True)
    out = Path(out) if out is not None else ids.run_root(paths, spec.id) / SERVE_CONFIG
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(dumps_canonical(cfg))
    return spec.id, out
