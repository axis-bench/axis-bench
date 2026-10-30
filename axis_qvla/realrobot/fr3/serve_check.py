"""The serve check: what a policy server's handshake declares against what its finetune run trained on.

check_serve first holds `--dataset` to its embodiment and finds the geometry its frames were made with
(axis_qvla.realrobot.fr3.dataset): measured from the stored pixels of an fr3 set (224 x 224),
the embodiment's declared geometry for your demonstrations in the training layout (fr3_droid, 320 x 180
frames that training and serving crop); a set whose geometry is not the record's stage image is not the
run's finetune set. It then holds the transform a server applies to raw frames (handshake
`image_geometry`, for every camera the client sends, and names any camera it does not send) to that
measured geometry: center_crop224 on the FR3 finetune set. The server must also serve the model
state order, the FR3 action units at 15 Hz in 8 dims with the run's action horizon, and that run's serve
condition, each compared as typed (axis_qvla.realrobot.fr3.handshake). An unreadable embodiment or dataset is
a named problem.

Offline tier (numpy, pyarrow, through the dataset module).
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from axis_qvla.bench.protocol import IMAGE_KEYS
from axis_qvla.realrobot.fr3.dataset import (DatasetError, check_training_layout, measure_geometry, open_set,
                                        training_geometry)
from axis_qvla.realrobot.fr3.embodiment import EmbodimentError, load_embodiment
from axis_qvla.realrobot.fr3.handshake import serve_fields
from axis_qvla.realrobot.fr3.line import IMAGES, Line, stage_of

__all__ = ["check_serve", "training_geometry"]


def _set_geometry(dataset_dir: Path, emb, stage: dict, run_id: str) -> tuple[str | None, list[str]]:
    """The geometry the run's finetune set was made with (None when it cannot be trusted), and its problems:
    measured from the stored pixels of an fr3 set, the declared one of a training layout (fr3_droid)."""
    try:
        if emb.geometry_default is None:
            have = measure_geometry(open_set(dataset_dir, emb), emb)
        else:
            check_training_layout(dataset_dir, emb)
            have = emb.geometry_default
    except DatasetError as exc:
        return None, [str(exc)]
    want = IMAGES.get(stage["image"])
    if have != want:
        if emb.geometry_default is not None:
            return have, [f"{emb.id}: its frames are served as {have}; {run_id} trains on {want} frames"]
        where = ", ".join(x for x in ("the record's stage 2", stage.get("dataset")) if x)
        return have, [f"{dataset_dir}: its frames measure as {have}; {run_id} trains on {want} frames ({where})"]
    return have, []


def check_serve(handshake: Mapping, line: Line, run_id: str, dataset_dir: Path, embodiment: str | None = None) -> list[str]:
    """`embodiment` is the finetune set's: the record's stage 2 (fr3, the FR3 finetune set) by default, or
    fr3_droid for your demonstrations in the training layout."""
    spec = line.runs.get(run_id)
    if spec is None:
        return [f"{run_id} is not a run of the line record"]
    if spec["stage"] != "finetune":
        return [f"{run_id} is not served: only finetune runs name a serve condition"]
    if not isinstance(handshake, Mapping):
        return [f"the handshake is a {type(handshake).__name__}, not a map"]
    f, problems = serve_fields(handshake)
    if f is None:
        return problems
    try:
        fr3 = load_embodiment(embodiment or line.stage2["embodiment"])
    except EmbodimentError as exc:
        return [str(exc)]
    trained, problems = _set_geometry(dataset_dir, fr3, stage_of(line, spec), run_id)
    problems += [f"image {cam}: the handshake declares no geometry for it" for cam in IMAGE_KEYS
                 if cam not in f.image_geometry]
    problems += [f"image {cam}: a camera the client does not send (it sends {', '.join(IMAGE_KEYS)})"
                 for cam in f.image_geometry if cam not in IMAGE_KEYS]
    if trained is not None:
        problems += [f"image {k}: the server applies geometry {g!r} to raw frames; the training frames were made "
                     f"with {trained!r}" for k, g in f.image_geometry.items() if k in IMAGE_KEYS and g != trained]
    if f.state_order != fr3.state_model:
        problems.append(f"state order {list(f.state_order)}, the model reads {list(fr3.state_model)}")
    if f.action_units != fr3.action_units:
        problems.append(f"action units {f.action_units!r}, the finetune data has {fr3.action_units!r}")
    if f.fps != fr3.fps:
        problems.append(f"action fps {f.fps}, the finetune data has {fr3.fps}")
    if f.dim != len(fr3.action_stored):
        problems.append(f"action dim {f.dim}, the finetune data has {len(fr3.action_stored)}")
    if f.horizon != stage_of(line, spec)["action_horizon"]:
        problems.append(f"action horizon {f.horizon}, {run_id} trained {stage_of(line, spec)['action_horizon']}")
    cond = line.serve.get(spec.get("serve"))
    if cond is None:
        return problems + [f"{run_id}: serve condition {spec.get('serve')!r} is not in the record's [serve]"]
    if f.prompt_suffix != cond["prompt_suffix"]:
        problems.append(f"prompt suffix {f.prompt_suffix!r}, {run_id} serves {cond['prompt_suffix']!r} "
                        f"({spec['serve']})")
    if f.guidance != float(cond["guidance"]):
        problems.append(f"guidance {f.guidance:g}, {run_id} serves {float(cond['guidance']):g} ({spec['serve']})")
    return problems
