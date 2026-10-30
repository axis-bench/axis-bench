"""An FR3 finetune set, held to its embodiment before anything trusts it, and the geometry its stored frames
were made with, measured from their pixels.

WHY: a directory, a repo id and a marker file are requests; only what the resolved set's metadata and
pixels state is a fact. Taken on trust, a set with 7-wide actions would give 7-wide norm statistics, a
directory holding an empty meta/info.json would pass as a finetune set, and a geometry read from a file
one tool writes would make a centre-cropped set read as letterboxed.

- open_set: the set as a LeRobot v2.1 dataset whose fps, `state` / `actions` widths, camera columns
  (frames stored in the parquet) and frame shape are the embodiment's; otherwise DatasetError naming
  each difference.
- check_training_layout: a training layout (`realdata training-set`: a roots index over LeRobot v3
  tasks) whose every task has the embodiment's fps, `observation.state` / `action` widths, cameras and
  frame shape; otherwise DatasetError naming each difference. Its frames keep the recorded geometry,
  which the embodiment declares (`geometry_default`), so nothing is measured;
- measure_geometry: a 16:9 recording letterboxed to a square keeps black bars above and below its
  content (49 rows each of 224 for 1280 x 720); a centre crop has content in every row. On sampled
  frames of each camera (the first, middle and last frame of up to five episodes spread over the set)
  it reads the two bands of rows a letterbox blackens, less a margin at the content edge, and the
  content rows between them. A frame is letterboxed when the brightest 1% of both bands' pixels are at
  most DARK and the brightest 1% of its content rows at least LIT; it is centre-cropped when the
  brightest 1% of both bands are at least LIT and neither band is flat (its 1st-to-99th percentile
  spread above FLAT: a scene has texture, a pad of any level has none). Anything else -- an all-black
  frame, a letterbox padded with grey -- is neither. Every sampled frame of every camera must agree;
  otherwise DatasetError says what was measured. There is no default.

Offline tier (numpy, pyarrow; Pillow to decode the stored PNG frames).
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from axis_qvla.realrobot.fr3.embodiment import Embodiment, load_embodiment
from axis_qvla.realrobot.realdata.lerobot_v21 import IMAGE, VIDEO, Dataset, decode_png

__all__ = ["DatasetError", "open_set", "check_training_layout", "measure_geometry", "training_geometry"]

MARGIN = 4          # rows left out at the content edge, where the resize blends bar and content
DARK = 16           # of 255: the brightest 1% of a letterbox bar's pixels
LIT = 48            # of 255: the brightest 1% of a band that shows the scene
FLAT = 8            # of 255: a band whose 1st-to-99th percentile spread is at most this is a pad (on the
                    # FR3 finetune set, across every episode, the scene bands spread 19 or more --
                    # the wrist camera's minimum; the exterior camera's is 59)
EPISODES = 5        # episodes sampled, spread over the set; three frames of each


class DatasetError(ValueError):
    """A dataset that is not a training set of the embodiment, or whose frame geometry cannot be measured."""


def _number(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _conformance(ds: Dataset, emb: Embodiment) -> list[str]:
    who = f"the {emb.id} embodiment's"
    problems = []
    fps = ds.info.get("fps")
    if not (_number(fps) and fps == emb.fps):
        problems.append(f"fps {fps!r}, {who} is {emb.fps}")
    feats = ds.features if isinstance(ds.features, Mapping) else {}
    for col, width in (("state", len(emb.state_stored)), ("actions", len(emb.action_stored))):
        f = feats.get(col)
        shape = f.get("shape") if isinstance(f, Mapping) else None
        if shape != [width]:
            problems.append(f"no {col} column" if f is None else f"{col} has shape {shape}, {who} is [{width}]")
    cams = {k: f.get("dtype") for k, f in feats.items() if isinstance(f, Mapping) and f.get("dtype") in (IMAGE, VIDEO)}
    want = {k: IMAGE for k in emb.image_keys}
    if cams != want:
        problems.append(f"camera columns {cams}, {who} are {want} (frames stored in the parquet)")
        return problems
    w, h = emb.image_stored
    problems += [f"{k} has shape {feats[k].get('shape')}, {who} is {[h, w, 3]}" for k in emb.image_keys
                 if feats[k].get("shape") != [h, w, 3]]
    return problems


def open_set(dataset_dir: Path, emb: Embodiment) -> Dataset:
    """The set at `dataset_dir`, opened and held to `emb`; DatasetError, naming the directory, otherwise."""
    try:
        ds = Dataset.open(dataset_dir)
        problems = _conformance(ds, emb)
    except (OSError, KeyError, TypeError, AttributeError, ValueError) as exc:
        raise DatasetError(f"{dataset_dir}: cannot read the {emb.id} training set "
                           f"({exc.__class__.__name__}: {exc})") from None
    if problems:
        raise DatasetError(f"{dataset_dir}: not an {emb.id} training set: " + "; ".join(problems))
    return ds


def check_training_layout(dataset_dir: Path, emb: Embodiment) -> None:
    """Hold a training layout (a roots index over LeRobot v3 tasks: `realdata training-set`) to `emb`:
    every task's fps, `observation.state` / `action` widths, cameras (video) and frame shape. DatasetError,
    naming the directory, otherwise."""
    import json

    dataset_dir = Path(dataset_dir)
    try:
        roots = json.loads((dataset_dir / "roots.json").read_text(encoding="utf-8"))
        base = dataset_dir / roots.get("base", ".")
        infos = {rel: json.loads((base / rel / "meta" / "info.json").read_text(encoding="utf-8"))
                 for rel in roots["roots"].values()}
    except (OSError, KeyError, TypeError, AttributeError, ValueError) as exc:
        raise DatasetError(f"{dataset_dir}: cannot read the {emb.id} training set "
                           f"({exc.__class__.__name__}: {exc})") from None
    who, (w, h) = f"the {emb.id} embodiment's", emb.image_stored
    problems = []
    for rel, info in sorted(infos.items()):
        feats = info.get("features") if isinstance(info.get("features"), Mapping) else {}
        if info.get("fps") != emb.fps:
            problems.append(f"{rel}: fps {info.get('fps')!r}, {who} is {emb.fps}")
        for col, width in (("observation.state", len(emb.state_model)), ("action", len(emb.action_stored))):
            shape = feats.get(col, {}).get("shape") if isinstance(feats.get(col), Mapping) else None
            if shape != [width]:
                problems.append(f"{rel}: {col} has shape {shape}, {who} is [{width}]")
        for k in emb.image_keys:
            f = feats.get(k) if isinstance(feats.get(k), Mapping) else {}
            if f.get("dtype") != VIDEO or f.get("shape") != [h, w, 3]:
                problems.append(f"{rel}: {k} has shape {f.get('shape')}, {who} is {[h, w, 3]}"
                                if f.get("dtype") == VIDEO else f"{rel}: no {k} video, {who} camera")
    if problems:
        raise DatasetError(f"{dataset_dir}: not an {emb.id} training set: " + "; ".join(problems))


def _bands(emb: Embodiment) -> int:
    """Rows per band: the letterbox bar of a recorded frame at the stored size, less MARGIN."""
    (rw, rh), (_, h) = emb.image_recorded, emb.image_stored
    rows = round((h - h * min(rw, rh) / max(rw, rh)) / 2) - MARGIN
    if rw <= rh or rows < 1:
        raise DatasetError(f"{emb.id}: a {rw} x {rh} recording letterboxed to {h} rows leaves no bar to measure")
    return rows


def _frames(ds: Dataset, col: str):
    n = len(ds.episodes)
    for i in sorted({int(round(x)) for x in np.linspace(0, n - 1, min(n, EPISODES))}):
        frames = ds.read_episode(int(ds.episodes[i]["episode_index"]), [col]).column(col)
        for r in sorted({0, len(frames) // 2, len(frames) - 1} if len(frames) else set()):
            yield decode_png(frames[r].as_py()["bytes"])


def _camera(ds: Dataset, col: str, emb: Embodiment, rows: int) -> tuple[str | None, str]:
    """(the camera's geometry, or None) and a sentence of what was measured."""
    w, h = emb.image_stored
    edge = rows + 2 * MARGIN                      # the content rows start a margin past the bar
    kinds, peaks, spreads, inner = [], [], [], []
    for f in _frames(ds, col):
        if f.shape != (h, w, 3):
            raise DatasetError(f"{ds.root}: a stored {col} frame is {f.shape}, not {(h, w, 3)}")
        bands = (f[:rows], f[h - rows:])
        top, bottom = (float(np.percentile(b, 99)) for b in bands)
        spread = min(float(np.percentile(b, 99) - np.percentile(b, 1)) for b in bands)
        lit = float(np.percentile(f[edge:h - edge], 99))
        peaks += [top, bottom]
        spreads.append(spread)
        inner.append(lit)
        if max(top, bottom) <= DARK and lit >= LIT:
            kinds.append("letterbox224")
        elif min(top, bottom) >= LIT and spread > FLAT:
            kinds.append("center_crop224")
        else:
            kinds.append(None)
    said = (f"{col}: of {len(kinds)} sampled frames, {kinds.count('letterbox224')} letterboxed, "
            f"{kinds.count('center_crop224')} with the scene in both bands, {kinds.count(None)} neither "
            f"(brightest 1% of the top and bottom {rows} rows: {min(peaks, default=0):g} to {max(peaks, default=0):g}, "
            f"their least spread {min(spreads, default=0):g}, the content rows' brightest 1% from "
            f"{min(inner, default=0):g}; a bar is at most {DARK} beside content of at least {LIT}, the scene at "
            f"least {LIT} with a spread above {FLAT})")
    return (kinds[0] if kinds and len(set(kinds)) == 1 else None), said


def measure_geometry(ds: Dataset, emb: Embodiment) -> str:
    """`letterbox224` or `center_crop224`, as the stored pixels of every camera show it; DatasetError
    saying what was measured when they show neither, or the cameras disagree."""
    rows = _bands(emb)
    try:
        measured = {col: _camera(ds, col, emb, rows) for col in emb.image_keys}
    except (OSError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, DatasetError):
            raise
        raise DatasetError(f"{ds.root}: cannot read its stored frames ({exc.__class__.__name__}: {exc})") from None
    unclear = [said for geo, said in measured.values() if geo is None]
    if unclear:
        raise DatasetError(f"{ds.root}: the stored frames are neither letterboxed nor centre-cropped: "
                           + "; ".join(unclear))
    geos = {geo for geo, _ in measured.values()}
    if len(geos) != 1:
        raise DatasetError(f"{ds.root}: the cameras disagree: "
                           + ", ".join(f"{col} {geo}" for col, (geo, _) in measured.items()))
    return geos.pop()


def training_geometry(dataset_dir: Path, embodiment: str = "fr3") -> str:
    """The transform that made a finetune set's frames from the raw recordings, measured from its pixels."""
    emb = load_embodiment(embodiment)
    return measure_geometry(open_set(dataset_dir, emb), emb)
