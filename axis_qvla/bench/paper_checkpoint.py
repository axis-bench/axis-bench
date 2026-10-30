"""A results-map checkpoint id (experiments/paper/cells.json `checkpoint.id`) as everything `bench
eval --checkpoint-id` needs to serve it, for every backbone.

Why: one id names a paper checkpoint for every backbone. Its first component is the backbone (so the
reference server), the results map gives its ref (so the local copy `qvla data get checkpoints/<id>` made,
axis_qvla.data.layout.checkpoint_dest; a LeRobot checkpoint is served from its pretrained_model/) and,
for LeRobot, the policy config its eval_command records. pi0.5's server config is rendered from the
paper run (axis_qvla.bench.openpi_serve).

Core tier: stdlib only.
"""
from __future__ import annotations

import dataclasses
import shlex
from pathlib import Path

BACKEND_OF_BACKBONE = {"pi05": "openpi", "smolvla": "smolvla", "xvla": "xvla"}
LEROBOT_MODEL_DIR = "pretrained_model"


class PaperCheckpointError(ValueError):
    """The id names no backbone, or is not a checkpoint of the results map."""


class NotInResultsMap(PaperCheckpointError):
    """The results map was read and holds no checkpoint of this id (not: the map cannot be read)."""


@dataclasses.dataclass(frozen=True)
class PaperCheckpoint:
    id: str
    backend: str
    ref: str               # <org>/<repo>@<tag>:<path>
    config: str | None     # the LeRobot policy config the eval_command records; None for pi0.5


def backend_of(checkpoint_id: str) -> str:
    """The reference server of a results-map checkpoint id, from its first component."""
    head = checkpoint_id.partition("/")[0]
    if head not in BACKEND_OF_BACKBONE:
        raise PaperCheckpointError(f"--checkpoint-id {checkpoint_id!r}: a results-map checkpoint id begins with "
                                   f"{', '.join(f'{k}/' for k in BACKEND_OF_BACKBONE)}")
    return BACKEND_OF_BACKBONE[head]


def default_run(checkpoint_id: str, serving: str) -> str:
    """The run a checkpoint id is filed under without --run: the id itself (its `/` as `.`) and the
    serving, so two checkpoints never share one by default -- the checkpoint directory's basename
    would (`pretrained_model` for every LeRobot copy, a step number for pi0.5)."""
    return f"{checkpoint_id.replace('/', '.')}_{serving}"


def _flag(tokens: list[str], flag: str) -> str | None:
    return tokens[tokens.index(flag) + 1] if flag in tokens[:-1] else None


def lookup(checkpoint_id: str, cells_bytes: bytes | None = None) -> PaperCheckpoint:
    """The results map's record of `checkpoint_id` (the committed map unless `cells_bytes`)."""
    from axis_qvla.paper.cells import load_cells
    from axis_qvla.paper.cli import DEFAULT_CELLS

    try:
        cells = load_cells(cells_bytes if cells_bytes is not None else DEFAULT_CELLS.read_bytes()).cells
    except (OSError, ValueError) as exc:
        raise PaperCheckpointError(f"--checkpoint-id {checkpoint_id}: the results map cannot be read ({exc})") from None
    hit = next((c for c in cells if c.checkpoint.get("id") == checkpoint_id), None)
    if hit is None:
        raise NotInResultsMap(f"--checkpoint-id {checkpoint_id} is not a checkpoint of the results map, so it "
                                   f"names no local copy: it needs --checkpoint <directory>")
    backend = backend_of(checkpoint_id)
    config = None if backend == "openpi" else _flag(shlex.split(hit.eval_command), "--config")
    return PaperCheckpoint(checkpoint_id, backend, str(hit.checkpoint["ref"]), config)


def local_dir(p: PaperCheckpoint, paths) -> Path:
    """The directory the server loads: the data-get copy of the checkpoint (pi0.5), or its
    pretrained_model/ (SmolVLA, X-VLA). `paths` is an axis_qvla.core.paths.Paths."""
    from axis_qvla.data.layout import LayoutError, checkpoint_dest

    try:
        d = checkpoint_dest(p.ref, paths)
    except LayoutError as exc:
        raise PaperCheckpointError(f"--checkpoint-id {p.id}: {exc}") from None
    except KeyError as exc:   # a host profile without the root the copy is under
        raise PaperCheckpointError(f"--checkpoint-id {p.id}: the host declares no root {exc}, under which "
                                   f"`qvla data get checkpoints/{p.id}` puts the local copy; add it to [roots], "
                                   f"or pass --checkpoint <directory>") from None
    return d if p.backend == "openpi" else d / LEROBOT_MODEL_DIR
