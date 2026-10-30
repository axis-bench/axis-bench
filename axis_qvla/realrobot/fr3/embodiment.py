"""One embodiment declaration (conf/embodiments/<id>.toml): state and action layout, units, image geometry.

The state is declared twice, as stored and in the order the model reads it; both name the same
quantities, so one is a permutation of the other and a consumer permutes by name, never by position.
`geometry_default` is the transform from a raw camera frame to the model's input (`center_crop224`,
`letterbox224`) when the embodiment fixes it: the rendered view stores raw 640 x 360 frames and its
training crops them. An embodiment whose stored frames already carry their geometry (fr3: 224 x 224,
made at export) declares none, because that is a property of each set's pixels; the FR3 checks measure
it (axis_qvla.realrobot.fr3.dataset). Offline tier (stdlib only).
"""
from __future__ import annotations

import dataclasses
import tomllib
from pathlib import Path

EMBODIMENTS = Path(__file__).resolve().parents[3] / "conf" / "embodiments"
SCHEMA = "qvla.embodiment/1"


class EmbodimentError(ValueError):
    """An embodiment declaration is unreadable, of another schema, or inconsistent with itself."""


@dataclasses.dataclass(frozen=True)
class Embodiment:
    id: str
    fps: int
    state_stored: tuple[str, ...]
    state_model: tuple[str, ...]
    action_stored: tuple[str, ...]
    action_units: str
    action_range: tuple[float, float]
    gripper_units: str
    image_keys: tuple[str, ...]
    image_recorded: tuple[int, int]
    image_stored: tuple[int, int]
    geometry_default: str | None


def load_embodiment(name_or_path: str | Path) -> Embodiment:
    """`fr3` reads conf/embodiments/fr3.toml; a path with a suffix is read as given."""
    path = Path(name_or_path)
    if not path.suffix:
        path = EMBODIMENTS / f"{name_or_path}.toml"
    try:
        d = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise EmbodimentError(f"{path}: cannot read ({exc.__class__.__name__}: {exc})") from None
    if d.get("schema") != SCHEMA:
        raise EmbodimentError(f"{path}: schema {d.get('schema')!r}, expected {SCHEMA!r}")
    try:
        e = Embodiment(d["id"], int(d["fps"]), tuple(d["state"]["stored"]), tuple(d["state"]["model"]),
                       tuple(d["action"]["stored"]), d["action"]["units"], tuple(d["action"]["range"]),
                       d["action"]["gripper_units"], tuple(d["images"]["keys"]), tuple(d["images"]["recorded"]),
                       tuple(d["images"]["stored"]), d["images"].get("geometry_default"))
    except KeyError as exc:
        raise EmbodimentError(f"{path}: missing {exc}") from None
    if len(set(e.state_stored)) != len(e.state_stored) or sorted(e.state_stored) != sorted(e.state_model):
        raise EmbodimentError(f"{path}: state.model {e.state_model} is not a permutation of state.stored {e.state_stored}")
    return e
