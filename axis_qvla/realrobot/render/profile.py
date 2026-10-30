"""A workspace profile: the exterior camera, the table and the background a render reproduces.

The renderer builds its arena from one profile and writes it beside the render (`workspace.toml`); the
gates and the overlay check read it back. `conf/workspaces/fr3_reference.toml` is the reference
workspace. Positions are metres in the robot base frame (x forward, y left, z up from the
table top, where the base stands); quaternions are w, x, y, z. The floor is `table.height` below the
base, and the wall stands on it. A surface's image is a path inside the profile's directory, so a
render keeps it beside its copy of the profile at the same relative path.

Render-tier package, stdlib only: the gates and the FR3 checks read it without MuJoCo.
"""
from __future__ import annotations

import dataclasses
import tomllib
from pathlib import Path, PurePosixPath

from axis_qvla.realrobot.render import rig

SCHEMA = "qvla.workspace_profile/1"
WORKSPACES = Path(__file__).resolve().parents[3] / "conf" / "workspaces"
REFERENCE = WORKSPACES / "fr3_reference.toml"
FILE = "workspace.toml"
# What a render writes at the top of its directory besides task_<id>/: a kept image must not land on one.
RENDER_NAMES = (FILE, "roots.json", "render_gates.json")


class ProfileError(ValueError):
    """A profile that is unreadable, of another schema, or missing or mistyping a field."""


@dataclasses.dataclass(frozen=True)
class Surface:
    rgb: tuple[float, float, float]
    rgb_jitter: float                           # one offset added to all three channels, drawn per episode
    emission: float
    material: tuple[float, float, float]        # reflectance, shininess, specular before jitter
    material_caps: tuple[float, float, float]   # their maxima
    material_jitter: float = 0.005
    texture: Path | None = None                 # an image on the surface: the profile's directory / its path there
    texture_repeat: tuple[float, float] = (1.0, 1.0)


@dataclasses.dataclass(frozen=True)
class Fixture:
    name: str
    pos: tuple[float, float, float]             # box centre
    half: tuple[float, float, float]            # half extents
    material: str                               # a key of [materials]


@dataclasses.dataclass(frozen=True)
class Table:
    height: float                               # floor to table top
    center: tuple[float, float]
    size: tuple[float, float, float]            # x and y extent, thickness of the top
    surface: Surface


@dataclasses.dataclass(frozen=True)
class Background:
    wall_distance: float                        # from the table's rear edge to the wall
    wall_half_width: float
    wall_height: float
    wall: Surface
    floor: Surface


@dataclasses.dataclass(frozen=True)
class WorkspaceProfile:
    id: str
    camera: rig.Camera                          # the exterior camera, in the render world
    table: Table
    materials: dict[str, tuple[float, float, float, float]]
    fixtures: tuple[Fixture, ...]
    background: Background
    text: str                                   # the profile file as read; a render keeps a copy
    directory: Path                             # where it was read; its images are paths inside it

    @property
    def cameras(self) -> tuple[rig.Camera, ...]:
        return (self.camera, rig.WRIST)

    def surfaces(self) -> tuple[tuple[str, Surface], ...]:
        """The surfaces whose colour is drawn per episode, in the order it is drawn."""
        return (("table", self.table.surface), ("floor", self.background.floor), ("wall", self.background.wall))

    def images(self) -> tuple[Path, ...]:
        """The image files the surfaces name, relative to the profile's directory."""
        return tuple(sorted({s.texture.relative_to(self.directory) for _, s in self.surfaces() if s.texture}))


def _table(v: object, where: str) -> dict:
    if not isinstance(v, dict):
        raise ProfileError(f"{where} must be a table, got {v!r}")
    return v


def _section(v: object, where: str, required: tuple[str, ...], optional: tuple[str, ...] = ()) -> dict:
    """`v` as a table with every key of `required` and no key outside `required` and `optional`."""
    d = _table(v, where)
    missing = [k for k in required if k not in d]
    if missing:
        raise ProfileError(f"{where} has no {missing[0]!r}")
    unknown = sorted(set(d) - {*required, *optional})
    if unknown:
        raise ProfileError(f"{where} has unknown keys {unknown}; it takes {[*required, *optional]}")
    return d


def _float(v: object, where: str) -> float:
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        raise ProfileError(f"{where} must be a number, got {v!r}")
    return float(v)


def _floats(v: object, n: int, where: str) -> tuple[float, ...]:
    if not (isinstance(v, list) and len(v) == n
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)):
        raise ProfileError(f"{where} must be {n} numbers, got {v!r}")
    return tuple(float(x) for x in v)


def _str(v: object, where: str) -> str:
    if not isinstance(v, str):
        raise ProfileError(f"{where} must be a string, got {v!r}")
    return v


def _texture(v: object, where: str, base: Path) -> Path | None:
    if v is None:
        return None
    rel = PurePosixPath(_str(v, where))
    if not rel.parts or rel.is_absolute() or ".." in rel.parts:
        raise ProfileError(f"{where} must be a path inside the profile's directory, got {v!r}")
    if rel.parts[0] in RENDER_NAMES or rel.parts[0].startswith("task_"):
        raise ProfileError(f"{where} starts with {rel.parts[0]!r}, a name a render writes beside its copy of the "
                           f"profile ({', '.join(RENDER_NAMES)}, task_*); rename the image")
    return base / rel


def _surface(v: object, where: str, base: Path) -> Surface:
    d = _section(v, where, ("rgb", "rgb_jitter", "emission", "material", "material_caps"),
                 ("material_jitter", "texture", "texture_repeat"))
    return Surface(_floats(d["rgb"], 3, f"{where}.rgb"), _float(d["rgb_jitter"], f"{where}.rgb_jitter"),
                   _float(d["emission"], f"{where}.emission"), _floats(d["material"], 3, f"{where}.material"),
                   _floats(d["material_caps"], 3, f"{where}.material_caps"),
                   _float(d.get("material_jitter", 0.005), f"{where}.material_jitter"),
                   _texture(d.get("texture"), f"{where}.texture", base),
                   _floats(d.get("texture_repeat", [1.0, 1.0]), 2, f"{where}.texture_repeat"))


def _fixture(v: object, where: str) -> Fixture:
    f = _section(v, where, ("name", "pos", "half", "material"))
    return Fixture(_str(f["name"], f"{where}.name"), _floats(f["pos"], 3, f"{where}.pos"),
                   _floats(f["half"], 3, f"{where}.half"), _str(f["material"], f"{where}.material"))


def _profile(d: dict, text: str, base: Path) -> WorkspaceProfile:
    _section(d, "the profile", ("schema", "id", "camera", "table", "background"), ("about", "materials", "fixtures"))
    t = _section(d["table"], "table", ("height", "center", "size", "surface"))
    table = Table(_float(t["height"], "table.height"), _floats(t["center"], 2, "table.center"),
                  _floats(t["size"], 3, "table.size"), _surface(t["surface"], "table.surface", base))
    c = _section(d["camera"], "camera", ("pos", "quat", "fovy"))
    x, y, z = _floats(c["pos"], 3, "camera.pos")
    camera = rig.Camera(rig.EXTERIOR_NAME, rig.EXTERIOR_KEY, None, (x, y, table.height + z),
                        _floats(c["quat"], 4, "camera.quat"), _float(c["fovy"], "camera.fovy"))
    materials = {k: _floats(_section(v, f"materials.{k}", ("rgba",))["rgba"], 4, f"materials.{k}.rgba")
                 for k, v in _table(d.get("materials", {}), "materials").items()}
    rows = d.get("fixtures", [])
    if not isinstance(rows, list):
        raise ProfileError(f"fixtures must be an array of tables ([[fixtures]]), got {rows!r}")
    fixtures = tuple(_fixture(f, f"fixtures[{i}]") for i, f in enumerate(rows))
    b = _section(d["background"], "background",
                 ("wall_distance", "wall_half_width", "wall_height", "wall", "floor"))
    background = Background(_float(b["wall_distance"], "background.wall_distance"),
                            _float(b["wall_half_width"], "background.wall_half_width"),
                            _float(b["wall_height"], "background.wall_height"),
                            _surface(b["wall"], "background.wall", base),
                            _surface(b["floor"], "background.floor", base))
    undefined = sorted({f.material for f in fixtures} - set(materials))
    if undefined:
        raise ProfileError(f"fixtures name materials {undefined} that [materials] does not define")
    return WorkspaceProfile(_str(d["id"], "id"), camera, table, materials, fixtures, background, text, base)


def load_profile(path: Path | str = REFERENCE) -> WorkspaceProfile:
    path = Path(path).resolve()
    try:
        text = path.read_text(encoding="utf-8")
        d = tomllib.loads(text)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ProfileError(f"{path}: cannot read ({exc.__class__.__name__}: {exc})") from None
    if d.get("schema") != SCHEMA:
        raise ProfileError(f"{path}: schema {d.get('schema')!r}, expected {SCHEMA!r}")
    try:
        return _profile(d, text, path.parent)
    except ProfileError as exc:
        raise ProfileError(f"{path}: {exc}") from None
