"""The serving fields of a policy server's handshake, in one place, as typed as the protocol declares them.

The handshake is the benchmark's policy-server protocol (docs/reference/policy-server.md,
axis_qvla.bench.protocol). The FR3 serve check needs its serving-contract keys
(axis_qvla.bench.protocol.CONTRACT_KEYS: `image_geometry`, `state_order`, `action_units`, `action_fps`,
`action_dim_out`, `prompt_suffix`) and the protocol's required `action_horizon`; a handshake that does
not declare one of them cannot be checked, which is reported, never guessed. `guidance_scale` is
optional in the protocol, with its documented default 0; an absent one is that default.
`image_geometry` maps each camera to the transform the server applies to the RAW frames it receives
(`raw`, `center_crop224`, `letterbox224`): for an FR3 policy that must be the transform that made its
training frames from the 1280 x 720 recordings.

Values are compared as they are typed, never coerced: a 15.9 Hz server is not a 15 Hz one, and the string
"15" is not a number. A bool is never a number here.

A `qvla bench check-server --report` file holds the handshake under `handshake`; it is read from there.
Offline tier (stdlib only).
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping

from axis_qvla.bench.protocol import CONTRACT_KEYS

KEYS = CONTRACT_KEYS + ("action_horizon",)
GUIDANCE_DEFAULT = 0.0      # docs/reference/policy-server.md: guidance_scale, optional, default 0


@dataclasses.dataclass(frozen=True)
class ServeFields:
    image_geometry: dict[str, str]
    state_order: tuple[str, ...]
    action_units: str
    fps: int | float
    dim: int
    prompt_suffix: str
    guidance: int | float
    horizon: int


def _number(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _integer(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _names(v: object) -> bool:
    return isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v)


def _cameras(v: object) -> bool:
    return isinstance(v, Mapping) and all(isinstance(k, str) and isinstance(g, str) for k, g in v.items())


_TYPES = {"image_geometry": (_cameras, "a map of camera to geometry"), "state_order": (_names, "a list of names"),
          "action_units": (lambda v: isinstance(v, str), "a string"), "action_fps": (_number, "a number"),
          "action_dim_out": (_integer, "an integer"), "prompt_suffix": (lambda v: isinstance(v, str), "a string"),
          "action_horizon": (_integer, "an integer"), "guidance_scale": (_number, "a number")}


def handshake_of(doc: Mapping) -> Mapping:
    """The handshake itself, from a bare handshake or from a check-server report that holds one."""
    inner = doc.get("handshake")
    return inner if isinstance(inner, Mapping) and "endpoint" in doc else doc


def serve_fields(h: Mapping) -> tuple[ServeFields | None, list[str]]:
    """(the fields, []) or (None, a problem per key the handshake does not declare or declares mistyped)."""
    h = handshake_of(h)
    problems = [f"the handshake does not declare {k}" for k in KEYS if k not in h]
    problems += [f"{k} is {type(h[k]).__name__} {h[k]!r}, not {what}" for k, (ok, what) in _TYPES.items()
                 if k in h and not ok(h[k])]
    if problems:
        return None, problems
    return ServeFields(dict(h["image_geometry"]), tuple(h["state_order"]), h["action_units"], h["action_fps"],
                       h["action_dim_out"], h["prompt_suffix"], h.get("guidance_scale", GUIDANCE_DEFAULT),
                       h["action_horizon"]), []
