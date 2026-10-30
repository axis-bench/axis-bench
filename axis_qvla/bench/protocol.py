"""The policy-server protocol `qvla bench eval` speaks, as constants and validators.

Why: evaluating a new policy needs the protocol written down. docs/reference/policy-server.md is
the reference; this module is the part of it a program can check: the handshake, the observation
layout and what a reply must hold. `qvla bench check-server` runs these checks against a live
server before anything else trusts it.

Core tier: stdlib only.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

PROTOCOL = "qvla.policy_server/1"
NOISE_KEY = "__noise__"            # the reserved request key; openpi_client defines the same string
OBS_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "state", "prompt")
STATE_DIM = 8                      # q0..q6 (rad) + gripper closedness in [0, 1]
MIN_ACTION_DIM = 8                 # 7 joint velocities (rad/s) + gripper closedness
GRIPPER_COLUMN = 7
DEFAULT_REPLAN = 8                 # actions executed open loop per request
RAW_HEIGHT, RAW_WIDTH = 360, 640   # image_geometry "raw": frames as rendered
IMAGE_KEYS = OBS_KEYS[:2]
# The serving contract a handshake may state (every key optional). The client sends each camera's
# frame at its embodiment's size, neither cropped nor padded: the simulator's raw RAW_HEIGHT x RAW_WIDTH
# render, or the FR3 line's 320 x 180 frame (the size openpi's DROID converter stores;
# axis_qvla.realrobot.fr3.droid_client). `image_geometry` says which transform the SERVER applies to each
# camera's frame before its model sees it, so a server can be checked against the geometry its
# checkpoint was trained with. The other keys carry the simulator's values; a server for another
# embodiment states its own under the same keys.
IMAGE_TRANSFORMS = ("raw", "center_crop224", "letterbox224")
CONTRACT_KEYS = ("image_geometry", "state_order", "action_units", "action_fps", "action_dim_out", "prompt_suffix")
STATE_ORDER = ("q0", "q1", "q2", "q3", "q4", "q5", "q6", "gripper")
ACTION_UNITS = "rad_s"             # columns 0-6 are joint velocities in rad/s
ACTION_FPS = 15                    # actions are executed at 15 Hz
_OPTIONAL = {"cfg_conditioning": bool, "guidance_scale": (int, float), "quality_tag": int,
             "beta": (int, float), "checkpoint_dir": str, "config_name": str, "protocol": str,
             "action_units": str, "action_fps": (int, float), "action_dim_out": int, "prompt_suffix": str}
# Handshake keys a resumed `--policy` run (axis_qvla.bench.sim.campaign) compares against an earlier
# real run of the same arm, grouped by what they mean: a server that now answers differently on
# any of these must not silently reuse that run's queue or manifest. Deliberately excludes an
# undeclared extra key (the protocol permits one unchecked, and it may be volatile -- a start
# time, a pid, a build stamp) and the endpoint itself (a restarted server may move host or port).
CONDITIONING_KEYS = ("cfg_conditioning", "guidance_scale", "quality_tag", "beta")
SHAPE_KEYS = ("action_horizon", "action_dim", "protocol")
IDENTITY_KEYS = ("checkpoint_dir", "config_name")


def _typed(value: object, typ: object) -> bool:
    if typ is bool:
        return isinstance(value, bool)
    return not isinstance(value, bool) and isinstance(value, typ)  # a bool is never a number here


def parse_endpoint(text: str) -> tuple[str, int]:
    """`HOST:PORT` -> (host, port); anything else raises ValueError naming the input."""
    host, sep, port = str(text).rpartition(":")
    if not sep or not host or not port.isdigit() or not 0 < int(port) < 65536:
        raise ValueError(f"a policy server is HOST:PORT, got {text!r}")
    return host, int(port)


def handshake_problems(md: object) -> list[str]:
    """Every way a server's first frame breaks the protocol, as sentences. Empty = conformant."""
    if not isinstance(md, Mapping):
        return [f"the handshake is a {type(md).__name__}, not a map"]
    out: list[str] = []
    if md.get("supports_client_noise") is not True:
        out.append(f"supports_client_noise must be true: the client sends each call's sampler noise "
                   f"under {NOISE_KEY!r}, and a server that ignored it would make results depend on "
                   f"request order")
    for key in ("action_horizon", "action_dim"):
        v = md.get(key)
        if not (_typed(v, int) and v > 0):
            out.append(f"{key} must be a positive integer (the shape of the {NOISE_KEY} tensor), got {v!r}")
    for key, typ in _OPTIONAL.items():
        if md.get(key) is not None and not _typed(md[key], typ):
            out.append(f"{key} has type {type(md[key]).__name__}")
    g = md.get("guidance_scale")
    if _typed(g, (int, float)) and float(g) != 0.0 and md.get("cfg_conditioning") is not True:
        out.append(f"guidance_scale {g} needs cfg_conditioning true: without a conditioned branch the "
                   f"guidance weight does nothing, and the result directory would claim it did")
    if md.get("protocol") not in (None, PROTOCOL):
        out.append(f"protocol {md['protocol']!r} is not {PROTOCOL!r}")
    return out + _contract_problems(md)


def _contract_problems(md: Mapping) -> list[str]:
    """The optional serving-contract keys, against what the simulator's client does."""
    out: list[str] = []
    geo = md.get("image_geometry")
    if geo is not None:
        if not isinstance(geo, Mapping) or not geo:
            out.append("image_geometry must map each camera key to the geometry its frames arrive in")
        else:
            for cam, g in geo.items():
                if cam not in IMAGE_KEYS:
                    out.append(f"image_geometry names {cam!r}, a camera the client does not send "
                               f"(it sends {', '.join(IMAGE_KEYS)})")
                elif g not in IMAGE_TRANSFORMS:
                    out.append(f"image_geometry[{cam!r}] is {g!r}, not one of {', '.join(IMAGE_TRANSFORMS)} "
                               f"(the transform the server applies to the raw frame it receives)")
    order = md.get("state_order")
    if order is not None and (not isinstance(order, (list, tuple)) or [*order] != [*STATE_ORDER]):
        out.append(f"state_order must be {list(STATE_ORDER)} (the client's state vector), got {order!r}")
    units = md.get("action_units")
    if _typed(units, str) and units != ACTION_UNITS:
        out.append(f"action_units {units!r}: the simulator executes joint velocities in rad/s "
                   f"({ACTION_UNITS!r}); a real-robot command is a different contract")
    fps = md.get("action_fps")
    if _typed(fps, (int, float)) and float(fps) != float(ACTION_FPS):
        out.append(f"action_fps {fps}: the client executes actions at {ACTION_FPS} Hz")
    dout, dim = md.get("action_dim_out"), md.get("action_dim")
    if _typed(dout, int) and (dout < MIN_ACTION_DIM or (_typed(dim, int) and dout > dim)):
        out.append(f"action_dim_out {dout} must be at least {MIN_ACTION_DIM} (7 joint velocities and the "
                   f"gripper) and at most action_dim")
    return out


def contract(md: Mapping) -> dict:
    """The serving-contract keys a handshake states, for the conformance report and the run record."""
    return {k: md[k] for k in CONTRACT_KEYS if k in md}


def reply_problems(reply: object, *, replan: int) -> list[str]:
    """Every way one reply breaks the protocol. Reads only `.shape` and `.dtype.kind`, so it needs
    no numpy; finiteness is checked by the caller, which has the array."""
    if not isinstance(reply, Mapping):
        return [f"the reply is a {type(reply).__name__}, not a map"]
    actions = reply.get("actions")
    shape = getattr(actions, "shape", None)
    kind = getattr(getattr(actions, "dtype", None), "kind", None)
    if shape is None or kind is None:
        return ["the reply has no 'actions' array"]
    out: list[str] = []
    if len(shape) != 2:
        out.append(f"actions must be 2-D (horizon, dim), got shape {tuple(shape)}")
    else:
        if shape[0] < replan:
            out.append(f"actions has {shape[0]} rows; the client executes the first {replan} before it "
                       f"observes again, so a reply needs at least {replan}")
        if shape[1] < MIN_ACTION_DIM:
            out.append(f"actions has {shape[1]} columns; the action dim is at least {MIN_ACTION_DIM} "
                       f"(7 joint velocities and the gripper)")
    if kind != "f":
        out.append(f"actions must be floating point, got dtype kind {kind!r}")
    return out


def add_check_server_arguments(ap: argparse.ArgumentParser) -> None:
    """The `bench check-server` options, here so building the CLI parser imports no numpy."""
    ap.add_argument("--policy", required=True, help="the running server, HOST:PORT")
    ap.add_argument("--benchmark", default=None,
                    help="declaration, as <version>/<name> or <name> (default franka_heldout24 of "
                         "axis-heldout-v1); supplies the prompt and the frame size/replan "
                         "(its execution.json, or 360 x 640 frames and replan 8 for a benchmark that "
                         "declares none)")
    ap.add_argument("--task", type=int, default=None, help="task whose prompt is sent (default: the first)")
    ap.add_argument("--calls", type=int, default=3, help="requests to send (default 3)")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="seconds to wait for the port, the websocket handshake AND every reply "
                         "including the first (default 120); raise it for a server with a cold "
                         "JIT compile, which can take minutes (pi0.5, X-VLA)")
    ap.add_argument("--report", type=Path, default=None,
                    help="also write the checks as JSON here (exit 2 when it cannot be written)")
